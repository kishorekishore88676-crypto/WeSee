from flask import Flask, render_template, request, redirect, session, Response, jsonify
import cv2
import time
import pyttsx3
from ultralytics import YOLO
import pytesseract
import os
import threading
import queue
import numpy as np
import logging
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "wesee_secret"

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CONFIG: Set Tesseract Path (Auto-detect Windows vs Linux)
if os.name == 'nt': # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: # Linux (Render/Docker)
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# ---------------- STATES ----------------
detect_objects = True
detect_text = False
voice_on = True
history = []
global_frame = None
frame_lock = threading.Lock()
current_detections = []
current_text_detections = []
last_text = ""
last_text_time = 0
search_query = ""
excluded_objects = []
MODEL_STATUS = "Initializing..."

# ---------------- DATABASE SETUP ----------------
# Use absolute path to ensure the database is found correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "wesee.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
        conn.commit()

init_db()

# ---------------- LOAD MODELS ----------------
logging.info("Loading YOLO model...")
model = None
try:
    model_yolo = YOLO("yolov8l-world.pt")
    # Define the specific custom objects you want to detect
    custom_classes = [
        "person", "watch", "phone", "laptop", "mouse", "keyboard", "monitor", "tv", "fan", "ceiling fan", 
        "light", "lamp", "bottle", "cup", "chair", "table", "remote", "book", "pen", "bag", "backpack", 
        "glasses", "sunglasses", "wallet", "keys", "headphones", "earbuds", "camera", "charger", "jacket", 
        "shoe", "door", "window", "cabinet", "shelf", "printer", "router", "switch", "socket", "trash can",
        "ring", "necklace", "bracelet", "earrings", "hat", "cap", "umbrella", "bicycle", "car", "motorcycle"
    ]
    model_yolo.set_classes(custom_classes)
    model = model_yolo
    MODEL_STATUS = "YOLO-World (Custom)"
    logging.info("YOLO-World model loaded with custom vocabulary.")
except Exception as e:
    logging.warning(f"YOLO-World failed to load (likely missing 'clip' or 'git'): {e}")
    logging.info("Attempting to fall back to standard YOLOv8 Medium model...")
    try:
        model = YOLO("yolov8m.pt")
        MODEL_STATUS = "YOLOv8-Medium (Standard)"
        logging.info("Successfully loaded standard YOLOv8 Medium model.")
        print("\n" + "!"*60)
        print("⚠️  WARNING: 'clip' library missing.")
        print("   Switched to STANDARD MODEL. Custom objects (chalk, jewels) will NOT be detected.")
        print("   To fix: Install Git, then run: pip install git+https://github.com/ultralytics/CLIP.git")
        print("!"*60 + "\n")
    except Exception as e2:
        MODEL_STATUS = "ERROR: No model loaded"
        logging.error(f"CRITICAL: Could not load any YOLO model. Object detection will be disabled. Error: {e2}")

# ---------------- HELPERS ----------------
tts_queue = queue.Queue()

def tts_loop():
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            logging.warning("pythoncom (pywin32) not found. TTS might be unstable on Windows.")
        
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        
        # Try to set a better voice (usually index 1 is female/clearer on Windows)
        try:
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
        except Exception:
            pass
        
        while True:
            try:
                # Get text with a timeout to allow checking for thread termination if needed
                text = tts_queue.get(timeout=1)
                
                if text:
                    engine.say(text)
                    engine.runAndWait()
                    time.sleep(0.1)
                    tts_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"TTS Loop Error: {e}")
                # Attempt to re-initialize engine if it crashes
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 160)
                except Exception as init_e:
                    logging.error(f"TTS Re-init failed: {init_e}")
                    time.sleep(1)
    except Exception as e:
        logging.critical(f"TTS Thread Critical Failure: {e}")

def speak(text):
    if voice_on:
        # Clear queue if it's backing up (too much lag) to ensure real-time feedback
        if tts_queue.qsize() > 10:
            with tts_queue.mutex:
                tts_queue.queue.clear()
        tts_queue.put(text)

def detect_direction(cx, width):
    if cx < width / 3:
        return "left"
    elif cx > 2 * width / 3:
        return "right"
    return "center"

# ---------------- BACKGROUND PROCESSING ----------------
def detection_loop():
    global current_detections, current_text_detections, last_text, last_text_time, history
    last_objects = set()
    
    while True:
        # Get the latest frame safely
        frame_input = None
        with frame_lock:
            if global_frame is not None:
                frame_input = global_frame.copy()
        
        if frame_input is None:
            time.sleep(0.1)
            continue

        h, w, _ = frame_input.shape
        timestamp = time.strftime("%H:%M:%S")

        # --- OBJECT DETECTION ---
        if detect_objects and model:
            try:
                # Increased confidence to 0.60 to ensure ONLY correct objects are shown (High Precision)
                # This removes "wrong" detections and guesses
                results = model(frame_input, conf=0.60, iou=0.4, verbose=False)
                new_detections = []
                current_labels = set()

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls].title()
                        
                        if label.lower() in excluded_objects:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        direction = detect_direction(cx, w)
                        
                        new_detections.append((x1, y1, x2, y2, label, direction))
                        current_labels.add(label)
                        
                        # Add to history (limit to 50 items)
                        history.insert(0, {"type": "object", "content": f"{label} ({direction})", "time": timestamp})
                        if len(history) > 50: history.pop()
                
                current_detections = new_detections

                # REMOVED AUTOMATIC VOICE - Only speak on tap
                last_objects = current_labels
            except Exception as e:
                logging.error(f"YOLO Error: {e}")
        else:
            current_detections = []
            last_objects = set() # Reset so it speaks again when re-enabled

        # --- OCR (TEXT READING) ---
        if detect_text:
            try:
                gray = cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)
                scale = 2
                resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                
                # Simplified Preprocessing: Otsu Thresholding is cleaner for text
                # This fixes "garbage text" detection
                _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Added 'lang' parameter for Hindi (hin) and Kannada (kan). 
                # NOTE: Ensure 'hin.traineddata' and 'kan.traineddata' are in your Tesseract/tessdata folder.
                data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config='--psm 11', lang='eng+hin+kan')
                new_text_detections = []
                found_words = []

                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 60:  # Higher confidence to reduce wrong text
                        word = data['text'][i].strip()
                        # Strict filter: Must be >1 char and contain alphanumeric (no random symbols)
                        if len(word) > 1 and any(c.isalnum() for c in word):
                            x = int(data['left'][i] / scale)
                            y = int(data['top'][i] / scale)
                            w_box = int(data['width'][i] / scale)
                            h_box = int(data['height'][i] / scale)
                            new_text_detections.append((x, y, w_box, h_box, word))
                            found_words.append(word)
                
                current_text_detections = new_text_detections

                if found_words:
                    sentence = " ".join(found_words)
                    current_time = time.time()
                    
                    # Fix: Prevent spamming. Only speak if text changed AND 3 seconds passed, 
                    # OR if it's the same text but 10 seconds passed.
                    if (sentence != last_text and (current_time - last_text_time > 3)) or \
                       (sentence == last_text and (current_time - last_text_time > 10)):
                            speak(f"Reading: {sentence}")
                            history.insert(0, {"type": "text", "content": sentence[:30] + "...", "time": timestamp})
                            if len(history) > 50: history.pop()
                            last_text = sentence
                            last_text_time = current_time
            except Exception as e:
                logging.error(f"OCR Error: {e}")
        else:
            current_text_detections = []

        # Increase sleep to reduce CPU usage (Lag Fix)
        # We don't need 30 detections per second. 4 detections/sec is enough for accuracy.
        time.sleep(0.25)

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global global_frame
    logging.info("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        logging.error("❌ Camera not opened - Running in Server Mode?")
        # Create a placeholder frame for the server
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "SERVER MODE: NO CAMERA", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode(".jpg", blank_frame)
        frame_bytes = buffer.tobytes()
        while True:
             yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
             time.sleep(1)
        return

    logging.info("✅ Camera opened")

    while True:
        success, frame = cap.read()
        if not success:
            logging.error("❌ Frame read failed")
            break

        h, w, _ = frame.shape

        # LOW LIGHT WARNING
        if frame.mean() < 40:
            cv2.putText(frame, "LOW LIGHT",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        # DRAW BOXES (Every Frame)
        for (x1, y1, x2, y2, label, direction) in current_detections:
            # Default color (Green)
            color = (0, 255, 0)
            
            # Check if object matches search query
            if search_query and search_query in label.lower():
                color = (0, 255, 255) # Yellow (BGR) for highlight
                # Show label only for searched item
                cv2.putText(frame, f"TARGET: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw simple border (Faster than transparent overlay)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for (x, y, w, h, word) in current_text_detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)

        with frame_lock:
            global_frame = frame.copy()

        # STREAM FRAME
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        print(f"Login attempt: {username}")
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            
            if user and check_password_hash(user[0], password):
                session["user"] = username
                return redirect("/dashboard")
            else:
                return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        try:
            print(f"Registering user: {username}")
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
            return redirect("/")
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username already exists")
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html", user=session["user"], model_status=MODEL_STATUS)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_object")
def toggle_object():
    global detect_objects, detect_text
    detect_objects = not detect_objects
    if detect_objects:
        detect_text = False
    return jsonify({"detect_objects": detect_objects, "detect_text": detect_text})

@app.route("/toggle_text")
def toggle_text():
    global detect_text, detect_objects
    detect_text = not detect_text
    if detect_text:
        detect_objects = False
    return jsonify({"detect_text": detect_text, "detect_objects": detect_objects})

@app.route("/toggle_voice")
def toggle_voice():
    global voice_on
    voice_on = not voice_on
    return jsonify({"voice": voice_on})

@app.route("/set_search", methods=["POST"])
def set_search():
    global search_query
    data = request.json
    search_query = data.get("query", "").lower().strip()
    return jsonify({"status": "ok", "query": search_query})

@app.route("/set_exclusion", methods=["POST"])
def set_exclusion():
    global excluded_objects
    data = request.json
    query = data.get("query", "").lower().strip()
    if query:
        excluded_objects = [x.strip() for x in query.split(",")]
    else:
        excluded_objects = []
    return jsonify({"status": "ok", "excluded": excluded_objects})

@app.route("/history")
def get_history():
    return jsonify(history[:15]) # Return top 15 recent items

@app.route("/tap", methods=["POST"])
def handle_tap():
    data = request.json
    rel_x = data.get('x', 0)
    rel_y = data.get('y', 0)
    
    found_name = ""
    
    with frame_lock:
        if global_frame is not None:
            h, w, _ = global_frame.shape
            click_x = int(rel_x * w)
            click_y = int(rel_y * h)
            
            # Check Objects
            if detect_objects:
                # Iterate backwards to find top-most box
                for (x1, y1, x2, y2, label, direction) in reversed(current_detections):
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        found_name = label
                        speak(found_name)
                        break
            
            # Check Text (if no object found or text mode is on)
            if not found_name and detect_text:
                for (x, y, w_box, h_box, word) in reversed(current_text_detections):
                    if x <= click_x <= (x + w_box) and y <= click_y <= (y + h_box):
                        found_name = word
                        speak(found_name)
                        break
    
    if found_name:
        return jsonify({"name": found_name})
    else:
        return jsonify({"name": ""})

@app.route("/capture")
def capture():
    global global_frame
    ret = False
    with frame_lock:
        if global_frame is not None:
            frame = global_frame.copy()
            ret = True

    if ret:
        os.makedirs("captures", exist_ok=True)
        path = f"captures/capture_{int(time.time())}.jpg"
        cv2.imwrite(path, frame)

    return "OK"

@app.route("/shutdown", methods=["POST"])
def shutdown():
    logging.info("Shutting down system...")
    # Forcefully exit the python process
    os._exit(0)

# ---------------- RUN ----------------
if __name__ == "__main__":
    threading.Thread(target=tts_loop, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    
    print("\n" + "="*50)
    print(f"✅ SERVER RUNNING: http://127.0.0.1:5000")
    print(f"   Active Model: {MODEL_STATUS}")
    print("="*50 + "\n")
    
    # use_reloader=False prevents threads from running twice, ensuring the link shows up
    app.run(debug=True, threaded=True, use_reloader=False, host='127.0.0.1', port=5000)
