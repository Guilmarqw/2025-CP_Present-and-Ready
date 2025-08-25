import os
import cv2
import time
import dlib
import base64
import torch
import numpy as np
import datetime
import threading
import face_recognition
from ultralytics import YOLO
from flask import Flask, Response, send_from_directory, request, jsonify
import mysql.connector
import smtplib
from email.message import EmailMessage
import random
import string

# =========================
# CONFIG
# =========================
USERNAME = "admin5610"
PASSWORD = "101Pok3r5610"
CAM_IP   = "192.168.1.29"
STREAM   = "stream2"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAM_IP}:554/{STREAM}"

WEIGHTS_PATH = "yolov8n-face.pt"
STREAM_WIDTH, STREAM_HEIGHT = 640, 360
DETECT_EVERY = 5
CONF_THRESH = 0.55

STABLE_TOLERANCE_FRAMES = 12
MAX_TRACKS = 128
EXPAND_BOX_RATIO = 0.25

ENABLE_RECOGNITION = True
TOLERANCE = 0.45
NUM_JITTERS_CROP = 1
KNOWN_DIR = "known_faces"

RECONNECT_COOLDOWN = 2.0
GRAB_SLEEP = 0.01
MAX_EMPTY_GRABS = 150

# MySQL Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",  # Replace with your MySQL username
    "password": "",  # Replace with your MySQL password
    "database": "faces"
}

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "lawrencetilde@gmail.com",  # Replace with your Gmail address
    "sender_password": "ufwxvjacdtftfcof",  # Replace with your Gmail App Password
    "default_recipient": "lawrencetilde@gmail.com"  # Replace with default recipient email
}

# =========================
# MySQL Setup
# =========================
def init_mysql():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                student_id VARCHAR(50) NOT NULL UNIQUE,
                course VARCHAR(100),
                year_section VARCHAR(50),
                image_path VARCHAR(255),
                registration_time DATETIME
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otps (
                email VARCHAR(255) PRIMARY KEY,
                otp VARCHAR(6) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] MySQL tables 'students' and 'otps' initialized.")
    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL initialization failed: {e}")
        raise SystemExit(1)

init_mysql()

# =========================
# Email Sending
# =========================
def send_confirmation_email(subject, body, recipient_email):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = recipient_email or EMAIL_CONFIG['default_recipient']
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        print(f"[INFO] Email sent to {recipient_email or EMAIL_CONFIG['default_recipient']}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to send email: {e}")
        return False

# =========================
# Utilities
# =========================
def expand_box(x1, y1, x2, y2, w, h, scale=EXPAND_BOX_RATIO):
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * scale), int(bh * scale)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(w - 1, x2 + px); ny2 = min(h - 1, y2 + py)
    return nx1, ny1, nx2, ny2

def enhance_lighting(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def safe_decode_data_url(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)

def image_bytes_to_rgb_array(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# =========================
# Load known faces
# =========================
known_face_encodings, known_face_names = [], []

def load_known_faces_from_disk():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    if ENABLE_RECOGNITION and os.path.isdir(KNOWN_DIR):
        for person in os.listdir(KNOWN_DIR):
            pdir = os.path.join(KNOWN_DIR, person)
            if not os.path.isdir(pdir):
                continue
            for fname in os.listdir(pdir):
                fpath = os.path.join(pdir, fname)
                if not fpath.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    img = face_recognition.load_image_file(fpath)
                    encs = face_recognition.face_encodings(img, model="large", num_jitters=2)
                    if encs:
                        known_face_encodings.append(encs[0])
                        known_face_names.append(person)
                        print(f"[INFO] Loaded {person} from {fname}")
                except Exception as e:
                    print(f"[WARN] Failed to encode {fpath}: {e}")
    print(f"[INFO] Known identities: {len(known_face_names)}")

def add_known_face_from_bytes(img_bytes: bytes, person_name: str) -> bool:
    try:
        rgb = image_bytes_to_rgb_array(img_bytes)
        if rgb is None:
            print("[WARN] Could not decode image bytes for encoding.")
            return False
        encs = face_recognition.face_encodings(rgb, model="large", num_jitters=2)
        if not encs:
            print("[WARN] No face encoding found in submitted registration image.")
            return False
        known_face_encodings.append(encs[0])
        known_face_names.append(person_name)
        print(f"[INFO] Encoding added in-memory for {person_name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to add in-memory encoding: {e}")
        return False

# Initialize known faces at startup
load_known_faces_from_disk()

# =========================
# Load YOLOv8-Face
# =========================
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"'{WEIGHTS_PATH}' not found. Download yolov8n-face.pt and place it next to this script.")

yolo = YOLO(WEIGHTS_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo.to(DEVICE)
print(f"[INFO] Using device: {DEVICE}  |  Model: {WEIGHTS_PATH}")

# =========================
# RTSP capture
# =========================
cap_lock = threading.Lock()
cap = None

def open_stream():
    global cap
    with cap_lock:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        if not cap.isOpened():
            print("[ERROR] Cannot open RTSP stream. Check IP/credentials and RTSP is enabled.")
            return False
        print("[INFO] RTSP connected.")
        return True

if not open_stream():
    raise SystemExit(1)

latest_frame = None
stop_flag = False

def grabber():
    global latest_frame, stop_flag
    empty_count = 0
    while not stop_flag:
        with cap_lock:
            ok, f = cap.read()
        if not ok:
            empty_count += 1
            if empty_count > MAX_EMPTY_GRABS:
                print("[WARN] Stream stalled. Reconnecting...")
                time.sleep(RECONNECT_COOLDOWN)
                if open_stream():
                    empty_count = 0
                else:
                    time.sleep(RECONNECT_COOLDOWN)
            else:
                time.sleep(0.01)
            continue
        empty_count = 0
        latest_frame = f
        time.sleep(GRAB_SLEEP)

grab_thread = threading.Thread(target=grabber, daemon=True)
grab_thread.start()

# =========================
# Tracking & attendance
# =========================
tracks = []
attendance = {}

def mark_attendance(name):
    if not ENABLE_RECOGNITION or name == "Unknown":
        return
    if name not in attendance:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance[name] = ts
        print(f"[ATTENDANCE] {name} at {ts}")

def update_trackers(rgb, frame, frame_idx):
    global tracks
    h, w = frame.shape[:2]
    kept = []
    for tr in tracks:
        try:
            tr["tracker"].update(rgb)
            pos = tr["tracker"].get_position()
            x1, y1 = int(pos.left()), int(pos.top())
            x2, y2 = int(pos.right()), int(pos.bottom())
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
            continue
        tr["box"] = (x1, y1, x2, y2)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = tr["name"] if ENABLE_RECOGNITION else "Face"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if frame_idx - tr["last_seen"] <= STABLE_TOLERANCE_FRAMES:
            kept.append(tr)
    tracks = kept

def refresh_with_detections(frame, rgb, frame_idx):
    global tracks
    if len(tracks) > MAX_TRACKS:
        tracks = tracks[:MAX_TRACKS]
    h, w = frame.shape[:2]
    frame_eq = enhance_lighting(frame)
    results = yolo.predict(source=frame_eq, verbose=False, conf=CONF_THRESH,
                           imgsz=max(STREAM_WIDTH, STREAM_HEIGHT), device=DEVICE)

    dets = []
    if results:
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w-1, x2)), int(min(h-1, y2))
                if conf >= CONF_THRESH and x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, conf))

    new_tracks = []
    for (x1, y1, x2, y2, conf) in dets:
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_BOX_RATIO)
        name = "Unknown"

        if ENABLE_RECOGNITION and len(known_face_encodings) > 0:
            crop = rgb[ey1:ey2, ex1:ex2]
            try:
                encs = face_recognition.face_encodings(crop, num_jitters=NUM_JITTERS_CROP, model="small")
                if encs:
                    enc = encs[0]
                    matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=TOLERANCE)
                    dists = face_recognition.face_distance(known_face_encodings, enc)
                    if len(dists) > 0:
                        best = int(np.argmin(dists))
                        if matches[best]:
                            name = known_face_names[best]
            except Exception:
                pass

        face_crop = frame[ey1:ey2, ex1:ex2]
        if face_crop.size > 0:
            save_dir = "captured_faces"
            os.makedirs(save_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{ts}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename))

        dtracker = dlib.correlation_tracker()
        try:
            dtracker.start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
        except Exception:
            continue
        new_tracks.append({
            "tracker": dtracker,
            "name": name if ENABLE_RECOGNITION else "Face",
            "last_seen": frame_idx,
            "box": (x1, y1, x2, y2)
        })
        mark_attendance(name)

    tracks[:] = new_tracks

# =========================
# Flask Setup and Routes
# =========================
app = Flask(__name__)

def generate_frames():
    frame_idx = 0
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with cap_lock:
            frame = latest_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        update_trackers(rgb, frame, frame_idx)
        if frame_idx % DETECT_EVERY == 0:
            refresh_with_detections(frame, rgb, frame_idx)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/send_otp', methods=['POST'])
def send_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        if not email.endswith("@wmsu.edu.ph"):
            return jsonify({"status": "error", "message": "Invalid WMSU email"}), 400
        otp = ''.join(random.choices(string.digits, k=6))
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM otps WHERE email = %s", (email,))
            cursor.execute("INSERT INTO otps (email, otp) VALUES (%s, %s)", (email, otp))
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as e:
            print(f"[WARN] OTP storage failed: {e}")
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500
        email_body = f"Your OTP for WMSU registration is {otp}. It expires in 5 minutes."
        if send_confirmation_email("WMSU OTP Verification", email_body, email):
            return jsonify({"status": "success", "message": "OTP sent to email"}), 200
        return jsonify({"status": "error", "message": "Failed to send OTP email"}), 500
    except Exception as e:
        print(f"[ERROR] Send OTP failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        otp = data.get("otp", "").strip()
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT otp, created_at FROM otps WHERE email = %s", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            stored_otp, created_at = result
            time_diff = (datetime.datetime.now() - created_at).total_seconds() / 60
            if time_diff > 5:
                return jsonify({"status": "error", "message": "OTP expired"}), 400
            if stored_otp == otp:
                conn = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM otps WHERE email = %s", (email,))
                conn.commit()
                cursor.close()
                conn.close()
                return jsonify({"status": "success", "message": "OTP verified"}), 200
        return jsonify({"status": "error", "message": "Invalid OTP"}), 400
    except mysql.connector.Error as e:
        print(f"[WARN] OTP verification failed: {e}")
        return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] Verify OTP failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def registration_page():
    return send_from_directory('Registration', 'Registration.html')

@app.route('/timer')
def timer_page():
    return send_from_directory('Timer', 'Timer.html')

@app.route('/camfootage')
def camfootage_page():
    return send_from_directory('Camfootage', 'camfootage.html')

@app.route('/summary')
def summary_page():
    return send_from_directory(os.path.join('Timer', 'SumSession'), 'Summary.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        student_id = (data.get("studentId") or "").strip()
        course = (data.get("course") or "").strip()
        year_section = (data.get("yearSection") or "").strip()
        face_images = data.get("faceImages") or {}
        recipient_email = (data.get("email") or "").strip()

        if not all([name, student_id, course, year_section, face_images.get("front"), face_images.get("left"), face_images.get("right")]):
            return jsonify({"status": "error", "message": "Missing required fields or face images"}), 400

        person_dir_name = name.replace(" ", "_")
        person_dir = os.path.join(KNOWN_DIR, person_dir_name)
        os.makedirs(person_dir, exist_ok=True)

        # Save all three images
        timestamp = int(time.time())
        image_paths = {}
        encoding_added = False
        for pose in ['front', 'left', 'right']:
            img_bytes = safe_decode_data_url(face_images[pose])
            filename = f"{student_id}_{pose}_{timestamp}.png"
            file_path = os.path.join(person_dir, filename)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            image_paths[pose] = file_path
            # Use front image for primary encoding
            if pose == 'front':
                encoding_added = add_known_face_from_bytes(img_bytes, person_dir_name)

        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor()
            registration_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Store only the front image path in the database for simplicity
            cursor.execute("""
                INSERT INTO students (name, student_id, course, year_section, image_path, registration_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (name, student_id, course, year_section, image_paths['front'], registration_time))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"[INFO] Stored {name} ({student_id}) in MySQL database")
        except mysql.connector.Error as e:
            print(f"[WARN] MySQL insertion failed: {e}")
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

        email_body = f"""
        Registration Confirmation

        Name: {name}
        Student ID: {student_id}
        Course: {course}
        Year/Section: {year_section}
        Registration Time: {registration_time}

        Your face (front, left, right profiles) has been successfully registered in the attendance system.
        """
        email_sent = send_confirmation_email("Face Attendance Registration Confirmation", email_body, recipient_email)

        print(f"[REGISTER] {name} ({student_id}) saved images at {person_dir}")
        return jsonify({
            "status": "success",
            "message": f"{name} registered successfully",
            "encoding_added": bool(encoding_added),
            "email_sent": email_sent,
            "next": "/timer"
        }), 200
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_flag = True
        time.sleep(0.05)
        with cap_lock:
            if cap is not None:
                cap.release()
        if ENABLE_RECOGNITION:
            with open("attendance_log.csv", "w") as f:
                f.write("Name,DateTime\n")
                for name, ts in attendance.items():
                    f.write(f"{name},{ts}\n")
            print("[INFO] Attendance saved to attendance_log.csv")