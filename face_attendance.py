import os
import cv2
import time
import dlib
import torch
import numpy as np
import datetime
import threading
import face_recognition
import mysql.connector
import smtplib
import random
import string
import json
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from flask import Flask, Response, send_from_directory, request, jsonify
from flask_cors import CORS
import logging
import ssl
from werkzeug.utils import secure_filename

# =========================
# CONFIG
# =========================
USERNAME = "admin5610"
PASSWORD = "101pok3r5610"
CAM_IP   = "192.168.254.113"
STREAM   = "stream1"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAM_IP}:554/{STREAM}"

WEIGHTS_PATH = "yolov8n-face.pt"
STREAM_WIDTH, STREAM_HEIGHT = 1280, 720
DETECT_EVERY = 5
CONF_THRESH = 0.45

STABLE_TOLERANCE_FRAMES = 12
MAX_TRACKS = 128
EXPAND_BOX_RATIO = 0.4

ENABLE_RECOGNITION = True
TOLERANCE = 0.7
NUM_JITTERS_CROP = 10
KNOWN_DIR = "known_faces"

RECONNECT_COOLDOWN = 2.0
GRAB_SLEEP = 0.01
MAX_EMPTY_GRABS = 150

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facedetect'
}

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'lawrencetilde@gmail.com',
    'password': 'ufwxvjacdtftfcof'
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(recipient_email, otp_code):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = recipient_email
        msg['Subject'] = "Your OTP Code for WMSU Face Attendance"
        
        body = f"""
        <html>
        <body>
            <h2>WMSU Face Attendance System</h2>
            <p>Your OTP code is: <strong>{otp_code}</strong></p>
            <p>This code will expire in 10 minutes.</p>
            <p>If you did not request this code, please ignore this email.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        logger.info(f"OTP email sent to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email to {recipient_email}: {e}")
        return False

# =========================
# Load known faces from database
# =========================
def load_known_faces_from_db():
    global known_face_encodings, known_face_names, known_face_ids
    known_face_encodings, known_face_names, known_face_ids = [], [], []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, first_name, last_name, face_encoding FROM students WHERE face_encoding IS NOT NULL")
        for (student_id, first_name, last_name, face_encoding) in cursor:
            try:
                # Handle different encoding formats
                if isinstance(face_encoding, str):
                    # Remove brackets and split by commas
                    encoding_str = face_encoding.strip('[]')
                    encoding = np.fromstring(encoding_str, sep=',', dtype=np.float64)
                else:
                    # Assume it's already a byte array or similar
                    encoding = np.frombuffer(face_encoding, dtype=np.float64)
                
                if encoding.size == 128:
                    known_face_encodings.append(encoding)
                    full_name = f"{first_name} {last_name}"
                    known_face_names.append(full_name)
                    known_face_ids.append(student_id)
                    logger.info(f"Loaded {full_name} ({student_id}) with encoding shape {encoding.shape}")
                else:
                    logger.warning(f"Invalid encoding size for {student_id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for {student_id}: {e}")
        cursor.close()
        conn.close()
        logger.info(f"Loaded {len(known_face_names)} known faces from database")
    except Exception as e:
        logger.error(f"Failed to load faces from database: {e}")

# Initialize known faces
load_known_faces_from_db()

# =========================
# Load YOLOv8-Face
# =========================
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"'{WEIGHTS_PATH}' not found. Download yolov8n-face.pt and place it next to this script.")

yolo = YOLO(WEIGHTS_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo.to(DEVICE)
logger.info(f"Using device: {DEVICE}  |  Model: {WEIGHTS_PATH}")

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
            logger.error("Cannot open RTSP stream. Check IP/credentials and RTSP is enabled.")
            return False
        logger.info("RTSP connected.")
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
                logger.warning("Stream stalled. Reconnecting...")
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

def mark_attendance(name, student_id):
    if not ENABLE_RECOGNITION or name == "Unknown" or student_id is None:
        return
    
    try:
        if student_id not in attendance:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance[student_id] = ts
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO attendance (student_id, name, timestamp) VALUES (%s, %s, %s)",
                    (student_id, name, ts)
                )
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Attendance recorded: {name} ({student_id}) at {ts}")
            except Exception as e:
                logger.error(f"Failed to save attendance to database: {e}")
    except Exception as e:
        logger.error(f"Failed to mark attendance for {name}: {e}")

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

def recognize_face(face_image, tolerance=0.6):
    """
    Enhanced face recognition with better handling for low-resolution images
    """
    try:
        # Preprocess image for better recognition
        if len(face_image.shape) == 3:
            # Convert to grayscale for some processing
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)
            
            # Apply a mild bilateral filter to reduce noise while keeping edges sharp
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = face_image
            
        # Generate face encoding with more jitters for better accuracy
        encodings = face_recognition.face_encodings(
            enhanced, 
            known_face_locations=[(0, enhanced.shape[1], enhanced.shape[0], 0)],  # Use entire image
            num_jitters=3,  # Increased from default for better accuracy
            model="large"   # Use large model for better accuracy
        )
        
        if not encodings:
            return "Unknown", None, float('inf')
            
        face_encoding = encodings[0]
        
        # Compare with known faces
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] <= tolerance:
                return known_face_names[best_match_index], known_face_ids[best_match_index], face_distances[best_match_index]
        
        return "Unknown", None, float('inf')
        
    except Exception as e:
        logger.error(f"Error in recognize_face: {e}")
        return "Unknown", None, float('inf')    

def refresh_with_detections(frame, rgb, frame_idx):
    global tracks
    if len(tracks) > MAX_TRACKS:
        tracks = tracks[:MAX_TRACKS]
    h, w = frame.shape[:2]
    frame_eq = enhance_lighting(frame)
    results = yolo.predict(source=frame_eq, verbose=False, conf=CONF_THRESH, imgsz=max(STREAM_WIDTH, STREAM_HEIGHT), device=DEVICE)
    dets = []
    if results:
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                x1 = int(max(0, x1)); y1 = int(max(0, y1))
                x2 = int(min(w-1, x2)); y2 = int(min(h-1, y2))
                if conf >= CONF_THRESH and x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, conf))
    
    logger.info(f"Frame {frame_idx}: Detected {len(dets)} faces with conf > {CONF_THRESH}")
    new_tracks = []
    
    for (x1, y1, x2, y2, conf) in dets:
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_BOX_RATIO)
        
        # Extract face region for recognition
        face_region = rgb[ey1:ey2, ex1:ex2]
        
        # Skip if face region is too small
        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            logger.info(f"Face region too small for recognition: {face_region.shape}")
            name = "Unknown"
        else:
            # Use our enhanced recognition function
            name, student_id, distance = recognize_face(face_region, TOLERANCE)
            
            if name != "Unknown":
                logger.info(f"Recognized {name} (ID: {student_id}) with distance {distance:.4f}")
                # Mark attendance for recognized face
                mark_attendance(name, student_id)
        
        # Create tracker for this face
        dtracker = dlib.correlation_tracker()
        try:
            dtracker.start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
            new_tracks.append({
                "tracker": dtracker, 
                "name": name, 
                "last_seen": frame_idx, 
                "box": (x1, y1, x2, y2)
            })
        except Exception as e:
            logger.error(f"Tracker error: {e}")
            continue
    
    tracks[:] = new_tracks

    

# =========================
# Flask streaming & API
# =========================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/send_otp', methods=['POST'])
def send_otp():
    email = request.json.get('email', '').strip()
    
    if not email or "@wmsu.edu.ph" not in email:
        logger.warning(f"Invalid email received: {email}")
        return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
    
    # Generate OTP
    otp_code = generate_otp()
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=10)
    
    # Store OTP in database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete any existing OTP for this email
        cursor.execute("DELETE FROM otp_codes WHERE email = %s", (email,))
        
        # Insert new OTP
        cursor.execute(
            "INSERT INTO otp_codes (email, otp_code, expires_at) VALUES (%s, %s, %s)",
            (email, otp_code, expires_at)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Send OTP via email
        if send_otp_email(email, otp_code):
            logger.info(f"OTP sent to {email}")
            return jsonify({'success': True, 'message': 'OTP sent successfully'})
        else:
            logger.error(f"Failed to send OTP email to {email}")
            return jsonify({'success': False, 'message': 'Failed to send OTP email'})
            
    except Exception as e:
        logger.error(f"Database error during OTP send for {email}: {e}")
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})

@app.route('/api/verify_otp', methods=['POST'])
def verify_otp():
    email = request.json.get('email', '').strip()
    otp_code = request.json.get('otp', '').strip()
    
    if not email or not otp_code:
        logger.warning("Missing email or OTP in verify_otp request")
        return jsonify({'success': False, 'message': 'Email and OTP are required'})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT otp_code, expires_at FROM otp_codes WHERE email = %s ORDER BY created_at DESC LIMIT 1",
            (email,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            logger.warning(f"No OTP found for email: {email}")
            return jsonify({'success': False, 'message': 'No OTP found for this email'})
        
        stored_otp, expires_at = result
        
        # Check if OTP is expired
        if datetime.datetime.now() > expires_at:
            logger.warning(f"OTP expired for email: {email}")
            return jsonify({'success': False, 'message': 'OTP has expired'})
        
        # Check if OTP matches
        if otp_code == stored_otp:
            logger.info(f"OTP verified successfully for {email}")
            return jsonify({'success': True, 'message': 'OTP verified successfully'})
        else:
            logger.warning(f"Invalid OTP provided for {email}")
            return jsonify({'success': False, 'message': 'Invalid OTP'})
            
    except Exception as e:
        logger.error(f"Database error during OTP verification for {email}: {e}")
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})

@app.route('/api/encode_face', methods=['POST'])
def encode_face():
    try:
        if 'image' not in request.files:
            logger.error("No image provided in encode_face request")
            return jsonify({'success': False, 'message': 'No image provided'})
        
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.error(f"Invalid image format: {filename}")
            return jsonify({'success': False, 'message': 'Invalid image format. Use JPG or PNG.'})
        
        # Load and process the image
        img = face_recognition.load_image_file(image_file)
        if img is None or img.size == 0:
            logger.error("Failed to load image")
            return jsonify({'success': False, 'message': 'Failed to load image'})
        
        # Enhance image for better face detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            enhanced_img = img
        
        # Detect faces
        face_locations = face_recognition.face_locations(enhanced_img, model="hog")
        if not face_locations:
            logger.warning("No face detected in image")
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Generate encodings with enhanced settings
        face_encodings = face_recognition.face_encodings(
            enhanced_img, 
            known_face_locations=face_locations, 
            num_jitters=5,  # Increased for better accuracy
            model="large"   # Use large model for better accuracy
        )
        
        if not face_encodings:
            logger.error("Could not generate face encoding")
            return jsonify({'success': False, 'message': 'Could not generate face encoding'})
        
        logger.info("Face encoding generated successfully")
        return jsonify({
            'success': True, 
            'encoding': face_encodings[0].tolist(),
            'message': 'Face encoding generated successfully'
        })
    except Exception as e:
        logger.error(f"Error encoding face: {str(e)}")
        return jsonify({'success': False, 'message': f'Error encoding face: {str(e)}'})

@app.route('/api/register_student', methods=['POST'])
def register_student():
    try:
        data = request.form
        email = data.get('email', '').strip()
        student_id = data.get('student_id', '').strip()
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        middle_name = data.get('middle_name', '').strip()
        course = data.get('course', '').strip()
        year_section = data.get('year_section', '').strip()
        face_encoding_data = data.get('face_encoding', '')
        
        # Validate inputs
        if not all([email, student_id, first_name, last_name, course, year_section, face_encoding_data]):
            logger.warning("Missing required fields in register_student request")
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if "@wmsu.edu.ph" not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        # Check if student already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id FROM students WHERE student_id = %s OR email = %s", 
                      (student_id, email))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            logger.warning(f"Student ID {student_id} or email {email} already exists")
            return jsonify({'success': False, 'message': 'Student ID or email already exists'})
        
        # Convert the face encoding string back to list and validate
        try:
            face_encoding = json.loads(face_encoding_data)
            if not isinstance(face_encoding, list) or len(face_encoding) != 128:
                raise ValueError("Invalid face encoding length")
            
            # Convert to string format for database storage
            encoding_str = "[" + ",".join(str(x) for x in face_encoding) + "]"
        except Exception as e:
            cursor.close()
            conn.close()
            logger.error(f"Invalid face encoding format: {e}")
            return jsonify({'success': False, 'message': 'Invalid face encoding format'})
        
        # Save student to database
        cursor.execute(
            """INSERT INTO students 
            (student_id, first_name, last_name, middle_name, course, year_section, email, face_encoding) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (student_id, first_name, last_name, middle_name or None, course, year_section, email, encoding_str)
        )
        
        # Save uploaded photo if provided
        if 'photo' in request.files:
            photo = request.files['photo']
            filename = secure_filename(photo.filename)
            if filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Create directory if it doesn't exist
                os.makedirs('student_photos', exist_ok=True)
                photo_path = f"student_photos/{student_id}.jpg"
                photo.save(photo_path)
                logger.info(f"Saved photo for {student_id} at {photo_path}")
                
                # Update photo path in database
                cursor.execute(
                    "UPDATE students SET photo_path = %s WHERE student_id = %s",
                    (photo_path, student_id)
                )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Reload known faces to include the new registration
        load_known_faces_from_db()
        
        logger.info(f"Student registered: {student_id} ({first_name} {last_name})")
        return jsonify({'success': True, 'message': 'Student registered successfully'})
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

def generate_frames():
    frame_idx = 0
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with cap_lock:
            frame = latest_frame.copy()  # keep native size
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

# Timer page
@app.route('/')
def timer_page():
    return send_from_directory('Timer', 'Timer.html')

# CamFootage page
@app.route('/camfootage')
def camfootage_page():
    return send_from_directory('CamFootage', 'CamFootage.html')

# Summary page
@app.route('/summary')
def summary_page():
    return send_from_directory(os.path.join('Timer', 'SumSession'), 'Summary.html')

# Student registration page
@app.route('/studentreg')
def studentreg_page():
    return send_from_directory('Registration', 'studentreg.html')

# Video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        ssl_context = None
        cert_path = 'cert.pem'
        key_path = 'key.pem'
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.load_cert_chain(cert_path, key_path)
            ssl_context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384')
            logger.info("Running server with HTTPS")
        else:
            logger.warning("SSL certificates not found. Running with HTTP")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context=ssl_context)
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
            logger.info("Attendance saved to attendance_log.csv")