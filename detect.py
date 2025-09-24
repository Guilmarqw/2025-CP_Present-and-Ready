import os
import cv2
import time
import dlib
import torch
import numpy as np
import datetime
import threading
import insightface
import mysql.connector
import smtplib
import random
import string
import json
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename

# =========================
# CONFIG
# =========================
RTSP_URL = "rtsp://admin:@101Pok3r5610@192.168.1.64:554/Streaming/Channels/101"
WEIGHTS_PATH = "yolov8n-face.pt"
STREAM_WIDTH, STREAM_HEIGHT = 3840, 2160
DETECT_EVERY = 8  # Default value, will be adjusted dynamically
CONF_THRESH = 0.45  
pose_embeddings = {}

STABLE_TOLERANCE_FRAMES = 20  # Increased for better lock stability
MAX_TRACKS = 128
EXPAND_BOX_RATIO = 0.4

ENABLE_RECOGNITION = True
TOLERANCE = 0.6  # InsightFace uses different distance metric
CONFIRMATION_THRESHOLD = 0.8  # Higher threshold for locking a track
KNOWN_DIR = "known_faces"

RECONNECT_COOLDOWN = 2.0
GRAB_SLEEP = 0.01
MAX_EMPTY_GRABS = 150

# Anti-spoofing configuration
LIVENESS_THRESHOLD = 150
MIN_FACE_SIZE = 40
HIGH_CONFIDENCE_THRESHOLD = 0.45
MEDIUM_CONFIDENCE_THRESHOLD = 0.55

# Performance optimization
PROCESSING_INTERVAL = 3
RESIZE_FACTOR = 0.75

# Distance settings
MAX_RECOGNITION_DISTANCE = 15
FACE_SIZE_FOR_DISTANCE = 80

# Locking configuration
LOCK_TIMEOUT_FRAMES = 60  # Frames before releasing lock if detection stops (~2s at 30 FPS)

# Face pose and feature thresholds (make them more lenient)
YAW_FRONTAL_THRESHOLD = 40  # Increased to allow wider head rotation
PITCH_FRONTAL_THRESHOLD = 35  # Increased to allow more vertical tilt
ROLL_THRESHOLD = 30  # Increased to allow more head tilt
YAW_SIDE_THRESHOLD = 20  # Decreased to make side profile detection easier
PITCH_UP_DOWN_THRESHOLD = 15  # Decreased to make up/down detection less strict
MAR_OPEN_THRESHOLD = 0.20  # Decreased to make open mouth detection more lenient
EAR_CLOSED_THRESHOLD = 0.35  # Increased to make closed eyes detection less restrictive
LIVENESS_THRESHOLD = 100  # Lowered to make liveness detection less strict



# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'fcee'
}

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'lawrencetilde@gmail.com',
    'password': 'ufwxvjacdtftfcof'
}

# InsightFace model
INSIGHTFACE_MODEL = "buffalo_l"

POSE_SEQUENCE = [
    'frontal', 'right', 'left', 'up', 'down', 'mouth_open', 'eyes_closed'
]

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Session & Attendance State
# =========================
# Will be set by frontend via /api/start_session
session_config = {
    'started_at': None,                 # datetime
    'late_threshold_minutes': 15,       # int
    'total_duration_minutes': 180,      # int
    'class_details': {}                 # arbitrary dict
}

# =========================
# Utilities
# =========================
def expand_box(x1, y1, x2, y2, w, h, scale=EXPAND_BOX_RATIO):
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * scale), int(bh * scale)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(w - 1, x2 + px); ny2 = min(h - 1, y2 + py)
    return nx1, ny1, nx2, ny2

def iou(box1, box2):
    """Calculate Intersection over Union for two boxes (x1,y1,x2,y2 format)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

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

def generate_invite_token(length=32):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

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

def calculate_ear(landmarks, eye_indices):
    eye_points = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(landmarks, mouth_indices):
    mouth_points = np.array([landmarks[i] for i in mouth_indices])
    A = np.linalg.norm(mouth_points[1] - mouth_points[7])
    B = np.linalg.norm(mouth_points[2] - mouth_points[6])
    C = np.linalg.norm(mouth_points[3] - mouth_points[5])
    D = np.linalg.norm(mouth_points[0] - mouth_points[4])
    mar = (A + B + C) / (3.0 * D)
    return mar

# =========================
# Load InsightFace model
# =========================
try:
    face_analysis = insightface.app.FaceAnalysis(name=INSIGHTFACE_MODEL)
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load InsightFace model: {e}")
    raise

# =========================
# Load known faces from database
# =========================
known_face_encodings = []
known_face_names = []
known_face_ids = []
known_face_types = []  # 'student' or 'instructor'

def load_known_faces_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_types
    known_face_encodings, known_face_names, known_face_ids, known_face_types = [], [], [], []  # Clear lists
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, first_name, last_name, face_encoding FROM students WHERE face_encoding IS NOT NULL")
        for (id, first_name, last_name, face_encoding) in cursor:
            try:
                if isinstance(face_encoding, str):
                    encoding_str = face_encoding.strip('[]')
                    if ', ' in encoding_str:
                        encoding_list = encoding_str.split(', ')
                    elif ',' in encoding_str:
                        encoding_list = encoding_str.split(',')
                    else:
                        encoding_list = encoding_str.split()
                    encoding = np.array([float(x) for x in encoding_list], dtype=np.float32)
                else:
                    encoding = np.frombuffer(face_encoding, dtype=np.float32)
                
                if encoding.size == 512:
                    known_face_encodings.append(encoding)
                    full_name = f"{first_name} {last_name}"
                    known_face_names.append(full_name)
                    known_face_ids.append(id)
                    known_face_types.append('student')
                    logger.info(f"Loaded student {full_name} ({id}) with encoding shape {encoding.shape}")
                else:
                    logger.warning(f"Invalid encoding size for student {id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for student {id}: {e}")
        cursor.close()
        conn.close()
        logger.info(f"Loaded {len(known_face_names)} known student faces from database")
    except Exception as e:
        logger.error(f"Failed to load student faces from database: {e}")

def load_known_instructors_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_types
    instructor_count = 0  # Track instructor faces separately
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT instructor_id, first_name, last_name, face_encoding FROM instructors WHERE face_encoding IS NOT NULL")
        for (id, first_name, last_name, face_encoding) in cursor:
            try:
                if isinstance(face_encoding, str):
                    encoding_str = face_encoding.strip('[]')
                    if ', ' in encoding_str:
                        encoding_list = encoding_str.split(', ')
                    elif ',' in encoding_str:
                        encoding_list = encoding_str.split(',')
                    else:
                        encoding_list = encoding_str.split()
                    encoding = np.array([float(x) for x in encoding_list], dtype=np.float32)
                else:
                    encoding = np.frombuffer(face_encoding, dtype=np.float32)
                
                if encoding.size == 512:
                    known_face_encodings.append(encoding)
                    full_name = f"{first_name} {last_name}"
                    known_face_names.append(full_name)
                    known_face_ids.append(id)
                    known_face_types.append('instructor')
                    instructor_count += 1
                    logger.info(f"Loaded instructor {full_name} ({id}) with encoding shape {encoding.shape}")
                else:
                    logger.warning(f"Invalid encoding size for instructor {id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for instructor {id}: {e}")
        cursor.close()
        conn.close()
        logger.info(f"Loaded {instructor_count} known instructor faces from database")
    except Exception as e:
        logger.error(f"Failed to load instructor faces from database: {e}")

# Initialize known faces
load_known_faces_from_db()
load_known_instructors_from_db()

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
# Laptop camera capture
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
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        if not cap.isOpened():
            logger.error("Cannot open RTSP stream. Check if stream is accessible.")
            return False
        logger.info("RTSP stream connected.")
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
                logger.warning("Camera stalled. Reconnecting...")
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
locked_tracks = {}  # Dict: id -> {'track': tr, 'last_seen': frame_idx, 'lock_start': frame_idx, 'type': 'student' or 'instructor'}
attendance = {}
tracking_history = {}

def mark_attendance(name, id, type):
    if type != 'student':
        return  # Only mark for students
    
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    today = current_time.strftime("%Y-%m-%d")
    if id in attendance and attendance[id].startswith(today):
        last_time_str = attendance[id]
        last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
        time_diff = (current_time - last_time).total_seconds() / 3600
        
        if time_diff < 4:
            return
    
    try:
        attendance[id] = time_str
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO attendance (student_id, name, timestamp) VALUES (%s, %s, %s)",
                (id, name, time_str)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Attendance recorded: {name} ({id}) at {time_str}")
        except Exception as e:
            logger.error(f"Failed to save attendance to database: {e}")
    except Exception as e:
        logger.error(f"Failed to mark attendance for {name}: {e}")

def update_trackers(rgb, frame, frame_idx):
    global tracks, locked_tracks
    h, w = frame.shape[:2]
    kept = []
    to_remove_locks = []
    
    # Check for timeout on locked tracks
    for id, lock_info in list(locked_tracks.items()):
        if frame_idx - lock_info['last_seen'] > LOCK_TIMEOUT_FRAMES:
            to_remove_locks.append(id)
            logger.info(f"Lock timeout for {id} - releasing track")
    
    for sid in to_remove_locks:
        del locked_tracks[sid]
    
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
        tr["last_seen"] = frame_idx
        
        # If track is locked, maintain identity even without re-recognition
        is_locked = tr["id"] in locked_tracks and locked_tracks[tr["id"]]['track'] is tr
        if is_locked:
            tr["confidence"] = max(tr["confidence"], 0.8)  # Maintain high confidence for locked tracks
            locked_tracks[tr["id"]]['last_seen'] = frame_idx
            logger.debug(f"Maintaining locked track for {tr['name']} ({tr['id']})")
            
            # Update tracking history for locked tracks
            if tr["id"] not in tracking_history:
                tracking_history[tr["id"]] = {
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "recognition_count": 1,
                    "name": tr["name"]
                }
            else:
                tracking_history[tr["id"]]["last_seen"] = frame_idx
                tracking_history[tr["id"]]["recognition_count"] += 1
        
        if tr["confidence"] > 0.7:
            color = (0, 255, 0)
        elif tr["confidence"] > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{tr['name']} ({tr['confidence']:.2f})" if ENABLE_RECOGNITION else "Face"
        if is_locked:
            label += " [LOCKED]"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if tr["id"] and tr["id"] in tracking_history:
            hist = tracking_history[tr["id"]]
            duration = (frame_idx - hist["first_seen"]) / 30
            mins = int(duration // 60)
            secs = int(duration % 60)
            time_text = f"Tracked: {mins}m {secs}s"
            cv2.putText(frame, time_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if frame_idx - tr["last_seen"] <= STABLE_TOLERANCE_FRAMES:
            kept.append(tr)
    tracks = kept

def cleanup_locked_tracks():
    global locked_tracks
    current_frame = 0  # This would need to be updated with the current frame index
    
    to_remove = []
    for id, lock_info in locked_tracks.items():
        if current_frame - lock_info['last_seen'] > LOCK_TIMEOUT_FRAMES * 2:  # Double timeout for cleanup
            to_remove.append(id)
            logger.info(f"Cleaning up locked track for {id}")
    
    for id in to_remove:
        del locked_tracks[id]

def enhanced_recognize_face(face_image, face_width_pixels, tolerance=0.6, is_locked_track=False):
    try:
        distance = estimate_distance(face_width_pixels)
        if distance > MAX_RECOGNITION_DISTANCE:
            logger.info(f"Face too far for recognition: {distance:.1f}m")
            return "Unknown", None, float('inf'), distance, 0.0, None
        
        if not detect_liveness_cctv(face_image):
            logger.info("Liveness detection failed, but continuing for CCTV")
        
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = face_image
        
        faces = face_analysis.get(enhanced)
        
        if not faces:
            return "Unknown", None, float('inf'), distance, 0.0, None
        
        face_embedding = faces[0].embedding
        
        # Check against students and instructors
        similarities = []
        names = known_face_names
        ids = known_face_ids
        types = known_face_types
        for known_embedding in known_face_encodings:
            dot_product = np.dot(face_embedding, known_embedding)
            norm_a = np.linalg.norm(face_embedding)
            norm_b = np.linalg.norm(known_embedding)
            similarity = dot_product / (norm_a * norm_b)
            similarities.append(similarity)
        
        if similarities:
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            distance_metric = 1 - best_similarity
            
            if is_locked_track or best_similarity >= (1 - tolerance):
                name = names[best_match_index]
                id = ids[best_match_index]
                type = types[best_match_index]
                if type == 'instructor':
                    name = "Instructor: " + name
                return name, id, distance_metric, distance, best_similarity, type
            else:
                logger.info(f"Closest match: {names[best_match_index]} with similarity {best_similarity:.4f} (threshold: {1 - tolerance:.4f})")
        
        return "Unknown", None, float('inf'), distance, 0.0, None
        
    except Exception as e:
        logger.error(f"Error in enhanced_recognize_face: {e}")
        return "Unknown", None, float('inf'), distance, 0.0, None

def detect_liveness_cctv(face_image):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Relaxed threshold for better detection in well-lit conditions
        adjusted_threshold = LIVENESS_THRESHOLD * 0.8  # Changed from 0.6 to 0.8
        if fm < adjusted_threshold:
            logger.warning(f"Liveness detection failed: variance {fm} < threshold {adjusted_threshold}")
            return False
        logger.info(f"Liveness detection passed: variance {fm}")
        return True
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return True  # Allow progression for CCTV scenarios
    
def estimate_distance(face_width_pixels):
    if face_width_pixels < 20:
        return float('inf')
    estimated_distance = (FACE_SIZE_FOR_DISTANCE * 2) / face_width_pixels
    return estimated_distance

def recognize_face_with_anti_spoofing(face_image, tolerance=0.6):
    if not detect_liveness_cctv(face_image):
        return "Unknown", None, float('inf'), False, 0.0
    name, student_id, distance, est_distance, confidence = enhanced_recognize_face(face_image, tolerance)
    return name, student_id, distance, est_distance, True, confidence

def refresh_with_detections(frame, rgb, frame_idx):
    global tracks, locked_tracks, DETECT_EVERY
    if len(tracks) > MAX_TRACKS:
        tracks = tracks[:MAX_TRACKS]
    
    # Adjust detection frequency based on number of tracks
    DETECT_EVERY = 4 if len(tracks) < 5 else 8
    
    h, w = frame.shape[:2]
    
    if frame_idx % DETECT_EVERY != 0:
        return
    
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    frame_eq = enhance_lighting(small_frame)
    results = yolo.predict(source=frame_eq, verbose=False, conf=CONF_THRESH, 
                          imgsz=640, device=DEVICE)
    
    dets = []
    if results:
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                
                x1 = int(max(0, x1 / RESIZE_FACTOR)); y1 = int(max(0, y1 / RESIZE_FACTOR))
                x2 = int(min(w-1, x2 / RESIZE_FACTOR)); y2 = int(min(h-1, y2 / RESIZE_FACTOR))
                
                if conf >= CONF_THRESH and x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, conf))
    
    logger.info(f"Frame {frame_idx}: Detected {len(dets)} faces with conf > {CONF_THRESH}")
    new_tracks = []
    
    # Store recently lost locked tracks for re-identification
    recently_lost = {sid: info for sid, info in locked_tracks.items() if frame_idx - info['last_seen'] <= 90}  # 3 seconds
    
    for (x1, y1, x2, y2, conf) in dets:
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_BOX_RATIO)
        
        face_region = rgb[ey1:ey2, ex1:ex2]
        face_width = x2 - x1
        
        if face_region.size == 0 or face_region.shape[0] < MIN_FACE_SIZE or face_region.shape[1] < MIN_FACE_SIZE:
            logger.info(f"Face region too small for recognition: {face_region.shape}")
            name = "Unknown"
            id = None
            type = None
            confidence = 0.0
        else:
            # Check if this detection matches any locked track
            matched_locked = None
            for sid, lock_info in list(locked_tracks.items()) + list(recently_lost.items()):
                lock_box = lock_info['track']['box']
                overlap_iou = iou((x1, y1, x2, y2), lock_box)
                if overlap_iou > 0.5:  # Good overlap with a locked track
                    matched_locked = sid
                    logger.info(f"Matched locked track for {lock_info['track']['name']} with IoU {overlap_iou:.2f}")
                    break
            
            # If this is a locked track, maintain identity even with lower confidence
            is_locked = matched_locked and matched_locked in locked_tracks
            name, id, distance, estimated_distance, confidence, type = enhanced_recognize_face(
                face_region, face_width, TOLERANCE, is_locked_track=is_locked
            )
            
            # If we have a locked track match but recognition returned Unknown, use the locked identity
            if is_locked and name == "Unknown":
                name = locked_tracks[matched_locked]['track']['name']
                id = matched_locked
                confidence = max(0.7, locked_tracks[matched_locked]['track']['confidence'] * 0.9)  # Slight decay
                type = locked_tracks[matched_locked]['type']
                logger.info(f"Maintaining locked identity for {name} despite low recognition confidence")
            
            if name != "Unknown" and not is_locked:
                logger.info(f"Recognized {name} (ID: {id}) with confidence {confidence:.4f} at {estimated_distance:.1f}m")
                mark_attendance(name, id, type)
        
        # Create or update track
        if matched_locked and matched_locked in locked_tracks:
            # Update existing locked track
            tr = locked_tracks[matched_locked]['track']
            tr['last_seen'] = frame_idx
            tr['confidence'] = max(tr['confidence'], confidence)
            tr['box'] = (x1, y1, x2, y2)
            
            # Update tracker position
            try:
                tr["tracker"].start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
            except Exception as e:
                logger.error(f"Tracker update error: {e}")
                # Create new tracker if update fails
                dtracker = dlib.correlation_tracker()
                dtracker.start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
                tr["tracker"] = dtracker
                
            new_tracks.append(tr)
        else:
            # Create new track
            dtracker = dlib.correlation_tracker()
            try:
                dtracker.start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
                tr = {
                    "tracker": dtracker, 
                    "name": name, 
                    "id": id,
                    "type": type,
                    "confidence": confidence,
                    "last_seen": frame_idx, 
                    "box": (x1, y1, x2, y2),
                    "recognition_count": 1 if id else 0
                }
                new_tracks.append(tr)
                
                # Lock the track if we have high confidence or multiple confirmations
                if id and (confidence >= CONFIRMATION_THRESHOLD or tr.get("recognition_count", 0) >= 3):
                    locked_tracks[id] = {
                        'track': tr,
                        'last_seen': frame_idx,
                        'lock_start': frame_idx,
                        'type': type
                    }
                    logger.info(f"Locked track for {type} {name} ({id}) with confidence {confidence:.4f}")
                    
            except Exception as e:
                logger.error(f"Tracker error: {e}")
                continue
    
    # Keep only tracks that are either new or locked
    tracks[:] = [tr for tr in new_tracks if tr['id'] in locked_tracks or tr not in tracks]

# =========================
# Flask streaming & API
# =========================
app = Flask(__name__)
CORS(app)

# Serve student photos
@app.route('/student_photos/<filename>')
def serve_photo(filename):
    return send_from_directory('student_photos', filename)

# Serve instructor photos
os.makedirs('instructor_photos', exist_ok=True)

@app.route('/instructor_photos/<filename>')
def serve_instructor_photo(filename):
    return send_from_directory('instructor_photos', filename)

@app.route('/api/get_students', methods=['GET'])
def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT student_id, first_name, last_name, course, year_section, photo_path FROM students")
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format student data for frontend
        formatted_students = [
            {
                'idNumber': s['student_id'],
                'name': f"{s['first_name']} {s['last_name']}",
                'course': s['course'],
                'section': s['year_section'],
                'photo': s['photo_path'] if s['photo_path'] else f"https://ui-avatars.com/api/?name={s['first_name']}+{s['last_name']}&background=random"
            }
            for s in students
        ]
        
        return jsonify({'success': True, 'students': formatted_students})
    except Exception as e:
        logger.error(f"Error fetching students: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_instructors', methods=['GET'])
def get_instructors():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT instructor_id, first_name, last_name, department, designation, photo_path FROM instructors")
        instructors = cursor.fetchall()
        cursor.close()
        conn.close()
        
        formatted_instructors = [
            {
                'idNumber': i['instructor_id'],
                'name': f"{i['first_name']} {i['last_name']}",
                'department': i['department'],
                'designation': i['designation'],
                'photo': i['photo_path'] if i['photo_path'] else f"https://ui-avatars.com/api/?name={i['first_name']}+{i['last_name']}&background=random"
            }
            for i in instructors
        ]
        
        return jsonify({'success': True, 'instructors': formatted_instructors})
    except Exception as e:
        logger.error(f"Error fetching instructors: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/generate_invite', methods=['POST'])
def generate_invite():
    try:
        token = generate_invite_token()
        expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO invites (token, expires_at) VALUES (%s, %s)",
            (token, expires_at)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        invite_link = f"{request.host_url}join/{token}"
        return jsonify({'success': True, 'link': invite_link})
    except Exception as e:
        logger.error(f"Error generating invite: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/join/<token>')
def join(token):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT expires_at, used FROM invites WHERE token = %s",
            (token,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return jsonify({'success': False, 'message': 'Invalid invite link'}), 404
        
        expires_at, used = result
        current_time = datetime.datetime.now()
        
        if used:
            return jsonify({'success': False, 'message': 'Invite link already used'}), 403
        if current_time > expires_at:
            return jsonify({'success': False, 'message': 'Invite link has expired'}), 403
            
        # Mark invite as used
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE invites SET used = TRUE WHERE token = %s",
            (token,)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return redirect(url_for('studentreg_page'))
    except Exception as e:
        logger.error(f"Error validating invite: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_student', methods=['POST'])
def update_student():
    try:
        data = request.json
        student_id = data.get('student_id')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        course = data.get('course')
        year_section = data.get('year_section')
        
        if not all([student_id, first_name, last_name, course, year_section]):
            return jsonify({'success': False, 'message': 'All fields are required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE students 
               SET first_name = %s, last_name = %s, course = %s, year_section = %s 
               WHERE student_id = %s""",
            (first_name, last_name, course, year_section, student_id)
        )
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})
            
        conn.commit()
        cursor.close()
        conn.close()
        
        load_known_faces_from_db()  # Refresh known faces if name changed
        return jsonify({'success': True, 'message': 'Student updated successfully'})
    except Exception as e:
        logger.error(f"Error updating student: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_student', methods=['POST'])
def delete_student():
    try:
        student_id = request.json.get('student_id')
        if not student_id:
            return jsonify({'success': False, 'message': 'Student ID is required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})
            
        conn.commit()
        cursor.close()
        conn.close()
        
        # Remove photo if exists
        photo_path = f"student_photos/{student_id}.jpg"
        if os.path.exists(photo_path):
            os.remove(photo_path)
            logger.info(f"Deleted photo for {student_id}")
            
        load_known_faces_from_db()  # Refresh known faces
        if student_id in locked_tracks:
            del locked_tracks[student_id]
        return jsonify({'success': True, 'message': 'Student deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_instructor', methods=['POST'])
def update_instructor():
    try:
        data = request.json
        instructor_id = data.get('instructor_id')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        department = data.get('department')
        designation = data.get('designation')
        
        if not all([instructor_id, first_name, last_name, department, designation]):
            return jsonify({'success': False, 'message': 'All fields are required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE instructors 
               SET first_name = %s, last_name = %s, department = %s, designation = %s 
               WHERE instructor_id = %s""",
            (first_name, last_name, department, designation, instructor_id)
        )
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Instructor not found'})
            
        conn.commit()
        cursor.close()
        conn.close()
        
        load_known_instructors_from_db()  # Refresh
        return jsonify({'success': True, 'message': 'Instructor updated successfully'})
    except Exception as e:
        logger.error(f"Error updating instructor: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_instructor', methods=['POST'])
def delete_instructor():
    try:
        instructor_id = request.json.get('instructor_id')
        if not instructor_id:
            return jsonify({'success': False, 'message': 'Instructor ID is required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM instructors WHERE instructor_id = %s", (instructor_id,))
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Instructor not found'})
            
        conn.commit()
        cursor.close()
        conn.close()
        
        # Remove photo if exists
        photo_path = f"instructor_photos/{instructor_id}.jpg"
        if os.path.exists(photo_path):
            os.remove(photo_path)
            logger.info(f"Deleted photo for instructor {instructor_id}")
            
        load_known_instructors_from_db()  # Refresh
        if instructor_id in locked_tracks:
            del locked_tracks[instructor_id]
        return jsonify({'success': True, 'message': 'Instructor deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting instructor: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/send_otp', methods=['POST'])
def send_otp():
    email = request.json.get('email', '').strip()
    
    if not email or "@wmsu.edu.ph" not in email:
        logger.warning(f"Invalid email received: {email}")
        return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
    
    otp_code = generate_otp()
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=10)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM otp_codes WHERE email = %s", (email,))
        
        cursor.execute(
            "INSERT INTO otp_codes (email, otp_code, expires_at) VALUES (%s, %s, %s)",
            (email, otp_code, expires_at)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
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
        
        if datetime.datetime.now() > expires_at:
            logger.warning(f"OTP expired for email: {email}")
            return jsonify({'success': False, 'message': 'OTP has expired'})
        
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
        logger.info(f"Received encode_face request for pose: {request.form.get('current_pose')}")
        
        if 'image' not in request.files:
            logger.error("No image provided in encode_face request")
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        current_pose = request.form.get('current_pose', POSE_SEQUENCE[0])
        if current_pose not in POSE_SEQUENCE:
            logger.warning(f"Invalid pose received: {current_pose}, defaulting to {POSE_SEQUENCE[0]}")
            current_pose = POSE_SEQUENCE[0]
        current_pose_index = POSE_SEQUENCE.index(current_pose)
        
        image_file = request.files['image']
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            logger.error("Failed to load image: Invalid or empty image data")
            return jsonify({
                'success': False,
                'message': 'Failed to load image. Ensure your webcam is working and try again.',
                'current_pose': current_pose,
                'next_pose': current_pose
            }), 400
        
        enhanced_img = enhance_lighting(img)
        
        if not detect_liveness_cctv(enhanced_img):
            logger.warning("Liveness detection failed")
            return jsonify({
                'success': False,
                'message': 'Ensure good lighting and clear face visibility.',
                'current_pose': current_pose,
                'next_pose': current_pose
            }), 400
        
        faces = face_analysis.get(enhanced_img)
        if not faces:
            logger.warning("No face detected in image during registration")
            return jsonify({
                'success': False,
                'message': 'No face detected. Please ensure your face is clearly visible and centered.',
                'current_pose': current_pose,
                'next_pose': current_pose
            }), 400
        
        face = faces[0]
        face_embedding = face.embedding
        yaw, pitch, roll = face.pose
        landmarks = face.landmark_2d_106
        
        left_eye_indices = [96, 97, 98, 99, 100, 101]
        left_ear = calculate_ear(landmarks, left_eye_indices)
        right_eye_indices = [90, 91, 92, 93, 94, 95]
        right_ear = calculate_ear(landmarks, right_eye_indices)
        mouth_indices = [76, 77, 78, 79, 80, 81, 82, 83]
        mar = calculate_mar(landmarks, mouth_indices)
        
        pose_results = {
            'is_frontal': bool(abs(yaw) <= 30 and abs(pitch) <= 25 and abs(roll) <= 25),  # Stricter for 4K
            'is_right': bool(yaw <= -15),  # Adjusted for clarity
            'is_left': bool(yaw >= 15),
            'is_up': bool(pitch <= -10),
            'is_down': bool(pitch >= 10),
            'is_mouth_open': bool(mar >= 0.25),
            'is_eyes_closed': bool((left_ear + right_ear) / 2 <= 0.3)
        }
        
        logger.info(f"Pose results for {current_pose}: {pose_results}, yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}, mar={mar:.3f}, left_ear={left_ear:.3f}, right_ear={right_ear:.3f}")
        
        pose_satisfied = False
        message = ""
        if current_pose == 'frontal' and pose_results['is_frontal']:
            pose_satisfied = True
            message = "Frontal pose detected successfully."
        elif current_pose == 'right' and pose_results['is_right']:
            pose_satisfied = True
            message = "Right pose detected successfully."
        elif current_pose == 'left' and pose_results['is_left']:
            pose_satisfied = True
            message = "Left pose detected successfully."
        elif current_pose == 'up' and pose_results['is_up']:
            pose_satisfied = True
            message = "Upward pose detected successfully."
        elif current_pose == 'down' and pose_results['is_down']:
            pose_satisfied = True
            message = "Downward pose detected successfully."
        elif current_pose == 'mouth_open' and pose_results['is_mouth_open']:
            pose_satisfied = True
            message = "Mouth open detected successfully."
        elif current_pose == 'eyes_closed' and pose_results['is_eyes_closed']:
            pose_satisfied = True
            message = "Eyes closed detected successfully."
        else:
            message = f"Please adjust to {current_pose} pose. Ensure good lighting and clear face visibility."
        
        if pose_satisfied:
            pose_embeddings[current_pose] = face_embedding.tolist()
            next_pose_index = min(current_pose_index + 1, len(POSE_SEQUENCE) - 1)
            next_pose = POSE_SEQUENCE[next_pose_index]
            logger.info(f"Pose {current_pose} satisfied, advancing to {next_pose}")
        else:
            next_pose = current_pose
            logger.info(f"Pose {current_pose} not satisfied, retrying")
        
        encoding_response = face_embedding.tolist() if current_pose == 'frontal' else []
        
        return jsonify({
            'success': pose_satisfied,
            'message': message,
            'current_pose': current_pose,
            'next_pose': next_pose,
            'encoding': encoding_response,
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
            'mar': float(mar),
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'is_frontal': bool(pose_results['is_frontal']),
            'is_left': bool(pose_results['is_left']),
            'is_right': bool(pose_results['is_right']),
            'is_up': bool(pose_results['is_up']),
            'is_down': bool(pose_results['is_down']),
            'is_mouth_open': bool(pose_results['is_mouth_open']),
            'is_eyes_closed': bool(pose_results['is_eyes_closed'])
        })
    except Exception as e:
        logger.error(f"Error encoding face: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error encoding face: {str(e)}',
            'current_pose': current_pose,
            'next_pose': current_pose,
            'encoding': []
        }), 500

        logger.info(f"Received encode_face request for pose: {request.form.get('current_pose')}")

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
        password = data.get('password', '').strip()
        
        if not all([email, student_id, first_name, last_name, course, year_section, password]):
            logger.warning("Missing required fields in register_student request")
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if "@wmsu.edu.ph" not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id FROM students WHERE student_id = %s OR email = %s", 
                      (student_id, email))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            logger.warning(f"Student ID {student_id} or email {email} already exists")
            return jsonify({'success': False, 'message': 'Student ID or email already exists'})
        
        # Average embeddings from multiple poses
        if len(pose_embeddings) >= 3:  # Require at least 3 poses
            avg_embedding = np.mean([np.array(pose_embeddings[p]) for p in POSE_SEQUENCE if p in pose_embeddings], axis=0)
            encoding_str = "[" + ",".join(str(x) for x in avg_embedding) + "]"
        else:
            face_encoding_data = data.get('face_encoding', '')
            try:
                face_encoding = json.loads(face_encoding_data)
                if not isinstance(face_encoding, list) or len(face_encoding) != 512:
                    raise ValueError("Invalid face encoding length")
                encoding_str = "[" + ",".join(str(x) for x in face_encoding) + "]"
            except Exception as e:
                cursor.close()
                conn.close()
                logger.error(f"Invalid face encoding format: {e}")
                return jsonify({'success': False, 'message': 'Invalid face encoding format'})
        
        photo_path = None
        if 'photo' in request.files:
            photo = request.files['photo']
            filename = secure_filename(photo.filename)
            if filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.makedirs('student_photos', exist_ok=True)
                photo_path = f"student_photos/{student_id}.jpg"
                photo.save(photo_path)
                logger.info(f"Saved photo for {student_id} at {photo_path}")
                
                img = cv2.imread(photo_path)
                if img is not None:
                    faces = face_analysis.get(img)
                    if faces:
                        photo_embedding = faces[0].embedding
                        scan_embedding = np.array(avg_embedding if len(pose_embeddings) >= 3 else face_encoding, dtype=np.float32)
                        dot_product = np.dot(photo_embedding, scan_embedding)
                        norm_photo = np.linalg.norm(photo_embedding)
                        norm_scan = np.linalg.norm(scan_embedding)
                        similarity = dot_product / (norm_photo * norm_scan)
                        distance = 1 - similarity
                        if distance > TOLERANCE:
                            os.remove(photo_path)
                            cursor.close()
                            conn.close()
                            return jsonify({'success': False, 'message': 'Uploaded photo does not match the face scan.'})
                    else:
                        os.remove(photo_path)
                        cursor.close()
                        conn.close()
                        return jsonify({'success': False, 'message': 'No face detected in uploaded photo.'})
        
        cursor.execute(
            """INSERT INTO students 
            (student_id, first_name, last_name, middle_name, course, year_section, email, face_encoding, photo_path, password) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (student_id, first_name, last_name, middle_name or None, course, year_section, email, encoding_str, photo_path, password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        load_known_faces_from_db()
        pose_embeddings.clear()  # Reset after registration
        
        logger.info(f"Student registered: {student_id} ({first_name} {last_name})")
        return jsonify({'success': True, 'message': 'Student registered successfully'})
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/register_instructor', methods=['POST'])
def register_instructor():
    try:
        data = request.form
        email = data.get('email', '').strip()
        instructor_id = data.get('instructor_id', '').strip()
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        middle_name = data.get('middle_name', '').strip()
        department = data.get('department', '').strip()
        designation = data.get('designation', '').strip()
        password = data.get('password', '').strip()
        
        if not all([email, instructor_id, first_name, last_name, department, designation, password]):
            logger.warning("Missing required fields in register_instructor request")
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if "@wmsu.edu.ph" not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT instructor_id FROM instructors WHERE instructor_id = %s OR email = %s", 
                      (instructor_id, email))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            logger.warning(f"Instructor ID {instructor_id} or email {email} already exists")
            return jsonify({'success': False, 'message': 'Instructor ID or email already exists'})
        
        # Average embeddings from multiple poses
        if len(pose_embeddings) >= 3:  # Require at least 3 poses
            avg_embedding = np.mean([np.array(pose_embeddings[p]) for p in POSE_SEQUENCE if p in pose_embeddings], axis=0)
            encoding_str = "[" + ",".join(str(x) for x in avg_embedding) + "]"
        else:
            face_encoding_data = data.get('face_encoding', '')
            try:
                face_encoding = json.loads(face_encoding_data)
                if not isinstance(face_encoding, list) or len(face_encoding) != 512:
                    raise ValueError("Invalid face encoding length")
                encoding_str = "[" + ",".join(str(x) for x in face_encoding) + "]"
            except Exception as e:
                cursor.close()
                conn.close()
                logger.error(f"Invalid face encoding format: {e}")
                return jsonify({'success': False, 'message': 'Invalid face encoding format'})
        
        photo_path = None
        if 'photo' in request.files:
            photo = request.files['photo']
            filename = secure_filename(photo.filename)
            if filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.makedirs('instructor_photos', exist_ok=True)
                photo_path = f"instructor_photos/{instructor_id}.jpg"
                photo.save(photo_path)
                logger.info(f"Saved photo for instructor {instructor_id} at {photo_path}")
                
                img = cv2.imread(photo_path)
                if img is not None:
                    faces = face_analysis.get(img)
                    if faces:
                        photo_embedding = faces[0].embedding
                        scan_embedding = np.array(avg_embedding if len(pose_embeddings) >= 3 else face_encoding, dtype=np.float32)
                        dot_product = np.dot(photo_embedding, scan_embedding)
                        norm_photo = np.linalg.norm(photo_embedding)
                        norm_scan = np.linalg.norm(scan_embedding)
                        similarity = dot_product / (norm_photo * norm_scan)
                        distance = 1 - similarity
                        if distance > TOLERANCE:
                            os.remove(photo_path)
                            cursor.close()
                            conn.close()
                            return jsonify({'success': False, 'message': 'Uploaded photo does not match the face scan.'})
                    else:
                        os.remove(photo_path)
                        cursor.close()
                        conn.close()
                        return jsonify({'success': False, 'message': 'No face detected in uploaded photo.'})
        
        cursor.execute(
            """INSERT INTO instructors 
            (instructor_id, first_name, last_name, middle_name, department, designation, email, face_encoding, photo_path, password) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (instructor_id, first_name, last_name, middle_name or None, department, designation, email, encoding_str, photo_path, password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        load_known_instructors_from_db()
        pose_embeddings.clear()  # Reset after registration
        
        logger.info(f"Instructor registered: {instructor_id} ({first_name} {last_name})")
        return jsonify({'success': True, 'message': 'Instructor registered successfully'})
    except Exception as e:
        logger.error(f"Instructor registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        with cap_lock:
            cam_ok = cap is not None and cap.isOpened()
        db_ok = get_db_connection().is_connected()
        model_ok = yolo is not None and face_analysis is not None
        return jsonify({
            'success': True,
            'camera': cam_ok,
            'database': db_ok,
            'models': model_ok,
            'active_tracks': len(tracks),
            'locked_tracks': len(locked_tracks)
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'success': False, 'message': str(e)})

def generate_frames():
    frame_idx = 0
    while True:
        with cap_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update trackers and refresh detections
        update_trackers(rgb, frame, frame_idx)
        refresh_with_detections(frame, rgb, frame_idx)
        frame_idx += 1
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def timer_page():
    return render_template('Timer.html')

@app.route('/camfootage')
def camfootage_page():
    return render_template('CamFootage.html')

@app.route('/summary')
def summary_page():
    return render_template('Summary.html')

@app.route('/studentreg')
def studentreg_page():
    return render_template('studentreg.html')

@app.route('/instructorreg')
def instructor_reg_page():
    return render_template('instructorreg.html')

@app.route('/AdminDB')
def admin_db_page():
    return render_template('AdminDB.html')

@app.route('/StudentDB')
def student_db_page():
    return render_template('StudentDB.html')

@app.route('/InstructorDB')
def instructor_db_page():
    return render_template('InstructorDB.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

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