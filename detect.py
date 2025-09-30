from functools import wraps
import os
import sys
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
from flask import Flask, Response, g, make_response, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque, defaultdict
import queue
from scipy.optimize import linear_sum_assignment
import bcrypt
import secrets
from datetime import datetime, timedelta
import hashlib


# =========================
# CONFIG
# =========================
RTSP_URL = "rtsp://admin:@101Pok3r5610@192.168.1.64:554/Streaming/Channels/101"
WEIGHTS_PATH = "yolov8n-face.pt"
STREAM_WIDTH, STREAM_HEIGHT = 3840, 2160
DETECT_EVERY = 8  # Default value, will be adjusted dynamically
CONF_THRESH = 0.35 
pose_embeddings = {}

# Add these global variables near the top with other globals (after line ~370 in your original code)
camera_available = False
use_dummy_feed = False
dummy_frame = None
latest_frame = None
stop_flag = False

STABLE_TOLERANCE_FRAMES = 20  # Increased for better lock stability
MAX_TRACKS = 128
EXPAND_BOX_RATIO = 0.4

PASSWORD_RESET_EXPIRE_HOURS = 24
OTP_RESEND_COOLDOWN = 30  # seconds
MAX_OTP_ATTEMPTS = 3

ENABLE_RECOGNITION = True
TOLERANCE = 0.6  # InsightFace uses different distance metric
CONFIRMATION_THRESHOLD = 0.8  # Higher threshold for locking a track
KNOWN_DIR = "known_faces"

RECONNECT_COOLDOWN = 2.0
GRAB_SLEEP = 0.01
MAX_EMPTY_GRABS = 150

# Anti-spoofing configuration
LIVENESS_THRESHOLD = 150
MIN_FACE_SIZE = 15
HIGH_CONFIDENCE_THRESHOLD = 0.45
MEDIUM_CONFIDENCE_THRESHOLD = 0.55

# Performance optimization
PROCESSING_INTERVAL = 3
RESIZE_FACTOR = 1

# Distance settings
MAX_RECOGNITION_DISTANCE = 8
FACE_SIZE_FOR_DISTANCE = 80

# Locking configuration
LOCK_TIMEOUT_FRAMES = 60  # Frames before releasing lock if detection stops (~2s at 30 FPS)

# Face pose and feature thresholds (make them more lenient)
YAW_FRONTAL_THRESHOLD = 45  # Increased from 40
PITCH_FRONTAL_THRESHOLD = 40  # Increased from 35
ROLL_THRESHOLD = 35  # Increased from 20
YAW_SIDE_THRESHOLD = 5  # Decreased from 8 - much more sensitive
PITCH_UP_DOWN_THRESHOLD = 5  # Decreased from 8 - much more sensitive
MAR_OPEN_THRESHOLD = 0.15  # Decreased from 0.20
EAR_CLOSED_THRESHOLD = 0.4  # Increased from 0.35
LIVENESS_THRESHOLD = 100  # Lowered to make liveness detection less strict



# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facesys',  # fcee for backup
    'autocommit': False,  # Explicitly set to False for transaction control
    'pool_name': 'mypool',
    'pool_size': 5
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
# Thread-safe state manager
# Add these utility functions before the Flask routes

def create_dummy_frame():
    """Create a dummy frame when camera is not available"""
    global dummy_frame
    if dummy_frame is None:
        # Create a black frame with text
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "CAMERA NOT AVAILABLE", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(dummy_frame, "Face Recognition System Running", (80, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(dummy_frame, "Connect RTSP camera to enable detection", (30, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return dummy_frame.copy()



def hash_password(password):
    """Hash a password for storing in database"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def generate_reset_token():
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(48)

def authenticate_user(email, password):
    """Authenticate user against database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check admins table first
        cursor.execute(
            "SELECT admin_id as user_id, first_name, last_name, password_hash, role, 'admin' as user_type FROM admins WHERE email = %s AND status = 'active'",
            (email,)
        )
        user = cursor.fetchone()
        
        # Check faculty table if not found in admins
        if not user:
            cursor.execute(
                "SELECT faculty_id as user_id, first_name, last_name, password_hash, role, 'faculty' as user_type FROM faculty WHERE email = %s AND status = 'active'",
                (email,)
            )
            user = cursor.fetchone()
        
        # Check students table if not found in faculty
        if not user:
            cursor.execute(
                "SELECT student_id as user_id, first_name, last_name, password_hash, 'student' as role, 'student' as user_type FROM students WHERE email = %s AND status = 'active'",
                (email,)
            )
            user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if user and verify_password(password, user['password_hash']):
            return user
        return None
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

def create_user_session(user_id, user_type):
    """Create a new user session"""
    try:
        session_token = generate_session_token()
        expires_at = datetime.now() + timedelta(hours=24)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clean up old sessions for this user
        cursor.execute(
            "DELETE FROM user_sessions WHERE user_id = %s AND user_type = %s",
            (user_id, user_type)
        )
        
        # Create new session
        cursor.execute(
            "INSERT INTO user_sessions (user_id, user_type, session_token, expires_at) VALUES (%s, %s, %s, %s)",
            (user_id, user_type, session_token, expires_at)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return session_token
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return None

def get_last_otp_time(email, purpose='password_reset'):
    """Get the time when the last OTP was sent"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT created_at FROM otp_codes WHERE email = %s AND purpose = %s ORDER BY created_at DESC LIMIT 1",
            (email, purpose)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return result[0]
        return None
        
    except Exception as e:
        logger.error(f"Error getting last OTP time: {e}")
        return None

# Update the mark_attendance function to include status
def mark_attendance_with_status(name, student_id, status='present', session_id=None):
    """Mark attendance with specific status"""
    if not student_id:
        return False
    
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if already marked today for this session
    if student_id in attendance:
        last_time_str = attendance[student_id]["time"]
        last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
        time_diff = (current_time - last_time).total_seconds() / 3600
        if time_diff < 1:  # Don't allow duplicate entries within 1 hour
            return False
    
    # Save in-memory
    attendance[student_id] = {"name": name, "time": time_str, "status": status}
    
    # Save to database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO attendance (student_id, name, timestamp, status, session_id) VALUES (%s, %s, %s, %s, %s)",
            (student_id, name, time_str, status, session_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Attendance recorded: {name} ({student_id}) - {status}")
        return True
    except Exception as e:
        logger.error(f"Failed to save attendance to database: {e}")
        return False

class ThreadSafeTrackManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._tracks = []
        self._locked_tracks = {}
        self._attendance = {}
        
    def update_tracks_atomic(self, update_func):
        with self._lock:
            return update_func(self._tracks, self._locked_tracks)
    
    def get_tracks_snapshot(self):
        with self._lock:
            return {
                'tracks': self._tracks.copy(),
                'locked_tracks': self._locked_tracks.copy(),
                'attendance': self._attendance.copy()
            }
    
    def mark_attendance_safe(self, person_id, name, person_type):
        with self._lock:
            if person_type != 'student':
                return False
                
            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if person_id in self._attendance:
                last_time = datetime.datetime.strptime(
                    self._attendance[person_id]["time"], 
                    "%Y-%m-%d %H:%M:%S"
                )
                if (current_time - last_time).total_seconds() < 14400:
                    return False
            
            self._attendance[person_id] = {"name": name, "time": time_str}
            
            # Async DB write
            threading.Thread(
                target=self._write_attendance_to_db,
                args=(person_id, name, time_str),
                daemon=True
            ).start()
            return True
    
    def _write_attendance_to_db(self, person_id, name, time_str):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO attendance (student_id, name, timestamp) VALUES (%s, %s, %s)",
                (person_id, name, time_str)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Attendance recorded: {name} ({person_id})")
        except Exception as e:
            logger.error(f"DB write failed: {e}")

# Initialize thread-safe manager
track_manager = ThreadSafeTrackManager()

class MotionPredictor:
    def __init__(self):
        self.velocity_history = {}
        
    def predict_next_position(self, track_id, current_box, frame_idx):
        if track_id not in self.velocity_history:
            return current_box
            
        history = self.velocity_history[track_id]
        if len(history) < 2:
            return current_box
            
        recent_velocities = history[-3:]
        avg_vx = np.mean([v[0] for v in recent_velocities])
        avg_vy = np.mean([v[1] for v in recent_velocities])
        
        x1, y1, x2, y2 = current_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        pred_cx = cx + avg_vx * 2
        pred_cy = cy + avg_vy * 2
        
        pred_x1 = max(0, pred_cx - w/2)
        pred_y1 = max(0, pred_cy - h/2)
        pred_x2 = pred_cx + w/2
        pred_y2 = pred_cy + h/2
        
        return (pred_x1, pred_y1, pred_x2, pred_y2)
    
    def update_velocity(self, track_id, prev_box, curr_box, frame_idx):
        if prev_box is None:
            return
            
        prev_cx = (prev_box[0] + prev_box[2]) / 2
        prev_cy = (prev_box[1] + prev_box[3]) / 2
        curr_cx = (curr_box[0] + curr_box[2]) / 2
        curr_cy = (curr_box[1] + curr_box[3]) / 2
        
        vx = curr_cx - prev_cx
        vy = curr_cy - prev_cy
        
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []
            
        self.velocity_history[track_id].append((vx, vy))
        
        if len(self.velocity_history[track_id]) > 10:
            self.velocity_history[track_id] = self.velocity_history[track_id][-10:]

def enhanced_iou_with_motion(box1, box2, predicted_box=None, motion_weight=0.3):
    base_iou = iou(box1, box2)
    
    if predicted_box is None:
        return base_iou
        
    predicted_iou = iou(predicted_box, box2)
    enhanced_score = (1 - motion_weight) * base_iou + motion_weight * predicted_iou
    return enhanced_score

def hungarian_assignment(tracks, detections, motion_predictor, frame_idx):
    if not tracks or not detections:
        return []
    
    cost_matrix = np.full((len(tracks), len(detections)), 1.0)
    
    for i, track in enumerate(tracks):
        track_id = track.get('id')
        current_box = track.get('box')
        
        if current_box is None:
            continue
            
        predicted_box = motion_predictor.predict_next_position(
            track_id, current_box, frame_idx
        )
        
        for j, (det_x1, det_y1, det_x2, det_y2, conf) in enumerate(detections):
            det_box = (det_x1, det_y1, det_x2, det_y2)
            
            enhanced_score = enhanced_iou_with_motion(
                current_box, det_box, predicted_box, motion_weight=0.4
            )
            
            confidence_bonus = min(0.2, conf * 0.3)
            identity_bonus = 0.1 if track.get('name') != 'Unknown' else 0
            
            total_score = enhanced_score + confidence_bonus + identity_bonus
            cost_matrix[i, j] = max(0, 1.0 - total_score)
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    assignments = []
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < 0.6:
            assignments.append((row, col, 1.0 - cost_matrix[row, col]))
    
    return assignments

# Initialize motion predictor
motion_predictor = MotionPredictor()

class IdentityVerificationManager:
    def __init__(self):
        self.active_identities = {}
        self.verification_threshold = 0.8
        self.max_simultaneous_same_id = 1
        self.verification_interval = 30
        
    def can_assign_identity(self, person_id, track_id, confidence):
        current_time = time.time()
        
        if person_id not in self.active_identities:
            return True
            
        active_info = self.active_identities[person_id]
        existing_track_id = active_info.get('track_id')
        
        if existing_track_id == track_id:
            return confidence >= self.verification_threshold
            
        last_verification = active_info.get('last_verification', 0)
        if current_time - last_verification > 5.0:
            logger.info(f"Identity {person_id} switching from track {existing_track_id} to {track_id}")
            return True
            
        logger.warning(f"Duplicate identity attempt: {person_id} already active on track {existing_track_id}")
        return False
    
    def assign_identity(self, person_id, track_id, confidence):
        if not self.can_assign_identity(person_id, track_id, confidence):
            return False
            
        current_time = time.time()
        self.active_identities[person_id] = {
            'track_id': track_id,
            'last_verification': current_time,
            'confidence_history': [confidence],
            'assignment_time': current_time
        }
        logger.info(f"Identity {person_id} assigned to track {track_id} with confidence {confidence:.3f}")
        return True
    
    def cleanup_stale_identities(self):
        current_time = time.time()
        stale_ids = [
            person_id for person_id, info in self.active_identities.items()
            if current_time - info['last_verification'] > 10.0
        ]
        
        for person_id in stale_ids:
            del self.active_identities[person_id]
            logger.info(f"Removed stale identity for {person_id}")

# Initialize identity manager
identity_manager = IdentityVerificationManager()

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
known_face_types = []  # 'student' or 'faculty'

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

def load_known_faculties_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_types
    faculty_count = 0  # Track faculty faces separately
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT faculty_id, first_name, last_name, face_encoding FROM faculty WHERE face_encoding IS NOT NULL")
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
                    known_face_types.append('faculty')
                    faculty_count += 1
                    logger.info(f"Loaded faculty {full_name} ({id}) with encoding shape {encoding.shape}")
                else:
                    logger.warning(f"Invalid encoding size for faculty {id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for faculty {id}: {e}")
        cursor.close()
        conn.close()
        logger.info(f"Loaded {faculty_count} known faculty faces from database")
    except Exception as e:
        logger.error(f"Failed to load faculty faces from database: {e}")

# Initialize known faces
load_known_faces_from_db()
load_known_faculties_from_db()

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
    """Modified to handle connection failures gracefully without webcam fallback"""
    global cap, camera_available, use_dummy_feed
    
    try:
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
            
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    camera_available = True
                    use_dummy_feed = False
                    logger.info("RTSP stream connected successfully.")
                    return True
                else:
                    cap.release()
                    cap = None
                    raise Exception("Cannot read frames from RTSP stream")
            else:
                raise Exception("Cannot open RTSP stream")
                
    except Exception as e:
        logger.warning(f"RTSP connection failed: {e}")
        logger.info("Switching to dummy feed mode - Flask app will continue running")
        camera_available = False
        use_dummy_feed = True
        cap = None
        logger.info("No camera available - using dummy feed")
        return True  # Continue app execution with dummy feed


def grabber():
    """Modified grabber to handle dummy feed and connection retries"""
    global latest_frame, stop_flag, camera_available, use_dummy_feed, cap
    empty_count = 0
    retry_interval = 0
    
    while not stop_flag:
        if use_dummy_feed:
            # Use dummy frame when no camera
            latest_frame = create_dummy_frame()
            time.sleep(0.1)  # Slower refresh for dummy feed
            
            # Periodically try to reconnect to RTSP
            retry_interval += 1
            if retry_interval > 100:  # Try every ~10 seconds
                retry_interval = 0
                logger.info("Attempting to reconnect to RTSP...")
                if open_stream():
                    continue
            continue
            
        # Normal camera grabbing logic
        with cap_lock:
            if cap is None:
                time.sleep(0.1)
                continue
                
            ok, f = cap.read()
            
        if not ok:
            empty_count += 1
            if empty_count > MAX_EMPTY_GRABS:
                logger.warning("Camera connection lost. Attempting reconnection...")
                time.sleep(RECONNECT_COOLDOWN)
                
                # Try to reconnect
                if not open_stream():
                    logger.warning("Reconnection failed - switching to dummy feed")
                    camera_available = False
                    use_dummy_feed = True
                empty_count = 0
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
locked_tracks = {}  # Dict: id -> {'track': tr, 'last_seen': frame_idx, 'lock_start': frame_idx, 'type': 'student' or 'faculty'}
attendance = {}
tracking_history = {}

def mark_attendance(name, id, type):
    if type != 'student':
        return
    
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    today = current_time.strftime("%Y-%m-%d")
    
    # Check if already marked today within 4 hours
    if id in attendance:
        last_time_str = attendance[id]["time"]
        last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
        time_diff = (current_time - last_time).total_seconds() / 3600
        if time_diff < 4:
            return
    
    # Save in-memory
    attendance[id] = {"name": name, "time": time_str}
    
    # Save to database
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
        logger.error(f"Failed to save attendance to database: {e}", exc_info=True)


def update_trackers(rgb, frame, frame_idx):
    global tracks, locked_tracks
    h, w = frame.shape[:2]
    kept = []
    to_remove_locks = []

    # Remove locked tracks when person disappears
    for sid, lock_info in list(locked_tracks.items()):
        frames_since_seen = frame_idx - lock_info.get('last_seen', frame_idx)
        if frames_since_seen > 30:  # ~1 second at 30fps
            to_remove_locks.append(sid)
            logger.info(f"Person {sid} disappeared - releasing lock")

    for sid in to_remove_locks:
        locked_tracks.pop(sid, None)

    for tr in list(tracks):
        tracker_ok = False
        tid = tr.get("id")
        is_locked = tid in locked_tracks

        try:
            tracker = tr.get("tracker")
            if tracker is None:
                continue

            pos = tracker.get_position()
            old_x1, old_y1 = int(pos.left()), int(pos.top())
            old_x2, old_y2 = int(pos.right()), int(pos.bottom())
            old_width = max(old_x2 - old_x1, 1)
            old_height = max(old_y2 - old_y1, 1)

            update_success = tracker.update(rgb)

            if update_success is not False and update_success is not None:
                tracker_quality = None
                try:
                    tracker_quality = float(update_success)
                except Exception:
                    tracker_quality = None

                pos = tracker.get_position()
                x1, y1 = int(pos.left()), int(pos.top())
                x2, y2 = int(pos.right()), int(pos.bottom())
                new_width = max(x2 - x1, 1)
                new_height = max(y2 - y1, 1)

                valid_position = (
                    x2 > x1 and y2 > y1 and
                    0 <= x1 < w and 0 <= y1 < h and
                    x2 <= w and y2 <= h
                )

                size_growth = max(new_width / old_width, new_height / old_height)
                size_reasonable = (
                    size_growth < 1.6 and
                    new_width < w * 0.9 and
                    new_height < h * 0.9
                )

                old_cx, old_cy = (old_x1 + old_x2) / 2.0, (old_y1 + old_y2) / 2.0
                new_cx, new_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                center_movement_sq = (new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2
                diag = (w ** 2 + h ** 2) ** 0.5
                movement_reasonable = center_movement_sq < (0.03 * (diag ** 2))

                min_size_ok = max(new_width, new_height) >= 20

                if valid_position and size_reasonable and movement_reasonable and min_size_ok:
                    tracker_ok = True
                    tr["box"] = (x1, y1, x2, y2)

                    # Dynamic confidence update
                    if not is_locked:
                        prev_conf = tr.get("confidence", 0.5)
                        if tracker_quality is not None:
                            qc = max(0.0, min(1.0, tracker_quality))
                            tr["confidence"] = min(0.95, prev_conf * 0.9 + qc * 0.1)
                        else:
                            # Decay slowly if no quality returned
                            tr["confidence"] = max(0.3, prev_conf * 0.995)

        except Exception as e:
            logger.debug(f"Tracker update failed for {tr.get('name','Unknown')}: {e}")
            tracker_ok = False

        if not tracker_ok:
            tr["consecutive_failures"] = tr.get("consecutive_failures", 0) + 1
            if tr.get("consecutive_failures", 0) >= 3:
                logger.info(f"Removing track {tr.get('name', 'Unknown')} - too many failures")
                if tid is not None and tid in locked_tracks:
                    locked_tracks.pop(tid, None)
                    logger.info(f"Removed lock for {tid}")
                continue

            if "box" in tr:
                try:
                    bx1, by1, bx2, by2 = tr["box"]
                    padding = min(12, max(4, (bx2 - bx1) // 6, (by2 - by1) // 6))
                    ex1 = max(0, bx1 - padding)
                    ey1 = max(0, by1 - padding)
                    ex2 = min(w - 1, bx2 + padding)
                    ey2 = min(h - 1, by2 + padding)

                    new_tracker = dlib.correlation_tracker()
                    new_tracker.start_track(rgb, dlib.rectangle(int(ex1), int(ey1), int(ex2), int(ey2)))
                    tr["tracker"] = new_tracker
                    tracker_ok = True
                except Exception as e:
                    logger.debug(f"Tracker reinit failed for {tr.get('name','Unknown')}: {e}")
        else:
            tr["consecutive_failures"] = 0

        if not tracker_ok:
            continue

        tr["last_seen"] = frame_idx
        if "start_frame" not in tr:
            tr["start_frame"] = frame_idx

        duration_frames = frame_idx - tr["start_frame"]
        duration_seconds = duration_frames / 30.0

        if is_locked:
            locked_tracks[tid]['last_seen'] = frame_idx
            locked_tracks[tid]['track'] = tr
            tr["confidence"] = max(tr.get("confidence", 0.6), 0.6)
        else:
            if tr.get("confidence", 0.0) >= 0.55 and tid:
                if tid not in locked_tracks:
                    locked_tracks[tid] = {
                        'track': tr,
                        'last_seen': frame_idx,
                        'lock_start': frame_idx,
                        'missed_detections': 0
                    }
                    logger.info(f"Track LOCKED for {tr.get('name','Unknown')} ({tid})")

        # === Status and box color tied together ===
        if is_locked:
            status_label = "[LOCKED]"
            status_color = (0, 200, 0)
            box_color = (0, 255, 0)
        elif tr.get("confidence", 0.0) >= 0.4:
            status_label = "[SCANNING]"
            status_color = (0, 180, 180)
            box_color = (0, 255, 255)
        else:
            status_label = "[DETECTING]"
            status_color = (0, 0, 180)
            box_color = (0, 0, 255)

        bx1, by1, bx2, by2 = tr.get("box", (0, 0, 0, 0))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        padding = 6

        display_name = tr.get("name") if tr.get("name") and tr.get("name") != "Unknown" else "Unknown"
        (name_w, name_h), _ = cv2.getTextSize(display_name, font, font_scale, thickness)
        name_y = max(10, by1 - 10)
        name_x2 = min(w - 1, bx1 + name_w + padding)
        cv2.rectangle(frame, (bx1, name_y - name_h - 4), (name_x2, name_y + 4), bg_color, -1)
        cv2.putText(frame, display_name, (bx1 + 4, name_y), font, font_scale, text_color, thickness)

        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        track_time_label = f"Time: {mins:02d}:{secs:02d}"
        (time_w, time_h), _ = cv2.getTextSize(track_time_label, font, font_scale - 0.1, thickness - 1)
        time_y = by2 + time_h + 10
        time_x2 = min(w - 1, bx1 + time_w + padding)
        cv2.rectangle(frame, (bx1, time_y - time_h - 4), (time_x2, time_y + 4), bg_color, -1)
        cv2.putText(frame, track_time_label, (bx1 + 4, time_y), font, font_scale - 0.1, text_color, thickness - 1)

        (status_w, status_h), _ = cv2.getTextSize(status_label, font, font_scale - 0.1, thickness - 1)
        status_y = time_y + status_h + 10
        status_x2 = min(w - 1, bx1 + status_w + padding)
        cv2.rectangle(frame, (bx1, status_y - status_h - 4), (status_x2, status_y + 4), status_color, -1)
        cv2.putText(frame, status_label, (bx1 + 4, status_y), font, font_scale - 0.1, text_color, thickness - 1)

        if not is_locked:
            conf_val = tr.get("confidence", 0.0)
            conf_label = f"Conf: {conf_val:.2f}"
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale - 0.1, thickness - 1)
            conf_y = status_y + conf_h + 10
            conf_x2 = min(w - 1, bx1 + conf_w + padding)
            cv2.rectangle(frame, (bx1, conf_y - conf_h - 4), (conf_x2, conf_y + 4), bg_color, -1)
            cv2.putText(frame, conf_label, (bx1 + 4, conf_y), font, font_scale - 0.1, text_color, thickness - 1)

        if tr.get("consecutive_failures", 0) < 3:
            kept.append(tr)

    tracks[:] = kept



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
        # Estimate distance based on face width in pixels
        distance = estimate_distance(face_width_pixels)

        # Too far for reliable recognition
        if distance > MAX_RECOGNITION_DISTANCE:
            logger.info(f"Face too far for recognition: {distance:.1f}m")
            return "Unknown", None, float('inf'), distance, 0.0, None

        # Image enhancement (better for low-light or far faces)
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = face_image

        # Extract embeddings
        faces = face_analysis.get(enhanced)
        if not faces:
            return "Unknown", None, float('inf'), distance, 0.0, None

        face_embedding = faces[0].embedding

        # Compare against known embeddings
        similarities = []
        for known_embedding in known_face_encodings:
            dot_product = np.dot(face_embedding, known_embedding)
            norm_a = np.linalg.norm(face_embedding)
            norm_b = np.linalg.norm(known_embedding)
            similarity = dot_product / (norm_a * norm_b)
            similarities.append(similarity)

        if similarities:
            best_match_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_match_index])

            # Confidence: convert similarity into "closeness" score
            confidence = best_similarity

            # Decide recognition
            if is_locked_track or confidence >= (1 - tolerance):
                name = known_face_names[best_match_index]
                id = known_face_ids[best_match_index]
                role_type = known_face_types[best_match_index]

                # Add role label (e.g., faculty)
                if role_type == 'faculty':
                    name = f"Faculty: {name}"

                return (
                    name,
                    id,
                    1 - confidence,   # recognition distance metric
                    distance,         # estimated distance in meters
                    confidence,       # similarity-based confidence
                    role_type
                )

        # Fallback if no match
        return "Unknown", None, float('inf'), distance, 0.0, None

    except Exception as e:
        logger.error(f"Error in enhanced_recognize_face: {e}")
        return "Unknown", None, float('inf'), None, 0.0, None


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

    # BETTER TRACK MANAGEMENT FOR 30+ ABOVE STUDENTS
    MAX_TOTAL_TRACKS = 50
    MAX_UNLOCKED_TRACKS = 20

    # Clean up old tracks first
    if len(tracks) > MAX_TOTAL_TRACKS:
        locked_track_objects = [info['track'] for info in locked_tracks.values()]
        unlocked_tracks = [tr for tr in tracks if tr not in locked_track_objects]
        unlocked_tracks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        unlocked_tracks = unlocked_tracks[:MAX_UNLOCKED_TRACKS]
        tracks[:] = locked_track_objects + unlocked_tracks
        logger.info(f"Track cleanup: kept {len(locked_track_objects)} locked + {len(unlocked_tracks)} unlocked = {len(tracks)} total")

    # DYNAMIC DETECTION FREQUENCY
    DETECT_EVERY = 2 if len(tracks) < 3 else (3 if len(tracks) < 6 else 5)
    h, w = frame.shape[:2]

    if frame_idx % DETECT_EVERY != 0:
        return

    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    frame_eq = enhance_lighting(small_frame)
    results = yolo.predict(source=frame_eq, verbose=False, conf=CONF_THRESH, imgsz=640, device=DEVICE)

    # Raw detections
    raw_dets = []
    if results:
        r = results[0]
        if getattr(r, "boxes", None) is not None:
            for b in r.boxes:
                try:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                except Exception:
                    continue

                x1 = int(max(0, x1 / RESIZE_FACTOR)); y1 = int(max(0, y1 / RESIZE_FACTOR))
                x2 = int(min(w - 1, x2 / RESIZE_FACTOR)); y2 = int(min(h - 1, y2 / RESIZE_FACTOR))

                if conf >= CONF_THRESH and x2 > x1 and y2 > y1:
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width >= 25 and box_height >= 25:
                        raw_dets.append((x1, y1, x2, y2, conf))

    # Apply Non-Maximum Suppression to remove duplicate detections
    dets = []
    if raw_dets:
        boxes_xywh = [[int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])] for d in raw_dets]
        scores = [float(d[4]) for d in raw_dets]

        try:
            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, CONF_THRESH, 0.4)
        except Exception as e:
            logger.exception(f"NMS failed: {e}")
            indices = []

        if indices is not None and len(indices) > 0:
            flat_indices = np.array(indices).flatten()
            for i in flat_indices:
                i = int(i)
                if 0 <= i < len(raw_dets):
                    dets.append(raw_dets[i])

    logger.info(f"Frame {frame_idx}: {len(raw_dets)} raw detections → {len(dets)} after NMS")
    new_tracks = []

    # Keep track of which locked tracks were matched
    matched_locked_tracks = set()
    used_detections = set()

    # FIRST PASS: Match locked tracks to detections
    for sid, lock_info in list(locked_tracks.items()):
        if sid in matched_locked_tracks:
            continue

        best_detection = None
        best_iou = 0.3
        best_idx = -1

        lock_box = lock_info.get('track', {}).get('box')
        if not lock_box:
            continue

        for idx, (x1, y1, x2, y2, conf) in enumerate(dets):
            if idx in used_detections:
                continue
            overlap_iou = iou((x1, y1, x2, y2), lock_box)
            if overlap_iou > best_iou:
                best_iou = overlap_iou
                best_detection = (x1, y1, x2, y2, conf)
                best_idx = idx

        if best_detection:
            used_detections.add(best_idx)
            matched_locked_tracks.add(sid)
            x1, y1, x2, y2, conf = best_detection
            tr = lock_info['track']
            tr['last_seen'] = frame_idx
            tr['box'] = (x1, y1, x2, y2)
            # For locked tracks, don't update confidence - just maintain tracking
            locked_tracks[sid]['last_seen'] = frame_idx
            locked_tracks[sid]['missed_detections'] = 0

            try:
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, 0.2)
                tr["tracker"].start_track(rgb, dlib.rectangle(int(ex1), int(ey1), int(ex2), int(ey2)))
            except Exception as e:
                logger.error(f"Tracker update error for locked track {tr.get('name','Unknown')}: {e}")
                dtracker = dlib.correlation_tracker()
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, 0.2)
                dtracker.start_track(rgb, dlib.rectangle(int(ex1), int(ey1), int(ex2), int(ey2)))
                tr["tracker"] = dtracker

            new_tracks.append(tr)
            logger.debug(f"Matched locked track {tr.get('name','Unknown')} with IoU {best_iou:.2f}")
        else:
            lock_info['missed_detections'] = lock_info.get('missed_detections', 0) + 1

    # SECOND PASS: Process remaining detections for new tracks
    for idx, (x1, y1, x2, y2, conf) in enumerate(dets):
        if idx in used_detections:
            continue

        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_BOX_RATIO)
        ex1, ey1, ex2, ey2 = map(int, (max(0, ex1), max(0, ey1), min(w - 1, ex2), min(h - 1, ey2)))
        face_region = rgb[ey1:ey2, ex1:ex2]
        face_width = x2 - x1

        # Check overlap with existing new_tracks
        overlaps_existing = False
        for existing_tr in new_tracks:
            existing_box = existing_tr.get('box')
            if existing_box and iou((x1, y1, x2, y2), existing_box) > 0.3:
                overlaps_existing = True
                break

        if overlaps_existing:
            logger.debug("Skipping detection that overlaps with existing track")
            continue

        if face_region.size == 0 or face_region.shape[0] < MIN_FACE_SIZE or face_region.shape[1] < MIN_FACE_SIZE:
            logger.debug(f"Face region too small or empty: {face_region.shape}")
            continue

        # Dynamic confidence updates for scanning tracks
        name = "Unknown"
        person_id = None
        ptype = None
        confidence = conf * 0.8  # Base confidence from YOLO

        if conf >= 0.5:  # Only recognize if YOLO confidence is decent
            tolerance = TOLERANCE * 1.1 if conf >= 0.8 else TOLERANCE
            try:
                name, person_id, distance, estimated_distance, recog_confidence, ptype = enhanced_recognize_face(
                    face_region, face_width, tolerance, is_locked_track=False
                )
                # Dynamic confidence calculation for scanning tracks
                if name != "Unknown":
                    confidence = min(1.0, (conf * 0.4) + (recog_confidence * 0.6))
                    if conf >= 0.9:
                        confidence = min(1.0, confidence * 1.1)
                else:
                    confidence = conf * 0.7  # Lower confidence for unknown faces
                    
            except Exception as e:
                logger.exception(f"enhanced_recognize_face failed: {e}")
                confidence = conf * 0.6

        # Create new track
        dtracker = dlib.correlation_tracker()
        try:
            ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, 0.2)
            dtracker.start_track(rgb, dlib.rectangle(int(ex1), int(ey1), int(ex2), int(ey2)))
            tr = {
                "tracker": dtracker,
                "name": name,
                "id": person_id,
                "type": ptype,
                "confidence": confidence,
                "last_seen": frame_idx,
                "box": (x1, y1, x2, y2),
                "start_frame": frame_idx,
                "recognition_count": 1 if person_id else 0
            }
            new_tracks.append(tr)

            # Lock at 0.6 confidence
            if person_id and confidence >= 0.6 and person_id not in locked_tracks:
                locked_tracks[person_id] = {
                    'track': tr,
                    'last_seen': frame_idx,
                    'lock_start': frame_idx,
                    'type': ptype,
                    'missed_detections': 0
                }
                logger.info(f"LOCKED track for {ptype} {name} ({person_id}) with confidence {confidence:.4f}")

        except Exception as e:
            logger.error(f"Tracker creation error: {e}")
            continue

    # Remove locked tracks that weren't matched
    tracks_to_remove = []
    for sid, lock_info in list(locked_tracks.items()):
        if lock_info.get('missed_detections', 0) > 3:
            tracks_to_remove.append(sid)
            logger.info(f"Removing locked track for {sid} - missed detections: {lock_info['missed_detections']}")

    for sid in tracks_to_remove:
        locked_tracks.pop(sid, None)

    # Update tracks list
    preserved_tracks = []
    for tr in tracks:
        if tr.get("id") in locked_tracks:
            preserved_tracks.append(tr)

    for tr in new_tracks:
        already_exists = False
        for existing_tr in preserved_tracks:
            if existing_tr.get('id') is not None and existing_tr.get('id') == tr.get('id'):
                already_exists = True
                break
        if not already_exists:
            preserved_tracks.append(tr)

    tracks[:] = preserved_tracks

    # Final cleanup - ensure no duplicate boxes
    final_tracks = []
    for i, tr in enumerate(tracks):
        is_duplicate = False
        for j, other_tr in enumerate(tracks):
            if i != j and iou(tr.get('box', (0, 0, 0, 0)), other_tr.get('box', (0, 0, 0, 0))) > 0.5:
                if (other_tr.get('id') in locked_tracks) or (other_tr.get('confidence', 0) > tr.get('confidence', 0)):
                    is_duplicate = True
                    break
        if not is_duplicate:
            final_tracks.append(tr)

    tracks[:] = final_tracks
    logger.info(f"Final tracks: {len(tracks)} (locked: {len(locked_tracks)})")


# =========================
# Flask streaming & API
# =========================
app = Flask(__name__)
app.template_folder = 'templates'  # Add this line
app.static_folder = 'static'       # Add this line
CORS(app)

# Add these Flask routes for authentication

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'})
        
        if "@wmsu.edu.ph" not in email:
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        # Authenticate user
        user = authenticate_user(email, password)
        if not user:
            return jsonify({'success': False, 'message': 'Invalid email or password'})
        
        # Create session
        session_token = create_user_session(user['user_id'], user['user_type'])
        if not session_token:
            return jsonify({'success': False, 'message': 'Failed to create session'})
        
        # Determine redirect URL based on role
        redirect_url = '/AdminDB'  # Default for admins and faculty
        if user['user_type'] == 'student':
            redirect_url = '/StudentLP'
        
        # Create response
        resp = jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['user_id'],
                'name': f"{user['first_name']} {user['last_name']}",
                'type': user['user_type'],
                'role': user.get('role', 'student')
            },
            'redirect_url': redirect_url
        })
        
        # Set session token as a cookie
        resp.set_cookie('session_token', session_token, httponly=True, secure=False, samesite='Strict')  # secure=False for non-HTTPS testing
        return resp
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'})

@app.route('/api/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.json
        email = data.get('email', '').strip()
        
        if not email or "@wmsu.edu.ph" not in email:
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        # Check cooldown period
        last_otp_time = get_last_otp_time(email, 'password_reset')
        if last_otp_time:
            time_diff = (datetime.now() - last_otp_time).total_seconds()
            if time_diff < OTP_RESEND_COOLDOWN:
                remaining = int(OTP_RESEND_COOLDOWN - time_diff)
                return jsonify({
                    'success': False, 
                    'message': f'Please wait {remaining} seconds before requesting another OTP',
                    'cooldown': remaining
                })
        
        # Check if user exists in any table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT email FROM admins WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute("SELECT email FROM faculty WHERE email = %s", (email,))
            user = cursor.fetchone()
            
        if not user:
            cursor.execute("SELECT email FROM students WHERE email = %s", (email,))
            user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not user:
            return jsonify({'success': False, 'message': 'Email not found in our system'})
        
        # Generate OTP
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=10)
        
        # Save OTP to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete old OTPs for this email
        cursor.execute("DELETE FROM otp_codes WHERE email = %s AND purpose = 'password_reset'", (email,))
        
        cursor.execute(
            "INSERT INTO otp_codes (email, otp_code, purpose, expires_at) VALUES (%s, %s, 'password_reset', %s)",
            (email, otp_code, expires_at)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Send OTP email
        if send_otp_email(email, otp_code):
            return jsonify({
                'success': True,
                'message': 'Password reset OTP sent to your email',
                'cooldown': OTP_RESEND_COOLDOWN
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to send OTP email'})
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'success': False, 'message': 'Failed to process password reset request'})

@app.route('/api/verify_reset_otp', methods=['POST'])
def verify_reset_otp():
    try:
        data = request.json
        email = data.get('email', '').strip()
        otp_code = data.get('otp', '').strip()
        new_password = data.get('new_password', '').strip()
        
        if not email or not otp_code or not new_password:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if len(new_password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
        
        # Verify OTP
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT otp_code, expires_at, attempts FROM otp_codes WHERE email = %s AND purpose = 'password_reset' AND used = FALSE ORDER BY created_at DESC LIMIT 1",
            (email,)
        )
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No valid OTP found for this email'})
        
        stored_otp, expires_at, attempts = result
        
        if datetime.now() > expires_at:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'OTP has expired'})
        
        if attempts >= MAX_OTP_ATTEMPTS:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Maximum OTP attempts exceeded'})
        
        if otp_code != stored_otp:
            # Increment attempts
            cursor.execute(
                "UPDATE otp_codes SET attempts = attempts + 1 WHERE email = %s AND purpose = 'password_reset'",
                (email,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid OTP'})
        
        # Mark OTP as used
        cursor.execute(
            "UPDATE otp_codes SET used = TRUE WHERE email = %s AND purpose = 'password_reset'",
            (email,)
        )
        
        # Update password in appropriate table
        password_hash = hash_password(new_password)
        
        # Try updating in admins table first
        cursor.execute("UPDATE admins SET password_hash = %s WHERE email = %s", (password_hash, email))
        if cursor.rowcount == 0:
            # Try faculty table
            cursor.execute("UPDATE faculty SET password_hash = %s WHERE email = %s", (password_hash, email))
            if cursor.rowcount == 0:
                # Try students table
                cursor.execute("UPDATE students SET password_hash = %s WHERE email = %s", (password_hash, email))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Password reset successfully'})
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'success': False, 'message': 'Failed to reset password'})

    
@app.route('/api/get_students_enhanced', methods=['GET'])
def get_students_enhanced():
    try:
        # Get query parameters for filtering
        department = request.args.get('department', '')
        course = request.args.get('course', '')
        year_section = request.args.get('year_section', '')
        status = request.args.get('status', 'active')
        search = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Build dynamic query
        where_conditions = ["status = %s"]
        params = [status]
        
        if course:
            where_conditions.append("course LIKE %s")
            params.append(f"%{course}%")
            
        if year_section:
            where_conditions.append("year_section LIKE %s")
            params.append(f"%{year_section}%")
            
        if search:
            where_conditions.append("(first_name LIKE %s OR last_name LIKE %s OR student_id LIKE %s OR email LIKE %s)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])
        
        where_clause = " AND ".join(where_conditions)
        offset = (page - 1) * limit
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM students WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['total']
        
        # Get paginated results
        query = f"""
            SELECT student_id, first_name, last_name, middle_name, course, year_section, 
                   email, photo_path, status, created_at, updated_at
            FROM students 
            WHERE {where_clause}
            ORDER BY last_name, first_name
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        cursor.execute(query, params)
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format student data for frontend
        formatted_students = []
        for s in students:
            formatted_students.append({
                'id': s['student_id'],
                'idNumber': s['student_id'],
                'firstName': s['first_name'],
                'lastName': s['last_name'],
                'middleName': s['middle_name'],
                'name': f"{s['first_name']} {s['middle_name'] + ' ' if s['middle_name'] else ''}{s['last_name']}",
                'course': s['course'],
                'yearSection': s['year_section'],
                'email': s['email'],
                'photo': f"/student_photos/{s['student_id']}.jpg" if s['photo_path'] else f"https://ui-avatars.com/api/?name={s['first_name']}+{s['last_name']}&background=random",
                'status': s['status'],
                'createdAt': s['created_at'].isoformat() if s['created_at'] else None,
                'updatedAt': s['updated_at'].isoformat() if s['updated_at'] else None
            })
        
        return jsonify({
            'success': True, 
            'students': formatted_students,
            'pagination': {
                'total': total_count,
                'page': page,
                'limit': limit,
                'pages': (total_count + limit - 1) // limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching students: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_faculty_enhanced', methods=['GET'])
def get_faculty_enhanced():
    try:
        # Get query parameters for filtering
        department = request.args.get('department', '')
        designation = request.args.get('designation', '')
        status = request.args.get('status', 'active')
        search = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Build dynamic query
        where_conditions = ["status = %s"]
        params = [status]
        
        if department:
            where_conditions.append("department LIKE %s")
            params.append(f"%{department}%")
            
        if designation:
            where_conditions.append("designation LIKE %s")
            params.append(f"%{designation}%")
            
        if search:
            where_conditions.append("(first_name LIKE %s OR last_name LIKE %s OR faculty_id LIKE %s OR email LIKE %s)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])
        
        where_clause = " AND ".join(where_conditions)
        offset = (page - 1) * limit
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM faculty WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['total']
        
        # Get paginated results
        query = f"""
            SELECT faculty_id, first_name, last_name, middle_name, department, designation, 
                   email, photo_path, status, role, created_at, updated_at
            FROM faculty 
            WHERE {where_clause}
            ORDER BY last_name, first_name
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        cursor.execute(query, params)
        faculty = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format faculty data for frontend
        formatted_faculty = []
        for f in faculty:
            formatted_faculty.append({
                'id': f['faculty_id'],
                'idNumber': f['faculty_id'],
                'firstName': f['first_name'],
                'lastName': f['last_name'],
                'middleName': f['middle_name'],
                'name': f"{f['first_name']} {f['middle_name'] + ' ' if f['middle_name'] else ''}{f['last_name']}",
                'department': f['department'],
                'designation': f['designation'],
                'email': f['email'],
                'photo': f"/faculty_photos/{f['faculty_id']}.jpg" if f['photo_path'] else f"https://ui-avatars.com/api/?name={f['first_name']}+{f['last_name']}&background=random",
                'status': f['status'],
                'role': f['role'],
                'createdAt': f['created_at'].isoformat() if f['created_at'] else None,
                'updatedAt': f['updated_at'].isoformat() if f['updated_at'] else None
            })
        
        return jsonify({
            'success': True, 
            'faculty': formatted_faculty,
            'pagination': {
                'total': total_count,
                'page': page,
                'limit': limit,
                'pages': (total_count + limit - 1) // limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_dashboard_stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics for admin overview"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student count by status
        cursor.execute("SELECT status, COUNT(*) as count FROM students GROUP BY status")
        student_stats = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Get faculty count by status
        cursor.execute("SELECT status, COUNT(*) as count FROM faculty GROUP BY status")
        faculty_stats = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Get recent registrations (last 30 days)
        cursor.execute("""
            SELECT 'student' as type, COUNT(*) as count 
            FROM students 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            UNION ALL
            SELECT 'faculty' as type, COUNT(*) as count 
            FROM faculty 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """)
        recent_registrations = {row['type']: row['count'] for row in cursor.fetchall()}
        
        # Get active invites count
        cursor.execute("""
            SELECT invite_type, COUNT(*) as count 
            FROM invites 
            WHERE expires_at > NOW() AND current_uses < max_uses 
            GROUP BY invite_type
        """)
        active_invites = {row['invite_type']: row['count'] for row in cursor.fetchall()}
        
        # Get attendance stats (today)
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) as present_today
            FROM attendance 
            WHERE DATE(timestamp) = CURDATE()
        """)
        attendance_today = cursor.fetchone()['present_today']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'students': {
                    'total': sum(student_stats.values()),
                    'active': student_stats.get('active', 0),
                    'inactive': student_stats.get('inactive', 0)
                },
                'faculty': {
                    'total': sum(faculty_stats.values()),
                    'active': faculty_stats.get('active', 0),
                    'inactive': faculty_stats.get('inactive', 0)
                },
                'recent_registrations': {
                    'students': recent_registrations.get('student', 0),
                    'faculty': recent_registrations.get('faculty', 0)
                },
                'active_invites': {
                    'student': active_invites.get('student', 0),
                    'faculty': active_invites.get('faculty', 0)
                },
                'attendance': {
                    'present_today': attendance_today
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bulk_update_students', methods=['POST'])
def bulk_update_students():
    """Bulk update student status or other fields"""
    try:
        data = request.json
        student_ids = data.get('student_ids', [])
        updates = data.get('updates', {})
        
        if not student_ids or not updates:
            return jsonify({'success': False, 'message': 'Student IDs and updates are required'})
        
        # Validate updates (only allow certain fields)
        allowed_fields = ['status', 'course', 'year_section']
        valid_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not valid_updates:
            return jsonify({'success': False, 'message': 'No valid fields to update'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build update query
        set_clause = ", ".join([f"{field} = %s" for field in valid_updates.keys()])
        placeholders = ", ".join(["%s"] * len(student_ids))
        
        query = f"""
            UPDATE students 
            SET {set_clause}, updated_at = NOW() 
            WHERE student_id IN ({placeholders})
        """
        
        params = list(valid_updates.values()) + student_ids
        cursor.execute(query, params)
        
        affected_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        # Refresh known faces if needed
        if 'status' in valid_updates:
            load_known_faces_from_db()
        
        return jsonify({
            'success': True, 
            'message': f'Updated {affected_rows} students successfully',
            'affected_rows': affected_rows
        })
        
    except Exception as e:
        logger.error(f"Error bulk updating students: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bulk_update_faculty', methods=['POST'])
def bulk_update_faculty():
    """Bulk update faculty status or other fields"""
    try:
        data = request.json
        faculty_ids = data.get('faculty_ids', [])
        updates = data.get('updates', {})
        
        if not faculty_ids or not updates:
            return jsonify({'success': False, 'message': 'Faculty IDs and updates are required'})
        
        # Validate updates (only allow certain fields)
        allowed_fields = ['status', 'department', 'designation', 'role']
        valid_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not valid_updates:
            return jsonify({'success': False, 'message': 'No valid fields to update'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build update query
        set_clause = ", ".join([f"{field} = %s" for field in valid_updates.keys()])
        placeholders = ", ".join(["%s"] * len(faculty_ids))
        
        query = f"""
            UPDATE faculty 
            SET {set_clause}, updated_at = NOW() 
            WHERE faculty_id IN ({placeholders})
        """
        
        params = list(valid_updates.values()) + faculty_ids
        cursor.execute(query, params)
        
        affected_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        # Refresh known faces if needed
        if 'status' in valid_updates:
            load_known_faculties_from_db()
        
        return jsonify({
            'success': True, 
            'message': f'Updated {affected_rows} faculty members successfully',
            'affected_rows': affected_rows
        })
        
    except Exception as e:
        logger.error(f"Error bulk updating faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_data', methods=['POST'])
def export_data():
    """Export students or faculty data to CSV"""
    try:
        data = request.json
        data_type = data.get('type', 'students')  # 'students' or 'faculty'
        filters = data.get('filters', {})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if data_type == 'students':
            query = """
                SELECT student_id, first_name, last_name, middle_name, course, 
                       year_section, email, status, created_at
                FROM students
                WHERE status = %s
                ORDER BY last_name, first_name
            """
            cursor.execute(query, [filters.get('status', 'active')])
            
        elif data_type == 'faculty':
            query = """
                SELECT faculty_id, first_name, last_name, middle_name, department, 
                       designation, email, role, status, created_at
                FROM faculty
                WHERE status = %s
                ORDER BY last_name, first_name
            """
            cursor.execute(query, [filters.get('status', 'active')])
        
        else:
            return jsonify({'success': False, 'message': 'Invalid data type'})
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not results:
            return jsonify({'success': False, 'message': 'No data found to export'})
        
        # Convert to CSV format
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        
        for row in results:
            # Convert datetime objects to strings
            formatted_row = {}
            for key, value in row.items():
                if hasattr(value, 'isoformat'):
                    formatted_row[key] = value.isoformat()
                else:
                    formatted_row[key] = value
            writer.writerow(formatted_row)
        
        csv_data = output.getvalue()
        output.close()
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'{data_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Update existing routes to use the enhanced functionality
@app.route('/api/get_students', methods=['GET'])
def get_students():
    """Legacy route - redirects to enhanced version"""
    return get_students_enhanced()

@app.route('/api/get_faculty', methods=['GET'])
def get_faculty():
    """Legacy route - redirects to enhanced version"""
    return get_faculty_enhanced()    

@app.route('/api/update_attendance_status', methods=['POST'])
def update_attendance_status():
    try:
        data = request.json
        student_id = data.get('student_id')
        status = data.get('status')  # 'present', 'late', 'absent', 'excused'
        remarks = data.get('remarks', '')
        
        if not student_id or not status:
            return jsonify({'success': False, 'message': 'Student ID and status are required'})
        
        if status not in ['present', 'late', 'absent', 'excused']:
            return jsonify({'success': False, 'message': 'Invalid status'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the most recent attendance record for this student
        cursor.execute(
            "UPDATE attendance SET status = %s, remarks = %s WHERE student_id = %s AND DATE(timestamp) = CURDATE() ORDER BY timestamp DESC LIMIT 1",
            (status, remarks, student_id)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No attendance record found to update'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Attendance status updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating attendance status: {e}")
        return jsonify({'success': False, 'message': str(e)})   
    
@app.route('/api/get_active_invites', methods=['GET'])
def get_active_invites():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get active invites that haven't expired
        cursor.execute("""
            SELECT token, expires_at, used, created_at 
            FROM invites 
            WHERE expires_at > NOW() AND used = 0
            ORDER BY created_at DESC
        """)
        
        invites = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format the invites data
        formatted_invites = []
        for invite in invites:
            # Calculate time remaining
            time_remaining = invite['expires_at'] - datetime.now()
            if time_remaining.total_seconds() > 0:
                days = time_remaining.days
                hours, remainder = divmod(time_remaining.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                if days > 0:
                    time_remaining_str = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    time_remaining_str = f"{hours}h {minutes}m"
                else:
                    time_remaining_str = f"{minutes}m"
                
                # Determine invite type (you may need to add a type column to your invites table)
                # For now, assuming all invites are for students, but you can modify this
                invite_type = 'student'  # or 'faculty' based on your logic
                
                formatted_invites.append({
                    'token': invite['token'],
                    'type': invite_type,
                    'link': f"{request.host_url}studentreg?token={invite['token']}" if invite_type == 'student' else f"{request.host_url}facultyreg?token={invite['token']}",
                    'created_at': invite['created_at'].isoformat(),
                    'expires_at': invite['expires_at'].isoformat(),
                    'time_remaining': time_remaining_str,
                    'uses': 0  # You may need to track this in your database
                })
        
        return jsonify({
            'success': True,
            'invites': formatted_invites
        })
        
    except Exception as e:
        logger.error(f"Error getting active invites: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/revoke_invite', methods=['POST'])
def revoke_invite():
    try:
        data = request.json
        token = data.get('token')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Mark the invite as used (revoked)
        cursor.execute("UPDATE invites SET used = 1 WHERE token = %s", (token,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invite not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Invite revoked successfully'})
        
    except Exception as e:
        logger.error(f"Error revoking invite: {e}")
        return jsonify({'success': False, 'message': str(e)})    

@app.route('/api/get_enhanced_dashboard_stats', methods=['GET'])
def get_enhanced_dashboard_stats():
    """Get comprehensive dashboard statistics including course breakdowns"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get total student count
        cursor.execute("SELECT COUNT(*) as total_students FROM students WHERE status = 'active'")
        total_students = cursor.fetchone()['total_students']
        
        # Get total faculty count  
        cursor.execute("SELECT COUNT(*) as total_faculty FROM faculty WHERE status = 'active'")
        total_faculty = cursor.fetchone()['total_faculty']
        
        # Get student count by course
        cursor.execute("""
            SELECT course, COUNT(*) as count 
            FROM students 
            WHERE status = 'active' 
            GROUP BY course 
            ORDER BY count DESC
        """)
        course_stats = cursor.fetchall()
        
        # Get CS, IT, and ACT student counts specifically
        cs_count = 0
        it_count = 0 
        act_count = 0
        
        for course in course_stats:
            course_name = course['course'].upper()
            if 'COMPUTER SCIENCE' in course_name or 'CS' in course_name:
                cs_count += course['count']
            elif 'INFORMATION TECHNOLOGY' in course_name or 'IT' in course_name:
                it_count += course['count']
            elif 'ACT' in course_name or 'ASSOCIATE IN COMPUTER TECHNOLOGY' in course_name:
                act_count += course['count']
        
        # Get faculty count by department
        cursor.execute("""
            SELECT department, COUNT(*) as count 
            FROM faculty 
            WHERE status = 'active' 
            GROUP BY department 
            ORDER BY count DESC
        """)
        department_stats = cursor.fetchall()
        
        # Get recent attendance (today)
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) as present_today,
                   COUNT(*) as total_attendance_records_today
            FROM attendance 
            WHERE DATE(timestamp) = CURDATE()
        """)
        attendance_today = cursor.fetchone()
        
        # Get attendance this week
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) as unique_students_week,
                   COUNT(*) as total_records_week
            FROM attendance 
            WHERE YEARWEEK(timestamp) = YEARWEEK(NOW())
        """)
        attendance_week = cursor.fetchone()
        
        # Get recent registrations (last 30 days)
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM students WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)) as students_30days,
                (SELECT COUNT(*) FROM faculty WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)) as faculty_30days
        """)
        recent_registrations = cursor.fetchone()
        
        # Get active invites count
        cursor.execute("""
            SELECT COUNT(*) as active_invites
            FROM invites 
            WHERE expires_at > NOW() AND used = 0
        """)
        active_invites = cursor.fetchone()['active_invites']
        
        # Get year/section distribution
        cursor.execute("""
            SELECT year_section, COUNT(*) as count 
            FROM students 
            WHERE status = 'active' 
            GROUP BY year_section 
            ORDER BY year_section
        """)
        year_section_stats = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Format the response to match your HTML structure
        response_data = {
            'success': True,
            'stats': {
                # Main dashboard cards
                'total_students': total_students,
                'total_faculty': total_faculty,
                'cs_students': cs_count,
                'it_students': it_count, 
                'act_students': act_count,
                
                # Detailed breakdowns
                'course_breakdown': [
                    {
                        'course': course['course'],
                        'count': course['count'],
                        'percentage': round((course['count'] / total_students * 100), 1) if total_students > 0 else 0
                    }
                    for course in course_stats
                ],
                
                'department_breakdown': [
                    {
                        'department': dept['department'],
                        'count': dept['count'],
                        'percentage': round((dept['count'] / total_faculty * 100), 1) if total_faculty > 0 else 0
                    }
                    for dept in department_stats
                ],
                
                'year_section_breakdown': [
                    {
                        'year_section': ys['year_section'],
                        'count': ys['count'],
                        'percentage': round((ys['count'] / total_students * 100), 1) if total_students > 0 else 0
                    }
                    for ys in year_section_stats
                ],
                
                # Attendance stats
                'attendance': {
                    'present_today': attendance_today['present_today'] or 0,
                    'total_records_today': attendance_today['total_attendance_records_today'] or 0,
                    'unique_students_week': attendance_week['unique_students_week'] or 0,
                    'total_records_week': attendance_week['total_records_week'] or 0
                },
                
                # Recent activity
                'recent_activity': {
                    'students_registered_30days': recent_registrations['students_30days'] or 0,
                    'faculty_registered_30days': recent_registrations['faculty_30days'] or 0,
                    'active_invites': active_invites
                },
                
                # Status summary
                'summary': {
                    'total_users': total_students + total_faculty,
                    'attendance_rate_today': round((attendance_today['present_today'] / total_students * 100), 1) if total_students > 0 else 0,
                    'most_popular_course': course_stats[0]['course'] if course_stats else 'N/A',
                    'largest_department': department_stats[0]['department'] if department_stats else 'N/A'
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error fetching enhanced dashboard stats: {e}")
        return jsonify({
            'success': False, 
            'message': str(e),
            'stats': {
                'total_students': 0,
                'total_faculty': 0,
                'cs_students': 0,
                'it_students': 0,
                'act_students': 0
            }
        })


@app.route('/api/get_course_distribution', methods=['GET'])
def get_course_distribution():
    """Get detailed course distribution for charts and analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get course distribution with year sections
        cursor.execute("""
            SELECT 
                course,
                year_section,
                COUNT(*) as student_count,
                GROUP_CONCAT(CONCAT(first_name, ' ', last_name) SEPARATOR ', ') as student_names
            FROM students 
            WHERE status = 'active'
            GROUP BY course, year_section
            ORDER BY course, year_section
        """)
        
        detailed_distribution = cursor.fetchall()
        
        # Get summary by course only
        cursor.execute("""
            SELECT 
                course,
                COUNT(*) as total_students,
                COUNT(DISTINCT year_section) as sections_count
            FROM students 
            WHERE status = 'active'
            GROUP BY course
            ORDER BY total_students DESC
        """)
        
        course_summary = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'detailed_distribution': detailed_distribution,
            'course_summary': course_summary
        })
        
    except Exception as e:
        logger.error(f"Error fetching course distribution: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/get_faculty_distribution', methods=['GET'])
def get_faculty_distribution():
    """Get detailed faculty distribution by department and designation"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get faculty distribution by department and designation
        cursor.execute("""
            SELECT 
                department,
                designation,
                COUNT(*) as faculty_count,
                GROUP_CONCAT(CONCAT(first_name, ' ', last_name) SEPARATOR ', ') as faculty_names
            FROM faculty 
            WHERE status = 'active'
            GROUP BY department, designation
            ORDER BY department, designation
        """)
        
        detailed_distribution = cursor.fetchall()
        
        # Get summary by department only
        cursor.execute("""
            SELECT 
                department,
                COUNT(*) as total_faculty,
                COUNT(DISTINCT designation) as designation_count
            FROM faculty 
            WHERE status = 'active'
            GROUP BY department
            ORDER BY total_faculty DESC
        """)
        
        department_summary = cursor.fetchall()
        
        # Get summary by designation only
        cursor.execute("""
            SELECT 
                designation,
                COUNT(*) as total_faculty,
                COUNT(DISTINCT department) as department_count
            FROM faculty 
            WHERE status = 'active'
            GROUP BY designation
            ORDER BY total_faculty DESC
        """)
        
        designation_summary = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'detailed_distribution': detailed_distribution,
            'department_summary': department_summary,
            'designation_summary': designation_summary
        })
        
    except Exception as e:
        logger.error(f"Error fetching faculty distribution: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Add this decorator function before the routes
# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = request.cookies.get('session_token')
        if not session_token:
            return redirect(url_for('login_page'))
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM user_sessions WHERE session_token = %s AND expires_at > NOW()",
                (session_token,)
            )
            session = cursor.fetchone()
            cursor.close()
            conn.close()
            if not session:
                return redirect(url_for('login_page'))
            g.user = session
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Serve student photos
@app.route('/student_photos/<filename>')
def serve_photo(filename):
    return send_from_directory('student_photos', filename)

# Serve faculty photos
os.makedirs('faculty_photos', exist_ok=True)

@app.route('/faculty_photos/<filename>')
def serve_faculty_photo(filename):
    return send_from_directory('faculty_photos', filename)

@app.route('/api/get_other_students', methods=['GET'])  # Different route path
def get_other_students():
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

@app.route('/api/get_faculty_list', methods=['GET'])
def get_faculty_list():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT faculty_id, first_name, last_name, department, designation, photo_path FROM faculty")
        faculty = cursor.fetchall()
        cursor.close()
        conn.close()
        
        formatted_faculty = [
            {
                'idNumber': i['faculty_id'],
                'name': f"{i['first_name']} {i['last_name']}",
                'department': i['department'],
                'designation': i['designation'],
                'photo': i['photo_path'] if i['photo_path'] else f"https://ui-avatars.com/api/?name={i['first_name']}+{i['last_name']}&background=random"
            }
            for i in faculty
        ]
        
        return jsonify({'success': True, 'faculty': formatted_faculty})
    except Exception as e:
        logger.error(f"Error fetching faculty: {e}")
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
    
@app.route('/api/generate_dynamic_invite', methods=['POST'])
def generate_dynamic_invite():
    try:
        data = request.get_json()
        invite_type = data.get('type')  # 'student' or 'faculty'
        expiry_days = int(data.get('days', 0))
        expiry_hours = int(data.get('hours', 0))
        expiry_minutes = int(data.get('minutes', 0))

        # Validate invite type
        if invite_type not in ['student', 'faculty']:
            return jsonify({'success': False, 'message': 'Invalid invite type'}), 400

        # Calculate expiration time
        expiry_delta = timedelta(days=expiry_days, hours=expiry_hours, minutes=expiry_minutes)
        if expiry_delta <= timedelta(0):
            expiry_delta = timedelta(hours=24)  # Default to 24 hours
        expires_at = datetime.now() + expiry_delta

        # Generate unique token
        token = secrets.token_urlsafe(32)

        # Insert into invites table - matching your actual schema
        conn = None
        cursor = None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Your schema: id, token, invite_type, created_by, max_uses, current_uses, notes, expires_at, used, created_at
            cursor.execute(
                """INSERT INTO invites 
                   (token, invite_type, max_uses, current_uses, expires_at, used) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (token, invite_type, 1, 0, expires_at, 0)
            )
            
            # Get the inserted ID
            invite_id = cursor.lastrowid
            
            # Commit BEFORE verification
            conn.commit()
            
            logger.info(f"Committed invite with ID {invite_id}, token: {token[:10]}...")
            
            # Verify the insert succeeded
            cursor.execute(
                "SELECT id, token, invite_type FROM invites WHERE id = %s",
                (invite_id,)
            )
            verify = cursor.fetchone()
            
            if not verify:
                raise Exception("Token insertion failed verification")
            
            logger.info(f"Generated {invite_type} invite token: {token} (ID: {invite_id})")
            
        except mysql.connector.Error as db_error:
            if conn:
                conn.rollback()
            logger.error(f"Database error during invite generation: {db_error}")
            raise Exception(f"Database error: {str(db_error)}")
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        # Generate the appropriate invite link
        if invite_type == 'student':
            invite_link = f"{request.host_url}studentreg?token={token}"
        else:
            invite_link = f"{request.host_url}facultyreg?token={token}"

        return jsonify({
            'success': True,
            'message': f'{invite_type.capitalize()} invite link generated successfully',
            'token': token,
            'link': invite_link,
            'expires_at': expires_at.isoformat(),
            'type': invite_type
        })

    except Exception as e:
        logger.error(f"Error generating invite: {str(e)}")
        return jsonify({'success': False, 'message': f'Error generating invite: {str(e)}'}), 500   

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

@app.route('/api/update_faculty', methods=['POST'])
def update_faculty():
    try:
        data = request.json
        faculty_id = data.get('faculty_id')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        department = data.get('department')
        designation = data.get('designation')

        if not all([faculty_id, first_name, last_name, department, designation]):
            return jsonify({'success': False, 'message': 'All fields are required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE faculty 
               SET first_name = %s, last_name = %s, department = %s, designation = %s 
               WHERE faculty_id = %s""",
            (first_name, last_name, department, designation, faculty_id)
        )
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Faculty Member not found'})
            
        conn.commit()
        cursor.close()
        conn.close()
        
        load_known_faculties_from_db()  # Refresh
        return jsonify({'success': True, 'message': 'Faculty Member updated successfully'})
    except Exception as e:
        logger.error(f"Error updating faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_faculty', methods=['POST'])
def delete_faculty():
    try:
        faculty_id = request.json.get('faculty_id')
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID is required'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faculty WHERE faculty_id = %s", (faculty_id,))
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Faculty Member not found'})

        conn.commit()
        cursor.close()
        conn.close()
        
        # Remove photo if exists
        photo_path = f"faculty_photos/{faculty_id}.jpg"
        if os.path.exists(photo_path):
            os.remove(photo_path)
            logger.info(f"Deleted photo for faculty {faculty_id}")

        load_known_faculties_from_db()  # Refresh
        if faculty_id in locked_tracks:
            del locked_tracks[faculty_id]
        return jsonify({'success': True, 'message': 'Faculty Member deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/send_otp', methods=['POST'])
def send_otp():
    email = request.json.get('email', '').strip()
    
    if not email or "@wmsu.edu.ph" not in email:
        logger.warning(f"Invalid email received: {email}")
        return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
    
    otp_code = generate_otp()
    expires_at = datetime.now() + timedelta(minutes=10)  # Use datetime.now() and timedelta directly
    
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
        
        if datetime.now() > expires_at:  # Fixed: Use datetime.now() instead of datetime.datetime.now()
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
        
        # Much more lenient pose detection thresholds
        pose_results = {
            'is_frontal': bool(abs(yaw) <= 45 and abs(pitch) <= 40 and abs(roll) <= 35),
            'is_right': bool(yaw <= -5),  # More lenient
            'is_left': bool(yaw >= 5),   # More lenient
            'is_up': bool(pitch <= -5),  # More lenient
            'is_down': bool(pitch >= 5), # More lenient
            'is_mouth_open': bool(mar >= 0.15),  # Much more lenient
            'is_eyes_closed': bool((left_ear + right_ear) / 2 <= 0.4)  # More lenient
        }
        
        logger.info(f"Pose results for {current_pose}: {pose_results}, yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}, mar={mar:.3f}, left_ear={left_ear:.3f}, right_ear={right_ear:.3f}")
        
        pose_satisfied = False
        message = ""
        
        # More flexible pose checking with fallback conditions
        if current_pose == 'frontal':
            pose_satisfied = pose_results['is_frontal']
            message = "Frontal pose detected successfully." if pose_satisfied else "Please face the camera directly."
        elif current_pose == 'right':
            pose_satisfied = pose_results['is_right'] or yaw <= -5  # Even more lenient fallback
            message = "Right pose detected successfully." if pose_satisfied else "Please turn your head to the right."
        elif current_pose == 'left':
            pose_satisfied = pose_results['is_left'] or yaw >= 5   # Even more lenient fallback
            message = "Left pose detected successfully." if pose_satisfied else "Please turn your head to the left."
        elif current_pose == 'up':
            pose_satisfied = pose_results['is_up'] or pitch <= -5  # More lenient fallback
            message = "Upward pose detected successfully." if pose_satisfied else "Please tilt your head up."
        elif current_pose == 'down':
            pose_satisfied = pose_results['is_down'] or pitch >= 5 # More lenient fallback
            message = "Downward pose detected successfully." if pose_satisfied else "Please tilt your head down."
        elif current_pose == 'mouth_open':
            pose_satisfied = pose_results['is_mouth_open'] or mar >= 0.12  # Very lenient
            message = "Mouth open detected successfully." if pose_satisfied else "Please open your mouth wider."
        elif current_pose == 'eyes_closed':
            pose_satisfied = pose_results['is_eyes_closed'] or ((left_ear + right_ear) / 2 <= 0.4)  # Very lenient
            message = "Eyes closed detected successfully." if pose_satisfied else "Please close your eyes."
        
        if pose_satisfied:
            pose_embeddings[current_pose] = face_embedding.tolist()
            next_pose_index = min(current_pose_index + 1, len(POSE_SEQUENCE) - 1)
            next_pose = POSE_SEQUENCE[next_pose_index]
            logger.info(f"Pose {current_pose} satisfied, advancing to {next_pose}")
        else:
            next_pose = current_pose
            logger.info(f"Pose {current_pose} not satisfied, retrying")
        
        encoding_response = face_embedding.tolist() if current_pose == 'frontal' else []
        
        # Convert all numpy types to native Python types to avoid JSON serialization issues
        return jsonify({
            'success': bool(pose_satisfied),  # Ensure it's a Python bool
            'message': str(message),
            'current_pose': str(current_pose),
            'next_pose': str(next_pose),
            'encoding': encoding_response,
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
            'mar': float(mar),
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            # Convert numpy bool_ to Python bool explicitly
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
        invite_token = data.get('invite_token', '').strip()  
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
        
        # Hash password
        password_hash = hash_password(password)
        
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
        if len(pose_embeddings) >= 3:
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
                
                # Verify photo matches face scan
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
        
        # Insert student with hashed password
        cursor.execute(
            """INSERT INTO students 
            (student_id, first_name, last_name, middle_name, course, year_section, email, face_encoding, photo_path, password_hash) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (student_id, first_name, last_name, middle_name or None, course, year_section, email, encoding_str, photo_path, password_hash)
        )
        conn.commit()  # COMMIT THE STUDENT INSERT FIRST
        cursor.close()
        conn.close()
        
        # Update invite token uses (ONLY ONCE, AFTER student is registered)
        if invite_token:
            try:
                conn_invite = get_db_connection()
                cursor_invite = conn_invite.cursor()

                # Increment current_uses
                cursor_invite.execute(
                    "UPDATE invites SET current_uses = current_uses + 1 WHERE token = %s",
                    (invite_token,)
                )

                # Check if max uses reached and mark as used
                cursor_invite.execute(
                    "SELECT current_uses, max_uses FROM invites WHERE token = %s",
                    (invite_token,)
                )
                result = cursor_invite.fetchone()

                if result and result[0] >= result[1]:
                    cursor_invite.execute(
                        "UPDATE invites SET used = 1 WHERE token = %s",
                        (invite_token,)
                    )
                    logger.info(f"Invite token {invite_token} has reached max uses and marked as used")

                conn_invite.commit()
                cursor_invite.close()
                conn_invite.close()

                logger.info(f"Incremented uses for invite token: {invite_token}")
            except Exception as e:
                logger.error(f"Failed to update invite uses: {e}")
        
        # Reload known faces
        load_known_faces_from_db()
        pose_embeddings.clear()
        
        logger.info(f"Student registered: {student_id} ({first_name} {last_name})")
        return jsonify({'success': True, 'message': 'Student registered successfully'})

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})
    

@app.route('/api/register_faculty', methods=['POST'])
def register_faculty():
    try:
        data = request.form
        email = data.get('email', '').strip()
        faculty_id = data.get('faculty_id', '').strip()
        invite_token = data.get('invite_token', '').strip()
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        middle_name = data.get('middle_name', '').strip()
        department = data.get('department', '').strip()
        designation = data.get('designation', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'moderator').strip()
        
        if not all([email, faculty_id, first_name, last_name, department, designation, password]):
            logger.warning("Missing required fields in register_faculty request")
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if "@wmsu.edu.ph" not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
        
        # Validate role
        if role not in ['super_admin', 'admin', 'moderator']:
            role = 'moderator'
        
        # Hash password
        password_hash = hash_password(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT faculty_id FROM faculty WHERE faculty_id = %s OR email = %s", 
                      (faculty_id, email))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            logger.warning(f"Faculty ID {faculty_id} or email {email} already exists")
            return jsonify({'success': False, 'message': 'Faculty ID or email already exists'})
        
        # Average embeddings from multiple poses
        if len(pose_embeddings) >= 3:
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
                os.makedirs('faculty_photos', exist_ok=True)
                photo_path = f"faculty_photos/{faculty_id}.jpg"
                photo.save(photo_path)
                logger.info(f"Saved photo for faculty {faculty_id} at {photo_path}")
                
                # Verify photo matches face scan
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
        
        # Insert faculty with hashed password and role
        cursor.execute(
            """INSERT INTO faculty 
            (faculty_id, first_name, last_name, middle_name, department, designation, email, face_encoding, photo_path, password_hash, role) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (faculty_id, first_name, last_name, middle_name or None, department, designation, email, encoding_str, photo_path, password_hash, role)
        )
        conn.commit()  # COMMIT THE FACULTY INSERT FIRST
        cursor.close()
        conn.close()

        # Update invite token uses (ONLY ONCE, AFTER faculty is registered)
        if invite_token:
            try:
                conn_invite = get_db_connection()
                cursor_invite = conn_invite.cursor()

                # Increment current_uses
                cursor_invite.execute(
                    "UPDATE invites SET current_uses = current_uses + 1 WHERE token = %s",
                    (invite_token,)
                )

                # Check if max uses reached and mark as used
                cursor_invite.execute(
                    "SELECT current_uses, max_uses FROM invites WHERE token = %s",
                    (invite_token,)
                )
                result = cursor_invite.fetchone()

                if result and result[0] >= result[1]:
                    cursor_invite.execute(
                        "UPDATE invites SET used = 1 WHERE token = %s",
                        (invite_token,)
                    )
                    logger.info(f"Invite token {invite_token} has reached max uses and marked as used")

                conn_invite.commit()
                cursor_invite.close()
                conn_invite.close()

                logger.info(f"Incremented uses for invite token: {invite_token}")
            except Exception as e:
                logger.error(f"Failed to update invite uses: {e}")
        
        # Reload known faces
        load_known_faculties_from_db()
        pose_embeddings.clear()
        
        logger.info(f"Faculty registered: {faculty_id} ({first_name} {last_name}) with role {role}")
        return jsonify({'success': True, 'message': 'Faculty registered successfully'})
        
    except Exception as e:
        logger.error(f"Faculty registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

# Modified health check to not fail on camera issues
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db_ok = True
        try:
            conn = get_db_connection()
            db_ok = conn.is_connected()
            conn.close()
        except Exception:
            db_ok = False
            
        model_ok = yolo is not None and face_analysis is not None
        
        return jsonify({
            'success': True,
            'camera': camera_available,
            'camera_type': 'rtsp' if camera_available and not use_dummy_feed else 'dummy',
            'database': db_ok,
            'models': model_ok,
            'active_tracks': len(tracks) if camera_available else 0,
            'locked_tracks': len(locked_tracks) if camera_available else 0
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': True,  # Keep this True so app doesn't fail
            'camera': False,
            'database': False,
            'models': False,
            'error': str(e)
        })


def generate_frames():
    """Modified to handle dummy feed scenario"""
    frame_idx = 0
    
    while True:
        with cap_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
            
        # Only do face processing if real camera is available
        if camera_available and not use_dummy_feed:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Update trackers and refresh detections
                update_trackers(rgb, frame, frame_idx)
                refresh_with_detections(frame, rgb, frame_idx)
                frame_idx += 1
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
        else:
            # Just display the dummy frame
            pass
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
# Add an API endpoint to check camera status
@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    return jsonify({
        'camera_available': camera_available,
        'using_dummy_feed': use_dummy_feed,
        'active_tracks': len(tracks) if camera_available else 0,
        'locked_tracks': len(locked_tracks) if camera_available else 0
    })

# Add an API endpoint to retry camera connection
@app.route('/api/reconnect_camera', methods=['POST'])
def reconnect_camera():
    try:
        if open_stream():
            return jsonify({
                'success': True, 
                'camera_available': camera_available,
                'message': 'Camera connected successfully' if camera_available else 'Using fallback feed'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Failed to connect to camera'
            })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Connection error: {str(e)}'
        })


# Routes
@app.route('/timer')
@login_required
def timer_page():
    return render_template('Timer.html')

@app.route('/camfootage')
@login_required
def camfootage_page():
    return render_template('CamFootage.html')

@app.route('/sidebar')
@login_required
def sidebar_page():
    user = g.get('user', {})
    user_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "Unknown User"
    return render_template('sidebar.html', user_name=user_name)

@app.route('/summary')
@login_required
def summary_page():
    return render_template('Summary.html')

@app.route('/schedule')
@login_required
def schedule_page():
    return render_template('schedule.html')

@app.route('/programs')
@login_required
def programs_page():
    return render_template('programs.html')

@app.route('/classsched')
@login_required
def classsched_page():
    return render_template('classsched.html')

@app.route('/studentreg')
def studentreg_page():
    """Student registration page - requires valid invite token"""
    token = request.args.get('token')
    
    if not token:
        return redirect(url_for('login_page'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all relevant fields including current_uses and max_uses
        cursor.execute(
            "SELECT expires_at, used, current_uses, max_uses, invite_type FROM invites WHERE token = %s",
            (token,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            logger.warning(f"Invalid invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        expires_at, used, current_uses, max_uses, invite_type = result
        
        # Validate invite type
        if invite_type != 'student':
            logger.warning(f"Wrong invite type for student registration: {invite_type}")
            return redirect(url_for('login_page'))
        
        # Check if token is used up (either marked as used OR current_uses >= max_uses)
        if used == 1 or current_uses >= max_uses:
            logger.warning(f"Used up invite token attempted: {token} (used={used}, {current_uses}/{max_uses})")
            return redirect(url_for('login_page'))
            
        # Check if token is expired
        if datetime.now() > expires_at:
            logger.warning(f"Expired invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        # Valid token - show registration page
        logger.info(f"Valid student invite token accessed: {token} ({current_uses}/{max_uses} uses)")
        return render_template('studentreg.html', token=token)
        
    except Exception as e:
        logger.error(f"Error validating invite token: {e}")
        return redirect(url_for('login_page'))

@app.route('/facultyreg')
def faculty_reg_page():
    """Faculty registration page - requires valid invite token"""
    token = request.args.get('token')
    
    if not token:
        return redirect(url_for('login_page'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all relevant fields including current_uses and max_uses
        cursor.execute(
            "SELECT expires_at, used, current_uses, max_uses, invite_type FROM invites WHERE token = %s",
            (token,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            logger.warning(f"Invalid invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        expires_at, used, current_uses, max_uses, invite_type = result
        
        # Validate invite type
        if invite_type != 'faculty':
            logger.warning(f"Wrong invite type for faculty registration: {invite_type}")
            return redirect(url_for('login_page'))
        
        # Check if token is used up (either marked as used OR current_uses >= max_uses)
        if used == 1 or current_uses >= max_uses:
            logger.warning(f"Used up invite token attempted: {token} (used={used}, {current_uses}/{max_uses})")
            return redirect(url_for('login_page'))
            
        # Check if token is expired
        if datetime.now() > expires_at:
            logger.warning(f"Expired invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        # Valid token - show registration page
        logger.info(f"Valid faculty invite token accessed: {token} ({current_uses}/{max_uses} uses)")
        return render_template('facultyreg.html', token=token)
        
    except Exception as e:
        logger.error(f"Error validating invite token: {e}")
        return redirect(url_for('login_page'))
    
@app.route('/subject')
def subject_page():
    return render_template('subject.html')

@app.route('/AdminDB')
@login_required
def admin_db_page():
    return render_template('AdminDB.html')

@app.route('/StudentDB')
@login_required
def student_db_page():
    return render_template('StudentDB.html')

@app.route('/FacultyDB')
@login_required
def faculty_db_page():
    return render_template('FacultyDB.html')

@app.route('/settings')
@login_required
def settings_page():
    return render_template('settings.html')

@app.route('/StudentLP')
@login_required
def student_lp_page():
    return render_template('StudentLP.html')

@app.route('/StudSettings')
@login_required
def student_settings_page():
    return render_template('StudSettings.html')

@app.route('/StudAttendance')
@login_required
def student_attendance_page():
    return render_template('StudAttendance.html')

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        session_token = request.cookies.get('session_token')
        
        if session_token:
            # Delete session from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM user_sessions WHERE session_token = %s",
                (session_token,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Session {session_token[:8]}... deleted")
        
        # Create response that clears the cookie
        resp = jsonify({'success': True, 'message': 'Logged out successfully'})
        resp.set_cookie('session_token', '', expires=0, httponly=True, secure=False, samesite='Strict')
        
        return resp
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'success': False, 'message': 'Logout failed'})

@app.route('/logout')
def logout_page():
    """Direct logout route that redirects to login"""
    session_token = request.cookies.get('session_token')
    
    if session_token:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE session_token = %s", (session_token,))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Logout error: {e}")
    
    resp = make_response(redirect(url_for('login_page')))
    resp.set_cookie('session_token', '', expires=0, httponly=True, secure=False, samesite='Strict')
    return resp

if __name__ == "__main__":
    # Initialize global variables (no need for global keyword here)
    latest_frame = None
    stop_flag = False
    camera_available = False
    use_dummy_feed = False
    dummy_frame = None
    
    # Try to connect to camera, but don't exit if it fails
    if not open_stream():
        logger.warning("Initial camera connection failed, but continuing with dummy feed")
    
    # Start the grabber thread
    grab_thread = threading.Thread(target=grabber, daemon=True)
    grab_thread.start()
    
    try:
        ssl_context = None
        cert_path = 'cert.pem'
        key_path = 'key.pem'
        
        # Setup SSL if certificates exist
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.load_cert_chain(cert_path, key_path)
            ssl_context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384')
            logger.info("Running server with HTTPS")
        else:
            logger.warning("SSL certificates not found. Running with HTTP")
        
        # Start Flask app
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context=ssl_context)
    
    finally:
        # Signal video capture loop to stop
        stop_flag = True
        time.sleep(0.05)
        
        # Release camera resource safely
        with cap_lock:
            if cap is not None:
                cap.release()
        
        # Save attendance to CSV if recognition is enabled
        if ENABLE_RECOGNITION:
            try:
                with open("attendance_log.csv", "w") as f:
                    f.write("ID,Name,DateTime\n")
                    for sid, data in attendance.items():
                        f.write(f"{sid},{data['name']},{data['time']}\n")
                logger.info("Attendance saved to attendance_log.csv")
            except Exception as e:
                logger.error(f"Failed to save attendance CSV: {e}", exc_info=True)

