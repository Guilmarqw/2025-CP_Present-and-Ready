import csv
from functools import wraps
import os
import sys
import cv2
import time
import dlib
import torch
import numpy as np
import datetime
import traceback
import threading
import insightface
import base64
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error
import smtplib
import torchreid
from torchreid import metrics
import random
import string
import json
import secrets
import uuid
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from flask import Flask, Response, g, make_response, render_template, request, jsonify, redirect, url_for, send_from_directory, session
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment
import bcrypt
import secrets
from datetime import datetime, timedelta, date
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from supervision import ByteTrack
from supervision.tracker.byte_tracker.core import ByteTrack
from typing import List, Dict, Any, Tuple



# =========================
# Flask streaming & API
# =========================
app = Flask(__name__)
app.secret_key = 'face-attendance-system-secret-key-2025'  # Add this line
app.template_folder = 'templates'
app.static_folder = 'static'
CORS(app)
# Add these Flask routes for authentication

app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key-change-in-production')

# =========================
# OPTIMIZED CONFIG FOR FIXED CODE
# =========================

# --- MOCK DEPENDENCIES ---
# In a real environment, replace these mock objects with actual library calls (e.g., insightface/deepface for analysis, actual CV2 for image ops).
try:
    import cv2
except ImportError:
    class MockCV2:
        def cvtColor(self, img, code): return img
        def createCLAHE(self, clipLimit, tileGridSize): return type('CLAHE', (object,), {'apply': lambda self, x: x})()
        def equalizeHist(self, gray): return gray
        def bilateralFilter(self, gray, d, sigmaColor, sigmaSpace): return gray
        COLOR_BGR2GRAY = 1
        COLOR_GRAY2BGR = 2
    cv2 = MockCV2()

class MockFaceObject:
    """Simulates the output of a face analysis model."""
    def __init__(self, embedding):
        self.embedding = embedding
        self.bbox = np.array([0, 0, 100, 100])
        self.det_score = 0.9

class MockFaceAnalysis:
    """Simulates the face analysis model."""
    def get(self, image):
        # Mocking a single face detection with a random embedding
        return [MockFaceObject(np.random.rand(128))]
face_analysis = MockFaceAnalysis()

# --- GLOBAL DATA STRUCTURES ---
tracks: List[Dict[str, Any]] = []
locked_tracks: Dict[str, Dict[str, Any]] = {}
pending_confirmations: Dict[str, Dict[str, Any]] = {}
UNKNOWN_FACES_FOR_ENROLLMENT: Dict[str, Dict[str, Any]] = {}

FACE_SCAN_START_TIME = None
FACE_SCAN_DURATION = 15  # seconds
is_face_scan_active_flag = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache after each YOLO call

face_scan_start_time = None

# Add these to your global variables
student_presence_tracker = {}  # Tracks when students are present/missing
current_session_id = None  # Make sure this is set

# Initialize FaceAnalysis with SCRFD
face_analysis = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analysis.prepare(ctx_id=0, det_size=(320, 320))  # Adjust det_size for your resolution


# Add these global variables at the top with your other globals
ACTIVE_FACE_TRACKS = {}
FACE_COOLDOWN_PERIOD = 30  # seconds
thread_safe_lock = threading.RLock()
FACE_SIMILARITY_THRESHOLD = 0.6

detectionStopped = False
current_fps = 30.0
frame_timestamps = []
skip_frame_counter = 0

session_start_time = None
session_total_duration_seconds = 3600  # Default 60 minutes
session_threshold_seconds = 900        # Default 15 minutes

pending_confirmations = {}  # {person_id: {'frames': [], 'body_boxes': [], 'name': str, 'type': str}}
locked_track_reid_features = {}
student_status = {}  # {student_id: 'absent' | 'present' | 'late'}
current_session_students = []  # List of student IDs for current class

# ✅ OPTIMIZED: Confirmation parameters for fast & accurate recognition
CONFIRMATION_FRAMES_REQUIRED = 2  # 2 consecutive frames = ~0.067 seconds
CONFIRMATION_SIMILARITY_THRESHOLD = 0.50  # Balanced threshold (not too strict, not too loose)
BODY_MATCH_IOU_THRESHOLD = 0.03  # Low threshold for better body matching
FACE_TO_BODY_VERTICAL_RATIO = 0.7  # Face should be in upper 60% of body
REID_DISTANCE_THRESHOLD = 0.4  # Strict ReID for overlap prevention
DETECT_EVERY = 1

current_rtsp_url = None
WEIGHTS_PATH = "yolov8n-face.pt"
STREAM_WIDTH, STREAM_HEIGHT = 1920, 1080
pose_embeddings = {}

# Add this near your other global variables
ATTENDANCE_CSV_FILE = "attendance_log.csv"
attendance_save_interval = 300  # 5 minutes

# Add these global variables near the top with other globals
camera_available = False
use_dummy_feed = False
dummy_frame = None
latest_frame = None
stop_flag = False

MAX_TRACKS = 100
MAX_UNLOCKED_TRACKS = 30
EXPAND_BOX_RATIO = 0.4

PASSWORD_RESET_EXPIRE_HOURS = 24
OTP_RESEND_COOLDOWN = 30  # seconds
MAX_OTP_ATTEMPTS = 3

ENABLE_RECOGNITION = True
TOLERANCE = 0.5  # InsightFace uses different distance metric
KNOWN_DIR = "known_faces"

GRAB_SLEEP = 0.01
MAX_EMPTY_GRABS = 150

# Distance settings
MAX_RECOGNITION_DISTANCE = 10
FACE_SIZE_FOR_DISTANCE = 90

# ✅ OPTIMIZED: Locking configuration
LOCK_TIMEOUT_FRAMES = 120  # 3 seconds at 30 FPS (was 60)
LOCK_MISS_THRESHOLD = 15  # 0.5 seconds before removing lock (faster cleanup)
PENDING_CONFIRMATION_TIMEOUT = 10  # Frames before cleaning up stale confirmations
ACTIVE_PENDING_WINDOW = 5  # Only draw confirmations seen in last 5 frames
MAX_TELEPORT_DISTANCE = 200  # Max pixels a person can move per frame

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facesys',
    'autocommit': False,
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
# Load YOLOv8-Face (Single initialization)
# =========================
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"'{WEIGHTS_PATH}' not found. Download yolov8n-face.pt and place it next to this script.")

# Define device once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize face detector
yolo = YOLO(WEIGHTS_PATH)
yolo.to(DEVICE)
logger.info(f"Using device: {DEVICE}  |  Model: {WEIGHTS_PATH}")

# Initialize body detector for person tracking (Single initialization with error handling)
try:
    body_detector = YOLO('yolov8n.pt')  # Standard YOLO for person detection
    body_detector.to(DEVICE)
    logger.info(f"✅ Body detector (YOLOv8n) loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"❌ Failed to load body detector: {e}")
    logger.info("Downloading yolov8n.pt model...")
    body_detector = YOLO('yolov8n.pt')  # This will auto-download
    body_detector.to(DEVICE)

# Initialize ByteTrack
byte_tracker = ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30
)
logger.info("✅ ByteTrack initialized for body tracking")

# =========================
# TorchReID OSNet Model - WORKING VERSION
# =========================
try:
    reid_model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        use_gpu=torch.cuda.is_available()
    )

    # Use the exact path to the cached file
    cache_path = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
    
    if os.path.exists(cache_path):
        torchreid.utils.load_pretrained_weights(reid_model, cache_path)
        logger.info(f"✅ TorchReID model loaded from cache: {cache_path}")
    else:
        logger.error("ReID model cache file not found")
        reid_model = None
        raise FileNotFoundError("ReID model cache file not found")
    
    reid_model.eval()
    if torch.cuda.is_available():
        reid_model = reid_model.cuda()
    logger.info(f"✅ TorchReID model loaded successfully on {DEVICE}")
    
except Exception as e:
    logger.error(f"Failed to load reid: {e}")
    reid_model = None

# =========================
# Load InsightFace model - FIXED CUDA INITIALIZATION
# =========================
try:
    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available_providers}")

    # ✅ SIMPLIFIED CUDA SETUP - Remove problematic options
    providers = []
    
    # Check if CUDA is available
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 🎯 SIMPLIFIED: Remove problematic 'memory_pattern' option
        provider_options = [
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            },
            {}
        ]
        ctx_id = 0
        logger.info("✅ Using CUDA + CPU providers")
    else:
        # Fallback to CPU only
        providers = ['CPUExecutionProvider']
        provider_options = [{}]
        ctx_id = -1
        logger.info("⚠️ CUDA not available, using CPU only")

    # Initialize FaceAnalysis
    face_analysis = insightface.app.FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        providers=providers,
        provider_options=provider_options
    )

    # Prepare with optimized settings
    face_analysis.prepare(
        ctx_id=ctx_id,
        det_size=(320, 320),  # Lower resolution = less GPU memory usage
        det_thresh=0.6
    )

    logger.info(f"🎯 InsightFace model '{INSIGHTFACE_MODEL}' loaded successfully")

    # Test the model to verify it works
    try:
        # Create a small test image to verify detection works
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_faces = face_analysis.get(test_img)
        logger.info("✅ InsightFace model test passed - detection method is available")
    except Exception as test_error:
        logger.error(f"❌ InsightFace model test failed: {test_error}")
        raise ValueError("FaceAnalysis detect method not working properly")

except Exception as e:
    logger.error(f"❌ Failed to load InsightFace model: {e}")
    
    # More robust fallback with multiple attempts
    fallback_success = False
    fallback_models = ['buffalo_l', 'antelopev2', 'buffalo_s']
    
    for model_name in fallback_models:
        try:
            logger.info(f"🔄 Attempting fallback with model: {model_name}")
            face_analysis = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            face_analysis.prepare(ctx_id=-1, det_size=(320, 320))
            
            # Test the fallback model
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_faces = face_analysis.get(test_img)
            
            logger.info(f"✅ Fallback successful with model: {model_name}")
            fallback_success = True
            INSIGHTFACE_MODEL = model_name  # Update the model name
            break
            
        except Exception as fallback_error:
            logger.warning(f"⚠️ Fallback with {model_name} failed: {fallback_error}")
            continue
    
    if not fallback_success:
        logger.error("❌ All initialization methods failed - InsightFace cannot be loaded")
        face_analysis = None
        ENABLE_RECOGNITION = False
    else:
        ENABLE_RECOGNITION = True

# =========================
# Utilities
# =========================
# Thread-safe state manager

def create_dummy_frame():
    """Create a dummy frame when no camera is available"""
    try:
        frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)  # Use new dimensions
        
        # Add background color (dark gray)
        frame[:] = (40, 40, 40)

        # Add text
        text = 'No Camera Connected'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Center text
        x = (STREAM_WIDTH - text_width) // 2
        y = (STREAM_HEIGHT + text_height) // 2

        # Text outline and fill
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Subtitle
        subtitle = 'Waiting for camera connection...'
        font_scale_small = 0.8
        thickness_small = 2
        (sub_width, sub_height), _ = cv2.getTextSize(subtitle, font, font_scale_small, thickness_small)
        sub_x = (STREAM_WIDTH - sub_width) // 2
        sub_y = y + 60

        cv2.putText(frame, subtitle, (sub_x, sub_y), font, font_scale_small, (0, 0, 0), thickness_small + 1, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (sub_x, sub_y), font, font_scale_small, (180, 180, 180), thickness_small, cv2.LINE_AA)

        return frame

    except Exception as e:
        logger.error(f"Error creating dummy frame: {e}")
        # Return a basic black frame as absolute fallback
        return np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)

def hash_password(password):
    """Hash a password for storing in database"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password, hashed_password):
    """Verify password with bcrypt with proper error handling"""
    try:
        if not plain_password or not hashed_password:
            logger.warning("Missing password or hash")
            return False
        
        # Check if the hash starts with bcrypt identifier
        if hashed_password.startswith('$2b$'):
            return bcrypt.checkpw(
                plain_password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        else:
            # Handle plain text passwords (for testing only - remove in production)
            logger.warning("Non-bcrypt hash detected")
            return plain_password == hashed_password
            
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def generate_reset_token():
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(48)

def authenticate_user(email, password):
    """Enhanced authentication with detailed logging"""
    conn = None
    cursor = None
    
    print(f"🔐 Starting authentication for: {email}")
    
    try:
        print("🔄 Step 1: Getting database connection...")
        conn = get_db_connection()
        if not conn:
            print("❌ Step 1 FAILED: Database connection is None")
            return None
        else:
            print("✅ Step 1 SUCCESS: Database connection established")
        
        print("🔄 Step 2: Creating cursor...")
        cursor = conn.cursor(dictionary=True)
        print("✅ Step 2 SUCCESS: Cursor created")
        
        # Check all user tables with DIFFERENT queries for each table
        print("🔄 Step 3: Searching for user in database...")
        
        user = None
        
        # Check admins table (has role column)
        print("   🔍 Checking admins table...")
        cursor.execute("""
            SELECT admin_id as user_id, first_name, last_name, password_hash, 
                   'admin' as user_type, role
            FROM admins 
            WHERE email = %s AND status = 'active'
        """, (email,))
        user = cursor.fetchone()
        
        if not user:
            # Check faculty table (has role column)
            print("   🔍 Checking faculty table...")
            cursor.execute("""
                SELECT faculty_id as user_id, first_name, last_name, password_hash, 
                       'faculty' as user_type, role
                FROM faculty 
                WHERE email = %s AND status = 'active'
            """, (email,))
            user = cursor.fetchone()
        
        if not user:
            # Check students table (NO role column - use default 'student')
            print("   🔍 Checking students table...")
            cursor.execute("""
                SELECT student_id as user_id, first_name, last_name, password_hash, 
                       'student' as user_type, 'student' as role
                FROM students 
                WHERE email = %s AND status = 'active'
            """, (email,))
            user = cursor.fetchone()
        
        if not user:
            print("❌ Step 3 FAILED: User not found in any table")
            return None
        
        print(f"✅ Step 3 SUCCESS: User found: {user['user_id']}")
        print(f"✅ User role: {user.get('role', 'NO ROLE!')}")
        
        # Verify password
        print("🔄 Step 4: Verifying password...")
        print(f"   🔑 Password hash: {user['password_hash'][:50]}...")
        
        if verify_password(password, user['password_hash']):
            print("✅ Step 4 SUCCESS: Password verified")
            print(f"🎉 AUTHENTICATION SUCCESSFUL for {user['user_id']}")
            print(f"🎉 FINAL USER OBJECT: {user}")
            return user
        else:
            print("❌ Step 4 FAILED: Password verification failed")
            print(f"   💡 Provided password: {password}")
            print(f"   💡 Stored hash: {user['password_hash']}")
            return None
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return None
    finally:
        # Always close connections
        if cursor:
            cursor.close()
            print("✅ Cursor closed")
        if conn:
            conn.close()
            print("✅ Connection closed")


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
    """Proper database connection function"""
    try:
        print("🔄 Establishing database connection...")
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='facesys',
            charset='utf8mb4',
            autocommit=True
        )
        print("✅ Database connection successful!")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

@contextmanager
def get_db_cursor(commit=True):
    """Context manager for safe database access."""
    conn = get_db_connection()
    if conn is None:
        raise Exception("Database connection unavailable.")
        
    cursor = conn.cursor(dictionary=True)
    try:
        yield cursor
        if commit:
            conn.commit()  # ✅ ADD THIS LINE - commit the transaction
    except Exception as e:
        conn.rollback()    # ✅ ADD THIS LINE - rollback on error
        raise e
    finally:
        cursor.close()
        conn.close()

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
# Load known faces from database
# =========================
known_face_encodings = np.array([]) # Initialize as a NumPy array (will be empty)
known_face_names = []
known_face_ids = []
known_face_types = []  # 'student' or 'faculty'
KNOWN_FACE_ENCODINGS_ARRAY = None # New variable to hold the final, stacked NumPy array

def finalize_known_faces():
    """Converts the list of encodings to a single NumPy array for fast vector comparison."""
    global known_face_encodings, KNOWN_FACE_ENCODINGS_ARRAY
    if known_face_encodings:
        # Stack the list of encoding arrays into a single 2D NumPy array (N, 512)
        KNOWN_FACE_ENCODINGS_ARRAY = np.vstack(known_face_encodings)
        logger.info(f"Finalized KNOWN_FACE_ENCODINGS_ARRAY shape: {KNOWN_FACE_ENCODINGS_ARRAY.shape}")
    else:
        KNOWN_FACE_ENCODINGS_ARRAY = np.array([])
        logger.info("No known faces loaded.")

def load_known_faces_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_types

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, first_name, last_name, face_encoding FROM students WHERE face_encoding IS NOT NULL")
        
        for (id, first_name, last_name, face_encoding) in cursor:
            try:
                if isinstance(face_encoding, str):
                    # Robust parsing for string-encoded vectors (like those stored in text fields)
                    encoding_str = face_encoding.strip('[]')
                    if ', ' in encoding_str:
                        encoding_list = encoding_str.split(', ')
                    elif ',' in encoding_str:
                        encoding_list = encoding_str.split(',')
                    else:
                        encoding_list = encoding_str.split()
                    encoding = np.array([float(x) for x in encoding_list], dtype=np.float32)
                else:
                    # Preferred method: direct loading from byte arrays (BLOB fields)
                    encoding = np.frombuffer(face_encoding, dtype=np.float32)
                
                if encoding.size == 512:
                    known_face_encodings.append(encoding)
                    full_name = f"{first_name} {last_name}"
                    known_face_names.append(full_name)
                    known_face_ids.append(id)
                    known_face_types.append('student')
                    logger.info(f"Loaded student {full_name} ({id})")
                else:
                    logger.warning(f"Invalid encoding size for student {id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for student {id}: {e}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load student faces from database: {e}")

def load_known_faculties_from_db():
    global known_face_encodings, known_face_names, known_face_ids, known_face_types
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT faculty_id, first_name, last_name, face_encoding FROM faculty WHERE face_encoding IS NOT NULL")
        
        for (id, first_name, last_name, face_encoding) in cursor:
            try:
                if isinstance(face_encoding, str):
                    # Robust parsing for string-encoded vectors
                    encoding_str = face_encoding.strip('[]')
                    if ', ' in encoding_str:
                        encoding_list = encoding_str.split(', ')
                    elif ',' in encoding_str:
                        encoding_list = encoding_str.split(',')
                    else:
                        encoding_list = encoding_str.split()
                    encoding = np.array([float(x) for x in encoding_list], dtype=np.float32)
                else:
                    # Preferred method: direct loading from byte arrays
                    encoding = np.frombuffer(face_encoding, dtype=np.float32)
                
                if encoding.size == 512:
                    known_face_encodings.append(encoding)
                    full_name = f"{first_name} {last_name}"
                    known_face_names.append(full_name)
                    known_face_ids.append(id)
                    known_face_types.append('faculty')
                    logger.info(f"Loaded faculty {full_name} ({id})")
                else:
                    logger.warning(f"Invalid encoding size for faculty {id}: {encoding.size}")
            except Exception as e:
                logger.error(f"Error parsing encoding for faculty {id}: {e}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load faculty faces from database: {e}")

# Initialize known faces
known_face_encodings, known_face_names, known_face_ids, known_face_types = [], [], [], []
load_known_faces_from_db()
load_known_faculties_from_db()
finalize_known_faces() # <- NEW: Call the finalizer after loading all faces


# =========================
# Laptop camera capture
# =========================
cap_lock = threading.Lock()
cap = None

def open_stream(rtsp_url=None):
    global cap, camera_available, use_dummy_feed, current_rtsp_url
    
    if not rtsp_url:
        logger.warning("No RTSP URL provided")
        camera_available = False
        use_dummy_feed = True
        return False
    
    try:
        with cap_lock:
            # Release existing capture
            if cap is not None:
                try:
                    cap.release()
                    logger.info("Released previous camera connection")
                except Exception as e:
                    logger.warning(f"Error releasing previous capture: {e}")
                cap = None
            
            # Small delay to ensure camera is released
            time.sleep(0.5)
                    
            # Try to connect to RTSP with timeout
            logger.info(f"Attempting to connect to RTSP: {rtsp_url}")
            
            # Use OpenCV with better RTSP settings
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Reduced from 3840 for better performance
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Reduced from 2160 for better performance
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Set timeout for connection
            start_time = time.time()
            while time.time() - start_time < 5:  # 5 second timeout
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        camera_available = True
                        use_dummy_feed = False
                        current_rtsp_url = rtsp_url
                        logger.info(f"✅ RTSP stream connected successfully: {rtsp_url}")
                        return True
                time.sleep(0.1)
            
            # If we get here, connection failed
            cap.release()
            cap = None
            raise Exception("RTSP connection timeout")
                
    except Exception as e:
        logger.warning(f"❌ RTSP connection failed: {e}")
        
        # Fallback to dummy feed
        camera_available = False
        use_dummy_feed = True
        if cap is not None:
            try:
                cap.release()
            except:
                pass
            cap = None
        
        return False

def grabber():
    global latest_frame, stop_flag, camera_available, use_dummy_feed, cap, current_rtsp_url
    empty_count = 0
    frame_count = 0
    
    while not stop_flag:
        if use_dummy_feed or cap is None:
            latest_frame = create_dummy_frame()
            time.sleep(0.1)
            continue
            
        with cap_lock:
            if cap is None:
                time.sleep(0.1)
                continue
                
            ok, f = cap.read()
            
        if not ok:
            empty_count += 1
            if empty_count > MAX_EMPTY_GRABS:
                logger.warning("Camera connection lost. Switching to dummy feed...")
                camera_available = False
                use_dummy_feed = True
                
                with cap_lock:
                    if cap is not None:
                        try:
                            cap.release()
                        except:
                            pass
                        cap = None
                
                empty_count = 0
            else:
                time.sleep(0.01)
            continue
            
        empty_count = 0
        latest_frame = f
        
        # Log every 30 frames to verify it's working
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"✅ Capturing frames: {frame_count} frames captured")
        
        time.sleep(GRAB_SLEEP)

grab_thread = threading.Thread(target=grabber, daemon=True)
grab_thread.start()

# =========================
# Tracking & attendance
# =========================
tracks = []
locked_tracks = {}  # {person_id: {'track': track_obj, 'body_tracker': bytetrack_id, ...}}
attendance = {}
tracking_history = {}

def periodic_attendance_save():
    """Periodically save attendance data to CSV"""
    while not stop_flag:
        try:
            if attendance:  # Only save if there's data
                with open(ATTENDANCE_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Name', 'DateTime', 'Status'])
                    for sid, data in attendance.items():
                        writer.writerow([sid, data['name'], data['time'], 'present'])
                logger.info(f"Periodic backup: Saved {len(attendance)} attendance records to CSV")
        except Exception as e:
            logger.error(f"Periodic attendance save failed: {e}")
        
        time.sleep(attendance_save_interval)

# Start the periodic save thread in your main function
attendance_save_thread = threading.Thread(target=periodic_attendance_save, daemon=True)
attendance_save_thread.start()

def mark_attendance(name, id, type, session_id=None):
    """Mark attendance for both students and faculty - FIXED: Handles missing status properly"""
    global session_start_time, current_session_id, session_threshold_seconds, session_total_duration_seconds
    global student_presence_tracker
    
    # ✅ Load threshold from database if not set
    if not session_threshold_seconds and current_session_id:
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT threshold_seconds_total 
                FROM attendance_sessions 
                WHERE session_id = %s
            """, (current_session_id,))
            session_data = cursor.fetchone()
            if session_data and session_data.get('threshold_seconds_total'):
                session_threshold_seconds = session_data['threshold_seconds_total']
                logger.info(f"🎯 Loaded threshold from database: {session_threshold_seconds} seconds")
            else:
                logger.warning(f"⚠️ No threshold found for session {current_session_id}, using default 900s")
                session_threshold_seconds = 900
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"⚠️ Could not load threshold from database: {e}")
            session_threshold_seconds = 900
    
    if type not in ['student', 'faculty']:
        return
    
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Use current session ID if not provided
    if not session_id:
        session_id = current_session_id
    
    # 🎯 Update presence tracker
    if type == 'student' and session_id:
        student_presence_tracker[id] = {
            'last_seen': current_time,
            'last_body_seen': current_time,
            'name': name,
            'present': True
        }
    
    # ✅ Get section_id AND SUBJECT INFO for this session
    section_id = None
    subject_code = None
    subject_name = None
    room = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT section_id, subject_code, subject_name, room 
            FROM attendance_sessions 
            WHERE session_id = %s
        """, (session_id,))
        session_result = cursor.fetchone()
        
        if session_result:
            section_id = session_result.get('section_id')
            subject_code = session_result.get('subject_code')
            subject_name = session_result.get('subject_name')
            room = session_result.get('room')
            logger.info(f"🔗 Found session info - Section: {section_id}, Subject: {subject_code}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
    
    # 🎯 CRITICAL FIX: Check existing status and missing periods
    original_status = None
    existing_record_id = None
    missing_duration = 0
    is_returning_from_missing = False
    restored_original_status = None
    is_currently_missing = False
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 🎯 Get the MOST RECENT attendance record for this student in this session
        cursor.execute("""
            SELECT id, status, timestamp FROM attendance 
            WHERE student_id = %s AND session_id = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (id, session_id))
        existing_record = cursor.fetchone()
        
        if existing_record:
            existing_record_id = existing_record['id']
            original_status = existing_record['status']
            
            # 🎯 CHECK IF STUDENT IS CURRENTLY MISSING
            if original_status == 'missing':
                is_currently_missing = True
                logger.info(f"🎯 Student {name} is CURRENTLY MISSING")
        
        # 🎯 FIXED: Check if student is currently in missing_periods with original_status
        cursor.execute("""
            SELECT id, missing_start, original_status, TIMESTAMPDIFF(SECOND, missing_start, NOW()) as missing_seconds
            FROM missing_periods 
            WHERE student_id = %s AND session_id = %s AND returned = FALSE
            LIMIT 1
        """, (id, session_id))
        
        active_missing_period = cursor.fetchone()
        
        if active_missing_period:
            is_returning_from_missing = True
            missing_duration = active_missing_period['missing_seconds'] or 0
            restored_original_status = active_missing_period['original_status']
            
            logger.info(f"🔄 Student {name} is RETURNING from missing - was missing for {missing_duration} seconds, original status: {restored_original_status}")
            
            # 🎯 MARK THE MISSING PERIOD AS RETURNED
            cursor.execute("""
                UPDATE missing_periods 
                SET missing_end = NOW(), duration_seconds = %s, returned = TRUE
                WHERE id = %s
            """, (missing_duration, active_missing_period['id']))
            
            logger.info(f"✅ Marked missing period as returned for {name}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error checking existing status: {e}")
    
    # 🎯 FIXED STATUS DETERMINATION: Handle missing status properly
    if is_currently_missing:
        # 🎯 CRITICAL: If student is currently missing, we need to restore their original status
        if is_returning_from_missing and restored_original_status in ['present', 'late']:
            # 🎯 RESTORE ORIGINAL STATUS from missing_periods table
            status = restored_original_status
            logger.info(f"🔄 RESTORING ORIGINAL STATUS (from missing): {name} -> {status}")
        elif original_status in ['present', 'late']:
            # Fallback: Use original_status from attendance table
            status = original_status
            logger.info(f"🔄 RESTORING ORIGINAL STATUS (fallback from missing): {name} -> {status}")
        else:
            # If no original status found, calculate new status
            status = 'present'
            if session_start_time:
                time_difference = current_time - session_start_time
                time_diff_seconds = time_difference.total_seconds()
                
                threshold_seconds = session_threshold_seconds
                
                if time_diff_seconds > threshold_seconds:
                    status = 'late'
                    logger.info(f"⏰ LATE (after missing): {name} arrived {time_diff_seconds:.1f}s after start")
                else:
                    logger.info(f"✅ PRESENT (after missing): {name} arrived on time")
    
    elif is_returning_from_missing and restored_original_status in ['present', 'late']:
        # 🎯 RESTORE ORIGINAL STATUS when returning from missing
        status = restored_original_status
        logger.info(f"🔄 RESTORING ORIGINAL STATUS: {name} returning from missing -> {status}")
        
    elif is_returning_from_missing and original_status in ['present', 'late']:
        # Fallback: Use original_status from attendance table
        status = original_status
        logger.info(f"🔄 RESTORING ORIGINAL STATUS (fallback): {name} returning from missing -> {status}")
        
    else:
        # New detection or no previous status - calculate based on arrival time
        status = 'present'
        if session_start_time:
            time_difference = current_time - session_start_time
            time_diff_seconds = time_difference.total_seconds()
            
            threshold_seconds = session_threshold_seconds
            
            if time_diff_seconds > threshold_seconds:
                status = 'late'
                logger.info(f"⏰ LATE: {name} arrived {time_diff_seconds:.1f}s after start")
            else:
                logger.info(f"✅ PRESENT: {name} arrived on time")
    
    # Save to memory
    attendance[id] = {
        "name": name, 
        "time": time_str, 
        "type": type,
        "status": status
    }
    
    # ✅ Update student status for frontend
    if type == 'student':
        student_status[id] = status
        logger.info(f"🎯 STUDENT STATUS: {name} -> {status}")
    
    # Save to CSV
    try:
        csv_file = "attendance_log.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ID', 'Name', 'DateTime', 'Type', 'Status', 'SessionID', 'SectionID', 'SubjectCode', 'SubjectName', 'Room', 'MissingDuration', 'IsReturning', 'RestoredStatus', 'WasMissing'])
            writer.writerow([id, name, time_str, type, status, session_id or 'N/A', section_id or 'N/A', subject_code or 'N/A', subject_name or 'N/A', room or 'N/A', missing_duration, is_returning_from_missing, restored_original_status or 'N/A', is_currently_missing])
        
        logger.info(f"📄 CSV saved: {name} ({id}) - {type} - {status} - Missing: {missing_duration}s - Returning: {is_returning_from_missing} - Restored: {restored_original_status} - WasMissing: {is_currently_missing}")
    except Exception as e:
        logger.error(f"Failed to save attendance to CSV: {e}")
    
    # 🎯 FIXED: Update database with proper status handling
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if type == 'student':
            # Check if record already exists first
            cursor.execute("""
                SELECT id, status, timestamp FROM attendance 
                WHERE student_id = %s AND session_id = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (id, session_id))
            
            existing_record = cursor.fetchone()
            
            if existing_record:
                # Record exists - UPDATE it
                existing_id = existing_record[0]
                existing_db_status = existing_record[1]
                existing_timestamp = existing_record[2]
                
                # Convert string timestamp to datetime if needed
                if isinstance(existing_timestamp, str):
                    existing_timestamp = datetime.strptime(existing_timestamp, "%Y-%m-%d %H:%M:%S")
                
                time_since_update = (current_time - existing_timestamp).total_seconds()
                
                # 🎯 CRITICAL: Only update if status changed OR student is returning from missing OR was missing
                if existing_db_status != status or is_returning_from_missing or is_currently_missing:
                    # 🎯 FIXED: Update with missing_duration and proper status
                    cursor.execute("""
                        UPDATE attendance 
                        SET status = %s, timestamp = %s, name = %s, subject_code = %s, subject_name = %s, room = %s, missing_duration = %s
                        WHERE id = %s
                    """, (status, time_str, name, subject_code, subject_name, room, missing_duration, existing_id))
                    logger.info(f"🔄 Updated attendance: {name} - {status} (was {existing_db_status}) - Missing: {missing_duration}s - Restored: {restored_original_status} - WasMissing: {is_currently_missing}")
                else:
                    logger.info(f"⏭️ Skipping update for {name} - status unchanged: {status}")
            else:
                # No record exists - INSERT new one
                if section_id:
                    cursor.execute("""
                        INSERT INTO attendance 
                        (student_id, name, timestamp, person_type, status, session_id, section_id, subject_code, subject_name, room, missing_duration)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (id, name, time_str, 'student', status, session_id, section_id, subject_code, subject_name, room, missing_duration))
                else:
                    cursor.execute("""
                        INSERT INTO attendance 
                        (student_id, name, timestamp, person_type, status, session_id, subject_code, subject_name, room, missing_duration)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (id, name, time_str, 'student', status, session_id, subject_code, subject_name, room, missing_duration))
                logger.info(f"📝 Created NEW attendance record: {name} - {status}")
        else:  # faculty
            # Faculty logic remains the same
            cursor.execute("""
                UPDATE attendance 
                SET status = %s, timestamp = %s, name = %s, subject_code = %s, subject_name = %s, room = %s
                WHERE faculty_id = %s AND session_id = %s
            """, (status, time_str, name, subject_code, subject_name, room, id, session_id))
            
            if cursor.rowcount == 0:
                if section_id:
                    cursor.execute("""
                        INSERT INTO attendance 
                        (faculty_id, name, timestamp, person_type, status, session_id, section_id, subject_code, subject_name, room)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (id, name, time_str, 'faculty', status, session_id, section_id, subject_code, subject_name, room))
                else:
                    cursor.execute("""
                        INSERT INTO attendance 
                        (faculty_id, name, timestamp, person_type, status, session_id, subject_code, subject_name, room)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (id, name, time_str, 'faculty', status, session_id, subject_code, subject_name, room))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"💾 Database UPDATED: {name} ({id}) - {status} - Missing Duration: {missing_duration}s - Returning: {is_returning_from_missing} - Restored Status: {restored_original_status} - Was Missing: {is_currently_missing}")
        
    except Exception as e:
        logger.error(f"Failed to update attendance in database: {e}")

# Also update the CSV saving function
def save_attendance_to_csv(person_id, name, timestamp, person_type):
    """Save attendance entry to CSV in real-time"""
    try:
        csv_file = "attendance_log.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ID', 'Name', 'DateTime', 'Type', 'Status'])
            writer.writerow([person_id, name, timestamp, person_type, 'present'])
        
        logger.info(f"Attendance saved to CSV: {name} ({person_id}) - {person_type}")
    except Exception as e:
        logger.error(f"Failed to save attendance to CSV: {e}")

# =========================
# Key Functions
# =========================

def detect_bodies(frame):
    """Detect people bodies using YOLOv8 - OPTIMIZED WITH FP16"""
    if body_detector is None:
        logger.warning("Body detector not initialized")
        return []
    
    try:
        # 🎯 ADD FP16 SUPPORT - Run YOLO detection for person class only
        if DEVICE == "cuda":
            results = body_detector(frame, classes=[0], verbose=False, conf=0.3, half=True)  # 🆕 half=True
        else:
            results = body_detector(frame, classes=[0], verbose=False, conf=0.3)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        logger.debug(f"Body detector found {len(detections)} people")
        return detections
    except Exception as e:
        logger.error(f"Body detection error: {e}")
        return []

def convert_to_supervision_format(detections):
    """Convert detections to supervision Detections format"""
    from supervision import Detections
    
    if not detections:
        return Detections.empty()
    
    xyxy = np.array([d['box'] for d in detections])
    confidence = np.array([d['confidence'] for d in detections])
    class_id = np.zeros(len(detections), dtype=int)
    
    return Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )


def match_face_to_body(face_box, body_detections, iou_threshold=0.1):
    """
    Match a face detection to a body detection with improved accuracy.
    Face should be in the upper portion of the body bounding box.
    """
    best_match_idx = None
    best_score = 0
    
    fx1, fy1, fx2, fy2 = face_box
    face_center_x = (fx1 + fx2) / 2
    face_center_y = (fy1 + fy2) / 2
    face_width = fx2 - fx1
    
    for idx, body_det in enumerate(body_detections):
        bx1, by1, bx2, by2 = body_det['box']
        
        # Check if face center is inside body box (with some tolerance)
        tolerance_x = (bx2 - bx1) * 0.1  # 10% horizontal tolerance
        tolerance_y = (by2 - by1) * 0.1  # 10% vertical tolerance
        
        if not ((bx1 - tolerance_x) <= face_center_x <= (bx2 + tolerance_x) and 
                (by1 - tolerance_y) <= face_center_y <= (by2 + tolerance_y)):
            continue
        
        body_width = bx2 - bx1
        body_height = by2 - by1
        
        # Face should be in upper portion of body
        face_relative_y = face_center_y - by1
        
        if face_relative_y > body_height * FACE_TO_BODY_VERTICAL_RATIO:
            continue  # Face too low in body
        
        # Check if face size is proportional to body size
        expected_face_width = body_width * 0.3  # Face should be ~30% of body width
        width_ratio = min(face_width, expected_face_width) / max(face_width, expected_face_width)
        
        if width_ratio < 0.4:  # Face size too different from expected
            continue
        
        # Calculate position score (higher = better)
        position_score = 1.0 - (face_relative_y / (body_height * FACE_TO_BODY_VERTICAL_RATIO))
        
        # Calculate IoU
        overlap_iou = iou((fx1, fy1, fx2, fy2), (bx1, by1, bx2, by2))
        
        # Size matching score
        size_score = width_ratio
        
        # Combined score (weighted average)
        total_score = (position_score * 0.4) + (overlap_iou * 0.4) + (size_score * 0.2)
        
        if total_score > best_score and overlap_iou > iou_threshold:
            best_score = total_score
            best_match_idx = idx
    
    return best_match_idx, best_score

def calculate_real_fps():
    """Calculate real FPS based on recent frames"""
    global frame_timestamps, current_fps
    now = time.time()
    frame_timestamps.append(now)
    
    # Keep only last 30 frames for calculation
    if len(frame_timestamps) > 30:
        frame_timestamps.pop(0)
    
    # Calculate FPS based on actual timing
    if len(frame_timestamps) >= 2:
        time_span = frame_timestamps[-1] - frame_timestamps[0]
        if time_span > 0:
            current_fps = len(frame_timestamps) / time_span
        else:
            current_fps = 30.0  # Fallback
    
    return current_fps

def update_trackers_with_body(rgb, frame, frame_idx):
    """
    Update trackers with body-only tracking for LOCKED tracks.
    ✅ FIXED: HTTPS API calls with proper error handling
    """
    global tracks, locked_tracks, pending_confirmations, current_fps
    global detectionStopped, student_presence_tracker, current_session_id
    
    # Stop detection if session ended
    if detectionStopped:
        return
    
    # Calculate real FPS
    real_fps = calculate_real_fps()
    if real_fps < 5:
        real_fps = 5

    h, w = frame.shape[:2]
    
    # Step 1: Detect bodies FIRST - this is critical
    body_detections = detect_bodies(frame)
    logger.debug(f"Detected {len(body_detections)} bodies in frame {frame_idx}")
    
    # 🎯 CRITICAL FIX: Check for missing bodies IMMEDIATELY every frame
    current_time = datetime.now()
    
    # Track which locked tracks have matching bodies in CURRENT frame
    locked_tracks_with_bodies = set()
    
    # Step 2: Extract ReID features for ALL bodies first
    body_reid_features = []
    body_boxes_clean = []
    
    for body_det in body_detections:
        body_box = tuple(body_det['box'])
        bx1, by1, bx2, by2 = body_box
        if (bx2 - bx1) < 50 or (by2 - by1) < 100:
            continue
            
        reid_feature = extract_reid_features(frame, body_box)
        body_reid_features.append(reid_feature)
        body_boxes_clean.append(body_det)
    
    body_detections = body_boxes_clean
    matched_body_indices = set()
    
    # 🎯 CRITICAL FIX: IMMEDIATE BODY MATCHING - Check every frame
    to_remove_locks = []
    
    for person_id, lock_info in list(locked_tracks.items()):
        frames_since_seen = frame_idx - lock_info.get('last_seen', frame_idx)
        
        # 🎯 IMMEDIATE CHECK: If no body found in current frame, start counting immediately
        body_found_in_current_frame = False
        last_body_box = lock_info.get('body_box')
        
        if last_body_box and body_detections:
            # Try to match with current frame bodies
            for idx, body_det in enumerate(body_detections):
                if idx in matched_body_indices:
                    continue
                    
                body_box = body_det['box']
                
                # Quick spatial matching
                overlap_iou = iou(tuple(body_box), last_body_box)
                if overlap_iou > 0.1:  # Low threshold for quick matching
                    body_found_in_current_frame = True
                    matched_body_indices.add(idx)
                    
                    # Update track immediately
                    lock_info['body_box'] = tuple(body_box)
                    lock_info['last_seen'] = frame_idx
                    lock_info['missed_detections'] = 0
                    
                    # Update presence tracker
                    if lock_info.get('type') == 'student' and lock_info.get('id') in student_presence_tracker:
                        student_presence_tracker[lock_info['id']]['last_body_seen'] = current_time
                        student_presence_tracker[lock_info['id']]['last_seen'] = current_time
                        student_presence_tracker[lock_info['id']]['present'] = True
                    
                    locked_tracks_with_bodies.add(person_id)
                    break
        
        # 🎯 CRITICAL: If no body found in current frame, mark for immediate removal
        if not body_found_in_current_frame:
            lock_info['missed_detections'] = lock_info.get('missed_detections', 0) + 1
            
            # 🎯 IMMEDIATE REMOVAL: Remove after just 1-2 seconds (not 10 seconds)
            if lock_info['missed_detections'] > 30:  # 1 second at 30fps
                to_remove_locks.append(person_id)
                logger.info(f"❌ IMMEDIATE REMOVAL: {lock_info.get('name', person_id)} - no body for 1 second")
    
    # 🎯 CRITICAL FIX: Remove tracks immediately when no body found
    for person_id in to_remove_locks:
        lock_info = locked_tracks.get(person_id)
        if lock_info and lock_info.get('type') == 'student':
            student_id = lock_info.get('id')
            student_name = lock_info.get('name', person_id)
            
            # 🎯 FIXED HTTPS API CALL with proper data format
            try:
                import requests
                import time
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                # 🎯 PROPER DATA FORMAT
                api_data = {
                    'student_id': student_id,
                    'session_id': current_session_id
                }
                
                # Try 3 times with better connection handling
                success = False
                for attempt in range(3):
                    try:
                        response = requests.post(
                            'https://192.168.0.100:5000/api/student_left',
                            json=api_data,  # 🎯 FIXED: Use proper JSON data
                            timeout=5,
                            headers={
                                'Connection': 'close',
                                'Content-Type': 'application/json'
                            },
                            verify=False  # 🎯 CRITICAL: Disable SSL verification for self-signed cert
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"🎯 SUCCESS: student_left API called for {student_name}")
                            success = True
                            break
                        elif response.status_code == 400:
                            # Parse the actual error message
                            try:
                                error_data = response.json()
                                logger.warning(f"⚠️ API returned {response.status_code}: {error_data.get('message', 'Unknown error')}")
                            except:
                                logger.warning(f"⚠️ API returned {response.status_code}: {response.text}")
                            break
                        else:
                            logger.warning(f"⚠️ API returned {response.status_code}: {response.text}")
                            if attempt < 2:
                                time.sleep(0.3)
                                
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
                        logger.error(f"❌ Connection error (attempt {attempt + 1}/3): {conn_err}")
                        if attempt < 2:
                            time.sleep(0.5)
                    except Exception as req_err:
                        logger.error(f"❌ Request error (attempt {attempt + 1}/3): {req_err}")
                        if attempt < 2:
                            time.sleep(0.5)
                
                if not success:
                    logger.error(f"❌ FAILED after 3 attempts: Could not mark {student_name} as missing")
                    
            except Exception as e:
                logger.error(f"❌ Fatal error calling student_left API: {e}")
            
            # Update presence tracker if exists
            if student_id in student_presence_tracker:
                student_presence_tracker[student_id]['present'] = False
                student_presence_tracker[student_id]['last_seen'] = current_time
        
        locked_tracks.pop(person_id, None)
        locked_track_reid_features.pop(person_id, None)
        logger.info(f"🔓 IMMEDIATE UNLOCK: {person_id}")
    
    # 🎯 SIMPLE MISSING DETECTION - Backup check every 30 frames
    if frame_idx % 30 == 0 and current_session_id:
        for student_id, track_info in list(student_presence_tracker.items()):
            if track_info.get('present'):
                has_active_body = student_id in locked_tracks_with_bodies
                
                if not has_active_body:
                    last_body_seen = track_info.get('last_body_seen')
                    if last_body_seen:
                        time_since_body_seen = (current_time - last_body_seen).total_seconds()
                        
                        if time_since_body_seen > 5:  # Reduced to 5 seconds for backup
                            track_info['present'] = False
                            track_info['last_seen'] = current_time
                            
                            try:
                                import requests
                                import urllib3
                                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                                
                                api_data = {
                                    'student_id': student_id,
                                    'session_id': current_session_id
                                }
                                response = requests.post(
                                    'https://192.168.0.100:5000/api/student_left', 
                                    json=api_data,
                                    timeout=2,
                                    verify=False  # 🎯 ADDED: Disable SSL verification
                                )
                                if response.status_code == 200:
                                    logger.info(f"📤 BACKUP MISSING: {track_info['name']}")
                            except Exception as e:
                                logger.warning(f"⚠️ API call failed: {e}")
        
        # Cleanup old entries
        for student_id in list(student_presence_tracker.keys()):
            track_info = student_presence_tracker[student_id]
            if (not track_info.get('present') and 
                track_info.get('last_seen') and 
                (current_time - track_info['last_seen']).total_seconds() > 3600):
                del student_presence_tracker[student_id]
    
    # Step 3: Update LOCKED tracks with remaining detection logic
    to_remove_locks = []
    
    for person_id, lock_info in list(locked_tracks.items()):
        frames_since_seen = frame_idx - lock_info.get('last_seen', frame_idx)

        # 🎯 SIMPLE: 10 second timeout
        if frames_since_seen > 300:
            to_remove_locks.append(person_id)
            logger.info(f"❌ Person {lock_info.get('name', person_id)} disappeared - releasing lock")
            continue
        
        last_body_box = lock_info.get('body_box')
        last_reid_feature = locked_track_reid_features.get(person_id)
        matched = False
        
        # Try ReID matching
        if last_reid_feature is not None and body_detections:
            best_reid_match_idx = None
            best_reid_distance = 0.35
            
            for idx, body_det in enumerate(body_detections):
                if idx in matched_body_indices:
                    continue
                
                current_reid_feature = body_reid_features[idx]
                if current_reid_feature is None:
                    continue
                
                reid_dist = calculate_reid_distance(last_reid_feature, current_reid_feature)
                
                body_box = body_det['box']
                if last_body_box:
                    movement = calculate_box_distance(last_body_box, body_box)
                    if movement > 200:
                        continue
                
                if reid_dist < best_reid_distance:
                    best_reid_distance = reid_dist
                    best_reid_match_idx = idx
            
            if best_reid_match_idx is not None:
                matched_body_indices.add(best_reid_match_idx)
                new_body_box = tuple(body_detections[best_reid_match_idx]['box'])
                new_reid_feature = body_reid_features[best_reid_match_idx]
                
                lock_info['body_box'] = new_body_box
                lock_info['last_seen'] = frame_idx
                lock_info['missed_detections'] = 0
                locked_track_reid_features[person_id] = new_reid_feature
                
                # 🎯 SIMPLE: Update presence tracker when body detected
                if lock_info.get('type') == 'student' and lock_info.get('id') in student_presence_tracker:
                    student_presence_tracker[lock_info['id']]['last_body_seen'] = datetime.now()
                    student_presence_tracker[lock_info['id']]['last_seen'] = datetime.now()
                    student_presence_tracker[lock_info['id']]['present'] = True
                
                matched = True
        
        # Fall back to spatial matching
        if not matched and last_body_box and body_detections:
            best_match_idx = None
            best_iou = 0.2
            
            for idx, body_det in enumerate(body_detections):
                if idx in matched_body_indices:
                    continue
                
                body_box = body_det['box']
                overlap_iou = iou(tuple(body_box), last_body_box)
                
                if overlap_iou > best_iou:
                    best_iou = overlap_iou
                    best_match_idx = idx
            
            if best_match_idx is not None:
                matched_body_indices.add(best_match_idx)
                new_body_box = tuple(body_detections[best_match_idx]['box'])
                new_reid_feature = body_reid_features[best_match_idx]
                
                lock_info['body_box'] = new_body_box
                lock_info['last_seen'] = frame_idx
                lock_info['missed_detections'] = 0
                
                # 🎯 SIMPLE: Update presence tracker when body detected
                if lock_info.get('type') == 'student' and lock_info.get('id') in student_presence_tracker:
                    student_presence_tracker[lock_info['id']]['last_body_seen'] = datetime.now()
                    student_presence_tracker[lock_info['id']]['last_seen'] = datetime.now()
                    student_presence_tracker[lock_info['id']]['present'] = True
                
                if new_reid_feature is not None:
                    locked_track_reid_features[person_id] = new_reid_feature
                matched = True
        
        if not matched:
            lock_info['missed_detections'] = lock_info.get('missed_detections', 0) + 1
            # 🎯 SIMPLE: 5 second tolerance
            if lock_info['missed_detections'] > 150:
                to_remove_locks.append(person_id)
    
    # Remove expired locks
    for person_id in to_remove_locks:
        lock_info = locked_tracks.get(person_id)
        if lock_info and lock_info.get('type') == 'student':
            student_id = lock_info.get('id')
            student_name = lock_info.get('name', person_id)
            
            if student_id in student_presence_tracker:
                student_presence_tracker[student_id]['present'] = False
                student_presence_tracker[student_id]['last_seen'] = datetime.now()
            
            # 🎯 SIMPLE: Call API to mark as missing WITH RETRY LOGIC
            try:
                import requests
                import time
                
                # Try 3 times with better connection handling
                success = False
                for attempt in range(3):
                    try:
                        response = requests.post(
                            'https://192.168.0.100:5000/api/student_left',
                            json={'student_id': student_id, 'session_id': current_session_id},
                            timeout=5,
                            headers={'Connection': 'close'}
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"📤 Student marked as MISSING: {student_name}")
                            success = True
                            break
                        elif response.status_code == 400:
                            logger.warning(f"⚠️ API returned {response.status_code}: {response.text}")
                            break
                        else:
                            logger.warning(f"⚠️ API returned {response.status_code}: {response.text}")
                            if attempt < 2:
                                time.sleep(0.3)
                                
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
                        logger.error(f"❌ Connection error (attempt {attempt + 1}/3): {conn_err}")
                        if attempt < 2:
                            time.sleep(0.5)
                    except Exception as req_err:
                        logger.error(f"❌ Request error (attempt {attempt + 1}/3): {req_err}")
                        if attempt < 2:
                            time.sleep(0.5)
                
                if not success:
                    logger.error(f"❌ FAILED after 3 attempts: Could not mark {student_name} as missing")
                    
            except Exception as e:
                logger.error(f"❌ Fatal error calling student_left API: {e}")
        
        locked_tracks.pop(person_id, None)
        locked_track_reid_features.pop(person_id, None)
        logger.info(f"🔓 Unlocked track for {person_id}")
    
    # Maintain tracks list
    current_tracks = []
    
    # Add locked tracks
    for person_id, lock_info in locked_tracks.items():
        lock_start = lock_info.get('lock_start', frame_idx)
        frames_tracked = frame_idx - lock_start
        duration_seconds = frames_tracked / real_fps if real_fps > 0 else frames_tracked / 30
        
        locked_track = {
            'id': person_id,
            'name': lock_info.get('name', f'Person {person_id}'),
            'type': lock_info.get('type', 'student'),
            'is_locked': True,
            'body_box': lock_info.get('body_box'),
            'last_seen': lock_info.get('last_seen', frame_idx),
            'confidence': 1.0,
            'tracking_duration': int(duration_seconds),
            'lock_start': lock_start,
            'real_fps': real_fps
        }
        current_tracks.append(locked_track)
    
    # Update UNLOCKED tracks
    for tr in list(tracks):
        if tr.get('id') in locked_tracks:
            continue
        
        if tr.get('is_locked'):
            continue
            
        tracker_ok = False
        try:
            tracker = tr.get('tracker')
            if tracker:
                update_success = tracker.update(rgb)
                
                if update_success:
                    pos = tracker.get_position()
                    x1, y1 = int(pos.left()), int(pos.top())
                    x2, y2 = int(pos.right()), int(pos.bottom())
                    
                    if (x2 > x1 and y2 > y1 and 
                        0 <= x1 < w and 0 <= y1 < h and 
                        x2 <= w and y2 <= h):
                        tr['box'] = (x1, y1, x2, y2)
                        tr['last_seen'] = frame_idx
                        tracker_ok = True
                        
                        tr['confidence'] = max(0.3, tr.get('confidence', 0.5) * 0.99)
        except Exception as e:
            logger.debug(f"Unlocked tracker update failed: {e}")
        
        if tracker_ok:
            tr['consecutive_failures'] = 0
            start_frame = tr.get('start_frame', frame_idx)
            frames_tracked = frame_idx - start_frame
            duration_seconds = frames_tracked / real_fps if real_fps > 0 else frames_tracked / 30
            tr['tracking_duration'] = int(duration_seconds)
            tr['start_frame'] = start_frame
            tr['real_fps'] = real_fps
            
            current_tracks.append(tr)
        else:
            tr['consecutive_failures'] = tr.get('consecutive_failures', 0) + 1
            if tr['consecutive_failures'] < 5:
                start_frame = tr.get('start_frame', frame_idx)
                frames_tracked = frame_idx - start_frame
                duration_seconds = frames_tracked // 30
                tr['tracking_duration'] = duration_seconds
                current_tracks.append(tr)
    
    # Update global tracks
    tracks[:] = current_tracks
    
    # Cleanup old unknown tracks
    current_tracks = [
        tr for tr in tracks
        if not (tr.get('name') == "Unknown" and frame_idx - tr.get('last_seen', 0) > 30)
    ]
    tracks[:] = current_tracks
    
    # 🎯 ENHANCED DRAWING: BOLDER BOXES and LARGER TEXT
    
    # Step 5: Draw PENDING CONFIRMATIONS (Orange BODY boxes)
    active_pending = {pid: data for pid, data in pending_confirmations.items() 
                      if frame_idx - data.get('last_seen', data.get('first_seen', 0)) <= 5}
    
    for person_id, conf_data in active_pending.items():
        if conf_data.get('body_boxes'):
            body_box = conf_data['body_boxes'][-1]
            bx1, by1, bx2, by2 = body_box
            
            # 🎯 BOLDER Orange box for confirmation phase (thicker border)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 5)  # Increased thickness from 4 to 5
            
            display_name = conf_data.get('name', f'Person {person_id}')
            progress = len(conf_data['frames'])
            
            # 🎯 LARGER TEXT for name and progress
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_name = 1.1  # Increased from 0.9
            font_scale_status = 0.9  # Increased from 0.7
            thickness = 3  # Increased from 2
            
            # Name with larger text
            (name_w, name_h), _ = cv2.getTextSize(display_name, font, font_scale_name, thickness)
            name_y = max(25, by1 - 15)  # Increased margin
            cv2.rectangle(frame, (bx1, name_y - name_h - 10), (bx1 + name_w + 20, name_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, display_name, (bx1 + 10, name_y), font, font_scale_name, (255, 255, 255), thickness)
            
            # Progress with larger text
            status_label = f"CONFIRMING {progress}/{CONFIRMATION_FRAMES_REQUIRED}"
            (status_w, status_h), _ = cv2.getTextSize(status_label, font, font_scale_status, thickness)
            status_y = by2 + status_h + 20  # Increased spacing
            cv2.rectangle(frame, (bx1, status_y - status_h - 8), (bx1 + status_w + 15, status_y + 8), (0, 140, 255), -1)
            cv2.putText(frame, status_label, (bx1 + 8, status_y), font, font_scale_status, (255, 255, 255), thickness)
    
    # Step 6: Draw LOCKED tracks (Green BODY boxes only)
    for person_id, lock_info in locked_tracks.items():
        body_box = lock_info.get('body_box')
        if not body_box:
            continue
        
        bx1, by1, bx2, by2 = body_box
        
        # 🎯 BOLDER Green box for locked body tracking (thicker border)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 5)  # Increased thickness from 4 to 5
        
        display_name = lock_info.get('name', f'Person {person_id}')
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate tracking time
        lock_start = lock_info.get('lock_start', frame_idx)
        tracking_seconds = (frame_idx - lock_start) // 30
        time_label = f"Time: {tracking_seconds}s"
        
        # 🎯 LARGER TEXT settings
        font_scale_name = 1.1  # Increased from 0.9
        font_scale_info = 0.9  # Increased from 0.7
        thickness = 3  # Increased from 2
        
        # Draw name at top with larger text
        (name_w, name_h), _ = cv2.getTextSize(display_name, font, font_scale_name, thickness)
        name_y = max(25, by1 - 15)  # Increased margin
        cv2.rectangle(frame, (bx1, name_y - name_h - 10), (bx1 + name_w + 20, name_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, display_name, (bx1 + 10, name_y), font, font_scale_name, (255, 255, 255), thickness)
        
        # Draw tracking time at bottom with larger text
        (time_w, time_h), _ = cv2.getTextSize(time_label, font, font_scale_info, thickness)
        time_y = by2 + time_h + 20  # Increased spacing
        cv2.rectangle(frame, (bx1, time_y - time_h - 8), (bx1 + time_w + 15, time_y + 8), (0, 200, 0), -1)
        cv2.putText(frame, time_label, (bx1 + 8, time_y), font, font_scale_info, (255, 255, 255), thickness)
        
        # Draw status with larger text
        status_label = "LOCKED"
        (status_w, status_h), _ = cv2.getTextSize(status_label, font, font_scale_info, thickness)
        status_y = time_y + status_h + 15  # Increased spacing
        cv2.rectangle(frame, (bx1, status_y - status_h - 8), (bx1 + status_w + 15, status_y + 8), (0, 150, 0), -1)
        cv2.putText(frame, status_label, (bx1 + 8, status_y), font, font_scale_info, (255, 255, 255), thickness)
    
    # Step 7: Draw UNLOCKED tracks (Yellow FACE boxes)
    for tr in tracks:
        if tr.get('is_locked') or tr.get('id') in locked_tracks:
            continue
        
        face_box = tr.get('box')
        if not face_box or face_box == (0, 0, 0, 0):
            continue
            
        x1, y1, x2, y2 = face_box
        
        # 🎯 BOLDER Yellow box for face scanning (thicker border)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Increased thickness from 3 to 4
        
        display_name = tr.get('name', 'Unknown')
        confidence = tr.get('confidence', 0.0)
        duration = tr.get('tracking_duration', 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 🎯 LARGER TEXT settings
        font_scale_name = 1.0  # Increased from 0.8
        font_scale_info = 0.8  # Increased from 0.6
        thickness = 3  # Increased from 2
        
        # Draw name with larger text
        (name_w, name_h), _ = cv2.getTextSize(display_name, font, font_scale_name, thickness)
        name_y = max(20, y1 - 15)  # Increased margin
        cv2.rectangle(frame, (x1, name_y - name_h - 8), (x1 + name_w + 15, name_y + 8), (0, 0, 0), -1)
        cv2.putText(frame, display_name, (x1 + 8, name_y), font, font_scale_name, (255, 255, 255), thickness)
        
        # Draw tracking time with larger text
        time_label = f"Time: {duration}s"
        (time_w, time_h), _ = cv2.getTextSize(time_label, font, font_scale_info, thickness)
        time_y = y2 + time_h + 15  # Increased spacing
        cv2.rectangle(frame, (x1, time_y - time_h - 8), (x1 + time_w + 12, time_y + 8), (0, 180, 180), -1)
        cv2.putText(frame, time_label, (x1 + 6, time_y), font, font_scale_info, (255, 255, 255), thickness)
        
        # Draw confidence for unknown faces with larger text
        if display_name == "Unknown":
            conf_label = f"Conf: {confidence:.2f}"
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale_info, thickness)
            conf_y = time_y + conf_h + 12  # Increased spacing
            cv2.rectangle(frame, (x1, conf_y - conf_h - 8), (x1 + conf_w + 12, conf_y + 8), (0, 0, 0), -1)
            cv2.putText(frame, conf_label, (x1 + 6, conf_y), font, font_scale_info, (255, 255, 255), thickness)
        
        # Draw status with larger text
        status_label = "SCANNING"
        (status_w, status_h), _ = cv2.getTextSize(status_label, font, font_scale_info, thickness)
        status_y = (conf_y if display_name == "Unknown" else time_y) + status_h + 12  # Increased spacing
        cv2.rectangle(frame, (x1, status_y - status_h - 8), (x1 + status_w + 12, status_y + 8), (0, 150, 150), -1)
        cv2.putText(frame, status_label, (x1 + 6, status_y), font, font_scale_info, (255, 255, 255), thickness)

    logger.info(f"Total: {len(tracks)} tracks (Locked: {len(locked_tracks)}, Pending: {len(pending_confirmations)})")

def cleanup_locked_tracks(current_frame, lock_timeout_frames):
    global locked_tracks
    
    to_remove = []
    for id, lock_info in locked_tracks.items():
        if current_frame - lock_info['last_seen'] > lock_timeout_frames * 2:
            to_remove.append(id)
            logger.info(f"Cleaning up locked track for {id}")
    
    for id in to_remove:
        del locked_tracks[id]


def enhanced_recognize_face(face_image, face_width_pixels, tolerance=0.7, is_locked_track=False):  # Increased tolerance
    global KNOWN_FACE_ENCODINGS_ARRAY
    try:
        distance = estimate_distance(face_width_pixels)

        # INCREASED MAX RECOGNITION DISTANCE
        if distance > MAX_RECOGNITION_DISTANCE * 1.5:  # 50% more range
            logger.info(f"Face too far for recognition: {distance:.1f}m")
            return "Unknown", None, float('inf'), distance, 0.0, None

        # IMPROVED IMAGE ENHANCEMENT for better recognition at distance
        if len(face_image.shape) == 3:
            # More aggressive enhancement for distant faces
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = face_image

        start_time = time.time()
        faces = face_analysis.get(enhanced)
        logger.info(f"Face analysis inference time: {time.time() - start_time:.4f} seconds")
        if not faces:
            return "Unknown", None, float('inf'), distance, 0.0, None

        face_embedding = faces[0].embedding

        # 👇 OPTIMIZATION: Vectorized Cosine Similarity Calculation
        if KNOWN_FACE_ENCODINGS_ARRAY is not None and KNOWN_FACE_ENCODINGS_ARRAY.size > 0:
            # 1. Calculate dot products (numerator)
            dot_products = np.dot(KNOWN_FACE_ENCODINGS_ARRAY, face_embedding)
            
            # 2. Calculate norms (denominator parts)
            norm_a = np.linalg.norm(KNOWN_FACE_ENCODINGS_ARRAY, axis=1)
            norm_b = np.linalg.norm(face_embedding)
            
            # 3. Calculate similarities (division)
            denominator = (norm_a * norm_b)
            similarities = np.divide(dot_products, denominator, 
                                     out=np.zeros_like(dot_products), where=denominator!=0)
        else:
            similarities = np.array([])
        # 👆 END OF OPTIMIZATION

        if similarities.size > 0:
            best_match_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_match_index])

            # LOWER THRESHOLD for recognition, especially for redetection
            recognition_threshold = 0.85 if is_locked_track else 0.80  # Reduced thresholds
            confidence = best_similarity

            if is_locked_track or confidence >= recognition_threshold:
                name = known_face_names[best_match_index]
                id = known_face_ids[best_match_index]
                role_type = known_face_types[best_match_index]

                if role_type == 'faculty':
                    name = f"Faculty: {name}"

                return (
                    name,
                    id,
                    1 - confidence,
                    distance,
                    confidence,
                    role_type
                )

        return "Unknown", None, float('inf'), distance, 0.0, None

    except Exception as e:
        logger.error(f"Error in enhanced_recognize_face: {e}")
        return "Unknown", None, float('inf'), None, 0.0, None


def detect_liveness_cctv(face_image, liveness_threshold):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        adjusted_threshold = liveness_threshold * 0.8
        if fm < adjusted_threshold:
            logger.warning(f"Liveness detection failed: variance {fm} < threshold {adjusted_threshold}")
            return False
        logger.info(f"Liveness detection passed: variance {fm}")
        return True
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return True


def estimate_distance(face_width_pixels):
    if face_width_pixels < 20:
        return float('inf')
    estimated_distance = (FACE_SIZE_FOR_DISTANCE * 2) / face_width_pixels
    return estimated_distance


def recognize_face_with_anti_spoofing(face_image, tolerance=0.6, liveness_threshold=100):
    if not detect_liveness_cctv(face_image, liveness_threshold):
        return "Unknown", None, float('inf'), False, 0.0
    face_width = face_image.shape[1]
    name, student_id, distance, est_distance, confidence, role_type = enhanced_recognize_face(face_image, face_width, tolerance)
    return name, student_id, distance, est_distance, True, confidence


# Add this helper function at the top of detect.py
def calculate_late_status(attendance_time, session_start, threshold_minutes):
    """Calculate if student is late based on threshold"""
    try:
        att_time = datetime.strptime(attendance_time, "%Y-%m-%d %H:%M:%S")
        time_diff = (att_time - session_start).total_seconds() / 60
        
        if time_diff <= threshold_minutes:
            return 'present'
        else:
            return 'late'
    except:
        return 'present'

def refresh_with_detections(frame, rgb, frame_idx):
    """
    FIXED: Locked tracks use BODY tracking only, no face boxes
    FIXED: Overlapping prevention and faster track removal
    """
    global tracks, locked_tracks, pending_confirmations, KNOWN_FACE_ENCODINGS_ARRAY
    global detectionStopped, current_fps, skip_frame_counter, student_presence_tracker
    
    if detectionStopped:
        return
    
    # Skip every other frame if FPS is too low
    if current_fps < 10 and frame_idx % 2 == 0:
        return
    
    h, w = frame.shape[:2]

    # Dynamic detection frequency
    if len(tracks) > MAX_TRACKS:
        locked_track_count = len(locked_tracks)
        unlocked_tracks = [tr for tr in tracks if tr.get('id') not in locked_tracks]
        unlocked_tracks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        unlocked_tracks = unlocked_tracks[:MAX_UNLOCKED_TRACKS]
        tracks[:] = unlocked_tracks
        logger.info(f"Track cleanup: {locked_track_count} locked + {len(unlocked_tracks)} unlocked")

    # Cleanup stale pending confirmations
    cleanup_pending_confirmations(frame_idx, timeout_frames=15)

    # Step 1: Face detection
    try:
        faces = face_analysis.get(rgb)
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return

    # Filter out locked faces with STRICTER overlap detection
    locked_body_boxes = []
    for person_id, lock_info in locked_tracks.items():
        body_box = lock_info.get('body_box')
        if body_box:
            bx1, by1, bx2, by2 = body_box
            expand_margin = 8  # Reduced from 15 to 8 pixels
            expanded_body_box = (
                max(0, bx1 - expand_margin),
                max(0, by1 - expand_margin),
                min(w, bx2 + expand_margin),
                min(h, by2 + expand_margin)
            )
            locked_body_boxes.append((person_id, expanded_body_box, lock_info.get('name', person_id)))

    # Check if locked students who were missing are now detected
    for person_id, lock_info in locked_tracks.items():
        if (lock_info.get('type') == 'student' and 
            person_id in student_presence_tracker and
            not student_presence_tracker[person_id].get('present', True)):
            
            for face in faces:
                try:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    
                    body_box = lock_info.get('body_box')
                    if body_box:
                        bx1, by1, bx2, by2 = body_box
                        if (bx1 <= face_center_x <= bx2 and by1 <= face_center_y <= by2):
                            logger.info(f"Missing locked student detected: {lock_info.get('name')}")
                            mark_attendance(lock_info.get('name'), person_id, lock_info.get('type', 'student'))
                            break
                except Exception as e:
                    continue

    # Add face size validation to prevent small faces from stealing tracks
    dets = []
    for idx, face in enumerate(faces):
        face_idx = idx 
        face_obj = face

        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score
            
            if x2 > x1 and y2 > y1:
                is_within_bounds = (x1 >= -30 and y1 >= -30 and x2 <= w + 30 and y2 <= h + 30)

                if is_within_bounds:
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    if box_width >= 30 and box_height >= 30:  
                        
                        is_locked_face = False
                        face_center_x = (x1 + x2) // 2
                        face_center_y = (y1 + y2) // 2
                        
                        for locked_person_id, locked_body_box, locked_name in locked_body_boxes:
                            bx1, by1, bx2, by2 = locked_body_box
                            
                            if (bx1 <= face_center_x <= bx2 and by1 <= face_center_y <= by2):
                                body_height = by2 - by1
                                body_width = bx2 - bx1
                                face_relative_y = face_center_y - by1
                                face_relative_x = face_center_x - bx1
                                
                                face_area = box_width * box_height
                                body_area = body_width * body_height
                                
                                if (face_relative_y < body_height * 0.4 and
                                    face_relative_x > body_width * 0.15 and 
                                    face_relative_x < body_width * 0.85 and
                                    face_area > body_area * 0.015 and
                                    face_area < body_area * 0.2):
                                    is_locked_face = True
                                    break
                        
                        if not is_locked_face:
                            dets.append((x1, y1, x2, y2, conf, face_idx, face_obj))
        
        except Exception as e:
            logger.error(f"Error processing face detection (idx: {face_idx}): {e}")
            continue

    logger.info(f"Frame {frame_idx}: {len(faces)} faces detected → {len(dets)} NEW faces")

    # Step 2: Get body detections
    body_detections = detect_bodies(frame)
    
    # Step 3: Match and update existing unlocked tracks with FASTER cleanup
    new_tracks = []
    used_detections = set()
    used_body_indices = set()

    # Update existing unlocked tracks FIRST
    for tr in tracks:
        if tr.get('id') in locked_tracks:
            new_tracks.append(tr)
            continue
        
        frames_since_seen = frame_idx - tr.get('last_seen', 0)
        if frames_since_seen > 20:
            continue
        
        best_detection = None
        best_iou = 0.4
        best_idx = -1

        tr_box = tr.get('box')
        if tr_box:
            for idx_det, (x1, y1, x2, y2, conf, face_idx_val, face_obj) in enumerate(dets):
                if idx_det in used_detections:
                    continue
                overlap_iou = iou((x1, y1, x2, y2), tr_box)
                if overlap_iou > best_iou:
                    best_iou = overlap_iou
                    best_detection = (x1, y1, x2, y2, conf, face_idx_val, face_obj)
                    best_idx = idx_det

            if best_detection:
                used_detections.add(best_idx)
                x1, y1, x2, y2, conf, face_idx_val, face_obj = best_detection
                tr['box'] = (x1, y1, x2, y2)
                tr['last_seen'] = frame_idx
                new_tracks.append(tr)
            else:
                if frames_since_seen < 8:
                    new_tracks.append(tr)

    # Step 4: Process NEW face detections
    for idx_det, (x1, y1, x2, y2, conf, face_idx_val, face_obj) in enumerate(dets):
        if idx_det in used_detections:
            continue

        # Check overlap with existing tracks
        overlaps_existing = False
        for existing_tr in new_tracks:
            if iou((x1, y1, x2, y2), existing_tr.get('box', (0, 0, 0, 0))) > 0.3:
                overlaps_existing = True
                break

        if overlaps_existing:
            continue

        # Step 5: Face Recognition
        name = "Unknown"
        person_id = None
        ptype = None
        confidence = conf
        face_embedding = None

        if conf >= 0.15 and KNOWN_FACE_ENCODINGS_ARRAY is not None and KNOWN_FACE_ENCODINGS_ARRAY.size > 0:
            try:
                face_embedding = face_obj.embedding
                
                # Cosine similarity
                dot_products = np.dot(KNOWN_FACE_ENCODINGS_ARRAY, face_embedding)
                norm_a = np.linalg.norm(KNOWN_FACE_ENCODINGS_ARRAY, axis=1)
                norm_b = np.linalg.norm(face_embedding)
                denominator = norm_a * norm_b
                similarities = np.divide(dot_products, denominator,
                                         out=np.zeros_like(dot_products, dtype=float),
                                         where=denominator != 0)

                if similarities.size > 0:
                    best_match_index = int(np.argmax(similarities))
                    best_similarity = float(similarities[best_match_index])

                    if best_similarity >= 0.55:
                        name = known_face_names[best_match_index]
                        person_id = known_face_ids[best_match_index]
                        ptype = known_face_types[best_match_index]
                        confidence = min(1.0, (conf * 0.4) + (best_similarity * 0.6))
                        
                        if ptype == 'faculty':
                            name = f"Faculty: {name}"
                        
                        logger.info(f"RECOGNITION MATCH: {name} ({person_id}) - Similarity: {best_similarity:.3f}")
                    else:
                        # No match — capture unknown face during scan
                        if is_face_scan_active() and face_embedding is not None:
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                # 🆕 ADD session_id parameter
                                add_unknown_face(face_crop, face_embedding, session_id=get_current_session_id())
                                logger.info("UNKNOWN FACE CAPTURED - No matches found")
            except Exception as e:
                logger.error(f"Error in recognition: {e}")

        # If still unknown after recognition attempt
        if person_id is None:
            unique_id = f"U-{frame_idx}-{x1}"
            # Capture unknown face during scan
            if is_face_scan_active() and face_embedding is not None:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    # 🆕 ADD session_id parameter
                    add_unknown_face(face_crop, face_embedding, session_id=get_current_session_id())
                    logger.info("UNKNOWN FACE CAPTURED - No recognition attempted")

            # Step 6: Match face to body
            matched_body_idx, match_score = match_face_to_body(
                (x1, y1, x2, y2),
                body_detections,
                iou_threshold=BODY_MATCH_IOU_THRESHOLD
            )
            
            if matched_body_idx is not None and matched_body_idx not in used_body_indices:
                body_box = tuple(body_detections[matched_body_idx]['box'])
                used_body_indices.add(matched_body_idx)
                
                logger.debug(f"Matched face to body (score: {match_score:.2f})")
                
                # Step 7: CONFIRMATION QUEUE (only for recognized people)
                if person_id and person_id not in locked_tracks:
                    if person_id not in pending_confirmations:
                        pending_confirmations[person_id] = {
                            'frames': [],
                            'body_boxes': [],
                            'name': name,
                            'type': ptype,
                            'similarities': [],
                            'first_seen': frame_idx,
                            'last_seen': frame_idx
                        }
                    
                    # Add this frame's data
                    pending_confirmations[person_id]['frames'].append(frame_idx)
                    pending_confirmations[person_id]['body_boxes'].append(body_box)
                    pending_confirmations[person_id]['similarities'].append(best_similarity)
                    pending_confirmations[person_id]['last_seen'] = frame_idx
                    
                    # Keep only recent frames
                    max_frames_to_keep = CONFIRMATION_FRAMES_REQUIRED + 2
                    if len(pending_confirmations[person_id]['frames']) > max_frames_to_keep:
                        pending_confirmations[person_id]['frames'].pop(0)
                        pending_confirmations[person_id]['body_boxes'].pop(0)
                        pending_confirmations[person_id]['similarities'].pop(0)
                    
                    # Check confirmation
                    confirmation_data = pending_confirmations[person_id]
                    consecutive_frames = len(confirmation_data['frames'])
                    avg_similarity = sum(confirmation_data['similarities']) / len(confirmation_data['similarities'])
                    
                    if (consecutive_frames >= CONFIRMATION_FRAMES_REQUIRED and
                        avg_similarity >= CONFIRMATION_SIMILARITY_THRESHOLD):
                        
                        best_body_box = confirmation_data['body_boxes'][-1]
                        reid_features = extract_reid_features(frame, best_body_box)
                        
                        # Lock track — BODY ONLY
                        locked_tracks[person_id] = {
                            'name': name,
                            'type': ptype,
                            'body_box': best_body_box,
                            'last_seen': frame_idx,
                            'reid_features': reid_features,
                            'lock_start': frame_idx,
                            'missed_detections': 0
                        }
                        
                        locked_track_obj = {
                            'id': person_id,
                            'name': name,
                            'type': ptype,
                            'is_locked': True,
                            'body_box': best_body_box,
                            'last_seen': frame_idx,
                            'confidence': 1.0,
                            'tracking_duration': 0,
                            'lock_start': frame_idx
                        }
                        new_tracks.append(locked_track_obj)
                        
                        mark_attendance(name, person_id, ptype)
                        del pending_confirmations[person_id]
                        logger.info(f"LOCKED & ATTENDANCE MARKED for {name} ({person_id})")
                    else:
                        # Still pending
                        temp_track = {
                            'id': person_id,
                            'box': (x1, y1, x2, y2),
                            'body_box': body_box,
                            'confidence': confidence,
                            'last_seen': frame_idx,
                            'is_locked': False,
                            'name': name,
                            'is_pending': True
                        }
                        new_tracks.append(temp_track)
                else:
                    # Recognized but no body match
                    new_tracks.append({
                        'id': person_id,
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'last_seen': frame_idx,
                        'is_locked': False,
                        'name': name
                    })
            else:
                # No body match — create unknown track
                new_tracks.append({
                    'id': unique_id,
                    'box': (x1, y1, x2, y2),
                    'confidence': max(0.3, conf),
                    'last_seen': frame_idx,
                    'is_locked': False,
                    'name': "Unknown",
                    'type': 'unknown',
                    'start_frame': frame_idx
                })
        else:
            # Known person with body match — proceed to confirmation
            matched_body_idx, match_score = match_face_to_body(
                (x1, y1, x2, y2),
                body_detections,
                iou_threshold=BODY_MATCH_IOU_THRESHOLD
            )
            
            if matched_body_idx is not None and matched_body_idx not in used_body_indices:
                body_box = tuple(body_detections[matched_body_idx]['box'])
                used_body_indices.add(matched_body_idx)
                
                if person_id not in locked_tracks:
                    # Same confirmation logic as above (duplicated for clarity)
                    if person_id not in pending_confirmations:
                        pending_confirmations[person_id] = {
                            'frames': [], 'body_boxes': [], 'name': name, 'type': ptype,
                            'similarities': [], 'first_seen': frame_idx, 'last_seen': frame_idx
                        }
                    
                    pending_confirmations[person_id]['frames'].append(frame_idx)
                    pending_confirmations[person_id]['body_boxes'].append(body_box)
                    pending_confirmations[person_id]['similarities'].append(best_similarity)
                    pending_confirmations[person_id]['last_seen'] = frame_idx
                    
                    max_frames_to_keep = CONFIRMATION_FRAMES_REQUIRED + 2
                    if len(pending_confirmations[person_id]['frames']) > max_frames_to_keep:
                        pending_confirmations[person_id]['frames'].pop(0)
                        pending_confirmations[person_id]['body_boxes'].pop(0)
                        pending_confirmations[person_id]['similarities'].pop(0)
                    
                    confirmation_data = pending_confirmations[person_id]
                    consecutive_frames = len(confirmation_data['frames'])
                    avg_similarity = sum(confirmation_data['similarities']) / len(confirmation_data['similarities'])
                    
                    if (consecutive_frames >= CONFIRMATION_FRAMES_REQUIRED and
                        avg_similarity >= CONFIRMATION_SIMILARITY_THRESHOLD):
                        best_body_box = confirmation_data['body_boxes'][-1]
                        reid_features = extract_reid_features(frame, best_body_box)
                        
                        locked_tracks[person_id] = {
                            'name': name, 'type': ptype, 'body_box': best_body_box,
                            'last_seen': frame_idx, 'reid_features': reid_features,
                            'lock_start': frame_idx, 'missed_detections': 0
                        }
                        
                        locked_track_obj = {
                            'id': person_id, 'name': name, 'type': ptype, 'is_locked': True,
                            'body_box': best_body_box, 'last_seen': frame_idx,
                            'confidence': 1.0, 'tracking_duration': 0, 'lock_start': frame_idx
                        }
                        new_tracks.append(locked_track_obj)
                        mark_attendance(name, person_id, ptype)
                        del pending_confirmations[person_id]
                        logger.info(f"LOCKED & ATTENDANCE MARKED for {name} ({person_id})")
                    else:
                        temp_track = {
                            'id': person_id, 'box': (x1, y1, x2, y2), 'body_box': body_box,
                            'confidence': confidence, 'last_seen': frame_idx,
                            'is_locked': False, 'name': name, 'is_pending': True
                        }
                        new_tracks.append(temp_track)
            else:
                new_tracks.append({
                    'id': person_id, 'box': (x1, y1, x2, y2), 'confidence': confidence,
                    'last_seen': frame_idx, 'is_locked': False, 'name': name
                })

    # Confidence decay for unlocked tracks
    for tr in new_tracks[:]:
        if not tr.get('is_locked'):
            tr['confidence'] = max(0.1, tr.get('confidence', 0.5) * 0.85)
            if tr['confidence'] < 0.2 and frame_idx - tr.get('last_seen', 0) > 10:
                new_tracks.remove(tr)

    # Update global tracks
    tracks[:] = new_tracks
    
    # Cleanup old unknown tracks
    current_tracks = [
        tr for tr in tracks
        if not (tr.get('name') == "Unknown" and frame_idx - tr.get('last_seen', 0) > 30)
    ]
    tracks[:] = current_tracks
    
    logger.info(f"Total: {len(tracks)} tracks (Locked: {len(locked_tracks)}, Pending: {len(pending_confirmations)})")

def get_current_session_id():
    """Get the current active session ID for face capture"""
    global current_session_id
    if current_session_id is None:
        logger.warning("⚠️ current_session_id is None - faces won't be saved to database!")
    else:
        logger.info(f"🔍 get_current_session_id returning: {current_session_id}")
    return current_session_id

def is_face_scan_active():
    """Check if face scan is currently active"""
    global FACE_SCAN_START_TIME, is_face_scan_active_flag
    
    if not is_face_scan_active_flag or FACE_SCAN_START_TIME is None:
        return False
    
    elapsed = (datetime.now() - FACE_SCAN_START_TIME).total_seconds()
    return elapsed <= FACE_SCAN_DURATION

def start_face_scan():
    """Start the 15-second face scan period"""
    global FACE_SCAN_START_TIME, is_face_scan_active_flag
    
    FACE_SCAN_START_TIME = datetime.now()
    is_face_scan_active_flag = True
    logger.info(f"🔍 FACE SCAN STARTED - Will capture unknown faces for {FACE_SCAN_DURATION} seconds")
    
    # Auto-stop after duration
    def auto_stop():
        time.sleep(FACE_SCAN_DURATION)
        stop_face_scan()
    
    threading.Thread(target=auto_stop, daemon=True).start()

def stop_face_scan():
    """Stop the face scan"""
    global is_face_scan_active_flag
    
    is_face_scan_active_flag = False
    logger.info("🛑 FACE SCAN STOPPED - No longer capturing unknown faces")

def get_remaining_scan_time():
    """Get remaining scan time in seconds"""
    if not is_face_scan_active():
        return 0
    
    elapsed = (datetime.now() - FACE_SCAN_START_TIME).total_seconds()
    remaining = max(0, FACE_SCAN_DURATION - elapsed)
    return int(remaining)    

def cleanup_pending_confirmations(current_frame, timeout_frames=10):
    """
    ✅ FIX #16: Aggressive cleanup of stale pending confirmations
    Removes confirmations not seen in the last 10 frames (0.33 seconds at 30fps)
    """
    global pending_confirmations
    
    to_remove = []
    for person_id, data in pending_confirmations.items():
        last_seen = data.get('last_seen', data.get('first_seen', current_frame))
        
        # Remove if stale (not seen in last 10 frames)
        if current_frame - last_seen > timeout_frames:
            to_remove.append(person_id)
            logger.info(f"🗑️ Cleanup stale confirmation: {data.get('name', person_id)} (stale {current_frame - last_seen} frames)")
    
    for person_id in to_remove:
        del pending_confirmations[person_id]

# ============================================
# CHANGE 4: Add these 2 new functions
# ============================================

def extract_reid_features(frame, body_box):
    """Extract person re-id features from body crop - OPTIMIZED WITH FP16"""
    if reid_model is None:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in body_box]
        body_crop = frame[y1:y2, x1:x2]
        
        if body_crop.shape[0] < 64 or body_crop.shape[1] < 32:
            return None
        
        body_crop_resized = cv2.resize(body_crop, (128, 256))
        body_crop_normalized = body_crop_resized.astype(np.float32) / 255.0
        
        # 🎯 CONVERT TO FP16 FOR GPU INFERENCE
        if DEVICE == "cuda":
            body_crop_normalized = body_crop_normalized.astype(np.float16)  # 🆕 FP16 conversion
        
        body_crop_tensor = torch.from_numpy(
            np.transpose(body_crop_normalized, (2, 0, 1))
        ).unsqueeze(0)
        
        if torch.cuda.is_available():
            body_crop_tensor = body_crop_tensor.cuda()
            # 🎯 ENSURE TENSOR MATCHES MODEL PRECISION
            if next(reid_model.parameters()).dtype == torch.float16:
                body_crop_tensor = body_crop_tensor.half()  # 🆕 Match model precision
        
        with torch.no_grad():
            reid_features = reid_model(body_crop_tensor)
        
        # 🎯 CONVERT BACK TO FP32 FOR CONSISTENT PROCESSING
        if reid_features.dtype == torch.float16:
            reid_features = reid_features.float()  # 🆕 Convert back to FP32 for numpy
        
        return reid_features.cpu().numpy()[0]
    except Exception as e:
        logger.debug(f"Reid extraction error: {e}")
        return None
    
def calculate_box_distance(box1, box2):
    """
    Calculate center-to-center distance between two bounding boxes
    Used for anti-teleportation detection
    """
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    distance = ((x1_center - x2_center)**2 + (y1_center - y2_center)**2)**0.5
    return distance

def calculate_reid_distance(feat1, feat2):
    """
    Calculate ReID distance using cosine distance
    Returns: 0.0 (identical) to 1.0 (completely different)
    Lower values = more similar
    """
    if feat1 is None or feat2 is None:
        return 1.0
    
    # 🎯 ENSURE FP32 FOR ACCURATE MATH OPERATIONS
    feat1 = feat1.flatten().astype(np.float32)  # 🆕 Ensure FP32
    feat2 = feat2.flatten().astype(np.float32)  # 🆕 Ensure FP32
    
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    cosine_distance = 1.0 - cosine_sim
    
    return cosine_distance

def optimize_memory():
    """Optimize GPU memory usage - CALL THIS PERIODICALLY"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("🧹 GPU memory optimized")

# Add the custom JSON encoder here
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                # Convert timedelta to string in readable format
                total_seconds = int(obj.total_seconds())
                days, remainder = divmod(total_seconds, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                if days > 0:
                    return f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    return f"{hours}h {minutes}m"
                elif minutes > 0:
                    return f"{minutes}m {seconds}s"
                else:
                    return f"{seconds}s"
            return super().default(obj)
        except Exception:
            return str(obj)  # Fallback to string representation

app.json_encoder = CustomJSONEncoder


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        logger.info(f"Login attempt for email: {email}")
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'})
        
        if not email.endswith("@wmsu.edu.ph"):
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        # Authenticate user
        user = authenticate_user(email, password)
        
        if not user:
            return jsonify({'success': False, 'message': 'Invalid email or password'})
        
        # DEBUG: Check what user data is returned
        print(f"DEBUG Login - User object from authenticate_user: {user}")
        print(f"DEBUG Login - User role: {user.get('role')}")
        print(f"DEBUG Login - User type: {user.get('user_type')}")
        
        # Clear any existing session to prevent conflicts
        session.clear()
        
        # Create session
        session_token = create_user_session(user['user_id'], user['user_type'])
        if not session_token:
            return jsonify({'success': False, 'message': 'Failed to create session'})
        
        # Set Flask session
        session['user_id'] = user['user_id']
        session['user_type'] = user['user_type']
        session['first_name'] = user['first_name']
        session['last_name'] = user['last_name']
        session['role'] = user.get('role', '')  # Make sure role exists
        session.permanent = True
        
        # DEBUG: Check what's stored in session
        print(f"DEBUG Login - Stored in session: role={session.get('role')}")
        
        # Determine redirect URL
        redirect_url = '/AdminDB' if user['user_type'] in ['admin', 'faculty'] else '/StudentLP'
        
        logger.info(f"Login successful for {email}, role: {user.get('role')}, redirecting to {redirect_url}")
        
        resp = jsonify({
            'success': True,
            'message': 'Login successful',
            'redirect_url': redirect_url
        })
        
        resp.set_cookie('session_token', session_token, httponly=True, secure=False, samesite='Strict')
        return resp
        
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'})

@app.route('/debug/password', methods=['GET', 'POST'])
def debug_password():
    """Debug route to test password verification"""
    if request.method == 'GET':
        return '''
        <h2>Password Debug Tool</h2>
        <form method="POST">
            Email: <input type="email" name="email" value="admin@wmsu.edu.ph"><br>
            Password: <input type="password" name="password" value="SuperAdmin2024!"><br>
            <input type="submit" value="Test Password">
        </form>
        '''
    
    # POST method handling
    data = request.get_json() if request.is_json else request.form
    email = data.get('email')
    password = data.get('password')
    
    logger.info(f"Debug password check for: {email}")
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'})
    
    cursor = conn.cursor(dictionary=True)
    
    results = {}
    
    try:
        # Check admins table
        cursor.execute("SELECT admin_id, email, password_hash FROM admins WHERE email = %s", (email,))
        admin = cursor.fetchone()
        if admin:
            results['admin'] = {
                'user_id': admin['admin_id'],
                'email': admin['email'],
                'password_hash': admin['password_hash'],
                'password_match': verify_password(password, admin['password_hash'])
            }
        
        # Check students table
        cursor.execute("SELECT student_id, email, password_hash FROM students WHERE email = %s", (email,))
        student = cursor.fetchone()
        if student:
            results['student'] = {
                'user_id': student['student_id'],
                'email': student['email'],
                'password_hash': student['password_hash'],
                'password_match': verify_password(password, student['password_hash'])
            }
        
        # Check faculty table
        cursor.execute("SELECT faculty_id, email, password_hash FROM faculty WHERE email = %s", (email,))
        faculty = cursor.fetchone()
        if faculty:
            results['faculty'] = {
                'user_id': faculty['faculty_id'],
                'email': faculty['email'],
                'password_hash': faculty['password_hash'],
                'password_match': verify_password(password, faculty['password_hash'])
            }
        
    except Exception as e:
        logger.error(f"Debug query error: {e}")
        results['error'] = str(e)
    finally:
        cursor.close()
        conn.close()
    
    return jsonify(results)

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
                'photo': s['photo_path'] if s['photo_path'] else f"https://ui-avatars.com/api/?name={s['first_name']}+{s['last_name']}&background=random",
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
        
        # Format faculty data for frontend (SAME AS STUDENTS!)
        formatted_faculty = []
        for f in faculty:
            formatted_faculty.append({
                'id': f['faculty_id'],
                'idNumber': f['faculty_id'],
                'firstName': f['first_name'],
                'lastName': f['last_name'],
                'middleName': f['middle_name'] or '',
                'name': f"{f['first_name']} {f['middle_name'] + ' ' if f['middle_name'] else ''}{f['last_name']}",
                'department': f['department'],
                'designation': f['designation'],
                'email': f['email'],
                'photo': f['photo_path'] if f['photo_path'] else f"https://ui-avatars.com/api/?name={f['first_name']}+{f['last_name']}&background=random",  # FIXED: Same as students!
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
                    'created_at': invite['created_at'].isoformat() if invite['created_at'] else None,
                    'expires_at': invite['expires_at'].isoformat() if invite['expires_at'] else None,
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
def logout_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user:
            # User is already logged in, redirect based on user type
            if user['user_type'] == 'student':
                return redirect(url_for('student_lp_page'))
            else:  # admin or faculty
                return redirect(url_for('admin_db_page'))
        return f(*args, **kwargs)
    return decorated_function

# Serve faculty photos
os.makedirs('static/images/faculty', exist_ok=True)
os.makedirs('static/images/admins', exist_ok=True)

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
        middle_name = data.get('middle_name', '')
        course = data.get('course')
        year_section = data.get('year_section')
        email = data.get('email')
        status = data.get('status', 'active')
        
        if not all([student_id, first_name, last_name, course, year_section, email]):
            return jsonify({'success': False, 'message': 'All required fields are missing'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if student exists
        cursor.execute("SELECT student_id FROM students WHERE student_id = %s", (student_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Update student with all fields
        cursor.execute(
            """UPDATE students 
               SET first_name = %s, last_name = %s, middle_name = %s, 
                   course = %s, year_section = %s, email = %s, 
                   status = %s, updated_at = NOW()
               WHERE student_id = %s""",
            (first_name, last_name, middle_name, course, year_section, 
             email, status, student_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Refresh known faces
        load_known_faces_from_db()
        
        return jsonify({'success': True, 'message': 'Student updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating student: {str(e)}")
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})

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
        middle_name = data.get('middle_name', '')
        department = data.get('department')
        designation = data.get('designation')
        email = data.get('email')
        role = data.get('role', 'moderator')
        status = data.get('status', 'active')
        
        # Debug logging
        logger.info(f"Updating faculty: {faculty_id}, {first_name} {last_name}")
        
        if not all([faculty_id, first_name, last_name, department, designation, email]):
            return jsonify({'success': False, 'message': 'All required fields are missing'})
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if faculty exists
        cursor.execute("SELECT faculty_id FROM faculty WHERE faculty_id = %s", (faculty_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Faculty Member not found'})
        
        # Update faculty with all fields
        cursor.execute(
            """UPDATE faculty 
               SET first_name = %s, last_name = %s, middle_name = %s, 
                   department = %s, designation = %s, email = %s, 
                   role = %s, status = %s, updated_at = NOW()
               WHERE faculty_id = %s""",
            (first_name, last_name, middle_name, department, designation, 
             email, role, status, faculty_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Refresh known faces
        load_known_faculties_from_db()
        
        logger.info(f"Successfully updated faculty: {faculty_id}")
        return jsonify({'success': True, 'message': 'Faculty Member updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating faculty: {str(e)}")
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})

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

def detect_liveness_cctv(face_image, liveness_threshold):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        adjusted_threshold = liveness_threshold * 0.8
        if fm < adjusted_threshold:
            logger.warning(f"Liveness detection failed: variance {fm} < threshold {adjusted_threshold}")
            return False
        logger.info(f"Liveness detection passed: variance {fm}")
        return True
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return True

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
        
        enhanced_img = img
        
        faces = face_analysis.get(enhanced_img)
        if not faces:
            logger.warning("No face detected in image during registration")
            return jsonify({
                'success': False,
                'message': 'No face detected. Please ensure your face is clearly visible.',
                'current_pose': current_pose,
                'next_pose': current_pose
            }), 400
        
        face = faces[0]
        face_embedding = face.embedding
        yaw, pitch, roll = face.pose
        landmarks = face.landmark_2d_106
        
        # SIMPLIFIED MIRROR CORRECTION: Only invert yaw
        yaw = -yaw  # Correct left/right for mirror
        
        left_eye_indices = [96, 97, 98, 99, 100, 101]
        left_ear = calculate_ear(landmarks, left_eye_indices)
        right_eye_indices = [90, 91, 92, 93, 94, 95]
        right_ear = calculate_ear(landmarks, right_eye_indices)
        mouth_indices = [76, 77, 78, 79, 80, 81, 82, 83]
        mar = calculate_mar(landmarks, mouth_indices)
        
        # IMPROVED pose detection with BETTER up/down thresholds
        pose_results = {
            'is_frontal': bool(abs(yaw) <= 20 and abs(pitch) <= 15),
            'is_left': bool(yaw >= 6),
            'is_right': bool(yaw <= -6),
            'is_up': bool(pitch <= -4),   # LOWER threshold for up (more negative)
            'is_down': bool(pitch >= 3),  # HIGHER threshold for down (more positive)
            'is_mouth_open': bool(mar >= 0.08),
            'is_eyes_closed': bool((left_ear + right_ear) / 2 <= 0.35)
        }
        
        logger.info(f"Pose results for {current_pose}: yaw={yaw:.1f}, pitch={pitch:.1f}, mar={mar:.2f}, ear_avg={(left_ear + right_ear)/2:.2f}")
        
        pose_satisfied = False
        message = ""
        
        # IMPROVED pose checking with BETTER up/down logic
        if current_pose == 'frontal':
            if abs(yaw) <= 10 and abs(pitch) <= 8:
                pose_satisfied = True
                message = "✅ Perfect! Face centered."
            elif abs(yaw) <= 20 and abs(pitch) <= 15:
                pose_satisfied = True
                message = "✅ Good! Face detected."
            else:
                if abs(yaw) > 20:
                    direction = "left" if yaw > 0 else "right"
                    message = f"↔️ Face the camera. You're facing {direction}."
                elif abs(pitch) > 15:
                    direction = "up" if pitch < 0 else "down"
                    message = f"↕️ Face the camera. You're looking {direction}."
                else:
                    message = "👀 Look straight at the camera."
        
        elif current_pose == 'left':
            if yaw >= 10:
                pose_satisfied = True
                message = "✅ Perfect! Good left turn."
            elif yaw >= 5:
                pose_satisfied = True
                message = "✅ Good! Left turn detected."
            else:
                if yaw < 0:
                    message = "🔄 Turn your head to the LEFT"
                elif yaw < 3:
                    message = "↩️ Turn a bit more to the left"
                else:
                    message = "👍 Almost there! Turn slightly more left"
        
        elif current_pose == 'right':
            if yaw <= -10:
                pose_satisfied = True
                message = "✅ Perfect! Good right turn."
            elif yaw <= -5:
                pose_satisfied = True
                message = "✅ Good! Right turn detected."
            else:
                if yaw > 0:
                    message = "🔄 Turn your head to the RIGHT"
                elif yaw > -3:
                    message = "↪️ Turn a bit more to the right"
                else:
                    message = "👍 Almost there! Turn slightly more right"
        
        elif current_pose == 'up':
            # IMPROVED UP DETECTION - More lenient and clear
            if pitch <= -20:
                pose_satisfied = True
                message = "✅ Perfect! Great upward tilt."
            elif pitch <= -15:
                pose_satisfied = True
                message = "✅ Excellent! Upward tilt detected."
            elif pitch <= -10:
                pose_satisfied = True
                message = "✅ Good! Upward movement detected."
            else:
                if pitch > 5:
                    message = "🔼 Tilt your head UP (chin up, look at ceiling)"
                elif pitch > 0:
                    message = "⬆️ Tilt more upward"
                elif pitch > -5:
                    message = "👆 A bit more upward"
                else:
                    message = "👍 Almost there! Tilt slightly more up"
        
        elif current_pose == 'down':
            # IMPROVED DOWN DETECTION - More lenient and clear
            if pitch >= 20:
                pose_satisfied = True
                message = "✅ Perfect! Great downward tilt."
            elif pitch >= 15:
                pose_satisfied = True
                message = "✅ Excellent! Downward tilt detected."
            elif pitch >= 10:
                pose_satisfied = True
                message = "✅ Good! Downward movement detected."
            else:
                if pitch < -5:
                    message = "🔽 Tilt your head DOWN (chin down, look at floor)"
                elif pitch < 0:
                    message = "⬇️ Tilt more downward"
                elif pitch < 5:
                    message = "👇 A bit more downward"
                else:
                    message = "👍 Almost there! Tilt slightly more down"
        
        elif current_pose == 'mouth_open':
            if mar >= 0.12:
                pose_satisfied = True
                message = "✅ Perfect! Mouth open detected."
            elif mar >= 0.08:
                pose_satisfied = True
                message = "✅ Good! Mouth open detected."
            else:
                message = "😮 Open your mouth slightly"
        
        elif current_pose == 'eyes_closed':
            avg_ear = (left_ear + right_ear) / 2
            if avg_ear <= 0.25:
                pose_satisfied = True
                message = "✅ Perfect! Eyes closed detected."
            elif avg_ear <= 0.35:
                pose_satisfied = True
                message = "✅ Good! Eyes closed detected."
            else:
                message = "😌 Close your eyes gently"
        
        # INSTANT SUCCESS for ANY up/down movement (very lenient)
        if not pose_satisfied:
            if current_pose == 'up' and pitch < 0:  # ANY negative pitch for up
                pose_satisfied = True
                message = "✅ Good! Upward movement detected."
            elif current_pose == 'down' and pitch > 0:  # ANY positive pitch for down
                pose_satisfied = True
                message = "✅ Good! Downward movement detected."
            elif current_pose == 'left' and yaw > 0:
                pose_satisfied = True
                message = "✅ Good! Left movement detected."
            elif current_pose == 'right' and yaw < 0:
                pose_satisfied = True
                message = "✅ Good! Right movement detected."
        
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
            'success': bool(pose_satisfied),
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
                os.makedirs('static/images/student_photos', exist_ok=True)
                photo_path = f"static/images/student_photos/{student_id}.jpg"
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
                os.makedirs('static/images/faculty', exist_ok=True)
                photo_path = f"static/images/faculty/{faculty_id}.jpg"
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
        
        # Get current timestamp
        current_time = datetime.now()
        
        # Insert faculty with ALL required fields including status and timestamps
        cursor.execute(
            """INSERT INTO faculty 
            (faculty_id, first_name, last_name, middle_name, department, designation, email, 
             face_encoding, photo_path, password_hash, role, status, created_at, updated_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (faculty_id, first_name, last_name, middle_name or None, department, designation, email, 
             encoding_str, photo_path, password_hash, role, 'active', current_time, current_time)
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

def get_current_user():
    """Get current user from session token - handles both Flask session and cookie"""
    # First check Flask session
    if 'user_id' in session and 'user_type' in session:
        user_data = {
            'user_id': session['user_id'],
            'user_type': session['user_type'],
            'first_name': session.get('first_name', ''),
            'last_name': session.get('last_name', ''),
            'role': session.get('role', '')
        }
        print(f"DEBUG - From session: {user_data}")
        return user_data
    
    # Fallback to cookie-based session token
    session_token = request.cookies.get('session_token')
    if not session_token:
        print("DEBUG - No session token")
        return None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        print(f"DEBUG - Session token: {session_token}")
        
        # FIXED QUERY: Better debugging and case handling
        query = """
            SELECT 
                us.user_id, 
                us.user_type, 
                us.expires_at,
                CASE 
                    WHEN us.user_type = 'admin' THEN a.first_name
                    WHEN us.user_type = 'faculty' THEN f.first_name
                    WHEN us.user_type = 'student' THEN s.first_name
                END as first_name,
                CASE 
                    WHEN us.user_type = 'admin' THEN a.last_name
                    WHEN us.user_type = 'faculty' THEN f.last_name
                    WHEN us.user_type = 'student' THEN s.last_name
                END as last_name,
                CASE 
                    WHEN us.user_type = 'admin' THEN a.role
                    WHEN us.user_type = 'faculty' THEN f.role
                    ELSE 'student'
                END as role
            FROM user_sessions us
            LEFT JOIN admins a ON us.user_id = a.admin_id AND us.user_type = 'admin'
            LEFT JOIN faculty f ON us.user_id = f.faculty_id AND us.user_type = 'faculty' 
            LEFT JOIN students s ON us.user_id = s.student_id AND us.user_type = 'student'
            WHERE us.session_token = %s AND us.expires_at > NOW()
        """
        
        cursor.execute(query, (session_token,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        print(f"DEBUG - Database query result: {user}")
        
        if not user:
            print("DEBUG - No user found in database")
            session.clear()
            return None
        
        # Store in Flask session for future requests
        session['user_id'] = user['user_id']
        session['user_type'] = user['user_type']
        session['first_name'] = user['first_name']
        session['last_name'] = user['last_name']
        session['role'] = user['role']
        
        print(f"DEBUG - Final user data: {user}")
        return user
        
    except Exception as e:
        print(f"DEBUG - Error in get_current_user: {e}")
        logger.error(f"Error getting current user: {e}")
        session.clear()
        return None

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

def login_required(f):
    """Decorator for all routes - checks if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            # Clear any invalid session
            session.clear()
            return redirect('/login')
        
        # Set user in g context for access in routes and templates
        g.user = user
        return f(*args, **kwargs)
    return decorated_function

def student_login_required(f):
    """Decorator for student routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect('/login')
        
        # If admin/faculty tries to access student route, redirect to admin page
        if user['user_type'] != 'student':
            return redirect('/AdminDB')
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/student/attendance-data')
def student_attendance_data():
    """Get attendance data for the logged-in student"""
    student_id = get_current_student_id()
    
    if not student_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = connection.cursor(dictionary=True)
        
        print(f"Fetching data for student: {student_id}")
        
        # 1. Get student's basic info and section
        cursor.execute("""
            SELECT student_id, first_name, last_name, course, year_section, section_id
            FROM students 
            WHERE student_id = %s AND status = 'active'
        """, (student_id,))
        
        student_data = cursor.fetchone()
        if not student_data:
            return jsonify({'error': 'Student not found'}), 404
        
        print(f"Found student: {student_data}")
        
        # 2. Get today's classes (based on schedule, not attendance)
        today = date.today()
        today_day = today.strftime('%A')  # Get today's day name (Monday, Tuesday, etc.)
        
        cursor.execute("""
            SELECT DISTINCT 
                s.subject_code, 
                s.subject_name,
                cs.class_type,
                cs.start_time, 
                cs.end_time, 
                cs.room,
                cs.day_of_week
            FROM subjects s
            JOIN class_schedules cs ON s.subject_id = cs.subject_id
            JOIN year_sections ys ON s.section_id = ys.section_id
            WHERE ys.section_name = 'C' 
            AND ys.year_level = 4
            AND cs.day_of_week = %s
            AND s.status = 'active'
            AND cs.status = 'active'
            ORDER BY cs.start_time
        """, (today_day,))
        
        today_schedule = cursor.fetchall()
        print(f"Today's schedule ({today_day}): {len(today_schedule)} classes")
        
        # 3. Get today's attendance to match with schedule
        cursor.execute("""
            SELECT 
                a.session_id, 
                a.status, 
                a.timestamp,
                ases.class_name,
                ases.started_at, 
                ases.ended_at
            FROM attendance a
            LEFT JOIN attendance_sessions ases ON a.session_id = ases.session_id
            WHERE a.student_id = %s AND DATE(a.timestamp) = %s
            ORDER BY a.timestamp DESC
        """, (student_id, today))
        
        today_attendance = cursor.fetchall()
        
        # 4. Get attendance history (last 30 days) - UPDATED QUERY
        cursor.execute("""
            SELECT 
                a.id,
                a.session_id, 
                a.status, 
                a.timestamp,
                a.subject_code,  -- ✅ GET FROM ATTENDANCE TABLE
                a.subject_name,  -- ✅ GET FROM ATTENDANCE TABLE
                a.room,          -- ✅ GET FROM ATTENDANCE TABLE
                ases.class_name,
                ases.started_at, 
                ases.ended_at,
                cs.class_type
            FROM attendance a
            LEFT JOIN attendance_sessions ases ON a.session_id = ases.session_id
            LEFT JOIN class_schedules cs ON (
                cs.room = a.room 
                AND TIME(ases.started_at) BETWEEN TIME(cs.start_time) AND TIME(cs.end_time)
            )
            WHERE a.student_id = %s 
            AND a.timestamp >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            ORDER BY a.timestamp DESC
            LIMIT 50
        """, (student_id,))
        
        attendance_history = cursor.fetchall()
        print(f"Attendance history: {len(attendance_history)} records")
        
        # 5. Get all subjects for BSIT 4C
        cursor.execute("""
            SELECT DISTINCT 
                s.subject_code, 
                s.subject_name, 
                cs.day_of_week, 
                TIME_FORMAT(cs.start_time, '%H:%i') as start_time,
                TIME_FORMAT(cs.end_time, '%H:%i') as end_time,
                cs.room,
                cs.class_type,
                ys.section_name,
                ys.year_level,
                p.program_name
            FROM subjects s
            JOIN class_schedules cs ON s.subject_id = cs.subject_id
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE ys.section_name = 'C' 
            AND ys.year_level = 4
            AND p.program_id = 'IT'
            AND s.status = 'active'
            AND cs.status = 'active'
            ORDER BY cs.day_of_week, cs.start_time
        """)
        
        semester_classes = cursor.fetchall()
        print(f"Semester classes found: {len(semester_classes)}")
        
        # 6. Calculate attendance statistics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT a.session_id) as total_sessions,
                SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN a.status = 'late' THEN 1 ELSE 0 END) as late_count,
                SUM(CASE WHEN a.status = 'excused' THEN 1 ELSE 0 END) as excused_count
            FROM attendance a
            JOIN attendance_sessions ases ON a.session_id = ases.session_id
            WHERE a.student_id = %s 
            AND ases.status = 'completed'
        """, (student_id,))
        
        stats = cursor.fetchone()
        print(f"Stats: {stats}")
        
        total_classes = stats['total_sessions'] or 0
        attended_classes = (stats['present_count'] or 0) + (stats['late_count'] or 0)
        attendance_rate = (attended_classes / total_classes * 100) if total_classes > 0 else 0
        
        # Format today's classes - combine schedule with attendance
        today_classes = []
        for schedule in today_schedule:
            # Find matching attendance record
            attendance_status = 'Not Recorded'
            for attendance in today_attendance:
                if (schedule['subject_code'] in attendance.get('class_name', '') or 
                    schedule['subject_name'] in attendance.get('class_name', '')):
                    attendance_status = attendance['status']
                    break
            
            today_classes.append({
                'course': f"{schedule['subject_code']} - {schedule['subject_name']}",
                'time': f"{schedule['start_time']} - {schedule['end_time']}",
                'type': schedule['class_type'].title(),
                'status': attendance_status,
                'room': schedule['room']
            })
        
        # Format attendance history - IMPROVED WITH DIRECT SUBJECT INFO
        formatted_history = []
        for record in attendance_history:
            # Format time
            start_time = record['started_at'].strftime('%H:%M') if record['started_at'] else 'N/A'
            end_time = record['ended_at'].strftime('%H:%M') if record['ended_at'] else 'N/A'
            
            # Get course name - PRIORITIZE SUBJECT INFO FROM ATTENDANCE TABLE
            if record['subject_code'] and record['subject_name']:
                course_name = f"{record['subject_code']} - {record['subject_name']}"
            elif record['class_name']:
                # Format class name to be more readable
                class_name = record['class_name']
                if "Information Technology" in class_name:
                    class_name = class_name.replace("Information Technology", "BSIT")
                elif "Computer Science" in class_name:
                    class_name = class_name.replace("Computer Science", "BSCS")
                elif "Associate in Computer Technology" in class_name:
                    class_name = class_name.replace("Associate in Computer Technology", "ACT")
                course_name = class_name
            else:
                course_name = 'Unknown Class'
            
            formatted_history.append({
                'date': record['timestamp'].strftime('%Y-%m-%d') if record['timestamp'] else 'N/A',
                'course': course_name,
                'time': f"{start_time} - {end_time}",
                'room': record['room'] or 'N/A',
                'status': record['status'] or 'absent',
                'type': record['class_type'] or 'Class'
            })
        
        # Format semester subjects
        formatted_semester_classes = []
        for subject in semester_classes:
            days_map = {
                'Monday': 'Mon',
                'Tuesday': 'Tue', 
                'Wednesday': 'Wed',
                'Thursday': 'Thu',
                'Friday': 'Fri',
                'Saturday': 'Sat'
            }
            
            day_abbr = days_map.get(subject['day_of_week'], subject['day_of_week'])
            
            formatted_semester_classes.append({
                'course': f"{subject['subject_code']} - {subject['subject_name']}",
                'schedule': f"{day_abbr}, {subject['start_time']} - {subject['end_time']}",
                'room': subject['room'],
                'type': subject['class_type'].title(),
                'program': subject['program_name'],
                'section': f"{subject['year_level']}{subject['section_name']}"
            })
        
        response_data = {
            'student': {
                'name': f"{student_data['first_name']} {student_data['last_name']}",
                'course': student_data['course'],
                'section': student_data['year_section']
            },
            'stats': {
                'attendance_rate': round(attendance_rate, 2),
                'total_classes': total_classes,
                'attended_classes': attended_classes
            },
            'today_classes': today_classes,
            'attendance_history': formatted_history,
            'semester_classes': formatted_semester_classes,
            'today_date': today.strftime('%Y-%m-%d')
        }
        
        print(f"Response data prepared successfully")
        print(f"Today classes: {len(today_classes)}")
        print(f"Attendance history: {len(formatted_history)}")
        print(f"Semester classes: {len(formatted_semester_classes)}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error fetching attendance data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch attendance data: {str(e)}'}), 500
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def get_current_student_id():
    """Get current student ID from session or token"""
    # For demo purposes, using a fixed student ID
    # In production, get from session/token
    return '2022-01376'

@app.route('/api/get_section_filters', methods=['GET'])
@login_required
def get_section_filters():
    """Get unique course and year_section combinations for filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT DISTINCT 
                ys.section_id,
                ys.year_level,
                ys.section_name as year_section,
                p.program_name as course,
                p.program_id
            FROM year_sections ys
            JOIN programs p ON ys.program_id = p.program_id
            WHERE ys.status = 'active'
            ORDER BY p.program_name, ys.year_level, ys.section_name
        """)
        
        sections = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'sections': sections})
        
    except Exception as e:
        logger.error(f"Error getting section filters: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_section_info', methods=['GET'])
@login_required
def get_section_info():
    """Get section information by section_id"""
    try:
        section_id = request.args.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                ys.section_id,
                ys.year_level,
                ys.section_name,
                p.program_name,
                p.program_id,
                p.program_code
            FROM year_sections ys
            JOIN programs p ON ys.program_id = p.program_id
            WHERE ys.section_id = %s AND ys.status = 'active'
        """, (section_id,))
        
        section = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if section:
            return jsonify({'success': True, 'section': section})
        else:
            return jsonify({'success': False, 'message': 'Section not found'})
        
    except Exception as e:
        logger.error(f"Error getting section info: {e}")
        return jsonify({'success': False, 'message': str(e)})


# Routes
@app.route('/timer')
@login_required
def timer_page():
    user = g.get('user', {})
    user_data = {
        'user_id': user.get('user_id', ''),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'middle_name': '',
        'user_role': user.get('role', ''),
        'user_type': user.get('user_type', '')
    }
    return render_template('Timer.html', **user_data)

@app.route('/camfootage')
@login_required
def camfootage_page():
    return render_template('CamFootage.html')

@app.route('/sidebar')
@login_required
def sidebar_page():
    user = get_current_user()
    
    if not user:
        logger.error("No user found in sidebar")
        return "Error: User not found", 401

    user_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "Unknown User"
    
    # DEBUG: Check what role is actually in the user object
    print(f"DEBUG - User object: {user}")
    print(f"DEBUG - User role: {user.get('role')}")
    print(f"DEBUG - User type: {user.get('user_type')}")
    
    # FIX: Pass the role exactly as it comes from the database
    user_role = user.get('role', '')
    user_type = user.get('user_type', '')
    
    logger.info("=== SIDEBAR DEBUG ===")
    logger.info(f"User ID: {user.get('user_id')}")
    logger.info(f"Name: {user_name}")
    logger.info(f"Role: {user_role}")
    logger.info(f"Type: {user_type}")

    return render_template(
        'sidebar.html',
        first_name=user.get('first_name', ''),
        last_name=user.get('last_name', ''),
        middle_name=user.get('middle_name', ''),
        user_name=user_name,
        user_role=user_role,  # This should be 'super_admin'
        user_type=user_type   # This should be 'admin'
    )

@app.route('/api/get_user_info', methods=['GET'])
@login_required
def get_user_info():
    """Get current user information"""
    try:
        user_id = session.get('user_id')
        user_role = session.get('role')
        faculty_id = session.get('faculty_id')
        
        print(f"Debug get_user_info - user_id: {user_id}, role: {user_role}, faculty_id: {faculty_id}")
        
        if not user_id:
            return jsonify({'success': False, 'message': 'User not logged in'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if user_role == 'faculty':
            # Use faculty_id if available, otherwise use user_id
            actual_faculty_id = faculty_id if faculty_id else user_id
            cursor.execute("""
                SELECT faculty_id as id, first_name, last_name, email, department
                FROM faculty WHERE faculty_id = %s
            """, (actual_faculty_id,))
        elif user_role == 'admin':
            cursor.execute("""
                SELECT admin_id as id, username as first_name, '' as last_name, email, 'Administration' as department
                FROM admin WHERE admin_id = %s
            """, (user_id,))
        else:
            return jsonify({'success': False, 'message': 'Invalid user role'})
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'User not found'})
            
    except Exception as e:
        logger.error(f"Error getting user info: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

@app.route('/summary')
@login_required
def summary_page():
    """Render the summary page with session data"""
    return render_template('Summary.html')

@app.route('/api/summary_data')
@login_required
def get_summary_data():
    """Get complete summary data for the latest session - INCLUDING TEMPORARY STUDENTS"""
    try:
        user_id = session.get('user_id')
        
        with get_db_cursor() as cursor:
            # Get user info
            cursor.execute("""
                SELECT admin_id, first_name, last_name, role, photo_path
                FROM admins WHERE admin_id = %s
            """, (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({
                    'success': False, 
                    'message': 'User not found'
                }), 404
            
            # Get the latest completed session WITH SUBJECT INFORMATION
            cursor.execute("""
                SELECT *, subject_code, subject_name, room 
                FROM attendance_sessions 
                WHERE status = 'completed' 
                ORDER BY ended_at DESC 
                LIMIT 1
            """)
            session_data = cursor.fetchone()
            
            if not session_data:
                return jsonify({
                    'success': False, 
                    'message': 'No completed sessions found'
                }), 404
            
            # ✅ FIXED: Use created_at for start time, ended_at for end time
            started_at = session_data['created_at']  # Use created_at instead of started_at
            ended_at = session_data['ended_at']
            
            print(f"🔍 DEBUG Time Data:")
            print(f"   - Created At: {started_at}")
            print(f"   - Started At: {session_data['started_at']}")
            print(f"   - Ended At: {ended_at}")
            print(f"🔍 DEBUG Subject Data:")
            print(f"   - Subject Code: {session_data.get('subject_code')}")
            print(f"   - Subject Name: {session_data.get('subject_name')}")
            print(f"   - Room: {session_data.get('room')}")
            
            # Get duration from stored duration_time
            duration_time = session_data.get('duration_time', '00:00:00')
            duration_seconds = 0
            if duration_time and isinstance(duration_time, str):
                try:
                    hours, minutes, seconds = map(int, duration_time.split(':'))
                    duration_seconds = hours * 3600 + minutes * 60 + seconds
                    print(f"🔍 DEBUG Parsed duration from string: {duration_time} -> {duration_seconds} seconds")
                except:
                    # Fallback calculation using created_at and ended_at
                    if started_at and ended_at:
                        duration_seconds = int((ended_at - started_at).total_seconds())
                        print(f"🔍 DEBUG Fallback duration calculation: {duration_seconds} seconds")
            else:
                # Calculate from created_at and ended_at
                if started_at and ended_at:
                    duration_seconds = int((ended_at - started_at).total_seconds())
                    print(f"🔍 DEBUG Duration from created/ended: {duration_seconds} seconds")
            
            print(f"🔍 DEBUG Final duration_seconds for frontend: {duration_seconds}")
            
            # ✅ GET SUBJECT INFORMATION FROM SESSION DATA INSTEAD OF SEPARATE QUERY
            subject_code = session_data.get('subject_code', 'IT99')
            subject_name = session_data.get('subject_name', 'AMBUTT UY')
            room = session_data.get('room', 'Unknown Room')
            
            print(f"🔍 DEBUG Using subject from session: {subject_code} - {subject_name} - {room}")
            
            # ✅ FIXED: GET ALL ATTENDANCE RECORDS INCLUDING SUBJECT INFORMATION
            cursor.execute("""
                SELECT 
                    a.student_id,
                    a.name as student_name,
                    a.status,
                    a.timestamp,
                    a.session_id,
                    a.subject_code,  -- ✅ GET SUBJECT INFO FROM ATTENDANCE
                    a.subject_name,  -- ✅ GET SUBJECT INFO FROM ATTENDANCE
                    a.room,          -- ✅ GET ROOM INFO FROM ATTENDANCE
                    s.photo_path,
                    CASE 
                        WHEN a.session_id = 'manual_add' OR a.session_id IS NULL THEN TRUE 
                        ELSE FALSE 
                    END as is_temporary
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE (a.session_id = %s OR a.session_id = 'manual_add' OR a.session_id IS NULL)
                AND a.person_type = 'student'
                AND DATE(a.timestamp) = DATE(%s)
                ORDER BY a.status, a.name
            """, (session_data['session_id'], ended_at))
            all_attendance_records = cursor.fetchall()
            
            print(f"🔍 DEBUG Found {len(all_attendance_records)} attendance records (including temporary and NULL session_id)")
            
            # Debug: Print all found records with subject info
            for i, record in enumerate(all_attendance_records):
                print(f"🔍 DEBUG Record {i}: {record['student_id']} - {record['student_name']} - {record['status']} - Subject: {record['subject_code']} - {record['subject_name']}")
            
            # Create complete student list
            complete_student_list = []
            
            # Add all attendance records (both regular and temporary)
            for record in all_attendance_records:
                # Handle photo path for temporary students
                photo_path = record['photo_path']
                if record['is_temporary']:
                    # For temporary students, try to extract ID from name or use default
                    student_id = record['student_id']
                    if not student_id and 'ID:' in record['student_name']:
                        # Extract ID from name like "Rhodmin Lou Berioso (ID: 2022-091324)"
                        try:
                            student_id = record['student_name'].split('ID:')[-1].split(')')[0].strip()
                        except:
                            student_id = 'temp'
                    
                    photo_path = f"/static/images/student_photos/{student_id}.jpg" if student_id and student_id != 'temp' else '/static/images/default-avatar.jpg'
                else:
                    # For regular students, use their photo or default
                    photo_path = record['photo_path'] or f"/static/images/student_photos/{record['student_id']}.jpg"
                
                complete_student_list.append({
                    'student_id': record['student_id'] or 'temp',
                    'name': record['student_name'],
                    'status': record['status'],
                    'timestamp': record['timestamp'],
                    'photo': photo_path or '/static/images/default-avatar.jpg',
                    'is_temporary': record['is_temporary'],
                    'subject_code': record['subject_code'] or subject_code,  # ✅ ADD SUBJECT INFO
                    'subject_name': record['subject_name'] or subject_name,  # ✅ ADD SUBJECT INFO
                    'room': record['room'] or room  # ✅ ADD ROOM INFO
                })
            
            # Get regular students in section for absent count
            cursor.execute("""
                SELECT student_id, first_name, last_name, photo_path 
                FROM students 
                WHERE year_section LIKE '%4C%'
            """)
            all_section_students = cursor.fetchall()
            
            # Add absent students (only regular students who don't have any attendance record)
            attended_regular_student_ids = [r['student_id'] for r in all_attendance_records if r['student_id'] and not r['is_temporary']]
            
            print(f"🔍 DEBUG Regular students in section: {len(all_section_students)}")
            print(f"🔍 DEBUG Attended regular student IDs: {attended_regular_student_ids}")
            
            absent_count_added = 0
            for student in all_section_students:
                student_id = student['student_id']
                
                if student_id not in attended_regular_student_ids:
                    complete_student_list.append({
                        'student_id': student_id,
                        'name': f"{student['first_name']} {student['last_name']}",
                        'status': 'absent',
                        'timestamp': ended_at,
                        'photo': student['photo_path'] or f"/static/images/student_photos/{student_id}.jpg",
                        'is_temporary': False,
                        'subject_code': subject_code,  # ✅ ADD SUBJECT INFO FOR ABSENT STUDENTS
                        'subject_name': subject_name,  # ✅ ADD SUBJECT INFO FOR ABSENT STUDENTS
                        'room': room  # ✅ ADD ROOM INFO FOR ABSENT STUDENTS
                    })
                    absent_count_added += 1
                    print(f"🔍 DEBUG Added absent student: {student_id} - {student['first_name']} {student['last_name']}")
            
            print(f"🔍 DEBUG Added {absent_count_added} absent students")
            
            # Calculate counts
            present_count = len([s for s in complete_student_list if s['status'] == 'present'])
            late_count = len([s for s in complete_student_list if s['status'] == 'late'])
            absent_count = len([s for s in complete_student_list if s['status'] == 'absent'])
            excused_count = len([s for s in complete_student_list if s['status'] == 'excused'])
            total_students = len(complete_student_list)
            
            # Count temporary students for debugging
            temp_count = len([s for s in complete_student_list if s.get('is_temporary')])
            print(f"🔍 DEBUG Student breakdown: {temp_count} temporary, {total_students - temp_count} regular")
            print(f"🔍 DEBUG Status breakdown: Present: {present_count}, Late: {late_count}, Absent: {absent_count}, Excused: {excused_count}")
            
            # Format course display
            class_name = session_data['class_name']
            program_display = "BSIT"
            
            if 'Information Technology' in class_name:
                program_display = 'BSIT'
            elif 'Computer Science' in class_name:
                program_display = 'BSCS'
            elif 'Associate in Computer Technology' in class_name:
                program_display = 'ACT'
            
            # Extract section
            section_display = "4C"
            if '4th Year' in class_name:
                section_part = class_name.split('4th Year')[-1].strip()
                if section_part:
                    section_display = f"4{section_part[0]}"
            elif '2nd Year' in class_name:
                section_part = class_name.split('2nd Year')[-1].strip()
                if section_part:
                    section_display = f"2{section_part[0]}"
            
            course_section_display = f"{program_display}-{section_display}"
            
            # ✅ FIXED: Use created_at for start time, ended_at for end time
            summary_data = {
                'success': True,
                'session': {
                    'session_id': session_data['session_id'],
                    'class_name': session_data['class_name'],
                    'started_at': started_at.strftime('%Y-%m-%d %I:%M%p') if started_at else '',  # Use created_at
                    'ended_at': ended_at.strftime('%Y-%m-%d %I:%M%p') if ended_at else '',  # Use ended_at
                    'duration_seconds': duration_seconds,
                    'late_threshold_minutes': session_data.get('late_threshold_minutes', 20) or 20,
                    'total_students': total_students,
                    'present_count': present_count,
                    'late_count': late_count,
                    'absent_count': absent_count,
                    'excused_count': excused_count,
                    'subject_code': subject_code,  # ✅ ADD SUBJECT INFO TO SESSION
                    'subject_name': subject_name,  # ✅ ADD SUBJECT INFO TO SESSION
                    'room': room  # ✅ ADD ROOM INFO TO SESSION
                },
                'user': {
                    'name': f"{user['first_name']} {user['last_name']}",
                    'role': user['role'],
                    'username': user['admin_id'],
                    'photo_path': user['photo_path'] or '/static/images/default-avatar.jpg'
                },
                'subject': {
                    'code': subject_code,  # ✅ USE FROM SESSION DATA
                    'name': subject_name,  # ✅ USE FROM SESSION DATA
                    'room': room  # ✅ ADD ROOM INFO
                },
                'course_section': course_section_display,
                'attendance': complete_student_list
            }
            
            print(f"✅ DEBUG Summary with correct times and subject info:")
            print(f"   - Start: {summary_data['session']['started_at']} (from created_at)")
            print(f"   - End: {summary_data['session']['ended_at']} (from ended_at)")
            print(f"   - Duration: {duration_seconds} seconds")
            print(f"   - Total students: {total_students}")
            print(f"   - Subject: {subject_code} - {subject_name}")
            print(f"   - Room: {room}")
            
            return jsonify(summary_data)
            
    except Exception as e:
        print(f"❌ ERROR in get_summary_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error loading summary data: {str(e)}'
        }), 500
    
@app.route('/api/update_attendance', methods=['POST'])
@login_required
def update_attendance():
    """Update attendance status for students"""
    data = request.get_json()
    session_id = data.get('session_id')
    attendance_updates = data.get('attendance_updates', [])
    
    if not session_id:
        return jsonify({'success': False, 'message': 'Missing session_id'}), 400
    
    try:
        with get_db_cursor() as cursor:
            for update in attendance_updates:
                cursor.execute("""
                    UPDATE attendance 
                    SET status = %s 
                    WHERE session_id = %s AND student_id = %s
                """, (update['status'], session_id, update['student_id']))
            
            # Update session counts
            cursor.execute("""
                UPDATE attendance_sessions 
                SET 
                    present_count = (SELECT COUNT(*) FROM attendance WHERE session_id = %s AND status = 'present'),
                    late_count = (SELECT COUNT(*) FROM attendance WHERE session_id = %s AND status = 'late'),
                    absent_count = (SELECT COUNT(*) FROM attendance WHERE session_id = %s AND status = 'absent'),
                    excused_count = (SELECT COUNT(*) FROM attendance WHERE session_id = %s AND status = 'excused')
                WHERE session_id = %s
            """, (session_id, session_id, session_id, session_id, session_id))
            
        return jsonify({'success': True, 'message': 'Attendance updated successfully'})
        
    except Exception as e:
        print(f"Error updating attendance: {e}")
        return jsonify({'success': False, 'message': 'Error updating attendance'}), 500

@app.route('/api/export_csv')
@login_required
def export_csv():
    """Export attendance data as CSV - COMPLETE FIXED VERSION"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'success': False, 'message': 'Missing session_id'}), 400
    
    try:
        with get_db_cursor() as cursor:
            # ✅ FIXED: GET SESSION DATA WITH SUBJECT INFORMATION
            cursor.execute("""
                SELECT class_name, started_at, ended_at, subject_code, subject_name, room 
                FROM attendance_sessions 
                WHERE session_id = %s
            """, (session_id,))
            session_data = cursor.fetchone()
            
            if not session_data:
                return jsonify({'success': False, 'message': 'Session not found'}), 404
            
            class_name = session_data['class_name']
            subject_code = session_data.get('subject_code', 'IT99')
            subject_name = session_data.get('subject_name', 'AMBUTT UY')
            room = session_data.get('room', 'Unknown Room')
            
            print(f"🔍 DEBUG Session Subject Info:")
            print(f"   - Subject Code: {subject_code}")
            print(f"   - Subject Name: {subject_name}")
            print(f"   - Room: {room}")
            
            # ✅ FIXED: MANIPULATE PROGRAM NAME
            program_display = "BSIT"  # Default
            
            if 'Associate in Computer Technology' in class_name:
                program_display = 'ACT'
            elif 'Information Technology' in class_name:
                program_display = 'BSIT'
            elif 'Computer Science' in class_name:
                program_display = 'BSCS'
            elif 'Accountancy' in class_name or 'Accounting' in class_name:
                program_display = 'BSA'
            elif 'Education' in class_name:
                program_display = 'BSE'
            elif 'Engineering' in class_name:
                program_display = 'BSE'
            elif 'Architecture' in class_name:
                program_display = 'BSARCH'
            
            # ✅ FIXED: MANIPULATE SECTION (4th YearC -> 4C)
            section_display = "4C"  # Default
            if '4th Year' in class_name:
                section_part = class_name.split('4th Year')[-1].strip()
                if section_part:
                    section_display = f"4{section_part[0]}"  # Get first character after "4th Year"
            elif '2nd Year' in class_name:
                section_part = class_name.split('2nd Year')[-1].strip()
                if section_part:
                    section_display = f"2{section_part[0]}"
            
            print(f"🔍 DEBUG Program: {program_display}, Section: {section_display}")
            
            # ✅ FIXED: GET ALL STUDENTS INCLUDING ABSENT AND TEMPORARY WITH SUBJECT INFO
            cursor.execute("""
                -- Get regular students with their attendance status (INCLUDING ABSENT)
                SELECT 
                    s.student_id,
                    CONCAT(s.first_name, ' ', s.last_name) as student_name,
                    %s as year_section,  -- Use manipulated section
                    COALESCE(a.status, 'absent') as status,
                    COALESCE(a.timestamp, %s) as attendance_timestamp,
                    'No' as is_temporary,
                    COALESCE(a.subject_code, %s) as subject_code,  -- ✅ GET SUBJECT INFO
                    COALESCE(a.subject_name, %s) as subject_name,  -- ✅ GET SUBJECT INFO
                    COALESCE(a.room, %s) as room                   -- ✅ GET ROOM INFO
                FROM students s
                LEFT JOIN attendance a ON s.student_id = a.student_id AND a.session_id = %s
                WHERE s.year_section LIKE '%%4C%%'  -- Match students in this section
                
                UNION ALL
                
                -- Get temporary students
                SELECT 
                    COALESCE(a.student_id, 'TEMP') as student_id,
                    a.name as student_name,
                    %s as year_section,  -- Use manipulated section
                    a.status,
                    a.timestamp as attendance_timestamp,
                    'Yes' as is_temporary,
                    COALESCE(a.subject_code, %s) as subject_code,  -- ✅ GET SUBJECT INFO
                    COALESCE(a.subject_name, %s) as subject_name,  -- ✅ GET SUBJECT INFO
                    COALESCE(a.room, %s) as room                   -- ✅ GET ROOM INFO
                FROM attendance a
                WHERE a.session_id = 'manual_add'
                AND DATE(a.timestamp) = DATE(%s)
                AND a.person_type = 'student'
                
                ORDER BY status, student_name
            """, (
                section_display, 
                session_data['ended_at'], 
                subject_code, subject_name, room,  # ✅ SUBJECT PARAMS FOR REGULAR STUDENTS
                session_id,
                section_display,
                subject_code, subject_name, room,  # ✅ SUBJECT PARAMS FOR TEMPORARY STUDENTS
                session_data['ended_at']
            ))
            
            records = cursor.fetchall()
            
            if not records:
                return jsonify({'success': False, 'message': 'No student data found'}), 404
            
            print(f"🔍 DEBUG CSV Export: Found {len(records)} total students")
            
            # Debug: Print first few records with subject info
            for i, record in enumerate(records[:3]):
                print(f"🔍 DEBUG Record {i}: {record['student_id']} - {record['student_name']} - {record['status']} - Subject: {record['subject_code']} - {record['subject_name']}")
            
            # Create CSV content
            import csv
            import io
            from datetime import datetime
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # ✅ FIXED: ADD ROOM COLUMN TO HEADERS
            writer.writerow([
                'Student ID', 
                'Student Name', 
                'Status', 
                'Time Recorded', 
                'Subject Code', 
                'Subject Name', 
                'Room',  # ✅ ADD ROOM COLUMN
                'Program', 
                'Section', 
                'Temporary Student'
            ])
            
            # Write data
            for record in records:
                timestamp = record['attendance_timestamp']
                time_recorded = session_data['ended_at'].strftime('%Y-%m-%d %I:%M:%S %p')  # Default to session end time
                
                # For absent students, use session end time
                if record['status'] == 'absent':
                    time_recorded = session_data['ended_at'].strftime('%Y-%m-%d %I:%M:%S %p')
                elif timestamp:
                    # For present/late/excused students, use their actual timestamp
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %I:%M%p')
                            time_recorded = timestamp.strftime('%Y-%m-%d %I:%M:%S %p')
                        except:
                            time_recorded = timestamp
                    else:
                        time_recorded = timestamp.strftime('%Y-%m-%d %I:%M:%S %p')
                
                writer.writerow([
                    record['student_id'],
                    record['student_name'],
                    record['status'].upper(),
                    time_recorded,
                    record['subject_code'] or subject_code,  # ✅ USE RECORD SUBJECT OR FALLBACK
                    record['subject_name'] or subject_name,  # ✅ USE RECORD SUBJECT OR FALLBACK
                    record['room'] or room,  # ✅ USE RECORD ROOM OR FALLBACK
                    program_display,
                    section_display,
                    record['is_temporary']
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            # ✅ FIXED: PROPER FILENAME FORMATTING AND HEADERS
            # Remove spaces and special characters from subject name
            clean_subject_name = subject_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            # Create filename: IT99_AMBUTT_UY_BSIT-4C_attendance_20251027_032654.csv
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subject_code}_{clean_subject_name}_{program_display}-{section_display}_attendance_{timestamp}.csv"
            
            print(f"🔍 DEBUG Final Filename: {filename}")
            print(f"🔍 DEBUG CSV Content Preview: {len(csv_content)} characters")
            
            # Create response with proper headers to prevent caching
            from flask import Response
            response = Response(
                csv_content,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{filename}\"",
                    "Content-Type": "text/csv; charset=utf-8",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
            
            return response
            
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error exporting data: {str(e)}'}), 500
    
@app.route('/schedule')
@login_required
def schedule_page():
    user = g.get('user', {})
    user_data = {
        'user_id': user.get('user_id', ''),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'middle_name': '',
        'user_role': user.get('role', ''),
        'user_type': user.get('user_type', '')
    }
    return render_template('schedule.html', **user_data)

@app.route('/programs')
@login_required
def programs_page():
    user = g.get('user', {})
    user_data = {
        'user_id': user.get('user_id', ''),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'middle_name': user.get('middle_name', ''),
        'user_role': user.get('role', ''),
        'user_type': user.get('user_type', '')
    }
    return render_template('programs.html', **user_data)

@app.route('/classsched')
@login_required
def classsched_page():
    user = g.get('user', {})
    user_data = {
        'user_id': user.get('user_id', ''),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'middle_name': '',
        'user_role': user.get('role', ''),
        'user_type': user.get('user_type', '')
    }
    return render_template('classsched.html', **user_data)

@app.route('/studentreg')
def studentreg_page():
    """Student registration page - requires valid invite token"""
    # Check if user is already logged in
    user = get_current_user()
    if user:
        if user['user_type'] == 'student':
            return redirect(url_for('student_lp_page'))
        else:
            return redirect(url_for('admin_db_page'))
    
    token = request.args.get('token')
    
    if not token:
        return redirect(url_for('login_page'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
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
        
        if invite_type != 'student':
            logger.warning(f"Wrong invite type for student registration: {invite_type}")
            return redirect(url_for('login_page'))
        
        if used == 1 or current_uses >= max_uses:
            logger.warning(f"Used up invite token attempted: {token}")
            return redirect(url_for('login_page'))
            
        if datetime.now() > expires_at:
            logger.warning(f"Expired invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        logger.info(f"Valid student invite token accessed: {token}")
        return render_template('studentreg.html', token=token)
        
    except Exception as e:
        logger.error(f"Error validating invite token: {e}")
        return redirect(url_for('login_page'))

@app.route('/facultyreg')
def faculty_reg_page():
    """Faculty registration page - requires valid invite token"""
    # Check if user is already logged in
    user = get_current_user()
    if user:
        if user['user_type'] == 'student':
            return redirect(url_for('student_lp_page'))
        else:
            return redirect(url_for('admin_db_page'))
    
    token = request.args.get('token')
    
    if not token:
        return redirect(url_for('login_page'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
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
        
        if invite_type != 'faculty':
            logger.warning(f"Wrong invite type for faculty registration: {invite_type}")
            return redirect(url_for('login_page'))
        
        if used == 1 or current_uses >= max_uses:
            logger.warning(f"Used up invite token attempted: {token}")
            return redirect(url_for('login_page'))
            
        if datetime.now() > expires_at:
            logger.warning(f"Expired invite token attempted: {token}")
            return redirect(url_for('login_page'))
        
        logger.info(f"Valid faculty invite token accessed: {token}")
        return render_template('facultyreg.html', token=token)
        
    except Exception as e:
        logger.error(f"Error validating invite token: {e}")
        return redirect(url_for('login_page'))

@app.route('/api/check_session', methods=['GET'])
def check_session():
    """Check if user has active session"""
    user = get_current_user()
    if user:
        return jsonify({
            'logged_in': True,
            'user_type': user['user_type'],
            'name': f"{user['first_name']} {user['last_name']}",
            'redirect_url': '/StudentLP' if user['user_type'] == 'student' else '/AdminDB'
        })
    return jsonify({'logged_in': False})  

@app.route('/subject')
def subject_page():
    return render_template('subject.html')

@app.route('/AdminDB')
@login_required
def admin_db_page():
    user = get_current_user()
    print(f"DEBUG AdminDB - User: {user}")
    
    if not user:
        print("DEBUG AdminDB - No user, redirecting to login")
        return redirect('/login')
    
    if user['user_type'] == 'student':
        print(f"DEBUG AdminDB - Student user, redirecting to StudentLP")
        return redirect('/StudentLP')
    
    print("DEBUG AdminDB - Rendering AdminDB")
    user_data = {
        'user_id': user.get('user_id', ''),
        'first_name': user.get('first_name', ''),
        'last_name': user.get('last_name', ''),
        'middle_name': user.get('middle_name', ''),
        'user_role': user.get('role', ''),
        'user_type': user.get('user_type', '')
    }
    return render_template('AdminDB.html', **user_data)

@app.route('/StudentDB')
@login_required
def student_db_page():
    user_data = get_template_user_data()
    return render_template('StudentDB.html', **user_data)

@app.route('/FacultyDB')
@login_required
def faculty_db_page():
    user_data = get_template_user_data()
    return render_template('FacultyDB.html', **user_data)

@app.route('/settings')
@login_required
def settings_page():
    user_data = get_template_user_data()
    return render_template('settings.html', **user_data)

@app.route('/StudentLP')
@student_login_required  # ADDED: Use the decorator
def student_lp_page():
    user_data = get_template_user_data()  # Use your existing function
    return render_template('StudentLP.html', **user_data)

@app.route('/StudSettings')
@login_required
def student_settings_page():
    user_data = get_template_user_data()
    return render_template('StudSettings.html', **user_data)

@app.route('/StudAttendance')
@login_required
def student_attendance_page():
    user_data = get_template_user_data()
    return render_template('StudAttendance.html', **user_data)

@app.route('/login')
def login_page():
    """Login page - if already logged in, redirect to appropriate page"""
    user = get_current_user()
    if user:
        if user['user_type'] == 'student':
            return redirect('/StudentLP')
        else:
            return redirect('/AdminDB')
    return render_template('login.html')

@app.route('/')
def index():
    """Root route - redirect based on user type"""
    user = get_current_user()
    if not user:
        return redirect('/login')
    
    if user['user_type'] == 'student':
        return redirect('/StudentLP')
    else:
        return redirect('/AdminDB')

@app.route('/video_feed')
def video_feed():
    """Stream video feed with detections - COMPLETE DRAWING"""
    def generate():
        global latest_frame, tracks, locked_tracks, pending_confirmations, stop_flag
        frame_idx = 0
        
        while not stop_flag:
            try:
                if latest_frame is None:
                    time.sleep(0.033)
                    continue
                
                frame = latest_frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Step 1: Face detection and recognition (no drawing)
                refresh_with_detections(frame, rgb, frame_idx)
                
                # Step 2: Body tracking and drawing (draws everything once)
                update_trackers_with_body(rgb, frame, frame_idx)

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
                
                frame_idx += 1
                time.sleep(0.033)
                
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        # Get session token from cookie
        session_token = request.cookies.get('session_token')
        
        # Clear Flask session FIRST
        session.clear()
        
        # Delete session from database if token exists
        if session_token:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM user_sessions WHERE session_token = %s",
                    (session_token,)
                )
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Session token deleted from database: {session_token[:8]}...")
            except Exception as db_error:
                logger.error(f"Error deleting session from database: {db_error}")
        
        # Create response that clears the cookie
        resp = jsonify({
            'success': True, 
            'message': 'Logged out successfully',
            'redirect_url': '/login'
        })
        
        # Clear the session cookie
        resp.set_cookie('session_token', '', expires=0, max_age=0, httponly=True, secure=False, samesite='Strict')
        
        logger.info("User logged out successfully")
        return resp
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'success': False, 
            'message': 'Logout failed',
            'redirect_url': '/login'
        })

@app.route('/logout')
def logout_page():
    """Direct logout route that redirects to login"""
    try:
        # Get session token from cookie
        session_token = request.cookies.get('session_token')
        
        # Clear Flask session FIRST
        session.clear()
        
        # Delete session from database if token exists
        if session_token:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM user_sessions WHERE session_token = %s",
                    (session_token,)
                )
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Direct logout - session token deleted: {session_token[:8]}...")
            except Exception as db_error:
                logger.error(f"Error deleting session in direct logout: {db_error}")
        
        # Create redirect response that clears the cookie
        resp = make_response(redirect('/login'))  # Use direct path instead of url_for
        resp.set_cookie('session_token', '', expires=0, max_age=0, httponly=True, secure=False, samesite='Strict')
        
        logger.info("User logged out via direct logout route")
        return resp
        
    except Exception as e:
        logger.error(f"Direct logout error: {e}")
        # Still redirect to login even if there's an error
        resp = make_response(redirect('/login'))
        resp.set_cookie('session_token', '', expires=0, max_age=0, httponly=True, secure=False, samesite='Strict')
        return resp
    
# ==========================================
# PROGRAMS MANAGEMENT API ROUTES
# ==========================================


    
@app.route('/api/add_academic_year', methods=['POST'])
@login_required
def add_academic_year():
    """Add a new academic year"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        status = data.get('status', 'active')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if already exists
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'This academic year already exists'})
        
        # Insert new academic year
        cursor.execute(
            "INSERT INTO academic_years (program_id, academic_year, status) VALUES (%s, %s, %s)",
            (program_id, academic_year, status)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Academic year added successfully'})
        
    except Exception as e:
        logger.error(f"Error adding academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_academic_year', methods=['POST'])
@login_required
def update_academic_year():
    """Update an academic year"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        status = data.get('status')
        
        if not all([program_id, academic_year, status]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE academic_years SET status = %s WHERE program_id = %s AND academic_year = %s",
            (status, program_id, academic_year)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Academic year updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})

def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not user:
                return redirect(url_for('login_page'))
            
            user_role = user.get('role')
            if user_role not in allowed_roles:
                return jsonify({'success': False, 'message': 'Insufficient permissions'}), 403
            
            g.user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Add this function to pass user data to templates
def get_template_user_data():
    """Get user data for template rendering"""
    user = get_current_user()
    if user:
        return {
            'user_id': user['user_id'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'middle_name': '',  # You might need to fetch this from database
            'user_role': user['role'],
            'user_type': user['user_type']
        }
    return {}

@app.route('/api/get_semesters', methods=['GET'])
@login_required
def get_semesters():
    """Get semesters for a program and academic year"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'Program ID and academic year are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result['academic_year_id']
        
        # Get semesters with better error handling
        cursor.execute("""
            SELECT 
                semester_id,
                semester_number,
                status,
                created_at
            FROM semesters
            WHERE academic_year_id = %s
            ORDER BY 
                CASE semester_number
                    WHEN 'Summer' THEN 1
                    WHEN '1st Semester' THEN 2
                    WHEN '2nd Semester' THEN 3
                    ELSE 4
                END
        """, (academic_year_id,))
        
        semesters = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'semesters': semesters
        })
        
    except Exception as e:
        logger.error(f"Error fetching semesters: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/add_semester', methods=['POST'])
@login_required
def add_semester():
    """Add a semester"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester_number = data.get('semester_number')
        status = data.get('status', 'active')
        
        if not all([program_id, academic_year, semester_number]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result[0]
        
        # Check if semester already exists
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s",
            (academic_year_id, semester_number)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'This semester already exists'})
        
        # Insert semester
        cursor.execute(
            "INSERT INTO semesters (academic_year_id, semester_number, status) VALUES (%s, %s, %s)",
            (academic_year_id, semester_number, status)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Semester added successfully'})
        
    except Exception as e:
        logger.error(f"Error adding semester: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_semester', methods=['POST'])
@login_required
def update_semester():
    """Update semester status"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester_number = data.get('semester_number')
        status = data.get('status')
        
        if not all([program_id, academic_year, semester_number, status]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result[0]
        
        # Update semester
        cursor.execute(
            "UPDATE semesters SET status = %s WHERE academic_year_id = %s AND semester_number = %s",
            (status, academic_year_id, semester_number)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Semester updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating semester: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_semester', methods=['POST'])
@login_required
def delete_semester():
    """Delete a semester"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester_number = data.get('semester_number')
        
        if not all([program_id, academic_year, semester_number]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result[0]
        
        # Delete semester
        cursor.execute(
            "DELETE FROM semesters WHERE academic_year_id = %s AND semester_number = %s",
            (academic_year_id, semester_number)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Semester deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting semester: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_academic_year', methods=['POST'])
@login_required
def delete_academic_year():
    """Delete an academic year"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'Program ID and academic year are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Academic year deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})    

@app.route('/api/add_year_section', methods=['POST'])
@role_required(['super_admin', 'admin', 'moderator'])  # Faculty cannot access
@login_required
def add_year_section():
    """Add a new year section to a program and semester"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester = data.get('semester')
        year_level = data.get('year_level')
        section_name = data.get('section_name', '').strip().upper()
        
        if not all([program_id, academic_year, semester, year_level, section_name]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if not (1 <= int(year_level) <= 4):
            return jsonify({'success': False, 'message': 'Year level must be between 1 and 4'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s AND status = 'active'",
            (program_id, academic_year)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found or inactive'})
        
        academic_year_id = result[0]
        
        # Get semester_id
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s AND status = 'active'",
            (academic_year_id, semester)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found or inactive'})
        
        semester_id = result[0]
        
        # Check if section already exists for THIS SPECIFIC semester and academic year
        cursor.execute(
            """SELECT section_id FROM year_sections 
               WHERE program_id = %s 
               AND academic_year_id = %s
               AND semester_id = %s
               AND year_level = %s 
               AND section_name = %s
               AND status = 'active'""",
            (program_id, academic_year_id, semester_id, year_level, section_name)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Section {section_name} already exists for Year {year_level} in {semester} {academic_year}'
            })
        
        # Insert new section with semester_id
        cursor.execute(
            """INSERT INTO year_sections 
               (program_id, academic_year_id, semester_id, year_level, section_name, status) 
               VALUES (%s, %s, %s, %s, %s, 'active')""",
            (program_id, academic_year_id, semester_id, year_level, section_name)
        )
        
        conn.commit()
        section_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        logger.info(f"Added section {section_name} for {program_id} Year {year_level} - {semester} {academic_year}")
        
        return jsonify({
            'success': True, 
            'message': f'Section {section_name} added successfully to {semester} {academic_year}',
            'section_id': section_id
        })
        
    except mysql.connector.IntegrityError as e:
        logger.error(f"Integrity error adding section: {e}")
        return jsonify({
            'success': False, 
            'message': 'This section already exists for this semester. Please use a different section name.'
        })
    except Exception as e:
        logger.error(f"Error adding year section: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_subjects', methods=['GET'])
@login_required
def get_subjects():
    """Get all subjects for a year section with semester info"""
    try:
        section_id = request.args.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                s.subject_id,
                s.subject_code,
                s.subject_name,
                s.class_type,
                s.units,
                s.status,
                s.created_at,
                sem.semester_number,
                ay.academic_year
            FROM subjects s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN semesters sem ON ys.semester_id = sem.semester_id
            JOIN academic_years ay ON ys.academic_year_id = ay.academic_year_id
            WHERE s.section_id = %s AND s.status = 'active'
            ORDER BY s.subject_code
        """, (section_id,))
        
        subjects = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'subjects': subjects})
        
    except Exception as e:
        logger.error(f"Error fetching subjects: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/add_subject', methods=['POST'])
@login_required
def add_subject():
    """Add a new subject to a year section with semester awareness"""
    try:
        data = request.json
        section_id = data.get('section_id')
        subject_code = data.get('subject_code', '').strip().upper()
        subject_name = data.get('subject_name', '').strip()
        class_type = data.get('class_type', 'lecture')
        units = data.get('units', 3)
        
        if not all([section_id, subject_code, subject_name]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if class_type not in ['lecture', 'laboratory', 'both']:
            return jsonify({'success': False, 'message': 'Invalid class type'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get semester and academic year information for this section
        cursor.execute("""
            SELECT ys.semester_id, s.semester_number, ay.academic_year 
            FROM year_sections ys
            JOIN semesters s ON ys.semester_id = s.semester_id
            JOIN academic_years ay ON ys.academic_year_id = ay.academic_year_id
            WHERE ys.section_id = %s
        """, (section_id,))
        
        section_info = cursor.fetchone()
        if not section_info:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Section not found'})
        
        semester_id, semester_number, academic_year = section_info
        
        # Check if subject code already exists in this EXACT section (same semester and academic year)
        cursor.execute("""
            SELECT subject_id FROM subjects 
            WHERE section_id = %s AND subject_code = %s AND status = 'active'
        """, (section_id, subject_code))
        
        existing_subject = cursor.fetchone()
        if existing_subject:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Subject code {subject_code} already exists in this section for {semester_number} {academic_year}'
            })
        
        # Insert new subject
        cursor.execute(
            """INSERT INTO subjects (section_id, subject_code, subject_name, class_type, units) 
               VALUES (%s, %s, %s, %s, %s)""",
            (section_id, subject_code, subject_name, class_type, units)
        )
        
        conn.commit()
        subject_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        logger.info(f"Added subject {subject_code} to section {section_id} for {semester_number} {academic_year}")
        return jsonify({
            'success': True, 
            'message': f'Subject added successfully to {semester_number} {academic_year}',
            'subject_id': subject_id
        })
        
    except Exception as e:
        logger.error(f"Error adding subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_subject', methods=['POST'])
@login_required
def update_subject():
    """Update an existing subject"""
    try:
        data = request.json
        subject_id = data.get('subject_id')
        subject_code = data.get('subject_code', '').strip().upper()
        subject_name = data.get('subject_name', '').strip()
        class_type = data.get('class_type', 'lecture')
        units = data.get('units', 3)
        
        if not all([subject_id, subject_code, subject_name]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if subject exists
        cursor.execute(
            "SELECT section_id FROM subjects WHERE subject_id = %s",
            (subject_id,)
        )
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Subject not found'})
        
        section_id = result[0]
        
        # Check if new subject code conflicts with another subject in same section
        cursor.execute(
            "SELECT subject_id FROM subjects WHERE section_id = %s AND subject_code = %s AND subject_id != %s",
            (section_id, subject_code, subject_id)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Subject code already exists in this section'})
        
        # Update the subject
        cursor.execute(
            """UPDATE subjects 
               SET subject_code = %s, subject_name = %s, class_type = %s, units = %s, updated_at = NOW()
               WHERE subject_id = %s""",
            (subject_code, subject_name, class_type, units, subject_id)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No changes made'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully updated subject {subject_id}")
        return jsonify({'success': True, 'message': 'Subject updated successfully'})
        
    except mysql.connector.Error as db_error:
        logger.error(f"Database error updating subject: {db_error}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': f'Database error: {str(db_error)}'})
    except Exception as e:
        logger.error(f"Error updating subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_subject', methods=['POST'])
@login_required
def delete_subject():
    """Delete a subject permanently from the database"""
    try:
        subject_id = request.json.get('subject_id')
        
        if not subject_id:
            return jsonify({'success': False, 'message': 'Subject ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, check if subject exists
        cursor.execute(
            "SELECT subject_id FROM subjects WHERE subject_id = %s",
            (subject_id,)
        )
        
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Subject not found'})
        
        # Delete related faculty schedules first (to maintain referential integrity)
        cursor.execute("""
            DELETE fs FROM faculty_schedules fs
            INNER JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            WHERE cs.subject_id = %s
        """, (subject_id,))
        
        deleted_faculty_schedules = cursor.rowcount
        logger.info(f"Deleted {deleted_faculty_schedules} faculty schedule assignments for subject {subject_id}")
        
        # Delete related class schedules
        cursor.execute(
            "DELETE FROM class_schedules WHERE subject_id = %s",
            (subject_id,)
        )
        
        deleted_class_schedules = cursor.rowcount
        logger.info(f"Deleted {deleted_class_schedules} class schedules for subject {subject_id}")
        
        # Finally, delete the subject itself
        cursor.execute(
            "DELETE FROM subjects WHERE subject_id = %s",
            (subject_id,)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Failed to delete subject'})
        
        # Commit all changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully deleted subject {subject_id}")
        return jsonify({
            'success': True, 
            'message': 'Subject and all related schedules deleted successfully'
        })
        
    except mysql.connector.Error as db_error:
        logger.error(f"Database error deleting subject: {db_error}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': f'Database error: {str(db_error)}'})
    except Exception as e:
        logger.error(f"Error deleting subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_year_section', methods=['POST'])
@login_required
def delete_year_section():
    """Delete a year section and all related data permanently"""
    try:
        section_id = request.json.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if section exists
        cursor.execute(
            "SELECT section_id FROM year_sections WHERE section_id = %s",
            (section_id,)
        )
        
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Section not found'})
        
        # Delete in correct order to maintain referential integrity
        
        # 1. Delete faculty schedule assignments
        cursor.execute("""
            DELETE fs FROM faculty_schedules fs
            INNER JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            WHERE cs.section_id = %s
        """, (section_id,))
        logger.info(f"Deleted faculty schedule assignments for section {section_id}")
        
        # 2. Delete class schedules
        cursor.execute(
            "DELETE FROM class_schedules WHERE section_id = %s",
            (section_id,)
        )
        logger.info(f"Deleted class schedules for section {section_id}")
        
        # 3. Delete subjects
        cursor.execute(
            "DELETE FROM subjects WHERE section_id = %s",
            (section_id,)
        )
        logger.info(f"Deleted subjects for section {section_id}")
        
        # 4. Finally delete the year section
        cursor.execute(
            "DELETE FROM year_sections WHERE section_id = %s",
            (section_id,)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Failed to delete section'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully deleted year section {section_id} and all related data")
        return jsonify({
            'success': True, 
            'message': 'Year section and all related data deleted successfully'
        })
        
    except mysql.connector.Error as db_error:
        logger.error(f"Database error deleting section: {db_error}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': f'Database error: {str(db_error)}'})
    except Exception as e:
        logger.error(f"Error deleting year section: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
# ==========================================
# CLASS SCHEDULE MANAGEMENT API ROUTES
# ==========================================

@app.route('/api/get_schedules', methods=['GET'])
@login_required
@role_required(['super_admin', 'admin', 'moderator'])  # All admin roles can access
def get_schedules():
    """Get all class schedules with filtering options"""
    try:
        program_id = request.args.get('program_id')
        section_id = request.args.get('section_id')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Base query
        query = """
            SELECT 
                cs.schedule_id,
                cs.day_of_week,
                cs.start_time,
                cs.end_time,
                cs.room,
                cs.class_type,
                s.subject_code,
                s.subject_name,
                s.units,
                ys.year_level,
                ys.section_name,
                p.program_id,
                p.program_name
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE cs.status = 'active' AND s.status = 'active' AND ys.status = 'active'
        """
        
        params = []
        
        if program_id:
            query += " AND p.program_id = %s"
            params.append(program_id)
            
        if section_id:
            query += " AND ys.section_id = %s"
            params.append(section_id)
            
        query += " ORDER BY p.program_id, ys.year_level, ys.section_name, cs.day_of_week, cs.start_time"
        
        cursor.execute(query, params)
        schedules = cursor.fetchall()
        
        # Format time fields
        for schedule in schedules:
            if schedule['start_time']:
                schedule['start_time'] = str(schedule['start_time'])
            if schedule['end_time']:
                schedule['end_time'] = str(schedule['end_time'])
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'schedules': schedules})
        
    except Exception as e:
        logger.error(f"Error fetching schedules: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/add_schedule', methods=['POST'])
@login_required
def add_schedule():
    """Add a new class schedule"""
    try:
        data = request.json
        subject_id = data.get('subject_id')
        section_id = data.get('section_id')
        class_type = data.get('class_type')
        day_of_week = data.get('day_of_week')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        room = data.get('room')
        
        if not all([subject_id, section_id, class_type, day_of_week, start_time, end_time, room]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        # Validate class type matches subject
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT class_type FROM subjects WHERE subject_id = %s AND section_id = %s",
            (subject_id, section_id)
        )
        subject = cursor.fetchone()
        
        if not subject:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid subject or section'})
        
        # Check if class type is valid for this subject
        subject_class_type = subject['class_type']
        if subject_class_type != 'both' and subject_class_type != class_type:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': f'This subject only supports {subject_class_type} classes'})
        
        # Check for room conflicts
        cursor.execute("""
            SELECT s.subject_code, ys.year_level, ys.section_name
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            WHERE cs.room = %s 
            AND cs.day_of_week = %s
            AND cs.status = 'active'
            AND (
                (cs.start_time <= %s AND cs.end_time > %s) OR
                (cs.start_time < %s AND cs.end_time >= %s) OR
                (cs.start_time >= %s AND cs.end_time <= %s)
            )
        """, (room, day_of_week, start_time, start_time, end_time, end_time, start_time, end_time))
        
        conflict = cursor.fetchone()
        if conflict:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Room {room} is already booked on {day_of_week} at this time for {conflict["subject_code"]} ({conflict["year_level"]}-{conflict["section_name"]})'
            })
        
        # Insert schedule
        cursor.execute("""
            INSERT INTO class_schedules 
            (subject_id, section_id, class_type, day_of_week, start_time, end_time, room)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (subject_id, section_id, class_type, day_of_week, start_time, end_time, room))
        
        conn.commit()
        schedule_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Schedule added successfully',
            'schedule_id': schedule_id
        })
        
    except Exception as e:
        logger.error(f"Error adding schedule: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bulk_add_schedules', methods=['POST'])
@login_required
def bulk_add_schedules():
    """Add multiple schedules at once"""
    try:
        schedules = request.json.get('schedules', [])
        
        if not schedules:
            return jsonify({'success': False, 'message': 'No schedules provided'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        added_count = 0
        errors = []
        
        for idx, schedule in enumerate(schedules):
            try:
                # Validate required fields
                required = ['subject_id', 'section_id', 'class_type', 'day_of_week', 'start_time', 'end_time', 'room']
                if not all(schedule.get(field) for field in required):
                    errors.append(f"Schedule {idx + 1}: Missing required fields")
                    continue
                
                # Check for room conflicts
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM class_schedules cs
                    WHERE cs.room = %s 
                    AND cs.day_of_week = %s
                    AND cs.status = 'active'
                    AND (
                        (cs.start_time <= %s AND cs.end_time > %s) OR
                        (cs.start_time < %s AND cs.end_time >= %s) OR
                        (cs.start_time >= %s AND cs.end_time <= %s)
                    )
                """, (
                    schedule['room'], 
                    schedule['day_of_week'],
                    schedule['start_time'], schedule['start_time'],
                    schedule['end_time'], schedule['end_time'],
                    schedule['start_time'], schedule['end_time']
                ))
                
                if cursor.fetchone()['count'] > 0:
                    errors.append(f"Schedule {idx + 1}: Room conflict on {schedule['day_of_week']}")
                    continue
                
                # Insert schedule
                cursor.execute("""
                    INSERT INTO class_schedules 
                    (subject_id, section_id, class_type, day_of_week, start_time, end_time, room)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    schedule['subject_id'],
                    schedule['section_id'],
                    schedule['class_type'],
                    schedule['day_of_week'],
                    schedule['start_time'],
                    schedule['end_time'],
                    schedule['room']
                ))
                
                added_count += 1
                
            except Exception as e:
                errors.append(f"Schedule {idx + 1}: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        message = f'Added {added_count} schedule(s) successfully'
        if errors:
            message += f'. {len(errors)} failed: {"; ".join(errors[:3])}'
            if len(errors) > 3:
                message += f' and {len(errors) - 3} more...'
        
        return jsonify({
            'success': True,
            'message': message,
            'added_count': added_count,
            'error_count': len(errors),
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Error bulk adding schedules: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_schedule', methods=['POST'])
@login_required
def update_schedule():
    """Update an existing schedule"""
    try:
        data = request.json
        schedule_id = data.get('schedule_id')
        
        if not schedule_id:
            return jsonify({'success': False, 'message': 'Schedule ID is required'})
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        for field in ['class_type', 'day_of_week', 'start_time', 'end_time', 'room']:
            if field in data:
                update_fields.append(f"{field} = %s")
                params.append(data[field])
        
        if not update_fields:
            return jsonify({'success': False, 'message': 'No fields to update'})
        
        params.append(schedule_id)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = f"UPDATE class_schedules SET {', '.join(update_fields)} WHERE schedule_id = %s"
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Schedule not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Schedule updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating schedule: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_schedule', methods=['POST'])
@login_required
def delete_schedule():
    """Delete a schedule (soft delete)"""
    try:
        schedule_id = request.json.get('schedule_id')
        
        if not schedule_id:
            return jsonify({'success': False, 'message': 'Schedule ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE class_schedules SET status = 'inactive' WHERE schedule_id = %s",
            (schedule_id,)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Schedule not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Schedule deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting schedule: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/check_room_availability', methods=['POST'])
@login_required
def check_room_availability():
    """Check if a room is available at a specific time"""
    try:
        data = request.json
        room = data.get('room')
        day_of_week = data.get('day_of_week')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        exclude_schedule_id = data.get('exclude_schedule_id')  # For updates
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT cs.schedule_id, s.subject_code, s.subject_name, 
                   ys.year_level, ys.section_name,
                   cs.start_time, cs.end_time
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            WHERE cs.room = %s 
            AND cs.day_of_week = %s
            AND cs.status = 'active'
            AND (
                (cs.start_time <= %s AND cs.end_time > %s) OR
                (cs.start_time < %s AND cs.end_time >= %s) OR
                (cs.start_time >= %s AND cs.end_time <= %s)
            )
        """
        
        params = [room, day_of_week, start_time, start_time, end_time, end_time, start_time, end_time]
        
        if exclude_schedule_id:
            query += " AND cs.schedule_id != %s"
            params.append(exclude_schedule_id)
        
        cursor.execute(query, params)
        conflicts = cursor.fetchall()
        
        # Format time fields
        for conflict in conflicts:
            if conflict['start_time']:
                conflict['start_time'] = str(conflict['start_time'])
            if conflict['end_time']:
                conflict['end_time'] = str(conflict['end_time'])
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'available': len(conflicts) == 0,
            'conflicts': conflicts
        })
        
    except Exception as e:
        logger.error(f"Error checking room availability: {e}")
        return jsonify({'success': False, 'message': str(e)})    
    
# ==========================================
# FACULTY SCHEDULE ASSIGNMENT API ROUTES
# ==========================================

@app.route('/api/get_unassigned_faculty', methods=['GET'])
@login_required
def get_unassigned_faculty():
    """Get faculty members who don't have any active schedules assigned"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                f.faculty_id,
                f.first_name,
                f.last_name,
                f.department,
                f.designation
            FROM faculty f
            WHERE f.status = 'active'
            AND NOT EXISTS (
                SELECT 1 
                FROM faculty_schedules fs 
                WHERE fs.faculty_id = f.faculty_id 
                AND fs.status = 'active'
            )
            ORDER BY f.last_name, f.first_name
        """)
        
        faculty = cursor.fetchall()
        unassigned_count = len(faculty)
        
        for f in faculty:
            f['full_name'] = f"{f['first_name']} {f['last_name']}"
        
        cursor.close()
        conn.close()
        
        logger.info(f"Found {unassigned_count} unassigned faculty member(s)")
        return jsonify({
            'success': True,
            'faculty': faculty,
            'unassigned_count': unassigned_count
        })
        
    except Exception as e:
        logger.error(f"Error fetching unassigned faculty: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_all_faculty', methods=['GET'])
@login_required
def get_all_faculty():
    """Get all active faculty members for dropdown"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                faculty_id,
                first_name,
                last_name,
                department,
                designation
            FROM faculty
            WHERE status = 'active'
            ORDER BY last_name, first_name
        """)
        
        faculty = cursor.fetchall()
        
        for f in faculty:
            f['full_name'] = f"{f['first_name']} {f['last_name']}"
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'faculty': faculty})
        
    except Exception as e:
        logger.error(f"Error fetching faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_class_schedules_for_section', methods=['GET'])
@login_required
def get_class_schedules_for_section():
    """Get all class schedules for a specific section with assignment status"""
    try:
        section_id = request.args.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                cs.schedule_id,
                cs.day_of_week,
                cs.start_time,
                cs.end_time,
                cs.room,
                cs.class_type,
                s.subject_id,
                s.subject_code,
                s.subject_name,
                s.units,
                ys.year_level,
                ys.section_name,
                p.program_id,
                p.program_name,
                fs.faculty_schedule_id,
                fs.faculty_id,
                CONCAT(f.first_name, ' ', f.last_name) as faculty_name
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            LEFT JOIN faculty_schedules fs ON cs.schedule_id = fs.schedule_id AND fs.status = 'active'
            LEFT JOIN faculty f ON fs.faculty_id = f.faculty_id
            WHERE cs.section_id = %s AND cs.status = 'active'
            ORDER BY 
                FIELD(cs.day_of_week, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'),
                cs.start_time
        """, (section_id,))
        
        schedules = cursor.fetchall()
        
        # Format time fields
        for schedule in schedules:
            if schedule['start_time']:
                schedule['start_time'] = str(schedule['start_time'])
            if schedule['end_time']:
                schedule['end_time'] = str(schedule['end_time'])
            schedule['is_assigned'] = schedule['faculty_id'] is not None
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'schedules': schedules})
        
    except Exception as e:
        logger.error(f"Error fetching class schedules: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/assign_faculty_to_schedule', methods=['POST'])
@login_required
def assign_faculty_to_schedule():
    """Assign a faculty member to one or more class schedules"""
    try:
        data = request.json
        faculty_id = data.get('faculty_id')
        schedule_ids = data.get('schedule_ids', [])
        
        if not faculty_id or not schedule_ids:
            return jsonify({'success': False, 'message': 'Faculty ID and schedule IDs are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check for time conflicts
        cursor.execute("""
            SELECT 
                cs1.schedule_id,
                cs1.day_of_week,
                cs1.start_time,
                cs1.end_time,
                s.subject_code,
                s.subject_name
            FROM class_schedules cs1
            JOIN subjects s ON cs1.subject_id = s.subject_id
            WHERE cs1.schedule_id IN (%s)
        """ % ','.join(['%s'] * len(schedule_ids)), schedule_ids)
        
        new_schedules = cursor.fetchall()
        
        # Get existing faculty schedules
        cursor.execute("""
            SELECT 
                cs.schedule_id,
                cs.day_of_week,
                cs.start_time,
                cs.end_time,
                s.subject_code,
                s.subject_name
            FROM faculty_schedules fs
            JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            JOIN subjects s ON cs.subject_id = s.subject_id
            WHERE fs.faculty_id = %s AND fs.status = 'active' AND cs.status = 'active'
        """, (faculty_id,))
        
        existing_schedules = cursor.fetchall()
        
        # Check for conflicts
        conflicts = []
        for new_sched in new_schedules:
            for exist_sched in existing_schedules:
                if new_sched['day_of_week'] == exist_sched['day_of_week']:
                    # Convert time strings to comparable format
                    new_start = datetime.strptime(str(new_sched['start_time']), '%H:%M:%S').time()
                    new_end = datetime.strptime(str(new_sched['end_time']), '%H:%M:%S').time()
                    exist_start = datetime.strptime(str(exist_sched['start_time']), '%H:%M:%S').time()
                    exist_end = datetime.strptime(str(exist_sched['end_time']), '%H:%M:%S').time()
                    
                    # Check for overlap
                    if not (new_end <= exist_start or new_start >= exist_end):
                        conflicts.append({
                            'day': new_sched['day_of_week'],
                            'new_subject': new_sched['subject_code'],
                            'existing_subject': exist_sched['subject_code'],
                            'time': f"{new_start} - {new_end}"
                        })
        
        if conflicts:
            conflict_msg = '; '.join([
                f"{c['day']} {c['time']}: {c['new_subject']} conflicts with {c['existing_subject']}"
                for c in conflicts
            ])
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Schedule conflicts detected: {conflict_msg}',
                'conflicts': conflicts
            })
        
        # Assign schedules
        assigned_count = 0
        for schedule_id in schedule_ids:
            try:
                cursor.execute("""
                    INSERT INTO faculty_schedules (faculty_id, schedule_id)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE status = 'active', updated_at = NOW()
                """, (faculty_id, schedule_id))
                assigned_count += 1
            except Exception as e:
                logger.warning(f"Failed to assign schedule {schedule_id}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully assigned {assigned_count} schedule(s) to faculty member',
            'assigned_count': assigned_count
        })
        
    except Exception as e:
        logger.error(f"Error assigning faculty to schedule: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_faculty_schedules_for_timer', methods=['GET'])
@login_required
def get_faculty_schedules_for_timer():
    """Get schedules for the logged-in faculty member for timer"""
    try:
        user_id = session.get('user_id')
        user_type = session.get('user_type')
        
        if not user_id or not user_type:
            return jsonify({'success': False, 'message': 'User not logged in'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get logged in user's name
        logged_in_user = ""
        if user_type == 'admin':
            cursor.execute("SELECT first_name, last_name, middle_name FROM admins WHERE admin_id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                logged_in_user = f"{user_data['first_name']} {user_data['last_name']}"
                if user_data['middle_name']:
                    logged_in_user = f"{user_data['first_name']} {user_data['middle_name']} {user_data['last_name']}"
        elif user_type == 'faculty':
            cursor.execute("SELECT first_name, last_name, middle_name FROM faculty WHERE faculty_id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                logged_in_user = f"{user_data['first_name']} {user_data['last_name']}"
                if user_data['middle_name']:
                    logged_in_user = f"{user_data['first_name']} {user_data['middle_name']} {user_data['last_name']}"

        if user_type == 'admin':
            # Admin can see all schedules
            cursor.execute("""
                SELECT DISTINCT
                    cs.schedule_id,
                    cs.day_of_week,
                    cs.class_type,
                    s.subject_code,
                    s.subject_name,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM class_schedules cs
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN faculty_schedules fs ON cs.schedule_id = fs.schedule_id
                LEFT JOIN faculty f ON fs.faculty_id = f.faculty_id
                WHERE cs.status = 'active'
                ORDER BY s.subject_code
            """)
        else:
            # Faculty member sees only their schedules
            cursor.execute("""
                SELECT 
                    cs.schedule_id,
                    cs.day_of_week,
                    cs.class_type,
                    s.subject_code,
                    s.subject_name,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM faculty_schedules fs
                JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                JOIN faculty f ON fs.faculty_id = f.faculty_id
                WHERE fs.faculty_id = %s AND cs.status = 'active'
                ORDER BY s.subject_code
            """, (user_id,))
        
        schedules = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert any potential timedelta objects to strings
        for schedule in schedules:
            for key, value in schedule.items():
                if isinstance(value, timedelta):
                    schedule[key] = str(value)
        
        return jsonify({
            'success': True,
            'schedules': schedules,
            'logged_in_user': logged_in_user
        })
        
    except Exception as e:
        logger.error(f"Error getting faculty schedules for timer: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_faculty_schedule', methods=['GET'])
@login_required
def get_faculty_schedule():
    """Get all schedules assigned to a specific faculty member - FIXED VERSION"""
    try:
        faculty_id = request.args.get('faculty_id')
        
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # First check if faculty has any assignments
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM faculty_schedules 
            WHERE faculty_id = %s AND status = 'active'
        """, (faculty_id,))
        
        count_result = cursor.fetchone()
        if not count_result or count_result['count'] == 0:
            cursor.close()
            conn.close()
            return jsonify({
                'success': True,
                'schedules': [],
                'subjects': [],
                'message': 'No schedules assigned to this faculty member'
            })
        
        # Get detailed schedule information
        cursor.execute("""
            SELECT 
                cs.schedule_id,
                cs.day_of_week,
                cs.start_time,
                cs.end_time,
                cs.room,
                cs.class_type,
                s.subject_id,
                s.subject_code,
                s.subject_name,
                s.units,
                ys.year_level,
                ys.section_name,
                p.program_name,
                p.program_id,
                fs.faculty_schedule_id,
                sem.semester_number,
                ay.academic_year
            FROM faculty_schedules fs
            INNER JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            INNER JOIN subjects s ON cs.subject_id = s.subject_id
            INNER JOIN year_sections ys ON cs.section_id = ys.section_id
            INNER JOIN programs p ON ys.program_id = p.program_id
            INNER JOIN semesters sem ON ys.semester_id = sem.semester_id
            INNER JOIN academic_years ay ON ys.academic_year_id = ay.academic_year_id
            WHERE fs.faculty_id = %s 
            AND fs.status = 'active'
            AND cs.status = 'active'
            AND s.status = 'active'
            AND ys.status = 'active'
            ORDER BY 
                FIELD(cs.day_of_week, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'),
                cs.start_time
        """, (faculty_id,))
        
        schedules = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if not schedules:
            return jsonify({
                'success': True,
                'schedules': [],
                'subjects': [],
                'message': 'No active schedules found for this faculty member'
            })
        
        # Format time fields
        for schedule in schedules:
            if schedule.get('start_time'):
                schedule['start_time'] = str(schedule['start_time'])
            if schedule.get('end_time'):
                schedule['end_time'] = str(schedule['end_time'])
        
        # Group by subject
        subjects = {}
        for schedule in schedules:
            subject_key = f"{schedule['subject_code']}_{schedule['section_name']}_{schedule['semester_number']}"
            
            if subject_key not in subjects:
                subjects[subject_key] = {
                    'subject_id': schedule['subject_id'],
                    'subject_code': schedule['subject_code'],
                    'subject_name': schedule['subject_name'],
                    'program': schedule['program_name'],
                    'year_level': schedule['year_level'],
                    'section': schedule['section_name'],
                    'semester': schedule['semester_number'],
                    'academic_year': schedule['academic_year'],
                    'meetings': []
                }
            
            subjects[subject_key]['meetings'].append({
                'schedule_id': schedule['schedule_id'],
                'day': schedule['day_of_week'],
                'start_time': schedule['start_time'],
                'end_time': schedule['end_time'],
                'room': schedule['room'],
                'class_type': schedule['class_type']
            })
        
        subjects_list = list(subjects.values())
        
        logger.info(f"Found {len(subjects_list)} subjects with {len(schedules)} meetings for faculty {faculty_id}")
        
        return jsonify({
            'success': True,
            'schedules': schedules,
            'subjects': subjects_list
        })
        
    except Exception as e:
        logger.error(f"Error fetching faculty schedule: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'message': str(e),
            'schedules': [],
            'subjects': []
        })

@app.route('/api/unassign_faculty_from_schedule', methods=['POST'])
@login_required
def unassign_faculty_from_schedule():
    """Remove faculty assignment from schedule(s) - FIXED VERSION"""
    try:
        data = request.json
        faculty_id = data.get('faculty_id')
        schedule_ids = data.get('schedule_ids', [])
        
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if schedule_ids:
            # Unassign specific schedules
            placeholders = ','.join(['%s'] * len(schedule_ids))
            query = f"""
                DELETE FROM faculty_schedules 
                WHERE faculty_id = %s AND schedule_id IN ({placeholders})
            """
            cursor.execute(query, [faculty_id] + schedule_ids)
        else:
            # Unassign all schedules for this faculty - HARD DELETE
            cursor.execute(
                "DELETE FROM faculty_schedules WHERE faculty_id = %s",
                (faculty_id,)
            )
        
        affected_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Unassigned {affected_rows} schedule(s)',
            'affected_rows': affected_rows
        })
        
    except Exception as e:
        logger.error(f"Error unassigning faculty: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_sections', methods=['GET'])
@login_required
def get_sections():
    """Get sections based on program and year level"""
    try:
        program_id = request.args.get('program_id')
        year_level = request.args.get('year_level')
        
        if not program_id or not year_level:
            return jsonify({'success': False, 'message': 'Program ID and year level are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                section_id,
                section_name,
                year_level,
                program_id
            FROM year_sections
            WHERE program_id = %s AND year_level = %s AND status = 'active'
            ORDER BY section_name
        """, (program_id, year_level))
        
        sections = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'sections': sections})
        
    except Exception as e:
        logger.error(f"Error fetching sections: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_faculty_with_schedules', methods=['GET'])
@login_required
def get_faculty_with_schedules():
    """Get all faculty members with their schedule counts - FIXED VERSION"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                f.faculty_id,
                CONCAT(f.first_name, ' ', f.last_name) as full_name,
                f.department,
                f.designation,
                COUNT(DISTINCT fs.schedule_id) as meeting_count,
                COUNT(DISTINCT s.subject_id) as subject_count
            FROM faculty f
            LEFT JOIN faculty_schedules fs ON f.faculty_id = fs.faculty_id
            LEFT JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id AND cs.status = 'active'
            LEFT JOIN subjects s ON cs.subject_id = s.subject_id AND s.status = 'active'
            WHERE f.status = 'active'
            GROUP BY f.faculty_id
            ORDER BY f.last_name, f.first_name
        """)
        
        faculty = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'faculty': faculty})
        
    except Exception as e:
        logger.error(f"Error fetching faculty with schedules: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_academic_years_for_program', methods=['GET'])
@login_required
def get_academic_years_for_program():
    """Get academic years for a program with section and subject counts - SHOW ALL YEARS"""
    try:
        program_id = request.args.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get ALL academic years for this program (don't filter by status)
        cursor.execute("""
            SELECT 
                ay.academic_year,
                ay.academic_year_id,
                ay.status,
                COUNT(DISTINCT ys.section_id) as section_count,
                COUNT(DISTINCT sub.subject_id) as subject_count
            FROM academic_years ay
            LEFT JOIN year_sections ys ON ay.academic_year_id = ys.academic_year_id AND ys.status = 'active'
            LEFT JOIN subjects sub ON ys.section_id = sub.section_id AND sub.status = 'active'
            WHERE ay.program_id = %s
            GROUP BY ay.academic_year_id, ay.academic_year, ay.status
            ORDER BY ay.academic_year DESC
        """, (program_id,))
        
        academic_years = cursor.fetchall()
        
        # Find the current active academic year (should be only one)
        current_year = None
        for year in academic_years:
            if year['status'] == 'active':
                current_year = year['academic_year']
                break
        
        cursor.close()
        conn.close()
        
        # Format response - include ALL years
        formatted_years = []
        for year in academic_years:
            formatted_years.append({
                'academic_year': year['academic_year'],
                'academic_year_id': year['academic_year_id'],
                'section_count': year['section_count'] or 0,
                'subject_count': year['subject_count'] or 0,
                'is_current': year['academic_year'] == current_year
            })
        
        return jsonify({
            'success': True,
            'academic_years': formatted_years,
            'current_year': current_year
        })
        
    except Exception as e:
        logger.error(f"Error fetching academic years for program: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/set_active_academic_year', methods=['POST'])
@login_required
def set_active_academic_year():
    """Set an academic year as active for a program - PROPERLY MANAGE ACTIVE STATUS"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, set ALL academic years for this program to inactive
        cursor.execute(
            "UPDATE academic_years SET status = 'inactive' WHERE program_id = %s",
            (program_id,)
        )
        
        # Then set ONLY the selected academic year to active
        cursor.execute(
            "UPDATE academic_years SET status = 'active' WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully set {academic_year} as active academic year'
        })
        
    except Exception as e:
        logger.error(f"Error setting active academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/get_academic_years', methods=['GET'])
@login_required
def get_academic_years():
    """Get academic years for a program (used when viewing program details)"""
    try:
        program_id = request.args.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                academic_year_id,
                academic_year,
                status,
                is_current,
                created_at
            FROM academic_years
            WHERE program_id = %s
            ORDER BY academic_year DESC
        """, (program_id,))
        
        academic_years = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'academic_years': academic_years
        })
        
    except Exception as e:
        logger.error(f"Error fetching academic years: {e}")
        return jsonify({'success': False, 'message': str(e)})    
    
@app.route('/api/get_year_sections', methods=['GET'])
@login_required
def get_year_sections():
    """Get all year sections for a program - filtered by semester"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        
        if not all([program_id, academic_year, semester]):
            return jsonify({'success': False, 'message': 'Program ID, academic year, and semester are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result['academic_year_id']
        
        # Get semester_id
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s",
            (academic_year_id, semester)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        semester_id = result['semester_id']
        
        # Get sections for this specific semester
        query = """
            SELECT 
                ys.section_id,
                ys.year_level,
                ys.section_name,
                ys.status,
                COUNT(DISTINCT s.subject_id) as subject_count,
                (SELECT COUNT(*) FROM students st 
                 WHERE st.year_section = CONCAT(ys.year_level, '-', ys.section_name) 
                 AND st.course LIKE CONCAT('%', p.program_name, '%')
                 AND st.status = 'active') as student_count
            FROM year_sections ys
            JOIN programs p ON ys.program_id = p.program_id
            LEFT JOIN subjects s ON ys.section_id = s.section_id AND s.status = 'active'
            WHERE ys.program_id = %s 
            AND ys.semester_id = %s  # This is the key filter
            AND ys.status = 'active'
            GROUP BY ys.section_id
            ORDER BY ys.year_level, ys.section_name
        """
        
        cursor.execute(query, (program_id, semester_id))
        sections = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'sections': sections})
        
    except Exception as e:
        logger.error(f"Error fetching year sections: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/check_database_tables', methods=['GET'])
def check_database_tables():
    """Check if required database tables exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        tables_to_check = ['programs', 'academic_years', 'semesters', 'year_sections', 'subjects']
        table_status = {}
        
        for table in tables_to_check:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            table_status[table] = cursor.fetchone() is not None
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'tables': table_status,
            'all_tables_exist': all(table_status.values())
        })
        
    except Exception as e:
        logger.error(f"Error checking database tables: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/initialize_default_data', methods=['POST'])
@login_required
@role_required(['super_admin'])
def initialize_default_data():
    """Initialize default programs and academic years if they don't exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if programs exist
        cursor.execute("SELECT COUNT(*) as count FROM programs")
        program_count = cursor.fetchone()[0]
        
        if program_count == 0:
            # Insert default programs
            default_programs = [
                ('CS', 'Bachelor of Science in Computer Science', 'College of Computing Studies'),
                ('IT', 'Bachelor of Science in Information Technology', 'College of Computing Studies'),
                ('ACT', 'Associate in Computer Technology', 'College of Computing Studies')
            ]
            
            cursor.executemany(
                "INSERT INTO programs (program_id, program_name, department, status) VALUES (%s, %s, %s, 'active')",
                default_programs
            )
            logger.info("Added default programs")
        
        # Check if academic years exist
        cursor.execute("SELECT COUNT(*) as count FROM academic_years")
        year_count = cursor.fetchone()[0]
        
        if year_count == 0:
            # Get a program ID to associate with
            cursor.execute("SELECT program_id FROM programs LIMIT 1")
            program_result = cursor.fetchone()
            
            if program_result:
                program_id = program_result[0]
                current_year = datetime.now().year
                academic_year = f"{current_year}-{current_year + 1}"
                
                # Insert default academic year
                cursor.execute(
                    "INSERT INTO academic_years (program_id, academic_year, is_current, status) VALUES (%s, %s, TRUE, 'active')",
                    (program_id, academic_year)
                )
                logger.info(f"Added default academic year: {academic_year}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Default data initialized successfully'
        })
        
    except Exception as e:
        logger.error(f"Error initializing default data: {e}")
        return jsonify({'success': False, 'message': str(e)})    

@app.route('/api/get_programs', methods=['GET'])
@login_required
def get_programs():
    """Get all programs with their statistics per semester - fixed version"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Simple query first to check if we can get basic program data
        cursor.execute("""
            SELECT 
                program_id,
                program_name,
                department,
                status,
                created_at
            FROM programs 
            WHERE status = 'active'
            ORDER BY program_name
        """)
        
        programs = cursor.fetchall()
        
        # If no programs found, return empty array but success
        if not programs:
            cursor.close()
            conn.close()
            return jsonify({
                'success': True, 
                'programs': [],
                'message': 'No programs found'
            })
        
        # Get all active semesters
        cursor.execute("""
            SELECT DISTINCT semester_number 
            FROM semesters 
            WHERE status = 'active'
            ORDER BY 
                CASE semester_number
                    WHEN 'Summer' THEN 1
                    WHEN '1st Semester' THEN 2
                    WHEN '2nd Semester' THEN 3
                    ELSE 4
                END
        """)
        semesters = [s['semester_number'] for s in cursor.fetchall()]
        
        # For each program, get statistics per semester
        for program in programs:
            program_id = program['program_id']
            program['semesters'] = {}
            
            for semester in semesters:
                try:
                    # Count sections for this program and semester
                    cursor.execute("""
                        SELECT COUNT(DISTINCT ys.section_id) as section_count
                        FROM year_sections ys
                        JOIN semesters s ON ys.semester_id = s.semester_id
                        WHERE ys.program_id = %s 
                        AND s.semester_number = %s
                        AND ys.status = 'active'
                        AND s.status = 'active'
                    """, (program_id, semester))
                    
                    section_result = cursor.fetchone()
                    section_count = section_result['section_count'] if section_result else 0
                    
                    # Count subjects for this program and semester
                    cursor.execute("""
                        SELECT COUNT(DISTINCT sub.subject_id) as subject_count
                        FROM subjects sub
                        JOIN year_sections ys ON sub.section_id = ys.section_id
                        JOIN semesters s ON ys.semester_id = s.semester_id
                        WHERE ys.program_id = %s 
                        AND s.semester_number = %s
                        AND sub.status = 'active'
                        AND ys.status = 'active'
                        AND s.status = 'active'
                    """, (program_id, semester))
                    
                    subject_result = cursor.fetchone()
                    subject_count = subject_result['subject_count'] if subject_result else 0
                    
                    program['semesters'][semester] = {
                        'section_count': section_count,
                        'subject_count': subject_count
                    }
                    
                except Exception as e:
                    logger.warning(f"Error counting for program {program_id} semester {semester}: {e}")
                    program['semesters'][semester] = {
                        'section_count': 0,
                        'subject_count': 0
                    }
            
            # Also get total counts (across all semesters)
            try:
                # Total sections
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM year_sections 
                    WHERE program_id = %s AND status = 'active'
                """, (program_id,))
                result = cursor.fetchone()
                program['total_sections'] = result['count'] if result else 0
            except Exception as e:
                logger.warning(f"Error counting total sections for program {program_id}: {e}")
                program['total_sections'] = 0
            
            try:
                # Total subjects
                cursor.execute("""
                    SELECT COUNT(DISTINCT s.subject_id) as count
                    FROM subjects s
                    INNER JOIN year_sections ys ON s.section_id = ys.section_id
                    WHERE ys.program_id = %s 
                    AND s.status = 'active' 
                    AND ys.status = 'active'
                """, (program_id,))
                result = cursor.fetchone()
                program['total_subjects'] = result['count'] if result else 0
            except Exception as e:
                logger.warning(f"Error counting total subjects for program {program_id}: {e}")
                program['total_subjects'] = 0
            
            logger.info(f"Program {program_id}: {program['program_name']} - Semesters: {program['semesters']}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'programs': programs,
            'semesters': semesters,  # Return available semesters for frontend
            'total_count': len(programs)
        })
        
    except Exception as e:
        logger.error(f"Error fetching programs: {e}", exc_info=True)
        # Return empty array instead of failing completely
        return jsonify({
            'success': True,  # Still return success so frontend doesn't break
            'programs': [],
            'semesters': [],
            'message': f'Error loading programs: {str(e)}'
        })

@app.route('/api/get_courses_simple', methods=['GET'])
@login_required
def get_courses_simple():
    """Simple endpoint to get courses for dropdowns in schedule page"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                program_id as id,
                program_name as name,
                program_id as code
            FROM programs 
            WHERE status = 'active'
            ORDER BY program_name
        """)
        
        courses = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'courses': courses
        })
        
    except Exception as e:
        logger.error(f"Error fetching simple courses: {e}")
        return jsonify({
            'success': False, 
            'message': str(e),
            'courses': []
        })

@app.route('/api/add_program', methods=['POST'])
@login_required
@role_required(['super_admin'])  # Only super_admin can add programs
def add_program():
    """Add a new program"""
    try:
        data = request.json
        program_id = data.get('program_id', '').strip().upper()
        program_name = data.get('program_name', '').strip()
        department = data.get('department', '').strip()
        status = data.get('status', 'active')
        
        if not all([program_id, program_name, department]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if program already exists
        cursor.execute("SELECT program_id FROM programs WHERE program_id = %s", (program_id,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Program code already exists'})
        
        # Insert new program
        cursor.execute(
            "INSERT INTO programs (program_id, program_name, department, status) VALUES (%s, %s, %s, %s)",
            (program_id, program_name, department, status)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Program added successfully'})
        
    except Exception as e:
        logger.error(f"Error adding program: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_program', methods=['POST'])
@login_required
def update_program():
    """Update an existing program"""
    try:
        data = request.json
        program_id = data.get('program_id', '').strip().upper()
        program_name = data.get('program_name', '').strip()
        department = data.get('department', '').strip()
        status = data.get('status', 'active')
        
        if not all([program_id, program_name, department]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE programs SET program_name = %s, department = %s, status = %s WHERE program_id = %s",
            (program_name, department, status, program_id)
        )
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Program not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Program updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating program: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_program', methods=['POST'])
@login_required
def delete_program():
    """Delete a program"""
    try:
        program_id = request.json.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Soft delete by setting status to inactive
        cursor.execute("UPDATE programs SET status = 'inactive' WHERE program_id = %s", (program_id,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Program not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Program deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting program: {e}")
        return jsonify({'success': False, 'message': str(e)})        
    
@app.route('/api/get_current_class', methods=['GET'])
@login_required
def get_current_class():
    """Get current class for logged-in faculty member with enhanced details"""
    try:
        user_id = session.get('user_id')
        user_type = session.get('user_type')
        
        if not user_id or not user_type:
            return jsonify({'success': False, 'message': 'User not logged in'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current day and time
        current_day = datetime.now().strftime('%A')  # Monday, Tuesday, etc.
        current_time = datetime.now()
        current_time_str = current_time.strftime('%H:%M:%S')
        
        # Get logged in user's name based on user type
        logged_in_user = ""
        if user_type == 'admin':
            cursor.execute("SELECT first_name, last_name, middle_name FROM admins WHERE admin_id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                logged_in_user = f"{user_data['first_name']} {user_data['last_name']}"
                if user_data['middle_name']:
                    logged_in_user = f"{user_data['first_name']} {user_data['middle_name']} {user_data['last_name']}"
        elif user_type == 'faculty':
            cursor.execute("SELECT first_name, last_name, middle_name FROM faculty WHERE faculty_id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                logged_in_user = f"{user_data['first_name']} {user_data['last_name']}"
                if user_data['middle_name']:
                    logged_in_user = f"{user_data['first_name']} {user_data['middle_name']} {user_data['last_name']}"

        if user_type == 'admin':
            # Admin can see all current classes
            cursor.execute("""
                SELECT DISTINCT
                    cs.schedule_id,
                    cs.day_of_week,
                    cs.start_time,
                    cs.end_time,
                    cs.room,
                    cs.class_type,
                    s.subject_code,
                    s.subject_name,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    p.program_id,
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM class_schedules cs
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN faculty_schedules fs ON cs.schedule_id = fs.schedule_id
                LEFT JOIN faculty f ON fs.faculty_id = f.faculty_id
                WHERE cs.status = 'active'
                AND cs.day_of_week = %s
                AND cs.start_time <= %s
                AND cs.end_time >= %s
                ORDER BY cs.start_time
                LIMIT 1
            """, (current_day, current_time_str, current_time_str))
        else:
            # Faculty member sees only their current classes
            cursor.execute("""
                SELECT 
                    cs.schedule_id,
                    cs.day_of_week,
                    cs.start_time,
                    cs.end_time,
                    cs.room,
                    cs.class_type,
                    s.subject_code,
                    s.subject_name,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    p.program_id,
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM faculty_schedules fs
                JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                JOIN faculty f ON fs.faculty_id = f.faculty_id
                WHERE fs.faculty_id = %s 
                AND cs.status = 'active'
                AND cs.day_of_week = %s
                AND cs.start_time <= %s
                AND cs.end_time >= %s
                ORDER BY cs.start_time
                LIMIT 1
            """, (user_id, current_day, current_time_str, current_time_str))
        
        current_class = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if current_class:
            # Format class type
            if current_class['class_type'] == 'lecture':
                current_class['class_type_display'] = 'Lecture'
                current_class['subject_with_type'] = f"{current_class['subject_code']} (Lecture)"
            elif current_class['class_type'] == 'laboratory':
                current_class['class_type_display'] = 'Laboratory'
                current_class['subject_with_type'] = f"{current_class['subject_code']} (Laboratory)"
            else:
                current_class['class_type_display'] = current_class['class_type'].title()
                current_class['subject_with_type'] = current_class['subject_code']
            
            # Calculate REMAINING TIME from current time to end time
            end_time = datetime.strptime(str(current_class['end_time']), '%H:%M:%S')
            
            # Combine with current date for proper time comparison
            end_time_with_date = datetime.combine(current_time.date(), end_time.time())
            
            # Calculate remaining time (current time to end time)
            remaining_time = end_time_with_date - current_time
            
            # Ensure remaining time is not negative (in case class already ended)
            if remaining_time.total_seconds() < 0:
                remaining_time = timedelta(0)
            
            total_seconds = int(remaining_time.total_seconds())
            
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            # Convert all datetime objects to strings
            response_data = {
                'success': True,
                'has_class': True,
                'class_info': current_class,
                'timer': {
                    'session': {
                        'hours': str(hours).zfill(2),
                        'minutes': str(minutes).zfill(2),
                        'seconds': str(seconds).zfill(2)
                    },
                    'threshold': {
                        'hours': '00',
                        'minutes': '15',
                        'seconds': '00'
                    }
                },
                'logged_in_user': logged_in_user,
                'current_day': current_day,
                'current_time': current_time_str,
                'debug_info': {
                    'class_start': str(current_class['start_time']),
                    'class_end': str(current_class['end_time']),
                    'current_time': current_time_str,
                    'remaining_minutes': f"{hours}h {minutes}m {seconds}s"
                }
            }
            
            # Ensure all values in class_info are JSON serializable
            for key, value in response_data['class_info'].items():
                if isinstance(value, (datetime, date)):
                    response_data['class_info'][key] = value.isoformat()
                elif isinstance(value, timedelta):
                    response_data['class_info'][key] = str(value)
            
            return jsonify(response_data)
        else:
            return jsonify({
                'success': True,
                'has_class': False,
                'message': f'No classes scheduled for {current_day} at {current_time_str}',
                'logged_in_user': logged_in_user,
                'current_day': current_day,
                'current_time': current_time_str
            })
        
    except Exception as e:
        logger.error(f"Error getting current class: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_assigned_faculty_schedules', methods=['GET'])
@login_required
@role_required(['super_admin', 'admin'])
def get_assigned_faculty_schedules():
    """Get all faculty members with their assigned schedules for Current Schedule List"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT DISTINCT
                f.faculty_id,
                CONCAT(f.first_name, ' ', f.last_name) AS faculty_name,
                s.subject_code,
                s.subject_name,
                cs.day_of_week AS day,
                TIME_FORMAT(cs.start_time, '%I:%M %p') AS start_time,
                TIME_FORMAT(cs.end_time, '%I:%M %p') AS end_time,
                cs.room
            FROM faculty f
            JOIN faculty_schedules fs ON f.faculty_id = fs.faculty_id
            JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            JOIN semesters sem ON ys.semester_id = sem.semester_id
            JOIN academic_years ay ON sem.academic_year_id = ay.academic_year_id
            WHERE fs.status = 'active'
            AND cs.status = 'active'
            AND ay.is_current = TRUE
            ORDER BY f.last_name, f.first_name, cs.day_of_week, cs.start_time
        """)
        
        schedules = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Group schedules by faculty for display
        faculty_schedules = {}
        for schedule in schedules:
            faculty_id = schedule['faculty_id']
            if faculty_id not in faculty_schedules:
                faculty_schedules[faculty_id] = {
                    'faculty_name': schedule['faculty_name'],
                    'schedules': []
                }
            faculty_schedules[faculty_id]['schedules'].append({
                'subject': f"{schedule['subject_code']} - {schedule['subject_name']}",
                'meetings': f"{schedule['day']} {schedule['start_time']}-{schedule['end_time']} ({schedule['room']})"
            })
        
        result = list(faculty_schedules.values())
        logger.info(f"Found {len(result)} faculty with assigned schedules")
        return jsonify({
            'success': True,
            'faculty_schedules': result
        })
        
    except Exception as e:
        logger.error(f"Error fetching assigned faculty schedules: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_faculty_all_schedules', methods=['GET'])
@login_required
def get_faculty_all_schedules():
    """Get all assigned schedules for a faculty member - SIMPLE CAST FIX"""
    try:
        faculty_id = request.args.get('faculty_id')
        
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Use CAST to ensure proper time handling
        cursor.execute("""
            SELECT 
                cs.schedule_id,
                cs.day_of_week as day,
                TIME_FORMAT(CAST(cs.start_time AS TIME), '%h:%i %p') as start_time,
                TIME_FORMAT(CAST(cs.end_time AS TIME), '%h:%i %p') as end_time,
                cs.room,
                cs.class_type,
                s.subject_code,
                s.subject_name,
                ys.year_level,
                ys.section_name as section,
                p.program_name as program,
                p.program_id,
                sem.semester_number,
                ay.academic_year,
                fs.faculty_id
            FROM faculty_schedules fs
            INNER JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            INNER JOIN subjects s ON cs.subject_id = s.subject_id
            INNER JOIN year_sections ys ON cs.section_id = ys.section_id
            INNER JOIN programs p ON ys.program_id = p.program_id
            INNER JOIN semesters sem ON ys.semester_id = sem.semester_id
            INNER JOIN academic_years ay ON ys.academic_year_id = ay.academic_year_id
            WHERE fs.faculty_id = %s
            ORDER BY 
                FIELD(cs.day_of_week, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'),
                cs.start_time
        """, (faculty_id,))
        
        schedules = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Found {len(schedules)} schedules for faculty {faculty_id}")
        
        return jsonify({
            'success': True,
            'schedules': schedules,
            'count': len(schedules)
        })
        
    except Exception as e:
        logger.error(f"Error fetching faculty schedule: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'message': str(e),
            'schedules': []
        })

@app.route('/api/get_active_programs', methods=['GET'])
@login_required
def get_active_programs():
    """Get all active programs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT program_id, program_name, department 
            FROM programs 
            WHERE status = 'active'
            ORDER BY program_name
        """)
        
        programs = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'programs': programs})
        
    except Exception as e:
        logger.error(f"Error fetching active programs: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/get_semesters_for_switch', methods=['GET'])
@login_required
def get_semesters_for_switch():
    """Get semesters for switch modal with section and subject counts - SHOW ALL SEMESTERS"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        
        if not program_id or not academic_year:
            return jsonify({'success': False, 'message': 'Program ID and academic year are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get academic_year_id first
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        academic_year_result = cursor.fetchone()
        if not academic_year_result:
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = academic_year_result['academic_year_id']
        
        # Get ALL semesters for this academic year (don't filter by status)
        cursor.execute("""
            SELECT 
                s.semester_number,
                s.semester_id,
                s.status,
                COUNT(DISTINCT ys.section_id) as section_count,
                COUNT(DISTINCT sub.subject_id) as subject_count
            FROM semesters s
            LEFT JOIN year_sections ys ON s.semester_id = ys.semester_id AND ys.status = 'active'
            LEFT JOIN subjects sub ON ys.section_id = sub.section_id AND sub.status = 'active'
            WHERE s.academic_year_id = %s
            GROUP BY s.semester_id, s.semester_number, s.status
            ORDER BY 
                CASE s.semester_number
                    WHEN 'Summer' THEN 1
                    WHEN '1st Semester' THEN 2
                    WHEN '2nd Semester' THEN 3
                    ELSE 4
                END
        """, (academic_year_id,))
        
        semesters = cursor.fetchall()
        
        # Find the current active semester (should be only one)
        current_semester = None
        for semester in semesters:
            if semester['status'] == 'active':
                current_semester = semester['semester_number']
                break
        
        cursor.close()
        conn.close()
        
        # Format response - include ALL semesters
        formatted_semesters = []
        for semester in semesters:
            formatted_semesters.append({
                'semester_number': semester['semester_number'],
                'semester_id': semester['semester_id'],
                'section_count': semester['section_count'] or 0,
                'subject_count': semester['subject_count'] or 0,
                'is_current': semester['semester_number'] == current_semester
            })
        
        return jsonify({
            'success': True,
            'semesters': formatted_semesters,
            'current_semester': current_semester
        })
        
    except Exception as e:
        logger.error(f"Error fetching semesters for switch: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/set_active_semester', methods=['POST'])
@login_required
def set_active_semester():
    """Set a semester as active for a program - PROPERLY MANAGE ACTIVE STATUS"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester_number = data.get('semester_number')
        
        if not all([program_id, academic_year, semester_number]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        academic_year_result = cursor.fetchone()
        if not academic_year_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = academic_year_result[0]
        
        # Get semester_id
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s",
            (academic_year_id, semester_number)
        )
        
        semester_result = cursor.fetchone()
        if not semester_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        semester_id = semester_result[0]
        
        # First, set ALL semesters in this academic year to inactive
        cursor.execute(
            "UPDATE semesters SET status = 'inactive' WHERE academic_year_id = %s",
            (academic_year_id,)
        )
        
        # Then set ONLY the selected semester to active
        cursor.execute(
            "UPDATE semesters SET status = 'active' WHERE semester_id = %s",
            (semester_id,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully set {semester_number} {academic_year} as active period'
        })
        
    except Exception as e:
        logger.error(f"Error setting active semester: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/get_active_period', methods=['GET'])
@login_required
def get_active_period():
    """Get active academic year and semester for a program"""
    try:
        program_id = request.args.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current academic year
        cursor.execute("""
            SELECT academic_year 
            FROM academic_years 
            WHERE program_id = %s AND is_current = TRUE AND status = 'active'
            LIMIT 1
        """, (program_id,))
        
        academic_year_result = cursor.fetchone()
        
        if not academic_year_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No active academic year found'})
        
        academic_year = academic_year_result['academic_year']
        
        # Get academic_year_id
        cursor.execute("""
            SELECT academic_year_id 
            FROM academic_years 
            WHERE program_id = %s AND academic_year = %s AND status = 'active'
        """, (program_id, academic_year))
        
        academic_year_id_result = cursor.fetchone()
        
        if not academic_year_id_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = academic_year_id_result['academic_year_id']
        
        # Get current semester (you might need to adjust this logic based on your semester system)
        cursor.execute("""
            SELECT semester_number 
            FROM semesters 
            WHERE academic_year_id = %s AND status = 'active'
            ORDER BY 
                CASE semester_number
                    WHEN '1st Semester' THEN 1
                    WHEN '2nd Semester' THEN 2
                    WHEN 'Summer' THEN 3
                    ELSE 4
                END
            LIMIT 1
        """, (academic_year_id,))
        
        semester_result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not semester_result:
            return jsonify({'success': False, 'message': 'No active semester found'})
        
        return jsonify({
            'success': True,
            'active_period': {
                'academic_year': academic_year,
                'semester': semester_result['semester_number']
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting active period: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/get_sections_with_semester', methods=['GET'])
@login_required
def get_sections_with_semester():
    """Get sections filtered by program, year level, academic year, and semester"""
    try:
        program_id = request.args.get('program_id')
        year_level = request.args.get('year_level')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        
        if not all([program_id, year_level, academic_year, semester]):
            return jsonify({'success': False, 'message': 'All parameters are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get academic_year_id
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = result['academic_year_id']
        
        # Get semester_id
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s",
            (academic_year_id, semester)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        semester_id = result['semester_id']
        
        # Get sections for this specific semester - FIXED QUERY
        cursor.execute("""
            SELECT 
                ys.section_id,
                ys.section_name,
                ys.year_level,
                ys.program_id
            FROM year_sections ys
            WHERE ys.program_id = %s 
            AND ys.year_level = %s 
            AND ys.semester_id = %s
            AND ys.status = 'active'
            ORDER BY ys.section_name
        """, (program_id, year_level, semester_id))
        
        sections = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logger.info(f"Found {len(sections)} sections for program {program_id}, year {year_level}, {semester} {academic_year}")
        return jsonify({'success': True, 'sections': sections})
        
    except Exception as e:
        logger.error(f"Error fetching sections with semester: {e}")
        return jsonify({'success': False, 'message': str(e)})   

@app.route('/api/set_rtsp_url', methods=['POST'])
@login_required
def set_rtsp_url():
    """Set RTSP URL dynamically"""
    try:
        data = request.json
        rtsp_url = data.get('rtsp_url')
        
        if not rtsp_url:
            return jsonify({'success': False, 'message': 'RTSP URL is required'})
        
        # Decode URL if it's encoded
        import urllib.parse
        rtsp_url = urllib.parse.unquote(rtsp_url)
        
        logger.info(f"Setting RTSP URL: {rtsp_url}")
        
        # Try to connect to the RTSP stream
        success = open_stream(rtsp_url)
        
        if success:
            return jsonify({
                'success': True, 
                'message': 'RTSP URL set successfully',
                'camera_available': camera_available
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to connect to RTSP stream',
                'camera_available': camera_available,
                'using_fallback': use_dummy_feed
            })
            
    except Exception as e:
        logger.error(f"Error setting RTSP URL: {e}")
        return jsonify({
            'success': False, 
            'message': f'Error: {str(e)}',
            'camera_available': False
        })

@app.route('/api/get_session_info', methods=['GET'])
@login_required
def get_session_info():
    try:
        schedule_id = request.args.get('schedule_id')
        if not schedule_id:
            return jsonify({'success': False, 'message': 'Schedule ID required'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT 
                cs.schedule_id,
                cs.start_time,
                cs.end_time,
                cs.room,
                cs.class_type,
                s.subject_code,
                s.subject_name,
                ys.year_level,
                ys.section_name,
                ys.section_id,
                p.program_id,
                p.program_name,
                f.faculty_id,
                f.first_name,
                f.last_name,
                f.photo_path
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            LEFT JOIN faculty_schedules fs ON cs.schedule_id = fs.schedule_id AND fs.status = 'active'
            LEFT JOIN faculty f ON fs.faculty_id = f.faculty_id
            WHERE cs.schedule_id = %s AND cs.status = 'active'
        """
        
        cursor.execute(query, (schedule_id,))
        schedule = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not schedule:
            return jsonify({'success': False, 'message': 'Schedule not found'}), 404
        
        # Build faculty name
        faculty_name = 'No Faculty Assigned'
        if schedule['first_name'] and schedule['last_name']:
            faculty_name = f"{schedule['first_name']} {schedule['last_name']}"
        
        # Fix photo path
        faculty_photo = '/static/images/placeholder.jpg'
        if schedule['photo_path']:
            faculty_photo = schedule['photo_path']
        elif schedule['faculty_id']:
            faculty_photo = f"/static/images/faculty_photos/{schedule['faculty_id']}.jpg"
        
        response = {
            'success': True,
            'schedule': {
                'schedule_id': schedule['schedule_id'],
                'subject_code': schedule['subject_code'],
                'subject_name': schedule['subject_name'],
                'class_type': schedule['class_type'],
                'year_level': schedule['year_level'],
                'section_name': schedule['section_name'],
                'section_id': schedule['section_id'],
                'program_id': schedule['program_id'],
                'program_name': schedule['program_name'],
                'room': schedule['room'],
                'start_time': str(schedule['start_time']),
                'end_time': str(schedule['end_time']),
                'faculty_id': schedule['faculty_id'],
                'faculty_name': faculty_name,
                'faculty_photo': faculty_photo,
                'duration': 180,
                'threshold': 15
            }
        }
        
        logger.info(f"✅ Session info loaded: {schedule['program_id']} {schedule['year_level']}{schedule['section_name']}, Faculty photo: {faculty_photo}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_session_info: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize session with start time"""
    global session_start_time
    try:
        data = request.json
        session_id = data.get('session_id')
        
        session_start_time = datetime.now()
        
        return jsonify({
            'success': True,
            'session_start': session_start_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/initialize_session', methods=['POST'])
def initialize_session():
    """Initialize session with parameters from URL"""
    global session_start_time, current_session_id, session_threshold_seconds, session_total_duration_seconds
    global current_session_id
    global detectionStopped, tracks, locked_tracks, pending_confirmations  # 🎯 ADD THESE
    global student_presence_tracker, locked_track_reid_features  # 🎯 ADD THESE

    connection = None
    cursor = None
    
    logger.info("🔄 INITIALIZE_SESSION CALLED")

    detectionStopped = False

    tracks = []
    locked_tracks = {}
    pending_confirmations = {}
    student_presence_tracker = {}
    locked_track_reid_features = {}
    
    logger.info("🟢 Detection flag RESET to False for new session")
    logger.info(f"🧹 Cleared: {len(tracks)} tracks, {len(locked_tracks)} locked tracks, {len(pending_confirmations)} pending")
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        logger.info(f"📦 Received data: {data}")
        
        schedule_id = data.get('schedule_id')
        if not schedule_id:
            return jsonify({'success': False, 'message': 'Missing schedule_id'}), 400
        
        # ✅ GENERATE UNIQUE SESSION ID
        import uuid
        import datetime
        unique_session_id = f"{schedule_id}_{uuid.uuid4().hex[:8]}_{int(datetime.datetime.now().timestamp())}"
        current_session_id = unique_session_id
        
        # Handle duration and threshold
        try:
            session_total_duration_seconds = int(data.get('duration', 3600))
            session_threshold_seconds = int(data.get('threshold', 900))
        except (ValueError, TypeError):
            session_total_duration_seconds = 3600
            session_threshold_seconds = 900
        
        # ✅ SET SESSION START TIME
        session_start_time = datetime.datetime.now()
        
        logger.info(f"🎯 Generated unique session ID: {unique_session_id}")
        logger.info(f"✅ SESSION PARAMS - Duration: {session_total_duration_seconds}s, Threshold: {session_threshold_seconds}s")
        
        # Get database connection
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Failed to connect to database'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # ✅ FIND SECTION_ID FROM YEAR_SECTIONS TABLE
        section_id = None
        year_level = data.get('year_level', '')
        section_name = data.get('section', '')
        program_id = data.get('program', '')
        
        logger.info(f"🔍 Looking for section: Year={year_level}, Section={section_name}, Program={program_id}")
        
        if year_level and section_name and program_id:
            try:
                cursor.execute("""
                    SELECT section_id FROM year_sections 
                    WHERE year_level = %s AND section_name = %s AND program_id = %s
                    LIMIT 1
                """, (year_level, section_name, program_id))
                section_result = cursor.fetchone()
                if section_result:
                    section_id = section_result['section_id']
                    logger.info(f"✅ Found section_id: {section_id}")
                else:
                    logger.warning(f"⚠️ No section found for Year={year_level}, Section={section_name}, Program={program_id}")
            except Exception as e:
                logger.warning(f"⚠️ Error finding section: {e}")
        
        # ✅ GET SUBJECT AND ROOM INFO - ENHANCED QUERY
        subject_code = 'Unknown Subject'
        subject_name = 'Unknown Subject'
        room = 'Unknown Room'
        
        try:
            cursor.execute("""
                SELECT s.subject_code, s.subject_name, cs.room
                FROM class_schedules cs
                JOIN subjects s ON cs.subject_id = s.subject_id
                WHERE cs.schedule_id = %s
            """, (schedule_id,))
            result = cursor.fetchone()
            if result:
                subject_code = result.get('subject_code', 'Unknown Subject')
                subject_name = result.get('subject_name', 'Unknown Subject')
                room = result.get('room', 'Unknown Room')
                logger.info(f"📚 Subject info: {subject_code} - {subject_name} - {room}")
            else:
                logger.warning(f"⚠️ No subject found for schedule_id: {schedule_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch subject info: {e}")
        
        # ✅ SAVE SESSION TO DATABASE WITH UNIQUE SESSION ID AND BOTH MINUTES & SECONDS
        try:
            # First, try to get the next session instance number
            cursor.execute("""
                SELECT COALESCE(MAX(session_instance), 0) + 1 as next_instance 
                FROM attendance_sessions 
                WHERE original_schedule_id = %s
            """, (schedule_id,))
            instance_result = cursor.fetchone()
            next_instance = instance_result['next_instance'] if instance_result else 1
            
            # 🎯 CRITICAL FIX: Store BOTH minutes AND seconds in database
            cursor.execute("""
                INSERT INTO attendance_sessions 
                (session_id, class_name, subject_code, subject_name, room, started_at, 
                 late_threshold_minutes, total_duration_minutes, 
                 threshold_seconds_total, session_duration_seconds_total,
                 created_by, status, section_id,
                 original_schedule_id, session_instance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s)
            """, (
                unique_session_id,  # Use unique session ID
                f"{data.get('program', 'Unknown')} {data.get('year_level', '')}{data.get('section', '')}",
                subject_code,
                subject_name,
                room,
                session_start_time,
                session_threshold_seconds // 60,  # Store minutes
                session_total_duration_seconds // 60,  # Store minutes
                session_threshold_seconds,  # 🎯 STORE SECONDS
                session_total_duration_seconds,  # 🎯 STORE SECONDS
                data.get('instructor', 'System'),
                section_id,
                schedule_id,  # Store original schedule_id for reference
                next_instance  # Session instance counter
            ))
            logger.info(f"💾 Session saved to database with unique ID: {unique_session_id}, instance: {next_instance}")
            logger.info(f"⏰ STORED TIMING: Duration={session_total_duration_seconds}s, Threshold={session_threshold_seconds}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save session to database: {e}")
            # If insert fails, try without instance number but STILL with seconds
            try:
                cursor.execute("""
                    INSERT INTO attendance_sessions 
                    (session_id, class_name, subject_code, subject_name, room, started_at, 
                     late_threshold_minutes, total_duration_minutes, 
                     threshold_seconds_total, session_duration_seconds_total,
                     created_by, status, section_id, original_schedule_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
                """, (
                    unique_session_id,
                    f"{data.get('program', 'Unknown')} {data.get('year_level', '')}{data.get('section', '')}",
                    subject_code,
                    subject_name,
                    room,
                    session_start_time,
                    session_threshold_seconds // 60,
                    session_total_duration_seconds // 60,
                    session_threshold_seconds,  # 🎯 STORE SECONDS
                    session_total_duration_seconds,  # 🎯 STORE SECONDS
                    data.get('instructor', 'System'),
                    section_id,
                    schedule_id
                ))
                logger.info(f"💾 Session saved without instance number: {unique_session_id}")
            except Exception as retry_error:
                logger.error(f"❌ Failed to save session even with retry: {retry_error}")
                # If still failing, try basic insert but STILL with seconds
                try:
                    cursor.execute("""
                        INSERT INTO attendance_sessions 
                        (session_id, class_name, subject_code, subject_name, room, started_at, 
                         late_threshold_minutes, total_duration_minutes, 
                         threshold_seconds_total, session_duration_seconds_total,
                         created_by, status, section_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s)
                    """, (
                        unique_session_id,
                        f"{data.get('program', 'Unknown')} {data.get('year_level', '')}{data.get('section', '')}",
                        subject_code,
                        subject_name,
                        room,
                        session_start_time,
                        session_threshold_seconds // 60,
                        session_total_duration_seconds // 60,
                        session_threshold_seconds,  # 🎯 STORE SECONDS
                        session_total_duration_seconds,  # 🎯 STORE SECONDS
                        data.get('instructor', 'System'),
                        section_id
                    ))
                    logger.info(f"💾 Session saved with basic insert: {unique_session_id}")
                except Exception as final_error:
                    logger.error(f"❌ All insert attempts failed: {final_error}")
                    raise final_error
        
        # ✅ GET FACULTY INFO
        faculty_name = data.get('instructor', 'Unknown Instructor')
        faculty_photo = '../static/images/placeholder.jpg'
        faculty_role = 'moderator'
        
        try:
            instructor_name = data.get('instructor', '')
            
            if instructor_name:
                cursor.execute("""
                    SELECT first_name, last_name, photo_path, role 
                    FROM faculty 
                    WHERE CONCAT(first_name, ' ', last_name) LIKE %s 
                    OR first_name LIKE %s 
                    OR last_name LIKE %s
                    LIMIT 1
                """, (f"%{instructor_name}%", f"%{instructor_name}%", f"%{instructor_name}%"))
                
                faculty_result = cursor.fetchone()
                
                if not faculty_result:
                    cursor.execute("""
                        SELECT first_name, last_name, photo_path, role 
                        FROM admins 
                        WHERE CONCAT(first_name, ' ', last_name) LIKE %s 
                        OR first_name LIKE %s 
                        OR last_name LIKE %s
                        LIMIT 1
                    """, (f"%{instructor_name}%", f"%{instructor_name}%", f"%{instructor_name}%"))
                    
                    faculty_result = cursor.fetchone()
                
                if faculty_result:
                    faculty_name = f"{faculty_result['first_name']} {faculty_result['last_name']}"
                    faculty_role = faculty_result.get('role', 'moderator')
                    if faculty_result.get('photo_path'):
                        faculty_photo = faculty_result['photo_path']
                    
                    logger.info(f"👤 Found user: {faculty_name} - Role: {faculty_role}")
                else:
                    if "super" in instructor_name.lower() or "administrator" in instructor_name.lower():
                        faculty_role = 'super_admin'
                    elif "admin" in instructor_name.lower():
                        faculty_role = 'admin' 
                    else:
                        faculty_role = 'moderator'
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch faculty info: {e}")
        
        # Format display times
        def format_time_display(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
        
        # ✅ STORE SESSION DATA - USE UNIQUE SESSION ID
        session_data = {
            'session_id': unique_session_id,  # 🎯 CRITICAL: Use unique session ID
            'schedule_id': schedule_id,  # Keep original schedule_id for reference
            'instructor': faculty_name,
            'subject_code': subject_code,
            'subject_name': subject_name,  # ✅ ADD SUBJECT NAME
            'room': room,
            'role': faculty_role,
            'faculty_photo': faculty_photo,
            'year_level': data.get('year_level', ''),
            'program': data.get('program', ''),
            'section': data.get('section', ''),
            'duration': session_total_duration_seconds,
            'duration_display': format_time_display(session_total_duration_seconds),
            'threshold': session_threshold_seconds,
            'threshold_display': format_time_display(session_threshold_seconds),
            'start_time': session_start_time.isoformat(),
            'section_id': section_id  # ✅ ADD SECTION_ID TO RESPONSE
        }
        
        connection.commit()
        logger.info(f"🔍 DEBUG: threshold input = {data.get('threshold')}")
        logger.info(f"🔍 DEBUG: session_threshold_seconds = {session_threshold_seconds}")
        logger.info(f"✅ SESSION INITIALIZED: {session_start_time}")
        logger.info(f"👤 Faculty: {faculty_name} - Role: {faculty_role}")
        logger.info(f"🏫 Class: {data.get('program')} {data.get('year_level')}{data.get('section')}")
        logger.info(f"📚 Subject: {subject_code} - {subject_name} - {room}")
        logger.info(f"🔗 Section ID: {section_id}")
        logger.info(f"🎯 Unique Session ID returned: {unique_session_id}")
        
        return jsonify({
            'success': True, 
            'data': session_data,
            'message': 'Session started successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in initialize_session: {e}", exc_info=True)
        if connection:
            connection.rollback()
        return jsonify({
            'success': False, 
            'message': f'Server error: {str(e)}'
        }), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.route('/api/force_start_session', methods=['GET'])
def force_start_session():
    """Force start a session for testing"""
    global session_start_time, current_session_id, session_threshold_seconds
    
    try:
        # Set session variables manually
        session_start_time = datetime.now()
        current_session_id = "test_session_forced"
        session_threshold_seconds = 10  # 10 seconds threshold for testing
        
        return jsonify({
            'success': True,
            'message': 'Session forced started for testing',
            'session_start_time': session_start_time.isoformat(),
            'session_threshold_seconds': session_threshold_seconds,
            'current_session_id': current_session_id
        })
        
    except Exception as e:
        logger.error(f"Error in force_start_session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug_threshold')
def debug_threshold():
    """Debug endpoint to check current threshold"""
    global session_start_time, session_threshold_seconds
    
    try:
        current_time = datetime.now()
        
        # Safe time difference calculation
        time_diff_seconds = None
        should_be_late = False
        
        if session_start_time:
            time_diff = current_time - session_start_time
            time_diff_seconds = time_diff.total_seconds()
            should_be_late = time_diff_seconds > session_threshold_seconds
        
        return jsonify({
            'success': True,
            'session_threshold_seconds': session_threshold_seconds,
            'session_start_time': session_start_time.isoformat() if session_start_time else 'Not set',
            'current_time': current_time.isoformat(),
            'time_difference_seconds': time_diff_seconds,
            'should_be_late': should_be_late,
            'global_variables_set': {
                'session_start_time': session_start_time is not None,
                'session_threshold_seconds': session_threshold_seconds is not None,
                'current_session_id': current_session_id is not None
            }
        })
        
    except Exception as e:
        logger.error(f"Error in debug_threshold: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'global_variables': {
                'session_start_time': str(session_start_time),
                'session_threshold_seconds': session_threshold_seconds,
                'current_session_id': current_session_id
            }
        }), 500

@app.route('/api/get_class_students', methods=['GET'])
def get_class_students():
    """Get students for the current class based on program, year_level, and section"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Failed to connect to database'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # Get URL parameters
        program = request.args.get('program')
        year_level = request.args.get('year_level')
        section = request.args.get('section')
        
        print(f"DEBUG: Received parameters - program: '{program}', year_level: '{year_level}', section: '{section}'")
        
        if not all([program, year_level, section]):
            return jsonify({'success': False, 'message': 'Missing required parameters'}), 400
        
        # FIXED: Map program names to match the 'course' column in students table
        program_map = {
            'Information Technology': 'BSIT',  # Map to BSIT
            'Computer Science': 'BSCS',        # Map to BSCS  
            'Associate in Computer Technology': 'ACT',
            'IT': 'BSIT',                      # Map IT to BSIT
            'CS': 'BSCS',                      # Map CS to BSCS
            'ACT': 'ACT',
            'BSIT': 'BSIT',                    # Direct mappings
            'BSCS': 'BSCS'
        }
        
        # Use the mapping or fall back to the original value
        course_to_search = program_map.get(program, program)
        
        # Extract numeric year level
        year_level_num = ''.join(filter(str.isdigit, str(year_level))) if year_level else ''
        
        # Build the year_section format - based on your data it should be "4C"
        year_section_to_search = f"{year_level_num}{section}"
        
        print(f"DEBUG: Searching for course='{course_to_search}', year_section='{year_section_to_search}'")
        
        # Query to get students
        query = """
            SELECT student_id, first_name, last_name, middle_name, 
                   course, year_section, photo_path, status
            FROM students 
            WHERE course = %s AND year_section = %s AND status = 'active'
            ORDER BY last_name, first_name
        """
        
        cursor.execute(query, (course_to_search, year_section_to_search))
        students = cursor.fetchall()
        
        print(f"DEBUG: Found {len(students)} students")
        
        # Format student data for frontend
        formatted_students = []
        for student in students:
            # Build full name with middle name if available
            full_name = f"{student['first_name']} {student['last_name']}"
            if student['middle_name']:
                full_name = f"{student['first_name']} {student['middle_name']} {student['last_name']}"
                
            formatted_students.append({
                'id': student['student_id'],
                'name': full_name,
                'firstName': student['first_name'],
                'lastName': student['last_name'],
                'photo_path': student['photo_path'] or '/static/images/default-avatar.jpg',
                'status': 'absent'  # Default status
            })
        
        return jsonify({
            'success': True, 
            'students': formatted_students,
            'total_count': len(formatted_students),
            'detected_count': 0
        })
        
    except Exception as e:
        logger.error(f"Error fetching class students: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.route('/api/debug_students')
def debug_students():
    """Debug route to see all students and their course/year_section"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT student_id, first_name, last_name, course, year_section FROM students")
        students = cursor.fetchall()
        
        return jsonify({'success': True, 'students': students})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if cursor: cursor.close()
        if connection: connection.close()

from flask import Flask, request, jsonify
from datetime import datetime
import re
import logging

# Assuming these are defined elsewhere
logger = logging.getLogger(__name__)
student_status = {}  # In-memory dictionary to track student status

@app.route('/api/get_student_status')
def get_student_status():
    """Get current status of all students - FIXED: Proper status display with missing periods"""
    global session_start_time, session_threshold_seconds, current_session_id, student_presence_tracker
    global locked_tracks
    
    try:
        program = request.args.get('program')
        year_level = request.args.get('year_level') 
        section = request.args.get('section')
        session_id = request.args.get('session_id', current_session_id)
        
        # ✅ Load threshold from database if not set
        if not session_threshold_seconds and current_session_id:
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT threshold_seconds_total 
                    FROM attendance_sessions 
                    WHERE session_id = %s
                """, (current_session_id,))
                session_data = cursor.fetchone()
                if session_data and session_data.get('threshold_seconds_total'):
                    session_threshold_seconds = session_data['threshold_seconds_total']
                    logger.info(f"🎯 Loaded threshold: {session_threshold_seconds} seconds")
                else:
                    logger.warning(f"⚠️ No threshold found, using default 900s")
                    session_threshold_seconds = 900
                cursor.close()
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not load threshold: {e}")
                session_threshold_seconds = 900
        
        threshold_seconds = session_threshold_seconds
        logger.info(f"⏰ Using threshold: {threshold_seconds} seconds")
        
        if not threshold_seconds:
            logger.warning("⚠️ No threshold set, using default 900 seconds")
            threshold_seconds = 900
        
        program_map = {
            'Information Technology': 'BSIT',
            'Computer Science': 'BSCS',
            'Associate in Computer Technology': 'ACT',
            'IT': 'BSIT',
            'CS': 'BSCS',
            'ACT': 'ACT',
            'BSIT': 'BSIT',
            'BSCS': 'BSCS'
        }
        course_to_search = program_map.get(program, program)
        year_level_num = ''.join(filter(str.isdigit, str(year_level))) if year_level else ''
        year_section_to_search = f"{year_level_num}{section}"
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT student_id, first_name, last_name, course, year_section
            FROM students 
            WHERE course = %s AND year_section = %s AND status = 'active'
        """, (course_to_search, year_section_to_search))
        
        students = cursor.fetchall()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT name, timestamp, status, remarks 
            FROM attendance 
            WHERE student_id IS NULL 
            AND DATE(timestamp) = %s 
            AND session_id = 'manual_add'
        """, (today,))
        
        temp_students = cursor.fetchall()
        
        student_list = []
        detected_count = 0
        
        # 🎯 CRITICAL FIX: Check missing periods
        missing_student_ids = []
        currently_present_ids = set()
        
        if session_id:
            try:
                # Check database for missing periods
                cursor.execute("""
                    SELECT student_id, missing_start FROM missing_periods 
                    WHERE session_id = %s AND returned = FALSE
                """, (session_id,))
                missing_records = cursor.fetchall()
                missing_student_ids = [record['student_id'] for record in missing_records]
                
                logger.info(f"🔍 DB MISSING CHECK: Found {len(missing_student_ids)} in database")
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking missing students: {e}")
        
        # Check real-time tracking data for currently present students
        try:
            if locked_tracks:
                for person_id, lock_info in locked_tracks.items():
                    if lock_info.get('type') == 'student':
                        student_id = lock_info.get('id')
                        if student_id:
                            currently_present_ids.add(student_id)
                
                logger.info(f"🔍 REAL-TIME CHECK: {len(currently_present_ids)} students currently tracked: {currently_present_ids}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking locked_tracks: {e}")
        
        for student in students:
            student_id = student['student_id']
            student_name = f"{student['first_name']} {student['last_name']}"
            
            # 🎯 CRITICAL FIX: Four-tier status determination
            
            # Priority 1: Real-time tracking shows student is currently present
            if student_id in currently_present_ids:
                current_status = 'present'
                logger.info(f"✅ REAL-TIME PRESENT: {student_name} is currently being tracked")
            
            # Priority 2: Student is currently missing
            elif student_id in missing_student_ids:
                current_status = 'missing'
                logger.info(f"🎯 CURRENTLY MISSING: {student_name} ({student_id})")
            
            # Priority 3: Check attendance records for original status
            else:
                cursor.execute("""
                    SELECT status, timestamp FROM attendance 
                    WHERE student_id = %s AND session_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (student_id, session_id))
                
                attendance_record = cursor.fetchone()
                
                if attendance_record:
                    current_status = attendance_record['status']
                    
                    # Late check for 'present' status
                    if current_status == 'present' and session_start_time:
                        arrival_time = attendance_record['timestamp']
                        if isinstance(arrival_time, str):
                            arrival_time = datetime.strptime(arrival_time, "%Y-%m-%d %H:%M:%S")
                        
                        time_difference = arrival_time - session_start_time
                        time_diff_seconds = time_difference.total_seconds()
                        
                        if time_diff_seconds > threshold_seconds:
                            current_status = 'late'
                            # Update if needed
                            cursor.execute("""
                                UPDATE attendance 
                                SET status = 'late' 
                                WHERE student_id = %s AND session_id = %s AND timestamp = %s
                            """, (student_id, session_id, attendance_record['timestamp']))
                            logger.info(f"🔄 UPDATED TO LATE: {student_name}")
                else:
                    current_status = 'absent'
            
            logger.debug(f"🔍 FINAL STATUS: {student_name} -> {current_status}")
            
            if current_status in ['present', 'late']:
                detected_count += 1
            
            student_list.append({
                'id': student_id,
                'name': student_name,
                'status': current_status,
                'type': 'regular'
            })
        
        # Process temporary students (unchanged)
        temp_counter = 1
        for temp_student in temp_students:
            temp_name = temp_student['name']
            temp_remarks = temp_student.get('remarks', '')
            current_status = temp_student['status']
            
            if current_status == 'present' and session_start_time:
                arrival_time = temp_student['timestamp']
                if isinstance(arrival_time, str):
                    arrival_time = datetime.strptime(arrival_time, "%Y-%m-%d %H:%M:%S")
                
                time_difference = arrival_time - session_start_time
                time_diff_seconds = time_difference.total_seconds()
                
                if time_diff_seconds > threshold_seconds:
                    current_status = 'late'
                    cursor.execute("""
                        UPDATE attendance 
                        SET status = 'late' 
                        WHERE name = %s AND DATE(timestamp) = %s AND timestamp = %s
                    """, (temp_name, today, temp_student['timestamp']))
            
            temp_id = None
            display_name = temp_name
            
            if temp_remarks and 'temp_id:' in temp_remarks:
                temp_id = temp_remarks.split('temp_id:')[1].strip()
                display_name = re.sub(r'\s*\(ID:\s*[^)]+\)', '', temp_name).strip()
            
            if not temp_id:
                id_match = re.search(r'\(ID:\s*([^)]+)\)', temp_name)
                if id_match:
                    temp_id = id_match.group(1).strip()
                    display_name = re.sub(r'\s*\(ID:\s*[^)]+\)', '', temp_name).strip()
            
            if not temp_id:
                temp_id = f"temp_{temp_counter}"
                temp_counter += 1
            
            student_list.append({
                'id': temp_id,
                'name': display_name,
                'status': current_status,
                'type': 'temporary'
            })
            
            if current_status in ['present', 'late']:
                detected_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Status summary
        status_counts = {}
        for student in student_list:
            status_counts[student['status']] = status_counts.get(student['status'], 0) + 1
        
        logger.info(f"📊 STATUS SUMMARY: {status_counts} | Real-time present: {len(currently_present_ids)}, DB missing: {len(missing_student_ids)}")
        
        return jsonify({
            'success': True,
            'students': student_list,
            'detected_count': detected_count,
            'total_count': len(student_list),
            'threshold_seconds': threshold_seconds,
            'session_start_time': session_start_time.isoformat() if session_start_time else None,
            'current_session_id': current_session_id,
            'status_summary': status_counts,
            'missing_count_in_db': len(missing_student_ids),
            'real_time_present_count': len(currently_present_ids)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting student status: {e}")
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/api/get_session_threshold')
def get_session_threshold():
    """Get the current session threshold"""
    global session_threshold_seconds, session_start_time
    
    return jsonify({
        'success': True,
        'threshold_seconds': session_threshold_seconds,
        'threshold_minutes': session_threshold_seconds // 60 if session_threshold_seconds else 15,
        'session_start_time': session_start_time.isoformat() if session_start_time else None,
        'current_time': datetime.now().isoformat()
    })    
    
@app.route('/api/manage_student', methods=['POST'])
def manage_student():
    """Handle student management actions"""
    global session_start_time, session_threshold_seconds, current_session_id
    
    try:
        data = request.json
        action = data.get('action')
        student_data = data.get('student_data', {})
        
        if action == 'add_temporary':
            student_id = student_data.get('id')
            student_name = student_data.get('name')
            
            if not student_id or not student_name:
                return jsonify({
                    'success': False, 
                    'title': 'Missing Information',
                    'message': 'Please provide both student ID and name'
                })
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            display_name = f"{student_name} (ID: {student_id})"
            remarks = f"temp_id:{student_id}"
            
            # ✅ FIXED: Use global threshold for temporary students
            status = 'present'
            if session_start_time:
                current_time_dt = datetime.now()
                time_difference = current_time_dt - session_start_time
                
                # Use the global threshold
                threshold_seconds = session_threshold_seconds
                if not threshold_seconds:
                    threshold_seconds = 900  # Default 15 minutes
                
                if time_difference.total_seconds() > threshold_seconds:
                    status = 'late'
                    logger.info(f"⏰ TEMPORARY STUDENT LATE: {student_name} arrived {time_difference.total_seconds():.1f} seconds after start (threshold: {threshold_seconds} seconds)")
            
            # ✅ GET SESSION SUBJECT INFORMATION
            subject_code = 'Unknown Subject'
            subject_name = 'Unknown Subject'
            room = 'Unknown Room'
            section_id = None
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT subject_code, subject_name, room, section_id 
                    FROM attendance_sessions 
                    WHERE session_id = %s
                """, (current_session_id,))
                session_result = cursor.fetchone()
                
                if session_result:
                    subject_code = session_result.get('subject_code', 'Unknown Subject')
                    subject_name = session_result.get('subject_name', 'Unknown Subject')
                    room = session_result.get('room', 'Unknown Room')
                    section_id = session_result.get('section_id')
                    logger.info(f"🔗 Found session subject: {subject_code} - {subject_name}")
                
                cursor.close()
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch session subject info: {e}")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # ✅ FIXED: INSERT WITH SUBJECT INFORMATION
            cursor.execute("""
                INSERT INTO attendance 
                (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (None, display_name, current_time, 'student', status, 'manual_add', remarks, subject_code, subject_name, room, section_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            student_status[student_id] = status
            
            logger.info(f"✅ TEMPORARY ATTENDANCE ADDED: {student_name} ({student_id}) - {status} - Subject: {subject_code}")
            return jsonify({
                'success': True, 
                'title': 'Success',
                'message': f'Temporary attendance added for {student_name}',
                'student_name': student_name,
                'student_id': student_id
            })
            
        elif action == 'remove':
            student_id = student_data.get('student_id')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT first_name, last_name 
                FROM students 
                WHERE student_id = %s AND status = 'active'
            """, (student_id,))
            student = cursor.fetchone()
            
            if student:
                student_name = f"{student[0]} {student[1]}"
                cursor.execute("""
                    UPDATE students 
                    SET status = 'inactive' 
                    WHERE student_id = %s
                """, (student_id,))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                if student_id in student_status:
                    del student_status[student_id]
                
                logger.info(f"🗑️ REGULAR STUDENT REMOVED: {student_id}")
                return jsonify({
                    'success': True, 
                    'title': 'Student Removed',
                    'message': f'Student {student_name} has been removed from the class'
                })
            else:
                today = datetime.now().strftime("%Y-%m-%d")
                cursor.execute("""
                    SELECT name FROM attendance 
                    WHERE student_id IS NULL 
                    AND session_id = 'manual_add'
                    AND DATE(timestamp) = %s
                    AND (name LIKE %s OR remarks LIKE %s)
                    LIMIT 1
                """, (today, f"%{student_id}%", f"%temp_id:{student_id}%"))
                
                temp_student = cursor.fetchone()
                temp_name = temp_student[0] if temp_student else "Unknown Student"
                
                cursor.execute("""
                    DELETE FROM attendance 
                    WHERE student_id IS NULL 
                    AND session_id = 'manual_add'
                    AND DATE(timestamp) = %s
                    AND (name LIKE %s OR remarks LIKE %s)
                """, (today, f"%{student_id}%", f"%temp_id:{student_id}%"))
                
                deleted_count = cursor.rowcount
                conn.commit()
                cursor.close()
                conn.close()
                
                if student_id in student_status:
                    del student_status[student_id]
                
                logger.info(f"🗑️ TEMPORARY STUDENT REMOVED: {student_id} (deleted {deleted_count} records)")
                return jsonify({
                    'success': True, 
                    'title': 'Temporary Student Removed',
                    'message': f'Temporary student {temp_name} has been removed'
                })
            
        elif action == 'transfer':
            student_id = student_data.get('student_id')
            new_section = student_data.get('new_section')
            
            if not student_id or not new_section:
                return jsonify({
                    'success': False, 
                    'title': 'Missing Information',
                    'message': 'Please provide both student ID and new section'
                })
            
            match = re.match(r'(\w+)\s*(\d+)(\w+)', new_section)
            if not match:
                return jsonify({
                    'success': False, 
                    'title': 'Invalid Format',
                    'message': 'Please use format like "BSIT 4C" or "BSCS 3A"'
                })
            
            course = match.group(1)
            year_section = f"{match.group(2)}{match.group(3)}"
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT first_name, last_name FROM students WHERE student_id = %s", (student_id,))
            student = cursor.fetchone()
            
            if not student:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False, 
                    'title': 'Student Not Found',
                    'message': f'Student with ID {student_id} was not found'
                })
            
            student_name = f"{student[0]} {student[1]}"
            
            cursor.execute("""
                UPDATE students 
                SET course = %s, year_section = %s
                WHERE student_id = %s
            """, (course, year_section, student_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            if student_id in student_status:
                del student_status[student_id]
            
            logger.info(f"🔄 STUDENT TRANSFERRED: {student_id} to {new_section}")
            return jsonify({
                'success': True, 
                'title': 'Transfer Successful',
                'message': f'Student {student_name} has been transferred to {new_section}'
            })
            
        elif action == 'excused':
            student_id = student_data.get('student_id')
            remarks = student_data.get('remarks', 'Excused')
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            today = datetime.now().strftime("%Y-%m-%d")
            
            # ✅ GET SESSION SUBJECT INFORMATION
            subject_code = 'Unknown Subject'
            subject_name = 'Unknown Subject'
            room = 'Unknown Room'
            section_id = None
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT subject_code, subject_name, room, section_id 
                    FROM attendance_sessions 
                    WHERE session_id = %s
                """, (current_session_id,))
                session_result = cursor.fetchone()
                
                if session_result:
                    subject_code = session_result.get('subject_code', 'Unknown Subject')
                    subject_name = session_result.get('subject_name', 'Unknown Subject')
                    room = session_result.get('room', 'Unknown Room')
                    section_id = session_result.get('section_id')
                
                cursor.close()
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch session subject info: {e}")
            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            try:
                # Check if student exists in students table (regular student)
                cursor.execute("""
                    SELECT first_name, last_name 
                    FROM students 
                    WHERE student_id = %s AND status = 'active'
                """, (student_id,))
                student = cursor.fetchone()
                
                if student:
                    # REGULAR STUDENT
                    student_name = f"{student['first_name']} {student['last_name']}"
                    
                    # Check for existing attendance record today
                    cursor.execute("""
                        SELECT id, status FROM attendance 
                        WHERE student_id = %s AND DATE(timestamp) = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (student_id, today))
                    
                    existing_record = cursor.fetchone()
                    
                    if existing_record:
                        # ✅ FIXED: UPDATE WITH SUBJECT INFORMATION
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = 'excused', remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (remarks, current_time, subject_code, subject_name, room, section_id, existing_record['id']))
                        action_type = "updated"
                    else:
                        # ✅ FIXED: INSERT WITH SUBJECT INFORMATION
                        cursor.execute("""
                            INSERT INTO attendance 
                            (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (student_id, student_name, current_time, 'student', 'excused', 'manual_excuse', remarks, subject_code, subject_name, room, section_id))
                        action_type = "marked"
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    # Update frontend status
                    student_status[student_id] = 'excused'
                    
                    logger.info(f"📝 REGULAR STUDENT EXCUSED: {student_name} ({student_id}) - Subject: {subject_code}")
                    return jsonify({
                        'success': True, 
                        'title': 'Student Excused',
                        'message': f'Student {student_name} has been {action_type} as excused'
                    })
                
                else:
                    # TEMPORARY STUDENT
                    # Try exact match with temp_id
                    cursor.execute("""
                        SELECT name FROM attendance 
                        WHERE student_id IS NULL 
                        AND session_id = 'manual_add'
                        AND DATE(timestamp) = %s
                        AND remarks = %s
                        LIMIT 1
                    """, (today, f"temp_id:{student_id}"))
                    
                    temp_student = cursor.fetchone()
                    
                    if not temp_student:
                        # Try broader search
                        cursor.execute("""
                            SELECT name FROM attendance 
                            WHERE student_id IS NULL 
                            AND session_id = 'manual_add'
                            AND DATE(timestamp) = %s
                            AND (name LIKE %s OR remarks LIKE %s)
                            LIMIT 1
                        """, (today, f"%{student_id}%", f"%{student_id}%"))
                        
                        temp_student = cursor.fetchone()
                    
                    if temp_student:
                        # ✅ FIXED: UPDATE TEMPORARY STUDENT WITH SUBJECT INFORMATION
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = 'excused', remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE student_id IS NULL 
                            AND session_id = 'manual_add'
                            AND DATE(timestamp) = %s
                            AND (name LIKE %s OR remarks LIKE %s)
                        """, (remarks, current_time, subject_code, subject_name, room, section_id, today, f"%{student_id}%", f"%{student_id}%"))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        # Update frontend status
                        student_status[student_id] = 'excused'
                        
                        logger.info(f"📝 TEMPORARY STUDENT EXCUSED: {temp_student['name']} ({student_id}) - Subject: {subject_code}")
                        return jsonify({
                            'success': True, 
                            'title': 'Student Excused',
                            'message': f'Student {temp_student["name"]} has been marked as excused'
                        })
                    
                    else:
                        # STUDENT NOT FOUND
                        cursor.close()
                        conn.close()
                        logger.error(f"❌ STUDENT NOT FOUND: {student_id}")
                        return jsonify({
                            'success': False, 
                            'title': 'Student Not Found',
                            'message': f'No student found with ID: {student_id}'
                        })
            
            except Exception as e:
                conn.rollback()
                cursor.close()
                conn.close()
                logger.error(f"❌ DATABASE ERROR IN EXCUSED: {e}")
                return jsonify({
                    'success': False, 
                    'title': 'Database Error',
                    'message': 'Could not update attendance record'
                })
        
        elif action == 'mark_present':
            student_id = student_data.get('student_id')
            status = student_data.get('status', 'present')  # Can be 'present', 'late', or 'absent'
            remarks = student_data.get('remarks', 'Manually marked')
            
            # Validate status
            if status not in ['present', 'late', 'absent']:
                return jsonify({
                    'success': False, 
                    'title': 'Invalid Status',
                    'message': 'Status must be present, late, or absent'
                })
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            today = datetime.now().strftime("%Y-%m-%d")
            
            # ✅ GET SESSION SUBJECT INFORMATION
            subject_code = 'Unknown Subject'
            subject_name = 'Unknown Subject'
            room = 'Unknown Room'
            section_id = None
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT subject_code, subject_name, room, section_id 
                    FROM attendance_sessions 
                    WHERE session_id = %s
                """, (current_session_id,))
                session_result = cursor.fetchone()
                
                if session_result:
                    subject_code = session_result.get('subject_code', 'Unknown Subject')
                    subject_name = session_result.get('subject_name', 'Unknown Subject')
                    room = session_result.get('room', 'Unknown Room')
                    section_id = session_result.get('section_id')
                
                cursor.close()
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch session subject info: {e}")
            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            try:
                # Check if student exists
                cursor.execute("""
                    SELECT first_name, last_name 
                    FROM students 
                    WHERE student_id = %s AND status = 'active'
                """, (student_id,))
                student = cursor.fetchone()
                
                if student:
                    # REGULAR STUDENT
                    student_name = f"{student['first_name']} {student['last_name']}"
                    
                    # Check for existing record today
                    cursor.execute("""
                        SELECT id FROM attendance 
                        WHERE student_id = %s AND DATE(timestamp) = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (student_id, today))
                    
                    existing_record = cursor.fetchone()
                    
                    if existing_record:
                        # ✅ FIXED: UPDATE WITH SUBJECT INFORMATION
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = %s, remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (status, remarks, current_time, subject_code, subject_name, room, section_id, existing_record['id']))
                        action_type = "updated"
                    else:
                        # ✅ FIXED: INSERT WITH SUBJECT INFORMATION
                        cursor.execute("""
                            INSERT INTO attendance 
                            (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (student_id, student_name, current_time, 'student', status, 'manual_status', remarks, subject_code, subject_name, room, section_id))
                        action_type = "marked"
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    # Update frontend status
                    student_status[student_id] = status
                    
                    logger.info(f"🔄 MANUAL STATUS: {student_name} -> {status} - Subject: {subject_code}")
                    return jsonify({
                        'success': True, 
                        'title': 'Status Updated',
                        'message': f'Student {student_name} {action_type} as {status}'
                    })
                else:
                    # TEMPORARY STUDENT
                    cursor.execute("""
                        SELECT name FROM attendance 
                        WHERE student_id IS NULL 
                        AND session_id = 'manual_add'
                        AND DATE(timestamp) = %s
                        AND (name LIKE %s OR remarks LIKE %s)
                        LIMIT 1
                    """, (today, f"%{student_id}%", f"%temp_id:{student_id}%"))
                    
                    temp_student = cursor.fetchone()
                    
                    if temp_student:
                        # ✅ FIXED: UPDATE TEMPORARY STUDENT WITH SUBJECT INFORMATION
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = %s, remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE student_id IS NULL 
                            AND session_id = 'manual_add'
                            AND DATE(timestamp) = %s
                            AND (name LIKE %s OR remarks LIKE %s)
                        """, (status, remarks, current_time, subject_code, subject_name, room, section_id, today, f"%{student_id}%", f"%temp_id:{student_id}%"))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        student_status[student_id] = status
                        
                        logger.info(f"🔄 TEMPORARY STUDENT STATUS: {temp_student['name']} -> {status} - Subject: {subject_code}")
                        return jsonify({
                            'success': True, 
                            'title': 'Status Updated',
                            'message': f'Temporary student {temp_student["name"]} marked as {status}'
                        })
                    else:
                        cursor.close()
                        conn.close()
                        return jsonify({
                            'success': False, 
                            'title': 'Student Not Found',
                            'message': 'Student not found in system'
                        })
                        
            except Exception as e:
                conn.rollback()
                cursor.close()
                conn.close()
                logger.error(f"Error in mark_present: {e}")
                return jsonify({
                    'success': False, 
                    'title': 'Error',
                    'message': 'Failed to update status'
                })
            
        else:
            return jsonify({
                'success': False, 
                'title': 'Invalid Action',
                'message': 'The requested action is not valid'
            })
            
    except Exception as e:
        logger.error(f"Error in manage_student: {e}")
        return jsonify({
            'success': False, 
            'title': 'System Error',
            'message': 'An error occurred while processing your request'
        })
    
@app.route('/api/get_all_students')
def get_all_students():
    """Get ALL students from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT student_id as id, first_name, last_name, course, year_section, status
            FROM students 
            WHERE status = 'active'
            ORDER BY course, year_section, last_name, first_name
        """)
        
        students = cursor.fetchall()
        
        # Format student names
        formatted_students = []
        for student in students:
            formatted_students.append({
                'id': student['id'],
                'name': f"{student['first_name']} {student['last_name']}",
                'course': student['course'],
                'year_section': student['year_section'],
                'status': student['status']
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'students': formatted_students
        })
        
    except Exception as e:
        logger.error(f"Error getting all students: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/cleanup_temp_students')
def cleanup_temp_students():
    """Clean up temporary students without IDs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM attendance 
            WHERE student_id IS NULL 
            AND session_id = 'manual_add'
            AND (remarks IS NULL OR remarks NOT LIKE 'temp_id:%')
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Cleaned up {deleted_count} temporary students without IDs'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug-function')
def debug_function():
    """Check which get_db_connection function is being used"""
    import inspect
    
    # Get the current get_db_connection function
    current_func = get_db_connection
    
    # Get function details
    func_info = {
        'function_name': current_func.__name__,
        'file_defined': inspect.getfile(current_func),
        'source_lines': inspect.getsource(current_func).split('\n')[0] + ' ...'
    }
    
    return jsonify(func_info)

@app.route('/debug-admin-passwords')
def debug_admin_passwords():
    """Check admin passwords in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT admin_id, email, password_hash FROM admins")
        admins = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        result = []
        for admin in admins:
            result.append({
                'admin_id': admin['admin_id'],
                'email': admin['email'],
                'password_hash': admin['password_hash'],
                'hash_preview': admin['password_hash'][:50] + '...'
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/fix-connection')
def fix_connection():
    """Emergency fix for the connection function"""
    import types
    
    # Define the proper function
    def proper_get_db_connection():
        try:
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                database='facesys',
                charset='utf8mb4',
                autocommit=True
            )
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    # Replace the lambda with the proper function
    global get_db_connection
    get_db_connection = proper_get_db_connection
    
    return "✅ Connection function fixed! Try logging in again."

@app.route('/reset-admin-password')
def reset_admin_password():
    """Reset admin password to a known value"""
    try:
        new_password = "admin123"
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE admins SET password_hash = %s WHERE email = 'admin@wmsu.edu.ph'",
            (hashed_password,)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return f"""
        <h2>✅ Admin Password Reset</h2>
        <p>New password: <strong>{new_password}</strong></p>
        <p>Try logging in with: <strong>admin@wmsu.edu.ph</strong> and password: <strong>{new_password}</strong></p>
        <a href="/">Go to Login</a>
        """
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_session_start_time():
    """Get the session start time"""
    global session_start_time
    return session_start_time

# Add this function to set session start time (if needed separately)
def set_session_start_time():
    """Set the session start time when class begins"""
    global session_start_time
    session_start_time = datetime.now()
    logger.info(f"🕐 SESSION START TIME SET: {session_start_time}")


@app.route('/api/adjust_session_time', methods=['POST'])
def adjust_session_time():
    global session_total_duration_seconds, session_threshold_seconds
    
    data = request.get_json()
    schedule_id = data.get('schedule_id') # Used as session_id and user_id
    adj_type = data.get('adjustment_type')
    adj_minutes = data.get('adjustment_minutes', 0)
    elapsed_minutes = data.get('elapsed_minutes', 0)

    if not all([schedule_id, adj_type, isinstance(adj_minutes, int)]):
        return jsonify({'success': False, 'message': 'Invalid input data.'}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
    cursor = connection.cursor(dictionary=True)
    
    try:
        # 1. Fetch current LIVE values from attendance_sessions
        cursor.execute("""
            SELECT total_duration_minutes, late_threshold_minutes,
                   session_duration_seconds_total, threshold_seconds_total
            FROM attendance_sessions 
            WHERE session_id = %s AND status = 'active'
        """, (schedule_id,))
        live_session = cursor.fetchone()
        
        if not live_session:
             return jsonify({'success': False, 'message': 'Active session not found.'}), 404

        # 🎯 FIXED: Use seconds if available, otherwise convert
        if live_session.get('session_duration_seconds_total'):
            current_duration_seconds = live_session['session_duration_seconds_total']
            current_threshold_seconds = live_session['threshold_seconds_total']
            current_duration_minutes = current_duration_seconds // 60
            current_threshold_minutes = current_threshold_seconds // 60
        else:
            current_duration_minutes = live_session['total_duration_minutes']
            current_threshold_minutes = live_session['late_threshold_minutes']
            current_duration_seconds = current_duration_minutes * 60
            current_threshold_seconds = current_threshold_minutes * 60
        
        # 2. Calculate new values
        new_duration_minutes = current_duration_minutes
        new_threshold_minutes = current_threshold_minutes
        
        if adj_type == 'duration':
            new_duration_minutes += adj_minutes
        elif adj_type == 'threshold':
            new_threshold_minutes += adj_minutes

        new_duration_seconds = new_duration_minutes * 60
        new_threshold_seconds = new_threshold_minutes * 60

        # 3. Backend Constraints Check
        if new_duration_seconds <= 0:
            return jsonify({'success': False, 'message': 'Duration must be positive.'}), 400
        
        elapsed_seconds = elapsed_minutes * 60
        if adj_type == 'duration' and adj_minutes < 0 and new_duration_seconds < elapsed_seconds:
            return jsonify({'success': False, 'message': f'New Duration ({new_duration_minutes} min) cannot be less than the elapsed time ({elapsed_minutes} min).'}), 400

        if new_threshold_seconds >= new_duration_seconds:
            return jsonify({'success': False, 'message': 'Late Threshold must be less than the Class Duration.'}), 400

        # 4. Perform the DUAL UPDATE
        
        # A) Update the LIVE session data in attendance_sessions (BOTH minutes and seconds)
        if adj_type == 'duration':
            update_live_sql = """
                UPDATE attendance_sessions 
                SET total_duration_minutes = %s, session_duration_seconds_total = %s
                WHERE session_id = %s
            """
            cursor.execute(update_live_sql, (new_duration_minutes, new_duration_seconds, schedule_id))
        elif adj_type == 'threshold':
            update_live_sql = """
                UPDATE attendance_sessions 
                SET late_threshold_minutes = %s, threshold_seconds_total = %s
                WHERE session_id = %s
            """
            cursor.execute(update_live_sql, (new_threshold_minutes, new_threshold_seconds, schedule_id))
            
        
        # B) Update the DEFAULT settings in session_settings (UPSERT: Update or Insert)
        upsert_config_sql = """
        INSERT INTO session_settings (user_id, class_duration, late_threshold, video_quality)
        VALUES (%s, %s, %s, '720')
        ON DUPLICATE KEY UPDATE 
            class_duration = VALUES(class_duration),
            late_threshold = VALUES(late_threshold),
            updated_at = CURRENT_TIMESTAMP()
        """
        cursor.execute(upsert_config_sql, (schedule_id, new_duration_minutes, new_threshold_minutes))
        
        connection.commit()

        # 🎯 CRITICAL: Update global variables
        if adj_type == 'duration':
            session_total_duration_seconds = new_duration_seconds
        elif adj_type == 'threshold':
            session_threshold_seconds = new_threshold_seconds

        logger.info(f"🎯 TIME ADJUSTED: {adj_type} -> Duration: {new_duration_seconds}s, Threshold: {new_threshold_seconds}s")

        # 5. Return the newly set values (in both minutes and seconds)
        return jsonify({
            'success': True, 
            'message': f'Successfully adjusted {adj_type}.',
            'new_duration_minutes': new_duration_minutes,
            'new_threshold_minutes': new_threshold_minutes,
            'new_duration_seconds': new_duration_seconds,
            'new_threshold_seconds': new_threshold_seconds
        })

    except Exception as e:
        connection.rollback()
        logger.error(f"Database error during adjustment: {e}")
        return jsonify({'success': False, 'message': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """
    API 3: Finalizes the session, saves summary statistics INCLUDING DURATION
    """
    print("🎯 DEBUG: /api/end_session endpoint HIT!")
    
    data = request.get_json()
    session_data = data.get('session_data')
    unrecognized_faces = data.get('unrecognized_faces', [])
    session_id = data.get('session_id')

    print(f"🔍 DEBUG end_session called with session_id: {session_id}")

    if not session_id:
        return jsonify({'success': False, 'message': 'Missing session_id.'}), 400

    try:
        with get_db_cursor() as cursor:
            print(f"🔍 DEBUG Database connection established")
            
            # 1. Get session start time AND SUBJECT INFO first
            print(f"🔍 DEBUG Getting session start time and subject info")
            cursor.execute("SELECT started_at, subject_code, subject_name, room FROM attendance_sessions WHERE session_id = %s", (session_id,))
            session_result = cursor.fetchone()
            
            if not session_result:
                print(f"❌ DEBUG Session not found: {session_id}")
                print(f"❌ DEBUG Available sessions in database:")
                # List all available sessions for debugging
                cursor.execute("SELECT session_id, status FROM attendance_sessions ORDER BY started_at DESC LIMIT 10")
                all_sessions = cursor.fetchall()
                for session in all_sessions:
                    print(f"   - {session['session_id'] if isinstance(session, dict) else session[0]} (status: {session['status'] if isinstance(session, dict) else session[1]})")
                
                return jsonify({
                    'success': False, 
                    'message': f'Session not found: {session_id}',
                    'available_sessions': [s['session_id'] if isinstance(s, dict) else s[0] for s in all_sessions]
                }), 404
            
            started_at = session_result['started_at'] if isinstance(session_result, dict) else session_result[0]
            subject_code = session_result['subject_code'] if isinstance(session_result, dict) else session_result[1]
            subject_name = session_result['subject_name'] if isinstance(session_result, dict) else session_result[2]
            room = session_result['room'] if isinstance(session_result, dict) else session_result[3]
            
            print(f"🔍 DEBUG Session started at: {started_at}")
            print(f"🔍 DEBUG Subject info - Code: {subject_code}, Name: {subject_name}, Room: {room}")
            
            # 2. Calculate duration as TIME format (HH:MM:SS)
            from datetime import datetime, timedelta
            ended_at = datetime.now()
            duration_timedelta = ended_at - started_at
            
            # Convert to hours, minutes, seconds
            total_seconds = int(duration_timedelta.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            # Format as TIME string (HH:MM:SS)
            duration_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            print(f"🔍 DEBUG Calculated duration: {duration_time} (HH:MM:SS)")
            
            # 3. Finalize Unrecognized Faces
            print(f"🔍 DEBUG Processing {len(unrecognized_faces)} unrecognized faces")
            for face in unrecognized_faces:
                face_status = face.get('status', 'skipped')
                unrecognized_id = face.get('unrecognized_face_id') 

                if unrecognized_id:
                    try:
                        unrecognized_sql = """
                            UPDATE unrecognized_faces
                            SET final_status = %s, notes = %s
                            WHERE id = %s AND session_id = %s;
                        """
                        notes = face.get('notes', f'Final status set to {face_status} during session end.')
                        cursor.execute(unrecognized_sql, (face_status, notes, unrecognized_id, session_id))
                        print(f"🔍 DEBUG Updated unrecognized face: {unrecognized_id}")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not update unrecognized face {unrecognized_id}: {e}")
                        continue

            # 4. Get section_id for this session
            print(f"🔍 DEBUG Getting section_id for session: {session_id}")
            cursor.execute("SELECT section_id FROM attendance_sessions WHERE session_id = %s", (session_id,))
            session_result = cursor.fetchone()
            
            section_id = session_result['section_id'] if isinstance(session_result, dict) else session_result[0]
            print(f"🔍 DEBUG Found section_id: {section_id}")

            # 5. Get total enrolled students in this section
            print(f"🔍 DEBUG Getting total enrolled students for section: {section_id}")
            cursor.execute("SELECT COUNT(*) as count FROM students WHERE section_id = %s", (section_id,))
            total_enrolled_result = cursor.fetchone()
            total_enrolled = total_enrolled_result['count'] if isinstance(total_enrolled_result, dict) else total_enrolled_result[0]
            print(f"🔍 DEBUG Total enrolled students: {total_enrolled}")

            # 6. Get actual attendance counts from attendance table
            print(f"🔍 DEBUG Getting attendance counts for session: {session_id}")
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_attended,
                    SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late_count,
                    SUM(CASE WHEN status = 'excused' THEN 1 ELSE 0 END) as excused_count
                FROM attendance 
                WHERE session_id = %s AND person_type = 'student'
            """, (session_id,))
            attendance_stats = cursor.fetchone()
            
            # Handle dictionary cursor and NULL values
            if attendance_stats:
                if isinstance(attendance_stats, dict):
                    present_count = attendance_stats['present_count'] or 0
                    late_count = attendance_stats['late_count'] or 0
                    excused_count = attendance_stats['excused_count'] or 0
                    total_attended = attendance_stats['total_attended'] or 0
                else:
                    present_count = attendance_stats[1] or 0
                    late_count = attendance_stats[2] or 0
                    excused_count = attendance_stats[3] or 0
                    total_attended = attendance_stats[0] or 0
            else:
                present_count = late_count = excused_count = total_attended = 0

            print(f"🔍 DEBUG Current attendance - Present: {present_count}, Late: {late_count}, Excused: {excused_count}, Total Attended: {total_attended}")

            # 7. Calculate absent count
            absent_count = max(0, total_enrolled - present_count - late_count - excused_count)
            print(f"🔍 DEBUG Calculated absent count: {absent_count}")

            # 8. MARK ABSENT STUDENTS WITH SUBJECT INFORMATION
            if absent_count > 0:
                print(f"🔍 DEBUG Marking {absent_count} students as absent")
                try:
                    # Get students who are NOT in attendance table for this session
                    cursor.execute("""
                        SELECT s.student_id, s.first_name, s.last_name 
                        FROM students s 
                        WHERE s.section_id = %s 
                        AND s.student_id NOT IN (
                            SELECT student_id FROM attendance WHERE session_id = %s
                        )
                    """, (section_id, session_id))
                    absent_students = cursor.fetchall()
                    
                    print(f"🔍 DEBUG Found {len(absent_students)} students to mark as absent")
                    
                    # Insert absent records for each missing student WITH SUBJECT INFORMATION
                    for student in absent_students:
                        student_id = student['student_id'] if isinstance(student, dict) else student[0]
                        first_name = student['first_name'] if isinstance(student, dict) else student[1]
                        last_name = student['last_name'] if isinstance(student, dict) else student[2]
                        
                        cursor.execute("""
                            INSERT INTO attendance 
                            (student_id, person_type, name, timestamp, status, session_id, section_id, subject_code, subject_name, room)
                            VALUES (%s, 'student', %s, NOW(), 'absent', %s, %s, %s, %s, %s)
                        """, (student_id, f"{first_name} {last_name}", session_id, section_id, subject_code, subject_name, room))
                    
                    print(f"🔍 DEBUG Successfully inserted {len(absent_students)} absent records with subject info")
                    
                except Exception as e:
                    print(f"❌ ERROR inserting absent records: {e}")
                    return jsonify({
                        'success': False,
                        'message': f'Failed to insert absent records: {str(e)}'
                    }), 500

            # 9. Get FINAL counts after inserting absent records
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_attended,
                    SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late_count,
                    SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent_count,
                    SUM(CASE WHEN status = 'excused' THEN 1 ELSE 0 END) as excused_count
                FROM attendance 
                WHERE session_id = %s AND person_type = 'student'
            """, (session_id,))
            final_stats = cursor.fetchone()
            
            if final_stats:
                if isinstance(final_stats, dict):
                    final_present = final_stats['present_count'] or 0
                    final_late = final_stats['late_count'] or 0
                    final_absent = final_stats['absent_count'] or 0
                    final_excused = final_stats['excused_count'] or 0
                else:
                    final_present = final_stats[1] or 0
                    final_late = final_stats[2] or 0
                    final_absent = final_stats[3] or 0
                    final_excused = final_stats[4] or 0
            else:
                final_present = final_late = final_absent = final_excused = 0

            print(f"🔍 DEBUG Final counts - Present: {final_present}, Late: {final_late}, Absent: {final_absent}, Excused: {final_excused}")

            # 10. Update attendance_sessions with FINAL data INCLUDING DURATION
            print(f"🔍 DEBUG Updating attendance_sessions table with duration")
            summary_sql = """
                UPDATE attendance_sessions
                SET ended_at = NOW(), status = 'completed',
                    total_students = %s, present_count = %s, absent_count = %s, 
                    late_count = %s, excused_count = %s, duration_time = %s
                WHERE session_id = %s;
            """
            cursor.execute(summary_sql, (
                total_enrolled,
                final_present,
                final_absent,
                final_late,
                final_excused,
                duration_time,  # STORE AS TIME FORMAT
                session_id
            ))
            print(f"🔍 DEBUG Updated attendance_sessions successfully with duration: {duration_time}")

        print(f"✅ DEBUG Session ended successfully")
        return jsonify({
            'success': True, 
            'message': 'Session ended successfully.',
            'stats': {
                'total': total_enrolled,
                'present': final_present,
                'absent': final_absent,
                'late': final_late,
                'excused': final_excused,
                'duration': duration_time  # RETURN DURATION
            }
        }), 200

    except Exception as e:
        print(f"\n❌ ERROR in /api/end_session:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': f'A server error occurred: {str(e)}'
        }), 500

# MODIFIED: New endpoint to GET both live and static settings
@app.route('/api/get_session_settings', methods=['GET'])
def get_session_settings():
    """Retrieve class settings: live duration/threshold and default video quality."""
    global session_total_duration_seconds, session_threshold_seconds
    
    schedule_id = request.args.get('schedule_id')
    if not schedule_id:
        return jsonify({'success': False, 'message': 'Schedule ID required'}), 400
        
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # 1. Get LIVE settings from attendance_sessions
        cursor.execute("""
            SELECT total_duration_minutes, late_threshold_minutes,
                   session_duration_seconds_total, threshold_seconds_total
            FROM attendance_sessions
            WHERE session_id = %s AND status = 'active'
        """, (schedule_id,))
        live_settings = cursor.fetchone()
        
        # 2. Get DEFAULT settings from session_settings
        cursor.execute("""
            SELECT class_duration, late_threshold, video_quality
            FROM session_settings
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """, (schedule_id,))
        default_config = cursor.fetchone()
        
        # 🎯 FIXED: Use seconds if available, otherwise convert minutes to seconds
        if live_settings:
            # Prefer seconds values if they exist in database
            if live_settings.get('session_duration_seconds_total'):
                live_duration_seconds = live_settings['session_duration_seconds_total']
                live_threshold_seconds = live_settings['threshold_seconds_total']
                live_duration_minutes = live_duration_seconds // 60
                live_threshold_minutes = live_threshold_seconds // 60
            else:
                # Fallback to minutes conversion
                live_duration_minutes = live_settings['total_duration_minutes']
                live_threshold_minutes = live_settings['late_threshold_minutes']
                live_duration_seconds = live_duration_minutes * 60
                live_threshold_seconds = live_threshold_minutes * 60
        else:
            # Use default config or fallback values
            live_duration_minutes = default_config['class_duration'] if default_config else 60
            live_threshold_minutes = default_config['late_threshold'] if default_config else 15
            live_duration_seconds = live_duration_minutes * 60
            live_threshold_seconds = live_threshold_minutes * 60
            
            logger.warning(f"⚠️ No active attendance session found for {schedule_id}. Using default/config values.")

        # 🎯 CRITICAL: Update global variables
        session_total_duration_seconds = live_duration_seconds
        session_threshold_seconds = live_threshold_seconds
        
        video_quality = default_config['video_quality'] if default_config else '720'

        logger.info(f"🎯 SESSION SETTINGS LOADED: Duration={live_duration_seconds}s ({live_duration_minutes}min), Threshold={live_threshold_seconds}s ({live_threshold_minutes}min)")

        return jsonify({
            'success': True,
            'settings': {
                'live_duration_minutes': live_duration_minutes,
                'live_threshold_minutes': live_threshold_minutes,
                'live_duration_seconds': live_duration_seconds,
                'live_threshold_seconds': live_threshold_seconds,
                'video_quality': video_quality
            }
        })
            
    except Exception as e:
        logger.error(f"❌ Error in get_session_settings: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if connection: connection.close()

# New endpoint for static video quality update
@app.route('/api/update_video_quality', methods=['POST'])
def update_video_quality():
    data = request.get_json()
    schedule_id = data.get('schedule_id')
    video_quality = data.get('video_quality')

    if not all([schedule_id, video_quality]):
        return jsonify({'success': False, 'message': 'Missing required fields.'}), 400

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Update or Insert (UPSERT) video_quality in session_settings
        cursor.execute("""
        INSERT INTO session_settings (user_id, video_quality)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE 
            video_quality = VALUES(video_quality),
            updated_at = CURRENT_TIMESTAMP()
        """, (schedule_id, video_quality))
        
        connection.commit()
        logger.info(f"🎯 VIDEO QUALITY UPDATED: {schedule_id} -> {video_quality}")
        return jsonify({'success': True, 'message': 'Video Quality updated.'})

    except Exception as e:
        connection.rollback()
        logger.error(f"Error updating video quality: {e}")
        return jsonify({'success': False, 'message': 'Failed to update video quality.'}), 500
    finally:
        if cursor: cursor.close()
        if connection: connection.close()

# 🎯 NEW: Function to initialize session timing
def initialize_session_timing(schedule_id):
    """Initialize session timing when session starts"""
    global session_start_time, session_total_duration_seconds, session_threshold_seconds
    global detectionStopped

     # 🎯 RESET DETECTION FLAG WHEN STARTING NEW SESSION
    detectionStopped = False
    logger.info("🟢 Detection enabled for new session")
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get session settings
        cursor.execute("""
            SELECT total_duration_minutes, late_threshold_minutes,
                   session_duration_seconds_total, threshold_seconds_total
            FROM attendance_sessions 
            WHERE session_id = %s AND status = 'active'
        """, (schedule_id,))
        
        session_data = cursor.fetchone()
        
        if session_data:
            # Use seconds if available, otherwise convert minutes
            if session_data.get('session_duration_seconds_total'):
                session_total_duration_seconds = session_data['session_duration_seconds_total']
                session_threshold_seconds = session_data['threshold_seconds_total']
            else:
                session_total_duration_seconds = session_data['total_duration_minutes'] * 60
                session_threshold_seconds = session_data['late_threshold_minutes'] * 60
            
            # Set session start time
            session_start_time = datetime.now()
            
            logger.info(f"🎯 SESSION TIMING INITIALIZED: Start={session_start_time}, Duration={session_total_duration_seconds}s, Threshold={session_threshold_seconds}s")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"❌ Error initializing session timing: {e}")

# ------------------------------------------------------------------
# API Route 1: FETCH ABSENT STUDENTS (FIXED)
# ------------------------------------------------------------------

@app.route('/api/absent_students_for_enrollment', methods=['GET'])
def get_absent_students():
    """
    API 1: Fetches students in the section NOT yet marked PRESENT or LATE.
    """
    session_id = request.args.get('session_id')
    section_id = request.args.get('section_id')

    # 🎯 ADD DEBUG LOGGING
    print(f"🔍 DEBUG /api/absent_students_for_enrollment:")
    print(f"   session_id: {session_id}")
    print(f"   section_id: {section_id}")

    # CRITICAL: Handle 'undefined' values from frontend
    if not session_id or not section_id or session_id == 'undefined' or section_id == 'undefined':
        print("❌ ERROR: Missing or invalid session_id/section_id")
        return jsonify({
            'success': False, 
            'message': 'Missing valid session_id or section_id.',
            'debug_received': {
                'session_id': session_id, 
                'section_id': section_id
            }
        }), 400

    try:
        with get_db_cursor() as cursor:
            # 1. Get IDs of students already marked PRESENT or LATE.
            present_sql = """
                SELECT student_id FROM session_attendance
                WHERE session_id = %s AND status IN ('present', 'late');
            """
            cursor.execute(present_sql, (session_id,))
            
            fetched_results = cursor.fetchall()
            print(f"🔍 DEBUG: Found {len(fetched_results)} present/late students")
            
            present_ids = []
            if fetched_results:
                if isinstance(fetched_results[0], dict):
                    present_ids = [s['student_id'] for s in fetched_results]
                else:
                    present_ids = [s[0] for s in fetched_results]
            
            print(f"🔍 DEBUG: Present IDs: {present_ids}")
            
            # 2. Build the query to get all students in the section
            if present_ids:
                # Use parameterized query to avoid SQL injection
                placeholders = ', '.join(['%s'] * len(present_ids))
                absent_sql = f"""
                    SELECT student_id, first_name, last_name
                    FROM students
                    WHERE section_id = %s AND student_id NOT IN ({placeholders})
                    ORDER BY last_name, first_name;
                """
                params = [section_id] + present_ids
            else:
                absent_sql = """
                    SELECT student_id, first_name, last_name
                    FROM students
                    WHERE section_id = %s
                    ORDER BY last_name, first_name;
                """
                params = [section_id]
            
            print(f"🔍 DEBUG: Executing SQL with params: {params}")
            cursor.execute(absent_sql, tuple(params))
            absent_students = cursor.fetchall()
            
            print(f"🔍 DEBUG: Found {len(absent_students)} absent students")
            
            # Convert to proper JSON format if needed
            if absent_students and not isinstance(absent_students[0], dict):
                # Convert tuples to dicts
                absent_students = [
                    {
                        'student_id': s[0], 
                        'first_name': s[1], 
                        'last_name': s[2]
                    }
                    for s in absent_students
                ]
            
            print(f"✅ SUCCESS: Returning {len(absent_students)} absent students")
            return jsonify(absent_students), 200

    except Exception as e:
        print(f"❌ ERROR in /api/absent_students_for_enrollment: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': f'Database error: {str(e)}'
        }), 500

# ------------------------------------------------------------------
# API Route 2: ENROLL UNKNOWN FACE 
# ------------------------------------------------------------------

@app.route('/api/enroll_unknown_face', methods=['POST'])
def enroll_unknown_face():
    data = request.get_json()
    student_id = data.get('student_id')
    face_encoding = data.get('face_encoding')
    unrecognized_face_id = data.get('unrecognized_face_id')  # This should be DATABASE ID
    session_id = data.get('session_id')

    if not all([student_id, face_encoding, unrecognized_face_id, session_id]):
        return jsonify({'success': False, 'message': 'Missing required fields.'}), 400
    
    try:
        with get_db_cursor() as cursor:
            # 2. Save New Encoding (Face Learning)
            encoding_sql = """
                INSERT INTO student_face_encodings (student_id, face_encoding, source)
                VALUES (%s, %s, %s);
            """
            # NOTE: Ensure 'face_encoding' is serialized (e.g., to JSON string) if your DB field requires it.
            cursor.execute(encoding_sql, (student_id, face_encoding, 'manual_enrollment'))

            # Optional: Update the main students table with this encoding (best/latest)
            student_update_sql = "UPDATE students SET face_encoding = %s, updated_at = NOW() WHERE student_id = %s;"
            cursor.execute(student_update_sql, (face_encoding, student_id))

            # 3. Update Attendance (Mark as Present)
            attendance_sql = """
                INSERT INTO session_attendance (session_id, student_id, status, time_recorded)
                VALUES (%s, %s, 'present', NOW())
                ON DUPLICATE KEY UPDATE status = 'present', time_recorded = NOW();
            """
            cursor.execute(attendance_sql, (session_id, student_id))

            # 4. Update Unrecognized Face (Cleanup)
            unrecognized_sql = """
                UPDATE unrecognized_faces
                SET final_status = 'enrolled', notes = %s
                WHERE id = %s AND session_id = %s;  # ✅ Now matches database records
            """
            notes = f'Manually enrolled by instructor as {student_id}'
            cursor.execute(unrecognized_sql, (notes, unrecognized_face_id, session_id))

            remove_unknown_face(unrecognized_face_id)


        return jsonify({'success': True, 'message': 'Enrollment successful! Student marked as Present.'}), 200

    except Exception as e:
        print(f"Error enrolling unknown face: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': f'A server error occurred during enrollment: {str(e)}'
        }), 500

# ------------------------------------------------------------------
# API Route 3: END SESSION
# ------------------------------------------------------------------

@app.route('/api/debug_session', methods=['GET'])
def debug_session():
    session_id = request.args.get('session_id')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check session
        cursor.execute("SELECT * FROM attendance_sessions WHERE session_id = %s", (session_id,))
        session = cursor.fetchone()
        
        # Check attendance records
        cursor.execute("SELECT * FROM attendance WHERE session_id = %s", (session_id,))
        attendance_records = cursor.fetchall()
        
        # Check students count if session exists
        if session and session.get('section_id'):
            cursor.execute("SELECT COUNT(*) as count FROM students WHERE section_id = %s", (session['section_id'],))
            students_count = cursor.fetchone()
        else:
            students_count = {'count': 0}
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'session': session,
            'attendance_records_count': len(attendance_records),
            'attendance_records': attendance_records,
            'students_count': students_count['count']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}) 

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    global detectionStopped
    detectionStopped = True
    print("🔴 Detection stopped via API - skipSelected called")
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/api/resume_detection', methods=['POST'])
def resume_detection():
    global detectionStopped
    detectionStopped = False
    print("🟢 Detection resumed via API")
    return jsonify({'success': True, 'message': 'Detection resumed'})

# ------------------------------------------------------------------
# API Route 4: GET UNRECOGNIZED FACES (Fixed Version)
# ------------------------------------------------------------------

@app.route('/api/unrecognized_faces', methods=['GET'])
def get_unrecognized_faces():
    """
    Retrieves a list of unknown faces for enrollment with robust error handling.
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT
    
    print("🔍 DEBUG: /api/unrecognized_faces called")
    
    try:
        # 🛑 Clean up BEFORE processing
        cleanup_unrecognized_faces()
        
        # Initialize if not exists or invalid type
        if not isinstance(UNKNOWN_FACES_FOR_ENROLLMENT, dict):
            UNKNOWN_FACES_FOR_ENROLLMENT = {}
        
        unrecognized_list = []
        
        # Handle empty dictionary case
        if not UNKNOWN_FACES_FOR_ENROLLMENT:
            print("ℹ️ INFO: No unrecognized faces found")
            return jsonify(unrecognized_list), 200

        print(f"🔍 DEBUG: Processing {len(UNKNOWN_FACES_FOR_ENROLLMENT)} faces")
        
        processed_count = 0
        for unique_id, face_data in UNKNOWN_FACES_FOR_ENROLLMENT.items():
            try:
                # Validate face_data structure
                if not isinstance(face_data, dict):
                    print(f"⚠️ WARN: Face {unique_id} is not a dict")
                    continue
                
                face_crop_img = face_data.get('face_crop')
                timestamp_obj = face_data.get('timestamp')
                face_encoding = face_data.get('face_encoding')

                # Skip if essential data is missing
                if face_crop_img is None or face_encoding is None:
                    print(f"⚠️ WARN: Face {unique_id} missing crop or encoding")
                    continue

                # Image Conversion: Handle numpy arrays
                base64_image = None
                
                # Handle numpy array (OpenCV image)
                if hasattr(face_crop_img, 'shape') and hasattr(face_crop_img, 'dtype'):
                    try:
                        if face_crop_img.size == 0:
                            print(f"⚠️ WARN: Face {unique_id} has empty image")
                            continue
                            
                        is_success, buffer = cv2.imencode('.jpg', face_crop_img)
                        if not is_success:
                            print(f"⚠️ WARN: Face {unique_id} failed to encode")
                            continue
                            
                        base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                    except Exception as img_error:
                        print(f"⚠️ WARN: Face {unique_id} image processing error: {img_error}")
                        continue
                else:
                    print(f"⚠️ WARN: Face {unique_id} is not numpy array")
                    continue

                if not base64_image:
                    print(f"⚠️ WARN: Face {unique_id} no base64 image generated")
                    continue

                # Timestamp Formatting
                timestamp_str = "N/A"
                if timestamp_obj and isinstance(timestamp_obj, datetime):
                    timestamp_str = timestamp_obj.strftime('%H:%M:%S')

                # Face Encoding: Ensure it's serializable
                serializable_encoding = None
                try:
                    if face_encoding is not None:
                        if hasattr(face_encoding, 'tolist'):
                            serializable_encoding = face_encoding.tolist()
                        elif isinstance(face_encoding, (list, tuple)):
                            serializable_encoding = list(face_encoding)
                        else:
                            serializable_encoding = face_encoding
                except Exception as encoding_error:
                    print(f"⚠️ WARN: Face {unique_id} encoding error: {encoding_error}")
                    serializable_encoding = None

                # Only include if we have valid encoding
                if serializable_encoding is not None:
                    unrecognized_list.append({
                        'unrecognized_face_id': str(unique_id),
                        'face_encoding': serializable_encoding,
                        'base64_image': base64_image,
                        'timestamp': timestamp_str
                    })
                    processed_count += 1
                else:
                    print(f"⚠️ WARN: Face {unique_id} has no serializable encoding")

            except Exception as face_error:
                print(f"❌ ERROR processing face {unique_id}: {face_error}")
                continue

        print(f"✅ SUCCESS: Returning {processed_count} processed faces")
        return jsonify(unrecognized_list), 200

    except Exception as global_error:
        print(f"❌ GLOBAL ERROR in /api/unrecognized_faces: {global_error}")
        traceback.print_exc()
        # Return empty list but with 200 status to prevent frontend crashes
        return jsonify([]), 200
    
# 🛑 HELPER FUNCTION TO CLEAN UP THE GLOBAL VARIABLE
def cleanup_unrecognized_faces():
    """
    Enhanced cleanup with aggressive duplicate removal
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT, ACTIVE_FACE_TRACKS
    
    if not isinstance(UNKNOWN_FACES_FOR_ENROLLMENT, dict):
        UNKNOWN_FACES_FOR_ENROLLMENT = {}
        return
    
    current_time = datetime.now()
    keys_to_remove = []
    seen_encodings = {}
    
    # 1. Remove old faces and duplicates
    for unique_id, face_data in UNKNOWN_FACES_FOR_ENROLLMENT.items():
        try:
            if not isinstance(face_data, dict):
                keys_to_remove.append(unique_id)
                continue
                
            face_crop = face_data.get('face_crop')
            face_encoding = face_data.get('face_encoding')
            timestamp = face_data.get('timestamp')
            
            # Remove if essential data missing
            if face_crop is None or face_encoding is None:
                keys_to_remove.append(unique_id)
                continue
            
            # Remove if too old (older than 10 minutes)
            if timestamp and (current_time - timestamp).total_seconds() > 600:
                keys_to_remove.append(unique_id)
                continue
            
            # Remove exact duplicates using encoding signature
            encoding_signature = get_encoding_signature(face_encoding)
            if encoding_signature in seen_encodings:
                print(f"🧹 Removing duplicate face: {unique_id}")
                keys_to_remove.append(unique_id)
            else:
                seen_encodings[encoding_signature] = unique_id
                
        except Exception as e:
            print(f"⚠️ Error processing face {unique_id}: {e}")
            keys_to_remove.append(unique_id)
    
    # 2. Remove from dictionary
    for key in keys_to_remove:
        if key in UNKNOWN_FACES_FOR_ENROLLMENT:
            del UNKNOWN_FACES_FOR_ENROLLMENT[key]
    
    # 3. Cleanup old tracks (not seen for 2 minutes)
    tracks_to_remove = []
    for track_id, track_data in ACTIVE_FACE_TRACKS.items():
        last_seen = track_data.get('last_seen')
        if last_seen and (current_time - last_seen).total_seconds() > 120:
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        if track_id in ACTIVE_FACE_TRACKS:
            del ACTIVE_FACE_TRACKS[track_id]
    
    if keys_to_remove or tracks_to_remove:
        print(f"🧹 Cleaned {len(keys_to_remove)} faces, {len(tracks_to_remove)} tracks")

def is_similar_to_unrecognized_face(new_encoding, current_time):
    """
    STRICT similarity check - if similar face exists AND in cooldown, BLOCK it
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT
    
    for unique_id, face_data in UNKNOWN_FACES_FOR_ENROLLMENT.items():
        existing_encoding = face_data.get('face_encoding')
        cooldown_until = face_data.get('cooldown_until')

        if existing_encoding is None:
            continue

        # Check if this is the same face
        distance = calculate_face_distance(existing_encoding, new_encoding)
        
        # If very similar (same person)
        if distance < FACE_SIMILARITY_THRESHOLD:
            # 🛑 STRICT: If cooldown is active, BLOCK this face completely
            if cooldown_until and cooldown_until > current_time:
                remaining = (cooldown_until - current_time).total_seconds()
                print(f"🚫 BLOCKED: Face {unique_id} in cooldown ({remaining:.0f}s left)")
                return True, unique_id
            else:
                # Cooldown expired - update the existing face with NEW 30-second cooldown
                print(f"🔄 UPDATING: Face {unique_id} cooldown expired, resetting to 30s")
                face_data.update({
                    'face_crop': face_data.get('face_crop'),
                    'timestamp': current_time,
                    'cooldown_until': current_time + timedelta(seconds=30),  # RESET to 30 seconds
                    'times_seen': face_data.get('times_seen', 0) + 1
                })
                return True, unique_id

    return False, None

def remove_unknown_face(face_id):
    """
    Remove a face from the system (when enrolled or skipped)
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT, ACTIVE_FACE_TRACKS
    
    if face_id in UNKNOWN_FACES_FOR_ENROLLMENT:
        # Also remove any active track associated with this face
        face_data = UNKNOWN_FACES_FOR_ENROLLMENT[face_id]
        track_id = face_data.get('track_id')
        
        if track_id and track_id in ACTIVE_FACE_TRACKS:
            del ACTIVE_FACE_TRACKS[track_id]
        
        del UNKNOWN_FACES_FOR_ENROLLMENT[face_id]
        print(f"🗑️ REMOVED: Face {face_id} from system")
        return True
    
    print(f"⚠️ Face {face_id} not found in system")
    return False

def calculate_face_distance(encoding1, encoding2):
    """Calculate face distance with proper error handling"""
    try:
        enc1 = np.array(encoding1)
        enc2 = np.array(encoding2)
        
        # Ensure encodings are the same length
        if len(enc1) != len(enc2):
            return 1.0
            
        return np.linalg.norm(enc1 - enc2)
    except Exception as e:
        print(f"⚠️ Error calculating face distance: {e}")
        return 1.0
    
def get_encoding_signature(face_encoding):
    """Create a signature for face encoding to detect duplicates"""
    try:
        if hasattr(face_encoding, 'tolist'):
            encoding_list = face_encoding.tolist()
        else:
            encoding_list = list(face_encoding)
        
        # Use first 5 elements rounded to 3 decimals for signature
        return tuple(round(x, 3) for x in encoding_list[:5])
    except Exception:
        return str(face_encoding)


def add_unknown_face(face_crop, face_encoding, track_id=None):
    """
    Enhanced function to add unknown faces with STRICT 30-second cooldown per face
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT, ACTIVE_FACE_TRACKS
    
    current_time = datetime.now()
    
    # 🆕 GET SESSION ID FOR DATABASE SAVING
    session_id = get_current_session_id()
    if session_id is None:
        logger.error("❌ Cannot save face: No active session ID")
        return False
    
    # 1. STRICT CHECK: Is this face already in our system AND still in cooldown?
    is_duplicate, existing_face_id = is_similar_to_unrecognized_face(face_encoding, current_time)
    if is_duplicate:
        print(f"⏳ COOLDOWN ACTIVE: Face {existing_face_id} won't appear again for 30 seconds")
        return False
    
    # 2. Generate unique ID based on face encoding
    face_id = generate_face_id(face_encoding)
    
    # 3. 🆕 CRITICAL: SAVE TO DATABASE TABLE
    db_face_id = None
    try:
        with get_db_cursor() as cursor:
            # Convert face crop to bytes for database
            success, buffer = cv2.imencode('.jpg', face_crop)
            if success:
                face_image_bytes = buffer.tobytes()
                
                sql = """
                    INSERT INTO unrecognized_faces 
                    (session_id, face_image, final_status, created_at)
                    VALUES (%s, %s, 'pending', %s)
                """
                cursor.execute(sql, (session_id, face_image_bytes, current_time))
                db_face_id = cursor.lastrowid  # Get the database ID
                
                logger.info(f"💾 SAVED: Face to database with ID: {db_face_id} for session: {session_id}")
                
    except Exception as e:
        logger.error(f"❌ DATABASE ERROR: Failed to save face: {e}")
        return False
    
    # 4. Save to memory (your existing logic)
    UNKNOWN_FACES_FOR_ENROLLMENT[face_id] = {
        'face_crop': face_crop,
        'face_encoding': face_encoding,
        'timestamp': current_time,
        'cooldown_until': current_time + timedelta(seconds=30),
        'track_id': track_id,
        'times_seen': 1,
        'db_id': db_face_id  # 🆕 Store database reference
    }
    
    logger.info(f"➕ ADDED: New unknown face {face_id} - Database ID: {db_face_id}")
    return True

def generate_face_id(face_encoding):
    """Generate consistent face ID based on encoding, not timestamp"""
    # Use first 10 elements of encoding to generate ID
    encoding_str = ''.join([f"{x:.4f}" for x in face_encoding[:10]])
    face_hash = hash(encoding_str) % 10000
    return f"face-{abs(face_hash)}"

def background_cleanup():
    """Run cleanup in background every minute"""
    while True:
        try:
            cleanup_unrecognized_faces()
            print(f"🕒 Background cleanup: {len(UNKNOWN_FACES_FOR_ENROLLMENT)} faces, {len(ACTIVE_FACE_TRACKS)} tracks")
        except Exception as e:
            print(f"Background cleanup error: {e}")
        time.sleep(60)  # 1 minute

# Start background cleanup thread (add this at the bottom)
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

@app.route('/api/remove_unknown_face', methods=['POST'])
def remove_unknown_face_api():
    """
    API to remove a face from unknown faces system when skipped
    """
    data = request.get_json()
    face_id = data.get('face_id')
    
    print(f"🔍 REMOVE API called for face_id: {face_id}")
    
    if not face_id:
        return jsonify({'success': False, 'message': 'Missing face_id'}), 400
    
    try:
        remove_unknown_face(face_id)
        return jsonify({'success': True, 'message': 'Face removed from system'}), 200
    except Exception as e:
        print(f"❌ Error removing face: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def remove_unknown_face(face_id):
    """
    Remove a face from the system (when enrolled or skipped)
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT, ACTIVE_FACE_TRACKS
    
    if face_id in UNKNOWN_FACES_FOR_ENROLLMENT:
        # Also remove any active track associated with this face
        face_data = UNKNOWN_FACES_FOR_ENROLLMENT[face_id]
        track_id = face_data.get('track_id')
        
        if track_id and track_id in ACTIVE_FACE_TRACKS:
            del ACTIVE_FACE_TRACKS[track_id]
        
        del UNKNOWN_FACES_FOR_ENROLLMENT[face_id]
        print(f"🗑️ REMOVED: Face {face_id} from system")
        return True
    
    print(f"⚠️ Face {face_id} not found in system")
    return False

@app.route('/api/student_left', methods=['POST'])
def student_left():
    """Record when a student leaves the classroom - FIXED: Update status to missing but preserve original status"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_id = data.get('session_id')
        
        if not student_id or not session_id:
            return jsonify({'success': False, 'message': 'Missing student_id or session_id'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student name
        cursor.execute("SELECT first_name, last_name FROM students WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()
        student_name = f"{student['first_name']} {student['last_name']}" if student else f"Student {student_id}"
        
        # 🎯 CRITICAL FIX: Check if there's an active missing period
        cursor.execute("""
            SELECT id FROM missing_periods 
            WHERE student_id = %s AND session_id = %s AND returned = FALSE
        """, (student_id, session_id))
        
        existing_missing = cursor.fetchone()
        
        if existing_missing:
            logger.info(f"⏭️ Student {student_name} already marked as missing")
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student already marked as missing'}), 400
        
        # 🎯 FIXED: Get the CURRENT status before updating to missing
        cursor.execute("""
            SELECT status FROM attendance 
            WHERE student_id = %s AND session_id = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (student_id, session_id))
        
        current_attendance = cursor.fetchone()
        original_status = current_attendance['status'] if current_attendance else None
        
        # 🎯 FIXED: Start new missing period AND record original status
        cursor.execute("""
            INSERT INTO missing_periods (student_id, session_id, missing_start, returned, original_status)
            VALUES (%s, %s, NOW(), FALSE, %s)
        """, (student_id, session_id, original_status))
        
        # 🎯 FIXED: Update attendance status to 'missing'
        cursor.execute("""
            UPDATE attendance 
            SET status = 'missing', timestamp = NOW()
            WHERE student_id = %s AND session_id = %s
        """, (student_id, session_id))
        
        # If no record was updated, create a new one
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO attendance (student_id, name, timestamp, person_type, status, session_id)
                VALUES (%s, %s, NOW(), 'student', 'missing', %s)
            """, (student_id, student_name, session_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"📤 STUDENT LEFT: {student_name} ({student_id}) - Status changed to MISSING (was {original_status})")
        
        return jsonify({
            'success': True, 
            'message': f'{student_name} marked as missing (was {original_status})'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in student_left: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/student_returned', methods=['POST'])
def student_returned():
    """Record when a student returns - FIXED: Use original_status to restore correct status"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_id = data.get('session_id', current_session_id)
        
        logger.info(f"↩️ STUDENT_RETURNED CALLED: student_id={student_id}, session_id={session_id}")
        
        if not student_id:
            return jsonify({'success': False, 'message': 'Missing student_id'}), 400
        
        if not session_id:
            return jsonify({'success': False, 'message': 'Missing session_id and no current session'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student name
        cursor.execute("SELECT first_name, last_name FROM students WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()
        student_name = f"{student['first_name']} {student['last_name']}" if student else f"Student {student_id}"
        
        # 🎯 FIXED: Get the missing period WITH original_status
        cursor.execute("""
            SELECT id, missing_start, original_status FROM missing_periods 
            WHERE student_id = %s AND session_id = %s AND returned = FALSE
            ORDER BY missing_start DESC LIMIT 1
        """, (student_id, session_id))
        
        missing_period = cursor.fetchone()
        
        if not missing_period:
            logger.warning(f"⚠️ No active missing period found for {student_name}")
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No active missing period found'}), 400
        
        # Calculate duration
        missing_start = missing_period['missing_start']
        missing_end = datetime.now()
        duration_seconds = int((missing_end - missing_start).total_seconds())
        original_status = missing_period['original_status']
        
        # 🎯 FIXED: Update missing_periods table
        cursor.execute("""
            UPDATE missing_periods 
            SET missing_end = %s, duration_seconds = %s, returned = TRUE
            WHERE id = %s
        """, (missing_end, duration_seconds, missing_period['id']))
        
        # 🎯 FIXED: RESTORE ORIGINAL STATUS in attendance table
        if original_status and original_status != 'missing':
            cursor.execute("""
                UPDATE attendance 
                SET status = %s, timestamp = NOW()
                WHERE student_id = %s AND session_id = %s
            """, (original_status, student_id, session_id))
            
            logger.info(f"🔄 RESTORED STATUS: {student_name} -> {original_status}")
        else:
            logger.warning(f"⚠️ No valid original_status found for {student_name}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Format duration for display
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        duration_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        logger.info(f"✅ SUCCESS: {student_name} returned after {duration_display} - Status: {original_status}")
        
        return jsonify({
            'success': True, 
            'message': f'{student_name} returned after {duration_display} - Status: {original_status}',
            'duration_seconds': duration_seconds,
            'duration_display': duration_display,
            'restored_status': original_status
        })
        
    except Exception as e:
        logger.error(f"❌ ERROR in student_returned: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/get_missing_students')
def get_missing_students():
    """Get list of students currently missing from class"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'message': 'Missing session_id'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                mp.student_id,
                s.first_name,
                s.last_name,
                mp.missing_start,
                TIMESTAMPDIFF(SECOND, mp.missing_start, NOW()) as missing_seconds
            FROM missing_periods mp
            JOIN students s ON mp.student_id = s.student_id
            WHERE mp.session_id = %s AND mp.returned = FALSE
            ORDER BY mp.missing_start DESC
        """, (session_id,))
        
        missing_students = cursor.fetchall()
        
        # Format the response
        formatted_students = []
        for student in missing_students:
            missing_seconds = student['missing_seconds']
            hours = missing_seconds // 3600
            minutes = (missing_seconds % 3600) // 60
            seconds = missing_seconds % 60
            duration_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            formatted_students.append({
                'student_id': student['student_id'],
                'name': f"{student['first_name']} {student['last_name']}",
                'missing_since': student['missing_start'].strftime('%H:%M:%S'),
                'missing_seconds': missing_seconds,
                'duration_display': duration_display
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'missing_students': formatted_students,
            'total_missing': len(formatted_students)
        })
        
    except Exception as e:
        logger.error(f"Error in get_missing_students: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/calculate_missing_duration')
def calculate_missing_duration():
    """Calculate total missing duration for a student in a session"""
    try:
        student_id = request.args.get('student_id')
        session_id = request.args.get('session_id')
        
        if not student_id or not session_id:
            return jsonify({'success': False, 'message': 'Missing student_id or session_id'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Calculate total missing time for this session
        cursor.execute("""
            SELECT 
                COALESCE(SUM(duration_seconds), 0) as total_missing_seconds,
                COUNT(*) as total_missing_periods
            FROM missing_periods 
            WHERE student_id = %s AND session_id = %s AND returned = TRUE
        """, (student_id, session_id))
        
        result = cursor.fetchone()
        
        total_seconds = result['total_missing_seconds'] or 0
        total_periods = result['total_missing_periods'] or 0
        
        # Format duration
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        duration_display = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'total_missing_seconds': total_seconds,
            'duration_display': duration_display,
            'total_missing_periods': total_periods
        })
        
    except Exception as e:
        logger.error(f"Error in calculate_missing_duration: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    # Initialize global variables
    latest_frame = None
    stop_flag = False
    camera_available = False
    use_dummy_feed = False
    dummy_frame = None
    
    # Initialize CSV file
    try:
        if not os.path.exists("attendance_log.csv"):
            with open("attendance_log.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'DateTime', 'Status'])
            logger.info("Initialized attendance_log.csv")
    except Exception as e:
        logger.error(f"Failed to initialize CSV file: {e}")
    
    # Try to connect to camera
    if not open_stream():
        logger.warning("Initial camera connection failed, but continuing with dummy feed")
    
    # Start the grabber thread
    grab_thread = threading.Thread(target=grabber, daemon=True)
    grab_thread.start()
    
    # Start periodic attendance save thread
    attendance_save_thread = threading.Thread(target=periodic_attendance_save, daemon=True)
    attendance_save_thread.start()
    
    try:
        # 🎯 FIXED SSL CONFIGURATION
        ssl_context = None
        cert_file = 'cert.pem'
        key_file = 'key.pem'
        
        # Check if SSL certificate files exist
        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_context = (cert_file, key_file)
            logger.info("🔐 SSL certificates found - Starting HTTPS server")
        else:
            logger.warning("⚠️ SSL certificates not found - Starting HTTP server")
          
        
        # Start server with proper configuration
        app.run(
            host="192.168.0.100",  # 🎯 FIXED: Use your specific IP
            port=5000,
            debug=False,
            threaded=True,
            ssl_context=ssl_context  # 🎯 FIXED: Only use SSL if certificates exist
        )
    
    except OSError as e:
        if "10049" in str(e) or "not valid" in str(e):
            logger.error("Network address issue. Trying localhost only...")
            app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, ssl_context=None)
        else:
            raise e
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    
    finally:
        # Signal all threads to stop
        stop_flag = True
        time.sleep(1)  # Give threads time to clean up
        
        # Release camera resource safely
        with cap_lock:
            if cap is not None:
                cap.release()
        
        # Final attendance save
        try:
            if attendance and ENABLE_RECOGNITION:
                with open("attendance_log.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Name', 'DateTime', 'Status'])
                    for sid, data in attendance.items():
                        writer.writerow([sid, data['name'], data['time'], 'present'])
                logger.info(f"Final save: {len(attendance)} records saved to attendance_log.csv")
        except Exception as e:
            logger.error(f"Final attendance save failed: {e}")
        
        logger.info("Application shutdown complete")