import csv
from functools import wraps
import hashlib
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
import supervision as sv
from typing import List, Dict, Any, Tuple



# =========================
# Flask streaming & API
# =========================
app = Flask(__name__)
app.secret_key = 'face-attendance-system-secret-key-2025'  
app.template_folder = 'templates'
app.static_folder = 'static'
CORS(app)

app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key-change-in-production')

# =========================
# OPTIMIZED CONFIG FOR FIXED CODE
# =========================

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

student_presence_tracker = {}  # Tracks when students are present/missing
current_session_id = None  

# Initialize FaceAnalysis with SCRFD
face_analysis = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analysis.prepare(ctx_id=0, det_size=(320, 320))  

FRAME_BUFFER = []
ACTIVE_FACE_TRACKS = {}
FACE_COOLDOWN_PERIOD = 30  # seconds
thread_safe_lock = threading.RLock()
FACE_SIMILARITY_THRESHOLD = 0.6

detectionStopped = False
current_fps = 30.0
frame_timestamps = []
skip_frame_counter = 0

session_start_time = None
session_total_duration_seconds = 3600 
session_threshold_seconds = 900       

pending_confirmations = {}  # {person_id: {'frames': [], 'body_boxes': [], 'name': str, 'type': str}}
locked_track_reid_features = {}
student_status = {}  # {student_id: 'absent' | 'present' | 'late'}
current_session_students = []  # List of student IDs for current class

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

ATTENDANCE_CSV_FILE = "attendance_log.csv"
attendance_save_interval = 300  # 5 minutes

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

# =========================
# ByteTrack Initialization - FIXED VERSION
# =========================
try:
    # Use the simpler initialization
    byte_tracker = ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )
    logger.info("✅ ByteTrack initialized successfully for body tracking")
    
    # Test ByteTrack with dummy detection - FIXED IMPORT
    test_detection = sv.Detections(  # ✅ FIXED: Changed from 'Detection' to 'Detections'
        xyxy=np.array([[100, 100, 200, 200]]),
        confidence=np.array([0.8]),
        class_id=np.array([0])
    )
    test_tracks = byte_tracker.update_with_detections(detections=test_detection)
    logger.info("✅ ByteTrack test passed - working correctly")
    
except Exception as e:
    logger.error(f"❌ ByteTrack initialization failed: {e}")
    # Create a simple fallback tracker
    class SimpleTracker:
        def __init__(self):
            self.track_id_counter = 0
            self.tracks = {}
            
        def update_with_detections(self, detections):
            tracks = []
            # Handle both array format and Detections object
            if hasattr(detections, 'xyxy'):
                # Detections object format
                for i, (xyxy, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                    track_id = self.track_id_counter + i
                    tracks.append(type('Track', (), {
                        'tracker_id': track_id,
                        'detection': np.concatenate([xyxy, [confidence]]),
                        'confidence': confidence
                    }))
            else:
                # Array format
                for i, detection in enumerate(detections):
                    track_id = self.track_id_counter + i
                    tracks.append(type('Track', (), {
                        'tracker_id': track_id,
                        'detection': detection,
                        'confidence': detection[4] if len(detection) > 4 else 0.8
                    }))
            self.track_id_counter += len(tracks)
            return tracks
    
    byte_tracker = SimpleTracker()
    logger.info("🔄 Using simple fallback tracker instead of ByteTrack")

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
    
    # ✅ ADD THIS RIGHT HERE - PROPER GPU SETUP:
    if torch.cuda.is_available():
        reid_model = reid_model.cuda()
        reid_model = reid_model.half()  # Use FP16 for maximum speed
        logger.info(f"✅ TorchReID model moved to GPU with FP16 precision")
    else:
        logger.info("⚠️ TorchReID model running on CPU")
    
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

    providers = []
    
    # Check if CUDA is available
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
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
            INSIGHTFACE_MODEL = model_name  
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
    
        frame[:] = (40, 40, 40)

        text = 'No Camera Connected'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

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
        logger.error(f"Database connection failed: {e}")
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
            conn.commit()  
    except Exception as e:
        conn.rollback()    
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
known_face_encodings = np.array([]) 
known_face_names = []
known_face_ids = []
known_face_types = []  # 'student' or 'faculty'
KNOWN_FACE_ENCODINGS_ARRAY = None 
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
finalize_known_faces() 

# =========================
# OPTIMIZED CAMERA CAPTURE
# =========================
cap_lock = threading.Lock()
cap = None
last_frame_time = time.time()

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
                except:
                    pass
                cap = None
            
            time.sleep(0.3)
                    
            logger.info(f"Connecting to RTSP: {rtsp_url}")
            
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  
            
            # Quick connection test
            start_time = time.time()
            while time.time() - start_time < 3:
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        camera_available = True
                        use_dummy_feed = False
                        current_rtsp_url = rtsp_url
                        logger.info(f"✅ Camera connected: 640x480 @ 30 FPS")
                        return True
                time.sleep(0.1)
            
            cap.release()
            cap = None
            raise Exception("Camera connection timeout")
                
    except Exception as e:
        logger.warning(f"❌ Camera connection failed: {e}")
        camera_available = False
        use_dummy_feed = True
        if cap is not None:
            cap.release()
            cap = None
        return False

def grabber():
    global latest_frame, stop_flag, camera_available, use_dummy_feed, cap, current_rtsp_url, last_frame_time
    empty_count = 0
    consecutive_failures = 0
    
    while not stop_flag:
        grab_start = time.time()
        
        if use_dummy_feed:
            latest_frame = create_dummy_frame()
            time.sleep(0.033)  # Consistent timing
            continue
            
        with cap_lock:
            if cap is None:
                time.sleep(0.01)
                continue
            ok, frame = cap.read()
            
        if not ok or frame is None:
            empty_count += 1
            consecutive_failures += 1
            
            if empty_count > 10 or consecutive_failures > 5:
                logger.warning("Camera connection issues, switching to dummy feed")
                camera_available = False
                use_dummy_feed = True
                consecutive_failures = 0
                
                # Attempt recovery in background
                if current_rtsp_url:
                    def attempt_recovery():
                        time.sleep(2.0)
                        logger.info("Attempting camera reconnection...")
                        open_stream(current_rtsp_url)
                    
                    recovery_thread = threading.Thread(target=attempt_recovery, daemon=True)
                    recovery_thread.start()
            else:
                time.sleep(0.01)
            continue
               
        empty_count = 0
        consecutive_failures = 0
        latest_frame = frame
        last_frame_time = time.time()
        
        grab_time = time.time() - grab_start
        sleep_time = max(0.001, 0.03 - grab_time)  # Target ~33 FPS grabbing
        time.sleep(sleep_time)

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
    """Mark attendance for both students and faculty - FIXED: Better manual status detection and duplicate prevention"""
    global session_start_time, current_session_id, session_threshold_seconds, session_total_duration_seconds
    global student_presence_tracker, student_status
    
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
    
    # 🎯 Update presence tracker (ALWAYS update this regardless of manual status)
    if type == 'student' and session_id:
        student_presence_tracker[id] = {
            'last_seen': current_time,
            'last_body_seen': current_time,
            'name': name,
            'present': True
        }
        logger.info(f"📍 PRESENCE TRACKER UPDATED: {name} is currently present")
    
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
    
    # 🎯 CRITICAL FIX: BETTER Manual Status Detection
    is_manual_status = False
    original_status = None
    existing_record_id = None
    existing_session_id = None
    existing_remarks = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 🎯 Get the MOST RECENT attendance record for this student in this session
        cursor.execute("""
            SELECT id, status, timestamp, session_id, remarks FROM attendance 
            WHERE student_id = %s AND session_id = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (id, session_id))
        existing_record = cursor.fetchone()
        
        if existing_record:
            existing_record_id = existing_record['id']
            original_status = existing_record['status']
            existing_session_id = existing_record.get('session_id')
            existing_remarks = existing_record.get('remarks', '')
            
            # 🎯 CONSISTENT MANUAL STATUS DETECTION (SAME AS ALL OTHER ENDPOINTS)
            manual_excuse_sessions = ['manual_excuse']  # Only for excused students
            manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
            manual_statuses = ['excused']  # Only truly manual statuses
            
            is_manual_status = (
                # Only specific manual session types (not manual_add)
                existing_session_id in manual_excuse_sessions or
                existing_session_id in manual_status_sessions or
                # Only specific manual statuses
                original_status in manual_statuses or
                # Only specific manual remarks (not temp_id)
                'Manually marked' in existing_remarks or
                'Manual status' in existing_remarks or
                'Manually marked as excused' in existing_remarks
            )
            
            if is_manual_status:
                logger.info(f"🔒 MANUAL STATUS DETECTED: {name} has manual status '{original_status}' (session: {existing_session_id}, remarks: {existing_remarks}) - PRESERVING")
            else:
                logger.info(f"🔄 AUTO STATUS: {name} has status '{original_status}' (session: {existing_session_id}) - CAN BE UPDATED")
            
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error checking existing status: {e}")
    
    if is_manual_status and original_status:
        if type == 'student':
            student_status[id] = original_status
        
        try:
            csv_file = "attendance_log.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['ID', 'Name', 'DateTime', 'Type', 'Status', 'SessionID', 'SectionID', 'SubjectCode', 'SubjectName', 'Room', 'MissingDuration', 'IsReturning', 'RestoredStatus', 'WasMissing', 'ManualStatus'])
                writer.writerow([id, name, time_str, type, original_status, session_id or 'N/A', section_id or 'N/A', subject_code or 'N/A', subject_name or 'N/A', room or 'N/A', 0, False, 'N/A', False, True])
            
            logger.info(f"📄 CSV saved: {name} - MANUAL STATUS PRESERVED: {original_status}")
        except Exception as e:
            logger.error(f"Failed to save attendance to CSV: {e}")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if existing_record_id:
                cursor.execute("""
                    UPDATE attendance 
                    SET timestamp = %s
                    WHERE id = %s
                """, (time_str, existing_record_id))
                logger.info(f"🔒 Updated timestamp only for {name} - Status '{original_status}' preserved")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update timestamp in database: {e}")
        
        # 🎯 CRITICAL: EARLY RETURN - don't process auto-detection logic
        return

    # 🎯 STANDARD STATUS DETERMINATION (only runs if NOT manual status)
    # Continue with your existing logic for non-manual statuses...
    missing_duration = 0
    is_returning_from_missing = False
    restored_original_status = None
    is_currently_missing = False
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 🎯 CHECK IF STUDENT IS CURRENTLY MISSING
        cursor.execute("""
            SELECT id, status FROM attendance 
            WHERE student_id = %s AND session_id = %s AND status = 'missing'
            ORDER BY timestamp DESC LIMIT 1
        """, (id, session_id))
        
        missing_record = cursor.fetchone()
        if missing_record:
            is_currently_missing = True
            logger.info(f"🎯 Student {name} is CURRENTLY MISSING")
        
        # 🎯 Check if student is currently in missing_periods
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
        logger.error(f"Error checking missing status: {e}")
    
    # 🎯 STATUS DETERMINATION LOGIC (your existing code continues here)
    if is_currently_missing:
        if is_returning_from_missing and restored_original_status in ['present', 'late']:
            status = restored_original_status
            logger.info(f"🔄 RESTORING ORIGINAL STATUS (from missing): {name} -> {status}")
        elif original_status in ['present', 'late']:
            status = original_status
            logger.info(f"🔄 RESTORING ORIGINAL STATUS (fallback from missing): {name} -> {status}")
        else:
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
        status = restored_original_status
        logger.info(f"🔄 RESTORING ORIGINAL STATUS: {name} returning from missing -> {status}")
        
    elif is_returning_from_missing and original_status in ['present', 'late']:
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
                writer.writerow(['ID', 'Name', 'DateTime', 'Type', 'Status', 'SessionID', 'SectionID', 'SubjectCode', 'SubjectName', 'Room', 'MissingDuration', 'IsReturning', 'RestoredStatus', 'WasMissing', 'ManualStatus'])
            writer.writerow([id, name, time_str, type, status, session_id or 'N/A', section_id or 'N/A', subject_code or 'N/A', subject_name or 'N/A', room or 'N/A', missing_duration, is_returning_from_missing, restored_original_status or 'N/A', is_currently_missing, False])
        
        logger.info(f"📄 CSV saved: {name} ({id}) - {type} - {status} - Missing: {missing_duration}s - Returning: {is_returning_from_missing} - Restored: {restored_original_status} - WasMissing: {is_currently_missing}")
    except Exception as e:
        logger.error(f"Failed to save attendance to CSV: {e}")
    
    # 🎯 Update database with proper status handling
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
                    # 🎯 Update with missing_duration and proper status
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
            results = body_detector(frame, classes=[0], verbose=False, conf=0.25, half=True)
        else:
            results = body_detector(frame, classes=[0], verbose=False, conf=0.25)
        
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
    FIXED: ByteTrack coordinate extraction - was treating bbox as track_id
    FIXED: Strong anti-swapping with strict ReID + spatial validation
    PRESERVED: Fast recognition and instant locking system
    """
    logger.info(f"🔍 refresh_with_detections CALLED - Frame {frame_idx}")
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
    # Step 2: Body detection
    body_detections = detect_bodies(frame)
    # Step 3: Use ByteTrack for BODY tracking - FIXED COORDINATE EXTRACTION
    body_tracks = []
    if len(body_detections) > 0:
        # Convert to proper ByteTrack format
        detections_list = []
        for body_det in body_detections:
            body_box = body_det['box']
            x1, y1, x2, y2 = body_box
            conf = body_det['confidence']
        
            # Filter small bodies
            if (x2 - x1) < 50 or (y2 - y1) < 100:
                continue
            
            detections_list.append([x1, y1, x2, y2, conf])
    
        if detections_list:
            detections_array = np.array(detections_list)
        
            try:
                # Use sv.Detections
                supervision_detections = sv.Detections(
                    xyxy=detections_array[:, :4],
                    confidence=detections_array[:, 4],
                    class_id=np.array([0] * len(detections_array))
                )
            
                tracks_byte = byte_tracker.update_with_detections(
                    detections=supervision_detections
                )
            
                if tracks_byte is not None and hasattr(tracks_byte, 'xyxy'):
                    # ByteTrack returns a Detections object with tracker_id attribute
                    xyxy = tracks_byte.xyxy
                    tracker_ids = tracks_byte.tracker_id if hasattr(tracks_byte, 'tracker_id') else None
                    confidences = tracks_byte.confidence if hasattr(tracks_byte, 'confidence') else np.array([0.8] * len(xyxy))
                    
                    if tracker_ids is not None:
                        for idx in range(len(xyxy)):
                            try:
                                x1, y1, x2, y2 = xyxy[idx]
                                track_id = int(tracker_ids[idx])
                                confidence = float(confidences[idx]) if idx < len(confidences) else 0.8
                                
                                body_box = (int(x1), int(y1), int(x2), int(y2))
                                
                                # Extract ReID features
                                reid_features = extract_reid_features(frame, body_box)
                                
                                body_tracks.append({
                                    'track_id': track_id,
                                    'body_box': body_box,
                                    'confidence': confidence,
                                    'reid_features': reid_features
                                })
                            except Exception as e:
                                logger.warning(f"Error processing track {idx}: {e}")
                                continue
                    else:
                        # No tracker_ids - fallback to index-based IDs
                        logger.warning("⚠️ ByteTrack returned detections without tracker_ids")
                        for idx in range(len(xyxy)):
                            x1, y1, x2, y2 = xyxy[idx]
                            confidence = float(confidences[idx]) if idx < len(confidences) else 0.8
                            
                            body_box = (int(x1), int(y1), int(x2), int(y2))
                            reid_features = extract_reid_features(frame, body_box)
                            
                            body_tracks.append({
                                'track_id': frame_idx * 1000 + idx,
                                'body_box': body_box,
                                'confidence': confidence,
                                'reid_features': reid_features
                            })
            
                logger.info(f"✅ ByteTrack: {len(body_tracks)} body tracks")
            
            except Exception as e:
                logger.error(f"❌ ByteTrack failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback with ReID features
                for i, det in enumerate(detections_list):
                    body_box = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                    reid_features = extract_reid_features(frame, body_box)
                
                    body_tracks.append({
                        'track_id': i + frame_idx * 1000,
                        'body_box': body_box,
                        'confidence': det[4],
                        'reid_features': reid_features
                    })
                logger.info(f"✅ Fallback: {len(body_tracks)} body tracks")
    # Filter out locked faces
    locked_body_boxes = []
    for person_id, lock_info in locked_tracks.items():
        body_box = lock_info.get('body_box')
        if body_box:
            bx1, by1, bx2, by2 = body_box
            expand_margin = 8
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
    dets = []
    face_embeddings = []
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
                            try:
                                embedding = face_obj.embedding
                                face_embeddings.append(embedding)
                            except:
                                face_embeddings.append(None)
                            
                            dets.append((x1, y1, x2, y2, conf, face_idx, face_obj))
    
        except Exception as e:
            logger.error(f"Error processing face detection (idx: {face_idx}): {e}")
            continue
    logger.info(f"Frame {frame_idx}: {len(faces)} faces detected → {len(dets)} NEW faces")
    # Step 4: ULTRA STRONG ReID MATCHING FOR LOCKED TRACKS (Anti-Swapping)
    new_tracks = []
    used_detections = set()
    used_body_tracks = set()
    # 🆕 IMPROVED: Ultra-strict ReID matching with multi-validation
    for person_id, lock_info in locked_tracks.items():
        last_reid_features = lock_info.get('reid_features')
        last_body_box = lock_info.get('body_box')
    
        if last_reid_features is not None and body_tracks:
            best_match_idx = None
            best_reid_distance = 0.18  # 🆕 ULTRA STRICT for maximum anti-swapping
            best_spatial_match = None
        
            # Try ReID matching with multi-stage validation
            for idx, body_track in enumerate(body_tracks):
                if idx in used_body_tracks:
                    continue
                
                current_reid_features = body_track['reid_features']
                if current_reid_features is None:
                    continue
            
                # Stage 1: Calculate ReID distance
                reid_dist = calculate_reid_distance(last_reid_features, current_reid_features)
                if reid_dist >= best_reid_distance:
                    continue
            
                # Stage 2: 🆕 STRICT SPATIAL CONSTRAINT - Movement limit
                if last_body_box:
                    movement = calculate_box_distance(last_body_box, body_track['body_box'])
                    # 🆕 Maximum 60 pixels movement (stricter than before)
                    if movement > 60:
                        continue
            
                # Stage 3: 🆕 STRICT SIZE CONSISTENCY - Prevent body size jumps
                if last_body_box:
                    last_area = (last_body_box[2] - last_body_box[0]) * (last_body_box[3] - last_body_box[1])
                    new_area = (body_track['body_box'][2] - body_track['body_box'][0]) * (body_track['body_box'][3] - body_track['body_box'][1])
                    
                    # Avoid division by zero
                    if max(last_area, new_area) == 0:
                        continue
                    
                    area_ratio = min(last_area, new_area) / max(last_area, new_area)
                    # 🆕 Body size must be within 70% similarity (stricter)
                    if area_ratio < 0.70:
                        continue
                
                # Stage 4: 🆕 ASPECT RATIO VALIDATION - Prevent shape distortion
                if last_body_box:
                    last_width = last_body_box[2] - last_body_box[0]
                    last_height = last_body_box[3] - last_body_box[1]
                    new_width = body_track['body_box'][2] - body_track['body_box'][0]
                    new_height = body_track['body_box'][3] - body_track['body_box'][1]
                    
                    if last_height > 0 and new_height > 0:
                        last_aspect = last_width / last_height
                        new_aspect = new_width / new_height
                        aspect_diff = abs(last_aspect - new_aspect)
                        # 🆕 Aspect ratio change must be minimal
                        if aspect_diff > 0.15:
                            continue
            
                # All validations passed - this is the best match
                best_reid_distance = reid_dist
                best_match_idx = idx
        
            if best_match_idx is not None:
                # Found matching body via strict ReID + spatial validation
                matched_body = body_tracks[best_match_idx]
                used_body_tracks.add(best_match_idx)
            
                lock_info['body_box'] = matched_body['body_box']
                lock_info['last_seen'] = frame_idx
                lock_info['missed_detections'] = 0
                
                if matched_body['reid_features'] is not None:
                    alpha = 0.92  #  Very high retention for maximum stability
                    lock_info['reid_features'] = (
                        alpha * last_reid_features +
                        (1 - alpha) * matched_body['reid_features']
                    )
            
                logger.info(f"✅ Locked {lock_info.get('name', person_id)} updated via ReID (dist: {best_reid_distance:.3f})")
            else:
                # No match - preserve last position and count miss
                lock_info['missed_detections'] = lock_info.get('missed_detections', 0) + 1
                logger.debug(f"❌ No ReID match for {lock_info.get('name', person_id)} - Preserving last detection")
    for person_id, lock_info in locked_tracks.items():
        lock_start = lock_info.get('lock_start', frame_idx)
        tracking_seconds = (frame_idx - lock_start) // 30
    
        locked_track_obj = {
            'id': person_id,
            'name': lock_info.get('name', f'Person {person_id}'),
            'type': lock_info.get('type', 'student'),
            'is_locked': True,
            'body_box': lock_info.get('body_box'),
            'last_seen': lock_info.get('last_seen', frame_idx),
            'confidence': 1.0,
            'tracking_duration': tracking_seconds,
            'lock_start': lock_start
        }
        new_tracks.append(locked_track_obj)
    for tr in tracks:
        if tr.get('id') in locked_tracks:
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
    # Step 5: Process NEW face detections (PRESERVED: Fast recognition)
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
        # Face Recognition (PRESERVED: Fast instant recognition)
        name = "Unknown"
        person_id = None
        ptype = None
        confidence = conf
        face_embedding = None
        best_similarity = 0
        if conf >= 0.15 and KNOWN_FACE_ENCODINGS_ARRAY is not None and KNOWN_FACE_ENCODINGS_ARRAY.size > 0:
            try:
                face_embedding = face_obj.embedding
            
                if len(known_face_names) == 0 or len(KNOWN_FACE_ENCODINGS_ARRAY) == 0:
                    logger.warning("No known faces loaded for recognition")
                    name = "Unknown"
                else:
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
                        if best_similarity >= 0.50:  # PRESERVED: Fast recognition threshold
                            name = known_face_names[best_match_index]
                            person_id = known_face_ids[best_match_index]
                            ptype = known_face_types[best_match_index]
                            confidence = min(1.0, (conf * 0.4) + (best_similarity * 0.6))
                        
                            if ptype == 'faculty':
                                name = f"Faculty: {name}"
                        
                            logger.info(f"✅ RECOGNITION MATCH: {name} ({person_id}) - Similarity: {best_similarity:.3f}")
                        else:
                            logger.info(f"❌ NO MATCH: Best similarity {best_similarity:.3f} < threshold 0.50")
                            name = "Unknown"
                        
            except Exception as e:
                logger.error(f"❌ Error in recognition: {e}")
                name = "Unknown"
        # Handle unknown faces
        if name == "Unknown" and face_embedding is not None:
            try:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0 and face_crop.shape[0] >= 30 and face_crop.shape[1] >= 30:
                    session_id = get_current_session_id()
                    if session_id:
                        success = add_unknown_face(face_crop, face_embedding, track_id=f"track-{frame_idx}-{x1}")
                        if success:
                            logger.info(f"📸 CAPTURED UNKNOWN FACE - Added to enrollment system")
            except Exception as e:
                logger.error(f"❌ Error capturing unknown face: {e}")
        # Match face to body with 🆕 ANTI-SWAP validation
        matched_body_track = None
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
    
        for body_track in body_tracks:
            if body_track['track_id'] in used_body_tracks:
                continue
            
            body_box = body_track['body_box']
            bx1, by1, bx2, by2 = body_box
        
            # Check if face is within upper body region
            body_height = by2 - by1
            upper_body_y1 = by1
            upper_body_y2 = by1 + int(body_height * 0.6)
        
            if (bx1 <= face_center_x <= bx2 and
                upper_body_y1 <= face_center_y <= upper_body_y2):
                
                # 🆕 ADDITIONAL: Validate face-body size ratio (anti-swap)
                face_area = (x2 - x1) * (y2 - y1)
                body_area = (bx2 - bx1) * (by2 - by1)
                face_body_ratio = face_area / body_area if body_area > 0 else 0
                
                # Face should be 1.5% to 20% of body area
                if 0.015 <= face_body_ratio <= 0.20:
                    matched_body_track = body_track
                    used_body_tracks.add(body_track['track_id'])
                    logger.info(f"✅ Face-Body matched: track_id={body_track['track_id']}, ratio={face_body_ratio:.3f}")
                    break
        # If still unknown after recognition attempt
        if person_id is None:
            unique_id = f"U-{frame_idx}-{x1}"
        
            if matched_body_track:
                track_obj = {
                    'id': unique_id,
                    'box': (x1, y1, x2, y2),
                    'body_box': matched_body_track['body_box'],
                    'byte_track_id': matched_body_track['track_id'],
                    'confidence': max(0.3, conf),
                    'last_seen': frame_idx,
                    'is_locked': False,
                    'name': "Unknown",
                    'type': 'unknown',
                    'start_frame': frame_idx,
                    'reid_features': matched_body_track['reid_features']
                }
                new_tracks.append(track_obj)
                logger.info(f"📍 Unknown face with body tracking")
            else:
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
                logger.info(f"📍 Unknown face without body")
        else:
            # Known person (PRESERVED: Fast locking)
            if matched_body_track and person_id not in locked_tracks:
                if person_id not in pending_confirmations:
                    pending_confirmations[person_id] = {
                        'id': person_id,
                        'frames': [],
                        'body_boxes': [],
                        'name': name,
                        'type': ptype,
                        'similarities': [],
                        'first_seen': frame_idx,
                        'last_seen': frame_idx,
                        'byte_track_id': matched_body_track['track_id'],
                        'reid_features': matched_body_track['reid_features']
                    }
                    logger.info(f"🆕 NEW PENDING: {name} (ID: {person_id})")
            
                pending_confirmations[person_id]['frames'].append(frame_idx)
                pending_confirmations[person_id]['body_boxes'].append(matched_body_track['body_box'])
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
            
                logger.info(f"🔄 PENDING PROGRESS: {name} - Frames: {consecutive_frames}, Avg Similarity: {avg_similarity:.3f}")
            
                # PRESERVED: Fast confirmation for instant locking
                if (consecutive_frames >= max(3, CONFIRMATION_FRAMES_REQUIRED - 2) and
                    avg_similarity >= CONFIRMATION_SIMILARITY_THRESHOLD - 0.05):
                
                    # LOCK with ReID features
                    locked_tracks[person_id] = {
                        'id': person_id,
                        'name': name,
                        'type': ptype,
                        'body_box': matched_body_track['body_box'],
                        'last_seen': frame_idx,
                        'reid_features': matched_body_track['reid_features'],
                        'lock_start': frame_idx,
                        'missed_detections': 0,
                        'byte_track_id': matched_body_track['track_id']
                    }
                
                    locked_track_obj = {
                        'id': person_id,
                        'name': name,
                        'type': ptype,
                        'is_locked': True,
                        'body_box': matched_body_track['body_box'],
                        'last_seen': frame_idx,
                        'confidence': 1.0,
                        'tracking_duration': 0,
                        'lock_start': frame_idx,
                        'byte_track_id': matched_body_track['track_id']
                    }
                    new_tracks.append(locked_track_obj)
                
                    mark_attendance(name, person_id, ptype)
                    del pending_confirmations[person_id]
                    logger.info(f"🔒 LOCKED & ATTENDANCE MARKED for {name} ({person_id})")
                else:
                    temp_track = {
                        'id': person_id,
                        'box': (x1, y1, x2, y2),
                        'body_box': matched_body_track['body_box'],
                        'confidence': confidence,
                        'last_seen': frame_idx,
                        'is_locked': False,
                        'name': name,
                        'is_pending': True,
                        'byte_track_id': matched_body_track['track_id']
                    }
                    new_tracks.append(temp_track)
            else:
                # No body match or already locked
                if person_id not in locked_tracks:
                    logger.info(f"⚠️ {name} recognized but no body match")
                new_tracks.append({
                    'id': person_id,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'last_seen': frame_idx,
                    'is_locked': False,
                    'name': name
                })
    # Confidence decay for unlocked tracks
    for tr in new_tracks[:]:
        if not tr.get('is_locked'):
            tr['confidence'] = max(0.1, tr.get('confidence', 0.5) * 0.85)
            if tr['confidence'] < 0.2 and frame_idx - tr.get('last_seen', 0) > 10:
                new_tracks.remove(tr)
    tracks[:] = new_tracks
    # Cleanup old unknown tracks
    current_tracks = [
        tr for tr in tracks
        if not (tr.get('name') == "Unknown" and frame_idx - tr.get('last_seen', 0) > 30)
    ]
    tracks[:] = current_tracks
    logger.info(f"Total: {len(tracks)} tracks (Locked: {len(locked_tracks)}, Pending: {len(pending_confirmations)})")

def update_trackers_with_body(rgb, frame, frame_idx):
    """
    Update trackers with body-only tracking for LOCKED tracks.
    ✅ FIXED: JSON serialization issues and better error handling
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
    
    # 🎯 CRITICAL FIX: Clean up student_presence_tracker from invalid entries at the start
    invalid_entries = []
    for student_id, track_info in list(student_presence_tracker.items()):
        if not student_id or not isinstance(student_id, str) or not student_id.startswith(('20', '19', '21')):
            invalid_entries.append(student_id)

    for invalid_id in invalid_entries:
        logger.warning(f"🚮 Removing invalid student_presence_tracker entry at start: {invalid_id}")
        del student_presence_tracker[invalid_id]
    
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
                    
                    lock_info['body_box'] = tuple(body_box)
                    lock_info['last_seen'] = frame_idx
                    lock_info['missed_detections'] = 0
                    
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
            
            # 🎯 FIXED: Validate student_id before proceeding
            if not student_id or not isinstance(student_id, str) or not student_id.startswith(('20', '19', '21')):
                logger.error(f"❌ INVALID STUDENT ID: {student_id} for {student_name} - skipping API call")
                if student_id in student_presence_tracker:
                    student_presence_tracker[student_id]['present'] = False
                    student_presence_tracker[student_id]['last_seen'] = current_time
                locked_tracks.pop(person_id, None)
                locked_track_reid_features.pop(person_id, None)
                logger.info(f"🔓 IMMEDIATE UNLOCK (INVALID ID): {person_id}")
                continue
            
            # 🎯 FIXED: BETTER Manual Status Detection BEFORE calling student_left API
            is_manual_status = False
            try:
                # First check if student has manual status
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT status, session_id, remarks FROM attendance 
                    WHERE student_id = %s AND session_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (student_id, current_session_id))
                
                attendance_record = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if attendance_record:
                    current_status = attendance_record['status']
                    current_session = attendance_record.get('session_id')
                    remarks = attendance_record.get('remarks') or ''  # Ensure string
                    
                    # 🎯 FIXED: Handle None values properly
                    if current_session is None:
                        current_session = ''
                    
                    # 🎯 BETTER MANUAL STATUS DETECTION - More specific
                    manual_excuse_sessions = ['manual_excuse']  # Only for excused students
                    manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
                    manual_statuses = ['excused']  # Only truly manual statuses
                    
                    is_manual_status = (
                        # Only specific manual session types (not manual_add)
                        current_session in manual_excuse_sessions or
                        current_session in manual_status_sessions or
                        # Only specific manual statuses
                        current_status in manual_statuses or
                        # Only specific manual remarks (not temp_id)
                        'Manually marked' in str(remarks) or  # FIXED: Convert to string
                        'Manual status' in str(remarks)       # FIXED: Convert to string
                    )
                    
                    if is_manual_status:
                        logger.info(f"🔒 SKIPPING MISSING: {student_name} has manual status '{current_status}'")
                        # Don't call student_left API for manual status students
                        # Just update presence tracker and continue
                        if student_id in student_presence_tracker:
                            student_presence_tracker[student_id]['present'] = False
                            student_presence_tracker[student_id]['last_seen'] = current_time
                        
                        locked_tracks.pop(person_id, None)
                        locked_track_reid_features.pop(person_id, None)
                        logger.info(f"🔓 IMMEDIATE UNLOCK (MANUAL STATUS): {person_id}")
                        continue  # Skip the API call entirely
                    else:
                        logger.info(f"🔄 AUTO STATUS: {student_name} can be marked as missing")
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking manual status: {e}")
            
            # 🎯 Only call student_left API if NOT manual status
            try:
                import requests
                import time
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                api_data = {
                    'student_id': str(student_id), 
                    'session_id': str(current_session_id)  
                }
                
                # Try 3 times with better connection handling
                success = False
                for attempt in range(3):
                    try:
                        response = requests.post(
                            'https://192.168.0.101:5000/api/student_left', 
                            json=api_data,  
                            timeout=3,
                            headers={
                                'Connection': 'close',
                                'Content-Type': 'application/json'
                            },
                            verify=False
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"✅ SUCCESS: student_left API called for {student_name}")
                            success = True
                            break
                        elif response.status_code == 400:
                            # Parse the actual error message
                            try:
                                error_data = response.json()
                                logger.warning(f"⚠️ API returned {response.status_code}: {error_data.get('message', 'Unknown error')}")
                                success = True  # Consider success if already marked
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
            
            if student_id in student_presence_tracker:
                student_presence_tracker[student_id]['present'] = False
                student_presence_tracker[student_id]['last_seen'] = current_time
        
        locked_tracks.pop(person_id, None)
        locked_track_reid_features.pop(person_id, None)
        logger.info(f"🔓 IMMEDIATE UNLOCK: {person_id}")
    
    if frame_idx % 15 == 0 and current_session_id:
        for student_id, track_info in list(student_presence_tracker.items()):
            # 🎯 CRITICAL FIX: Validate student_id is actually a string, not the built-in id function
            if student_id and isinstance(student_id, str) and student_id.startswith(('20', '19', '21')):  # Adjust pattern based on your student ID format
                if track_info.get('present'):
                    # 🎯 FIXED: Check if this student has any active locked track
                    has_active_body = False
                    for locked_id, lock_info in locked_tracks.items():
                        if lock_info.get('type') == 'student' and lock_info.get('id') == student_id:
                            if locked_id in locked_tracks_with_bodies:
                                has_active_body = True
                                break
                    
                    if not has_active_body:
                        last_body_seen = track_info.get('last_body_seen')
                        if last_body_seen:
                            time_since_body_seen = (current_time - last_body_seen).total_seconds()
                            
                            if time_since_body_seen > 3:  # Reduced to 3 seconds for backup
                                track_info['present'] = False
                                track_info['last_seen'] = current_time
                                
                                # 🎯 ADDED: BETTER Manual Status Detection before marking as missing
                                try:
                                    conn = get_db_connection()
                                    cursor = conn.cursor(dictionary=True)
                                    cursor.execute("""
                                        SELECT status, session_id, remarks FROM attendance 
                                        WHERE student_id = %s AND session_id = %s
                                        ORDER BY timestamp DESC LIMIT 1
                                    """, (student_id, current_session_id))
                                    
                                    attendance_record = cursor.fetchone()
                                    cursor.close()
                                    conn.close()
                                    
                                    if attendance_record:
                                        current_status = attendance_record['status']
                                        current_session = attendance_record.get('session_id')
                                        remarks = attendance_record.get('remarks') or ''
                                        
                                        # 🎯 BETTER MANUAL STATUS DETECTION
                                        manual_excuse_sessions = ['manual_excuse']  # Only for excused students
                                        manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
                                        manual_statuses = ['excused']  # Only truly manual statuses
                                        
                                        is_manual_status = (
                                            current_session in manual_excuse_sessions or
                                            current_session in manual_status_sessions or
                                            current_status in manual_statuses or
                                            'Manually marked' in str(remarks) or  # FIXED: Convert to string
                                            'Manual status' in str(remarks)       # FIXED: Convert to string
                                        )
                                        
                                        if is_manual_status:
                                            logger.info(f"🔒 SKIPPING MISSING: {track_info['name']} has manual status '{current_status}'")
                                            continue  # Skip API call for manual status students
                                        else:
                                            logger.info(f"🔄 AUTO STATUS: {track_info['name']} can be marked as missing")
                                    
                                except Exception as e:
                                    logger.warning(f"⚠️ Error checking manual status: {e}")
                                
                                # Only call API if not manual status
                                try:
                                    import requests
                                    import urllib3
                                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                                    
                                    # 🎯 FIXED: Proper data format with validation
                                    if student_id and isinstance(student_id, str) and student_id.startswith(('20', '19', '21')):
                                        api_data = {
                                            'student_id': str(student_id),
                                            'session_id': str(current_session_id)
                                        }
                                        response = requests.post(
                                            'http://localhost:5000/api/student_left',  # FIXED: Use localhost
                                            json=api_data,
                                            timeout=2
                                        )
                                        if response.status_code == 200:
                                            logger.info(f"📤 BACKUP MISSING: {track_info['name']}")
                                except Exception as e:
                                    logger.warning(f"⚠️ API call failed: {e}")
            else:
                # 🎯 CLEANUP: Remove invalid entries from student_presence_tracker
                logger.warning(f"🚮 Removing invalid student_presence_tracker entry: {student_id}")
                del student_presence_tracker[student_id]
        
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
            
            # 🎯 FIXED: Validate student_id before proceeding
            if not student_id or not isinstance(student_id, str) or not student_id.startswith(('20', '19', '21')):
                logger.error(f"❌ INVALID STUDENT ID: {student_id} for {student_name} - skipping API call")
                if student_id in student_presence_tracker:
                    student_presence_tracker[student_id]['present'] = False
                    student_presence_tracker[student_id]['last_seen'] = datetime.now()
                locked_tracks.pop(person_id, None)
                locked_track_reid_features.pop(person_id, None)
                logger.info(f"🔓 Unlocked track for {person_id} (INVALID ID)")
                continue
            
            # 🎯 FIXED: BETTER Manual Status Detection BEFORE calling student_left API
            is_manual_status = False
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT status, session_id, remarks FROM attendance 
                    WHERE student_id = %s AND session_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (student_id, current_session_id))
                
                attendance_record = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if attendance_record:
                    current_status = attendance_record['status']
                    current_session = attendance_record.get('session_id')
                    remarks = attendance_record.get('remarks') or ''  # Ensure string
                    
                    # 🎯 FIXED: Handle None values properly
                    if current_session is None:
                        current_session = ''
                    
                    # 🎯 BETTER MANUAL STATUS DETECTION
                    manual_excuse_sessions = ['manual_excuse']  # Only for excused students
                    manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
                    manual_statuses = ['excused']  # Only truly manual statuses
                    
                    is_manual_status = (
                        current_session in manual_excuse_sessions or
                        current_session in manual_status_sessions or
                        current_status in manual_statuses or
                        'Manually marked' in str(remarks) or  # FIXED: Convert to string
                        'Manual status' in str(remarks)       # FIXED: Convert to string
                    )
                    
                    if is_manual_status:
                        logger.info(f"🔒 SKIPPING MISSING: {student_name} has manual status '{current_status}'")
                        # Don't call student_left API for manual status students
                        # Just update presence tracker and continue
                        if student_id in student_presence_tracker:
                            student_presence_tracker[student_id]['present'] = False
                            student_presence_tracker[student_id]['last_seen'] = datetime.now()
                        
                        locked_tracks.pop(person_id, None)
                        locked_track_reid_features.pop(person_id, None)
                        logger.info(f"🔓 Unlocked track for {person_id} (MANUAL STATUS)")
                        continue  # Skip the API call entirely
                    else:
                        logger.info(f"🔄 AUTO STATUS: {student_name} can be marked as missing")
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking manual status: {e}")
            
            if student_id in student_presence_tracker:
                student_presence_tracker[student_id]['present'] = False
                student_presence_tracker[student_id]['last_seen'] = datetime.now()
            
            # 🎯 SIMPLE: Call API to mark as missing WITH RETRY LOGIC (only if not manual status)
            try:
                import requests
                import time
                
                # 🎯 FIXED: Proper data format
                api_data = {
                    'student_id': str(student_id),  # FIXED: Ensure string
                    'session_id': str(current_session_id)  # FIXED: Ensure string
                }
                
                # Try 3 times with better connection handling
                success = False
                for attempt in range(3):
                    try:
                        response = requests.post(
                            'https://192.168.0.101:5000/api/student_left',  # FIXED: Use localhost
                            json=api_data,  # FIXED: Use json parameter
                            timeout=3,
                            headers={'Connection': 'close'}
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"✅ Student marked as MISSING: {student_name}")
                            success = True
                            break
                        elif response.status_code == 400:
                            logger.warning(f"⚠️ API returned {response.status_code}: Student already marked as missing")
                            success = True  # Consider success
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
    
    current_tracks = []
    
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


@app.route('/video_feed')
def video_feed():
    """Stream video feed - RESTORED WORKING VERSION WITH DETECTION"""
    def generate():
        global latest_frame, tracks, locked_tracks, pending_confirmations, stop_flag
        frame_idx = 0
        last_processing_time = time.time()
        
        while not stop_flag:
            frame_start = time.time()
            
            try:
                # 🎯 ORIGINAL FRAME ACCESS - KEEP THIS
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                
                frame = latest_frame.copy()
                current_time = time.time()
                
                # 🎯 CRITICAL: PROCESS DETECTION EVERY FRAME
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 🎯 ALWAYS RUN DETECTION - DON'T SKIP FRAMES
                refresh_with_detections(frame, rgb, frame_idx)
                update_trackers_with_body(rgb, frame, frame_idx)
                
                last_processing_time = current_time
                frame_idx += 1
                
                # 🎯 OPTIMIZED ENCODING (keep this)
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 80,  # Good balance
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      frame_bytes + b'\r\n')
                
                # 🎯 REASONABLE TIMING CONTROL
                processing_time = time.time() - frame_start
                target_frame_time = 0.033  # ~30 FPS
                sleep_time = max(0.001, target_frame_time - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                time.sleep(0.01)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        
        # ✅ FIXED: CONSISTENT PRECISION HANDLING
        if torch.cuda.is_available():
            # Convert to FP16 and move to GPU
            body_crop_tensor = torch.from_numpy(
                np.transpose(body_crop_normalized, (2, 0, 1))
            ).unsqueeze(0).half().cuda()  # ✅ FIXED: Consistent FP16 on GPU
        else:
            # CPU fallback
            body_crop_tensor = torch.from_numpy(
                np.transpose(body_crop_normalized, (2, 0, 1))
            ).unsqueeze(0).float()
        
        with torch.no_grad():
            reid_features = reid_model(body_crop_tensor)
        
        # ✅ FIXED: Always convert to CPU numpy for consistency
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
    """Aggressive memory cleanup for smooth performance"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.debug("🧹 Aggressive memory optimization completed")

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

def check_detection_status():
    """Check why detection isn't running"""
    global detectionStopped, current_session_id, latest_frame
    
    logger.info(f"🔍 DETECTION STATUS CHECK:")
    logger.info(f"   - detectionStopped: {detectionStopped}")
    logger.info(f"   - current_session_id: {current_session_id}")
    logger.info(f"   - latest_frame: {'Available' if latest_frame is not None else 'None'}")
    logger.info(f"   - ENABLE_RECOGNITION: {ENABLE_RECOGNITION}")
    
    # Check camera
    try:
        if cap is not None:
            logger.info(f"   - Camera: Connected")
        else:
            logger.error("   - Camera: NOT CONNECTED")
    except:
        logger.error("   - Camera: ERROR")

# Call this function to see what's wrong
check_detection_status()

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
        program_id = request.args.get('program_id', '')
        year_level = request.args.get('year_level', '')
        section_id = request.args.get('section_id', '')
        curriculum_id = request.args.get('curriculum_id', '')  # ADD THIS
        status = request.args.get('status', 'active')
        search = request.args.get('search', '')
        page = int(request.args.get('page', 1))     
        limit = int(request.args.get('limit', 50))  
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # UPDATE: Added curricula join
        from_joins = """
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            JOIN curricula c ON s.curriculum_id = c.curriculum_id  # ADD THIS JOIN
        """
        
        where_conditions = ["s.status = %s"]
        params = [status]
        
        if program_id:
            where_conditions.append("p.program_id = %s") 
            params.append(program_id)

        if year_level:
            where_conditions.append("ys.year_level = %s")
            params.append(year_level)
            
        if section_id:
            where_conditions.append("s.section_id = %s") 
            params.append(section_id)
            
        # ADD THIS: Curriculum filter - THIS IS THE KEY FIX!
        if curriculum_id:
            where_conditions.append("s.curriculum_id = %s")
            params.append(curriculum_id)
            
        if search:
            where_conditions.append("(s.first_name LIKE %s OR s.last_name LIKE %s OR s.student_id LIKE %s OR s.email LIKE %s)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])
        
        where_clause = " AND ".join(where_conditions) 
        offset = (page - 1) * limit
        
        count_query = f"SELECT COUNT(s.student_id) as total {from_joins} WHERE {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['total']
        
        query = f"""
            SELECT 
                s.student_id, s.first_name, s.last_name, s.middle_name, 
                s.email, s.photo_path, s.status, s.created_at, s.updated_at,
                p.program_name AS course, 
                p.program_id,
                ys.section_name,
                ys.year_level,
                s.section_id,
                s.curriculum_id,  # ADD THIS
                c.curriculum_year,  # ADD THIS
                c.curriculum_name   # ADD THIS
            {from_joins}
            WHERE {where_clause}
            ORDER BY s.last_name, s.first_name
            LIMIT %s OFFSET %s
        """
        
        params.extend([limit, offset]) 
        cursor.execute(query, params)
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        
        formatted_students = []
        for s in students:
            year_section_display = f"{s['year_level']}-{s['section_name']}"

            formatted_students.append({
                'id': s['student_id'],
                'idNumber': s['student_id'],
                'firstName': s['first_name'],
                'lastName': s['last_name'],
                'middleName': s['middle_name'],
                'name': f"{s['first_name']} {s['middle_name'] + ' ' if s['middle_name'] else ''}{s['last_name']}",
                'course': s['course'],
                'yearSection': year_section_display, 
                'email': s['email'],
                'photo': s['photo_path'] if s['photo_path'] else f"https://ui-avatars.com/api/?name={s['first_name']}+{s['last_name']}&background=random",
                'status': s['status'],
                'createdAt': s['created_at'].isoformat() if s['created_at'] else None,
                'updatedAt': s['updated_at'].isoformat() if s['updated_at'] else None,
                'program_id': s['program_id'],
                'year_level': s['year_level'],
                'section_id': s['section_id'],
                'curriculum_id': s['curriculum_id'],  # ADD THIS
                'curriculum_year': s['curriculum_year']  # ADD THIS
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
                'photo': f['photo_path'] if f['photo_path'] else f"https://ui-avatars.com/api/?name={f['first_name']}+{f['last_name']}&background=random",  
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
    """Export students or faculty data to CSV - FIXED: Using correct schema"""
    try:
        data = request.json
        data_type = data.get('type', 'students')  # 'students' or 'faculty'
        filters = data.get('filters', {})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if data_type == 'students':
            # ✅ CORRECTED: Join with year_sections and programs to get proper data
            query = """
                SELECT 
                    s.student_id, 
                    s.first_name, 
                    s.last_name, 
                    s.middle_name, 
                    p.program_name as course,
                    CONCAT(ys.year_level, ys.section_name) as year_section,
                    s.email, 
                    s.status, 
                    s.created_at
                FROM students s
                JOIN year_sections ys ON s.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                WHERE s.status = %s
                ORDER BY s.last_name, s.first_name
            """
            cursor.execute(query, [filters.get('status', 'active')])
            
        elif data_type == 'faculty':
            query = """
                SELECT 
                    faculty_id, 
                    first_name, 
                    last_name, 
                    middle_name, 
                    department, 
                    designation, 
                    email, 
                    role, 
                    status, 
                    created_at
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
    """Get comprehensive dashboard statistics including program breakdowns - SHOW ONLY EXISTING PROGRAMS"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get total student count
        cursor.execute("SELECT COUNT(*) as total_students FROM students WHERE status = 'active'")
        total_students = cursor.fetchone()['total_students']
        
        # Get total faculty count
        cursor.execute("SELECT COUNT(*) as total_faculty FROM faculty WHERE status = 'active'")
        total_faculty = cursor.fetchone()['total_faculty']
        
        # Get student count by PROGRAM (only programs that have students)
        cursor.execute("""
            SELECT 
                p.program_id,
                p.program_name,
                COUNT(s.student_id) as count 
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            GROUP BY p.program_id, p.program_name
            ORDER BY count DESC
        """)
        program_stats = cursor.fetchall()
        
        # Create program counts dictionary - ONLY FOR EXISTING PROGRAMS
        program_counts = {}
        for program in program_stats:
            program_id = program['program_id'].upper()
            program_counts[program_id] = program['count']
        
        # Get ALL active programs to know what programs exist
        cursor.execute("SELECT program_id, program_name FROM programs WHERE status = 'active'")
        all_active_programs = cursor.fetchall()
        
        # Initialize counts for ALL existing programs (even if they have 0 students)
        cs_count = 0
        it_count = 0 
        act_count = 0
        
        # Only count programs that actually exist in the system
        existing_programs = {}
        for program in all_active_programs:
            program_id = program['program_id'].upper()
            existing_programs[program_id] = program['program_name']
            
            # Set count for each existing program (0 if no students)
            if program_id == 'CS':
                cs_count = program_counts.get('CS', 0)
            elif program_id == 'IT':
                it_count = program_counts.get('IT', 0)
            elif program_id == 'ACT':
                act_count = program_counts.get('ACT', 0)
        
        # Get faculty count by department
        cursor.execute("""
            SELECT department, COUNT(*) as count 
            FROM faculty 
            WHERE status = 'active' 
            GROUP BY department 
            ORDER BY count DESC
        """)
        department_stats = cursor.fetchall()
        
        # Recent attendance (today)
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) as present_today,
                   COUNT(*) as total_attendance_records_today
            FROM attendance 
            WHERE DATE(timestamp) = CURDATE()
        """)
        attendance_today = cursor.fetchone()
        
        # Attendance this week
        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) as unique_students_week,
                   COUNT(*) as total_records_week
            FROM attendance 
            WHERE YEARWEEK(timestamp) = YEARWEEK(NOW())
        """)
        attendance_week = cursor.fetchone()
        
        # Recent registrations (last 30 days)
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM students WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)) as students_30days,
                (SELECT COUNT(*) FROM faculty WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)) as faculty_30days
        """)
        recent_registrations = cursor.fetchone()
        
        # Active invites count
        cursor.execute("""
            SELECT COUNT(*) as active_invites
            FROM invites 
            WHERE expires_at > NOW() AND used = 0
        """)
        active_invites = cursor.fetchone()['active_invites']
        
        cursor.close()
        conn.close()
        
        # Build response with ONLY existing programs
        response_data = {
            'success': True,
            'stats': {
                # Main dashboard cards
                'total_students': total_students,
                'total_faculty': total_faculty,
                
                # Program counts - ONLY FOR EXISTING PROGRAMS
                'cs_students': cs_count,
                'it_students': it_count, 
                'act_students': act_count,
                
                # List of existing programs to help frontend decide what to display
                'existing_programs': existing_programs,
                
                # Detailed breakdowns
                'course_breakdown': [
                    {
                        'course': program['program_name'],
                        'count': program['count'],
                        'percentage': round((program['count'] / total_students * 100), 1) if total_students > 0 else 0
                    }
                    for program in program_stats
                ],
                
                'department_breakdown': [
                    {
                        'department': dept['department'],
                        'count': dept['count'],
                        'percentage': round((dept['count'] / total_faculty * 100), 1) if total_faculty > 0 else 0
                    }
                    for dept in department_stats
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
                    'most_popular_course': program_stats[0]['program_name'] if program_stats else 'N/A',
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
                'act_students': 0,
                'existing_programs': {},
                'course_breakdown': [],
                'department_breakdown': [],
                'attendance': {'present_today': 0, 'total_records_today': 0, 'unique_students_week': 0, 'total_records_week': 0},
                'recent_activity': {'students_registered_30days': 0, 'faculty_registered_30days': 0, 'active_invites': 0},
                'summary': {'total_users': 0, 'attendance_rate_today': 0, 'most_popular_course': 'N/A', 'largest_department': 'N/A'}
            }
        })


@app.route('/api/get_active_system_overview', methods=['GET'])
def get_active_system_overview():
    """Get current active academic year with ALL active curricula"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # FIXED: Get ALL ACTIVE academic years across ALL programs
        cursor.execute("""
            SELECT ay.academic_year_id, ay.academic_year, ay.program_id, p.program_name
            FROM academic_years ay
            JOIN programs p ON ay.program_id = p.program_id
            WHERE ay.status = 'active'
            ORDER BY ay.academic_year DESC, p.program_name
        """)
        active_academic_years = cursor.fetchall()

        active_programs = list(set([year['program_name'] for year in active_academic_years]))
        active_programs.sort()  # Sort alphabetically
        
        if not active_academic_years:
            cursor.close()
            conn.close()
            return jsonify({
                'success': True,
                'overview': {
                    'active_academic_year': 'No active academic year set',
                    'active_semester': 'N/A',
                    'total_sections': 0,
                    'total_subjects': 0,
                    'total_units': 0,
                    'has_active_period': False,
                    'detailed_breakdown': [],
                    'active_curricula': [],
                    'active_programs': active_programs
                }
            })
        
        # Get all academic years and program IDs
        academic_years = list(set([year['academic_year'] for year in active_academic_years]))
        program_ids = [year['program_id'] for year in active_academic_years]
        academic_year_ids = [year['academic_year_id'] for year in active_academic_years]
        
        # Use the most common academic year for display
        most_common_year = max(set(academic_years), key=academic_years.count)
        
        print(f"DEBUG: Active Academic Years - {academic_years}")
        print(f"DEBUG: Active Programs - {program_ids}")
        print(f"DEBUG: Academic Year IDs - {academic_year_ids}")
        
        # Get ALL active curricula for ALL active academic years
        format_strings = ','.join(['%s'] * len(academic_years))
        cursor.execute(f"""
            SELECT DISTINCT c.curriculum_id, c.curriculum_name, c.curriculum_year, p.program_name
            FROM curricula c
            JOIN programs p ON c.program_id = p.program_id
            WHERE c.academic_year IN ({format_strings}) AND c.status = 'active'
            ORDER BY p.program_name, c.curriculum_year DESC
        """, tuple(academic_years))
        
        active_curricula = cursor.fetchall()
        print(f"DEBUG: Active curricula found: {[c['curriculum_name'] for c in active_curricula]}")
        
        # Get ALL semesters for ALL active academic years
        format_strings = ','.join(['%s'] * len(academic_year_ids))
        cursor.execute(f"""
            SELECT s.semester_id, s.semester_number, p.program_name, c.curriculum_name, c.curriculum_year
            FROM semesters s
            JOIN academic_years ay ON s.academic_year_id = ay.academic_year_id
            JOIN programs p ON ay.program_id = p.program_id
            JOIN curricula c ON s.curriculum_id = c.curriculum_id
            WHERE ay.academic_year_id IN ({format_strings})
            ORDER BY p.program_name, c.curriculum_year, s.semester_number
        """, tuple(academic_year_ids))
        
        active_semesters = cursor.fetchall()
        print(f"DEBUG: All semesters found: {len(active_semesters)}")
        
        if not active_semesters:
            cursor.close()
            conn.close()
            return jsonify({
                'success': True,
                'overview': {
                    'active_academic_year': most_common_year,
                    'active_semester': 'No semesters found',
                    'total_sections': 0,
                    'total_subjects': 0,
                    'total_units': 0,
                    'has_active_period': False,
                    'detailed_breakdown': [],
                    'active_curricula': active_curricula
                }
            })
        
        # Get semester IDs
        semester_ids = [s['semester_id'] for s in active_semesters]
        
        # Get unique semester numbers
        semester_numbers = list(set([s['semester_number'] for s in active_semesters]))
        active_semester_display = ", ".join(semester_numbers)
        
        # Get data from ALL semesters
        format_strings = ','.join(['%s'] * len(semester_ids))
        
        # Total sections
        cursor.execute(f"""
            SELECT COUNT(DISTINCT ys.section_id) as total_sections
            FROM year_sections ys
            WHERE ys.semester_id IN ({format_strings}) AND ys.status = 'active'
        """, tuple(semester_ids))
        total_sections = cursor.fetchone()['total_sections'] or 0
        
        # Total subjects and units
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT sub.subject_id) as total_subjects,
                COALESCE(SUM(sub.units), 0) as total_units
            FROM subjects sub
            JOIN year_sections ys ON sub.section_id = ys.section_id
            WHERE ys.semester_id IN ({format_strings}) AND sub.status = 'active' AND ys.status = 'active'
        """, tuple(semester_ids))
        subjects_result = cursor.fetchone()
        total_subjects = subjects_result['total_subjects'] or 0
        total_units = subjects_result['total_units'] or 0
        
        # Detailed breakdown by curriculum
        detailed_breakdown = []
        cursor.execute(f"""
            SELECT 
                ys.section_name,
                ys.year_level,
                p.program_name,
                s.semester_number,
                c.curriculum_name,
                c.curriculum_year,
                COUNT(DISTINCT sub.subject_id) as subject_count,
                COALESCE(SUM(sub.units), 0) as section_units
            FROM year_sections ys
            JOIN semesters s ON ys.semester_id = s.semester_id
            JOIN academic_years ay ON s.academic_year_id = ay.academic_year_id
            JOIN programs p ON ay.program_id = p.program_id
            JOIN curricula c ON s.curriculum_id = c.curriculum_id
            LEFT JOIN subjects sub ON ys.section_id = sub.section_id AND sub.status = 'active'
            WHERE ys.semester_id IN ({format_strings}) AND ys.status = 'active'
            GROUP BY ys.section_id, ys.section_name, ys.year_level, p.program_name, 
                     s.semester_number, c.curriculum_name, c.curriculum_year
            ORDER BY p.program_name, c.curriculum_year DESC, ys.year_level, ys.section_name
        """, tuple(semester_ids))
        
        sections_data = cursor.fetchall()
        print(f"DEBUG: Found {len(sections_data)} section records")
        
        for section in sections_data:
            detailed_breakdown.append({
                'program': section['program_name'],
                'curriculum': section['curriculum_name'],
                'curriculum_year': section['curriculum_year'],
                'semester': section['semester_number'],
                'year_level': f"Year {section['year_level']}",
                'section_name': section['section_name'],
                'subject_count': section['subject_count'],
                'section_units': section['section_units']
            })
            print(f"DEBUG Section: {section['program_name']} - {section['curriculum_name']} ({section['curriculum_year']}) - {section['section_name']}")
        
        cursor.close()
        conn.close()
        
        overview_data = {
            'success': True,
            'overview': {
                'active_academic_year': most_common_year,
                'active_semester': active_semester_display,
                'total_sections': total_sections,
                'total_subjects': total_subjects,
                'total_units': total_units,
                'has_active_period': True,
                'detailed_breakdown': detailed_breakdown,
                'active_curricula': active_curricula,
                'active_programs': active_programs 
            }
        }
        
        print(f"DEBUG FINAL: {total_sections} sections, {total_subjects} subjects, {total_units} units")
        print(f"DEBUG FINAL Curricula: {[c['curriculum_name'] for c in active_curricula]}")
        return jsonify(overview_data)
        
    except Exception as e:
        logger.error(f"Error fetching active system overview: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'overview': {
                'active_academic_year': 'Error loading',
                'active_semester': 'Error',
                'total_sections': 0,
                'total_subjects': 0,
                'total_units': 0,
                'has_active_period': False,
                'detailed_breakdown': [],
                'active_curricula': [],
                'active_programs': active_programs
            }
        })
    
@app.route('/api/get_course_distribution', methods=['GET'])
def get_course_distribution():
    """Get detailed course distribution for charts and analytics - FIXED: Using correct schema"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get course distribution with year sections using proper joins
        cursor.execute("""
            SELECT 
                p.program_name as course,
                CONCAT(ys.year_level, ys.section_name) as year_section,
                COUNT(*) as student_count,
                GROUP_CONCAT(CONCAT(s.first_name, ' ', s.last_name) SEPARATOR ', ') as student_names
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            GROUP BY p.program_name, ys.year_level, ys.section_name
            ORDER BY p.program_name, ys.year_level, ys.section_name
        """)
        
        detailed_distribution = cursor.fetchall()
        
        # Get summary by course only
        cursor.execute("""
            SELECT 
                p.program_name as course,
                COUNT(*) as total_students,
                COUNT(DISTINCT CONCAT(ys.year_level, ys.section_name)) as sections_count
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            GROUP BY p.program_name
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
os.makedirs('static/images/faculty_photos', exist_ok=True)
os.makedirs('static/images/admin_photos', exist_ok=True)

@app.route('/api/get_other_students', methods=['GET'])  # Different route path
def get_other_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # ✅ CORRECTED: Join with year_sections and programs to get proper data
        cursor.execute("""
            SELECT 
                s.student_id,
                s.first_name, 
                s.last_name, 
                p.program_name as course,
                CONCAT(ys.year_level, ys.section_name) as year_section,
                s.photo_path
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            ORDER BY p.program_name, ys.year_level, ys.section_name, s.last_name, s.first_name
        """)
        
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
        original_student_id = data.get('original_student_id')  # Old ID
        new_student_id = data.get('student_id')  # New ID (might be same)
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        middle_name = data.get('middle_name', '')
        section_id = data.get('section_id')
        email = data.get('email')
        status = data.get('status', 'active')
        
        # Validation
        if not all([original_student_id, new_student_id, first_name, last_name, section_id, email]):
            return jsonify({'success': False, 'message': 'All required fields are missing'})
            
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if original student exists
        cursor.execute("SELECT student_id FROM students WHERE student_id = %s", (original_student_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # If student_id is being changed, check if new ID already exists
        if original_student_id != new_student_id:
            cursor.execute("SELECT student_id FROM students WHERE student_id = %s", (new_student_id,))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'success': False, 'message': f'Student ID {new_student_id} already exists'})
        
        # Update student record
        cursor.execute(
            """UPDATE students 
               SET student_id = %s,
                   first_name = %s, 
                   last_name = %s, 
                   middle_name = %s, 
                   section_id = %s, 
                   email = %s, 
                   status = %s, 
                   updated_at = NOW()
               WHERE student_id = %s""",
            (new_student_id, first_name, last_name, middle_name, 
             section_id, email, status, original_student_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Refresh known faces
        try:
            load_known_faces_from_db()
        except:
            pass
        
        return jsonify({
            'success': True, 
            'message': 'Student updated successfully',
            'updated_student_id': new_student_id
        })
        
    except Exception as e:
        logger.error(f"Error updating student: {str(e)}")
        import traceback
        traceback.print_exc()
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
    expires_at = datetime.now() + timedelta(minutes=10)  
    
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
        
        if datetime.now() > expires_at: 
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
            'is_left': bool(yaw >= 5),
            'is_right': bool(yaw <= -4),
            'is_up': bool(pitch <= -1),   # LOWER threshold for up (more negative)
            'is_down': bool(pitch >= 1),  # HIGHER threshold for down (more positive)
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
        email = data.get('email', '').strip().lower()
        student_id = data.get('student_id', '').strip()
        invite_token = data.get('invite_token', '').strip()  
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        middle_name = data.get('middle_name', '').strip()
        section_id = data.get('section_id', '').strip()
        program_id = data.get('program_id', '').strip()  # Get program_id from frontend
        password = data.get('password', '').strip()
        
        if not all([email, student_id, first_name, last_name, section_id, program_id, password]):
            logger.warning("Missing required fields in register_student request")
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if "@wmsu.edu.ph" not in email:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Invalid WMSU email address'})
        
        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'})
        
        # Extract curriculum year from student ID (first 4 digits)
        if len(student_id) >= 4:
            curriculum_year = student_id[:4]
            # Validate it's a reasonable year (between 2000 and current year + 1)
            current_year = datetime.now().year
            if not (2000 <= int(curriculum_year) <= current_year + 1):
                curriculum_year = str(current_year)  # Fallback to current year
        else:
            # Fallback if student ID format is unexpected
            current_year = datetime.now().year
            curriculum_year = str(current_year)
        
        password_hash = hash_password(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if student already exists
        cursor.execute("SELECT student_id, email FROM students WHERE student_id = %s OR email = %s", 
                      (student_id, email))
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            conn.close()
            if existing[0] == student_id:
                logger.warning(f"Student ID {student_id} already exists")
                return jsonify({'success': False, 'message': 'Student ID already exists'})
            else:
                logger.warning(f"Email {email} already registered")
                return jsonify({'success': False, 'message': 'Email already registered'})
        
        # Find the appropriate curriculum
        current_academic_year = f"{datetime.now().year}-{datetime.now().year + 1}"
        
        cursor.execute(
            """SELECT curriculum_id FROM curricula 
            WHERE program_id = %s AND academic_year = %s AND curriculum_year = %s AND status = 'active'
            ORDER BY effective_date DESC LIMIT 1""",
            (program_id, current_academic_year, curriculum_year)
        )
        
        curriculum_result = cursor.fetchone()
        curriculum_id = curriculum_result[0] if curriculum_result else None
        
        # If no exact match found, try to find the closest curriculum
        if not curriculum_id:
            cursor.execute(
                """SELECT curriculum_id FROM curricula 
                WHERE program_id = %s AND academic_year = %s AND status = 'active'
                ORDER BY ABS(CAST(curriculum_year AS SIGNED) - %s) LIMIT 1""",
                (program_id, current_academic_year, int(curriculum_year))
            )
            curriculum_result = cursor.fetchone()
            curriculum_id = curriculum_result[0] if curriculum_result else None
        
        # Log curriculum assignment
        if curriculum_id:
            logger.info(f"Assigned curriculum {curriculum_id} to student {student_id} (program: {program_id}, year: {curriculum_year})")
        else:
            logger.warning(f"No active curriculum found for student {student_id} (program: {program_id}, year: {curriculum_year})")
        
        face_encoding_data = data.get('face_encoding', '')
        encoding_str = None
        
        try:
            if face_encoding_data:
                face_encoding = json.loads(face_encoding_data)
                if isinstance(face_encoding, list) and len(face_encoding) == 512:
                    encoding_str = "[" + ",".join(str(x) for x in face_encoding) + "]"
                else:
                    raise ValueError("Invalid face encoding length")
            else:
                raise ValueError("No face encoding provided")
        except Exception as e:
            cursor.close()
            conn.close()
            logger.error(f"Invalid face encoding format: {e}")
            return jsonify({'success': False, 'message': 'Invalid face encoding. Please complete face scanning.'})
        
        photo_path = None
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo and photo.filename:
                filename = secure_filename(photo.filename)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    os.makedirs('static/images/student_photos', exist_ok=True)
                    photo_path = f"static/images/student_photos/{student_id}.jpg"
                    
                    try:
                        photo.save(photo_path)
                        logger.info(f"Saved photo for {student_id} at {photo_path}")
                    except Exception as e:
                        logger.error(f"Failed to save photo: {e}")
                        photo_path = None
        
        # Insert student with curriculum_id
        cursor.execute(
            """INSERT INTO students 
            (student_id, first_name, last_name, middle_name, section_id, email, 
             face_encoding, photo_path, password_hash, curriculum_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (student_id, first_name, last_name, middle_name or None, section_id, email, 
             encoding_str, photo_path, password_hash, curriculum_id)
        )
        conn.commit()
        
        # Handle invite token (your existing code)
        if invite_token:
            try:
                conn_invite = get_db_connection()
                cursor_invite = conn_invite.cursor()

                cursor_invite.execute(
                    "UPDATE invites SET current_uses = current_uses + 1 WHERE token = %s",
                    (invite_token,)
                )

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
        
        cursor.close()
        conn.close()
        
        try:
            load_known_faces_from_db()
        except Exception as e:
            logger.warning(f"Failed to reload known faces: {e}")
        
        logger.info(f"Student registered successfully: {student_id} ({first_name} {last_name}) with curriculum {curriculum_id}")
        return jsonify({'success': True, 'message': 'Student registered successfully'})

    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})
    

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
                photo_path = f"static/images/faculty_photos/{faculty_id}.jpg"
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

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    return jsonify({
        'camera_available': camera_available,
        'using_dummy_feed': use_dummy_feed,
        'active_tracks': len(tracks) if camera_available else 0,
        'locked_tracks': len(locked_tracks) if camera_available else 0
    })

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
    """Get attendance data for the logged-in student - FIXED: Using correct schema"""
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
        
        # 1. Get student's basic info with proper joins
        cursor.execute("""
            SELECT 
                s.student_id, 
                s.first_name, 
                s.last_name,
                p.program_name,
                ys.year_level, 
                ys.section_name,
                s.section_id
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.student_id = %s AND s.status = 'active'
        """, (student_id,))
        
        student_data = cursor.fetchone()
        if not student_data:
            return jsonify({'error': 'Student not found'}), 404
        
        print(f"Found student: {student_data}")
        
        # Extract student's section info for queries
        student_section_id = student_data['section_id']
        student_program = student_data['program_name']
        student_year_level = student_data['year_level']
        student_section_name = student_data['section_name']
        
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
            WHERE s.section_id = %s
            AND cs.day_of_week = %s
            AND s.status = 'active'
            AND cs.status = 'active'
            ORDER BY cs.start_time
        """, (student_section_id, today_day))
        
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
        
        # 5. Get all subjects for the student's section
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
            WHERE ys.section_id = %s
            AND s.status = 'active'
            AND cs.status = 'active'
            ORDER BY 
                CASE cs.day_of_week 
                    WHEN 'Monday' THEN 1
                    WHEN 'Tuesday' THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4
                    WHEN 'Friday' THEN 5
                    WHEN 'Saturday' THEN 6
                    ELSE 7
                END,
                cs.start_time
        """, (student_section_id,))
        
        semester_classes = cursor.fetchall()
        print(f"Semester classes found: {len(semester_classes)}")
        
        # 6. Calculate attendance statistics - FIXED: Count unique sessions
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT ases.session_id) as total_classes,
                COUNT(DISTINCT CASE WHEN a.status IN ('present', 'late') THEN ases.session_id END) as attended_classes
            FROM attendance_sessions ases
            LEFT JOIN attendance a ON (
                a.session_id = ases.session_id 
                AND a.student_id = %s
                AND a.status IN ('present', 'late')
            )
            WHERE ases.status = 'completed'
            AND ases.section_id = %s
        """, (student_id, student_section_id))
        
        stats = cursor.fetchone()
        print(f"Stats: {stats}")
        
        total_classes = stats['total_classes'] or 0
        attended_classes = stats['attended_classes'] or 0
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
                'id': student_data['student_id'],
                'name': f"{student_data['first_name']} {student_data['last_name']}",
                'course': student_data['program_name'],  # Use program_name instead of course
                'section': f"{student_data['year_level']}{student_data['section_name']}"  # Combine year_level and section_name
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
        print(f"Attendance stats: {attended_classes}/{total_classes} = {attendance_rate}%")
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
    
    print(f"DEBUG - User object: {user}")
    print(f"DEBUG - User role: {user.get('role')}")
    print(f"DEBUG - User type: {user.get('user_type')}")
    
    user_role = user.get('role', '')
    user_type = user.get('user_type', '')
    user_id = user.get('user_id', '') 
    
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
        user_role=user_role,
        user_type=user_type,
        user_id=user_id  
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
    """Get complete summary data for the latest session - FIXED: Handles empty status and missing students"""
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
            
            session_id = session_data['session_id']
            section_id = session_data['section_id']
            started_at = session_data['created_at']
            ended_at = session_data['ended_at']
            
            print(f"🔍 DEBUG Session: {session_id}")
            print(f"   - Section ID: {section_id}")
            print(f"   - Created At: {started_at}")
            print(f"   - Ended At: {ended_at}")
            
            # Get duration
            duration_time = session_data.get('duration_time', '00:00:00')
            duration_seconds = 0
            if duration_time and isinstance(duration_time, str):
                try:
                    hours, minutes, seconds = map(int, duration_time.split(':'))
                    duration_seconds = hours * 3600 + minutes * 60 + seconds
                except:
                    if started_at and ended_at:
                        duration_seconds = int((ended_at - started_at).total_seconds())
            else:
                if started_at and ended_at:
                    duration_seconds = int((ended_at - started_at).total_seconds())
            
            subject_code = session_data.get('subject_code', 'IT99')
            subject_name = session_data.get('subject_name', 'AMBUTT UY')
            room = session_data.get('room', 'Unknown Room')
            
            # 🎯 FIXED: Get attendance records and convert empty/missing status to 'absent'
            cursor.execute("""
                WITH LatestAttendance AS (
                    SELECT 
                        a.student_id,
                        a.name as student_name,
                        CASE 
                            WHEN a.status IS NULL OR a.status = '' OR a.status = 'missing' THEN 'absent'
                            ELSE a.status
                        END as status,
                        a.timestamp,
                        a.session_id,
                        a.subject_code,
                        a.subject_name,
                        a.room,
                        s.photo_path,
                        CASE 
                            WHEN a.student_id IS NULL OR a.student_id = '' THEN TRUE 
                            WHEN NOT EXISTS (SELECT 1 FROM students WHERE student_id = a.student_id) THEN TRUE
                            ELSE FALSE 
                        END as is_temporary,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.student_id 
                            ORDER BY a.timestamp DESC
                        ) as rn
                    FROM attendance a
                    LEFT JOIN students s ON a.student_id = s.student_id
                    WHERE a.session_id = %s
                    AND a.person_type = 'student'
                )
                SELECT * FROM LatestAttendance WHERE rn = 1
                ORDER BY 
                    CASE 
                        WHEN status = 'present' THEN 1
                        WHEN status = 'late' THEN 2
                        WHEN status = 'excused' THEN 3
                        WHEN status = 'absent' THEN 4
                        ELSE 5
                    END,
                    student_name
            """, (session_id,))
            
            unique_attendance_records = cursor.fetchall()
            
            print(f"🔍 DEBUG Found {len(unique_attendance_records)} UNIQUE attendance records")
            
            # Get regular students from the actual section
            cursor.execute("""
                SELECT student_id, first_name, last_name, photo_path 
                FROM students 
                WHERE section_id = %s AND status = 'active'
            """, (section_id,))
            
            all_section_students = cursor.fetchall()
            
            # Get list of students who have attendance records
            attended_student_ids = set()
            for r in unique_attendance_records:
                if r['student_id'] and not r['is_temporary']:
                    attended_student_ids.add(r['student_id'])
            
            print(f"🔍 DEBUG Regular students in section {section_id}: {len(all_section_students)}")
            print(f"🔍 DEBUG Students with attendance records: {len(attended_student_ids)}")
            
            # Create complete student list
            complete_student_list = []
            
            # Add all attendance records (includes converted absent from empty/missing status)
            for record in unique_attendance_records:
                student_id = record['student_id']
                
                # Handle temporary students (manually added)
                if record['is_temporary']:
                    if not student_id and 'ID:' in record['student_name']:
                        try:
                            student_id = record['student_name'].split('ID:')[-1].split(')')[0].strip()
                        except:
                            student_id = 'temp'
                    
                    photo_path = f"/static/images/student_photos/{student_id}.jpg" if student_id and student_id != 'temp' else '/static/images/default-avatar.jpg'
                else:
                    photo_path = record['photo_path'] or f"/static/images/student_photos/{student_id}.jpg"
                
                complete_student_list.append({
                    'student_id': student_id or 'temp',
                    'name': record['student_name'],
                    'status': record['status'],  # Already converted to 'absent' if it was empty/missing
                    'timestamp': record['timestamp'],
                    'photo': photo_path or '/static/images/default-avatar.jpg',
                    'is_temporary': record['is_temporary'],
                    'subject_code': record['subject_code'] or subject_code,
                    'subject_name': record['subject_name'] or subject_name,
                    'room': record['room'] or room
                })
                
                print(f"🔍 DEBUG Added from records: {student_id} - Status: {record['status']}")
            
            # Add students who have NO attendance record at all (never detected)
            absent_count_added = 0
            for student in all_section_students:
                student_id = student['student_id']
                
                # Only add if student has NO attendance record for this session
                if student_id not in attended_student_ids:
                    complete_student_list.append({
                        'student_id': student_id,
                        'name': f"{student['first_name']} {student['last_name']}",
                        'status': 'absent',
                        'timestamp': ended_at,
                        'photo': student['photo_path'] or f"/static/images/student_photos/{student_id}.jpg",
                        'is_temporary': False,
                        'subject_code': subject_code,
                        'subject_name': subject_name,
                        'room': room
                    })
                    absent_count_added += 1
                    print(f"🔍 DEBUG Added never-detected student as absent: {student_id}")
            
            print(f"🔍 DEBUG Added {absent_count_added} never-detected students as absent")
            
            # Calculate counts from FINAL list
            present_count = len([s for s in complete_student_list if s['status'] == 'present'])
            late_count = len([s for s in complete_student_list if s['status'] == 'late'])
            absent_count = len([s for s in complete_student_list if s['status'] == 'absent'])
            excused_count = len([s for s in complete_student_list if s['status'] == 'excused'])
            total_students = len(complete_student_list)
            
            # Count temporary students for debugging
            temp_count = len([s for s in complete_student_list if s.get('is_temporary')])
            print(f"🔍 DEBUG FINAL Student breakdown: {temp_count} temporary, {total_students - temp_count} regular")
            print(f"🔍 DEBUG FINAL Status breakdown: Present: {present_count}, Late: {late_count}, Absent: {absent_count}, Excused: {excused_count}")
            
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
            
            summary_data = {
                'success': True,
                'session': {
                    'session_id': session_id,
                    'class_name': class_name,
                    'started_at': started_at.strftime('%Y-%m-%d %I:%M%p') if started_at else '',
                    'ended_at': ended_at.strftime('%Y-%m-%d %I:%M%p') if ended_at else '',
                    'duration_seconds': duration_seconds,
                    'late_threshold_minutes': session_data.get('late_threshold_minutes', 20) or 20,
                    'total_students': total_students,
                    'present_count': present_count,
                    'late_count': late_count,
                    'absent_count': absent_count,
                    'excused_count': excused_count,
                    'subject_code': subject_code,
                    'subject_name': subject_name,
                    'room': room
                },
                'user': {
                    'name': f"{user['first_name']} {user['last_name']}",
                    'role': user['role'],
                    'username': user['admin_id'],
                    'photo_path': user['photo_path'] or '/static/images/default-avatar.jpg'
                },
                'subject': {
                    'code': subject_code,
                    'name': subject_name,
                    'room': room
                },
                'course_section': course_section_display,
                'attendance': complete_student_list
            }
            
            print(f"✅ FINAL SUMMARY: {total_students} total students, {present_count} present, {late_count} late, {absent_count} absent, {excused_count} excused")
            
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
    """Export attendance data as CSV - TODAY'S SESSION ONLY (REGULAR + TEMPORARY + ABSENT) - Alphabetically sorted"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'success': False, 'message': 'Missing session_id'}), 400
    
    try:
        with get_db_cursor() as cursor:
            # ✅ GET CURRENT SESSION DATA
            cursor.execute("""
                SELECT class_name, started_at, ended_at, subject_code, subject_name, room, section_id
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
            section_id = session_data.get('section_id')
            session_date = session_data['started_at'].date()
            
            print(f"🔍 DEBUG Current Session: {session_id}")
            print(f"   - Subject: {subject_code} - {subject_name}")
            print(f"   - Room: {room}")
            print(f"   - Section ID: {section_id}")
            print(f"   - Session Date: {session_date}")
            
            # ✅ GET PROGRAM AND SECTION FROM SESSION
            program_display = "BSIT"  # Default
            if 'Associate in Computer Technology' in class_name:
                program_display = 'ACT'
            elif 'Information Technology' in class_name:
                program_display = 'BSIT'
            elif 'Computer Science' in class_name:
                program_display = 'BSCS'
            
            section_display = "4C"  # Default
            if '4th Year' in class_name:
                section_part = class_name.split('4th Year')[-1].strip()
                if section_part:
                    section_display = f"4{section_part[0]}"
            elif '2nd Year' in class_name:
                section_part = class_name.split('2nd Year')[-1].strip()
                if section_part:
                    section_display = f"2{section_part[0]}"
            
            print(f"🔍 DEBUG Program: {program_display}, Section: {section_display}")
            
            # ✅ FIXED: HANDLE EMPTY STATUS AND GET ALL STUDENTS - SORTED ALPHABETICALLY
            cursor.execute("""
                WITH attendance_records AS (
                    -- Get attendance records with status cleanup
                    SELECT 
                        a.student_id,
                        a.name as student_name,
                        CASE 
                            WHEN a.status IS NULL OR a.status = '' OR a.status = 'missing' THEN 'absent'
                            ELSE a.status
                        END as status,
                        a.timestamp,
                        a.subject_code,
                        a.subject_name,
                        a.room,
                        a.remarks,
                        CASE 
                            WHEN a.student_id IS NULL OR a.student_id = '' THEN TRUE
                            ELSE FALSE
                        END as is_temporary
                    FROM attendance a
                    WHERE a.session_id = %s
                    AND a.person_type = 'student'
                ),
                all_students AS (
                    -- Regular students from section (with their attendance or marked absent)
                    SELECT 
                        s.student_id,
                        CONCAT(s.first_name, ' ', s.last_name) as student_name,
                        %s as year_section,
                        COALESCE(ar.status, 'absent') as status,
                        COALESCE(ar.timestamp, %s) as attendance_timestamp,
                        'No' as is_temporary,
                        COALESCE(ar.subject_code, %s) as subject_code,
                        COALESCE(ar.subject_name, %s) as subject_name,
                        COALESCE(ar.room, %s) as room,
                        ar.remarks
                    FROM students s
                    LEFT JOIN attendance_records ar ON s.student_id = ar.student_id AND ar.is_temporary = FALSE
                    WHERE s.section_id = %s AND s.status = 'active'
                    
                    UNION ALL
                    
                    -- Temporary students (manually added)
                    SELECT 
                        CASE 
                            WHEN ar.student_name LIKE '%(ID: %' THEN 
                                TRIM(SUBSTRING(
                                    ar.student_name, 
                                    LOCATE('(ID: ', ar.student_name) + 5,
                                    LOCATE(')', ar.student_name, LOCATE('(ID: ', ar.student_name)) - (LOCATE('(ID: ', ar.student_name) + 5)
                                ))
                            ELSE CONCAT('TEMP-', LPAD(ar.student_id, 4, '0'))
                        END as student_id,
                        CASE 
                            WHEN ar.student_name LIKE '%(ID: %' THEN 
                                TRIM(SUBSTRING(ar.student_name, 1, LOCATE('(ID: ', ar.student_name) - 1))
                            ELSE ar.student_name 
                        END as student_name,
                        %s as year_section,
                        ar.status,
                        ar.timestamp as attendance_timestamp,
                        'Yes' as is_temporary,
                        COALESCE(ar.subject_code, %s) as subject_code,
                        COALESCE(ar.subject_name, %s) as subject_name,
                        COALESCE(ar.room, %s) as room,
                        ar.remarks
                    FROM attendance_records ar
                    WHERE ar.is_temporary = TRUE
                )
                
                SELECT 
                    student_id,
                    student_name,
                    year_section,
                    status,
                    attendance_timestamp,
                    is_temporary,
                    subject_code,
                    subject_name,
                    room,
                    remarks
                FROM all_students
                ORDER BY student_name ASC
            """, (
                session_id,
                section_display,
                session_data['ended_at'],
                subject_code, subject_name, room,
                section_id,
                section_display,
                subject_code, subject_name, room
            ))
            
            records = cursor.fetchall()
            
            if not records:
                return jsonify({'success': False, 'message': 'No student data found for today\'s session'}), 404
            
            print(f"🔍 DEBUG CSV Export: Found {len(records)} total students in today's session")
            
            # Debug breakdown
            regular_students = [r for r in records if r['is_temporary'] == 'No']
            temp_students = [r for r in records if r['is_temporary'] == 'Yes']
            present_students = [r for r in records if r['status'] == 'present']
            late_students = [r for r in records if r['status'] == 'late']
            absent_students = [r for r in records if r['status'] == 'absent']
            excused_students = [r for r in records if r['status'] == 'excused']
            
            print(f"🔍 DEBUG Today's Session Breakdown:")
            print(f"   - Regular students: {len(regular_students)}")
            print(f"   - Temporary students: {len(temp_students)}")
            print(f"   - Present: {len(present_students)}")
            print(f"   - Late: {len(late_students)}")
            print(f"   - Absent: {len(absent_students)}")
            print(f"   - Excused: {len(excused_students)}")
            
            # Debug first few students (alphabetically)
            print(f"🔍 DEBUG First 10 Students (Alphabetical Order):")
            for i, record in enumerate(records[:10]):
                print(f"   {i+1}. {record['student_name']} ({record['student_id']}) - {record['status']}")
            
            # Create CSV content
            import csv
            import io
            from datetime import datetime
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Headers
            writer.writerow([
                'Student ID', 
                'Student Name', 
                'Status', 
                'Time Recorded', 
                'Subject Code', 
                'Subject Name', 
                'Room',
                'Program', 
                'Section', 
                'Temporary Student'
            ])
            
            # Write data for ALL students (already sorted alphabetically by query)
            for record in records:
                timestamp = record['attendance_timestamp']
                
                # Handle timestamps
                if record['status'] == 'absent' and not timestamp:
                    time_recorded = session_data['ended_at'].strftime('%Y-%m-%d %I:%M:%S %p')
                elif timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                            time_recorded = timestamp.strftime('%Y-%m-%d %I:%M:%S %p')
                        except:
                            time_recorded = timestamp
                    else:
                        time_recorded = timestamp.strftime('%Y-%m-%d %I:%M:%S %p')
                else:
                    time_recorded = session_data['ended_at'].strftime('%Y-%m-%d %I:%M:%S %p')
                
                # Clean up student ID for temporary students
                student_id = record['student_id']
                if record['is_temporary'] == 'Yes' and (not student_id or student_id.startswith('TEMP')):
                    student_id = f"TEMP-{hash(record['student_name']) % 10000:04d}"
                
                writer.writerow([
                    student_id,
                    record['student_name'],
                    record['status'].upper(),
                    time_recorded,
                    record['subject_code'] or subject_code,
                    record['subject_name'] or subject_name,
                    record['room'] or room,
                    program_display,
                    section_display,
                    record['is_temporary']
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            # Create filename
            clean_subject_name = subject_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subject_code}_{clean_subject_name}_{program_display}-{section_display}_attendance_{timestamp}.csv"
            
            print(f"✅ EXPORT SUCCESS: {len(records)} students from today's session (A-Z sorted)")
            print(f"   - Regular: {len(regular_students)}")
            print(f"   - Temporary: {len(temp_students)}")
            print(f"   - Present: {len(present_students)}")
            print(f"   - Late: {len(late_students)}")
            print(f"   - Absent: {len(absent_students)}")
            print(f"   - Excused: {len(excused_students)}")
            
            # Create response
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
@student_login_required  
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
    """Add academic year and automatically create semesters with PROPER NAMES"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        status = data.get('status', 'active')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'Program ID and academic year are required'})
        
        # Validate academic year format
        if not re.match(r'^\d{4}-\d{4}$', academic_year):
            return jsonify({'success': False, 'message': 'Academic year must be in YYYY-YYYY format'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if this academic year already exists for THIS SPECIFIC PROGRAM
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': f'Academic year {academic_year} already exists for this program'})
        
        # If setting as active, first deactivate all other academic years for THIS PROGRAM
        if status == 'active':
            cursor.execute(
                "UPDATE academic_years SET status = 'inactive' WHERE program_id = %s",
                (program_id,)
            )
        
        # Insert academic year
        cursor.execute(
            "INSERT INTO academic_years (program_id, academic_year, status) VALUES (%s, %s, %s)",
            (program_id, academic_year, status)
        )
        
        # Get the newly created academic_year_id
        academic_year_id = cursor.lastrowid
        
        # AUTOMATICALLY CREATE SEMESTERS WITH PROPER NAMES
        semester_names = ['1st Semester', '2nd Semester', '3rd Semester']
        
        for semester_name in semester_names:
            # Insert semester for each active curriculum in this program
            cursor.execute("""
                INSERT INTO semesters (academic_year_id, semester_number, status, curriculum_id) 
                SELECT %s, %s, 'active', c.curriculum_id
                FROM curricula c 
                WHERE c.program_id = %s AND c.status = 'active' AND c.academic_year = %s
            """, (academic_year_id, semester_name, program_id, academic_year))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Academic year {academic_year} added successfully with semesters created'
        })
        
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
        original_academic_year = data.get('original_academic_year')  # Get original value
        academic_year = data.get('academic_year')  # This is the NEW value
        status = data.get('status')
        
        if not all([program_id, original_academic_year, academic_year, status]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Use original_academic_year in WHERE clause, new academic_year in SET
        cursor.execute(
            "UPDATE academic_years SET academic_year = %s, status = %s WHERE program_id = %s AND academic_year = %s",
            (academic_year, status, program_id, original_academic_year)  # Use original in WHERE
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

@app.route('/api/get_semesters_for_curriculum', methods=['GET'])
@login_required
def get_semesters_for_curriculum():
    """Get semesters for a specific curriculum - SHOW ALL SEMESTERS"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        curriculum_id = request.args.get('curriculum_id')
        
        if not all([program_id, academic_year, curriculum_id]):
            return jsonify({'success': False, 'message': 'Program ID, academic year, and curriculum ID are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get academic_year_id first
        cursor.execute(
            "SELECT academic_year_id FROM academic_years WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        academic_year_result = cursor.fetchone()
        
        if not academic_year_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Academic year not found'})
        
        academic_year_id = academic_year_result['academic_year_id']
        
        # Get ALL semesters for this specific curriculum - SIMPLIFIED QUERY
        cursor.execute("""
            SELECT 
                s.semester_id,
                s.semester_number,
                s.status,
                s.curriculum_id,
                (SELECT COUNT(DISTINCT ys.section_id) 
                 FROM year_sections ys 
                 WHERE ys.semester_id = s.semester_id AND ys.status = 'active') as section_count
            FROM semesters s
            WHERE s.academic_year_id = %s AND s.curriculum_id = %s
            ORDER BY 
                CASE s.semester_number
                    WHEN 'Summer' THEN 1
                    WHEN '1st Semester' THEN 2
                    WHEN '2nd Semester' THEN 3
                    ELSE 4
                END
        """, (academic_year_id, curriculum_id))
        
        semesters = cursor.fetchall()
        
        # ADD DEBUG LOGGING
        logger.info(f"Found {len(semesters)} semesters for curriculum {curriculum_id}")
        for semester in semesters:
            logger.info(f"Semester: {semester['semester_number']} - Status: {semester['status']}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'semesters': semesters,
            'debug_info': {  # ADD DEBUG INFO
                'total_semesters': len(semesters),
                'active_semesters': len([s for s in semesters if s['status'] == 'active']),
                'inactive_semesters': len([s for s in semesters if s['status'] == 'inactive'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching semesters for curriculum: {e}")
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
        curriculum_id = data.get('curriculum_id')
        
        if not all([program_id, academic_year, semester_number, curriculum_id]):
            return jsonify({'success': False, 'message': 'All fields including curriculum are required'})
        
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
        
        # Check if semester already exists for this academic_year AND curriculum
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s AND curriculum_id = %s",
            (academic_year_id, semester_number, curriculum_id)
        )
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': f'Semester "{semester_number}" already exists for this curriculum'})
        
        cursor.execute(
            "INSERT INTO semesters (academic_year_id, semester_number, status, curriculum_id) VALUES (%s, %s, %s, %s)",
            (academic_year_id, semester_number, status, curriculum_id)
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
        
        # Update the semester status
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
        curriculum_id = data.get('curriculum_id')  # ADD THIS
        
        if not all([program_id, academic_year, semester_number, curriculum_id]):  # UPDATE THIS
            return jsonify({'success': False, 'message': 'All fields including curriculum are required'})
        
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
        
        # Delete semester WITH curriculum_id
        cursor.execute(
            "DELETE FROM semesters WHERE academic_year_id = %s AND semester_number = %s AND curriculum_id = %s",
            (academic_year_id, semester_number, curriculum_id)  # ADD curriculum_id
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
    """Permanently delete an academic year"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'Program ID and Academic Year are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Permanent deletion
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
        
        return jsonify({'success': True, 'message': 'Academic year deleted permanently'})
        
    except Exception as e:
        logger.error(f"Error deleting academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})

 
# ==========================================
# CLASS SCHEDULE MANAGEMENT API ROUTES
# ==========================================

@app.route('/api/get_schedules', methods=['GET'])
@login_required
@role_required(['super_admin', 'admin', 'moderator'])
def get_schedules():
    """Get all class schedules with filtering options"""
    try:
        program_id = request.args.get('program_id')
        section_id = request.args.get('section_id')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Base query - UPDATED to include curriculum
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
                p.program_name,
                c.curriculum_id,
                c.curriculum_name  -- ADD THIS
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- ADD THIS JOIN
            WHERE cs.status = 'active' AND s.status = 'active' AND ys.status = 'active'
        """
        
        params = []
        
        if program_id:
            query += " AND p.program_id = %s"
            params.append(program_id)
            
        if section_id:
            query += " AND ys.section_id = %s"
            params.append(section_id)
            
        query += " ORDER BY c.curriculum_name, p.program_id, ys.year_level, ys.section_name, cs.day_of_week, cs.start_time"
        
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
    """Get all class schedules for a specific section with assignment status AND CURRICULUM"""
    try:
        section_id = request.args.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # UPDATED QUERY TO INCLUDE CURRICULUM
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
                CONCAT(f.first_name, ' ', f.last_name) as faculty_name,
                c.curriculum_year  -- ADD CURRICULUM YEAR
            FROM class_schedules cs
            JOIN subjects s ON cs.subject_id = s.subject_id
            JOIN year_sections ys ON cs.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
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
    """Get schedules for the logged-in faculty member for timer - WITH CURRICULUM"""
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
            # Admin can see all subjects (using subject_id as schedule_id for frontend compatibility)
            cursor.execute("""
                SELECT DISTINCT
                    s.subject_id as schedule_id,  -- Using subject_id as schedule_id for frontend
                    s.subject_code,
                    s.subject_name,
                    s.class_type,
                    s.units,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    p.program_name as program,  -- Duplicate for frontend compatibility
                    c.curriculum_year  -- ADD CURRICULUM YEAR
                FROM subjects s
                JOIN year_sections ys ON s.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
                WHERE s.status = 'active'
                ORDER BY s.subject_code
            """)
        else:
            # Faculty member sees only their assigned subjects
            cursor.execute("""
                SELECT DISTINCT
                    s.subject_id as schedule_id,  -- Using subject_id as schedule_id for frontend
                    s.subject_code,
                    s.subject_name,
                    s.class_type,
                    s.units,
                    ys.year_level,
                    ys.section_name,
                    p.program_name,
                    p.program_name as program,  -- Duplicate for frontend compatibility
                    c.curriculum_year  -- ADD CURRICULUM YEAR
                FROM faculty_schedules fs
                JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON s.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
                WHERE fs.faculty_id = %s AND s.status = 'active'
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
        logger.error(f"Error getting faculty subjects for timer: {e}", exc_info=True)
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
def get_academic_years_for_program():
    """Get academic years for a program"""
    try:
        program_id = request.args.get('program_id')
        
        print(f"=== DEBUG: Fetching academic years for program: {program_id} ===")
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get ALL academic years for this program - REMOVED active_semester_id references
        cursor.execute("""
            SELECT 
                ay.academic_year,
                ay.academic_year_id,
                ay.status,
                COUNT(DISTINCT ys.section_id) as section_count,
                COUNT(DISTINCT sub.subject_id) as subject_count
            FROM academic_years ay
            LEFT JOIN year_sections ys ON (
                ay.academic_year_id = ys.academic_year_id 
                AND ys.status = 'active'
                AND ys.program_id = %s
            )
            LEFT JOIN subjects sub ON (
                ys.section_id = sub.section_id 
                AND sub.status = 'active'
            )
            WHERE ay.program_id = %s
            GROUP BY ay.academic_year_id, ay.academic_year, ay.status
            ORDER BY ay.academic_year DESC
        """, (program_id, program_id))
        
        academic_years = cursor.fetchall()
        print(f"DEBUG: Processed {len(academic_years)} academic years for response")
        
        # Find the current active academic year
        current_year = None
        for year in academic_years:
            if year['status'] == 'active':
                current_year = year['academic_year']
                break
        
        cursor.close()
        conn.close()
        
        formatted_years = []
        for year in academic_years:
            formatted_years.append({
                'academic_year': year['academic_year'],
                'academic_year_id': year['academic_year_id'],
                'status': year['status'],
                'section_count': year['section_count'] or 0,
                'subject_count': year['subject_count'] or 0,
                'is_current': year['academic_year'] == current_year
            })
        
        print(f"DEBUG: Sending response: {formatted_years}")
        
        return jsonify({
            'success': True,
            'academic_years': formatted_years,
            'current_year': current_year
        })
        
    except Exception as e:
        logger.error(f"Error fetching academic years for program: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_active_academic_year', methods=['GET'])
def get_active_academic_year():
    """Get the current active academic year for the system"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT academic_year FROM academic_years 
            WHERE status = 'active'
            ORDER BY academic_year_id DESC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return jsonify({
                'success': True,
                'academic_year': result['academic_year']
            })
        else:
            # Fallback: calculate based on current date
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if current_month >= 6:  # June or later
                academic_year = f"{current_year}-{current_year + 1}"
            else:
                academic_year = f"{current_year - 1}-{current_year}"
                
            return jsonify({
                'success': True,
                'academic_year': academic_year
            })
            
    except Exception as e:
        print(f"Error getting active academic year: {str(e)}")
        # Final fallback
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if current_month >= 6:
            academic_year = f"{current_year}-{current_year + 1}"
        else:
            academic_year = f"{current_year - 1}-{current_year}"
            
        return jsonify({
            'success': True,
            'academic_year': academic_year
        })


@app.route('/api/set_active_academic_year', methods=['POST'])
@login_required
def set_active_academic_year():
    """Set an academic year as active - USING STATUS ONLY"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        
        if not all([program_id, academic_year]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, set ALL academic years for THIS PROGRAM to inactive
        cursor.execute("UPDATE academic_years SET status = 'inactive' WHERE program_id = %s", (program_id,))
        
        # Then set ONLY the selected academic year to active for THIS PROGRAM
        cursor.execute(
            "UPDATE academic_years SET status = 'active' WHERE program_id = %s AND academic_year = %s",
            (program_id, academic_year)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully set {academic_year} as active academic year for {program_id}'
        })
        
    except Exception as e:
        logger.error(f"Error setting active academic year: {e}")
        return jsonify({'success': False, 'message': str(e)})



@app.route('/api/get_year_sections', methods=['GET'])
@login_required
def get_year_sections():
    """Get all year sections for a program - filtered by semester AND curriculum"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        curriculum_id = request.args.get('curriculum_id')
        
        if not all([program_id, academic_year, semester, curriculum_id]):
            return jsonify({'success': False, 'message': 'Program ID, academic year, semester, and curriculum are required'})
        
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
        
        # Get semester_id with curriculum filter
        cursor.execute(
            """SELECT semester_id FROM semesters 
               WHERE academic_year_id = %s AND semester_number = %s AND curriculum_id = %s""",
            (academic_year_id, semester, curriculum_id)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found for this curriculum'})
        
        semester_id = result['semester_id']
        
        # Get sections for this specific semester AND curriculum - FIXED QUERY
        query = """
            SELECT 
                ys.section_id,
                ys.year_level,
                ys.section_name,
                ys.status,
                ys.academic_year_id,
                ys.curriculum_id,
                COUNT(DISTINCT s.subject_id) as subject_count,
                (SELECT COUNT(*) 
                 FROM students st 
                 WHERE st.section_id = ys.section_id 
                 AND st.status = 'active') as student_count
            FROM year_sections ys
            LEFT JOIN subjects s ON ys.section_id = s.section_id AND s.status = 'active'
            WHERE ys.program_id = %s 
            AND ys.academic_year_id = %s  # Link to academic_years table
            AND ys.semester_id = %s       # Link to semesters table  
            AND ys.curriculum_id = %s     # Link to curricula table
            AND ys.status = 'active'
            GROUP BY ys.section_id, ys.year_level, ys.section_name, ys.status, 
                     ys.academic_year_id, ys.curriculum_id
            ORDER BY ys.year_level, ys.section_name
        """
        
        cursor.execute(query, (program_id, academic_year_id, semester_id, curriculum_id))
        sections = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'sections': sections})
        
    except Exception as error:
        logger.error(f"Error fetching year sections: {error}")
        return jsonify({'success': False, 'message': str(error)})
    
@app.route('/api/add_year_section', methods=['POST'])
@login_required
def add_year_section():
    """Add a new year section with curriculum context"""
    try:
        data = request.get_json()
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester = data.get('semester')
        year_level = data.get('year_level')
        section_name = data.get('section_name')
        curriculum_id = data.get('curriculum_id')
        
        if not all([program_id, academic_year, semester, year_level, section_name, curriculum_id]):
            return jsonify({'success': False, 'message': 'All fields including curriculum are required'})
        
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
        
        # Get semester_id with curriculum filter
        cursor.execute(
            """SELECT semester_id FROM semesters 
               WHERE academic_year_id = %s AND semester_number = %s AND curriculum_id = %s""",
            (academic_year_id, semester, curriculum_id)
        )
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found for this curriculum'})
        
        semester_id = result['semester_id']
        
        # Check if section already exists for this semester and curriculum
        cursor.execute(
            """SELECT section_id FROM year_sections 
               WHERE program_id = %s AND semester_id = %s 
               AND year_level = %s AND section_name = %s""",
            (program_id, semester_id, year_level, section_name)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Section already exists for this semester'})
        
        # Insert new section WITH academic_year_id
        cursor.execute(
            """INSERT INTO year_sections 
               (program_id, academic_year_id, semester_id, year_level, section_name, curriculum_id, status) 
               VALUES (%s, %s, %s, %s, %s, %s, 'active')""",
            (program_id, academic_year_id, semester_id, year_level, section_name, curriculum_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Section added successfully'})
        
    except Exception as e:
        logger.error(f"Error adding year section: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_subjects', methods=['GET'])
@login_required
def get_subjects():
    """Get all subjects for a section"""
    try:
        section_id = request.args.get('section_id')
        
        if not section_id:
            return jsonify({'success': False, 'message': 'Section ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get subjects for this section
        cursor.execute("""
            SELECT 
                subject_id,
                subject_code,
                subject_name, 
                class_type,
                units,
                status,
                created_at,
                updated_at
            FROM subjects 
            WHERE section_id = %s AND status = 'active'
            ORDER BY subject_code
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
    """Add a new subject to a section"""
    try:
        data = request.get_json()
        section_id = data.get('section_id')
        subject_code = data.get('subject_code')
        subject_name = data.get('subject_name')
        class_type = data.get('class_type')
        units = data.get('units')
        
        if not all([section_id, subject_code, subject_name, class_type, units]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if subject code already exists in this section
        cursor.execute(
            "SELECT subject_id FROM subjects WHERE section_id = %s AND subject_code = %s",
            (section_id, subject_code)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Subject code already exists in this section'})
        
        # Insert new subject
        cursor.execute(
            """INSERT INTO subjects 
               (section_id, subject_code, subject_name, class_type, units, status) 
               VALUES (%s, %s, %s, %s, %s, 'active')""",
            (section_id, subject_code, subject_name, class_type, units)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Subject added successfully'})
        
    except Exception as e:
        logger.error(f"Error adding subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_subject', methods=['POST'])
@login_required
def update_subject():
    """Update an existing subject"""
    try:
        data = request.get_json()
        subject_id = data.get('subject_id')
        subject_code = data.get('subject_code')
        subject_name = data.get('subject_name')
        class_type = data.get('class_type')
        units = data.get('units')
        
        if not all([subject_id, subject_code, subject_name, class_type, units]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if subject code already exists in the same section (excluding current subject)
        cursor.execute("""
            SELECT subject_id FROM subjects 
            WHERE subject_code = %s 
            AND section_id = (SELECT section_id FROM subjects WHERE subject_id = %s)
            AND subject_id != %s
        """, (subject_code, subject_id, subject_id))
        
        existing = cursor.fetchone()
        if existing:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Subject code already exists in this section'})
        
        # Update subject
        cursor.execute(
            """UPDATE subjects 
               SET subject_code = %s, subject_name = %s, class_type = %s, units = %s, updated_at = NOW()
               WHERE subject_id = %s""",
            (subject_code, subject_name, class_type, units, subject_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Subject updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_subject', methods=['POST'])
@login_required
def delete_subject():
    """Delete a subject (soft delete by setting status to inactive)"""
    try:
        data = request.get_json()
        subject_id = data.get('subject_id')
        
        if not subject_id:
            return jsonify({'success': False, 'message': 'Subject ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Soft delete by setting status to inactive
        cursor.execute(
            "UPDATE subjects SET status = 'inactive', updated_at = NOW() WHERE subject_id = %s",
            (subject_id,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Subject deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting subject: {e}")
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
        
        cursor.execute("SELECT COUNT(*) as count FROM programs")
        program_count = cursor.fetchone()[0]
        
        if program_count == 0:
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
        
        cursor.execute("SELECT COUNT(*) as count FROM academic_years")
        year_count = cursor.fetchone()[0]
        
        if year_count == 0:
            cursor.execute("SELECT program_id FROM programs LIMIT 1")
            program_result = cursor.fetchone()
            
            if program_result:
                program_id = program_result[0]
                current_year = datetime.now().year
                academic_year = f"{current_year}-{current_year + 1}"
                
                cursor.execute(
                    "INSERT INTO academic_years (program_id, academic_year, status) VALUES (%s, %s, 'active')",
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
def get_programs():
    """Get all programs with their statistics per semester - SHOW ALL PROGRAMS"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                program_id,
                program_name,
                department,
                status,  
                created_at
            FROM programs 
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
        
        # Get all active semesters (keep this as is)
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
            'semesters': semesters,
            'total_count': len(programs)
        })
        
    except Exception as e:
        logger.error(f"Error fetching programs: {e}", exc_info=True)
        # Return empty array instead of failing completely
        return jsonify({
            'success': False,  # Changed to False on error
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
    """Delete a program permanently"""
    try:
        program_id = request.json.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if program exists first
        cursor.execute("SELECT program_id FROM programs WHERE program_id = %s", (program_id,))
        program = cursor.fetchone()
        
        if not program:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Program not found'})
        
        # Permanent deletion
        cursor.execute("DELETE FROM programs WHERE program_id = %s", (program_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Program deleted permanently'})
        
    except Exception as e:
        logger.error(f"Error deleting program: {e}")
        return jsonify({'success': False, 'message': str(e)})     
    
@app.route('/api/get_current_class', methods=['GET'])
@login_required
def get_current_class():
    """Get current class for logged-in faculty member with enhanced details - WITH CURRICULUM"""
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
                    c.curriculum_year,  -- ADD CURRICULUM YEAR
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM class_schedules cs
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
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
                    c.curriculum_year,  -- ADD CURRICULUM YEAR
                    CONCAT(f.first_name, ' ', f.last_name) as instructor_name
                FROM faculty_schedules fs
                JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
                JOIN subjects s ON cs.subject_id = s.subject_id
                JOIN year_sections ys ON cs.section_id = ys.section_id
                JOIN programs p ON ys.program_id = p.program_id
                LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
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
            
            # IMPROVED: Calculate REMAINING TIME from current time to end time
            try:
                # Handle time parsing more safely
                if isinstance(current_class['end_time'], str):
                    end_time = datetime.strptime(current_class['end_time'], '%H:%M:%S').time()
                else:
                    end_time = current_class['end_time']
                
                # Combine with current date for proper time comparison
                end_datetime = datetime.combine(current_time.date(), end_time)
                
                # Calculate remaining time (current time to end time)
                remaining_time = end_datetime - current_time
                
                # Ensure remaining time is not negative (in case class already ended)
                if remaining_time.total_seconds() < 0:
                    remaining_time = timedelta(0)
                
                total_seconds = int(remaining_time.total_seconds())
                
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                
            except Exception as time_error:
                logger.warning(f"Error calculating remaining time: {time_error}")
                # Fallback to default 1 hour if time calculation fails
                hours, minutes, seconds = 1, 0, 0
            
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
                    'remaining_minutes': f"{hours}h {minutes}m {seconds}s",
                    'curriculum': current_class.get('curriculum_year', 'N/A')
                }
            }
            
            # Ensure all values in class_info are JSON serializable
            for key, value in response_data['class_info'].items():
                if isinstance(value, (datetime, date)):
                    response_data['class_info'][key] = value.isoformat()
                elif isinstance(value, timedelta):
                    response_data['class_info'][key] = str(value)
                elif hasattr(value, 'isoformat'):  # Handle time objects
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
            AND ay.status = 'active'  
            AND sem.status = 'active'  
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
    """Get all assigned schedules for a faculty member - WITH CURRICULUM"""
    try:
        faculty_id = request.args.get('faculty_id')
        
        if not faculty_id:
            return jsonify({'success': False, 'message': 'Faculty ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # UPDATED QUERY TO INCLUDE CURRICULUM
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
                fs.faculty_id,
                c.curriculum_year  -- ADD CURRICULUM YEAR
            FROM faculty_schedules fs
            INNER JOIN class_schedules cs ON fs.schedule_id = cs.schedule_id
            INNER JOIN subjects s ON cs.subject_id = s.subject_id
            INNER JOIN year_sections ys ON cs.section_id = ys.section_id
            INNER JOIN programs p ON ys.program_id = p.program_id
            INNER JOIN semesters sem ON ys.semester_id = sem.semester_id
            INNER JOIN academic_years ay ON ys.academic_year_id = ay.academic_year_id
            LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id  -- LEFT JOIN CURRICULA
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
    """Get semesters for switch modal - GROUP BY semester_number to show one per type"""
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
        
        cursor.execute("""
            SELECT 
                s.semester_number,
                -- Check if ANY of this semester type has active status
                MAX(CASE WHEN s.status = 'active' THEN 1 ELSE 0 END) as has_active,
                -- Count how many curricula have this semester
                COUNT(DISTINCT s.curriculum_id) as curriculum_count,
                -- Aggregate section count across ALL curricula
                SUM(
                    (SELECT COUNT(DISTINCT ys.section_id) 
                     FROM year_sections ys 
                     WHERE ys.semester_id = s.semester_id AND ys.status = 'active')
                ) as section_count,
                -- Aggregate subject count across ALL curricula
                SUM(
                    (SELECT COUNT(DISTINCT sub.subject_id) 
                     FROM subjects sub 
                     JOIN year_sections ys ON sub.section_id = ys.section_id
                     WHERE ys.semester_id = s.semester_id AND sub.status = 'active')
                ) as subject_count
            FROM semesters s
            WHERE s.academic_year_id = %s
            GROUP BY s.semester_number
            ORDER BY 
                CASE s.semester_number
                    WHEN '1st Semester' THEN 1
                    WHEN '2nd Semester' THEN 2
                    WHEN 'Summer' THEN 3
                    ELSE 4
                END
        """, (academic_year_id,))
        
        semesters = cursor.fetchall()
        
        # Find the current active semester (check if any semester type has active status)
        current_semester = None
        for semester in semesters:
            if semester['has_active']:
                current_semester = semester['semester_number']
                break
        
        cursor.close()
        conn.close()
        
        # Format response - now only one entry per semester type
        formatted_semesters = []
        for semester in semesters:
            formatted_semesters.append({
                'semester_number': semester['semester_number'],
                'section_count': semester['section_count'] or 0,
                'subject_count': semester['subject_count'] or 0,
                'is_active': bool(semester['has_active']),  # Renamed to is_active for clarity
                'curriculum_count': semester['curriculum_count'] or 0
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
    """Set a semester as ACTIVE - REMOVED academic_years update"""
    try:
        data = request.json
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        semester_number = data.get('semester_number')
        
        if not all([program_id, academic_year, semester_number]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
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
        
        academic_year_id = academic_year_result['academic_year_id']
        
        # Get the semester_id for this specific semester
        cursor.execute(
            "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND semester_number = %s LIMIT 1",
            (academic_year_id, semester_number)
        )
        semester_result = cursor.fetchone()
        if not semester_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Semester not found'})
        
        semester_id = semester_result['semester_id']
        
        # FIRST: Set ALL semesters in this academic year to INACTIVE
        cursor.execute(
            "UPDATE semesters SET status = 'inactive' WHERE academic_year_id = %s",
            (academic_year_id,)
        )
        
        # SECOND: Set the specific semester to ACTIVE
        cursor.execute(
            "UPDATE semesters SET status = 'active' WHERE semester_id = %s",
            (semester_id,)
        )
        
        # REMOVED: Update academic_years table with active_semester_id
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully set {semester_number} as active semester for {academic_year}'
        })
        
    except Exception as e:
        logger.error(f"Error setting active semester: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_active_curricula', methods=['GET'])
@login_required
def get_active_curricula():
    """Get active curricula for a program"""
    try:
        program_id = request.args.get('program_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT curriculum_id, curriculum_name, curriculum_year, description, status
            FROM curricula 
            WHERE program_id = %s AND status = 'active'
            ORDER BY curriculum_year DESC
        """, (program_id,))
        
        curricula = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'curricula': curricula})
        
    except Exception as e:
        logger.error(f"Error fetching active curricula: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_available_year_levels', methods=['GET'])
@login_required
def get_available_year_levels():
    """Get distinct year levels available for a program and curriculum"""
    try:
        program_id = request.args.get('program_id')
        curriculum_id = request.args.get('curriculum_id')
        
        if not program_id:
            return jsonify({'success': False, 'message': 'Program ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if curriculum_id:
            # Get year levels for specific program and curriculum
            cursor.execute("""
                SELECT DISTINCT year_level 
                FROM year_sections 
                WHERE program_id = %s AND curriculum_id = %s AND status = 'active'
                ORDER BY year_level
            """, (program_id, curriculum_id))
        else:
            # Get year levels for program (all curricula)
            cursor.execute("""
                SELECT DISTINCT year_level 
                FROM year_sections 
                WHERE program_id = %s AND status = 'active'
                ORDER BY year_level
            """, (program_id,))
        
        year_levels = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'year_levels': year_levels})
        
    except Exception as e:
        logger.error(f"Error fetching year levels: {e}")
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
        
        cursor.execute("""
            SELECT academic_year, academic_year_id
            FROM academic_years 
            WHERE program_id = %s AND status = 'active'
            LIMIT 1
        """, (program_id,))
        
        academic_year_result = cursor.fetchone()
        
        if not academic_year_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No active academic year found'})
        
        academic_year = academic_year_result['academic_year']
        academic_year_id = academic_year_result['academic_year_id']
        
        # Get active semester
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
def get_sections_with_semester():
    """Get sections for student registration - FIXED CURRICULUM HANDLING"""
    try:
        program_id = request.args.get('program_id')
        year_level = request.args.get('year_level')
        academic_year = request.args.get('academic_year')
        semester = request.args.get('semester')
        curriculum_year = request.args.get('curriculum_year')  # This is curriculum_year like "2023"
        
        print(f"=== DEBUG: Fetching sections ===")
        print(f"Program: {program_id}, Year: {year_level}, Academic Year: {academic_year}")
        print(f"Semester: {semester}, Curriculum Year: {curriculum_year}")
        
        if not all([program_id, year_level, academic_year, semester]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})
        
        # AUTO CONVERT semester numbers to semester names
        semester_map = {
            '1': '1st Semester',
            '2': '2nd Semester', 
            '3': '3rd Semester',
            'first': '1st Semester',
            'second': '2nd Semester',
            'third': '3rd Semester'
        }
        
        semester_name = semester_map.get(semester.lower(), semester)
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
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
        
        academic_year_id = academic_year_result['academic_year_id']
        
        # FIXED: Properly handle curriculum_year parameter
        curriculum_id = None
        
        # If curriculum_year is provided, find the corresponding curriculum_id
        if curriculum_year and curriculum_year != 'null' and curriculum_year != 'undefined':
            cursor.execute(
                "SELECT curriculum_id FROM curricula WHERE program_id = %s AND curriculum_year = %s AND academic_year = %s AND status = 'active'",
                (program_id, curriculum_year, academic_year)
            )
            curriculum_result = cursor.fetchone()
            
            if curriculum_result:
                curriculum_id = curriculum_result['curriculum_id']
                print(f"Found curriculum_id: {curriculum_id} for curriculum_year: {curriculum_year}")
            else:
                print(f"No curriculum found for year: {curriculum_year}")
                # Don't return error here - just proceed without curriculum filter
        
        # Get semester_id - FIXED QUERY
        if curriculum_id:
            # Use the found curriculum_id
            cursor.execute(
                "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND curriculum_id = %s AND (semester_number = %s OR semester_number LIKE %s)",
                (academic_year_id, curriculum_id, semester_name, f'%{semester}%')
            )
        else:
            # Fallback: get any semester if no curriculum provided or not found
            cursor.execute(
                "SELECT semester_id FROM semesters WHERE academic_year_id = %s AND (semester_number = %s OR semester_number LIKE %s) LIMIT 1",
                (academic_year_id, semester_name, f'%{semester}%')
            )
        
        semester_result = cursor.fetchone()
        if not semester_result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': f'Semester not found. Tried: {semester_name}'})
        
        semester_id = semester_result['semester_id']
        
        # Get sections for this semester - FIXED QUERY
        query = """
            SELECT 
                ys.section_id,
                ys.section_name,
                ys.year_level,
                ys.curriculum_id,
                c.curriculum_year,
                c.curriculum_name,
                COUNT(DISTINCT s.subject_id) as subject_count
            FROM year_sections ys
            LEFT JOIN curricula c ON ys.curriculum_id = c.curriculum_id
            LEFT JOIN subjects s ON ys.section_id = s.section_id AND s.status = 'active'
            WHERE ys.program_id = %s 
            AND ys.year_level = %s
            AND ys.semester_id = %s
            AND ys.status = 'active'
        """
        
        params = [program_id, year_level, semester_id]
        
        # Add curriculum filter if we found a curriculum_id
        if curriculum_id:
            query += " AND ys.curriculum_id = %s"
            params.append(curriculum_id)
        
        query += " GROUP BY ys.section_id, ys.section_name, ys.year_level, ys.curriculum_id, c.curriculum_year, c.curriculum_name ORDER BY ys.section_name"
        
        cursor.execute(query, params)
        sections = cursor.fetchall()
        
        print(f"DEBUG: Found {len(sections)} sections")
        for section in sections:
            print(f"Section: {section['section_name']}, Curriculum: {section.get('curriculum_year', 'N/A')}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'sections': sections,
            'debug_info': {
                'program': program_id,
                'year_level': year_level,
                'academic_year': academic_year,
                'semester_requested': semester,
                'semester_used': semester_name,
                'curriculum_year_requested': curriculum_year,
                'curriculum_id_used': curriculum_id,
                'sections_found': len(sections)
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching sections for registration: {e}")
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
    global detectionStopped, tracks, locked_tracks, pending_confirmations  
    global student_presence_tracker, locked_track_reid_features 
    global student_status

    connection = None
    cursor = None
    
    logger.info("🔄 INITIALIZE_SESSION CALLED")

    detectionStopped = False

    tracks = []
    locked_tracks = {}
    pending_confirmations = {}
    student_presence_tracker = {}
    locked_track_reid_features = {}
    student_status = {}
    
    logger.info("🟢 Detection flag RESET to False for new session")
    logger.info(f"🧹 Cleared: {len(tracks)} tracks, {len(locked_tracks)} locked tracks, {len(pending_confirmations)} pending")
    logger.info(f"🧹 Cleared student_status: {len(student_status)} entries")
    
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
        
        # 🎯 CRITICAL FIX: CLEAR PREVIOUS TEMPORARY STUDENTS FOR THIS SESSION
        try:
            cursor.execute("""
                DELETE FROM temporary_students 
                WHERE session_id = %s
            """, (unique_session_id,))
            
            deleted_count = cursor.rowcount
            logger.info(f"🧹 CLEARED {deleted_count} temporary students for new session: {unique_session_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not clear temporary students: {e}")
        
        # ✅ CORRECTED SECTION LOOKUP
        section_id = None
        year_level = data.get('year_level', '')
        section_name = data.get('section', '')
        program_id = data.get('program', '')

        logger.info(f"🔍 Looking for section: Year={year_level}, Section={section_name}, Program={program_id}")

        if year_level and section_name and program_id:
            try:
                # 🎯 EXTRACT NUMERIC YEAR LEVEL (convert "4th Year" to 4)
                year_level_clean = ''.join(filter(str.isdigit, year_level))
                if year_level_clean:
                    year_level_num = int(year_level_clean)
                else:
                    year_level_num = None
                    logger.warning(f"⚠️ Could not extract numeric year level from: {year_level}")
                
                # 🎯 MAP PROGRAM NAME TO PROGRAM ID
                program_map = {
                    'Information Technology': 'IT',
                    'Computer Science': 'CS',
                    'Associate in Computer Technology': 'ACT',
                    'IT': 'IT',
                    'CS': 'CS', 
                    'ACT': 'ACT'
                }
                program_id_to_search = program_map.get(program_id, program_id)
                
                # 🎯 CLEAN SECTION NAME
                section_to_search = section_name.upper().strip()
                
                logger.info(f"🔍 Searching section with: Year={year_level_num}, Section={section_to_search}, Program={program_id_to_search}")
                
                # 🎯 USE THE CORRECT QUERY FROM YOUR DATABASE
                cursor.execute("""
                    SELECT 
                        ys.section_id,
                        ys.program_id,
                        p.program_name,
                        ys.year_level,
                        ys.section_name,
                        ys.semester_id,
                        ys.academic_year_id,
                        ys.status
                    FROM year_sections ys
                    JOIN programs p ON ys.program_id = p.program_id
                    WHERE ys.program_id = %s AND ys.year_level = %s AND ys.section_name = %s
                    ORDER BY ys.section_id
                """, (program_id_to_search, year_level_num, section_to_search))
                
                section_result = cursor.fetchone()
                if section_result:
                    section_id = section_result['section_id']
                    logger.info(f"✅ Found section_id: {section_id}")
                    logger.info(f"📋 Section details: Program={section_result['program_name']}, Year={section_result['year_level']}, Section={section_result['section_name']}")
                else:
                    logger.warning(f"⚠️ No section found for Program={program_id_to_search}, Year={year_level_num}, Section={section_to_search}")
                    
                    # 🎯 FALLBACK: Try to find any active section for this program
                    cursor.execute("""
                        SELECT section_id FROM year_sections 
                        WHERE program_id = %s AND status = 'active'
                        LIMIT 1
                    """, (program_id_to_search,))
                    
                    fallback_result = cursor.fetchone()
                    if fallback_result:
                        section_id = fallback_result['section_id']
                        logger.info(f"🔄 Using fallback section_id: {section_id}")
                    else:
                        logger.error(f"❌ No sections found for program: {program_id_to_search}")
                
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
        
        # 🎯 ADD PROGRAM DISPLAY NAME HELPER FUNCTION
        def get_program_display_name(program_id):
            """Get display name for program ID"""
            program_map = {
                'IT': 'BSIT',
                'CS': 'BSCS', 
                'ACT': 'ACT',
                'Information Technology': 'BSIT',
                'Computer Science': 'BSCS',
                'Associate in Computer Technology': 'ACT'
            }
            return program_map.get(program_id, program_id)
            
        program_display = get_program_display_name(data.get('program', ''))
        
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
            'program_display': program_display,  # 🆕 ADD PROGRAM DISPLAY NAME
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
    """Get students for the current class including both regular and temporary students"""
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
        session_id = request.args.get('session_id')
        
        logger.info(f"🔍 get_class_students CALLED: program='{program}', year_level='{year_level}', section='{section}', session_id='{session_id}'")
        
        if not all([program, year_level, section]):
            return jsonify({'success': False, 'message': 'Missing required parameters'}), 400
        
        # 🎯 CORRECTED: Extract numeric year level from "4th Year"
        year_level_clean = ''.join(filter(str.isdigit, year_level))
        year_level_num = int(year_level_clean) if year_level_clean else None
        
        # Map program names to program_ids
        program_map = {
            'Information Technology': 'IT',
            'Computer Science': 'CS',        
            'Associate in Computer Technology': 'ACT',
            'IT': 'IT',
            'CS': 'CS',
            'ACT': 'ACT'
        }
        
        program_id_to_search = program_map.get(program, program)
        section_to_search = section.upper().strip()
        
        logger.info(f"🔍 Cleaned parameters: program_id='{program_id_to_search}', year_level={year_level_num}, section='{section_to_search}'")
        
        # 🎯 STEP 1: FIRST FIND THE SECTION_ID (USING THE SAME LOGIC AS initialize_session)
        section_id = None
        cursor.execute("""
            SELECT section_id FROM year_sections 
            WHERE program_id = %s AND year_level = %s AND section_name = %s
        """, (program_id_to_search, year_level_num, section_to_search))
        
        section_result = cursor.fetchone()
        if section_result:
            section_id = section_result['section_id']
            logger.info(f"✅ Found section_id: {section_id}")
        else:
            logger.error(f"❌ No section found for {program_id_to_search} {year_level_num}{section_to_search}")
            return jsonify({'success': False, 'message': 'Section not found'}), 404
        
        # 🎯 STEP 2: GET REGULAR STUDENTS USING SECTION_ID (NOT PROGRAM/YEAR/SECTION)
        regular_query = """
            SELECT 
                s.student_id, 
                s.first_name, 
                s.last_name, 
                s.middle_name, 
                s.photo_path, 
                s.status,
                p.program_name,
                ys.year_level, 
                ys.section_name
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.section_id = %s 
            AND s.status = 'active'
            ORDER BY s.last_name, s.first_name
        """
        
        cursor.execute(regular_query, (section_id,))
        regular_students = cursor.fetchall()
        
        logger.info(f"🔍 Found {len(regular_students)} regular students in section {section_id}")
        
        # 🎯 STEP 3: GET TEMPORARY STUDENTS - FIXED: Query attendance table, not temporary_students
        temporary_students = []
        if session_id:
            temp_query = """
                SELECT 
                    name,
                    status,
                    remarks,
                    timestamp
                FROM attendance 
                WHERE session_id = %s 
                AND student_id IS NULL
                ORDER BY timestamp DESC
            """
            cursor.execute(temp_query, (session_id,))
            temp_results = cursor.fetchall()
            
            for temp_student in temp_results:
                # Extract ID from remarks if available, otherwise generate one
                temp_id = None
                display_name = temp_student['name']
                
                if temp_student.get('remarks') and 'temp_id:' in temp_student['remarks']:
                    temp_id = temp_student['remarks'].split('temp_id:')[1].strip()
                else:
                    # Generate ID from name and timestamp
                    temp_id = f"temp_{hash(temp_student['name'] + str(temp_student['timestamp'])) % 10000}"
                
                temporary_students.append({
                    'id': temp_id,
                    'name': display_name,
                    'photo_path': '/static/images/default-avatar.jpg',
                    'status': temp_student['status'] or 'present',
                    'type': 'temporary'
                })
        
        logger.info(f"🔍 Found {len(temporary_students)} temporary students")
        
        # 🎯 STEP 4: FORMAT STUDENTS
        formatted_students = []
        
        for student in regular_students:
            full_name = f"{student['first_name']} {student['last_name']}"
            if student['middle_name']:
                full_name = f"{student['first_name']} {student['middle_name']} {student['last_name']}"
                
            formatted_students.append({
                'id': student['student_id'],
                'name': full_name,
                'firstName': student['first_name'],
                'lastName': student['last_name'],
                'photo_path': student['photo_path'] or '/static/images/default-avatar.jpg',
                'status': 'absent',  # Default status
                'type': 'regular',
                'program': student['program_name'],
                'year_level': student['year_level'],
                'section': student['section_name']
            })
        
        for temp_student in temporary_students:
            formatted_students.append(temp_student)
        
        # 🎯 STEP 5: UPDATE STATUSES FROM ATTENDANCE - FIXED: Handle temporary students
        if session_id:
            # Update regular students
            attendance_query = """
                SELECT student_id, status 
                FROM attendance 
                WHERE session_id = %s 
                AND student_id IS NOT NULL
                AND DATE(timestamp) = CURDATE()
                ORDER BY timestamp DESC
            """
            cursor.execute(attendance_query, (session_id,))
            attendance_records = cursor.fetchall()
            
            # Create a mapping of student_id to latest status
            status_map = {}
            for record in attendance_records:
                if record['student_id'] not in status_map:
                    status_map[record['student_id']] = record['status']
            
            for student in formatted_students:
                if student['type'] == 'regular' and student['id'] in status_map:
                    student['status'] = status_map[student['id']]
                
                # Temporary students already have their status from the attendance query above
        
        detected_count = len([s for s in formatted_students if s['status'] in ['present', 'late']])
        
        logger.info(f"📊 Final student count: {len(formatted_students)} total, {detected_count} detected")
        
        return jsonify({
            'success': True, 
            'students': formatted_students,
            'total_count': len(formatted_students),
            'regular_count': len(regular_students),
            'temporary_count': len(temporary_students),
            'detected_count': detected_count,
            'section_id': section_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching class students: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.route('/api/debug_section_students')
def debug_section_students():
    """Debug route to check students in section 26"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check section 26 details
        cursor.execute("SELECT * FROM year_sections WHERE section_id = 26")
        section = cursor.fetchone()
        
        # Check students in section 26
        cursor.execute("SELECT * FROM students WHERE section_id = 26 AND status = 'active'")
        students = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'section': section,
            'students_count': len(students),
            'students': students
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/debug_students')
def debug_students():
    """Debug route to see all students and their program/year_level/section"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # ✅ CORRECTED: Join with year_sections and programs to get proper data
        cursor.execute("""
            SELECT 
                s.student_id, 
                s.first_name, 
                s.last_name, 
                p.program_name,
                ys.year_level, 
                ys.section_name,
                CONCAT(ys.year_level, ys.section_name) as display_section
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            ORDER BY p.program_name, ys.year_level, ys.section_name
        """)
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
    """Get current status of all students - FIXED: Proper missing/present logic"""
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
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 🎯 CRITICAL FIX: GET SECTION_ID FROM ATTENDANCE_SESSIONS
        section_id = None
        cursor.execute("""
            SELECT section_id FROM attendance_sessions 
            WHERE session_id = %s
        """, (session_id,))
        
        session_data = cursor.fetchone()
        if session_data and session_data.get('section_id'):
            section_id = session_data['section_id']
            logger.info(f"🎯 Using section_id from session: {section_id}")
        else:
            logger.warning(f"⚠️ No section_id found for session {session_id}, falling back to program/year/section lookup")
            # Fallback to old method
            program_map = {
                'Information Technology': 'IT',
                'Computer Science': 'CS', 
                'Associate in Computer Technology': 'ACT',
                'IT': 'IT',
                'CS': 'CS',
                'ACT': 'ACT'
            }
            program_id_to_search = program_map.get(program, program)
            
            # Extract numeric year level
            year_level_clean = ''.join(filter(str.isdigit, year_level))
            year_level_num = int(year_level_clean) if year_level_clean else None
            section_to_search = section.upper() if section else None
            
            cursor.execute("""
                SELECT section_id FROM year_sections 
                WHERE program_id = %s AND year_level = %s AND section_name = %s
            """, (program_id_to_search, year_level_num, section_to_search))
            
            section_result = cursor.fetchone()
            if section_result:
                section_id = section_result['section_id']
                logger.info(f"🔄 Fallback: Found section_id: {section_id}")
        
        if not section_id:
            logger.error(f"❌ No section_id found for session {session_id}")
            return jsonify({'success': False, 'message': 'Section not found'}), 404
        
        # 🎯 CRITICAL FIX: GET STUDENTS BY SECTION_ID (NOT PROGRAM/YEAR/SECTION)
        cursor.execute("""
            SELECT 
                s.student_id, 
                s.first_name, 
                s.last_name,
                ys.program_id,
                p.program_name,
                ys.year_level, 
                ys.section_name
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.section_id = %s 
            AND s.status = 'active'
        """, (section_id,))
        
        students = cursor.fetchall()
        
        logger.info(f"🔍 Found {len(students)} students for section_id: {section_id}")
        
        # ✅ CRITICAL FIX: Get temporary students from CURRENT SESSION only
        cursor.execute("""
            SELECT name, timestamp, status, remarks, session_id
            FROM attendance 
            WHERE student_id IS NULL 
            AND session_id = %s
        """, (session_id,))
        
        temp_students = cursor.fetchall()
        
        student_list = []
        detected_count = 0
        
        # 🎯 CRITICAL FIX: Check missing periods - INITIALIZE as empty list
        missing_student_ids = []  # Initialize as empty list to avoid None
        currently_present_ids = set()
        
        if session_id:
            try:
                # Check database for CURRENTLY missing students (not returned yet)
                cursor.execute("""
                    SELECT student_id FROM missing_periods 
                    WHERE session_id = %s AND returned = FALSE
                """, (session_id,))
                missing_records = cursor.fetchall()
                missing_student_ids = [record['student_id'] for record in missing_records] if missing_records else []
                
                logger.info(f"🔍 DB MISSING CHECK: Found {len(missing_student_ids)} CURRENTLY missing students: {missing_student_ids}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking missing students: {e}")
                missing_student_ids = []  # Ensure it's not None
        
        # Check real-time tracking data for currently present students
        try:
            if locked_tracks:
                for person_id, lock_info in locked_tracks.items():
                    if lock_info and isinstance(lock_info, dict) and lock_info.get('type') == 'student':  # ✅ Added type check
                        student_id = lock_info.get('id')
                        if student_id and isinstance(student_id, str):  # ✅ Ensure student_id is string
                            currently_present_ids.add(student_id)
                
                logger.info(f"🔍 REAL-TIME CHECK: {len(currently_present_ids)} students currently tracked: {list(currently_present_ids)}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking locked_tracks: {e}")
            currently_present_ids = set()  # Ensure it's not None
        
        safe_missing_ids = list(missing_student_ids) if missing_student_ids is not None else []
        safe_present_ids = list(currently_present_ids) if currently_present_ids is not None else []
        
        manual_status_students = set()
        
        for student in students:
            student_id = student['student_id']
            student_name = f"{student['first_name']} {student['last_name']}"
            
            # Priority 1: Check for MANUAL STATUS (excused, etc.) - HIGHEST PRIORITY
            cursor.execute("""
                SELECT status, session_id, remarks, timestamp FROM attendance 
                WHERE student_id = %s AND session_id = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (student_id, session_id))
            
            attendance_record = cursor.fetchone()
            
            is_manual_status = False
            current_status = None
            original_status = attendance_record['status'] if attendance_record else None
            
            if attendance_record:
                current_status = attendance_record['status']
                current_session = attendance_record.get('session_id')
                remarks = attendance_record.get('remarks') or ''
                
                manual_excuse_sessions = ['manual_excuse']  # Only for excused students
                manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
                manual_statuses = ['excused']  # Only truly manual statuses
                
                is_manual_status = (
                    # Only specific manual session types (not manual_add)
                    current_session in manual_excuse_sessions or
                    current_session in manual_status_sessions or
                    # Only specific manual statuses
                    current_status in manual_statuses or
                    # Only specific manual remarks (not temp_id)
                    'Manually marked' in remarks or
                    'Manual status' in remarks or
                    'Manually marked as excused' in remarks
                )
            
            # Priority 1A: If manual status exists, use it and skip other checks
            if is_manual_status:
                logger.info(f"🔒 MANUAL STATUS PROTECTED: {student_name} -> {current_status}")
                student_list.append({
                    'id': student_id,
                    'name': student_name,
                    'status': current_status,  # Use the manual status
                    'type': 'regular',
                    'program': student.get('program_name', program),
                    'year_level': student.get('year_level'),
                    'section': student.get('section_name')
                })
                manual_status_students.add(student_id)  # Track manual status students
                continue  
            
            # 🎯 FIXED LOGIC: Check if student is CURRENTLY DETECTED first (highest priority after manual)
            if student_id in safe_present_ids:
                # Student is currently being tracked - they CANNOT be missing!
                current_status = 'present'
                
                # 🎯 Check if they were previously missing and need to be marked as returned
                if student_id in safe_missing_ids:
                    # Student has returned from missing - restore original status
                    try:
                        cursor.execute("""
                            SELECT original_status FROM missing_periods 
                            WHERE student_id = %s AND session_id = %s AND returned = FALSE
                            ORDER BY missing_start DESC LIMIT 1
                        """, (student_id, session_id))
                        
                        missing_record = cursor.fetchone()
                        if missing_record:
                            original_status = missing_record['original_status']
                            current_status = original_status  # Restore their original status (present/late)
                            
                            # Mark the missing period as returned
                            cursor.execute("""
                                UPDATE missing_periods 
                                SET missing_end = NOW(), returned = TRUE
                                WHERE student_id = %s AND session_id = %s AND returned = FALSE
                            """, (student_id, session_id))
                            
                            # Update attendance record to restore original status
                            cursor.execute("""
                                UPDATE attendance 
                                SET status = %s, timestamp = NOW()
                                WHERE student_id = %s AND session_id = %s
                            """, (original_status, student_id, session_id))
                            
                            logger.info(f"🔄 STUDENT RETURNED: {student_name} -> {original_status}")
                    except Exception as e:
                        logger.error(f"❌ Error marking student as returned: {e}")
                
                logger.info(f"✅ CURRENTLY PRESENT: {student_name}")
            
            # Priority 3: Student is currently marked as missing in database
            elif student_id in safe_missing_ids:
                current_status = 'missing'
                logger.info(f"🎯 CURRENTLY MISSING: {student_name} ({student_id})")
            
            # Priority 4: Check attendance records for original status (fallback)
            else:
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
                            cursor.execute("""
                                UPDATE attendance 
                                SET status = 'late' 
                                WHERE student_id = %s AND session_id = %s AND timestamp = %s
                            """, (student_id, session_id, attendance_record['timestamp']))
                            logger.info(f"🔄 UPDATED TO LATE: {student_name}")
                else:
                    current_status = 'absent'
            
            if current_status is None:
                current_status = 'absent'
                logger.warning(f"⚠️ Status was None for {student_name}, defaulting to 'absent'")
            
            logger.debug(f"🔍 FINAL STATUS: {student_name} -> {current_status}")
            
            if current_status in ['present', 'late']:
                detected_count += 1
            
            student_list.append({
                'id': student_id,
                'name': student_name,
                'status': current_status,
                'type': 'regular',
                'program': student.get('program_name', program),
                'year_level': student.get('year_level'),
                'section': student.get('section_name')
            })
        
        temp_counter = 1
        for temp_student in temp_students:
            temp_name = temp_student['name']
            temp_remarks = temp_student.get('remarks') or ''
            current_status = temp_student['status']
            temp_session_id = temp_student.get('session_id')
            
            if temp_session_id != session_id:
                logger.debug(f"🔍 Skipping temporary student from different session: {temp_name} (session: {temp_session_id})")
                continue
            
            is_manual_temp = False
            temp_remarks = temp_student.get('remarks') or ''
            
            if 'Manually marked' in temp_remarks or 'Manual status' in temp_remarks:
                is_manual_temp = True
                logger.info(f"🔒 MANUAL TEMPORARY STATUS: {temp_name} -> {current_status}")
            
            # Only apply late conversion for non-manual temporary students
            if current_status == 'present' and session_start_time and not is_manual_temp:
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
                        WHERE name = %s AND session_id = %s AND timestamp = %s
                    """, (temp_name, session_id, temp_student['timestamp']))
            
            temp_id = None
            display_name = temp_name
            
            if 'temp_id:' in temp_remarks:
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
            
            existing_temp = next((s for s in student_list if s['id'] == temp_id and s['type'] == 'temporary'), None)
            if existing_temp:
                logger.warning(f"⚠️ DUPLICATE TEMPORARY STUDENT: {temp_id} - {display_name}")
                existing_temp['status'] = current_status
                existing_temp['name'] = display_name
            else:
                student_list.append({
                    'id': temp_id,
                    'name': display_name,
                    'status': current_status,
                    'type': 'temporary',
                    'program': program,
                    'year_level': year_level,
                    'section': section
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
        
        logger.info(f"📊 STATUS SUMMARY: {status_counts} | Real-time present: {len(safe_present_ids)}, DB missing: {len(safe_missing_ids)}, Manual protected: {len(manual_status_students)}")
        
        return jsonify({
            'success': True,
            'students': student_list,
            'detected_count': detected_count,
            'total_count': len(student_list),
            'threshold_seconds': threshold_seconds,
            'session_start_time': session_start_time.isoformat() if session_start_time else None,
            'current_session_id': current_session_id,
            'status_summary': status_counts,
            'missing_count_in_db': len(safe_missing_ids),
            'real_time_present_count': len(safe_present_ids),
            'manual_status_count': len(manual_status_students),
            'section_id_used': section_id  # 🆕 ADD FOR DEBUGGING
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting student status: {e}")
        import traceback
        logger.error(f"❌ Stack trace: {traceback.format_exc()}")
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
    """Handle student management actions - FIXED: Using correct schema with programs and year_sections"""
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
            
            # ✅ CRITICAL FIX: Check if temporary student already exists in THIS session
            cursor.execute("""
                SELECT id, status FROM attendance 
                WHERE student_id IS NULL 
                AND session_id = %s
                AND remarks = %s
                LIMIT 1
            """, (current_session_id, remarks))
            
            existing_temp_student = cursor.fetchone()
            
            if existing_temp_student:
                # ✅ UPDATE existing temporary student instead of creating duplicate
                cursor.execute("""
                    UPDATE attendance 
                    SET status = %s, timestamp = %s,
                        subject_code = %s, subject_name = %s, room = %s, section_id = %s
                    WHERE id = %s
                """, (status, current_time, subject_code, subject_name, room, section_id, existing_temp_student[0]))
                
                logger.info(f"🔄 TEMPORARY STUDENT UPDATED: {student_name} ({student_id}) - {status}")
            else:
                # ✅ INSERT new temporary student
                actual_session_id = current_session_id if current_session_id else 'manual_add'
                
                cursor.execute("""
                    INSERT INTO attendance 
                    (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (None, display_name, current_time, 'student', status, actual_session_id, remarks, subject_code, subject_name, room, section_id))
                
                logger.info(f"✅ TEMPORARY ATTENDANCE ADDED: {student_name} ({student_id}) - {status}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            student_status[student_id] = status
            
            return jsonify({
                'success': True, 
                'title': 'Success',
                'message': f'Temporary attendance added for {student_name}',
                'student_name': student_name,
                'student_id': student_id,
                'student_data': {
                    'id': student_id,
                    'name': student_name,
                    'status': status,
                    'type': 'temporary'
                }
            })
            
        elif action == 'remove':
            student_id = student_data.get('student_id')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # ✅ CORRECTED: Check students table with proper schema
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
                cursor.execute("""
                    DELETE FROM attendance 
                    WHERE student_id IS NULL 
                    AND session_id = %s
                    AND (name LIKE %s OR remarks LIKE %s)
                """, (current_session_id, f"%{student_id}%", f"%temp_id:{student_id}%"))
                
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
                    'message': f'Temporary student with ID {student_id} has been removed'
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
            
            # Parse new section format (e.g., "IT 4A", "CS 3B", "ACT 2C")
            match = re.match(r'(\w+)\s*(\d+)(\w+)', new_section)
            if not match:
                return jsonify({
                    'success': False, 
                    'title': 'Invalid Format',
                    'message': 'Please use format like "IT 4A" or "CS 3B"'
                })
            
            program_id = match.group(1).upper()  # IT, CS, ACT
            year_level = int(match.group(2))     # 1, 2, 3, 4
            section_name = match.group(3).upper()  # A, B, C
            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # ✅ CORRECTED: Get student info with proper schema
            cursor.execute("""
                SELECT s.first_name, s.last_name, ys.section_id
                FROM students s
                JOIN year_sections ys ON s.section_id = ys.section_id
                WHERE s.student_id = %s
            """, (student_id,))
            
            student = cursor.fetchone()
            
            if not student:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False, 
                    'title': 'Student Not Found',
                    'message': f'Student with ID {student_id} was not found'
                })
            
            student_name = f"{student['first_name']} {student['last_name']}"
            
            # Find the target section_id
            cursor.execute("""
                SELECT section_id FROM year_sections 
                WHERE program_id = %s AND year_level = %s AND section_name = %s
                LIMIT 1
            """, (program_id, year_level, section_name))
            
            target_section = cursor.fetchone()
            
            if not target_section:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False, 
                    'title': 'Section Not Found',
                    'message': f'Section {program_id} {year_level}{section_name} was not found'
                })
            
            # ✅ CORRECTED: Update student's section_id
            cursor.execute("""
                UPDATE students 
                SET section_id = %s
                WHERE student_id = %s
            """, (target_section['section_id'], student_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            if student_id in student_status:
                del student_status[student_id]
            
            logger.info(f"🔄 STUDENT TRANSFERRED: {student_id} to {program_id} {year_level}{section_name}")
            return jsonify({
                'success': True, 
                'title': 'Transfer Successful',
                'message': f'Student {student_name} has been transferred to {program_id} {year_level}{section_name}'
            })
            
        elif action == 'excused':
            student_id = student_data.get('student_id')
            remarks = student_data.get('remarks', 'Excused')
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
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
                    
                    # ✅ CRITICAL FIX: Check for existing attendance record in CURRENT SESSION
                    cursor.execute("""
                        SELECT id, status FROM attendance 
                        WHERE student_id = %s AND session_id = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (student_id, current_session_id))
                    
                    existing_record = cursor.fetchone()
                    
                    if existing_record:
                        # ✅ UPDATE existing record (PREVENTS DUPLICATE)
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = 'excused', remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (remarks, current_time, subject_code, subject_name, room, section_id, existing_record['id']))
                        action_type = "updated"
                    else:
                        cursor.execute("""
                            INSERT INTO attendance 
                            (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (student_id, student_name, current_time, 'student', 'excused', current_session_id, remarks, subject_code, subject_name, room, section_id))
                        action_type = "marked"
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    student_status[student_id] = 'excused'
                    
                    logger.info(f"📝 REGULAR STUDENT EXCUSED: {student_name} ({student_id}) - {action_type}")
                    return jsonify({
                        'success': True, 
                        'title': 'Student Excused',
                        'message': f'Student {student_name} has been {action_type} as excused'
                    })
                
                else:
                    cursor.execute("""
                        SELECT id, name FROM attendance 
                        WHERE student_id IS NULL 
                        AND session_id = %s
                        AND (name LIKE %s OR remarks LIKE %s)
                        LIMIT 1
                    """, (current_session_id, f"%{student_id}%", f"%temp_id:{student_id}%"))
                    
                    temp_student = cursor.fetchone()
                    
                    if temp_student:
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = 'excused', remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (remarks, current_time, subject_code, subject_name, room, section_id, temp_student['id']))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        student_status[student_id] = 'excused'
                        
                        logger.info(f"📝 TEMPORARY STUDENT EXCUSED: {temp_student['name']} ({student_id})")
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
            status = student_data.get('status', 'present')  # Can be 'present', 'late', 'absent', 'excused'
            remarks = student_data.get('remarks', 'Manually marked')
            
            # Validate status
            if status not in ['present', 'late', 'absent', 'excused']:
                return jsonify({
                    'success': False, 
                    'title': 'Invalid Status',
                    'message': 'Status must be present, late, absent, or excused'
                })
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
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
                    
                    # ✅ CRITICAL FIX: Check for existing record in CURRENT SESSION
                    cursor.execute("""
                        SELECT id FROM attendance 
                        WHERE student_id = %s AND session_id = %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, (student_id, current_session_id))
                    
                    existing_record = cursor.fetchone()
                    
                    if existing_record:
                        # ✅ UPDATE existing record (PREVENTS DUPLICATE)
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = %s, remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (status, remarks, current_time, subject_code, subject_name, room, section_id, existing_record['id']))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        student_status[student_id] = status
                        
                        logger.info(f"🔄 MANUAL STATUS UPDATED: {student_name} -> {status}")
                        return jsonify({
                            'success': True, 
                            'title': 'Status Updated',
                            'message': f'Student {student_name} status updated to {status}'
                        })
                    else:
                        # ✅ INSERT new record if none exists in this session
                        cursor.execute("""
                            INSERT INTO attendance 
                            (student_id, name, timestamp, person_type, status, session_id, remarks, subject_code, subject_name, room, section_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (student_id, student_name, current_time, 'student', status, current_session_id, remarks, subject_code, subject_name, room, section_id))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        student_status[student_id] = status
                        
                        logger.info(f"✅ MANUAL ATTENDANCE ADDED: {student_name} -> {status}")
                        return jsonify({
                            'success': True, 
                            'title': 'Attendance Added',
                            'message': f'Student {student_name} marked as {status}'
                        })
                else:
                    # TEMPORARY STUDENT
                    # ✅ CRITICAL FIX: Check in CURRENT SESSION only
                    cursor.execute("""
                        SELECT id, name FROM attendance 
                        WHERE student_id IS NULL 
                        AND session_id = %s
                        AND (name LIKE %s OR remarks LIKE %s)
                        LIMIT 1
                    """, (current_session_id, f"%{student_id}%", f"%temp_id:{student_id}%"))
                    
                    temp_student = cursor.fetchone()
                    
                    if temp_student:
                        # ✅ UPDATE existing temporary student (PREVENTS DUPLICATE)
                        cursor.execute("""
                            UPDATE attendance 
                            SET status = %s, remarks = %s, timestamp = %s,
                                subject_code = %s, subject_name = %s, room = %s, section_id = %s
                            WHERE id = %s
                        """, (status, remarks, current_time, subject_code, subject_name, room, section_id, temp_student['id']))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        student_status[student_id] = status
                        
                        logger.info(f"🔄 TEMPORARY STUDENT STATUS UPDATED: {temp_student['name']} -> {status}")
                        return jsonify({
                            'success': True, 
                            'title': 'Status Updated',
                            'message': f'Temporary student {temp_student["name"]} status updated to {status}'
                        })
                    else:
                        cursor.close()
                        conn.close()
                        logger.warning(f"❌ NO ATTENDANCE RECORD: Temporary student ({student_id}) in session {current_session_id}")
                        return jsonify({
                            'success': False, 
                            'title': 'No Attendance Record',
                            'message': f'No attendance record found for temporary student with ID {student_id} in current session'
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
    """Get ALL students from the database - FIXED: Using correct schema with joins"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # ✅ CORRECTED: Join with year_sections and programs to get proper data
        cursor.execute("""
            SELECT 
                s.student_id as id, 
                s.first_name, 
                s.last_name, 
                p.program_name as course,
                ys.year_level,
                ys.section_name,
                CONCAT(ys.year_level, ys.section_name) as year_section,
                s.status
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active'
            ORDER BY p.program_name, ys.year_level, ys.section_name, s.last_name, s.first_name
        """)
        
        students = cursor.fetchall()
        
        # Format student names and create display info
        formatted_students = []
        for student in students:
            formatted_students.append({
                'id': student['id'],
                'name': f"{student['first_name']} {student['last_name']}",
                'course': student['course'],
                'year_section': student['year_section'],
                'year_level': student['year_level'],
                'section_name': student['section_name'],
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

def get_program_display_name(program_id):
    """Get display name for program ID"""
    program_map = {
        'IT': 'BSIT',
        'CS': 'BSCS', 
        'ACT': 'ACT',
        'Information Technology': 'BSIT',
        'Computer Science': 'BSCS',
        'Associate in Computer Technology': 'ACT'
    }
    return program_map.get(program_id, program_id)

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

def set_session_start_time():
    """Set the session start time when class begins"""
    global session_start_time
    session_start_time = datetime.now()
    logger.info(f"🕐 SESSION START TIME SET: {session_start_time}")


@app.route('/api/adjust_session_time', methods=['POST'])
def adjust_session_time():
    global session_total_duration_seconds, session_threshold_seconds
    
    data = request.get_json()
    schedule_id = data.get('schedule_id') 
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
    FIXED: Prevents duplicate absent records and handles temporary students properly
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
            cursor.execute("SELECT COUNT(*) as count FROM students WHERE section_id = %s AND status = 'active'", (section_id,))
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
                    SUM(CASE WHEN status = 'excused' THEN 1 ELSE 0 END) as excused_count,
                    SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent_count
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
                    absent_count = attendance_stats['absent_count'] or 0
                    total_attended = attendance_stats['total_attended'] or 0
                else:
                    present_count = attendance_stats[1] or 0
                    late_count = attendance_stats[2] or 0
                    excused_count = attendance_stats[3] or 0
                    absent_count = attendance_stats[4] or 0
                    total_attended = attendance_stats[0] or 0
            else:
                present_count = late_count = excused_count = absent_count = total_attended = 0

            print(f"🔍 DEBUG Current attendance - Present: {present_count}, Late: {late_count}, Excused: {excused_count}, Absent: {absent_count}, Total Attended: {total_attended}")

            # 7. ✅ CRITICAL FIX: Calculate absent count PROPERLY (considering temporary students)
            # Total records should equal total enrolled + temporary students
            # But absent count should only consider enrolled students who are actually absent
            
            # Get count of temporary students in this session
            cursor.execute("""
                SELECT COUNT(*) as temp_count 
                FROM attendance 
                WHERE session_id = %s 
                AND person_type = 'student' 
                AND student_id IS NULL
            """, (session_id,))
            temp_result = cursor.fetchone()
            temp_student_count = temp_result['temp_count'] if isinstance(temp_result, dict) else temp_result[0]
            
            print(f"🔍 DEBUG Temporary students in session: {temp_student_count}")
            
            # ✅ CORRECT absent calculation: Only enrolled students who don't have attendance records
            cursor.execute("""
                SELECT COUNT(*) as actual_absent_count
                FROM students s 
                WHERE s.section_id = %s 
                AND s.status = 'active'
                AND s.student_id NOT IN (
                    SELECT student_id 
                    FROM attendance 
                    WHERE session_id = %s 
                    AND student_id IS NOT NULL
                    AND status IN ('present', 'late', 'excused')
                )
            """, (section_id, session_id))
            
            absent_result = cursor.fetchone()
            actual_absent_count = absent_result['actual_absent_count'] if isinstance(absent_result, dict) else absent_result[0]
            
            print(f"🔍 DEBUG Actual absent students (enrolled but not present/late/excused): {actual_absent_count}")

            # 8. ✅ CRITICAL FIX: MARK ABSENT STUDENTS WITH PROPER DUPLICATE CHECK
            if actual_absent_count > 0:
                print(f"🔍 DEBUG Marking {actual_absent_count} students as absent")
                try:
                    # Get students who are enrolled but NOT marked as present/late/excused in this session
                    cursor.execute("""
                        SELECT s.student_id, s.first_name, s.last_name 
                        FROM students s 
                        WHERE s.section_id = %s 
                        AND s.status = 'active'
                        AND s.student_id NOT IN (
                            SELECT student_id 
                            FROM attendance 
                            WHERE session_id = %s 
                            AND student_id IS NOT NULL
                            AND status IN ('present', 'late', 'excused')
                        )
                    """, (section_id, session_id))
                    absent_students = cursor.fetchall()
                    
                    print(f"🔍 DEBUG Found {len(absent_students)} students to mark as absent")
                    
                    # ✅ CRITICAL FIX: Check for existing absent records before inserting
                    absent_records_added = 0
                    for student in absent_students:
                        student_id = student['student_id'] if isinstance(student, dict) else student[0]
                        first_name = student['first_name'] if isinstance(student, dict) else student[1]
                        last_name = student['last_name'] if isinstance(student, dict) else student[2]
                        
                        # ✅ CHECK if absent record already exists for this student in this session
                        cursor.execute("""
                            SELECT id FROM attendance 
                            WHERE session_id = %s AND student_id = %s AND status = 'absent'
                        """, (session_id, student_id))
                        existing_absent = cursor.fetchone()
                        
                        if not existing_absent:
                            # Only insert if no absent record exists
                            cursor.execute("""
                                INSERT INTO attendance 
                                (student_id, person_type, name, timestamp, status, session_id, section_id, subject_code, subject_name, room)
                                VALUES (%s, 'student', %s, NOW(), 'absent', %s, %s, %s, %s, %s)
                            """, (student_id, f"{first_name} {last_name}", session_id, section_id, subject_code, subject_name, room))
                            absent_records_added += 1
                        else:
                            print(f"🔍 DEBUG Absent record already exists for student {student_id}, skipping")
                    
                    print(f"🔍 DEBUG Successfully inserted {absent_records_added} new absent records (skipped {len(absent_students) - absent_records_added} duplicates)")
                    
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
                    final_total = final_stats['total_attended'] or 0
                else:
                    final_present = final_stats[1] or 0
                    final_late = final_stats[2] or 0
                    final_absent = final_stats[3] or 0
                    final_excused = final_stats[4] or 0
                    final_total = final_stats[0] or 0
            else:
                final_present = final_late = final_absent = final_excused = final_total = 0

            print(f"🔍 DEBUG Final counts - Present: {final_present}, Late: {final_late}, Absent: {final_absent}, Excused: {final_excused}, Total Records: {final_total}")

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
                total_enrolled,  # Only count enrolled students (excludes temporary)
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
                'duration': duration_time,  # RETURN DURATION
                'temporary_students': temp_student_count
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

@app.route('/api/absent_students_for_enrollment', methods=['GET'])
def get_absent_students():
    """
    API 1: Fetches ALL students in the session's class who haven't been marked PRESENT or LATE in this session.
    FIXED: Using correct schema with programs and year_sections
    """
    session_id = request.args.get('session_id')
    section_id = request.args.get('section_id')

    print(f"🔍 DEBUG /api/absent_students_for_enrollment:")
    print(f"   session_id: {session_id}")
    print(f"   section_id: {section_id}")

    if not session_id or session_id == 'undefined':
        print("❌ ERROR: Missing session_id")
        return jsonify({
            'success': False, 
            'message': 'Missing valid session_id.'
        }), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 🎯 STEP 1: Get session details to find the class info
        print("🔍 DEBUG: Fetching session details...")
        cursor.execute("""
            SELECT session_id, class_name, subject_name, section_id 
            FROM attendance_sessions 
            WHERE session_id = %s
        """, (session_id,))
        session_data = cursor.fetchone()
        
        if not session_data:
            print(f"❌ ERROR: Session {session_id} not found")
            return jsonify({
                'success': False,
                'message': f'Session {session_id} not found.'
            }), 404
        
        print(f"🔍 DEBUG: Session found: {session_data}")
        
        # 🎯 STEP 2: Get section details from year_sections
        session_section_id = session_data.get('section_id')
        section_info = None
        
        if session_section_id:
            print(f"🔍 DEBUG: Fetching section details for section_id: {session_section_id}")
            cursor.execute("""
                SELECT ys.section_id, ys.program_id, ys.year_level, ys.section_name, p.program_name
                FROM year_sections ys
                JOIN programs p ON ys.program_id = p.program_id
                WHERE ys.section_id = %s
            """, (session_section_id,))
            section_info = cursor.fetchone()
            print(f"🔍 DEBUG: Section info: {section_info}")
        
        # 🎯 STEP 3: Get students already marked present/late in THIS session
        print("🔍 DEBUG: Finding students already detected in this session...")
        present_sql = """
            SELECT DISTINCT student_id 
            FROM attendance 
            WHERE session_id = %s AND status IN ('present', 'late')
        """
        cursor.execute(present_sql, (session_id,))
        present_students = cursor.fetchall()
        present_ids = [student['student_id'] for student in present_students] if present_students else []
        
        print(f"🔍 DEBUG: Already detected: {len(present_ids)} students")
        
        # 🎯 STEP 4: Try multiple approaches to find students
        
        # First, let's see what students actually exist in the database
        print("🔍 DEBUG: Checking available students in database...")
        cursor.execute("""
            SELECT DISTINCT 
                p.program_name, 
                ys.year_level, 
                ys.section_name,
                ys.section_id
            FROM students s
            JOIN year_sections ys ON s.section_id = ys.section_id
            JOIN programs p ON ys.program_id = p.program_id
            WHERE s.status = 'active' 
            LIMIT 10
        """)
        available_students = cursor.fetchall()
        print(f"🔍 DEBUG: Available student patterns: {available_students}")
        
        # 🎯 STEP 5: Try to find students using multiple criteria
        undetected_students = []
        
        # Method 1: Try using section_id from session (BEST METHOD)
        if session_section_id:
            print(f"🔍 DEBUG: Trying to find students by section_id: {session_section_id}")
            if present_ids:
                placeholders = ', '.join(['%s'] * len(present_ids))
                students_sql = f"""
                    SELECT 
                        s.student_id, 
                        s.first_name, 
                        s.last_name,
                        p.program_name,
                        ys.year_level, 
                        ys.section_name,
                        CONCAT(ys.year_level, ys.section_name) as display_section
                    FROM students s
                    JOIN year_sections ys ON s.section_id = ys.section_id
                    JOIN programs p ON ys.program_id = p.program_id
                    WHERE s.section_id = %s 
                    AND s.status = 'active'
                    AND s.student_id NOT IN ({placeholders})
                    ORDER BY s.last_name, s.first_name
                """
                cursor.execute(students_sql, [session_section_id] + present_ids)
            else:
                students_sql = """
                    SELECT 
                        s.student_id, 
                        s.first_name, 
                        s.last_name,
                        p.program_name,
                        ys.year_level, 
                        ys.section_name,
                        CONCAT(ys.year_level, ys.section_name) as display_section
                    FROM students s
                    JOIN year_sections ys ON s.section_id = ys.section_id
                    JOIN programs p ON ys.program_id = p.program_id
                    WHERE s.section_id = %s 
                    AND s.status = 'active'
                    ORDER BY s.last_name, s.first_name
                """
                cursor.execute(students_sql, [session_section_id])
            
            undetected_students = cursor.fetchall()
            print(f"🔍 DEBUG: Found {len(undetected_students)} students by section_id")
        
        # Method 2: If no students found by section_id, try by program and year_level/section
        if not undetected_students and section_info:
            print(f"🔍 DEBUG: Trying to find students by program/year_level/section...")
            program_id = section_info.get('program_id')
            year_level = section_info.get('year_level')
            section_name = section_info.get('section_name')
            
            if program_id and year_level and section_name:
                print(f"🔍 DEBUG: Searching for program_id='{program_id}', year_level={year_level}, section_name='{section_name}'")
                
                if present_ids:
                    placeholders = ', '.join(['%s'] * len(present_ids))
                    students_sql = f"""
                        SELECT 
                            s.student_id, 
                            s.first_name, 
                            s.last_name,
                            p.program_name,
                            ys.year_level, 
                            ys.section_name,
                            CONCAT(ys.year_level, ys.section_name) as display_section
                        FROM students s
                        JOIN year_sections ys ON s.section_id = ys.section_id
                        JOIN programs p ON ys.program_id = p.program_id
                        WHERE ys.program_id = %s 
                        AND ys.year_level = %s 
                        AND ys.section_name = %s
                        AND s.status = 'active'
                        AND s.student_id NOT IN ({placeholders})
                        ORDER BY s.last_name, s.first_name
                    """
                    cursor.execute(students_sql, [program_id, year_level, section_name] + present_ids)
                else:
                    students_sql = """
                        SELECT 
                            s.student_id, 
                            s.first_name, 
                            s.last_name,
                            p.program_name,
                            ys.year_level, 
                            ys.section_name,
                            CONCAT(ys.year_level, ys.section_name) as display_section
                        FROM students s
                        JOIN year_sections ys ON s.section_id = ys.section_id
                        JOIN programs p ON ys.program_id = p.program_id
                        WHERE ys.program_id = %s 
                        AND ys.year_level = %s 
                        AND ys.section_name = %s
                        AND s.status = 'active'
                        ORDER BY s.last_name, s.first_name
                    """
                    cursor.execute(students_sql, [program_id, year_level, section_name])
                
                undetected_students = cursor.fetchall()
                print(f"🔍 DEBUG: Found {len(undetected_students)} students by program/year/section")
        
        # Method 3: Last resort - get ALL active students (for debugging)
        if not undetected_students:
            print("🔍 DEBUG: No students found with specific criteria, getting ALL active students for debugging...")
            if present_ids:
                placeholders = ', '.join(['%s'] * len(present_ids))
                students_sql = f"""
                    SELECT 
                        s.student_id, 
                        s.first_name, 
                        s.last_name,
                        p.program_name,
                        ys.year_level, 
                        ys.section_name,
                        CONCAT(ys.year_level, ys.section_name) as display_section
                    FROM students s
                    JOIN year_sections ys ON s.section_id = ys.section_id
                    JOIN programs p ON ys.program_id = p.program_id
                    WHERE s.status = 'active'
                    AND s.student_id NOT IN ({placeholders})
                    ORDER BY p.program_name, ys.year_level, ys.section_name, s.last_name, s.first_name
                """
                cursor.execute(students_sql, present_ids)
            else:
                students_sql = """
                    SELECT 
                        s.student_id, 
                        s.first_name, 
                        s.last_name,
                        p.program_name,
                        ys.year_level, 
                        ys.section_name,
                        CONCAT(ys.year_level, ys.section_name) as display_section
                    FROM students s
                    JOIN year_sections ys ON s.section_id = ys.section_id
                    JOIN programs p ON ys.program_id = p.program_id
                    WHERE s.status = 'active'
                    ORDER BY p.program_name, ys.year_level, ys.section_name, s.last_name, s.first_name
                """
                cursor.execute(students_sql)
            
            all_students = cursor.fetchall()
            print(f"🔍 DEBUG: Total active students in database: {len(all_students)}")
            print(f"🔍 DEBUG: All student patterns: {[(s['student_id'], s['program_name'], s['year_level'], s['section_name']) for s in all_students[:5]]}")
            
            # For now, return all students so we can see what's available
            undetected_students = all_students
        
        print(f"✅ FINAL RESULT: Found {len(undetected_students)} students for enrollment")
        
        # Log the results
        for student in undetected_students[:5]:  # Log first 5 only
            print(f"   👤 {student['student_id']}: {student['first_name']} {student['last_name']} ({student['program_name']} {student['year_level']}{student['section_name']})")
        
        if len(undetected_students) > 5:
            print(f"   ... and {len(undetected_students) - 5} more students")
        
        cursor.close()
        conn.close()
        
        return jsonify(undetected_students), 200

    except Exception as e:
        print(f"❌ ERROR in /api/absent_students_for_enrollment: {str(e)}")
        import traceback
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
    """
    ENROLL UNKNOWN FACE - FIXED: Stores multiple face encodings per student
    """
    data = request.get_json()
    student_id = data.get('student_id')
    face_encoding = data.get('face_encoding')
    unrecognized_face_id = data.get('unrecognized_face_id')
    session_id = data.get('session_id')

    print(f"🎯 ENROLLMENT STARTED:")
    print(f"   Student: {student_id}")
    print(f"   Face ID: {unrecognized_face_id}")
    print(f"   Session: {session_id}")

    if not all([student_id, face_encoding, unrecognized_face_id, session_id]):
        return jsonify({
            'success': False, 
            'message': 'Missing required fields.'
        }), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 1. VALIDATE STUDENT EXISTS
        student_check_sql = "SELECT student_id, first_name, last_name FROM students WHERE student_id = %s"
        cursor.execute(student_check_sql, (student_id,))
        student_data = cursor.fetchone()
        
        if not student_data:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Student {student_id} not found in database.'
            }), 400

        student_name = f"{student_data['first_name']} {student_data['last_name']}"

        # 2. 🎯 STORE MULTIPLE FACE ENCODINGS (don't overwrite, just add new)
        print("💾 Adding new face encoding to student_face_encodings table...")
        encoding_sql = """
            INSERT INTO student_face_encodings (student_id, face_encoding, source, created_at)
            VALUES (%s, %s, %s, NOW())
        """
        cursor.execute(encoding_sql, (student_id, json.dumps(face_encoding), 'manual_enrollment'))
        encoding_id = cursor.lastrowid
        print(f"✅ New face encoding saved to encodings table with ID: {encoding_id}")

        # 3. Check if student already has attendance record for this session
        print("🔍 Checking existing attendance record...")
        check_attendance_sql = """
            SELECT id, status FROM attendance 
            WHERE student_id = %s AND session_id = %s
        """
        cursor.execute(check_attendance_sql, (student_id, session_id))
        existing_attendance = cursor.fetchone()
        
        if existing_attendance:
            print(f"📝 Updating existing attendance record (was: {existing_attendance['status']})...")
            update_attendance_sql = """
                UPDATE attendance 
                SET status = 'present', timestamp = NOW(), name = %s
                WHERE id = %s
            """
            cursor.execute(update_attendance_sql, (student_name, existing_attendance['id']))
            print(f"✅ Updated existing attendance from '{existing_attendance['status']}' to 'present'")
        else:
            # Create new attendance record
            print("📝 Creating new attendance record...")
            attendance_sql = """
                INSERT INTO attendance 
                (student_id, session_id, status, timestamp, name)
                VALUES (%s, %s, 'present', NOW(), %s)
            """
            cursor.execute(attendance_sql, (student_id, session_id, student_name))
            print("✅ New attendance marked as present")

        # 4. Handle unrecognized face cleanup
        print("🗑️ Cleaning up unrecognized face...")
        
        if unrecognized_face_id.startswith('db_'):
            # Database face - update status
            db_id = unrecognized_face_id.replace('db_', '')
            print(f"   Updating database face {db_id}...")
            
            update_sql = """
                UPDATE unrecognized_faces 
                SET final_status = 'enrolled', 
                    notes = %s,
                    updated_at = NOW()
                WHERE id = %s
            """
            notes = f'Enrolled as student {student_id} ({student_name})'
            cursor.execute(update_sql, (notes, db_id))
            print(f"✅ Database face {db_id} marked as enrolled")
            
        else:
            # Memory-only face - save to database for record keeping
            print(f"   Memory face {unrecognized_face_id} - saving to database...")
            insert_sql = """
                INSERT INTO unrecognized_faces 
                (session_id, face_encoding, final_status, notes, created_at)
                VALUES (%s, %s, 'enrolled', %s, NOW())
            """
            notes = f'Enrolled as student {student_id} ({student_name}) - was memory-only face'
            cursor.execute(insert_sql, (session_id, json.dumps(face_encoding), notes))
            print(f"✅ Memory face saved to database as enrolled")

        # 5. Remove from memory
        print("🧹 Removing from memory...")
        remove_success = remove_unknown_face(unrecognized_face_id)
        print(f"   Memory removal: {'Success' if remove_success else 'Failed'}")

        # 6. 🎯 UPDATE GLOBAL KNOWN FACES ARRAY WITH ALL ENCODINGS
        print("🔄 Updating known faces cache with ALL encodings...")
        try:
            global KNOWN_FACE_ENCODINGS_ARRAY, known_face_names, known_face_ids, known_face_types
            
            # Reload ALL face encodings from database
            known_face_encodings = []
            known_face_names = []
            known_face_ids = []
            known_face_types = []
            
            sql = """
                SELECT sfe.student_id, sfe.face_encoding, s.first_name, s.last_name
                FROM student_face_encodings sfe
                JOIN students s ON sfe.student_id = s.student_id
                WHERE sfe.face_encoding IS NOT NULL
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            
            for row in results:
                try:
                    student_id_db = row['student_id']
                    encoding_json = row['face_encoding']
                    
                    if encoding_json:
                        encoding_list = json.loads(encoding_json)
                        known_face_encodings.append(np.array(encoding_list))
                        known_face_names.append(student_id_db)
                        known_face_ids.append(student_id_db)
                        known_face_types.append('student')
                        
                except Exception as e:
                    print(f"⚠️ Error loading face encoding for {student_id_db}: {e}")
                    continue
            
            if known_face_encodings:
                KNOWN_FACE_ENCODINGS_ARRAY = np.array(known_face_encodings)
                print(f"✅ Updated known faces cache: {len(known_face_encodings)} encodings from database")
            else:
                KNOWN_FACE_ENCODINGS_ARRAY = np.array([])
                print("⚠️ No known faces in cache")
                
        except Exception as cache_error:
            print(f"⚠️ Error updating face cache: {cache_error}")

        # 7. Commit all changes
        conn.commit()
        print("💫 All database changes committed")

        # Close connection
        cursor.close()
        conn.close()

        return jsonify({
            'success': True, 
            'message': f'Student {student_name} enrolled successfully! New face encoding added.'
        }), 200

    except Exception as e:
        print(f"❌ ERROR in enrollment: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure connection is closed even on error
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        except:
            pass
            
        return jsonify({
            'success': False,
            'message': f'Enrollment failed: {str(e)}'
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
    global detectionStopped, FRAME_BUFFER
    detectionStopped = True
    FRAME_BUFFER = []  # 🎯 CLEAR THE BUFFER
    print("🔴 Detection stopped via API - buffer cleared")
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/api/resume_detection', methods=['POST'])
def resume_detection():
    global detectionStopped, FRAME_BUFFER
    detectionStopped = False
    FRAME_BUFFER = []  # 🎯 CLEAR THE BUFFER
    print("🟢 Detection resumed via API - buffer cleared")
    return jsonify({'success': True, 'message': 'Detection resumed'})

# ------------------------------------------------------------------
# API Route 4: GET UNRECOGNIZED FACES 
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
        
        # 🆕 CRITICAL FIX: Load faces from database first
        load_faces_from_database()
        
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

def load_faces_from_database():
    """
    Load unrecognized faces from database into memory
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT
    
    try:
        session_id = get_current_session_id()
        if not session_id:
            print("❌ No active session ID for loading faces")
            return
        
        with get_db_cursor() as cursor:
            sql = """
                SELECT id, face_image, face_encoding, image_path, created_at 
                FROM unrecognized_faces 
                WHERE session_id = %s AND final_status = 'pending'
                ORDER BY created_at DESC
            """
            cursor.execute(sql, (session_id,))
            results = cursor.fetchall()
            
            print(f"🔍 Loading {len(results)} faces from database")
            
            for row in results:
                try:
                    db_id = row['id'] if isinstance(row, dict) else row[0]
                    face_image_bytes = row['face_image'] if isinstance(row, dict) else row[1]
                    face_encoding_json = row['face_encoding'] if isinstance(row, dict) else row[2]
                    image_path = row['image_path'] if isinstance(row, dict) else row[3]
                    created_at = row['created_at'] if isinstance(row, dict) else row[4]
                    
                    # Skip if already in memory
                    memory_face_id = f"db_{db_id}"
                    if memory_face_id in UNKNOWN_FACES_FOR_ENROLLMENT:
                        continue
                    
                    # Convert bytes back to numpy image
                    face_crop_img = None
                    if face_image_bytes:
                        try:
                            nparr = np.frombuffer(face_image_bytes, np.uint8)
                            face_crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if face_crop_img is None:
                                # Try to load from image_path if byte decoding fails
                                if image_path and os.path.exists(image_path):
                                    face_crop_img = cv2.imread(image_path)
                        except Exception as img_error:
                            print(f"⚠️ Error decoding face image for db_id {db_id}: {img_error}")
                            continue
                    
                    # Convert JSON back to numpy array
                    face_encoding = None
                    if face_encoding_json:
                        try:
                            encoding_list = json.loads(face_encoding_json)
                            face_encoding = np.array(encoding_list)
                        except Exception as encoding_error:
                            print(f"⚠️ Error decoding face encoding for db_id {db_id}: {encoding_error}")
                            continue
                    
                    if face_crop_img is not None and face_encoding is not None:
                        UNKNOWN_FACES_FOR_ENROLLMENT[memory_face_id] = {
                            'face_crop': face_crop_img,
                            'face_encoding': face_encoding,
                            'timestamp': created_at,
                            'cooldown_until': created_at + timedelta(seconds=30),
                            'track_id': None,
                            'times_seen': 1,
                            'db_id': db_id,
                            'image_path': image_path
                        }
                        print(f"📥 LOADED from DB: Face db_{db_id}")
                        
                except Exception as e:
                    print(f"❌ ERROR loading face from DB: {e}")
                    continue
                    
    except Exception as e:
        print(f"❌ ERROR loading faces from database: {e}")

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
    Check for similar faces (different people) with cooldown
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT
    
    # 🎯 Use a reasonable threshold for similar faces (different people)
    SIMILARITY_THRESHOLD = 0.4  # Normal threshold for similar faces
    
    for unique_id, face_data in UNKNOWN_FACES_FOR_ENROLLMENT.items():
        existing_encoding = face_data.get('face_encoding')
        cooldown_until = face_data.get('cooldown_until')

        if existing_encoding is None:
            continue

        # Check if this is a similar face
        distance = calculate_face_distance(existing_encoding, new_encoding)
        
        # If similar (but not exact duplicate) and in cooldown, block it
        if distance < SIMILARITY_THRESHOLD:
            # 🛑 If cooldown is active, BLOCK this similar face
            if cooldown_until and cooldown_until > current_time:
                remaining = (cooldown_until - current_time).total_seconds()
                print(f"🚫 BLOCKED: Similar face {unique_id} in cooldown ({remaining:.0f}s left)")
                return True, unique_id
            else:
                # Cooldown expired - update the existing face
                print(f"🔄 UPDATING: Similar face {unique_id} cooldown expired")
                face_data.update({
                    'timestamp': current_time,
                    'cooldown_until': current_time + timedelta(seconds=30),
                    'times_seen': face_data.get('times_seen', 0) + 1
                })
                return True, unique_id

    return False, None

def calculate_face_distance(encoding1, encoding2):
    """Calculate face distance with proper error handling and normalization"""
    try:
        enc1 = np.array(encoding1)
        enc2 = np.array(encoding2)
        
        # Ensure encodings are the same length
        if len(enc1) != len(enc2):
            return 1.0
            
        # Normalize encodings for better distance calculation
        enc1_norm = enc1 / np.linalg.norm(enc1)
        enc2_norm = enc2 / np.linalg.norm(enc2)
        
        # Calculate cosine distance (better for face recognition)
        distance = np.arccos(np.clip(np.dot(enc1_norm, enc2_norm), -1.0, 1.0)) / np.pi
        
        return distance
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
    Enhanced function to add unknown faces with STRICT duplicate prevention
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT, ACTIVE_FACE_TRACKS
    
    current_time = datetime.now()
    
    session_id = get_current_session_id()
    if session_id is None:
        logger.error("❌ Cannot save face: No active session ID")
        return False
    
    existing_face_id = find_existing_face_by_encoding(face_encoding)
    if existing_face_id:
        print(f"🎯 EXACT DUPLICATE FOUND: Face {existing_face_id} already exists - skipping")
        
        if existing_face_id in UNKNOWN_FACES_FOR_ENROLLMENT:
            UNKNOWN_FACES_FOR_ENROLLMENT[existing_face_id].update({
                'timestamp': current_time,
                'cooldown_until': current_time + timedelta(seconds=30),
                'times_seen': UNKNOWN_FACES_FOR_ENROLLMENT[existing_face_id].get('times_seen', 0) + 1
            })
            print(f"🔄 UPDATED: Existing face {existing_face_id} timestamp and cooldown")
        
        return False
    
    is_similar, similar_face_id = is_similar_to_unrecognized_face(face_encoding, current_time)
    if is_similar:
        print(f"⏳ SIMILAR FACE IN COOLDOWN: Face {similar_face_id} - skipping duplicate")
        return False
    
    face_id = generate_face_id(face_encoding)
    
    if face_id in UNKNOWN_FACES_FOR_ENROLLMENT:
        print(f"🎯 FACE ID COLLISION: {face_id} already exists - updating")
        UNKNOWN_FACES_FOR_ENROLLMENT[face_id].update({
            'face_crop': face_crop,  
            'timestamp': current_time,
            'cooldown_until': current_time + timedelta(seconds=30),
            'times_seen': UNKNOWN_FACES_FOR_ENROLLMENT[face_id].get('times_seen', 0) + 1,
            'track_id': track_id
        })
        return True
    
    image_path = save_face_image_to_file(face_crop, session_id, track_id)
    if not image_path:
        logger.error("❌ Failed to save face image to file")
        return False
    
    db_face_id = None
    try:
        with get_db_cursor() as cursor:
            # Convert face crop to bytes for database (optional)
            face_image_bytes = None
            try:
                success, buffer = cv2.imencode('.jpg', face_crop)
                if success:
                    face_image_bytes = buffer.tobytes()
            except Exception as img_error:
                logger.warning(f"⚠️ Could not encode face image to bytes: {img_error}")
            
            sql = """
                INSERT INTO unrecognized_faces 
                (session_id, section_id, face_encoding, image_path, face_image, 
                 detection_confidence, bounding_box, final_status, notes, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            section_id = get_current_section_id() or "default"
            encoding_json = json.dumps(face_encoding.tolist()) if hasattr(face_encoding, 'tolist') else json.dumps(face_encoding)
            detection_confidence = 0.8
            bounding_box_json = json.dumps({"track_id": track_id}) if track_id else None
            notes = f"Auto-detected - Track: {track_id}" if track_id else "Auto-detected"
            
            cursor.execute(sql, (
                session_id, 
                section_id, 
                encoding_json, 
                image_path, 
                face_image_bytes,
                detection_confidence,
                bounding_box_json,
                'pending',
                notes,
                current_time
            ))
            
            db_face_id = cursor.lastrowid
            logger.info(f"💾 SAVED: Face to database with ID: {db_face_id}")
            
    except Exception as e:
        logger.error(f"❌ DATABASE ERROR: Failed to save face: {e}")
        # Try fallback insert
        try:
            with get_db_cursor() as cursor:
                sql = """
                    INSERT INTO unrecognized_faces 
                    (session_id, section_id, face_encoding, image_path, final_status)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    session_id, 
                    "default", 
                    encoding_json, 
                    image_path, 
                    'pending'
                ))
                db_face_id = cursor.lastrowid
                logger.info(f"💾 SAVED (fallback): Face to database with ID: {db_face_id}")
        except Exception as fallback_error:
            logger.error(f"❌ DATABASE FALLBACK ALSO FAILED: {fallback_error}")
            return False
    
    # Save to memory
    UNKNOWN_FACES_FOR_ENROLLMENT[face_id] = {
        'face_crop': face_crop,
        'face_encoding': face_encoding,
        'timestamp': current_time,
        'cooldown_until': current_time + timedelta(seconds=30),
        'track_id': track_id,
        'times_seen': 1,
        'db_id': db_face_id,
        'image_path': image_path
    }
    
    logger.info(f"➕ ADDED: New unknown face {face_id} - Database ID: {db_face_id}")
    return True

def find_existing_face_by_encoding(new_encoding):
    """
    STRICT check for EXACT same face encoding (not just similar)
    Returns existing face_id if exact match found, None otherwise
    """
    global UNKNOWN_FACES_FOR_ENROLLMENT
    
    # 🎯 VERY STRICT threshold for exact duplicates
    EXACT_DUPLICATE_THRESHOLD = 0.01  # Almost exact match
    
    for existing_id, face_data in UNKNOWN_FACES_FOR_ENROLLMENT.items():
        existing_encoding = face_data.get('face_encoding')
        
        if existing_encoding is None:
            continue
            
        # Calculate distance between encodings
        distance = calculate_face_distance(existing_encoding, new_encoding)
        
        # If distance is very small, it's the exact same face
        if distance < EXACT_DUPLICATE_THRESHOLD:
            print(f"🎯 EXACT DUPLICATE DETECTED: distance={distance:.6f} (threshold: {EXACT_DUPLICATE_THRESHOLD})")
            return existing_id
    
    return None
    
    # Create a stable string representation of the encoding
    encoding_str = ''.join([f"{x:.8f}" for x in encoding_list])
    face_hash = hashlib.md5(encoding_str.encode()).hexdigest()[:12]
    return f"face-{face_hash}"

def save_face_image_to_file(face_crop, session_id, track_id=None):
    """
    Save face crop to file system and return the path
    """
    try:
        # Create directory if not exists
        os.makedirs('unknown_faces', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        track_suffix = f"_{track_id}" if track_id else ""
        filename = f"unknown_face_{session_id}_{timestamp}{track_suffix}.jpg"
        filepath = os.path.join('unknown_faces', filename)
        
        # Save image
        success = cv2.imwrite(filepath, face_crop)
        if success:
            return filepath
        else:
            logger.error(f"❌ Failed to save face image to: {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error saving face image to file: {e}")
        return None

def get_current_section_id():
    """
    Get current section ID - you need to implement this based on your application
    """
    # TODO: Implement based on your application logic
    # This could be from a global variable, configuration, or database
    return "default_section"  # Replace with actual implementation


def generate_face_id(face_encoding):
    """Generate consistent face ID based on encoding - same encoding = same ID"""
    # Use the entire encoding to generate a consistent hash
    if hasattr(face_encoding, 'tolist'):
        encoding_list = face_encoding.tolist()
    else:
        encoding_list = list(face_encoding)
    
    # Create a stable string representation of the encoding
    encoding_str = ''.join([f"{x:.8f}" for x in encoding_list])
    face_hash = hashlib.md5(encoding_str.encode()).hexdigest()[:12]
    return f"face-{face_hash}"

def background_cleanup():
    """Run cleanup in background every minute"""
    while True:
        try:
            cleanup_unrecognized_faces()
            print(f"🕒 Background cleanup: {len(UNKNOWN_FACES_FOR_ENROLLMENT)} faces, {len(ACTIVE_FACE_TRACKS)} tracks")
        except Exception as e:
            print(f"Background cleanup error: {e}")
        time.sleep(60)  # 1 minute

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
        success = remove_unknown_face(face_id)
        if success:
            return jsonify({'success': True, 'message': 'Face removed from system'}), 200
        else:
            return jsonify({'success': False, 'message': 'Face not found'}), 404
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

# Start background cleanup thread (add this at the bottom)
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

@app.route('/api/student_left', methods=['POST'])
def student_left():
    """Record when a student leaves the classroom - FIXED: Prevents duplicate records"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_id = data.get('session_id')
        
        if not student_id or not session_id:
            logger.error(f"❌ MISSING PARAMS: student_id={student_id}, session_id={session_id}")
            return jsonify({'success': False, 'message': 'Missing student_id or session_id'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # ✅ FIXED: Better student lookup with error handling
        student_name = f"Student {student_id}"  # Default name
        
        try:
            cursor.execute("SELECT first_name, last_name FROM students WHERE student_id = %s", (student_id,))
            student = cursor.fetchone()
            if student:
                student_name = f"{student['first_name']} {student['last_name']}"
            else:
                logger.warning(f"⚠️ Student ID {student_id} not found in students table, using default name")
        except Exception as e:
            logger.warning(f"⚠️ Error fetching student name: {e}, using default name")
        
        # Check current attendance status
        cursor.execute("""
            SELECT id, status, session_id, remarks FROM attendance 
            WHERE student_id = %s AND session_id = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (student_id, session_id))
        
        current_attendance = cursor.fetchone()
        
        if current_attendance:
            current_status = current_attendance['status']
            current_session = current_attendance.get('session_id')
            # ✅ FIXED: Handle None remarks safely
            remarks = current_attendance.get('remarks') or ''  # Convert None to empty string
            
            manual_excuse_sessions = ['manual_excuse']  # Only for excused students
            manual_status_sessions = ['manual_status']  # Only for manual present/late/absent
            manual_statuses = ['excused']  # Only truly manual statuses
            
            is_manual_status = (
                # Only specific manual session types (not manual_add)
                current_session in manual_excuse_sessions or
                current_session in manual_status_sessions or
                # Only specific manual statuses
                current_status in manual_statuses or
                # Only specific manual remarks (not temp_id) - Now safe because remarks is always string
                'Manually marked' in remarks or
                'Manual status' in remarks
            )
            
            if is_manual_status:
                logger.info(f"🔒 PRESERVING MANUAL STATUS: {student_name} has manual status '{current_status}' - NOT marking as missing")
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False, 
                    'message': f'Manual status preserved: {student_name} remains {current_status}'
                }), 400
            else:
                logger.info(f"🔄 AUTO STATUS: {student_name} has status '{current_status}' - CAN mark as missing")
        
        # Check for existing missing record
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
        
        # Get original status for tracking
        original_status = current_attendance['status'] if current_attendance else None
        
        # Insert missing period record
        cursor.execute("""
            INSERT INTO missing_periods (student_id, session_id, missing_start, returned, original_status)
            VALUES (%s, %s, NOW(), FALSE, %s)
        """, (student_id, session_id, original_status))
        
        # Update or create attendance record
        if current_attendance:
            cursor.execute("""
                UPDATE attendance 
                SET status = 'missing', timestamp = NOW()
                WHERE id = %s
            """, (current_attendance['id'],))
            logger.info(f"🔄 UPDATED existing attendance record to 'missing' for {student_name}")
        else:
            cursor.execute("""
                INSERT INTO attendance (student_id, name, timestamp, person_type, status, session_id)
                VALUES (%s, %s, NOW(), 'student', 'missing', %s)
            """, (student_id, student_name, session_id))
            logger.info(f"📝 CREATED new 'missing' attendance record for {student_name}")
        
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
        import traceback
        logger.error(f"❌ Stack trace: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/student_returned', methods=['POST'])
def student_returned():
    """Record when a student returns - FIXED: Prevents duplicate records"""
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
        
        # 🎯 CRITICAL FIX: Check for existing attendance record BEFORE updating
        cursor.execute("""
            SELECT id, status FROM attendance 
            WHERE student_id = %s AND session_id = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (student_id, session_id))
        
        existing_attendance = cursor.fetchone()
        
        if existing_attendance:
            # 🎯 UPDATE existing record (PREVENTS DUPLICATE)
            cursor.execute("""
                UPDATE attendance 
                SET status = %s, timestamp = NOW()
                WHERE id = %s
            """, (original_status, existing_attendance['id']))
            logger.info(f"🔄 UPDATED existing attendance record: {student_name} -> {original_status}")
        else:
            # 🎯 Only INSERT if no record exists
            cursor.execute("""
                INSERT INTO attendance 
                (student_id, name, timestamp, person_type, status, session_id)
                VALUES (%s, %s, NOW(), 'student', %s, %s)
            """, (student_id, student_name, original_status, session_id))
            logger.info(f"📝 CREATED new attendance record: {student_name} -> {original_status}")
        
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

@app.route('/api/get_my_profile', methods=['GET'])
def get_my_profile():
    """Fetches profile information for the currently logged-in user."""
    
    user_id = session.get('user_id')
    user_type = session.get('user_type')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authorized'}), 401
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        profile = None
        
        if user_type == 'student':
            query = """
                SELECT 
                    s.student_id as id, s.first_name, s.middle_name, s.last_name, s.email,
                    p.program_id, p.program_name,
                    ys.year_level, ys.section_id, ys.section_name,
                    'student' as user_type
                FROM students s
                LEFT JOIN year_sections ys ON s.section_id = ys.section_id
                LEFT JOIN programs p ON ys.program_id = p.program_id
                WHERE s.student_id = %s
            """
            cursor.execute(query, (user_id,))
            profile = cursor.fetchone()
            
        elif user_type == 'faculty':
            query = """
                SELECT 
                    faculty_id as id, first_name, middle_name, last_name, email,
                    department, 'faculty' as user_type, role
                FROM faculty
                WHERE faculty_id = %s
            """
            cursor.execute(query, (user_id,))
            profile = cursor.fetchone()

        elif user_type == 'admin':
            query = """
                SELECT 
                    admin_id as id, first_name, middle_name, last_name, email,
                    'admin' as user_type, role
                FROM admins
                WHERE admin_id = %s
            """
            cursor.execute(query, (user_id,))
            profile = cursor.fetchone()
            
        cursor.close()
        conn.close()
        
        if not profile:
            return jsonify({'success': False, 'message': 'Profile not found'}), 404
            
        return jsonify({'success': True, 'profile': profile})
        
    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_my_profile', methods=['POST'])
def update_my_profile():
    """
    Updates the logged-in user's profile information.
    Handles 'student' differently from 'admin'/'faculty'.
    """
    
    # Get user info from session
    user_id = session.get('user_id')
    user_type = session.get('user_type')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authorized'}), 401
        
    try:
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get common fields
        first_name = data.get('first_name')
        middle_name = data.get('middle_name')
        last_name = data.get('last_name')
        email = data.get('email')

        if not all([first_name, last_name, email]):
            return jsonify({'success': False, 'message': 'First name, last name, and email are required'}), 400

        if user_type == 'student':
            section_id = data.get('section_id')
            if not section_id:
                return jsonify({'success': False, 'message': 'Section is required for students'}), 400
            
            query = """
                UPDATE students
                SET first_name = %s, middle_name = %s, last_name = %s, email = %s, section_id = %s
                WHERE student_id = %s
            """
            params = (first_name, middle_name or None, last_name, email, section_id, user_id)
            table = 'students'

        elif user_type == 'faculty':
            query = """
                UPDATE faculty
                SET first_name = %s, middle_name = %s, last_name = %s, email = %s
                WHERE faculty_id = %s
            """
            params = (first_name, middle_name or None, last_name, email, user_id)
            table = 'faculty'

        elif user_type == 'admin':
            query = """
                UPDATE admins
                SET first_name = %s, middle_name = %s, last_name = %s, email = %s
                WHERE admin_id = %s
            """
            params = (first_name, middle_name or None, last_name, email, user_id)
            table = 'admins'
        
        else:
            return jsonify({'success': False, 'message': 'Invalid user type'}), 400

        cursor.execute(query, params)
        conn.commit()
        
        # Update session with new name
        session['first_name'] = first_name
        session['last_name'] = last_name

        cursor.close()
        conn.close()
        
        logger.info(f"User {user_id} ({user_type}) updated their profile.")
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/change_my_password', methods=['POST'])
def change_my_password():
    """Updates the logged-in user's password, checking the correct table."""
    
    # Get user info from session
    user_id = session.get('user_id')
    user_type = session.get('user_type')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authorized'}), 401
        
    try:
        data = request.json
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not all([current_password, new_password]):
            return jsonify({'success': False, 'message': 'All password fields are required'}), 400
            
        if len(new_password) < 8:
             return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Determine which table to query
        if user_type == 'student':
            table, id_column = 'students', 'student_id'
        elif user_type == 'faculty':
            table, id_column = 'faculty', 'faculty_id'
        elif user_type == 'admin':
            table, id_column = 'admins', 'admin_id'
        else:
            return jsonify({'success': False, 'message': 'Invalid user type'}), 400

        # 1. Get current password hash from the correct table
        cursor.execute(f"SELECT password_hash FROM {table} WHERE {id_column} = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # 2. Check if the current password is correct (using YOUR function)
        if not verify_password(current_password, user['password_hash']):
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Incorrect current password'}), 403
            
        # 3. Hash the new password and update the database
        new_password_hash = hash_password(new_password)
        
        cursor.execute(
            f"UPDATE {table} SET password_hash = %s WHERE {id_column} = %s",
            (new_password_hash, user_id)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"User {user_id} ({user_type}) changed their password.")
        return jsonify({'success': True, 'message': 'Password updated successfully'})
        
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/check_email', methods=['POST'])
def check_email():
    """Check if email is already registered"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'success': False, 'message': 'Email is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check in students table
        cursor.execute("SELECT student_id FROM students WHERE email = %s AND status = 'active'", (email,))
        student = cursor.fetchone()
        
        # Also check in faculty table if needed
        cursor.execute("SELECT faculty_id FROM faculty WHERE email = %s AND status = 'active'", (email,))
        faculty = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if student or faculty:
            return jsonify({
                'already_registered': True,
                'message': 'This email is already registered in the system.'
            })
        else:
            return jsonify({
                'already_registered': False,
                'message': 'Email is available for registration.'
            })
            
    except Exception as e:
        logger.error(f"Error checking email: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/student/<student_id>/details')
@login_required
def get_student_details(student_id):
    """Get student details with attendance records"""
    try:
        with get_db_cursor() as cursor:
            # Get student information with program and section details via JOIN
            cursor.execute("""
                SELECT 
                    s.student_id,
                    s.first_name,
                    s.middle_name,
                    s.last_name,
                    s.email,
                    s.status,
                    s.photo_path,
                    p.program_name as course,
                    CONCAT(
                        CASE ys.year_level
                            WHEN 1 THEN '1st'
                            WHEN 2 THEN '2nd'
                            WHEN 3 THEN '3rd'
                            WHEN 4 THEN '4th'
                        END,
                        ' Year ',
                        ys.section_name
                    ) as year_section,
                    ys.program_id,
                    ys.year_level
                FROM students s
                LEFT JOIN year_sections ys ON s.section_id = ys.section_id
                LEFT JOIN programs p ON ys.program_id = p.program_id
                WHERE s.student_id = %s
            """, (student_id,))
            
            student = cursor.fetchone()
            
            if not student:
                return jsonify({
                    'success': False,
                    'message': 'Student not found'
                }), 404
            
            # Get attendance records (latest 50 records for scrolling)
            cursor.execute("""
                SELECT 
                    a.timestamp,
                    CASE 
                        WHEN a.status IS NULL OR a.status = '' OR a.status = 'missing' THEN 'absent'
                        ELSE a.status
                    END as status,
                    a.subject_code,
                    a.subject_name,
                    a.room
                FROM attendance a
                WHERE a.student_id = %s
                AND a.person_type = 'student'
                ORDER BY a.timestamp DESC
                LIMIT 50
            """, (student_id,))
            
            attendance_records = cursor.fetchall()
            
            # Format attendance records
            formatted_attendance = []
            for record in attendance_records:
                formatted_attendance.append({
                    'timestamp': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if record['timestamp'] else '',
                    'status': record['status'],
                    'subject_code': record['subject_code'],
                    'subject_name': record['subject_name'],
                    'room': record['room']
                })
            
            return jsonify({
                'success': True,
                'student': {
                    'student_id': student['student_id'],
                    'first_name': student['first_name'],
                    'middle_name': student['middle_name'] or '',
                    'last_name': student['last_name'],
                    'email': student['email'],
                    'course': student['course'] or 'N/A',
                    'year_section': student['year_section'] or 'N/A',
                    'status': student['status'],
                    'photo_path': student['photo_path'] or '/static/images/default-avatar.jpg',
                    'program_id': student['program_id'],
                    'year_level': student['year_level']
                },
                'attendance': formatted_attendance
            })
            
    except Exception as e:
        print(f"❌ ERROR in get_student_details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error loading student details: {str(e)}'
        }), 500


@app.route('/api/get_curricula', methods=['GET'])
def get_curricula():
    """Get all curricula for a specific program and academic year"""
    try:
        program_id = request.args.get('program_id')
        academic_year = request.args.get('academic_year')
        
        if not program_id or not academic_year:
            return jsonify({
                'success': False,
                'message': 'Program ID and Academic Year are required'
            }), 400
        
        with get_db_cursor(commit=False) as cursor:
            # FIXED QUERY - count sections via semesters table
            query = """
                SELECT 
                    c.curriculum_id,
                    c.curriculum_name,
                    c.curriculum_year,
                    c.description,
                    c.status,
                    c.effective_date,
                    -- Student count using subquery
                    (SELECT COUNT(DISTINCT s.student_id) 
                     FROM students s 
                     WHERE s.curriculum_id = c.curriculum_id 
                     AND s.status = 'active') as student_count,
                    -- Section count - FIXED: count via semesters table
                    (SELECT COUNT(DISTINCT ys.section_id) 
                     FROM year_sections ys 
                     JOIN semesters sem ON ys.semester_id = sem.semester_id
                     WHERE sem.curriculum_id = c.curriculum_id 
                     AND ys.status = 'active') as section_count
                FROM curricula c
                WHERE c.program_id = %s 
                AND c.academic_year = %s
                ORDER BY c.curriculum_year DESC
            """
            
            cursor.execute(query, (program_id, academic_year))
            curricula = cursor.fetchall()
        
        curriculum_list = []
        for curr in curricula:
            curriculum_list.append({
                'curriculum_id': curr['curriculum_id'],
                'curriculum_name': curr['curriculum_name'],
                'curriculum_year': curr['curriculum_year'],
                'description': curr['description'],
                'status': curr['status'],
                'effective_date': curr['effective_date'].strftime('%Y-%m-%d') if curr['effective_date'] else None,
                'student_count': curr['student_count'] or 0,
                'section_count': curr['section_count'] or 0
            })
        
        return jsonify({
            'success': True,
            'curricula': curriculum_list
        })
        
    except Exception as e:
        logger.error(f"Error in get_curricula: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/add_curriculum', methods=['POST'])
def add_curriculum():
    """Add a new curriculum"""
    try:
        data = request.get_json()
        program_id = data.get('program_id')
        academic_year = data.get('academic_year')
        curriculum_name = data.get('curriculum_name')
        curriculum_year = data.get('curriculum_year')
        description = data.get('description', '')
        status = data.get('status', 'active')
        effective_date = data.get('effective_date')
        
        if not all([program_id, academic_year, curriculum_name, curriculum_year]):
            return jsonify({
                'success': False,
                'message': 'Program ID, Academic Year, Curriculum Name, and Curriculum Year are required'
            }), 400
        
        with get_db_cursor(commit=True) as cursor:
            # Check if curriculum already exists
            cursor.execute("""
                SELECT curriculum_id FROM curricula 
                WHERE program_id = %s 
                AND academic_year = %s 
                AND curriculum_year = %s
            """, (program_id, academic_year, curriculum_year))
            
            if cursor.fetchone():
                return jsonify({
                    'success': False,
                    'message': f'Curriculum year "{curriculum_year}" already exists for {academic_year}'
                }), 400
            
            # Insert new curriculum
            cursor.execute("""
                INSERT INTO curricula 
                (program_id, academic_year, curriculum_name, curriculum_year, description, status, effective_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (program_id, academic_year, curriculum_name, curriculum_year, description, status, effective_date))
        
        return jsonify({
            'success': True,
            'message': f'Curriculum "{curriculum_name}" added successfully!'
        })
        
    except Exception as e:
        logger.error(f"Error in add_curriculum: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/update_curriculum', methods=['POST'])
def update_curriculum():
    """Update an existing curriculum"""
    try:
        data = request.get_json()
        curriculum_id = data.get('curriculum_id')
        curriculum_name = data.get('curriculum_name')
        curriculum_year = data.get('curriculum_year')
        description = data.get('description')
        status = data.get('status')
        effective_date = data.get('effective_date')
        
        if not all([curriculum_id, curriculum_name, curriculum_year, status]):
            return jsonify({
                'success': False,
                'message': 'Curriculum ID, Name, Year, and Status are required'
            }), 400
        
        with get_db_cursor(commit=True) as cursor:
            # Update curriculum
            cursor.execute("""
                UPDATE curricula 
                SET curriculum_name = %s, 
                    curriculum_year = %s,
                    description = %s,
                    status = %s,
                    effective_date = %s
                WHERE curriculum_id = %s
            """, (curriculum_name, curriculum_year, description, status, effective_date, curriculum_id))
        
        return jsonify({
            'success': True,
            'message': 'Curriculum updated successfully!'
        })
        
    except Exception as e:
        logger.error(f"Error in update_curriculum: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/delete_curriculum', methods=['POST'])
@login_required
def delete_curriculum():
    """Permanently delete a curriculum"""
    try:
        curriculum_id = request.json.get('curriculum_id')
        
        if not curriculum_id:
            return jsonify({'success': False, 'message': 'Curriculum ID is required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Permanent deletion
        cursor.execute("DELETE FROM curricula WHERE curriculum_id = %s", (curriculum_id,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Curriculum not found'})
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Curriculum deleted permanently'})
        
    except Exception as e:
        logger.error(f"Error deleting curriculum: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/get_curriculum_details', methods=['GET'])
def get_curriculum_details():
    """Get details of a specific curriculum"""
    try:
        curriculum_id = request.args.get('curriculum_id')
        
        if not curriculum_id:
            return jsonify({
                'success': False,
                'message': 'Curriculum ID is required'
            }), 400
        
        with get_db_cursor(commit=False) as cursor:
            # FIXED QUERY - using subqueries instead of joins to avoid cartesian product
            query = """
                SELECT 
                    c.curriculum_id,
                    c.program_id,
                    c.academic_year,
                    c.curriculum_name,
                    c.curriculum_year,
                    c.description,
                    c.status,
                    c.effective_date,
                    -- Student count using subquery
                    (SELECT COUNT(DISTINCT s.student_id) 
                     FROM students s 
                     WHERE s.curriculum_id = c.curriculum_id 
                     AND s.status = 'active') as student_count,
                    -- Section count using subquery
                    (SELECT COUNT(DISTINCT ys.section_id) 
                     FROM year_sections ys 
                     WHERE ys.curriculum_id = c.curriculum_id 
                     AND ys.status = 'active') as section_count
                FROM curricula c
                WHERE c.curriculum_id = %s
            """
            
            cursor.execute(query, (curriculum_id,))
            curriculum = cursor.fetchone()
        
        if not curriculum:
            return jsonify({
                'success': False,
                'message': 'Curriculum not found'
            }), 404
        
        return jsonify({
            'success': True,
            'curriculum': {
                'curriculum_id': curriculum['curriculum_id'],
                'program_id': curriculum['program_id'],
                'academic_year': curriculum['academic_year'],
                'curriculum_name': curriculum['curriculum_name'],
                'curriculum_year': curriculum['curriculum_year'],
                'description': curriculum['description'],
                'status': curriculum['status'],
                'effective_date': curriculum['effective_date'].strftime('%Y-%m-%d') if curriculum['effective_date'] else None,
                'student_count': curriculum['student_count'] or 0,
                'section_count': curriculum['section_count'] or 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_curriculum_details: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == "__main__":
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
            host="192.168.0.100", 
            port=5000,
            debug=False,
            threaded=True,
            ssl_context=ssl_context  
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