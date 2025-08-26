import os
import cv2
import time
import dlib
import torch
import numpy as np
import datetime
import threading
import face_recognition
from ultralytics import YOLO
# pylint: disable=import-error
from flask import Flask, Response, render_template, send_from_directory

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

# =========================
# Load known faces
# =========================
known_face_encodings, known_face_names = [], []
if ENABLE_RECOGNITION and os.path.isdir(KNOWN_DIR):
    for person in os.listdir(KNOWN_DIR):
        pdir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            fpath = os.path.join(pdir, fname)
            if not fpath.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = face_recognition.load_image_file(fpath)
            encs = face_recognition.face_encodings(img, model="large", num_jitters=2)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(person)
                print(f"[INFO] Loaded {person} from {fname}")
print(f"[INFO] Known identities: {len(known_face_names)}")

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
        dtracker = dlib.correlation_tracker()
        try:
            dtracker.start_track(rgb, dlib.rectangle(ex1, ey1, ex2, ey2))
        except Exception:
            continue
        new_tracks.append({"tracker": dtracker, "name": name if ENABLE_RECOGNITION else "Face", "last_seen": frame_idx, "box": (x1, y1, x2, y2)})
        mark_attendance(name)
    tracks[:] = new_tracks

# =========================
# Flask streaming
# =========================
app = Flask(__name__)

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


# 1) Timer page (Timer/Timer.html)
@app.route('/')
def timer_page():
    return send_from_directory('Timer', 'Timer.html')

# 2) CamFootage page (templates/CamFootage.html)
@app.route('/camfootage')
def camfootage_page():
    return render_template('CamFootage.html')

# 3) Summary page (Timer/SumSession/Summary.html)
@app.route('/summary')
def summary_page():
    return send_from_directory(os.path.join('Timer', 'SumSession'), 'Summary.html')

# 4) Video feed (used in CamFootage.html)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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