import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
WEIGHTS_PATH = "yolov8n-face.pt"
CONF_THRESH = 0.45
RESIZE_FACTOR = 0.75
MIN_FACE_SIZE = 40

def test_imports():
    """Test if all required libraries are imported correctly."""
    try:
        logger.info("Testing imports...")
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Ultralytics YOLO version: {YOLO.__module__}")
        logger.info("All imports successful")
        return True
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False

def enhance_lighting(bgr):
    """Enhance image lighting for better face detection."""
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
        lab = cv2.merge([L, A, B])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.error(f"Error in enhance_lighting: {e}")
        return bgr

def test_face_detection(image_path):
    """Test face detection on an image with 30+ people, drawing green bounding boxes."""
    if not os.path.exists(WEIGHTS_PATH):
        logger.error(f"'{WEIGHTS_PATH}' not found. Download yolov8n-face.pt and place it in the same directory.")
        return

    try:
        # Load YOLOv8-Face model
        yolo = YOLO(WEIGHTS_PATH)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        yolo.to(DEVICE)
        logger.info(f"Using device: {DEVICE}  |  Model: {WEIGHTS_PATH}")

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        h, w = frame.shape[:2]
        logger.info(f"Image dimensions: {w}x{h}")

        # Resize and preprocess image
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        frame_eq = enhance_lighting(small_frame)

        # Run face detection
        results = yolo.predict(source=frame_eq, verbose=False, conf=CONF_THRESH, imgsz=640, device=DEVICE)

        dets = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])

                # Scale coordinates back to original image size
                x1 = int(max(0, x1 / RESIZE_FACTOR))
                y1 = int(max(0, y1 / RESIZE_FACTOR))
                x2 = int(min(w-1, x2 / RESIZE_FACTOR))
                y2 = int(min(h-1, y2 / RESIZE_FACTOR))

                # Filter detections based on confidence and size
                if conf >= CONF_THRESH and x2 > x1 and y2 > y1 and (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                    dets.append((x1, y1, x2, y2, conf))

        logger.info(f"Detected {len(dets)} faces in the image")

        # Draw green bounding boxes on the original image
        for (x1, y1, x2, y2, conf) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box (BGR: 0, 255, 0)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the output image
        output_path = "output_detected_faces.jpg"
        cv2.imwrite(output_path, frame)
        logger.info(f"Saved output image with detected faces to {output_path}")

        # Verify if 30+ faces were detected
        if len(dets) >= 30:
            logger.info("Successfully detected 30 or more faces")
        else:
            logger.warning(f"Detected {len(dets)} faces, expected 30 or more. Consider adjusting CONF_THRESH or MIN_FACE_SIZE.")

    except Exception as e:
        logger.error(f"Error during face detection: {e}")

if __name__ == "__main__":
    if test_imports():
        # Replace with the actual path to your image containing 30+ people
        image_path = "static/images/IMG_20250418_153747_639.jpg"
        if os.path.exists(image_path):
            test_face_detection(image_path)
        else:
            logger.error(f"Image path does not exist: {image_path}")
    else:
        logger.error("Cannot proceed with face detection due to import errors")