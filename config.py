"""
config.py
Central configuration for the Real-Time Object Detection & Logging Platform.
All tuneable parameters live here so no magic numbers are scattered in code.
"""

import os
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── YOLO / Model Settings ──────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "yolov8n.pt")   # pre-trained weights
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

# Classes the model is *allowed* to detect.  Empty list → detect everything.
# Example: ["person", "cell phone", "laptop", "book", "bottle"]
ALLOWED_CLASSES: list = os.getenv(
    "ALLOWED_CLASSES", "person,cell phone,laptop,bottle"
).split(",")

# ── Camera Settings ─────────────────────────────────────────────────────────
CAMERA_SOURCE: int = int(os.getenv("CAMERA_SOURCE", "0"))  # 0 = default webcam
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_LIMIT: int = 30  # cap so we don't flood the DB

# ── Database Settings (loaded from .env — never hard-coded) ─────────────────
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
DB_USER: str = os.getenv("DB_USER", "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "vision_platform")
DB_TABLE: str = os.getenv("DB_TABLE", "detection_logs")

# ── UI / Streamlit Settings ─────────────────────────────────────────────────
PAGE_TITLE: str = "🔍 Real-Time Object Detection Platform"
PAGE_ICON: str = "🔎"
LOG_REFRESH_SECONDS: int = 5   # how often the log table refreshes
MAX_LOG_ROWS_DISPLAY: int = 50  # rows shown in the dashboard table