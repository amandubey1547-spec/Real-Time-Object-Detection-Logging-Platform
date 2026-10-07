"""
ui.py
Streamlit-based dashboard that shows:
  • Live annotated video feed
  • Real-time detection metrics
  • Recent database logs
"""

import streamlit as st
import cv2
import pandas as pd
import time
from datetime import datetime
from collections import Counter

import config
from detector import ObjectDetector
from database import DatabaseManager


# ═══════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
)

st.title("🔍 Real-Time Object Detection & Logging Platform")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — User Controls
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=config.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Only detections above this confidence are shown & logged.",
    )

    allowed_classes_input = st.text_area(
        "Allowed Classes (comma-separated)",
        value=",".join(config.ALLOWED_CLASSES) if config.ALLOWED_CLASSES else "",
        help="Leave empty to detect ALL classes.",
    )
    allowed_classes = (
        [c.strip() for c in allowed_classes_input.split(",") if c.strip()]
        if allowed_classes_input.strip()
        else []
    )

    camera_source = st.selectbox(
        "Camera Source",
        options=[0, 1],
        format_func=lambda x: "Default Webcam" if x == 0 else f"Camera {x}",
    )

    log_to_db = st.checkbox("Log Detections to MySQL", value=True)
    show_fps = st.checkbox("Show FPS", value=True)

    st.markdown("---")
    st.header("🗄️ Database Status")
    db_status_placeholder = st.empty()

# ═══════════════════════════════════════════════════════════════════════════
# Initialise objects (cached across reruns)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_detector():
    return ObjectDetector(
        confidence=confidence_threshold,
        allowed_classes=allowed_classes if allowed_classes else None,
    )

@st.cache_resource
def get_database():
    return DatabaseManager()

detector = get_detector()
db = get_database()

# Update detector settings live
detector.confidence = confidence_threshold
detector.allowed_classes = allowed_classes

# ═══════════════════════════════════════════════════════════════════════════
# Layout — Two columns
# ═══════════════════════════════════════════════════════════════════════════
col_video, col_stats = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Live Metrics")
    fps_placeholder = st.empty()
    count_placeholder = st.empty()
    class_count_placeholder = st.empty()

st.markdown("---")
st.subheader("📋 Recent Detection Logs")
log_table_placeholder = st.empty()

# ═══════════════════════════════════════════════════════════════════════════
# Video Loop
# ═══════════════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(camera_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

frame_count = 0
start_time = time.time()
detection_counter = Counter()
last_log_refresh = time.time()

# Session state for stopping
if "running" not in st.session_state:
    st.session_state.running = True

stop_button = st.button("⏹️ Stop Detection")

if stop_button:
    st.session_state.running = False

if not st.session_state.running:
    cap.release()
    db.close()
    st.success("Detection stopped. Refresh the page to restart.")
    st.stop()

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Could not read from camera. Check your camera source.")
        break

    # ── Detect & Annotate ───────────────────────────────────────────────────
    annotated_frame, detections = detector.detect_and_annotate(frame)

    # ── Draw detection count on frame ───────────────────────────────────────
    cv2.putText(
        annotated_frame,
        f"Detections: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # ── Log to database ─────────────────────────────────────────────────────
    if log_to_db and detections and db.is_connected():
        db.insert_detections_batch(detections)

    # ── Update metrics ──────────────────────────────────────────────────────
    for det in detections:
        detection_counter[det["object_class"]] += 1

    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # ── Convert BGR → RGB for Streamlit ─────────────────────────────────────
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # ── Update UI elements ──────────────────────────────────────────────────
    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    if show_fps:
        fps_placeholder.metric("FPS", f"{fps:.1f}")

    count_placeholder.metric(
        "Total Detections (session)", sum(detection_counter.values())
    )

    # Class-wise counts
    if detection_counter:
        class_df = pd.DataFrame(
            list(detection_counter.items()),
            columns=["Object Class", "Count"],
        ).sort_values("Count", ascending=False)
        class_count_placeholder.dataframe(class_df, use_container_width=True, hide_index=True)

    # ── Refresh log table periodically ──────────────────────────────────────
    if time.time() - last_log_refresh > config.LOG_REFRESH_SECONDS:
        if db.is_connected():
            logs = db.fetch_recent_logs(limit=config.MAX_LOG_ROWS_DISPLAY)
            if logs:
                log_df = pd.DataFrame(logs)
                log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
                log_df["confidence"] = log_df["confidence"].round(4)
                log_table_placeholder.dataframe(
                    log_df, use_container_width=True, hide_index=True
                )
            else:
                log_table_placeholder.info("No logs yet.")
        else:
            log_table_placeholder.warning("Database not connected.")

        # DB status in sidebar
        if db.is_connected():
            total = db.fetch_total_detections()
            db_status_placeholder.success(
                f"✅ Connected\n\nTotal rows: {total:,}"
            )
        else:
            db_status_placeholder.error("❌ Not connected")

        last_log_refresh = time.time()

    # ── Small sleep to prevent CPU hogging ──────────────────────────────────
    time.sleep(1.0 / config.FPS_LIMIT)

# ── Cleanup ─────────────────────────────────────────────────────────────────
cap.release()
db.close()