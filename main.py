"""
main.py
Entry point for the Real-Time Object Detection & Logging Platform.

Usage:
    # Launch the Streamlit dashboard (recommended)
    python main.py --mode ui

    # Run headless (no UI, just console output + DB logging)
    python main.py --mode headless

    # Run headless with a video file instead of webcam
    python main.py --mode headless --source video.mp4
"""

import argparse
import sys
import time
from collections import Counter

import cv2

import config
from detector import ObjectDetector
from database import DatabaseManager


def run_headless(source=None):
    """Run detection without a UI — prints to console and logs to DB."""
    detector = ObjectDetector()
    db = DatabaseManager()

    cap = cv2.VideoCapture(source if source else config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    print("[INFO] Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()
    counter = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        annotated, detections = detector.detect_and_annotate(frame)

        # Log to DB
        if detections and db.is_connected():
            db.insert_detections_batch(detections)

        for det in detections:
            counter[det["object_class"]] += 1

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Console output
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}  |  Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Object Detection — Press Q to Quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

    print(f"\n{'='*50}")
    print(f"Session Summary")
    print(f"{'='*50}")
    print(f"Total frames : {frame_count}")
    print(f"Average FPS  : {fps:.1f}")
    print(f"Detections by class:")
    for cls, cnt in counter.most_common():
        print(f"  {cls:>20s} : {cnt}")
    print(f"{'='*50}")


def run_ui():
    """Launch the Streamlit dashboard."""
    import subprocess

    print("[INFO] Launching Streamlit dashboard …")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "ui.py",
         "--server.port=8501", "--server.address=localhost"],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection & Logging Platform"
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "headless"],
        default="ui",
        help="Run mode: 'ui' (Streamlit dashboard) or 'headless' (OpenCV window)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source for headless mode (path to file or camera index)",
    )
    args = parser.parse_args()

    if args.mode == "ui":
        run_ui()
    else:
        source = int(args.source) if args.source and args.source.isdigit() else args.source
        run_headless(source=source)


if __name__ == "__main__":
    main()