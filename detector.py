"""
detector.py
Wraps the Ultralytics YOLO model and provides frame-by-frame detection
with bounding-box drawing, class filtering, and confidence thresholding.
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

import config


class ObjectDetector:
    """Encapsulates YOLO model loading, inference, and annotation."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        allowed_classes: Optional[List[str]] = None,
    ):
        print(f"[Detector] Loading model `{model_name}` …")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou
        self.allowed_classes = (
            [c.strip() for c in allowed_classes if c.strip()]
            if allowed_classes
            else []
        )
        self.class_names = self.model.model.names  # {0: 'person', 1: 'bicycle', …}
        print(f"[Detector] Model loaded. Classes available: "
              f"{len(self.class_names)}")
        print(f"[Detector] Allowed classes filter: "
              f"{self.allowed_classes if self.allowed_classes else 'ALL'}")

    # ── Core Detection ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO on a single BGR frame and return a list of detections.

        Each detection dict contains:
            object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
        Only objects whose class is in `allowed_classes` (if set) and
        whose confidence ≥ threshold are returned.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.class_names.get(cls_id, f"id_{cls_id}")

                # ── Class filter ───────────────────────────────────────
                if self.allowed_classes and cls_name not in self.allowed_classes:
                    continue

                # ── Confidence filter ──────────────────────────────────
                if conf < self.confidence:
                    continue

                # ── Bounding box (xywh → x,y,top-left + w,h) ──────────
                xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, xyxy)
                bbox_x, bbox_y = x1, y1
                bbox_w, bbox_h = x2 - x1, y2 - y1

                detections.append({
                    "object_class": cls_name,
                    "confidence": conf,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                })

        return detections

    # ── Drawing ────────────────────────────────────────────────────────────
    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw bounding boxes + labels on a copy of the frame.

        Returns:
            Annotated frame (BGR).
        """
        annotated = frame.copy()

        # Colour palette (deterministic per class name)
        def _color(name: str) -> tuple:
            hash_val = hash(name)
            r = (hash_val & 0xFF0000) >> 16
            g = (hash_val & 0x00FF00) >> 8
            b = hash_val & 0x0000FF
            return (int(r), int(g), int(b))

        for det in detections:
            x = det["bbox_x"]
            y = det["bbox_y"]
            w = det["bbox_w"]
            h = det["bbox_h"]
            cls_name = det["object_class"]
            conf = det["confidence"]

            color = _color(cls_name)

            # Rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label background
            label = f"{cls_name} {conf:.2f}"
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                annotated,
                (x, y - text_h - baseline - 4),
                (x + text_w, y),
                color,
                -1,
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    # ── Convenience ────────────────────────────────────────────────────────
    def detect_and_annotate(self, frame: np.ndarray):
        """Run detection AND draw on the frame in one call."""
        detections = self.detect(frame)
        annotated = self.draw_detections(frame, detections)
        return annotated, detections