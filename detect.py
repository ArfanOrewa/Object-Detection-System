"""
detect.py — Real-time object detection for the assistive Object Detection System.

Uses Ultralytics YOLOv8n (nano model, ~6MB, CPU-friendly, auto-downloads on
first run). All detection logic lives in importable functions so it can be
unit-tested without a camera; `run_detection_loop()` is the live webcam entry
point used by app.py.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

MODEL_NAME = "yolov8n.pt"      # nano model: fast on CPU, downloads automatically
DEFAULT_CONFIDENCE = 0.5       # minimum confidence to keep a detection


@dataclass
class Detection:
    """A single detected object."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.bbox
        return (x1 + x2) / 2.0


# ------------------------------------------------------------------ model

def load_model(model_path: str = MODEL_NAME):
    """Load (and auto-download on first run) the YOLOv8 nano model."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


def detect_objects(model, frame: np.ndarray,
                   conf_threshold: float = DEFAULT_CONFIDENCE) -> List[Detection]:
    """
    Run inference on a single BGR frame.

    Returns a list of Detection objects with confidence >= conf_threshold,
    sorted by confidence (highest first).
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    detections: List[Detection] = []
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(Detection(
                label=names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
            ))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# ------------------------------------------------- geometry / prioritisation

def largest_detection(detections: List[Detection]) -> Optional[Detection]:
    """Return the detection with the largest bounding box (closest object)."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.area)


def estimate_distance(det: Detection, frame_shape: Tuple[int, int]) -> float:
    """
    Rough distance estimate in metres from bounding-box size.

    Heuristic (uncalibrated): assumes a typical object fills ~1/3 of the frame
    height at ~2 metres. distance ≈ 2.0 * (frame_h / 3) / box_h.
    This is approximate — good enough for "near / a few metres" warnings,
    not for measurement. Documented as heuristic on purpose.
    """
    frame_h = frame_shape[0]
    x1, y1, x2, y2 = det.bbox
    box_h = max(1, y2 - y1)
    distance = 2.0 * (frame_h / 3.0) / box_h
    return round(max(0.3, min(distance, 20.0)), 1)


def horizontal_position(det: Detection, frame_width: int) -> str:
    """Classify the object's horizontal position: left / center / right."""
    third = frame_width / 3.0
    cx = det.center_x
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "center"


def format_announcement(det: Detection, frame_shape: Tuple[int, int]) -> str:
    """Build the spoken sentence, e.g. 'Obstacle: chair, 2 meters, center'."""
    h, w = frame_shape[0], frame_shape[1]
    distance = estimate_distance(det, frame_shape)
    position = horizontal_position(det, w)
    return f"Obstacle: {det.label}, {distance} meters, {position}"


# --------------------------------- natural announcements (assistive UX)

def distance_description(det: Detection, frame_shape: Tuple[int, int]) -> str:
    """
    Everyday distance terms for speech: 'close' / 'about N meters' / 'far'.
    Larger bounding box = closer object.
    """
    d = estimate_distance(det, frame_shape)
    if d < 1.0:
        return "close"
    if d <= 3.0:
        n = max(1, round(d))
        return f"about {n} meter" if n == 1 else f"about {n} meters"
    return "far"


def position_phrase(det: Detection, frame_width: int) -> str:
    """Natural position phrase: 'on your left' / 'in the center' / 'on your right'."""
    pos = horizontal_position(det, frame_width)
    return {"left": "on your left",
            "center": "in the center",
            "right": "on your right"}[pos]


def describe_object(det: Detection, frame_shape: Tuple[int, int]) -> str:
    """
    Primary announcement: name first, then distance and position.
    e.g. 'banana, close, on your left'. Show a banana -> app says 'banana'.
    """
    h, w = frame_shape[0], frame_shape[1]
    return (f"{det.label}, {distance_description(det, frame_shape)}, "
            f"{position_phrase(det, w)}")


def _article(word: str) -> str:
    """'a' vs 'an' for natural speech."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def describe_scene(detections: List[Detection],
                   frame_shape: Tuple[int, int],
                   max_objects: int = 5) -> str:
    """
    Full on-demand scene summary, largest objects first.
    e.g. 'I see: a person on the left, a chair in the center, and a banana
    on the right.' Returns 'I don't see anything clearly.' when empty.
    """
    if not detections:
        return "I don't see anything clearly."
    w = frame_shape[1]
    ranked = sorted(detections, key=lambda d: d.area, reverse=True)[:max_objects]
    phrases = [f"{_article(d.label)} {d.label} {position_phrase(d, w)}"
               for d in ranked]
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} and {phrases[1]}"
    else:
        body = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"I see: {body}."


# ------------------------------------------------------------------ drawing

def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of the frame; return it."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# ------------------------------------------------------- live capture loop

def run_detection_loop(model, camera_index: int = 0,
                       conf_threshold: float = DEFAULT_CONFIDENCE,
                       on_frame=None, on_key=None):
    """
    Capture from webcam, run detection, draw overlay, show FPS.

    `on_frame(frame, detections, fps)` is an optional callback invoked every
    frame (used by app.py to drive voice feedback). `on_key(key, frame,
    detections)` is invoked for every keypress; return True from it to quit
    the loop. Built-in: 'q' always quits.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}. "
                           "Check that a webcam is connected.")

    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from camera.")
                break

            detections = detect_objects(model, frame, conf_threshold)
            annotated = draw_detections(frame, detections)

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if on_frame is not None:
                on_frame(frame, detections, fps)

            cv2.imshow("Object Detection — Assistive View (q quit, d describe, t text)",
                       annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if on_key is not None and key not in (0, 255):
                if on_key(chr(key), frame, detections):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Loading {MODEL_NAME} ...")
    _model = load_model()
    print("Starting webcam. Press 'q' to quit.")
    run_detection_loop(_model)
