"""
app.py — Object Detection System: assistive app for visually impaired users.

Primary interaction: HOLD AN OBJECT UP TO THE CAMERA and the app identifies it
and SPEAKS its name out loud — e.g. show a banana → app says "banana".

Combines real-time YOLOv8 detection (detect.py) with offline voice feedback
(voice.py) and offline text reading (ocr.py):

  * Every frame, detections are ranked by bounding-box size (largest = closest).
  * NEW objects are announced immediately ("banana, close, on your left");
    the same object is not repeated within 10 seconds.
  * Objects closer than URGENT_DISTANCE_M bypass the cooldown (urgent warning).
  * Press 'd' any time for a full scene description:
    "I see: a person on the left, a chair in the center, and a banana on the right."
  * Press 't' to toggle text-reading mode: the app reads signs/labels/pages aloud.
  * A sighted-helper panel lists all currently detected objects on screen.

Keys:  q = quit   d = describe scene   t = toggle text-reading mode

Usage:
    python app.py                 # camera 0, voice on
    python app.py --camera 1       # use a different camera
    python app.py --no-voice      # detection + on-screen panel only

100% offline: no API keys, no internet needed after the models download.
"""

import argparse
import time

import cv2

from detect import (MODEL_NAME, describe_object, describe_scene,
                    draw_detections, estimate_distance, largest_detection,
                    load_model, run_detection_loop)
from voice import VoiceFeedback

try:
    from ocr import TextReader
    _OCR_IMPORT_OK = True
except Exception:
    TextReader = None
    _OCR_IMPORT_OK = False

URGENT_DISTANCE_M = 1.0   # closer than this -> announce immediately (force)
COOLDOWN_S = 10.0         # per-object re-announcement cooldown
OCR_INTERVAL_S = 3.0      # minimum seconds between OCR reads in text mode


def parse_args():
    p = argparse.ArgumentParser(description="Assistive object detection app.")
    p.add_argument("--camera", type=int, default=0,
                   help="webcam index (default: 0)")
    p.add_argument("--no-voice", action="store_true",
                   help="disable voice feedback (visual panel only)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="detection confidence threshold (default: 0.5)")
    return p.parse_args()


def draw_helper_panel(frame, detections, text_mode=False):
    """
    On-screen panel for sighted helpers: lists every detected object with
    its confidence, plus the current mode and key hints.
    """
    panel_w, row_h = 300, 26
    rows = max(1, len(detections))
    panel_h = 34 + row_h * min(rows, 8) + 30
    x0 = frame.shape[1] - panel_w - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 10), (x0 + panel_w, 10 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    mode = "TEXT MODE" if text_mode else "OBJECT MODE"
    cv2.putText(frame, f"{mode}  (q/d/t)", (x0 + 10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    for i, det in enumerate(detections[:8]):  # cap at 8 rows
        y = 34 + 26 + row_h * i
        cv2.putText(frame, f"- {det.label} ({det.confidence:.0%})",
                    (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1)
    return frame


def main():
    args = parse_args()

    print(f"Loading model ({MODEL_NAME}) ...")
    model = load_model()
    voice = VoiceFeedback(cooldown=COOLDOWN_S, enabled=not args.no_voice)
    reader = TextReader() if _OCR_IMPORT_OK else None
    print(f"Voice feedback: {'OFF' if args.no_voice else 'ON'}")
    print(f"Text reader: {'available' if reader and reader.available else 'NOT installed (pip install easyocr)'}")
    print(f"Opening camera {args.camera} ... keys: q=quit  d=describe  t=text mode")

    state = {"text_mode": False, "last_ocr": 0.0}

    def on_key(key, frame, detections):
        if key == "d":
            # On-demand scene description.
            voice.announce("__scene__", describe_scene(detections, frame.shape),
                           force=True)
        elif key == "t":
            # Toggle text-reading mode.
            state["text_mode"] = not state["text_mode"]
            if state["text_mode"]:
                if reader and reader.available:
                    voice.announce("__mode__", "Text reading mode on. "
                                   "Hold text up to the camera.", force=True)
                else:
                    state["text_mode"] = False
                    voice.announce("__mode__", "Text reader not installed. "
                                   "Run pip install easyocr.", force=True)
            else:
                voice.announce("__mode__", "Object mode.", force=True)
        return False  # never quit from here ('q' is handled by the loop)

    def on_frame(frame, detections, fps):
        # Sighted-helper panel (drawn every frame).
        draw_helper_panel(frame, detections, text_mode=state["text_mode"])

        if state["text_mode"] and reader and reader.available:
            # Throttled OCR: read aloud at most every OCR_INTERVAL_S seconds.
            now = time.time()
            if now - state["last_ocr"] >= OCR_INTERVAL_S:
                state["last_ocr"] = now
                try:
                    spoken = reader.describe_text(frame)
                except Exception as exc:
                    spoken = f"Text reading failed: {exc}"
                voice.announce("__ocr__", spoken, force=True)
            return  # in text mode, object announcements are paused

        # Object mode: smart announcements.
        # 1. Update visibility so reappearing objects count as NEW.
        voice.mark_visible([d.label for d in detections])
        # 2. Rank largest (closest) first; announce the top new one.
        for det in sorted(detections, key=lambda d: d.area, reverse=True):
            text = describe_object(det, frame.shape)
            urgent = estimate_distance(det, frame.shape) < URGENT_DISTANCE_M
            if voice.announce(det.label, text, force=urgent):
                break  # one announcement per frame at most

    run_detection_loop(model, camera_index=args.camera,
                       conf_threshold=args.conf,
                       on_frame=on_frame, on_key=on_key)
    print("Goodbye.")


if __name__ == "__main__":
    main()
