# Object Detection System — Assistive App for the Visually Impaired

A **low-cost, 100% offline** assistive technology app: point any webcam at a
scene and the system identifies objects in real time, announcing them out loud.
**Hold an object up to the camera and the app speaks its name** — show a banana
→ app says *"banana, close, on your right"*.

## How the assistive flow works

1. **See** — OpenCV captures webcam frames; YOLOv8n (nano model, CPU-friendly)
   detects everyday objects (person, chair, bottle, banana, …) with bounding boxes.
2. **Prioritise** — Detections are ranked by bounding-box size (largest = closest).
3. **Speak** — pyttsx3 (offline text-to-speech, no API key, no internet) announces
   the object with everyday distance terms and position:
   *"banana, close, on your left"* · *"chair, about 2 meters, in the center"* ·
   *"person, far, on your right"*.
4. **Smart announcements** — a NEW object is spoken immediately; the same object
   is never repeated within 10 seconds; if several new objects appear, the
   closest/largest is announced first. Objects closer than 1 metre bypass the
   cooldown as urgent warnings.
5. **On demand** —
   - Press **d**: full scene description —
     *"I see: a person on the left, a chair in the center, and a banana on the right."*
   - Press **t**: text-reading mode — offline OCR reads signs, labels, and book
     pages aloud (*"Text reads: …"*).
6. **Assist** — An on-screen panel lists all currently detected objects with
   confidences, so a sighted helper can follow along.

```
Webcam → YOLOv8n detection → closest NEW object → voice announcement
                        ├── d → scene description
                        └── t → text-reading mode (OCR)
```

## Tech stack

| Layer     | Technology                                      |
|-----------|-------------------------------------------------|
| Language  | Python 3.9+                                     |
| Detection | Ultralytics YOLOv8n, OpenCV                     |
| Voice     | pyttsx3 (offline TTS)                           |
| Text      | EasyOCR (offline, models download once)         |
| Math      | NumPy                                           |
| Tests     | unittest (no camera/mic needed)                 |

**100% offline & free:** the YOLOv8n weights (~6 MB) and EasyOCR models download
once on first run; after that everything — detection, speech, text reading —
works with no internet, no API keys, and no accounts.

## Keyboard controls

| Key | Action                                              |
|-----|-----------------------------------------------------|
| q   | Quit                                                |
| d   | Describe the full scene out loud                    |
| t   | Toggle text-reading mode (OCR) on/off               |

## Project structure

```
object-detection-system/
├── app.py            # main assistive app (detection + voice + OCR + keys)
├── detect.py         # YOLOv8 detection + distance/position/scene phrasing
├── voice.py          # VoiceFeedback: offline TTS, cooldown, smart logic
├── ocr.py            # TextReader: offline OCR for text-reading mode
├── test_assist.py    # unit tests (no camera/mic needed)
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```

## Quick start

```bash
pip install -r requirements.txt
python app.py                 # camera 0, voice on
python app.py --camera 1      # different webcam
python app.py --no-voice      # visual panel only
```

Hold an object up to the camera → the app speaks its name. Press **d** for a
scene description, **t** for text-reading mode, **q** to quit. See
[HOW_TO_RUN.md](HOW_TO_RUN.md) for OS-specific setup (Windows / macOS / Linux).

## Design notes

- **Low-cost by design:** runs on any laptop webcam + CPU; no GPU, no cloud.
- **Distance is a heuristic** (uncalibrated): good for "near / a few metres"
  warnings, not measurement — documented in `detect.estimate_distance`.
- **Graceful degradation:** if no audio device exists, voice falls back to
  console logging instead of crashing; if no camera is found, the app exits
  with a clear message.
