# How to Run — Object Detection System

Works on **Windows**, **macOS**, and **Linux**. You need Python 3.9+, a webcam,
and speakers/headphones for voice feedback.

> **Note:** the first run downloads the YOLOv8n model (~6 MB) — after that the
> app is fully offline.

## 1. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the app
```bash
python app.py                 # default webcam, voice on
python app.py --camera 1       # use a different webcam
python app.py --no-voice      # visual panel only, no speech
python app.py --conf 0.4       # lower detection confidence threshold
```

A video window opens showing detections. Keyboard: **q** quit,
**d** describe scene aloud, **t** toggle text-reading mode (reads signs/labels
via offline OCR).

## 4. Run the tests (no camera/mic needed)
```bash
python test_assist.py
```

## Troubleshooting

- **"Could not open camera"** — another app may be using the webcam, or try
  `--camera 1` / `--camera 2`.
- **No voice on Linux** — install eSpeak: `sudo apt install espeak` (pyttsx3
  uses it as its speech driver).
- **Slow FPS on old hardware** — the nano model is already the fastest; closing
  other apps helps.
