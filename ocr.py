"""
ocr.py — Text-reading mode for the assistive Object Detection System.

Uses EasyOCR (pip-installable, fully offline after its models download once —
no API key, no internet needed at runtime). Reads signs, labels, and book
pages out loud when the user toggles text mode with the 't' key.

Graceful degradation: if EasyOCR is not installed, the reader reports itself
as unavailable instead of crashing, and the app tells the user why.
"""

from typing import List

import numpy as np


class TextReader:
    """Offline OCR reader with lazy model loading."""

    def __init__(self):
        self._reader = None
        self._available: bool | None = None  # None = not checked yet

    @property
    def available(self) -> bool:
        """True if EasyOCR imported successfully."""
        if self._available is None:
            try:
                import easyocr  # noqa: F401
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr
            # Downloads detection/recognition models on first use (~100MB),
            # then works fully offline. CPU mode for low-cost hardware.
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def read_text(self, frame: np.ndarray,
                  min_confidence: float = 0.4) -> str:
        """
        Extract readable text from a BGR frame.

        Returns the detected words joined into a single string ("" if none).
        Raises RuntimeError if EasyOCR is not installed.
        """
        if not self.available:
            raise RuntimeError(
                "EasyOCR is not installed. Run: pip install easyocr")
        reader = self._ensure_reader()
        # detail=1 -> [(bbox, text, confidence), ...]
        results = reader.readtext(frame, detail=1)
        words: List[str] = [text.strip() for _, text, conf in results
                            if conf >= min_confidence and text.strip()]
        return " ".join(words)

    def describe_text(self, frame: np.ndarray) -> str:
        """Spoken form: 'Text reads: ...' or a nothing-found message."""
        text = self.read_text(frame)
        if not text:
            return "No readable text in view."
        return f"Text reads: {text}"


if __name__ == "__main__":
    # Demo on a synthetic image with known text (no camera needed).
    import cv2
    img = np.full((200, 600, 3), 255, np.uint8)
    cv2.putText(img, "HELLO WORLD", (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    r = TextReader()
    print("available:", r.available)
    if r.available:
        print(r.describe_text(img))
