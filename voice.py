"""
voice.py — Offline voice feedback for the assistive Object Detection System.

Uses pyttsx3 (fully offline text-to-speech, no API key, no internet).
Announcements run on a background thread so detection never blocks on speech.

Key behaviours (all unit-tested in test_assist.py):
  * Per-object cooldown: the same object label is not re-announced within
    `cooldown` seconds (default 10).
  * Priority: the caller passes the most important detection first
    (app.py sends the largest/closest one); `force=True` bypasses cooldown
    for urgent warnings.
  * Graceful degradation: if pyttsx3 cannot initialise (e.g. no audio device
    on a headless machine), announcements fall back to console logging instead
    of crashing.
"""

import queue
import threading
import time
from typing import Dict, Optional


class VoiceFeedback:
    def __init__(self, cooldown: float = 10.0, enabled: bool = True,
                 rate: int = 160):
        """
        Args:
            cooldown: seconds before the same object label may be announced again.
            enabled:  False disables speech entirely (log-only mode).
            rate:     speech rate (words per minute) for the TTS engine.
        """
        self.cooldown = cooldown
        self.enabled = enabled
        self._last_spoken: Dict[str, float] = {}
        self._engine = None
        self._audio_ok = False

        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self._audio_ok = True
            except Exception as exc:  # no audio device, missing driver, etc.
                print(f"[voice] Audio unavailable ({exc}); "
                      "falling back to console log.")
                self._engine = None
                self._audio_ok = False

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._speak_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ internals

    def _speak_loop(self) -> None:
        """Background thread: speak queued messages one at a time."""
        while True:
            text = self._queue.get()
            try:
                if self._audio_ok and self._engine is not None:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    print(f"[voice] {text}")
            except Exception as exc:
                # Never let a speech failure kill detection.
                print(f"[voice] speech failed ({exc}); message was: {text}")
            finally:
                self._queue.task_done()

    def _cooling_down(self, label: str, now: Optional[float] = None) -> bool:
        """True if `label` was announced within the cooldown window."""
        now = time.time() if now is None else now
        last = self._last_spoken.get(label)
        return last is not None and (now - last) < self.cooldown

    def mark_visible(self, labels) -> None:
        """
        Tell the voice system which object labels are currently visible.

        Labels that disappear from view have their cooldown timestamp cleared,
        so when they reappear they count as NEW and are announced immediately
        instead of being suppressed by an old cooldown.
        """
        visible = set(labels)
        for label in list(self._last_spoken):
            if label not in visible:
                del self._last_spoken[label]

    # --------------------------------------------------------------- public

    def announce(self, label: str, text: str, force: bool = False) -> bool:
        """
        Queue an announcement for `label` (e.g. "chair") with spoken `text`.

        Returns True if the announcement was queued, False if it was skipped
        (cooldown active, or speech disabled). `force=True` bypasses cooldown
        for urgent warnings (e.g. obstacle closer than 1 metre).
        """
        if not self.enabled:
            return False
        if not force and self._cooling_down(label):
            return False
        self._last_spoken[label] = time.time()
        self._queue.put(text)
        return True

    def pending(self) -> int:
        """Number of announcements waiting in the speech queue."""
        return self._queue.qsize()

    def wait_until_done(self, timeout: float = 30.0) -> None:
        """Block until the speech queue is empty (useful in tests/demos)."""
        self._queue.join()


if __name__ == "__main__":
    v = VoiceFeedback(cooldown=5.0)
    v.announce("person", "Obstacle: person, 1.5 meters, center")
    v.announce("person", "Obstacle: person, 1.5 meters, center")  # skipped
    v.announce("chair", "Obstacle: chair, 2 meters, left")
    v.wait_until_done()
    print("demo done")
