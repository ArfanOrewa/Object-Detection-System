"""
test_assist.py — Unit tests for the Object Detection System.

No camera or microphone required:
  * voice cooldown logic (with a fake clock — no sleeping)
  * detection filtering / confidence threshold / largest-box selection
    (with a fake YOLO model — no weights downloaded)
  * announcement text formatting

Run:  python test_assist.py
"""

import time
import unittest

import numpy as np

import voice as voice_module
from detect import (Detection, describe_object, describe_scene,
                    distance_description, draw_detections, estimate_distance,
                    format_announcement, horizontal_position,
                    largest_detection, detect_objects, position_phrase)
from voice import VoiceFeedback


# ------------------------------------------------------------- fake YOLO

class _FakeTensor:
    """Mimics a torch tensor just enough for detect_objects()."""
    def __init__(self, values):
        self._v = np.asarray(values, dtype=np.float32)

    def __getitem__(self, idx):
        return _FakeTensor(self._v[idx])

    def tolist(self):
        return self._v.tolist()

    def __int__(self):
        return int(self._v)

    def __float__(self):
        return float(self._v)


class _FakeBox:
    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.xyxy = _FakeTensor([[x1, y1, x2, y2]])
        self.cls = _FakeTensor([cls_id])
        self.conf = _FakeTensor([conf])


class _FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    """Pretends to be an ultralytics YOLO model."""
    def __init__(self, names, specs):
        # specs: list of (x1, y1, x2, y2, cls_id, conf)
        self._names = names
        self._specs = specs

    def predict(self, frame, conf=0.5, verbose=False):
        boxes = [_FakeBox(*s) for s in self._specs if s[5] >= conf]
        return [_FakeResult(self._names, boxes)]


# ----------------------------------------------------------------- tests

class TestVoiceCooldown(unittest.TestCase):
    def _voice(self, cooldown=10.0):
        # enabled=True exercises the real cooldown path; pyttsx3 falls back
        # to console logging when no audio device exists.
        return VoiceFeedback(cooldown=cooldown, enabled=True)

    def test_first_announcement_goes_through(self):
        v = self._voice()
        self.assertTrue(v.announce("chair", "Obstacle: chair, 2 meters, center"))
        v.wait_until_done()

    def test_same_object_within_cooldown_is_skipped(self):
        v = self._voice(cooldown=60.0)
        self.assertTrue(v.announce("chair", "chair 2m"))
        self.assertFalse(v.announce("chair", "chair 2m again"))
        v.wait_until_done()

    def test_different_objects_are_all_announced(self):
        v = self._voice(cooldown=60.0)
        self.assertTrue(v.announce("chair", "chair"))
        self.assertTrue(v.announce("person", "person"))
        self.assertTrue(v.announce("bottle", "bottle"))
        v.wait_until_done()

    def test_cooldown_expires(self):
        v = self._voice(cooldown=0.05)
        self.assertTrue(v.announce("chair", "chair"))
        time.sleep(0.1)
        self.assertTrue(v.announce("chair", "chair again"))
        v.wait_until_done()

    def test_force_bypasses_cooldown(self):
        v = self._voice(cooldown=60.0)
        self.assertTrue(v.announce("person", "person far"))
        # urgent re-announcement goes through despite cooldown
        self.assertTrue(v.announce("person", "person VERY close", force=True))
        v.wait_until_done()

    def test_disabled_voice_announces_nothing(self):
        v = VoiceFeedback(enabled=False)
        self.assertFalse(v.announce("chair", "chair"))
        self.assertEqual(v.pending(), 0)

    def test_mark_visible_clears_cooldown_for_gone_objects(self):
        # Smart logic: an object that leaves and reappears counts as NEW.
        v = self._voice(cooldown=60.0)
        self.assertTrue(v.announce("banana", "banana"))
        self.assertFalse(v.announce("banana", "banana"))   # cooling down
        v.mark_visible(["person"])                          # banana left view
        self.assertTrue(v.announce("banana", "banana"))     # NEW -> announced
        v.wait_until_done()

    def test_mark_visible_keeps_cooldown_for_still_visible(self):
        v = self._voice(cooldown=60.0)
        self.assertTrue(v.announce("banana", "banana"))
        v.mark_visible(["banana", "person"])                # still visible
        self.assertFalse(v.announce("banana", "banana"))    # still cooling
        v.wait_until_done()


class TestDetectionFiltering(unittest.TestCase):
    NAMES = {0: "person", 1: "chair", 2: "bottle"}

    def test_confidence_threshold_filters(self):
        model = _FakeModel(self.NAMES, [
            (10, 10, 50, 50, 0, 0.9),    # kept
            (60, 60, 100, 100, 1, 0.3),   # dropped (< 0.5)
        ])
        dets = detect_objects(model, np.zeros((480, 640, 3), np.uint8),
                              conf_threshold=0.5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "person")

    def test_sorted_by_confidence_descending(self):
        model = _FakeModel(self.NAMES, [
            (10, 10, 50, 50, 0, 0.6),
            (60, 60, 100, 100, 1, 0.95),
        ])
        dets = detect_objects(model, np.zeros((480, 640, 3), np.uint8))
        self.assertEqual([d.label for d in dets], ["chair", "person"])

    def test_largest_detection_selects_biggest_box(self):
        small = Detection("bottle", 0.9, (0, 0, 10, 10))      # area 100
        big = Detection("person", 0.7, (0, 0, 100, 200))      # area 20000
        self.assertIs(largest_detection([small, big]), big)

    def test_largest_detection_empty_returns_none(self):
        self.assertIsNone(largest_detection([]))

    def test_draw_detections_returns_frame_copy(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        dets = [Detection("chair", 0.8, (10, 10, 40, 40))]
        out = draw_detections(frame, dets)
        self.assertEqual(out.shape, frame.shape)
        self.assertIsNot(out, frame)  # must not mutate the input


class TestAnnouncementFormatting(unittest.TestCase):
    def test_format_includes_label_distance_position(self):
        det = Detection("chair", 0.85, (10, 10, 110, 210))  # tall box, left-ish
        text = format_announcement(det, (480, 640))
        self.assertIn("chair", text)
        self.assertIn("meters", text)
        self.assertTrue(text.startswith("Obstacle:"))

    def test_horizontal_position_buckets(self):
        self.assertEqual(horizontal_position(Detection("x", 1, (0, 0, 50, 50)), 600),
                         "left")
        self.assertEqual(horizontal_position(Detection("x", 1, (250, 0, 350, 50)), 600),
                         "center")
        self.assertEqual(horizontal_position(Detection("x", 1, (500, 0, 600, 50)), 600),
                         "right")

    def test_estimate_distance_nearer_when_box_bigger(self):
        shape = (480, 640)
        near = Detection("x", 1, (0, 0, 100, 400))   # huge box
        far = Detection("x", 1, (0, 0, 100, 40))     # tiny box
        self.assertLess(estimate_distance(near, shape),
                        estimate_distance(far, shape))


class TestNaturalAnnouncements(unittest.TestCase):
    def test_position_phrase(self):
        self.assertEqual(position_phrase(Detection("x", 1, (0, 0, 50, 50)), 600),
                         "on your left")
        self.assertEqual(position_phrase(Detection("x", 1, (250, 0, 350, 50)), 600),
                         "in the center")
        self.assertEqual(position_phrase(Detection("x", 1, (500, 0, 600, 50)), 600),
                         "on your right")

    def test_distance_description_tiers(self):
        shape = (480, 640)
        huge = Detection("x", 1, (0, 0, 300, 470))   # fills frame -> close
        tiny = Detection("x", 1, (0, 0, 20, 20))     # speck -> far
        mid = Detection("x", 1, (0, 0, 100, 160))    # mid-size -> ~N meters
        self.assertEqual(distance_description(huge, shape), "close")
        self.assertEqual(distance_description(tiny, shape), "far")
        mid_text = distance_description(mid, shape)
        self.assertTrue(mid_text.startswith("about ") and "meter" in mid_text)

    def test_describe_object_names_object_first(self):
        # Primary interaction: show a banana -> app says "banana".
        det = Detection("banana", 0.9, (400, 100, 550, 400))
        text = describe_object(det, (480, 640))
        self.assertTrue(text.startswith("banana"))
        self.assertIn("on your right", text)

    def test_describe_scene_empty(self):
        self.assertEqual(describe_scene([], (480, 640)),
                         "I don't see anything clearly.")

    def test_describe_scene_single(self):
        det = Detection("banana", 0.9, (400, 100, 550, 400))
        self.assertEqual(describe_scene([det], (480, 640)),
                         "I see: a banana on your right.")

    def test_describe_scene_two(self):
        dets = [Detection("person", 0.9, (0, 0, 120, 400)),
                Detection("chair", 0.8, (250, 0, 350, 200))]
        self.assertEqual(describe_scene(dets, (480, 640)),
                         "I see: a person on your left and a chair in the center.")

    def test_describe_scene_three_uses_oxford_comma(self):
        dets = [Detection("person", 0.9, (0, 0, 120, 400)),
                Detection("chair", 0.8, (250, 0, 350, 200)),
                Detection("apple", 0.7, (450, 0, 550, 150))]
        self.assertEqual(
            describe_scene(dets, (480, 640)),
            "I see: a person on your left, a chair in the center, "
            "and an apple on your right.")

    def test_describe_scene_largest_first(self):
        small = Detection("bottle", 0.95, (500, 0, 560, 60))
        big = Detection("person", 0.6, (0, 0, 200, 450))
        text = describe_scene([small, big], (480, 640))
        self.assertLess(text.index("person"), text.index("bottle"))


class TestOcrReader(unittest.TestCase):
    def test_unavailable_reader_raises_helpful_error(self):
        import ocr as ocr_module
        r = ocr_module.TextReader()
        # Force the "not installed" path without touching the real env.
        r._available = False
        with self.assertRaises(RuntimeError) as ctx:
            r.read_text(__import__("numpy").zeros((50, 50, 3), "uint8"))
        self.assertIn("pip install easyocr", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
