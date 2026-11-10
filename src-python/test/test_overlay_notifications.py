"""NotificationQueue: overlay notifications are shown one after another."""

import time
import unittest

from models.overlay.overlay_notifications import (
    DWELL_MIN_SECONDS,
    MAX_PENDING_ITEMS,
    NotificationQueue,
)


class _Harness:
    def __init__(self, max_dwell=5.0):
        self.now = 100.0
        self.shown = []  # (time, payload)
        self.queue = NotificationQueue(
            lambda payload: self.shown.append((self.now, payload)),
            max_dwell=lambda: max_dwell,
            clock=lambda: self.now,
            threaded=False,
        )

    def submit(self, payload, key=None, units=20.0):
        self.queue.submit(payload, key=key, units=units)

    def run(self, seconds, step=0.1):
        end = self.now + seconds
        while self.now < end:
            self.now = round(self.now + step, 6)
            self.queue.tick()

    @property
    def payloads(self):
        return [payload for _, payload in self.shown]

    @property
    def times(self):
        return [t for t, _ in self.shown]


class NotificationQueueTests(unittest.TestCase):
    def test_first_notification_is_shown_immediately(self):
        h = _Harness()
        h.submit("a")
        self.assertEqual(h.shown, [(100.0, "a")])

    def test_next_notification_waits_until_the_first_was_readable(self):
        h = _Harness()
        h.submit("a")
        h.run(0.5)
        h.submit("b")
        self.assertEqual(h.payloads, ["a"])
        h.run(5)
        self.assertEqual(h.payloads, ["a", "b"])
        self.assertGreaterEqual(h.times[1] - h.times[0], DWELL_MIN_SECONDS - 1e-6)

    def test_notification_after_a_pause_is_shown_at_once(self):
        h = _Harness()
        h.submit("a")
        h.run(20)
        h.submit("b")
        self.assertEqual(h.times[1], h.now)

    def test_all_are_shown_in_order(self):
        h = _Harness()
        for name in "abcde":
            h.submit(name)
        h.run(60)
        self.assertEqual(h.payloads, list("abcde"))

    def test_longer_text_stays_longer(self):
        short = _Harness()
        short.submit("a", units=5)
        short.submit("b")
        short.run(30)
        long = _Harness()
        long.submit("a", units=60)
        long.submit("b")
        long.run(30)
        self.assertGreater(long.times[1] - long.times[0], short.times[1] - short.times[0])

    def test_never_waits_longer_than_the_overlay_keeps_it_visible(self):
        h = _Harness(max_dwell=3.0)
        h.submit("a", units=500)
        h.submit("b")
        h.run(30)
        self.assertLessEqual(h.times[1] - h.times[0], 3.0 + 0.11)

    def test_same_utterance_updates_in_place(self):
        h = _Harness()
        h.submit("hello", key=("speaker", 1), units=5)
        h.run(1)
        h.submit("hello there", key=("speaker", 1), units=10)
        self.assertEqual(h.payloads, ["hello", "hello there"])
        self.assertEqual(h.times[1], h.now)  # no waiting: it is the same notification

    def test_update_of_a_waiting_utterance_replaces_it(self):
        h = _Harness()
        h.submit("first")
        h.submit("draft", key=("speaker", 2))
        h.submit("draft, longer", key=("speaker", 2))
        h.run(30)
        self.assertEqual(h.payloads, ["first", "draft, longer"])

    def test_growing_notification_does_not_hold_up_the_next_one_forever(self):
        h = _Harness()
        h.submit("v1", key=("speaker", 1), units=10)
        h.submit("other", key=("speaker", 2))
        for i in range(2, 8):
            h.run(1)
            h.submit(f"v{i}", key=("speaker", 1), units=10 * i)
        h.run(10)
        self.assertIn("other", h.payloads)

    def test_pace_picks_up_when_several_are_waiting(self):
        relaxed = _Harness()
        relaxed.submit("a", units=60)
        relaxed.submit("b")
        relaxed.run(30)
        busy = _Harness()
        busy.submit("a", units=60)
        for name in "bcd":
            busy.submit(name)
        busy.run(30)
        self.assertLess(busy.times[1] - busy.times[0], relaxed.times[1] - relaxed.times[0])

    def test_oldest_unread_are_dropped_beyond_the_cutoff(self):
        h = _Harness()
        total = MAX_PENDING_ITEMS + 6
        for i in range(total):
            h.submit(f"n{i:02d}")
        h.run(120)
        self.assertEqual(h.payloads[0], "n00")
        self.assertEqual(h.payloads[-1], f"n{total - 1:02d}")
        self.assertEqual(len(h.payloads) + h.queue.dropped, total)
        self.assertGreater(h.queue.dropped, 0)
        self.assertEqual(h.payloads, sorted(h.payloads))

    def test_clear(self):
        h = _Harness()
        h.submit("a")
        h.submit("b")
        h.queue.clear()
        h.run(30)
        self.assertEqual(h.payloads, ["a"])
        # and the next one is shown at once
        h.submit("c")
        self.assertEqual(h.payloads, ["a", "c"])

    def test_background_thread_shows_waiting_notifications(self):
        shown = []
        queue = NotificationQueue(shown.append, max_dwell=lambda: 0.05)
        try:
            import models.overlay.overlay_notifications as module
            original = (module.DWELL_MIN_SECONDS, module.FAST_DWELL_MIN_SECONDS)
            module.DWELL_MIN_SECONDS = module.FAST_DWELL_MIN_SECONDS = 0.05
            try:
                queue.submit("a")
                queue.submit("b")
                deadline = time.monotonic() + 3
                while len(shown) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                module.DWELL_MIN_SECONDS, module.FAST_DWELL_MIN_SECONDS = original
            self.assertEqual(shown, ["a", "b"])
        finally:
            queue.stop()


if __name__ == "__main__":
    unittest.main()
