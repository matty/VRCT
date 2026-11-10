"""ChatboxPlayer: every message is played through, in order, at a readable pace."""

import time
import unittest

from models.osc.chatbox_pages import CHATBOX_MAX_CHARS, buildStreamPages, chatboxLength
from models.osc.chatbox_player import (
    DWELL_MAX_SECONDS,
    DWELL_MIN_SECONDS,
    MAX_PENDING_PAGES,
    MIN_SEND_INTERVAL_SECONDS,
    ChatboxPlayer,
    readingUnits,
)


def _render(translation, message):
    translation_part = "\n".join(translation)
    if translation and message:
        return message + "\n" + translation_part
    return translation_part if translation else message


SENTENCES = [
    "Hello, my name is Matty and I'm from England.",
    "I have been playing VRChat for about three years now.",
    "This world is really beautiful, isn't it?",
    "I come here almost every night.",
    "Sometimes I bring my friends along and we just sit by the lake and talk for hours.",
]
TRANSLATED = [
    "こんにちは、私の名前はマティで、イギリス出身です。",
    "VRChatを始めて約3年になります。",
    "この世界は本当に美しいですね。",
    "ほぼ毎晩ここに来ます。",
    "時々友達を連れてきて、湖のほとりに座って何時間も話します。",
]


def _shown(texts, line):
    """Everything shown on line `line` of the pages (0 = original), markers removed."""
    parts = []
    for text in texts:
        body = text.rsplit(" (", 1)[0] if text.endswith(")") and "/" in text.rsplit(" (", 1)[-1] else text
        lines = body.split("\n")
        if line < len(lines):
            parts.append(lines[line])
    return "".join("".join(parts).split())


class _Harness:
    """A player driven by a fake clock."""

    def __init__(self):
        self.now = 1000.0
        self.sent = []  # (time, text, notification)
        self.player = ChatboxPlayer(self._send, clock=lambda: self.now, threaded=False)

    def _send(self, text, notification):
        self.sent.append((self.now, text, notification))

    def submit(self, message, translation=(), key=None, notification=True):
        self.player.submit(message, list(translation), _render, key=key, notification=notification)
        self.player.tick()

    def run(self, seconds, step=0.1):
        """Advance the clock, ticking like the background thread would."""
        end = self.now + seconds
        while self.now < end:
            self.now = round(self.now + step, 6)
            self.player.tick()

    @property
    def texts(self):
        return [text for _, text, _ in self.sent]

    @property
    def times(self):
        return [t for t, _, _ in self.sent]


class SingleMessageTests(unittest.TestCase):
    def test_message_is_sent_immediately(self):
        h = _Harness()
        h.submit("hello", ["hola"])
        self.assertEqual(h.sent, [(1000.0, "hello\nhola", True)])

    def test_message_that_fits_is_sent_exactly_as_formatted(self):
        h = _Harness()
        h.submit("hello", ["hola"], key=("mic", 1))
        self.assertEqual(h.texts, ["hello\nhola"])

    def test_nothing_more_is_sent_while_idle(self):
        h = _Harness()
        h.submit("hello")
        h.run(60)
        self.assertEqual(len(h.sent), 1)
        self.assertIsNone(h.player.tick())

    def test_empty_message_sends_nothing(self):
        h = _Harness()
        h.submit("   ", key=("mic", 1))
        self.assertEqual(h.sent, [])


class QueueTests(unittest.TestCase):
    def test_a_second_message_waits_until_the_first_was_readable(self):
        h = _Harness()
        h.submit("first message")
        h.run(0.5)
        h.submit("second message")
        self.assertEqual(h.texts, ["first message"])
        h.run(DWELL_MIN_SECONDS)
        self.assertEqual(h.texts, ["first message", "second message"])
        self.assertGreaterEqual(h.times[1] - h.times[0], DWELL_MIN_SECONDS - 1e-6)

    def test_a_message_after_a_pause_is_sent_at_once(self):
        h = _Harness()
        h.submit("first message")
        h.run(30)
        h.submit("second message")
        self.assertEqual(h.texts, ["first message", "second message"])
        self.assertAlmostEqual(h.times[1], h.now)

    def test_messages_are_played_in_order_and_none_is_lost(self):
        h = _Harness()
        for i in range(5):
            h.submit(f"message number {i}")
        h.run(60)
        self.assertEqual(h.texts, [f"message number {i}" for i in range(5)])

    def test_sends_are_never_closer_than_the_rate_limit(self):
        h = _Harness()
        for i in range(6):
            h.submit(f"m{i}", key=("mic", i))
            h.run(0.3)
        h.run(60)
        gaps = [b - a for a, b in zip(h.times, h.times[1:])]
        self.assertTrue(all(gap >= MIN_SEND_INTERVAL_SECONDS - 1e-6 for gap in gaps), gaps)

    def test_longer_pages_stay_up_longer(self):
        short = _Harness()
        short.submit("ok")
        short.submit("next")
        short.run(30)
        long = _Harness()
        long.submit("word " * 28)
        long.submit("next")
        long.run(30)
        self.assertGreater(long.times[1] - long.times[0], short.times[1] - short.times[0])
        self.assertLessEqual(long.times[1] - long.times[0], DWELL_MAX_SECONDS + 0.11)

    def test_wide_characters_count_double(self):
        self.assertEqual(readingUnits("abc def"), 6)
        self.assertEqual(readingUnits("こんにちは"), 10)

    def test_notification_sound_follows_each_message(self):
        h = _Harness()
        h.submit("one", notification=True)
        h.submit("two", notification=False)
        h.run(30)
        self.assertEqual([flag for _, _, flag in h.sent], [True, False])


class LongMessageTests(unittest.TestCase):
    def test_long_message_is_played_page_by_page(self):
        h = _Harness()
        h.submit(" ".join(SENTENCES), ["".join(TRANSLATED)])
        h.run(60)
        self.assertGreater(len(h.sent), 1)
        for text in h.texts:
            self.assertLessEqual(chatboxLength(text), CHATBOX_MAX_CHARS)
        # Nothing is lost: the pages add up to the whole original and translation.
        self.assertEqual(_shown(h.texts, 0), "".join(" ".join(SENTENCES).split()))
        self.assertEqual(_shown(h.texts, 1), "".join(TRANSLATED))
        for index, text in enumerate(h.texts, start=1):
            self.assertTrue(text.endswith(f" ({index}/{len(h.texts)})"), text)

    def test_only_the_first_page_plays_the_notification_sound(self):
        h = _Harness()
        h.submit(" ".join(SENTENCES), ["".join(TRANSLATED)])
        h.run(60)
        self.assertEqual([flag for _, _, flag in h.sent], [True] + [False] * (len(h.sent) - 1))

    def test_next_message_waits_for_every_page(self):
        h = _Harness()
        h.submit(" ".join(SENTENCES), ["".join(TRANSLATED)])
        h.submit("after")
        h.run(60)
        self.assertEqual(h.texts[-1], "after")
        self.assertEqual(_shown(h.texts[:-1], 1), "".join(TRANSLATED))


class GrowingMessageTests(unittest.TestCase):
    """Transcription sends longer and longer versions of the same utterance."""

    def _grow(self, h, key, upto):
        h.submit(" ".join(SENTENCES[:upto]), ["".join(TRANSLATED[:upto])], key=key)

    def test_growing_page_is_updated_in_place(self):
        h = _Harness()
        h.submit("Hello my", ["こんにちは"], key=("mic", 1))
        h.run(3)
        h.submit("Hello my name is Matty", ["こんにちは、マティです"], key=("mic", 1))
        self.assertEqual(h.texts, ["Hello my\nこんにちは", "Hello my name is Matty\nこんにちは、マティです"])
        # the first version pinged; the update of the same utterance does not
        self.assertEqual([flag for _, _, flag in h.sent], [True, False])

    def test_unchanged_version_is_not_sent_again(self):
        h = _Harness()
        h.submit("Hello", key=("mic", 1))
        h.run(3)
        h.submit("Hello", key=("mic", 1))
        h.run(3)
        self.assertEqual(len(h.sent), 1)

    def test_update_respects_the_rate_limit(self):
        h = _Harness()
        h.submit("Hello", key=("mic", 1))
        h.run(0.5)
        h.submit("Hello my name", key=("mic", 1))
        self.assertEqual(len(h.sent), 1)
        h.run(MIN_SEND_INTERVAL_SECONDS)
        self.assertEqual(h.texts[-1], "Hello my name")
        self.assertGreaterEqual(h.times[1] - h.times[0], MIN_SEND_INTERVAL_SECONDS - 1e-6)

    def test_pages_already_read_are_not_shown_again(self):
        h = _Harness()
        for upto in range(1, len(SENTENCES) + 1):
            self._grow(h, ("mic", 1), upto)
            h.run(3)  # a new, longer version every 3 seconds
        h.run(30)

        self.assertGreater(len(h.sent), 2)
        for text in h.texts:
            self.assertLessEqual(chatboxLength(text), CHATBOX_MAX_CHARS)
        # The first sentence only ever appears on the first page: no later
        # send starts the utterance over.
        starts = [text for text in h.texts if text.startswith("Hello, my name is Matty")]
        self.assertTrue(all("(2/" not in text for text in starts))
        first_page_final = starts[-1]
        after = h.texts[h.texts.index(first_page_final) + 1:]
        self.assertTrue(after)
        self.assertTrue(all("Hello, my name is Matty" not in text for text in after), after)
        # ...and everything up to the end of the utterance was shown.
        self.assertIn("talk for hours.", h.texts[-1])
        self.assertIn(TRANSLATED[-1][-8:], h.texts[-1])

    def test_everything_said_in_a_long_monologue_reaches_the_chatbox(self):
        h = _Harness()
        words = [f"w{i:03d}" for i in range(120)]
        # 4 utterances of 30 words; each grows 10 words at a time, 3 s apart.
        for phrase in range(4):
            for upto in (10, 20, 30):
                h.submit(" ".join(words[phrase * 30:phrase * 30 + upto]), key=("mic", phrase))
                h.run(3)
        h.run(60)
        shown = " ".join(h.texts)
        for word in words:
            self.assertIn(word, shown)
        self.assertEqual(h.player.skipped_pages, 0)
        # In order: the first time each word appears follows the previous word's.
        positions = [shown.index(word) for word in words]
        self.assertEqual(positions, sorted(positions))

    def test_later_version_of_a_finished_message_continues_where_it_left_off(self):
        h = _Harness()
        h.submit("one two three", key=("mic", 1))
        h.run(3)
        h.submit("someone else", key=("speaker", 9))  # takes over the chatbox
        h.run(5)
        h.submit("one two three four five", key=("mic", 1))
        h.run(10)
        self.assertEqual(h.texts[0], "one two three")
        self.assertEqual(h.texts[1], "someone else")
        self.assertTrue(h.texts[2].startswith("four five"), h.texts[2])
        # a continuation does not ping again
        self.assertEqual([flag for _, _, flag in h.sent], [True, True, False])

    def test_shrunk_version_does_not_break_the_player(self):
        h = _Harness()
        self._grow(h, ("mic", 1), 5)
        h.run(8)
        h.submit("Hello.", ["こんにちは。"], key=("mic", 1))
        h.submit("next", key=("mic", 2))
        h.run(30)
        self.assertEqual(h.texts[-1], "next")

    def test_changed_number_of_translations_starts_the_message_over(self):
        h = _Harness()
        h.submit("hello", ["hola"], key=("mic", 1))
        h.run(3)
        h.submit("hello there", ["hola", "bonjour"], key=("mic", 1))
        h.run(3)
        self.assertEqual(h.texts[-1], "hello there\nhola\nbonjour")


class BacklogTests(unittest.TestCase):
    def test_pages_are_read_faster_when_several_are_waiting(self):
        relaxed = _Harness()
        relaxed.submit("word " * 28)
        relaxed.submit("next")
        relaxed.run(30)
        busy = _Harness()
        busy.submit("word " * 28)
        for i in range(4):
            busy.submit(f"waiting {i}")
        busy.run(30)
        self.assertLess(busy.times[1] - busy.times[0], relaxed.times[1] - relaxed.times[0])

    def test_oldest_unread_pages_are_skipped_beyond_the_cutoff(self):
        h = _Harness()
        total = MAX_PENDING_PAGES + 10
        for i in range(total):
            h.submit(f"message {i:02d}")
        h.run(120)
        shown = h.texts
        self.assertEqual(shown[0], "message 00")  # already on screen: never skipped
        self.assertEqual(shown[-1], f"message {total - 1:02d}")  # the newest always arrives
        self.assertLess(len(shown), total)
        self.assertGreater(h.player.skipped_pages, 0)
        self.assertEqual(len(shown) + h.player.skipped_pages, total)
        # what is shown stays in order
        self.assertEqual(shown, sorted(shown))

    def test_no_skipping_at_the_cutoff(self):
        h = _Harness()
        for i in range(MAX_PENDING_PAGES + 1):
            h.submit(f"message {i:02d}")
        h.run(120)
        self.assertEqual(h.player.skipped_pages, 0)
        self.assertEqual(len(h.sent), MAX_PENDING_PAGES + 1)

    def test_clear_forgets_what_was_not_shown(self):
        h = _Harness()
        h.submit("one")
        h.submit("two")
        h.player.clear()
        h.run(30)
        self.assertEqual(h.texts, ["one"])


class StreamPagesTests(unittest.TestCase):
    def test_full_pages_do_not_change_when_the_text_grows(self):
        first_pages = None
        for upto in range(3, len(SENTENCES) + 1):
            texts = [" ".join(SENTENCES[:upto]), "".join(TRANSLATED[:upto])]
            pages = buildStreamPages(texts, [0, 0], 0, _render)
            body = pages[0][0].rsplit(" (", 1)[0]
            if first_pages is None:
                first_pages = body
            self.assertEqual(body, first_pages)

    def test_consumed_counts_cover_the_whole_text(self):
        texts = [" ".join(SENTENCES), "".join(TRANSLATED)]
        pages = buildStreamPages(texts, [0, 0], 0, _render)
        self.assertEqual([sum(c[i] for _, c in pages) for i in range(2)], [len(texts[0]), len(texts[1])])

    def test_resuming_from_offsets_gives_the_following_pages(self):
        texts = [" ".join(SENTENCES), "".join(TRANSLATED)]
        pages = buildStreamPages(texts, [0, 0], 0, _render)
        rest = buildStreamPages(texts, pages[0][1], 1, _render)
        self.assertEqual([page for page, _ in rest], [page for page, _ in pages[1:]])

    def test_every_page_fits(self):
        texts = ["word " * 200, "語" * 300, "단어 " * 150]
        for page, _ in buildStreamPages(texts, [0, 0, 0], 0, _render, max_pages=50):
            self.assertLessEqual(chatboxLength(page), CHATBOX_MAX_CHARS)

    def test_nothing_left(self):
        self.assertEqual(buildStreamPages(["abc"], [3], 1, _render), [])


class ThreadedPlayerTests(unittest.TestCase):
    def test_background_thread_plays_the_queue(self):
        sent = []
        player = ChatboxPlayer(lambda text, notification: sent.append(text))
        try:
            player.submit("hello", [], _render)
            deadline = time.monotonic() + 2
            while not sent and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(sent, ["hello"])
        finally:
            player.stop()

    def test_submit_does_not_block(self):
        player = ChatboxPlayer(lambda text, notification: None)
        try:
            started = time.monotonic()
            for i in range(20):
                player.submit(f"message {i}", [], _render)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            player.stop()

    def test_a_failing_send_does_not_kill_the_player(self):
        calls = []

        def send(text, notification):
            calls.append(text)
            if len(calls) == 1:
                raise OSError("network unreachable")

        from unittest.mock import patch
        with patch("models.osc.chatbox_player.errorLogging"):
            player = ChatboxPlayer(send)
            try:
                player.submit("first", [], _render)
                deadline = time.monotonic() + 2
                while not calls and time.monotonic() < deadline:
                    time.sleep(0.01)
                time.sleep(0.05)
                player.submit("second", [], _render)
                deadline = time.monotonic() + 4
                while len(calls) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(calls, ["first", "second"])
            finally:
                player.stop()


if __name__ == "__main__":
    unittest.main()
