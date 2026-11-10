"""Plays chatbox messages through in order, at a pace they can be read.

The chatbox shows one message of at most 144 characters, and each new
`/chatbox/input` replaces whatever was there. Sending every message the
moment it is produced therefore loses text in a long conversation: a message
is replaced before anyone has read it, and whatever does not fit in 144
characters is never shown at all.

The player sits between the pipeline and OSC:

- Messages are queued and shown one after another. A message stays up long
  enough to be read (by its length) before the next one replaces it.
- A message that does not fit is shown page by page (chatbox_pages.py).
- Transcription keeps sending longer versions of the utterance that is still
  being spoken. They share a `key`: the newer version replaces the older one
  in place. The page on screen grows as long as everything still fits on it;
  once it does not, the next page starts right after what was on screen, so
  nothing is shown twice and nothing is skipped.
- VRChat drops chatbox updates that come too fast, so sends are spaced out.
- When the speaker gets too far ahead of the reader, the oldest unread pages
  are skipped so the chatbox keeps up with the conversation.
"""

import time
import unicodedata
from collections import OrderedDict
from threading import Condition, Thread
from typing import Callable, Hashable, List, Optional, Sequence, Tuple

from .chatbox_pages import buildStreamPages

try:
    from utils import errorLogging, printLog
except Exception:  # pragma: no cover - same fallback as osc.py
    def errorLogging() -> None:
        import traceback
        print("Error occurred:", traceback.format_exc())

    def printLog(log: str, data=None) -> None:
        print(log, data)

# VRChat rate-limits chatbox updates: sent too fast, they are dropped and the
# chatbox is muted for a while. Never send more often than this.
MIN_SEND_INTERVAL_SECONDS = 1.5

# How long a page stays up before the next one may replace it:
# BASE + (what one reader has to read) / speed, between MIN and MAX.
# A page holds the original and its translation(s), and a reader reads one of
# them, so the amount to read is the page divided by the number of texts on
# it. East Asian characters count double.
DWELL_BASE_SECONDS = 1.0
DWELL_UNITS_PER_SECOND = 18.0
DWELL_MIN_SECONDS = 2.5
DWELL_MAX_SECONDS = 7.0
# With this many pages waiting, read faster to catch up.
BACKLOG_FAST_PAGES = 3
FAST_DWELL_UNITS_PER_SECOND = 28.0
FAST_DWELL_MIN_SECONDS = 2.0
FAST_DWELL_MAX_SECONDS = 4.5
# More pages than this waiting behind the one on screen: skip the oldest when
# the page turns. At the fast pace this is about half a minute of reading.
MAX_PENDING_PAGES = 6

_REMEMBERED_KEYS = 64

Render = Callable[[List[str], str], str]


def readingUnits(text: str) -> float:
    """Amount of reading in `text`: a wide (CJK) character counts as two."""
    units = 0.0
    for char in text:
        if char.isspace():
            continue
        units += 2.0 if unicodedata.east_asian_width(char) in ("W", "F") else 1.0
    return units


def _commonPrefixLength(a: str, b: str) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


class _Entry:
    """One message in the queue."""

    def __init__(self, key: Optional[Hashable], texts: Sequence[str], render: Render, notification: bool) -> None:
        self.key = key
        self.render = render
        self.notification = notification
        self.notified = False
        self.texts = [text or "" for text in texts]
        # Progress through the message: characters of each text that were on
        # pages already read, and how many pages that was.
        self.offsets = [0] * len(self.texts)
        self.completed = 0
        self._cache: Optional[List[Tuple[str, List[int]]]] = None

    @property
    def readers(self) -> int:
        return max(1, sum(1 for text in self.texts if text.strip()))

    def update(self, texts: Sequence[str], render: Render) -> None:
        texts = [text or "" for text in texts]
        if len(texts) != len(self.texts):
            # A target language was switched on or off mid-utterance.
            self.offsets = [0] * len(texts)
            self.completed = 0
        self.texts = texts
        self.render = render
        self._cache = None

    def pages(self) -> List[Tuple[str, List[int]]]:
        """Pages from the one that should be on screen now onwards."""
        if self._cache is None:
            # One page more than may wait, so a single long message never
            # counts as a backlog on its own.
            self._cache = buildStreamPages(
                self.texts, self.offsets, self.completed, self.render, max_pages=MAX_PENDING_PAGES + 1
            )
        return self._cache

    def advance(self, consumed: Sequence[int]) -> None:
        """The page on screen has been read."""
        self.completed += 1
        self.offsets = [offset + used for offset, used in zip(self.offsets, consumed)]
        self._cache = None


class ChatboxPlayer:
    def __init__(
        self,
        send: Callable[[str, bool], None],
        clock: Callable[[], float] = time.monotonic,
        threaded: bool = True,
    ) -> None:
        """`send(text, notification)` puts one page in the chatbox.

        With `threaded=False` nothing runs in the background: the caller
        drives the player with `tick()` (used by the tests).
        """
        self._send = send
        self._clock = clock
        self._threaded = threaded
        self._cond = Condition()
        self._thread: Optional[Thread] = None
        self._stopped = False
        self._queue: List[_Entry] = []
        # Where growing messages that have left the queue had got to, so a
        # later version continues instead of starting over.
        self._progress: "OrderedDict[Hashable, Tuple[List[int], int]]" = OrderedDict()
        self._on_screen: Optional[_Entry] = None
        self._screen_text = ""
        self._shown_at = 0.0
        self._new_units = 0.0
        # Characters of each text of the message that the page on screen holds.
        self._screen_consumed: List[int] = []
        self._last_send: Optional[float] = None
        self.skipped_pages = 0

    # --- public -----------------------------------------------------------

    def submit(
        self,
        message: str,
        translation: Sequence[str],
        render: Render,
        key: Optional[Hashable] = None,
        notification: bool = True,
    ) -> None:
        """Queue a message, or replace the queued message with the same key.

        `render(translation, message)` formats a page from (parts of) the
        translations and the original.
        """
        texts = [message] + list(translation)
        with self._cond:
            entry = None
            if key is not None:
                entry = next((queued for queued in self._queue if queued.key == key), None)
            if entry is not None:
                entry.update(texts, render)
            else:
                entry = _Entry(key, texts, render, notification)
                if key is not None and key in self._progress:
                    offsets, completed = self._progress.pop(key)
                    if len(offsets) == len(entry.offsets):
                        entry.offsets, entry.completed = list(offsets), completed
                        # The start of this message already pinged.
                        entry.notified = True
                self._queue.append(entry)
            self._wake()

    def clear(self) -> None:
        """Forget everything that has not been shown yet."""
        with self._cond:
            self._queue.clear()
            self._progress.clear()
            self._on_screen = None
            self._wake()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._queue.clear()
            self._cond.notify_all()

    def pendingPages(self) -> int:
        """Pages waiting behind the one on screen."""
        with self._cond:
            return self._pendingPages()

    def tick(self, now: Optional[float] = None) -> Optional[float]:
        """Send whatever is due. Returns the seconds until something else is
        due, or None when the player is waiting for a new message."""
        with self._cond:
            return self._tick(self._clock() if now is None else now)

    # --- internals (call with the lock held) ---------------------------------

    def _wake(self) -> None:
        if not self._threaded:
            return
        if self._thread is None or not self._thread.is_alive():
            self._stopped = False
            self._thread = Thread(target=self._run, name="chatbox-player", daemon=True)
            self._thread.start()
        self._cond.notify_all()

    def _run(self) -> None:
        with self._cond:
            while not self._stopped:
                try:
                    wait = self._tick(self._clock())
                except Exception:
                    # Never let one bad message stop the chatbox for good.
                    errorLogging()
                    self._queue.clear()
                    self._on_screen = None
                    wait = None
                self._cond.wait(timeout=wait)

    def _pendingPages(self) -> int:
        total = sum(len(entry.pages()) for entry in self._queue)
        if self._on_screen is not None and self._queue and self._queue[0] is self._on_screen:
            total -= 1
        return max(0, total)

    def _dwell(self) -> float:
        fast = self._pendingPages() >= BACKLOG_FAST_PAGES
        speed = FAST_DWELL_UNITS_PER_SECOND if fast else DWELL_UNITS_PER_SECOND
        low = FAST_DWELL_MIN_SECONDS if fast else DWELL_MIN_SECONDS
        high = FAST_DWELL_MAX_SECONDS if fast else DWELL_MAX_SECONDS
        return min(high, max(low, DWELL_BASE_SECONDS + self._new_units / speed))

    def _retire(self, entry: _Entry) -> None:
        self._queue.remove(entry)
        if entry.key is not None:
            self._progress[entry.key] = (list(entry.offsets), entry.completed)
            self._progress.move_to_end(entry.key)
            while len(self._progress) > _REMEMBERED_KEYS:
                self._progress.popitem(last=False)
        if self._on_screen is entry:
            self._on_screen = None

    def _show(self, entry: _Entry, text: str, consumed: Sequence[int], now: float, replaces: str) -> None:
        notification = entry.notification and not entry.notified
        entry.notified = True
        # Only what was added needs reading when the page on screen grew.
        added = text[_commonPrefixLength(text, replaces):] if replaces else text
        self._new_units = readingUnits(added) / entry.readers
        self._on_screen = entry
        self._screen_text = text
        self._screen_consumed = list(consumed)
        self._shown_at = now
        self._last_send = now
        self._send(text, notification)

    def _skipBacklog(self) -> None:
        """Called when the page turns: drop the oldest unread pages while more
        than MAX_PENDING_PAGES are waiting."""
        skipped = 0
        while self._queue and self._pendingPages() > MAX_PENDING_PAGES:
            entry = self._queue[0]
            pages = entry.pages()
            if pages:
                entry.advance(pages[0][1])
            if not entry.pages():
                self._retire(entry)
            skipped += 1
        if skipped:
            self.skipped_pages += skipped
            printLog(f"[chatbox] skipped {skipped} unread page(s) to keep up with the conversation")

    def _tick(self, now: float) -> Optional[float]:
        while True:
            if not self._queue:
                return None
            entry = self._queue[0]
            pages = entry.pages()
            if not pages:
                self._retire(entry)
                continue
            text, consumed = pages[0]
            more_in_entry = len(pages) > 1
            gap = 0.0 if self._last_send is None else self._last_send + MIN_SEND_INTERVAL_SECONDS - now

            if self._on_screen is not entry:
                if gap > 0:
                    return gap
                self._show(entry, text, consumed, now, replaces="")
                continue

            # The page on screen belongs to this message. While everything
            # that is left still fits on it, keep it up to date. Once it does
            # not, leave it alone: the next page will start right after what
            # is on screen now.
            if text != self._screen_text and not more_in_entry:
                if gap > 0:
                    return gap
                self._show(entry, text, consumed, now, replaces=self._screen_text)
                continue

            if not more_in_entry and len(self._queue) == 1:
                # Nothing else to show: the page stays until a message comes.
                return None

            wait = self._shown_at + self._dwell() - now
            if wait > 0:
                return wait
            # Read: turn the page, past what was actually on screen.
            entry.advance(self._screen_consumed)
            self._on_screen = None
            if not entry.pages():
                self._retire(entry)
            self._skipBacklog()
