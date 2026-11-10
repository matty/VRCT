"""Shows overlay notifications one after another instead of replacing them.

The small overlay log shows the latest received message for a few seconds.
Each new message used to replace it at once, so in a busy conversation a
message was gone before it could be read.

Messages are queued here and each stays up long enough to be read (by its
length) before the next one takes its place. Results that belong to the same
utterance - transcription sends longer and longer versions while someone is
still talking - share a key and update the notification in place. When more
are waiting than can be read in time, the oldest ones that were never shown
are dropped so the overlay keeps up with the conversation.
"""

import time
from threading import Condition, Thread
from typing import Any, Callable, Hashable, List, Optional

try:
    from utils import errorLogging, printLog
except Exception:  # pragma: no cover
    def errorLogging() -> None:
        import traceback
        print("Error occurred:", traceback.format_exc())

    def printLog(log: str, data=None) -> None:
        print(log, data)

# How long a notification stays before the next may replace it:
# BASE + reading units / speed, between MIN and the overlay's own display
# duration (after which it fades anyway). Same reading model as the chatbox
# player; a wide (CJK) character is two units.
DWELL_BASE_SECONDS = 1.0
DWELL_UNITS_PER_SECOND = 18.0
DWELL_MIN_SECONDS = 2.0
# With this many waiting, read faster to catch up.
BACKLOG_FAST_ITEMS = 2
FAST_DWELL_UNITS_PER_SECOND = 28.0
FAST_DWELL_MIN_SECONDS = 1.5
# More than this waiting: the oldest unread ones are dropped.
MAX_PENDING_ITEMS = 5


class _Item:
    def __init__(self, payload: Any, key: Optional[Hashable], units: float) -> None:
        self.payload = payload
        self.key = key
        self.units = units


class NotificationQueue:
    def __init__(
        self,
        show: Callable[[Any], None],
        max_dwell: Callable[[], float] = lambda: 5.0,
        clock: Callable[[], float] = time.monotonic,
        threaded: bool = True,
    ) -> None:
        """`show(payload)` displays one notification. `max_dwell()` is how
        long the overlay keeps a notification visible at most (its display
        duration setting). With `threaded=False` the caller drives the queue
        with `tick()` (used by the tests)."""
        self._show = show
        self._max_dwell = max_dwell
        self._clock = clock
        self._threaded = threaded
        self._cond = Condition()
        self._thread: Optional[Thread] = None
        self._stopped = False
        self._pending: List[_Item] = []
        self._current: Optional[_Item] = None
        # When the notification on screen was last shown or updated, and how
        # much of it was new then (= what still had to be read).
        self._shown_at = 0.0
        self._unread_units = 0.0
        self.dropped = 0

    def submit(self, payload: Any, key: Optional[Hashable] = None, units: float = 0.0) -> None:
        """Show `payload`, now or when what is on screen has been read.

        `units` is the amount of reading in it (chatbox_player.readingUnits).
        A payload with the key of the notification on screen, or of one that
        is waiting, replaces it.
        """
        with self._cond:
            now = self._clock()
            if key is not None and self._current is not None and self._current.key == key:
                # The same utterance, longer: update in place. Only what was
                # added needs reading.
                self._unread_units = max(0.0, units - self._current.units)
                self._current = _Item(payload, key, units)
                self._shown_at = now
                self._show(payload)
                self._wake()
                return
            waiting = next((item for item in self._pending if key is not None and item.key == key), None)
            if waiting is not None:
                waiting.payload, waiting.units = payload, units
            else:
                self._pending.append(_Item(payload, key, units))
                dropped = 0
                while len(self._pending) > MAX_PENDING_ITEMS:
                    self._pending.pop(0)
                    dropped += 1
                if dropped:
                    self.dropped += dropped
                    printLog(f"[overlay] dropped {dropped} unread notification(s) to keep up with the conversation")
            self._tick(now)
            self._wake()

    def clear(self) -> None:
        with self._cond:
            self._pending.clear()
            self._current = None

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._pending.clear()
            self._cond.notify_all()

    def pending(self) -> int:
        with self._cond:
            return len(self._pending)

    def tick(self, now: Optional[float] = None) -> Optional[float]:
        """Show what is due. Returns the seconds until something else is
        due, or None when nothing is waiting."""
        with self._cond:
            return self._tick(self._clock() if now is None else now)

    # --- internals (call with the lock held) ---------------------------------

    def _dwell(self, units: float) -> float:
        fast = len(self._pending) >= BACKLOG_FAST_ITEMS
        speed = FAST_DWELL_UNITS_PER_SECOND if fast else DWELL_UNITS_PER_SECOND
        low = FAST_DWELL_MIN_SECONDS if fast else DWELL_MIN_SECONDS
        try:
            high = float(self._max_dwell())
        except Exception:
            high = 5.0
        return max(low, min(max(low, high), DWELL_BASE_SECONDS + units / speed))

    def _tick(self, now: float) -> Optional[float]:
        while self._pending:
            if self._current is not None:
                # Evaluated on every tick: the pace picks up as more arrive.
                wait = self._shown_at + self._dwell(self._unread_units) - now
                if wait > 0:
                    return wait
            item = self._pending.pop(0)
            self._current = item
            self._shown_at = now
            self._unread_units = item.units
            self._show(item.payload)
        return None

    def _wake(self) -> None:
        if not self._threaded:
            return
        if self._thread is None or not self._thread.is_alive():
            self._stopped = False
            self._thread = Thread(target=self._run, name="overlay-notifications", daemon=True)
            self._thread.start()
        self._cond.notify_all()

    def _run(self) -> None:
        with self._cond:
            while not self._stopped:
                try:
                    wait = self._tick(self._clock())
                except Exception:
                    errorLogging()
                    self._pending.clear()
                    wait = None
                self._cond.wait(timeout=wait)
