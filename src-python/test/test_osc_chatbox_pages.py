"""OSCHandler: sending a chatbox message as pages, one after another."""

import threading
import time
import unittest
from unittest.mock import MagicMock

from models.osc.osc import CHATBOX_PAGE_INTERVAL_SECONDS, OSCHandler


def _handler():
    handler = OSCHandler(ip_address="127.0.0.1", port=9000)
    handler.udp_client = MagicMock()
    return handler


def _sent(handler):
    return [call.args[1] for call in handler.udp_client.send_message.call_args_list]


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class SendMessageTests(unittest.TestCase):
    def test_single_message_is_sent_as_before(self):
        handler = _handler()
        handler.sendMessage("hello", notification=False)
        handler.udp_client.send_message.assert_called_once_with("/chatbox/input", ["hello", True, False])

    def test_empty_message_is_not_sent(self):
        handler = _handler()
        handler.sendMessage("")
        handler.udp_client.send_message.assert_not_called()

    def test_page_interval_stays_clear_of_vrchats_rate_limit(self):
        self.assertGreaterEqual(CHATBOX_PAGE_INTERVAL_SECONDS, 2.0)


class SendMessagePagesTests(unittest.TestCase):
    def test_first_page_is_sent_immediately(self):
        handler = _handler()
        handler.sendMessagePages(["one", "two"], notification=True, interval=30)
        self.assertEqual(_sent(handler), [["one", True, True]])
        handler.cancelPendingPages()

    def test_following_pages_are_sent_in_order_without_the_notification_sound(self):
        handler = _handler()
        handler.sendMessagePages(["one", "two", "three"], notification=True, interval=0.02)
        self.assertTrue(_wait_for(lambda: len(_sent(handler)) == 3))
        self.assertEqual(
            _sent(handler),
            [["one", True, True], ["two", True, False], ["three", True, False]],
        )

    def test_pages_wait_for_the_interval(self):
        handler = _handler()
        handler.sendMessagePages(["one", "two"], interval=0.2)
        time.sleep(0.05)
        self.assertEqual(len(_sent(handler)), 1)
        self.assertTrue(_wait_for(lambda: len(_sent(handler)) == 2))

    def test_a_new_message_drops_the_pages_still_waiting(self):
        handler = _handler()
        handler.sendMessagePages(["one", "two", "three"], interval=0.1)
        handler.sendMessage("new")
        time.sleep(0.3)
        self.assertEqual([args[0] for args in _sent(handler)], ["one", "new"])

    def test_a_new_paged_message_replaces_the_old_one(self):
        handler = _handler()
        handler.sendMessagePages(["a1", "a2", "a3"], interval=0.1)
        handler.sendMessagePages(["b1", "b2"], interval=0.02)
        self.assertTrue(_wait_for(lambda: len(_sent(handler)) == 3))
        time.sleep(0.25)
        self.assertEqual([args[0] for args in _sent(handler)], ["a1", "b1", "b2"])

    def test_cancel_drops_the_pages_still_waiting(self):
        handler = _handler()
        handler.sendMessagePages(["one", "two"], interval=0.05)
        handler.cancelPendingPages()
        time.sleep(0.15)
        self.assertEqual([args[0] for args in _sent(handler)], ["one"])

    def test_empty_pages_are_skipped(self):
        handler = _handler()
        handler.sendMessagePages(["", "one", ""], interval=0.02)
        time.sleep(0.1)
        self.assertEqual([args[0] for args in _sent(handler)], ["one"])

    def test_no_pages_sends_nothing(self):
        handler = _handler()
        handler.sendMessagePages([])
        handler.udp_client.send_message.assert_not_called()

    def test_the_caller_is_not_blocked(self):
        handler = _handler()
        started = time.monotonic()
        handler.sendMessagePages(["one", "two", "three"], interval=30)
        self.assertLess(time.monotonic() - started, 0.5)
        handler.cancelPendingPages()

    def test_pages_follow_a_changed_osc_target(self):
        # setOscIpAddress/setOscPort replace udp_client: later pages must go
        # to the new target, not the client captured when sending started.
        handler = _handler()
        handler.sendMessagePages(["one", "two"], interval=0.05)
        new_client = MagicMock()
        handler.udp_client = new_client
        self.assertTrue(_wait_for(lambda: new_client.send_message.call_count == 1))
        new_client.send_message.assert_called_once_with("/chatbox/input", ["two", True, False])

    def test_page_threads_end(self):
        handler = _handler()
        before = threading.active_count()
        handler.sendMessagePages(["one", "two"], interval=0.01)
        self.assertTrue(_wait_for(lambda: len(_sent(handler)) == 2))
        self.assertTrue(_wait_for(lambda: threading.active_count() <= before))


if __name__ == "__main__":
    unittest.main()
