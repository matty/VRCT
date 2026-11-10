"""OSCHandler: chatbox messages go through the player; direct sends still work."""

import time
import unittest
from unittest.mock import MagicMock

from models.osc.osc import OSCHandler


def _render(translation, message):
    translation_part = "\n".join(translation)
    if translation and message:
        return message + "\n" + translation_part
    return translation_part if translation else message


def _handler():
    handler = OSCHandler(ip_address="127.0.0.1", port=9000)
    handler.udp_client = MagicMock()
    return handler


def _sent(client):
    return [call.args[1] for call in client.send_message.call_args_list]


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class SendMessageTests(unittest.TestCase):
    def test_direct_send_goes_out_immediately(self):
        handler = _handler()
        handler.sendMessage("hello", notification=False)
        handler.udp_client.send_message.assert_called_once_with("/chatbox/input", ["hello", True, False])

    def test_empty_message_is_not_sent(self):
        handler = _handler()
        handler.sendMessage("")
        handler.udp_client.send_message.assert_not_called()

    def test_direct_send_drops_what_the_player_had_queued(self):
        handler = _handler()
        try:
            handler.playMessage("first", [], _render)
            handler.playMessage("second", [], _render)
            self.assertTrue(_wait_for(lambda: len(_sent(handler.udp_client)) == 1))
            handler.sendMessage("direct")
            time.sleep(0.2)
            self.assertEqual([args[0] for args in _sent(handler.udp_client)], ["first", "direct"])
            self.assertEqual(handler.chatbox_player.pendingPages(), 0)
        finally:
            handler.chatbox_player.stop()


class PlayMessageTests(unittest.TestCase):
    def test_played_message_reaches_the_chatbox(self):
        handler = _handler()
        try:
            handler.playMessage("hello", ["hola"], _render, key=("mic", 1), notification=True)
            self.assertTrue(_wait_for(lambda: len(_sent(handler.udp_client)) == 1))
            self.assertEqual(_sent(handler.udp_client), [["hello\nhola", True, True]])
        finally:
            handler.chatbox_player.stop()

    def test_the_caller_is_not_blocked(self):
        handler = _handler()
        try:
            started = time.monotonic()
            for i in range(10):
                handler.playMessage(f"message {i}", [], _render)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            handler.chatbox_player.stop()

    def test_clear_drops_the_messages_still_waiting(self):
        handler = _handler()
        try:
            handler.playMessage("one", [], _render)
            handler.playMessage("two", [], _render)
            self.assertTrue(_wait_for(lambda: len(_sent(handler.udp_client)) == 1))
            handler.clearMessageQueue()
            self.assertEqual(handler.chatbox_player.pendingPages(), 0)
        finally:
            handler.chatbox_player.stop()

    def test_pages_follow_a_changed_osc_target(self):
        # setOscIpAddress/setOscPort replace udp_client: what is still queued
        # must go to the new target, not the client in use when it was queued.
        handler = _handler()
        try:
            old_client = handler.udp_client
            handler.playMessage("one", [], _render)
            handler.playMessage("two", [], _render)
            self.assertTrue(_wait_for(lambda: old_client.send_message.call_count == 1))
            new_client = MagicMock()
            handler.udp_client = new_client
            self.assertTrue(_wait_for(lambda: new_client.send_message.call_count == 1, timeout=6.0))
            new_client.send_message.assert_called_once_with("/chatbox/input", ["two", True, True])
            self.assertEqual(old_client.send_message.call_count, 1)
        finally:
            handler.chatbox_player.stop()


if __name__ == "__main__":
    unittest.main()
