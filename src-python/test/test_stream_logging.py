"""One log entry per utterance, not one per interim result."""

import unittest
from unittest.mock import MagicMock

from model import Model
from models.overlay.overlay_image import OverlayImage


def _model():
    model = object.__new__(Model)
    model._inited = True
    model.logger = MagicMock()
    model._pending_log_lines = {}
    return model


def _lines(model):
    return [call.args[0] for call in model.logger.info.call_args_list]


class LogFileTests(unittest.TestCase):
    def test_interim_lines_are_not_written(self):
        model = _model()
        model.logMessage("[SENT] Hello", key=("mic", 1), final=False)
        model.logMessage("[SENT] Hello my name is", key=("mic", 1), final=False)
        self.assertEqual(_lines(model), [])

    def test_final_line_is_written_once(self):
        model = _model()
        model.logMessage("[SENT] Hello", key=("mic", 1), final=False)
        model.logMessage("[SENT] Hello my name is Matty", key=("mic", 1), final=True)
        self.assertEqual(_lines(model), ["[SENT] Hello my name is Matty"])
        model.logUtteranceEnd(("mic", 1))
        self.assertEqual(len(_lines(model)), 1)

    def test_end_of_utterance_writes_the_latest_interim_line(self):
        model = _model()
        model.logMessage("[SENT] Hello", key=("mic", 1), final=False)
        model.logMessage("[SENT] Hello my name is", key=("mic", 1), final=False)
        model.logUtteranceEnd(("mic", 1))
        self.assertEqual(_lines(model), ["[SENT] Hello my name is"])

    def test_utterances_are_kept_apart(self):
        model = _model()
        model.logMessage("[SENT] mine", key=("mic", 1), final=False)
        model.logMessage("[RECEIVED] theirs", key=("speaker", 1), final=False)
        model.logUtteranceEnd(("speaker", 1))
        self.assertEqual(_lines(model), ["[RECEIVED] theirs"])
        model.logUtteranceEnd(("mic", 1))
        self.assertEqual(_lines(model), ["[RECEIVED] theirs", "[SENT] mine"])

    def test_line_without_a_key_is_written_at_once(self):
        model = _model()
        model.logMessage("[CHAT] hi")
        self.assertEqual(_lines(model), ["[CHAT] hi"])

    def test_flush_writes_everything_that_is_waiting(self):
        model = _model()
        model.logMessage("[SENT] a", key=("mic", 1), final=False)
        model.logMessage("[RECEIVED] b", key=("speaker", 2), final=False)
        model.flushLogMessages()
        self.assertEqual(_lines(model), ["[SENT] a", "[RECEIVED] b"])
        model.flushLogMessages()
        self.assertEqual(len(_lines(model)), 2)

    def test_stopping_the_logger_writes_what_is_waiting_first(self):
        model = _model()
        logger = model.logger
        model.logMessage("[SENT] a", key=("mic", 1), final=False)
        model.stopLogger()
        logger.info.assert_called_once_with("[SENT] a")
        self.assertIsNone(model.logger)

    def test_end_of_unknown_utterance_is_ignored(self):
        model = _model()
        model.logUtteranceEnd(("mic", 99))
        self.assertEqual(_lines(model), [])

    def test_no_logger(self):
        model = _model()
        model.logger = None
        model.logMessage("[SENT] a", key=("mic", 1), final=True)  # must not raise


class LargeOverlayLogTests(unittest.TestCase):
    def _log(self, overlay_image, message, entry_id=None):
        overlay_image.createOverlayImageLargeLog(
            "send", message, "English", ["訳"], ["Japanese"], [], [[]], entry_id=entry_id,
        )

    def test_same_utterance_replaces_its_line(self):
        overlay_image = OverlayImage()
        self._log(overlay_image, "Hello", entry_id=("mic", 1))
        self._log(overlay_image, "Hello my name is", entry_id=("mic", 1))
        self._log(overlay_image, "Hello my name is Matty", entry_id=("mic", 1))
        self.assertEqual([log["message"] for log in overlay_image.message_log], ["Hello my name is Matty"])

    def test_earlier_conversation_is_not_pushed_out_by_drafts(self):
        overlay_image = OverlayImage()
        for i in range(4):
            self._log(overlay_image, f"earlier {i}", entry_id=("speaker", i))
        for text in ("I", "I am", "I am from", "I am from England", "I am from England and you?"):
            self._log(overlay_image, text, entry_id=("mic", 9))
        self.assertEqual(
            [log["message"] for log in overlay_image.message_log],
            ["earlier 0", "earlier 1", "earlier 2", "earlier 3", "I am from England and you?"],
        )

    def test_replaced_line_keeps_its_place_and_time(self):
        overlay_image = OverlayImage()
        self._log(overlay_image, "mine", entry_id=("mic", 1))
        overlay_image.message_log[0]["datetime"] = "12:34"
        self._log(overlay_image, "theirs", entry_id=("speaker", 1))
        self._log(overlay_image, "mine, longer", entry_id=("mic", 1))
        self.assertEqual([log["message"] for log in overlay_image.message_log], ["mine, longer", "theirs"])
        self.assertEqual(overlay_image.message_log[0]["datetime"], "12:34")

    def test_without_an_id_every_message_is_a_new_line(self):
        overlay_image = OverlayImage()
        self._log(overlay_image, "ok")
        self._log(overlay_image, "ok")
        self.assertEqual(len(overlay_image.message_log), 2)

    def test_log_is_still_limited_to_five_lines(self):
        overlay_image = OverlayImage()
        for i in range(8):
            self._log(overlay_image, f"m{i}", entry_id=("mic", i))
        self.assertEqual([log["message"] for log in overlay_image.message_log], [f"m{i}" for i in range(3, 8)])


if __name__ == "__main__":
    unittest.main()
