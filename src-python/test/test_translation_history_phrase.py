"""One utterance = one entry in the translation context.

Transcription keeps re-sending the growing buffer of an utterance, so one
utterance produces several results ("Hello" -> "Hello my name is" ->
"Hello my name is Matty"). Each used to be appended to the translation
history. LLM engines and DeepL only read the 5 most recent entries, so a
single long utterance filled the context with its own drafts and pushed out
what the other person had said.
"""

import unittest
from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import MagicMock, patch

from model import Model
from models.transcription.transcription_transcriber import AudioTranscriber


class FakeAudioSource:
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    channels = 1


def _model():
    # object.__new__ bypasses the singleton so the shared `model` is untouched.
    model = object.__new__(Model)
    model._inited = True
    model.translation_history = []
    model.translation_history_max_items = 20
    return model


def _texts(model):
    return [(item["source"], item["text"]) for item in model.translation_history]


class TranslationHistoryPhraseTests(unittest.TestCase):
    def test_results_of_the_same_utterance_replace_each_other(self):
        model = _model()
        model.addTranslationHistory("mic", "Hello", phrase_id=7)
        model.addTranslationHistory("mic", "Hello my name is", phrase_id=7)
        model.addTranslationHistory("mic", "Hello my name is Matty", phrase_id=7)
        self.assertEqual(_texts(model), [("mic", "Hello my name is Matty")])

    def test_a_new_utterance_is_a_new_entry(self):
        model = _model()
        model.addTranslationHistory("mic", "Hello", phrase_id=7)
        model.addTranslationHistory("mic", "How are you", phrase_id=8)
        self.assertEqual(_texts(model), [("mic", "Hello"), ("mic", "How are you")])

    def test_the_other_persons_message_is_not_pushed_out(self):
        model = _model()
        model.addTranslationHistory("speaker", "Where are you from?", phrase_id=1)
        for text in ("I", "I am from", "I am from England", "I am from England and", "I am from England and you?"):
            model.addTranslationHistory("mic", text, phrase_id=2)
        recent = model.getTranslationHistory(max_items=5)
        self.assertEqual(
            [(item["source"], item["text"]) for item in recent],
            [("speaker", "Where are you from?"), ("mic", "I am from England and you?")],
        )

    def test_entry_keeps_its_place_when_another_source_spoke_meanwhile(self):
        model = _model()
        model.addTranslationHistory("mic", "So I was", phrase_id=3)
        model.addTranslationHistory("speaker", "Yeah", phrase_id=4)
        model.addTranslationHistory("mic", "So I was thinking", phrase_id=3)
        self.assertEqual(_texts(model), [("mic", "So I was thinking"), ("speaker", "Yeah")])

    def test_same_number_from_another_source_is_a_different_utterance(self):
        model = _model()
        model.addTranslationHistory("mic", "Hello", phrase_id=5)
        model.addTranslationHistory("speaker", "Hi", phrase_id=5)
        self.assertEqual(_texts(model), [("mic", "Hello"), ("speaker", "Hi")])

    def test_without_a_phrase_id_every_message_is_appended(self):
        # Typed chat and OCR have no phrase id: identical messages are separate.
        model = _model()
        model.addTranslationHistory("chat", "ok")
        model.addTranslationHistory("chat", "ok")
        self.assertEqual(_texts(model), [("chat", "ok"), ("chat", "ok")])

    def test_empty_text_is_ignored(self):
        model = _model()
        model.addTranslationHistory("mic", "Hello", phrase_id=1)
        model.addTranslationHistory("mic", "   ", phrase_id=1)
        self.assertEqual(_texts(model), [("mic", "Hello")])

    def test_history_is_still_capped(self):
        model = _model()
        for i in range(30):
            model.addTranslationHistory("mic", f"message {i}", phrase_id=i)
        self.assertEqual(len(model.translation_history), 20)
        self.assertEqual(model.translation_history[-1]["text"], "message 29")


@patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
class TranscriberPhraseIdTests(unittest.TestCase):
    def _transcriber(self, speaker=False, **kwargs):
        transcriber = AudioTranscriber(speaker, FakeAudioSource(), 3, 10, "Google", **kwargs)
        transcriber.audio_recognizer.recognize_google = MagicMock(return_value=("hello", 0.9))
        return transcriber

    def test_interim_results_of_a_growing_utterance_share_a_phrase_id(self, _):
        transcriber = self._transcriber()
        now = datetime.now()
        audio_queue = Queue()
        ids = []
        for offset in (0, 100, 200):
            audio_queue.put((b"\x01\x00", now + timedelta(milliseconds=offset)))
            self.assertTrue(transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"]))
            ids.append(transcriber.getTranscript()["phrase_id"])
        self.assertEqual(len(set(ids)), 1)

    def test_a_silence_gap_starts_a_new_phrase_id(self, _):
        transcriber = self._transcriber()
        now = datetime.now()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", now))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        first = transcriber.getTranscript()["phrase_id"]

        # Longer than phrase_timeout (3 s) after the previous chunk.
        audio_queue.put((b"\x02\x00", now + timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        results = []
        while transcriber.hasTranscript():
            results.append(transcriber.getTranscript())
        second = [r["phrase_id"] for r in results if r["text"]]

        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second[0])

    def test_vad_fragments_share_an_id_until_the_utterance_ends(self, _):
        transcriber = self._transcriber(vad_segmented=True)
        now = datetime.now()
        audio_queue = Queue()
        ids = []
        for reason in ("max_duration", "max_duration", "silence"):
            audio_queue.put((b"\x01\x00" * 160, now, reason))
            self.assertTrue(transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"]))
            ids.append(transcriber.getTranscript()["phrase_id"])
        self.assertEqual(len(set(ids)), 1)

        audio_queue.put((b"\x01\x00" * 160, now, "silence"))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        self.assertNotIn(transcriber.getTranscript()["phrase_id"], ids)

    def test_mic_and_speaker_never_share_a_phrase_id(self, _):
        mic = self._transcriber(speaker=False)
        speaker = self._transcriber(speaker=True)
        now = datetime.now()
        ids = []
        for transcriber in (mic, speaker):
            audio_queue = Queue()
            audio_queue.put((b"\x01\x00", now))
            transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
            ids.append(transcriber.getTranscript()["phrase_id"])
        self.assertNotEqual(ids[0], ids[1])

    def test_clearing_the_transcriber_starts_a_new_phrase_id(self, _):
        transcriber = self._transcriber()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        first = transcriber.getTranscript()["phrase_id"]
        transcriber.clearTranscriptData()
        audio_queue.put((b"\x01\x00", datetime.now()))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        self.assertNotEqual(transcriber.getTranscript()["phrase_id"], first)


if __name__ == "__main__":
    unittest.main()
