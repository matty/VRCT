"""Nothing is dropped from long, continuous speech.

- A phrase that reaches the length limit is cut at a quiet point and the
  rest is carried into the next phrase, instead of being cut wherever the
  last chunk happened to end.
- Every result says which utterance it belongs to and whether it is the
  last one for it, and an utterance whose last result was only an interim
  one is closed explicitly, so consumers can tell drafts from the final
  text.
"""

import unittest
from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from models.transcription.transcription_transcriber import (
    MAX_PHRASE_DURATION_SECONDS,
    AudioTranscriber,
)

RATE = 16000


class FakeAudioSource:
    SAMPLE_RATE = RATE
    SAMPLE_WIDTH = 2
    channels = 1


def _tone(seconds: float, level: int = 8000) -> np.ndarray:
    return (np.sin(np.arange(int(seconds * RATE)) * 0.3) * level).astype("<i2")


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype="<i2")


def _drain(transcriber):
    results = []
    while transcriber.hasTranscript():
        results.append(transcriber.getTranscript())
    return results


@patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
class CarryOverTests(unittest.TestCase):
    def _transcriber(self, **kwargs):
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google", **kwargs)
        sent = []

        def recognize(audio_data, **_kwargs):
            sent.append(audio_data.get_raw_data())
            return (f"text{len(sent)}", 0.9)

        transcriber.audio_recognizer.recognize_google = MagicMock(side_effect=recognize)
        # Keep the audio exactly as queued so the cut position can be checked.
        transcriber._preprocessAudio = lambda audio_data: audio_data
        return transcriber, sent

    def test_long_phrase_is_cut_at_the_quiet_point(self, _):
        transcriber, sent = self._transcriber()
        # 16 s of speech with a 0.3 s pause that ends 1.5 s before the end.
        audio = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        audio_queue = Queue()
        audio_queue.put((audio.tobytes(), datetime.now()))

        self.assertTrue(transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"]))

        self.assertEqual(len(sent), 1)
        head_seconds = len(sent[0]) / 2 / RATE
        self.assertGreater(head_seconds, 14.2)
        self.assertLess(head_seconds, 14.5)
        tail = transcriber.audio_sources["last_sample"]
        # Nothing is lost: head + carried tail are exactly the queued audio.
        self.assertEqual(sent[0] + tail, audio.tobytes())

    def test_carried_tail_starts_the_next_phrase(self, _):
        transcriber, sent = self._transcriber()
        audio = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        now = datetime.now()
        audio_queue = Queue()
        audio_queue.put((audio.tobytes(), now))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        tail = transcriber.audio_sources["last_sample"]

        more = _tone(2.0).tobytes()
        audio_queue.put((more, now + timedelta(seconds=1)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        self.assertEqual(sent[1], tail + more)

    def test_carried_tail_is_not_sent_on_its_own_while_speech_may_continue(self, _):
        transcriber, sent = self._transcriber()
        audio = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        audio_queue = Queue()
        audio_queue.put((audio.tobytes(), datetime.now()))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        self.assertFalse(transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"]))
        self.assertEqual(len(sent), 1)

    def test_carried_tail_is_sent_when_no_more_audio_arrives(self, _):
        transcriber, sent = self._transcriber()
        audio = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        audio_queue = Queue()
        # Recorded long enough ago that phrase_timeout has already passed.
        audio_queue.put((audio.tobytes(), datetime.now() - timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        tail = transcriber.audio_sources["last_sample"]
        self.assertTrue(tail)

        self.assertTrue(transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"]))

        self.assertEqual(sent[1], tail)
        self.assertEqual(transcriber.audio_sources["last_sample"], b"")

    def test_every_sample_is_transcribed_exactly_once_over_a_long_monologue(self, _):
        transcriber, sent = self._transcriber()
        rng = np.random.default_rng(0)
        # 50 s of "speech" with a short dip every ~2 s, delivered in 3 s chunks.
        pieces = []
        while sum(len(p) for p in pieces) < 50 * RATE:
            pieces += [_tone(float(rng.uniform(1.5, 2.5))), _silence(0.15)]
        audio = np.concatenate(pieces)[: 50 * RATE]
        start = datetime.now() - timedelta(seconds=600)
        audio_queue = Queue()
        for index, offset in enumerate(range(0, len(audio), 3 * RATE)):
            audio_queue.put((audio[offset:offset + 3 * RATE].tobytes(), start + timedelta(seconds=2.9 * index)))

        for _ in range(100):
            transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
            _drain(transcriber)
            if audio_queue.empty() and not transcriber.audio_sources["last_sample"]:
                break

        # With the queue pre-filled only final sends happen, so the sent
        # clips are consecutive, non-overlapping pieces of the stream.
        self.assertEqual(b"".join(sent), audio.tobytes())
        self.assertGreaterEqual(len(sent), 4)
        for clip in sent:
            self.assertLessEqual(len(clip) / 2 / RATE, MAX_PHRASE_DURATION_SECONDS + 3)

    def test_stereo_buffer_is_cut_on_a_frame_boundary(self, _):
        source = FakeAudioSource()
        source.channels = 2
        transcriber = AudioTranscriber(False, source, 3, 10, "Google")
        transcriber._finalizeAndTranscribe = MagicMock(return_value=True)
        mono = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        stereo = np.repeat(mono, 2)
        audio_queue = Queue()
        audio_queue.put((stereo.tobytes(), datetime.now()))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        tail = transcriber.audio_sources["last_sample"]
        self.assertEqual(len(tail) % 4, 0)
        self.assertEqual((len(stereo.tobytes()) - len(tail)) % 4, 0)

    def test_vad_safety_net_also_carries_the_rest(self, _):
        transcriber, sent = self._transcriber(vad_segmented=True)
        audio = np.concatenate([_tone(14.2), _silence(0.3), _tone(1.5)])
        audio_queue = Queue()
        audio_queue.put((audio.tobytes(), datetime.now(), "max_duration"))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        tail = transcriber.audio_sources["last_sample"]
        self.assertTrue(tail)
        self.assertLess(len(tail) / 2 / RATE, 3.0)
        # The sent clip is padded with silence for VAD; the audio itself is intact.
        self.assertIn(audio.tobytes()[: len(audio.tobytes()) - len(tail)], sent[0])


@patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
class PhraseEndSignalTests(unittest.TestCase):
    def _transcriber(self, texts=None, **kwargs):
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google", **kwargs)
        replies = iter(texts) if texts is not None else None

        def recognize(audio_data, **_kwargs):
            return (next(replies) if replies is not None else "hello", 0.9)

        transcriber.audio_recognizer.recognize_google = MagicMock(side_effect=recognize)
        return transcriber

    def test_interim_results_are_not_final(self, _):
        transcriber = self._transcriber()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (result,) = _drain(transcriber)
        self.assertEqual(result["text"], "hello")
        self.assertIs(result["final"], False)

    def test_utterance_that_ended_on_an_interim_result_is_closed(self, _):
        transcriber = self._transcriber()
        audio_queue = Queue()
        # Old enough that the next call sees phrase_timeout already passed.
        audio_queue.put((b"\x01\x00", datetime.now() - timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (interim,) = _drain(transcriber)
        self.assertIs(interim["final"], False)

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (closing,) = _drain(transcriber)

        self.assertEqual(closing["text"], "")
        self.assertIs(closing["final"], True)
        self.assertEqual(closing["phrase_id"], interim["phrase_id"])
        # The audio was already transcribed: it is not sent again.
        self.assertEqual(transcriber.audio_recognizer.recognize_google.call_count, 1)

    def test_closing_is_signalled_once(self, _):
        transcriber = self._transcriber()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now() - timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        _drain(transcriber)
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        self.assertEqual(_drain(transcriber), [])

    def test_final_result_closes_the_utterance_itself(self, _):
        transcriber = self._transcriber(vad_segmented=True)
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00" * 160, datetime.now(), "silence"))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (result,) = _drain(transcriber)
        self.assertEqual(result["text"], "hello")
        self.assertIs(result["final"], True)

    def test_new_utterance_closes_the_previous_one_first(self, _):
        transcriber = self._transcriber()
        now = datetime.now()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", now))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (first,) = _drain(transcriber)

        audio_queue.put((b"\x02\x00", now + timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        closing, second = _drain(transcriber)

        self.assertEqual((closing["text"], closing["final"], closing["phrase_id"]), ("", True, first["phrase_id"]))
        self.assertEqual(second["text"], "hello")
        self.assertNotEqual(second["phrase_id"], first["phrase_id"])

    def test_final_send_that_recognises_nothing_still_closes_the_utterance(self, _):
        # interim result, then the final (longer) clip comes back empty.
        transcriber = self._transcriber(vad_segmented=True)
        transcriber.audio_recognizer.recognize_google = MagicMock(side_effect=[("hello", 0.9), ("", 0.0)])
        now = datetime.now()
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00" * 160, now, "max_duration"))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (interim,) = _drain(transcriber)

        audio_queue.put((b"\x01\x00" * 160, now, "silence"))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        (closing,) = _drain(transcriber)

        self.assertEqual((closing["text"], closing["final"], closing["phrase_id"]), ("", True, interim["phrase_id"]))

    def test_utterance_without_any_result_is_not_signalled(self, _):
        transcriber = self._transcriber()
        transcriber.audio_recognizer.recognize_google = MagicMock(return_value=("", 0.0))
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now() - timedelta(seconds=10)))
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])
        self.assertEqual(_drain(transcriber), [])


if __name__ == "__main__":
    unittest.main()
