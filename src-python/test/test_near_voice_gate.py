import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from models.transcription.near_voice_gate import NearVoiceGate, NearVoiceSettings
from models.transcription.transcription_transcriber import AudioTranscriber

SR = 16000


def _voice(seconds: float, dbfs: float, seed: int = 0, rate: int = SR) -> np.ndarray:
    """Speech-like signal (RMS = dbfs): 200 Hz harmonics, amplitude-modulated at 4 Hz (syllables)."""
    t = np.arange(int(rate * seconds)) / rate
    tone = sum(np.sin(2 * np.pi * 200 * k * t + seed) / k for k in range(1, 6))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t + seed)
    x = tone * env
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)).astype(np.float32)


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestNearVoiceGate(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.gate = NearVoiceGate(NearVoiceSettings(margin_db=12.0, window_sec=120.0), clock=self.clock)

    def test_first_clip_sets_reference_and_passes(self) -> None:
        near = _voice(2.0, -26)
        out = self.gate.process(near, SR)
        np.testing.assert_allclose(out, near, atol=1e-6)
        # The reference is the level of the loudest 10% of frames (speech peaks), a few dB above RMS.
        self.assertTrue(-26 <= self.gate.reference_db <= -21, self.gate.reference_db)

    def _establish(self, dbfs: float = -26) -> None:
        for _ in range(3):
            self.gate.process(_voice(2.0, dbfs), SR)
            self.clock.now += 5

    def test_far_only_clip_is_dropped(self) -> None:
        self._establish()
        self.assertIsNone(self.gate.process(_voice(3.0, -45, seed=1), SR))

    def test_far_voice_around_near_voice_is_attenuated(self) -> None:
        near = _voice(2.0, -26)
        far = _voice(1.5, -45, seed=1)
        clip = np.concatenate([far, near + _voice(2.0, -45, seed=1), far])
        out = self.gate.process(clip, SR)
        lead = slice(0, int(1.0 * SR))  # far only, outside the 400 ms hold time
        self.assertLess(_db(out[lead]), _db(clip[lead]) - 30)
        middle = slice(int(1.6 * SR), int(3.4 * SR))
        self.assertAlmostEqual(_db(out[middle]), _db(clip[middle]), delta=0.5)

    def test_history_expires_so_far_voices_return_after_silence(self) -> None:
        self._establish()
        self.clock.now += 121
        self.assertIsNotNone(self.gate.process(_voice(3.0, -45, seed=1), SR))

    def test_one_shout_does_not_cut_normal_voice(self) -> None:
        for _ in range(4):
            self.gate.process(_voice(2.0, -26), SR)
            self.clock.now += 5
        self.gate.process(_voice(2.0, -12), SR)  # laughing / shouting, 14 dB louder
        self.clock.now += 5
        normal = _voice(2.0, -26)
        np.testing.assert_allclose(self.gate.process(normal, SR), normal, atol=1e-6)

    def test_far_chatter_does_not_lower_reference(self) -> None:
        self._establish()
        for _ in range(10):
            self.clock.now += 5
            self.assertIsNone(self.gate.process(_voice(3.0, -45, seed=1), SR))

    def test_loud_first_clip_does_not_lock_out_normal_voice(self) -> None:
        self.gate.process(_voice(2.0, -12), SR)  # session starts with a laugh
        for _ in range(4):
            self.clock.now += 5
            self.assertIsNotNone(self.gate.process(_voice(2.0, -26), SR))

    def test_quiet_noise_does_not_set_reference(self) -> None:
        noise = (np.random.default_rng(0).standard_normal(SR) * 10 ** (-70 / 20)).astype(np.float32)
        self.gate.process(noise, SR)
        self.assertIsNone(self.gate.reference_db)

    def test_digital_silence_is_ignored(self) -> None:
        self.assertIs(self.gate.process(np.zeros(SR, np.float32), SR).dtype, np.dtype(np.float32))
        self.assertIsNone(self.gate.reference_db)


class FakeAudioSource:
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    channels = 1


class TestTranscriberSkipsFarClips(unittest.TestCase):
    def test_far_only_clip_is_not_sent_to_the_engine(self) -> None:
        with patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False):
            transcriber = AudioTranscriber(True, FakeAudioSource(), 3, 10, "Google", near_voice_only=True)
        transcriber.near_gate._clock = FakeClock()
        provider = MagicMock()
        provider.transcribe.return_value = ("hello", 0.9, True)

        def run(audio: np.ndarray) -> None:
            transcriber.audio_sources["last_sample"] = np.rint(audio * 32767).astype("<i2").tobytes()
            with patch.object(transcriber, "_resolve_provider", return_value=provider):
                transcriber._finalizeAndTranscribe(["English"], ["United States"], -0.8, 0.6, 0)

        for _ in range(3):
            run(_voice(2.0, -26))
        self.assertEqual(provider.transcribe.call_count, 3)
        run(_voice(2.0, -45, seed=1))
        self.assertEqual(provider.transcribe.call_count, 3)
        self.assertEqual(transcriber.near_voice_skips, 1)

    def test_whisper_gets_16khz_audio(self) -> None:
        source = FakeAudioSource()
        source.SAMPLE_RATE = 48000
        with patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False):
            transcriber = AudioTranscriber(False, source, 3, 10, "Google")
        transcriber.transcription_engine = "Whisper"
        transcriber.whisper_model = MagicMock()
        transcriber.audio_sources["last_sample"] = np.rint(_voice(1.0, -26, rate=48000) * 32767).astype("<i2").tobytes()
        provider = MagicMock()
        provider.transcribe.return_value = ("hello", 0.9, True)
        with patch.object(transcriber, "_resolve_provider", return_value=provider):
            transcriber._finalizeAndTranscribe(["English"], ["United States"], -0.8, 0.6, 0)
        sent = provider.transcribe.call_args.args[0]
        self.assertEqual(sent.sample_rate, 16000)
        self.assertEqual(len(sent.get_raw_data()), 16000 * 2)

    def test_google_keeps_device_rate(self) -> None:
        source = FakeAudioSource()
        source.SAMPLE_RATE = 48000
        with patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False):
            transcriber = AudioTranscriber(False, source, 3, 10, "Google")
        transcriber.audio_sources["last_sample"] = np.rint(_voice(1.0, -26) * 32767).astype("<i2").tobytes()
        provider = MagicMock()
        provider.transcribe.return_value = ("hello", 0.9, True)
        with patch.object(transcriber, "_resolve_provider", return_value=provider):
            transcriber._finalizeAndTranscribe(["English"], ["United States"], -0.8, 0.6, 0)
        self.assertEqual(provider.transcribe.call_args.args[0].sample_rate, 48000)


if __name__ == "__main__":
    unittest.main()
