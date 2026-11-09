import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from speech_recognition import AudioData

from models.transcription import audio_preprocess as ap
from models.transcription.transcription_transcriber import AudioTranscriber

SR = 16000


def _db(samples: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-12)


def _tone_in_noise(seconds: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A 440 Hz tone from 1 s to 2 s over steady white noise + 50 Hz hum."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * ((t > 1) & (t < 2))
    noise = 0.02 * rng.standard_normal(t.size) + 0.2 * np.sin(2 * np.pi * 50 * t)
    return (tone + noise).astype(np.float32), tone.astype(np.float32)


class TestStft(unittest.TestCase):
    def test_round_trip_is_lossless(self) -> None:
        for sr in (16000, 44100, 48000):
            x = np.random.default_rng(1).standard_normal(sr * 2).astype(np.float32) * 0.1
            n = ap._frame_length(sr)
            window = np.hanning(n + 1)[:-1]
            y = ap._istft(ap._stft(x, n, n // 4, window), n, n // 4, window, x.size)
            np.testing.assert_allclose(y, x, atol=1e-5)


class TestPreprocess(unittest.TestCase):
    def test_removes_hum_and_noise_but_keeps_speech_band(self) -> None:
        x, _ = _tone_in_noise()
        y = ap.preprocess_float(x, SR, ap.PreprocessSettings(noise_reduction=True, normalize=False))
        noise_only = slice(0, SR // 2)
        tone_part = slice(int(1.2 * SR), int(1.8 * SR))
        # Hum (high-pass) + noise floor (gate): well over 12 dB less noise.
        self.assertLess(_db(y[noise_only]), _db(x[noise_only]) - 15)
        # The tone loses at most ~1.5 dB (its level without hum/noise is -13.5 dB).
        self.assertGreater(_db(y[tone_part]), -13.5 - 1.5)

    def test_attenuation_is_capped(self) -> None:
        rng = np.random.default_rng(2)
        noise = (0.05 * rng.standard_normal(SR * 2)).astype(np.float32)
        settings = ap.PreprocessSettings(highpass_hz=0, noise_reduction=True, normalize=False, max_attenuation_db=12.0)
        y = ap.preprocess_float(noise, SR, settings)
        self.assertGreater(_db(y), _db(noise) - 12.5)

    def test_short_clip_skips_gate_but_keeps_highpass(self) -> None:
        x = (0.05 * np.random.default_rng(3).standard_normal(SR // 4)).astype(np.float32)
        gate_only = ap.PreprocessSettings(highpass_hz=0, noise_reduction=True, normalize=False)
        self.assertIs(ap.spectral_filter(x, SR, gate_only), x)
        t = np.arange(SR // 4) / SR
        hum = (0.2 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
        y = ap.spectral_filter(x + hum, SR, ap.DEFAULT_SETTINGS)
        self.assertLess(_db(y), _db(x + hum) - 6)

    def test_digital_silence_padding_is_kept_and_ignored(self) -> None:
        x, _ = _tone_in_noise()
        padded = np.concatenate([np.zeros(SR // 2, np.float32), x, np.zeros(SR, np.float32)])
        y = ap.preprocess_float(padded, SR, ap.PreprocessSettings(noise_reduction=True, normalize=False))
        self.assertFalse(np.any(y[: SR // 2]))
        self.assertFalse(np.any(y[-SR:]))
        core = y[SR // 2 : SR // 2 + x.size]
        # Same result as without the padding: the zeros were not used as the noise floor.
        np.testing.assert_allclose(core, ap.preprocess_float(x, SR, ap.PreprocessSettings(noise_reduction=True, normalize=False)), atol=1e-6)

    def test_normalize_boosts_quiet_speech_without_clipping(self) -> None:
        x, _ = _tone_in_noise()
        quiet = x * 0.05
        y = ap.normalize_level(quiet, SR, ap.DEFAULT_SETTINGS)
        self.assertGreater(_db(y), _db(quiet) + 15)
        self.assertLessEqual(np.max(np.abs(y)), 10 ** (-1 / 20) + 1e-6)

    def test_normalize_gain_is_capped(self) -> None:
        x, _ = _tone_in_noise()
        tiny = x * 0.001
        y = ap.normalize_level(tiny, SR, ap.DEFAULT_SETTINGS)
        self.assertAlmostEqual(_db(y) - _db(tiny), ap.DEFAULT_SETTINGS.max_gain_db, places=3)

    def test_silence_and_empty_input_pass_through(self) -> None:
        self.assertEqual(ap.preprocess_pcm16(b"", SR), b"")
        zeros = np.zeros(SR, dtype="<i2").tobytes()
        self.assertEqual(ap.preprocess_pcm16(zeros, SR), zeros)

    def test_audio_data_round_trip_keeps_format(self) -> None:
        x, _ = _tone_in_noise()
        pcm = np.rint(x * 32767).astype("<i2").tobytes()
        audio = AudioData(pcm, 48000, 2)
        out = ap.preprocess_audio_data(audio)
        self.assertIsInstance(out, AudioData)
        self.assertEqual(out.sample_rate, 48000)
        self.assertEqual(out.sample_width, 2)
        self.assertEqual(len(out.get_raw_data()), len(pcm))
        self.assertIs(ap.preprocess_audio_data(audio, None), audio)


class FakeAudioSource:
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    channels = 1


class TestTranscriberHook(unittest.TestCase):
    def _transcriber(self, noise_reduction: bool) -> AudioTranscriber:
        with patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False):
            transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google", noise_reduction=noise_reduction)
        transcriber.audio_sources["last_sample"] = np.ones(1600, dtype="<i2").tobytes()
        return transcriber

    def _run(self, transcriber: AudioTranscriber) -> MagicMock:
        provider = MagicMock()
        provider.transcribe.return_value = ("hello", 0.9, True)
        with patch.object(transcriber, "_resolve_provider", return_value=provider):
            transcriber._finalizeAndTranscribe(["English"], ["United States"], -0.8, 0.6, 0)
        return provider

    def test_always_cleans_up_and_gate_follows_setting(self) -> None:
        for enabled in (False, True):
            transcriber = self._transcriber(enabled)
            cleaned = AudioData(b"\x00\x00" * 10, 16000, 2)
            with patch("models.transcription.transcription_transcriber.preprocess_audio_data", return_value=cleaned) as pre:
                provider = self._run(transcriber)
            self.assertIs(provider.transcribe.call_args.args[0], cleaned)
            settings = pre.call_args.args[1]
            self.assertEqual(settings.noise_reduction, enabled)
            self.assertGreater(settings.highpass_hz, 0)
            self.assertTrue(settings.normalize)
            self.assertEqual(transcriber.getTranscript()["text"], "hello")

    def test_preprocess_failure_falls_back_to_original(self) -> None:
        transcriber = self._transcriber(True)
        with patch("models.transcription.transcription_transcriber.preprocess_audio_data", side_effect=RuntimeError):
            provider = self._run(transcriber)
        sent = provider.transcribe.call_args.args[0]
        self.assertEqual(sent.get_raw_data(), np.ones(1600, dtype="<i2").tobytes())


if __name__ == "__main__":
    unittest.main()
