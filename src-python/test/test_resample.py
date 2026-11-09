import unittest

import numpy as np

from models.transcription.audio_vad import Pcm16MonoNormalizer
from models.transcription.resample import AntiAliasFilter, resample_to_16k, _fast_length


def _tone(freq: float, rate: int, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _db_rel(y: np.ndarray, amp: float = 0.5) -> float:
    y = y[2000:-2000].astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) / (amp / np.sqrt(2)) + 1e-12)


class TestResampleTo16k(unittest.TestCase):
    def test_keeps_speech_band_and_removes_aliases(self) -> None:
        for rate in (44100, 48000):
            for freq in (300, 3000, 6500):
                y = resample_to_16k(_tone(freq, rate), rate)
                self.assertEqual(y.size, 16000)
                self.assertAlmostEqual(_db_rel(y), 0.0, delta=0.2)
            for freq in (9000, 12000, 15000, 20000):
                # audioop.ratecv folds these back into 0-8 kHz at 0 dB.
                self.assertLess(_db_rel(resample_to_16k(_tone(freq, rate), rate)), -60)

    def test_low_rates_pass_through(self) -> None:
        x = _tone(1000, 16000)
        self.assertIs(resample_to_16k(x, 16000), x)

    def test_fast_length(self) -> None:
        for n in (1, 7, 720001, 529201):
            m = _fast_length(n)
            self.assertGreaterEqual(m, n)
            r = m
            for p in (2, 3, 5):
                while r % p == 0:
                    r //= p
            self.assertEqual(r, 1)


class TestAntiAliasFilter(unittest.TestCase):
    def _run(self, x: np.ndarray, rate: int, chunk: int = 2048) -> np.ndarray:
        pcm = (x * 32767).astype("<i2").tobytes()
        f = AntiAliasFilter(rate)
        out = b"".join(f.process(pcm[i:i + chunk]) for i in range(0, len(pcm), chunk))
        return np.frombuffer(out, "<i2").astype(np.float32) / 32767

    def test_streaming_matches_one_shot(self) -> None:
        x = np.random.default_rng(0).standard_normal(48000).astype(np.float32) * 0.1
        np.testing.assert_array_equal(self._run(x, 48000, 1000), self._run(x, 48000, x.size * 2))

    def test_passes_speech_band_and_blocks_above_8khz(self) -> None:
        self.assertAlmostEqual(_db_rel(self._run(_tone(3000, 48000), 48000)), 0.0, delta=0.2)
        self.assertLess(_db_rel(self._run(_tone(12000, 48000), 48000)), -60)

    def test_disabled_at_16khz(self) -> None:
        pcm = (_tone(7000, 16000) * 32767).astype("<i2").tobytes()
        self.assertEqual(AntiAliasFilter(16000).process(pcm), pcm)


class TestNormalizerAliasing(unittest.TestCase):
    def test_vad_normalizer_does_not_fold_high_frequencies(self) -> None:
        pcm = (_tone(15000, 48000) * 32767).astype("<i2").tobytes()
        normalizer = Pcm16MonoNormalizer(48000, 2, 1)
        out = b"".join(normalizer.process(pcm[i:i + 4096]) for i in range(0, len(pcm), 4096))
        y = np.frombuffer(out, "<i2").astype(np.float32) / 32767
        self.assertLess(_db_rel(y), -60)


if __name__ == "__main__":
    unittest.main()
