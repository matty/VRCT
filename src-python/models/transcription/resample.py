"""折り返し (エイリアシング) の無い 16 kHz へのリサンプリング。

`audioop.ratecv` はローパスフィルタを掛けずにサンプルを間引く。48 kHz から
16 kHz に落とすと、8 kHz より上の成分 (マイクのサー音、子音の高域、
キーボードの打鍵音) がそのままのレベルで音声帯域に折り返す
(例: 15 kHz → 1 kHz、12 kHz → 4 kHz)。Whisper・OpenAI 互換 API・Deepgram へ
送る音声 (`get_raw_data(convert_rate=16000)`) と、VAD 経路の正規化
(`audio_vad.Pcm16MonoNormalizer`) がこれを使っていた。

- `resample_to_16k`: クリップ全体を FFT でリサンプリングする (理想的な
  ローパス + 間引き)。
- `AntiAliasFilter`: ストリーミング用。`ratecv` の前に掛ける FIR ローパス。
  状態 (直前の入力) をチャンクをまたいで持つ。
"""

from typing import Optional

import numpy as np

TARGET_RATE = 16000
# 通過域の上限と阻止域の始まり。8 kHz (16 kHz のナイキスト) より上を落とす。
_PASS_HZ = 7200.0
_STOP_HZ = 8000.0


def _fast_length(n: int) -> int:
    """n 以上で、素因数が 2・3・5 だけの長さ (FFT が速い)。"""
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """モノラル float の音声を 16 kHz にする (16 kHz より低いレートはそのまま)。"""
    if sample_rate <= TARGET_RATE or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    out_len = int(round(samples.size * TARGET_RATE / sample_rate))
    n = _fast_length(samples.size)
    # FFT の長さを伸ばした分だけ出力も伸ばし、後で切る (n * 16000 / sr が整数になる長さにはならないので丸める)。
    n_out = int(round(n * TARGET_RATE / sample_rate))
    spectrum = np.fft.rfft(samples.astype(np.float64), n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    ramp = np.clip((freqs - _PASS_HZ) / (_STOP_HZ - _PASS_HZ), 0.0, 1.0)
    gain = 0.5 + 0.5 * np.cos(np.pi * ramp)
    keep = n_out // 2 + 1
    out = np.fft.irfft((spectrum * gain)[:keep], n=n_out) * (n_out / n)
    return out[:out_len].astype(np.float32)


def _lowpass_taps(sample_rate: int) -> np.ndarray:
    # カイザー窓 FIR。阻止域 約 80 dB、遷移帯 _PASS_HZ〜_STOP_HZ。
    atten_db = 80.0
    width = 2 * np.pi * (_STOP_HZ - _PASS_HZ) / sample_rate
    count = int(np.ceil((atten_db - 8) / (2.285 * width))) | 1
    beta = 0.1102 * (atten_db - 8.7)
    cutoff = (_PASS_HZ + _STOP_HZ) / 2 / sample_rate
    t = np.arange(count) - (count - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(count, beta)
    return taps / taps.sum()


class AntiAliasFilter:
    """16 kHz に間引く前に掛けるストリーミング FIR ローパス (int16 PCM)。

    出力は入力と同じ長さ・同じレート。群遅延 (タップ数/2 サンプル、48 kHz で
    約 3 ms) だけ遅れる。
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.enabled = sample_rate > TARGET_RATE
        self._taps = _lowpass_taps(sample_rate) if self.enabled else np.ones(1)
        self._history: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._history = None

    def process(self, pcm16: bytes) -> bytes:
        if not self.enabled or not pcm16:
            return pcm16
        x = np.frombuffer(pcm16[: len(pcm16) - len(pcm16) % 2], dtype="<i2").astype(np.float64)
        if self._history is None:
            self._history = np.zeros(self._taps.size - 1)
        extended = np.concatenate([self._history, x])
        y = np.convolve(extended, self._taps, mode="valid")
        self._history = extended[-(self._taps.size - 1):]
        return np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes()
