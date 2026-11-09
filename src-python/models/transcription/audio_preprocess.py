"""確定した発話クリップを、音声認識に送る直前に整える。

マイクが拾う自分の声にも、スピーカー (ループバック) 側で拾う他プレイヤーの
声にも、一定の背景ノイズが乗っている: PC のファン、エアコン、電源のハム音、
安いマイクのサー音、風や振動の低音。このモジュールは `AudioTranscriber` が
送ろうとしているクリップ (Whisper / Google / クラウド API のどれでも) から
それを取り除く。numpy だけで動く (追加の依存もバンドルするモデルも無い)。

フレーズが区切られた後、クリップ全体に一度だけ掛ける。発話の始まりと
終わりを決める VAD / エネルギー閾値は元の音声のままなので、調整済みの
閾値には影響しない。

処理は次の 3 段:

1. ハイパスフィルタ (`highpass_hz`、既定 80 Hz)。音声は約 85 Hz (男声の
   最も低い基本周波数) より下にほとんどエネルギーを持たないが、低音の
   ゴロゴロ・ハム音・DC オフセットはそこにある。ノイズゲートと同じ STFT
   上のゲインとして、なだらかに減衰させる。

2. スペクトルノイズゲート (定常ノイズのみ)。ノイズのスペクトルはクリップ
   自身から分位点ベースのノイズ推定 (QBNE、Stahl et al., 2000) で求める:
   各周波数ビンで音声が鳴っているのは一部の時間だけなので、時間方向の
   振幅の低い分位点は、間 (ま) の無いクリップでもノイズフロアのよい推定に
   なる。フロアに近い時間-周波数セルをソフトマスクで減衰させ、マスクは
   時間・周波数方向に平滑化する (ミュージカルノイズ対策)。減衰量には
   上限を設ける (`max_attenuation_db`、既定 12 dB)。

   この上限が重要。Whisper はノイズを含む音声で学習されており、強い
   ノイズ除去が残す歪みのほうがノイズそのものより苦手。上限 20 dB では
   12 dB より認識精度が下がった (docs/transcription_noise_reduction_2026-11-08.md)。

   他の人の話し声・音楽・キーボードの打鍵音のような非定常なノイズは
   除去しない (レベルが推定フロアより高いので素通りする)。それどころか
   後ろで他の人が話していると精度が下がった (声がフロアを押し上げ、
   目的の声まで削る) ため、このゲートは既定で無効 (`noise_reduction`)。
   ハイパスと正規化は常に掛ける。

3. 音量の正規化。小さい声 (遠くのプレイヤー、ゲインの低いマイク) を目標の
   音声レベルまで持ち上げる。上限は `max_gain_db`、ピークは -1 dBFS で
   クリップしない。Whisper は log-mel 入力をクリップごとに正規化するので
   影響は小さいが、Google やクラウドのエンジンは音声をそのまま符号化する。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .near_voice_gate import NearVoiceGate
from .resample import TARGET_RATE, resample_to_16k


@dataclass(frozen=True)
class PreprocessSettings:
    highpass_hz: float = 80.0
    # スペクトルノイズゲート。既定は無効 (他の人の声が重なると悪化するため)。
    noise_reduction: bool = False
    # 各ビンの時間方向の振幅のうち、ノイズフロアとみなす分位点。
    noise_quantile: float = 0.2
    # フロア × 10^(gate_low_db/20) 未満のセルは最大まで減衰、
    # フロア × 10^(gate_high_db/20) 以上はそのまま、その間はなだらかに。
    gate_low_db: float = 3.0
    gate_high_db: float = 12.0
    max_attenuation_db: float = 12.0
    normalize: bool = True
    target_speech_dbfs: float = -20.0
    max_gain_db: float = 20.0
    peak_dbfs: float = -1.0


DEFAULT_SETTINGS = PreprocessSettings()

# これより短いとノイズフロアを推定するにはフレームが足りない。
_MIN_GATE_SECONDS = 0.5
_FRAME_SECONDS = 0.032  # 16 kHz で 512 サンプル (Silero VAD と同じ)


def _frame_length(sample_rate: int) -> int:
    return int(2 ** round(np.log2(sample_rate * _FRAME_SECONDS)))


def _highpass_gain(freqs: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """cutoff/2 未満は 0、cutoff 以上は 1、その間はレイズドコサイン。

    スペクトルに急な段差を作らない (リンギングを避ける) ため。
    ビン 0 (DC オフセット) は常に 0。
    """
    if cutoff_hz <= 0:
        return np.ones_like(freqs)
    low = cutoff_hz / 2
    ramp = np.clip((freqs - low) / (cutoff_hz - low), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ramp)


def _stft(samples: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    pad = n_fft // 2
    padded = np.pad(samples, (pad, pad + n_fft), mode="constant")
    frame_count = 1 + (samples.size + pad * 2 - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frame_count)[:, None]
    return np.fft.rfft(padded[idx] * window, axis=1)


def _istft(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    pad = n_fft // 2
    frame_count = frames.shape[0]
    overlap = n_fft // hop
    total = hop * (frame_count - 1) + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    # 重畳加算: フレーム i は [i*hop, i*hop + n_fft) = hop 幅のチャンク `overlap` 個。
    # 全フレームの j 番目のチャンクを 1 回のスライスで足す。
    chunks = frames.reshape(frame_count, overlap, hop)
    window_chunks = (window ** 2).reshape(overlap, hop)
    for j in range(overlap):
        out[j * hop:j * hop + frame_count * hop] += chunks[:, j, :].reshape(-1)
        norm[j * hop:j * hop + frame_count * hop] += np.tile(window_chunks[j], frame_count)
    out /= np.maximum(norm, 1e-8)
    return out[pad:pad + length].astype(np.float32)


def _box_average(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return values
    pad = [(0, 0)] * values.ndim
    pad[axis] = (radius + 1, radius)
    csum = np.cumsum(np.pad(values, pad, mode="edge"), axis=axis)
    width = 2 * radius + 1
    upper = np.take(csum, np.arange(width, csum.shape[axis]), axis=axis)
    lower = np.take(csum, np.arange(0, csum.shape[axis] - width), axis=axis)
    return (upper - lower) / width


def _smooth(mask: np.ndarray, time_radius: int, freq_radius: int) -> np.ndarray:
    """マスクを近傍のフレーム・ビンで箱型平均する。"""
    return _box_average(_box_average(mask, time_radius, axis=0), freq_radius, axis=1)


def _gate_gain(magnitude: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    """定常ノイズのフロアを減衰させるセルごとのゲイン (モジュールの docstring 参照)。"""
    noise_floor = np.quantile(magnitude, settings.noise_quantile, axis=0)
    noise_floor = np.maximum(noise_floor, 1e-10)
    snr_db = 20 * np.log10(np.maximum(magnitude, 1e-10) / noise_floor)
    mask = np.clip(
        (snr_db - settings.gate_low_db) / (settings.gate_high_db - settings.gate_low_db), 0.0, 1.0
    )
    # ±2 フレーム (hop = 1/4 フレームで ±16 ms)、±2 ビン (16 kHz で ±62 Hz)。
    mask = _smooth(mask, time_radius=2, freq_radius=2)
    floor_gain = 10 ** (-settings.max_attenuation_db / 20)
    return floor_gain + (1.0 - floor_gain) * mask


def spectral_filter(samples: np.ndarray, sample_rate: int, settings: PreprocessSettings) -> np.ndarray:
    """ハイパスフィルタとスペクトルノイズゲートを 1 回の STFT で掛ける。

    `_MIN_GATE_SECONDS` より短いクリップはゲートを省く (ノイズフロアを
    推定するにはフレームが足りない)。ハイパスは掛ける。
    """
    use_gate = settings.noise_reduction and samples.size >= sample_rate * _MIN_GATE_SECONDS
    if settings.highpass_hz <= 0 and not use_gate:
        return samples
    n_fft = _frame_length(sample_rate)
    if samples.size < n_fft:
        return samples
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    spec = _stft(samples, n_fft, hop, window)
    gain = _highpass_gain(np.fft.rfftfreq(n_fft, d=1.0 / sample_rate), settings.highpass_hz)[None, :]
    if use_gate:
        gain = gain * _gate_gain(np.abs(spec), settings)
    return _istft(spec * gain, n_fft, hop, window, samples.size)


def normalize_level(samples: np.ndarray, sample_rate: int, settings: PreprocessSettings) -> np.ndarray:
    """音声レベルを `target_speech_dbfs` に近づける (クリップさせない)。

    音声レベルは 32 ms フレームのうち大きい方から 30% の RMS。前後の無音や
    間 (ま) に引きずられないようにするため。
    """
    if samples.size == 0:
        return samples
    frame = _frame_length(sample_rate)
    usable = samples[: samples.size - samples.size % frame] if samples.size >= frame else samples
    frames = usable.reshape(-1, frame) if usable.size >= frame else usable.reshape(1, -1)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    loud = np.sort(frame_rms)[-max(1, int(np.ceil(frame_rms.size * 0.3))):]
    speech_rms = float(np.sqrt(np.mean(loud ** 2)))
    peak = float(np.max(np.abs(samples)))
    if speech_rms < 1e-6 or peak < 1e-6:
        return samples
    gain_db = settings.target_speech_dbfs - 20 * np.log10(speech_rms)
    gain_db = min(gain_db, settings.max_gain_db)
    gain_db = min(gain_db, settings.peak_dbfs - 20 * np.log10(peak))
    return (samples * (10 ** (gain_db / 20))).astype(np.float32)


def preprocess_float(
    samples: np.ndarray,
    sample_rate: int,
    settings: PreprocessSettings = DEFAULT_SETTINGS,
    near_gate: Optional[NearVoiceGate] = None,
) -> Optional[np.ndarray]:
    """[-1, 1] のモノラル float32 に、有効な処理を掛ける。

    前後のデジタル無音 (完全な 0。`AudioTranscriber._padWithSilenceForVad`
    が付けるパディングなど) はそのまま残し、その間だけを処理する。
    含めると 0 がノイズフロアとみなされ、ゲートが何もしなくなる。

    `near_gate` を渡すと遠くの声だけの区間を落とす (near_voice_gate 参照)。
    クリップ全体が遠くの声だけなら None を返す。
    """
    samples = samples.astype(np.float32, copy=False)
    nonzero = np.flatnonzero(samples)
    if nonzero.size == 0:
        return samples
    start, end = int(nonzero[0]), int(nonzero[-1]) + 1
    core = _process(samples[start:end], sample_rate, settings, near_gate)
    if core is None:
        return None
    out = samples.copy()
    out[start:end] = core
    return out


def _process(
    samples: np.ndarray,
    sample_rate: int,
    settings: PreprocessSettings,
    near_gate: Optional[NearVoiceGate],
) -> Optional[np.ndarray]:
    out = spectral_filter(samples, sample_rate, settings)
    if near_gate is not None:
        # 正規化の前に掛ける (レベルで遠近を判定するので、元のレベルのまま見る)。
        out = near_gate.process(out, sample_rate)
        if out is None:
            return None
    if settings.normalize:
        out = normalize_level(out, sample_rate, settings)
    return np.clip(out, -1.0, 1.0)


def preprocess_pcm16(
    data: bytes, sample_rate: int, settings: PreprocessSettings = DEFAULT_SETTINGS
) -> bytes:
    """`preprocess_float` の、モノラル 16bit リトルエンディアン PCM バイト列版。"""
    samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2")
    if samples.size == 0:
        return data
    out = preprocess_float(samples.astype(np.float32) / 32768.0, sample_rate, settings)
    return np.rint(out * 32767.0).astype("<i2").tobytes()


def preprocess_audio_data(
    audio_data,
    settings: Optional[PreprocessSettings] = DEFAULT_SETTINGS,
    near_gate: Optional[NearVoiceGate] = None,
    resample_16k: bool = False,
):
    """speech_recognition の `AudioData` (モノラル) を整えたコピーを返す。

    サンプル幅は問わない。結果は 16bit。`resample_16k` なら 16 kHz に
    折り返し無しでリサンプリングする (後段の `get_raw_data(convert_rate=16000)`
    が `audioop.ratecv` を使わずに済む)。クリップ全体が遠くの声だけなら None。
    """
    if settings is None and near_gate is None and not resample_16k:
        return audio_data
    raw = audio_data.get_raw_data(convert_width=2)
    samples = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2").astype(np.float32) / 32768.0
    rate = audio_data.sample_rate
    if resample_16k and rate > TARGET_RATE:
        samples = resample_to_16k(samples, rate)
        rate = TARGET_RATE
    if samples.size and (settings is not None or near_gate is not None):
        samples = preprocess_float(samples, rate, settings or PreprocessSettings(highpass_hz=0, normalize=False), near_gate)
        if samples is None:
            return None
    cleaned = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return type(audio_data)(cleaned, rate, 2)
