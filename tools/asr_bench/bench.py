"""Whisper WER/CER with and without audio preprocessing.

usage: bench.py MODEL_DIR SET CONDITION VARIANTS... -> appends JSON lines to results.jsonl
Transcription settings mirror LocalWhisperProvider (beam 5, temp 0, forced
language, logprob/no_speech filters -0.8 / 0.6, VAD-style silence padding).
"""
from pathlib import Path
import json, pickle, re, sys, time, unicodedata, zlib
import numpy as np
import jiwer
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src-python"))
from models.transcription import audio_preprocess as ap  # noqa: E402

model_dir, set_name, condition = sys.argv[1:4]
variants = sys.argv[4:]
bench = pickle.load(open("bench.pkl", "rb"))
data = bench["sets"][set_name]
lang = set_name[:2]
rng = np.random.default_rng(zlib.crc32(f"{set_name}|{condition}".encode()))


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def fit(noise, n):
    if noise.size < n:
        noise = np.tile(noise, int(np.ceil(n / noise.size)))
    start = rng.integers(0, noise.size - n + 1)
    return noise[start:start + n]


def pink(n):
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / 16000)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    p = np.fft.irfft(spec, n)
    t = np.arange(n) / 16000
    hum = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3))  # mains hum + harmonics
    return (p / rms(p) + 0.7 * hum / rms(hum)).astype(np.float32)


def make_noise(kind, n):
    if kind == "pink":
        return pink(n)
    if kind == "babble":
        srcs = rng.choice(len(data["babble"]), 4, replace=False)
        return sum(fit(data["babble"][i], n) / rms(fit(data["babble"][i], n)) for i in srcs)
    if kind == "stationary":
        return fit(bench["noise"]["stationary"][rng.integers(len(bench["noise"]["stationary"]))], n)
    if kind == "keyboard":
        return fit(bench["noise"]["keyboard"][rng.integers(len(bench["noise"]["keyboard"]))], n)
    raise ValueError(kind)


def mix(clean, cond):
    if cond == "clean":
        return clean
    kind, snr = cond.rsplit("@", 1)
    # 0.3 s of noise-only lead-in/out, like a VAD pre-roll and hangover.
    lead = np.zeros(int(0.3 * 16000), np.float32)
    padded = np.concatenate([lead, clean, lead])
    noise = make_noise(kind, padded.size)
    noise = noise / rms(noise) * rms(clean) / (10 ** (float(snr) / 20))
    out = padded + noise
    peak = np.max(np.abs(out))
    return (out / peak * 0.9 if peak > 0.99 else out).astype(np.float32)


SETTINGS = {
    "none": None,
    "hpf_norm": ap.PreprocessSettings(noise_reduction=False),
    "gate12": ap.PreprocessSettings(noise_reduction=True),
    "gate6": ap.PreprocessSettings(noise_reduction=True, max_attenuation_db=6.0),
    "gate20": ap.PreprocessSettings(noise_reduction=True, max_attenuation_db=20.0),
    "gate12_q10": ap.PreprocessSettings(noise_reduction=True, noise_quantile=0.1),
    "gate12_soft": ap.PreprocessSettings(noise_reduction=True, gate_low_db=0.0, gate_high_db=9.0),
    "gate12_q10_soft": ap.PreprocessSettings(noise_reduction=True, noise_quantile=0.1, gate_low_db=0.0, gate_high_db=9.0),
}


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    if lang == "ja":
        return re.sub(r"[\s\W_]+", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", s)).strip()


model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
mixed = [(mix(a, condition), ref) for a, ref in data["test"]]
for variant in variants:
    refs, hyps, t_pre = [], [], 0.0
    for audio, ref in mixed:
        x = audio
        if SETTINGS[variant] is not None:
            t0 = time.perf_counter()
            x = ap.preprocess_float(audio, 16000, SETTINGS[variant])
            t_pre += time.perf_counter() - t0
        x = np.concatenate([np.zeros(4800, np.float32), x, np.zeros(8000, np.float32)])
        segs, info = model.transcribe(
            x, beam_size=5, temperature=0.0, log_prob_threshold=-0.8, no_speech_threshold=0.6,
            language=lang, word_timestamps=False, without_timestamps=True, task="transcribe",
        )
        text = "".join(s.text for s in segs if not (s.avg_logprob < -0.8 or s.no_speech_prob > 0.6))
        refs.append(norm_text(ref))
        hyps.append(norm_text(text))
    if lang == "ja":
        err = jiwer.cer(refs, hyps)
    else:
        err = jiwer.wer(refs, hyps)
    empty = sum(1 for h in hyps if not h)
    rec = dict(model=model_dir.split("/")[-1], set=set_name, cond=condition, variant=variant,
               err=round(err * 100, 2), empty=empty, n=len(refs), pre_ms=round(t_pre / len(refs) * 1000, 1))
    print(json.dumps(rec), flush=True)
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
