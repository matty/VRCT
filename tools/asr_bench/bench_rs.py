"""48 kHz -> 16 kHz resampling: audioop.ratecv (current) vs anti-aliased.

usage: bench_rs.py MODEL_DIR COND VARIANTS...   (JSUT, native 48 kHz)
"""
from pathlib import Path
import audioop, io, json, re, sys, unicodedata, zlib, pickle
import numpy as np, jiwer, soundfile as sf, soxr, pyarrow.parquet as pq
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src-python"))
from models.transcription import resample as rs  # noqa: E402

model_dir, cond = sys.argv[1:3]
variants = sys.argv[3:]
rng = np.random.default_rng(zlib.crc32(f"rs|{cond}".encode()))
SR = 48000

items = []
for row in pq.read_table("data/data/test-00000-of-00004.parquet").to_pylist():
    a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
    a = a.mean(axis=1); assert sr == SR
    if 1.5 <= a.size / SR <= 12.0:
        items.append((a, row["transcription"]))
    if len(items) >= 40:
        break
noise = pickle.load(open("bench.pkl", "rb"))["noise"]
esc_raw = {}


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def esc48(kind):
    # Raw ESC-50 at 44.1 kHz -> 48 kHz (keeps the high band, unlike bench.pkl).
    if kind not in esc_raw:
        cats = {"keyboard": ("keyboard_typing", "mouse_click"),
                "stationary": ("vacuum_cleaner", "airplane", "engine", "rain", "wind", "washing_machine", "train", "helicopter")}[kind]
        out = []
        for r in pq.read_table("data/data/train-00000-of-00002-2f1ab7b824ec751f.parquet").to_pylist():
            if r["category"] in cats:
                b, s = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                b = soxr.resample(b.mean(axis=1), s, SR, quality="VHQ")
                if rms(b) > 1e-3:
                    out.append(b)
        esc_raw[kind] = out
    return esc_raw[kind]


def mix(a):
    if cond == "clean":
        return a
    kind, snr = cond.rsplit("@", 1)
    if kind == "hiss":
        nz = rng.standard_normal(a.size)  # white: mic preamp hiss, flat up to 24 kHz
    else:
        src = esc48(kind)[rng.integers(len(esc48(kind)))]
        src = np.tile(src, int(np.ceil(a.size / src.size)) + 1)
        st = rng.integers(0, src.size - a.size); nz = src[st:st + a.size]
    out = a + nz / rms(nz) * rms(a) / 10 ** (float(snr) / 20)
    pk = np.max(np.abs(out))
    return (out / pk * 0.9 if pk > 0.99 else out).astype(np.float32)


def to16(x, variant):
    if variant == "ratecv":
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        y, _ = audioop.ratecv(pcm, 2, 1, SR, 16000, None)
        return np.frombuffer(y, "<i2").astype(np.float32) / 32768
    if variant == "soxr":
        return soxr.resample(x, SR, 16000, quality="VHQ").astype(np.float32)
    if variant == "fft":
        return rs.resample_to_16k(x, SR)
    if variant == "fir_ratecv":
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        f = rs.AntiAliasFilter(SR)
        pcm = b"".join(f.process(pcm[i:i + 2048]) for i in range(0, len(pcm), 2048))
        y, _ = audioop.ratecv(pcm, 2, 1, SR, 16000, None)
        return np.frombuffer(y, "<i2").astype(np.float32) / 32768
    raise ValueError(variant)


def norm_text(s):
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", s).lower())


model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
mixed = [(mix(a), r) for a, r in items]
for variant in variants:
    refs, hyps = [], []
    for x, ref in mixed:
        y = np.concatenate([np.zeros(4800, np.float32), to16(x, variant), np.zeros(8000, np.float32)])
        segs, _ = model.transcribe(y, beam_size=5, temperature=0.0, log_prob_threshold=-0.8,
                                   no_speech_threshold=0.6, language="ja", without_timestamps=True)
        refs.append(norm_text(ref))
        hyps.append(norm_text("".join(s.text for s in segs if not (s.avg_logprob < -0.8 or s.no_speech_prob > 0.6))))
    rec = dict(model=model_dir.split("/")[-1], cond=cond, variant=variant, err=round(jiwer.cer(refs, hyps) * 100, 2),
               empty=sum(1 for h in hyps if not h))
    print(json.dumps(rec), flush=True)
    with open("results_rs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
