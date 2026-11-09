"""Near talker vs background talkers.

usage: bench_near.py MODEL_DIR SET COND VARIANTS...
COND: solo | far{D}x{N}  (far talkers D dB below the near talker, N talkers)
Sequence: near clips (far chatter 1.5 s before, during, 1.0 s after) with a
far-only clip after every 2nd near clip. Near-voice gate state carries over.
"""
from pathlib import Path
import json, pickle, re, sys, unicodedata, zlib
import numpy as np, jiwer
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src-python"))
from models.transcription import audio_preprocess as ap  # noqa: E402
from models.transcription import near_voice_gate as nvg  # noqa: E402

model_dir, set_name, cond = sys.argv[1:4]
variants = sys.argv[4:]
bench = pickle.load(open("bench.pkl", "rb"))
data = bench["sets"][set_name]
lang = set_name[:2]
rng = np.random.default_rng(zlib.crc32(f"near|{set_name}|{cond}".encode()))
SR = 16000
NEAR_DBFS = -26.0


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def active_rms(a):
    f = a[: a.size // 512 * 512].reshape(-1, 512)
    r = np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1))
    return float(np.sqrt(np.mean(r[r > r.max() * 0.1] ** 2)) + 1e-12)


def far_stream(n, talkers):
    out = np.zeros(n, np.float32)
    for _ in range(talkers):
        parts, total = [], 0
        while total < n:
            u = data["babble"][rng.integers(len(data["babble"]))]
            u = u / active_rms(u)
            parts += [u, np.zeros(int(SR * rng.uniform(0.1, 0.4)), np.float32)]
            total += u.size + parts[-1].size
        s = np.concatenate(parts)
        st = rng.integers(0, s.size - n + 1)
        out += s[st:st + n]
    return out / max(active_rms(out), 1e-9)


def pink(n):
    s = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / SR)
    s[1:] /= np.sqrt(f[1:]); s[0] = 0; p = np.fft.irfft(s, n)
    return p / rms(p)


ALT = None
if cond == "solo":
    D, N = None, 0
elif cond.startswith("shout"):
    # One talker; every 5th clip is SHOUT dB louder (laughing / shouting).
    D, N, ALT = None, 0, -float(cond[5:])
elif cond.startswith("alt"):
    # Two near talkers whose mic levels differ: every 2nd clip is ALT dB quieter.
    D, N, ALT = None, 0, float(cond[3:])
else:
    m = re.fullmatch(r"far(\d+)x(\d+)", cond); D, N = float(m.group(1)), int(m.group(2))
near_amp = 10 ** (NEAR_DBFS / 20)
items = []  # (kind, audio, ref)
for i, (a, ref) in enumerate(data["test"]):
    a = a / active_rms(a) * near_amp
    if ALT is not None and ((ALT > 0 and i % 2 == 1) or (ALT < 0 and i % 5 == 0)):
        a = a * 10 ** (-ALT / 20)
    if N:
        lead, tail = int(1.5 * SR), int(1.0 * SR)
        x = np.concatenate([np.zeros(lead, np.float32), a, np.zeros(tail, np.float32)])
        x = x + far_stream(x.size, N) * near_amp * 10 ** (-D / 20)
    else:
        x = a.copy()
    x = x + pink(x.size) * 10 ** (-60 / 20)
    items.append(("near", x.astype(np.float32), ref))
    if N and i % 2 == 1:
        n = int(SR * rng.uniform(2.0, 4.0))
        f = far_stream(n, N) * near_amp * 10 ** (-D / 20) + pink(n) * 10 ** (-60 / 20)
        items.append(("far", f.astype(np.float32), ""))

VARIANTS = {"base": None}


TF = {}


def tf_mask(y, margin_db, atten_db=12.0):
    """Within the clip, attenuate STFT cells far below the near talker's spectrum.

    Near spectrum per bin = 90th percentile over frames whose level is within
    6 dB of the clip's top level (the near talker's loud frames).
    """
    n = 512; hop = 128; w = np.hanning(n + 1)[:-1].astype(np.float32)
    spec = ap._stft(y, n, hop, w); mag = np.abs(spec)
    fl = 10 * np.log10(np.mean(mag ** 2, axis=1) + 1e-20)
    top = np.mean(np.sort(fl)[-max(1, fl.size // 10):])
    near_frames = mag[fl >= top - 6]
    ref = np.quantile(near_frames, 0.9, axis=0) + 1e-10
    rel = 20 * np.log10(mag / ref + 1e-12)
    m = np.clip((rel + margin_db) / 6.0 + 1.0, 0.0, 1.0)  # 0 below -margin-6, 1 above -margin
    m = ap._smooth(m, 1, 1)
    floor = 10 ** (-atten_db / 20)
    return ap._istft(spec * (floor + (1 - floor) * m), n, hop, w, y.size)


def parse_variant(v):
    if v == "final":
        return nvg.NearVoiceSettings()
    if v.startswith("fin_"):
        kw = {}
        for part in v[4:].split("_"):
            key, val = part[0], float(part[1:])
            kw[{"f": "frame_margin_db", "h": "hold_ms", "a": "attenuation_db"}[key]] = val
        return nvg.NearVoiceSettings(**kw)
    if v.startswith("med_m"):
        return nvg.NearVoiceSettings(margin_db=float(v[5:]))

    # near_m{margin}[_d{window sec}][_h{hold}][_tf{db}]
    m = re.fullmatch(r"near_m([\d.]+)(?:_d([\d.]+))?(?:_h([\d.]+))?(?:_tf([\d.]+))?", v)
    if m.group(4): TF[v] = float(m.group(4))
    kw = {"margin_db": float(m.group(1))}
    if m.group(2): kw["window_sec"] = float(m.group(2))
    if m.group(3): kw["hold_ms"] = float(m.group(3))
    return nvg.NearVoiceSettings(**kw)


for v in variants:
    if v != "base":
        VARIANTS[v] = parse_variant(v)
PRE = ap.PreprocessSettings()  # hpf + normalize (current app default)


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    if lang == "ja":
        return re.sub(r"[\s\W_]+", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", s)).strip()


model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
for variant in variants:
    clock = [0.0]
    cfg = VARIANTS[variant]
    if cfg is None:
        gate = None
    else:
        gate = nvg.NearVoiceGate(cfg, clock=lambda: clock[0])
    refs, hyps, leaks, leak_chars, skipped_near = [], [], 0, 0, 0
    for kind, x, ref in items:
        clock[0] += x.size / SR + 1.0
        y = ap.spectral_filter(x, SR, PRE)
        if gate is not None:
            y = gate.process(y, SR)
            if y is not None and variant in TF:
                y = tf_mask(y, TF[variant])
        if y is None:
            text = ""
            if kind == "near":
                skipped_near += 1
        else:
            y = np.clip(ap.normalize_level(y, SR, PRE), -1, 1)
            y = np.concatenate([np.zeros(4800, np.float32), y, np.zeros(8000, np.float32)])
            segs, _ = model.transcribe(y, beam_size=5, temperature=0.0, log_prob_threshold=-0.8,
                                       no_speech_threshold=0.6, language=lang, without_timestamps=True)
            text = "".join(s.text for s in segs if not (s.avg_logprob < -0.8 or s.no_speech_prob > 0.6))
        if kind == "near":
            refs.append(norm_text(ref)); hyps.append(norm_text(text))
        elif norm_text(text):
            leaks += 1; leak_chars += len(norm_text(text))
    err = jiwer.cer(refs, hyps) if lang == "ja" else jiwer.wer(refs, hyps)
    n_far = sum(1 for k, _, _ in items if k == "far")
    rec = dict(model=model_dir.split("/")[-1], set=set_name, cond=cond, variant=variant, err=round(err * 100, 2),
               empty=sum(1 for h in hyps if not h), skipped_near=skipped_near, leaks=leaks, n_far=n_far,
               leak_chars=leak_chars)
    print(json.dumps(rec), flush=True)
    with open("results_near.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
