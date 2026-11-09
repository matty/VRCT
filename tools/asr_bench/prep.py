"""Build the benchmark set: clean utterances + noise clips, all 16 kHz mono float32.

Downloads the source parquet files from the Hugging Face Hub into ./data
(about 1.1 GB) and writes ./bench.pkl. See README.md.
"""
import io, pickle, sys
from huggingface_hub import hf_hub_download
import numpy as np, soundfile as sf, soxr, pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(1234)


def decode(cell):
    a, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 16000:
        a = soxr.resample(a, sr, 16000)
    return a.astype(np.float32)


def load(path, text_col, n, min_s=1.5, max_s=12.0, skip=0):
    t = pq.read_table(path).to_pylist()
    out = []
    for row in t[skip:]:
        a = decode(row["audio"])
        if not (min_s <= a.size / 16000 <= max_s):
            continue
        out.append((a, row[text_col]))
        if len(out) >= n * 2:
            break
    return out


for repo, filename in [
    ("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet"),
    ("japanese-asr/ja_asr.reazonspeech_test", "data/test-00000-of-00002.parquet"),
    ("japanese-asr/ja_asr.jsut_basic5000", "data/test-00000-of-00004.parquet"),
    ("ashraq/esc50", "data/train-00000-of-00002-2f1ab7b824ec751f.parquet"),
]:
    hf_hub_download(repo, filename, repo_type="dataset", local_dir="data")

sets = {
    "en_libri": load("data/clean/validation-00000-of-00001.parquet", "text", N),
    "ja_jsut": load("data/data/test-00000-of-00004.parquet", "transcription", N),
    "ja_reazon": load("data/data/test-00000-of-00002.parquet", "transcription", N),
}
esc = pq.read_table("data/data/train-00000-of-00002-2f1ab7b824ec751f.parquet").to_pylist()
noise = {"stationary": [], "keyboard": []}
for row in esc:
    if row["category"] in ("vacuum_cleaner", "airplane", "engine", "rain", "wind", "washing_machine", "train", "helicopter"):
        noise["stationary"].append(decode(row["audio"]))
    elif row["category"] in ("keyboard_typing", "mouse_click"):
        noise["keyboard"].append(decode(row["audio"]))
# Keep only noise clips that are not mostly silent.
for k in noise:
    noise[k] = [a for a in noise[k] if np.sqrt(np.mean(a ** 2)) > 1e-3]
out = {}
for name, items in sets.items():
    # First N are test utterances, the rest are babble sources.
    out[name] = {"test": items[:N], "babble": [a for a, _ in items[N:]]}
    print(name, len(items[:N]), "babble", len(out[name]["babble"]))
print({k: len(v) for k, v in noise.items()})
pickle.dump({"sets": out, "noise": noise}, open("bench.pkl", "wb"))
