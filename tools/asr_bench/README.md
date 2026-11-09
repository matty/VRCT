# ASR input benchmarks

Scripts behind `docs/transcription_noise_reduction_2026-11-08.md`. They run faster-whisper
with the same settings as `LocalWhisperProvider` on public speech mixed with noise or
background talkers, and print WER (English) / CER (Japanese).

Not part of the app. Use a separate Python 3.11 venv:

```bash
pip install faster-whisper==1.1.1 numpy==1.26.4 "pyarrow<17" soundfile soxr jiwer huggingface_hub requests
```

Run everything from a scratch directory (it writes `data/`, `bench.pkl` and `results*.jsonl` there):

```bash
python path/to/tools/asr_bench/prep.py 40
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base', local_dir='models/base')"
```

| Script | What it measures | Example |
|---|---|---|
| `bench.py` | high-pass / normalization / noise gate vs noise (`clean`, `pink@5`, `stationary@0`, `babble@5`, `keyboard@5`) | `python bench.py models/base ja_jsut stationary@5 none hpf_norm gate12` |
| `bench_near.py` | near-voice gate vs background talkers (`solo`, `far15x1`, `far10x4`, `shout10`, `alt12`) | `python bench_near.py models/base ja_reazon far15x1 base final near_m10` |
| `bench_rs.py` | 48 kHz -> 16 kHz resampling (`ratecv`, `soxr`, `fft`, `fir_ratecv`), JSUT at 48 kHz | `python bench_rs.py models/base hiss@20 ratecv fft` |

Sets: `en_libri`, `ja_jsut`, `ja_reazon` (40 utterances each). Differences of about +-2
points are within noise at that size. Each (set, condition, variant) takes 1-4 minutes on
one CPU core with `base`; run conditions in parallel with `xargs -P`.
