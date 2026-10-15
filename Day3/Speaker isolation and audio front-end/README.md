# Modular Speaker-Isolation + MMS Benchmark

This task folder runs front-end experiments with the Hindi adapter of
`facebook/mms-1b-all`; Whisper is not used. The MMS adapter is the accessible
Hindi-capable model from the six-model set documented in the repository README.
Existing benchmark results report 69.57% WER on `short_turn`, so treat its
absolute accuracy as a limitation and focus this experiment on paired front-end
changes. IndicConformer and SraVaani scored better in that README but require
Hugging Face gated-model access.

## Environment

The task uses its own `.venv` directory. From the repository root in
PowerShell:

```powershell
python -m venv ".\Day3\Speaker isolation and audio front-end\.venv"
& ".\Day3\Speaker isolation and audio front-end\.venv\Scripts\python.exe" -m pip install -r ".\Day3\Speaker isolation and audio front-end\requirements.txt"
```

The pinned CUDA build targets the RTX 3050 / CUDA 12.4 setup. Model weights are
downloaded from Hugging Face on first run. Pyannote diarization additionally
requires accepted model terms and a locally configured `HF_TOKEN`.

## Run

Run one short-turn smoke test:

```powershell
& ".\Day3\Speaker isolation and audio front-end\.venv\Scripts\python.exe" ".\Day3\Speaker isolation and audio front-end\manifest_frontend_eval.py" --mode short-turn --limit 3
```

Run all 50 `short_turn` manifest records plus the synthetic stereo experiment:

```powershell
& ".\Day3\Speaker isolation and audio front-end\.venv\Scripts\python.exe" ".\Day3\Speaker isolation and audio front-end\manifest_frontend_eval.py"
```

The VAD sweep compares raw MMS with four Silero configurations and records
pooled WER/CER, total latency, processed duration, and the number of utterances
reduced to zero samples. The stereo test puts two similarly sized manifest
recordings on separate channels, then compares raw downmix, right-channel, and
right-channel+VAD against the right-channel reference. Diarization stages are
added only when `HF_TOKEN` is available; speaker selection uses the longest
detected duration rather than a hard-coded pyannote label.

Results are written under `evaluation_results/` as JSON and CSV files, including
the exact synthetic stereo WAV used in the comparison.
