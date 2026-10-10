# Round 2 measurement workflow

## What changed

- All WER/CER, normalization, token alignment, English-word accuracy, and
  wrong-script scoring use one implementation in
  `modelcodesday1/evaluate_asr.py`. Terminal punctuation including danda is
  stripped on both sides; Devanagari nukta and vowel signs are retained.
- Aggregate WER/CER are pooled edit counts divided by pooled reference length.
  Aggregate RTF is total inference wall time divided by total audio duration.
- Wrong-script scoring aligns word sequences, then detects English references
  emitted in Devanagari and Hindi references emitted in Latin. English-word
  accuracy is exact-match count over aligned English reference tokens.
- Day 3 delegates WER/CER to the same scorer. Its long-form notebook command
  reads the long-form reference from the frozen manifest, not a Whisper output.
  Its saved VAD settings are threshold `0.40`, minimum speech `100 ms`, minimum
  silence `200 ms`, and padding `150 ms` (the actual command-cell values).
- SraVaani records actual single-utterance timings in the Round 2 runner. Its
  legacy batched runner now records batch wall time under a batch ID and does
  not pretend that dividing one batch's time evenly is per-file latency.
- Nemotron now pads the final partial chunk and emits a terminal flush/right-
  context feature chunk. Treat long-form results as unverified until rerun.

## In-scope model set

| Key | Model | Decoder | Vocabulary / script coverage | Hindi claim | Streaming / long input |
|---|---|---|---|---|---|
| `indicconformer` | ai4bharat/indic-conformer-600m-multilingual | CTC | Indic vocabulary; inspect loaded checkpoint | Yes | 45-second chunked evaluation |
| `sraavani` | ARTPARK-IISc/SraVaani-1.0 | Hybrid TDT-CTC | Indic vocabulary; inspect loaded checkpoint | Yes | 30-second chunks, 1-second overlap in this run |
| `nemotron` | nvidia/nemotron-3.5-asr-streaming-0.6b | RNNT | Multilingual; record emitted scripts | Yes, hi-IN | Native cache-aware streaming; default 6-token lookahead / 560 ms |
| `parakeet_v2` | nvidia/parakeet-tdt-0.6b-v2 | TDT | 1,024-piece English vocabulary; no Devanagari expected | No | NeMo offline call; full-attention inputs up to about 24 minutes |
| `parakeet_v3` | nvidia/parakeet-tdt-0.6b-v3 | TDT | 8,192-piece vocabulary for 25 European languages; Hindi absent | No | NeMo chunked-streaming path exists; this comparison uses offline inference; local attention supports longer inputs |
| `mms` | facebook/mms-1b-all | wav2vec2 CTC | Hindi adapter; runtime vocabulary size is recorded | Yes, via `hin` adapter | 30-second windows with 2-second overlap in this run |

The Parakeet v2/v3 rows are deliberate negative controls for Hindi. Preserve
their raw hypotheses and report whether the actual output is Latin, Devanagari,
English paraphrase, or empty. Do not infer that a poor Hindi score measures
general model quality; it also measures tokenizer/training-language coverage.

## Run

Use the already frozen manifest. Run on a supported Linux/CUDA host for NeMo
and the NVIDIA models; the repository's current Windows Python 3.8 environment
is not a supported NeMo setup. First install a current PyTorch/Transformers
environment and NeMo ASR in that host. Then run from the repository root:

```powershell
python .\modelcodesday1\run_frozen_benchmark.py --device cuda --threads 4
python .\modelcodesday1\build_comparison_table.py
```

For a development smoke test, select a single model and use a copied manifest
containing a few rows; do not change the committed frozen manifest. The runner
loads models one at a time, saves every raw hypothesis and per-record metric to
`modelbenchmark_day2/<model>_evaluation.json`, and records device, GPU, CPU
count, thread count, decoder config, reference source, and peak process memory.
The second command writes the combined model × slice table to
`modelbenchmark_day2/model_by_slice_results.csv`. Failed models are recorded in
`modelbenchmark_day2/run_errors.json` rather than silently omitted.

The runner records `beam_size` as `null` because these adapters use their
model-default/greedy decode APIs; do not compare its timings to a beam-search
run. Benchmarking uses batch size one to provide actual utterance latency and
avoid inventing per-item times from aggregate batches.

## Day 3 long-form VAD

Open `Day3/day3b_colab_execution_without_denoiser.ipynb` in Colab after
uploading the frozen manifest and `w5Jyq3XMbb3WwiKQ.wav`. The revised long-form
cell obtains `long_form_separate.reference`, prints its source/count and the
settings above, then runs raw and VAD stages with that same reference. The
recorded notebook output is not a new result until the cell is actually run on
the GPU. The old “VAD makes WER worse” claim is withdrawn until then.

## Round 1 files

`modelbenchmark_day1/` and the original Whisper runners are retained for audit
history only. They are out of scope for the new comparison and must not be
combined with Round 2 rows. Corrected model scores and a ranking are not yet
available until the new run completes on a common device and the references
have been human-audited.