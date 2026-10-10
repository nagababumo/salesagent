# Frozen STT evaluation set

This directory contains a deterministic test manifest assembled from the
Gram Vaani and MUCS datasets supplied for the measurement rework. Audio is
referenced in place; it is not copied into this repository.

## Artifacts

- `frozen_eval_manifest.csv` — one row per evaluated utterance, with transcript,
  provenance, recording/speaker grouping, time offsets, sample rate, duration,
  normalized reference word count, and augmentation metadata.
- `frozen_eval_manifest_summary.json` — counts and aggregate metadata by slice,
  deterministic MUCS speaker assignments, selection rules, and limitations.
- `../scripts/build_frozen_eval_manifest.py` — reproducible manifest builder.

## Slices

| Slice | N | Source | Selection |
|---|---:|---|---|
| `clean_hindi` | 100 | Gram Vaani `GV_Eval_3h` | Hindi-script references with no Latin letters; at least 15 words; 3–30 seconds. |
| `short_turn` | 50 | Gram Vaani `GV_Eval_3h` | At most 10 reference words and at most 8 seconds. |
| `telephony_from_clean_hindi` | 100 | Same utterances as `clean_hindi` | Deterministic 8 kHz mu-law/noise augmentation; preserve each clean reference and record the seed. |
| `code_mixed_mucs` | 100 | MUCS Hindi-English test release | Reference contains both Latin and Devanagari; 2–25 words and 1–20 seconds; only held-out speaker and recording groups. |
| `long_form_separate` | 1 | Existing 413-second MUCS recording in `data/` | Separate long-form score; do not pool into the short-form aggregate. |

The telephony rows intentionally reuse their clean source audio and reference;
the augmentation recipe and seed indicate how to create the degraded input at
evaluation time. MUCS utterances point to a source recording plus start/end
offsets, so crop those intervals before inference.

## Split and selection

Selection is deterministic (`saleagent-stt-eval-v1`) and uses SHA-256 ordering.
MUCS speakers are assigned as whole groups to train/dev/test before utterances
are selected; the test portion is about 20% of speakers. No MUCS recording is
split across those groups. Speaker `103085` is reserved for test because the
separate long-form recording belongs to that speaker. Only held-out test
examples appear in the CSV.
Within Gram Vaani, clean and short-turn slices have disjoint recording groups,
and known quality flags (`inaudible`, `audio_break`, `audio_jump`) are excluded.
The previous `01-00159` benchmark recording is excluded from the new Gram
Vaani slices.

## Known limitations — do not overstate these results

1. The Gram Vaani release has no speaker-ID field. Its clean and short-turn
   examples are recording-group-disjoint, but speaker disjointness cannot be
   verified from the supplied metadata. Keep this source evaluation-only until
   speaker identities are available; do not use these clips for training.
2. References are dataset-provided, not yet manually checked against audio.
   Audit them before publishing final scores, especially MUCS script choices
   and the concatenated long-form reference. Rows explicitly mark this status.
3. The quality-filtered Gram Vaani release has fewer than 50 samples under a
   strict <=5-word definition. The operational short-turn slice therefore uses
   <=10 words and <=8 seconds; it is not a pure one-word confirmation set.
4. The short-form strata come from different sources (Gram Vaani and MUCS).
   Report source alongside the slice and avoid treating between-source
   differences as a controlled acoustic comparison. Clean vs telephony is the
   paired comparison on the same Gram Vaani references.

## Rebuild

From the repository root in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_frozen_eval_manifest.py
```

Optional `--mucs-root`, `--gv-root`, `--workspace`, and `--output-dir`
arguments override the local dataset locations. The default paths match the
provided Windows folders. Rebuilding overwrites only the generated manifest
and summary in this directory. Keep the CSV unchanged once model evaluation
begins; change the seed/version if the evaluation set must be revised.