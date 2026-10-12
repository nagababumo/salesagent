
from __future__ import annotations

"""Evaluate NVIDIA Parakeet TDT models (v2 and v3) on evaluation manifest.

Optimized for GPU acceleration with batch-level processing to maximize throughput.
"""

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf
import torch

# Add the folder containing evaluate_asr.py to python path
evaluate_dir = Path(__file__).resolve().parent.parent.parent / "modelcodesday1"
sys.path.append(str(evaluate_dir))

from evaluate_asr import (
    aggregate_metrics,
    load_audio,
    make_telephony_audio,
    metric,
    NORMALIZATION_POLICY,
    resample,
    runtime_metadata,
    script_aware_metrics,
)

# ==============================================================================
# EMBEDDED MODULE: Parakeet TDT Helper (v2 & v3)
# ==============================================================================
class EmbeddedParakeetTDT:
    @staticmethod
    def build_model(model_id: str, device: str):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("Parakeet runs require NVIDIA NeMo ASR (nemo_toolkit[asr]).") from exc
        
        print(f"Loading {model_id} on {device}...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        return model.to(device).eval()

    @staticmethod
    def transcribe_batch(model, audio_list: list[np.ndarray], sample_rates: list[int], device: str, batch_size: int = 16) -> list[str]:
        """Transcribe a batch of audio arrays efficiently using temporary files and NeMo native batching."""
        wav_paths = []
        temp_handles = []
        
        try:
            for audio, sr in zip(audio_list, sample_rates):
                resampled_audio = resample(audio, sr, 16000).astype(np.float32)
                handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_handles.append(handle)
                wav_path = handle.name
                sf.write(wav_path, resampled_audio, 16000, subtype="PCM_16")
                wav_paths.append(wav_path)

            # Run batched GPU inference
            with torch.inference_mode():
                hypotheses = model.transcribe(wav_paths, batch_size=batch_size, return_hypotheses=True)

            if isinstance(hypotheses, tuple):
                hypotheses = hypotheses[0]

            results = []
            for hyp in hypotheses:
                text = str(getattr(hyp, "text", hyp)).strip()
                results.append(text)
            return results

        finally:
            for handle in temp_handles:
                try:
                    Path(handle.name).unlink(missing_ok=True)
                except Exception:
                    pass


# ==============================================================================
# MAIN BENCHMARK ENGINE
# ==============================================================================
MODEL_METADATA = {
    "parakeet_v2": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v2",
        "decoder": "TDT",
        "claims_hindi": "no (English optimized, record emissions)",
    },
    "parakeet_v3": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "decoder": "TDT",
        "claims_hindi": "multilingual support across 25 languages",
    },
}

def _torch_memory_reset(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def _peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / (1024 * 1024)

def _load_row_audio(row: dict[str, str]) -> tuple[np.ndarray, int]:
    audio, sample_rate = load_audio(Path(row["audio_path"]))
    start = float(row.get("start_seconds", 0.0))
    end = float(row.get("end_seconds", len(audio) / sample_rate))
    begin_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(len(audio), int(round(end * sample_rate)))
    audio = np.asarray(audio[begin_sample:end_sample], dtype=np.float32)
    
    aug = str(row.get("augmentation", "none")).lower()
    if aug != "none" and aug != "nan":
        seed_val = row.get("augmentation_seed", 0)
        seed = int(float(seed_val)) if str(seed_val).replace('.', '', 1).isdigit() else 0
        audio, sample_rate = make_telephony_audio(audio, sample_rate, seed)
    return audio, sample_rate


def evaluate_model(key: str, manifest_path: Path, output_path: Path, device: str, language: str, batch_size: int, threads: int | None) -> dict[str, Any]:
    if threads is not None:
        torch.set_num_threads(threads)
        
    model_id = MODEL_METADATA[key]["model_id"]
    model = EmbeddedParakeetTDT.build_model(model_id, device)
    
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    
    records_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_meta = runtime_metadata(beam_size=None)
    run_meta["device"] = str(device)
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Batch collection containers for high throughput processing
    batch_rows = []
    batch_audios = []
    batch_sample_rates = []

    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file:
        for row in manifest:
            try:
                audio, sample_rate = _load_row_audio(row)
                batch_rows.append(row)
                batch_audios.append(audio)
                batch_sample_rates.append(sample_rate)

                # Process once batch size is reached
                if len(batch_rows) >= batch_size:
                    _process_and_write_batch(model, batch_rows, batch_audios, batch_sample_rates, key, device, raw_file, records_by_slice)
                    batch_rows, batch_audios, batch_sample_rates = [], [], []

            except Exception as row_exc:
                print(f"[{key}] SKIPPED row {row.get('utterance_id')}: {row_exc}", file=sys.stderr)

        # Process remainder
        if batch_rows:
            _process_and_write_batch(model, batch_rows, batch_audios, batch_sample_rates, key, device, raw_file, records_by_slice)

    results: dict[str, Any] = {
        "model_key": key,
        **MODEL_METADATA[key],
        "manifest": str(manifest_path.resolve()),
        "raw_output_jsonl": str(raw_path.resolve()),
        "slices": {},
    }
    for slice_name, records in sorted(records_by_slice.items()):
        results["slices"][slice_name] = {
            "count": len(records),
            "aggregate": aggregate_metrics(records),
            "records": records,
        }
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _process_and_write_batch(model, rows, audios, sample_rates, key, device, raw_file, records_by_slice):
    _torch_memory_reset(device)
    started = time.perf_counter()
    
    # Run batched inference via helper
    hypotheses = EmbeddedParakeetTDT.transcribe_batch(model, audios, sample_rates, device, batch_size=len(audios))
    
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = _peak_memory_mb(device)
    per_item_elapsed = elapsed / len(rows)

    for row, audio, sr, hypothesis in zip(rows, audios, sample_rates, hypotheses):
        duration = len(audio) / sr if sr else 0.0
        record = {
            "utterance_id": row.get("utterance_id", ""),
            "audio": row.get("audio_path", ""),
            "slice": row.get("slice", "default"),
            "reference": row.get("reference", ""),
            "hypothesis": hypothesis,
            "duration_seconds": duration,
            "inference_seconds": per_item_elapsed,
            "rtf": per_item_elapsed / max(duration, 1e-9),
            "peak_process_memory_mb": peak_memory,
            "metrics": metric(row.get("reference", ""), hypothesis),
            "script_aware": script_aware_metrics(row.get("reference", ""), hypothesis),
        }
        raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        raw_file.flush()
        records_by_slice[row.get("slice", "default")].append(record)
        print(f"[{key}][{row.get('slice', 'default')}] {row.get('utterance_id')} -> {hypothesis[:30]}... ({per_item_elapsed:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/rework_resultsAndArtifacts/frozen_eval_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/rework_resultsAndArtifacts")
    parser.add_argument("--models", nargs="+", choices=["parakeet_v2", "parakeet_v3"], default=["parakeet_v2", "parakeet_v3"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--language", default="hi-IN")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch size for accelerated processing")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Evaluation manifest not found: {args.manifest}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    errors = {}
    for key in args.models:
        output_path = args.output_dir / f"{key}_evaluation.json"
        try:
            evaluate_model(key, args.manifest, output_path, device, args.language, args.batch_size, args.threads)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"[{key}] FAILED: {errors[key]}", file=sys.stderr)
            
    if errors:
        (args.output_dir / "run_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()