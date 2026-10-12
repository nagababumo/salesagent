
from __future__ import annotations

"""Evaluate NVIDIA Nemotron ASR on evaluation manifest.

Single-file implementation containing the embedded Nemotron ASR adapter and benchmark runner.
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
import sys
from pathlib import Path

# Add the folder containing evaluate_asr.py to python path
evaluate_dir = (
    Path(__file__).resolve().parent.parent.parent / "modelcodesday1"
)
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
# EMBEDDED MODULE: Nemotron Helper
# ==============================================================================
class EmbeddedNemotron:
    @staticmethod
    def build_model(model_id: str, device: str):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("Nemotron runs require NVIDIA NeMo ASR (nemo_toolkit[asr]).") from exc
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        return model.to(device).eval()

    @staticmethod
    def transcribe(model, audio: np.ndarray, sample_rate: int, language: str, device: str) -> str:
        audio = resample(audio, sample_rate, 16000).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
            cfg = model.get_transcribe_config()
            cfg.target_lang = language if language else "hi-IN"
            hypotheses = model.transcribe([str(wav_path)], override_config=cfg)
        finally:
            wav_path.unlink(missing_ok=True)
            
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        first = hypotheses[0] if hypotheses else ""
        return str(getattr(first, "text", first)).strip()


# ==============================================================================
# MAIN BENCHMARK ENGINE
# ==============================================================================
MODEL_METADATA = {
    "nemotron": {
        "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "decoder": "RNNT",
        "has_devanagari": "Hindi locale supported",
        "claims_hindi": "yes",
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

def _make_adapter(key: str, device: str, language: str, lookahead_tokens: int):
    if key == "nemotron":
        model = EmbeddedNemotron.build_model(MODEL_METADATA[key]["model_id"], device)
        def transcribe(audio: np.ndarray, sample_rate: int) -> str:
            return EmbeddedNemotron.transcribe(model, audio, sample_rate, language, device)
        return model, transcribe, {"sample_rate_hz": 16000}

    raise ValueError(f"Unknown model key: {key}")


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


def evaluate_model(key: str, manifest_path: Path, output_path: Path, device: str, language: str, lookahead_tokens: int, threads: int | None) -> dict[str, Any]:
    if threads is not None:
        torch.set_num_threads(threads)
    model, transcribe, config = _make_adapter(key, device, language, lookahead_tokens)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    
    records_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_meta = runtime_metadata(beam_size=None)
    run_meta["device"] = str(device)
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file:
        for row in manifest:
            try:
                audio, sample_rate = _load_row_audio(row)
                duration = len(audio) / sample_rate if sample_rate else 0.0
                _torch_memory_reset(device)
                started = time.perf_counter()
                hypothesis = transcribe(audio, sample_rate)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak_memory = _peak_memory_mb(device)
                
                record = {
                    "utterance_id": row.get("utterance_id", ""),
                    "audio": row.get("audio_path", ""),
                    "slice": row.get("slice", "default"),
                    "reference": row.get("reference", ""),
                    "hypothesis": hypothesis,
                    "duration_seconds": duration,
                    "inference_seconds": elapsed,
                    "rtf": elapsed / max(duration, 1e-9),
                    "peak_process_memory_mb": peak_memory,
                    "metrics": metric(row.get("reference", ""), hypothesis),
                    "script_aware": script_aware_metrics(row.get("reference", ""), hypothesis),
                }
                raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_file.flush()
                records_by_slice[row.get("slice", "default")].append(record)
                print(f"[{key}][{row.get('slice', 'default')}] {row.get('utterance_id')} {elapsed:.3f}s")
            except Exception as row_exc:
                print(f"[{key}] SKIPPED row {row.get('utterance_id')}: {row_exc}", file=sys.stderr)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/rework_resultsAndArtifacts/frozen_eval_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_METADATA), default=sorted(MODEL_METADATA))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--language", default="hi-IN")
    parser.add_argument("--lookahead-tokens", type=int, default=6)
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
            evaluate_model(key, args.manifest, output_path, device, args.language, args.lookahead_tokens, args.threads)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"[{key}] FAILED: {errors[key]}", file=sys.stderr)
            
    if errors:
        (args.output_dir / "run_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()