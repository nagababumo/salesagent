
from __future__ import annotations

"""Evaluate ARTPARK SraVaani-1.0 ASR model on evaluation manifest.

Single-file implementation containing the embedded SraVaani ASR adapter, chunking/merging logic, and benchmark runner.
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
from huggingface_hub import snapshot_download
from transformers import AutoModel

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
# EMBEDDED MODULE: SraVaani Official Pipeline Helper
# ==============================================================================
class EmbeddedSraVaani:
    @staticmethod
    def build_model(model_id: str, device: str):
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Downloading/loading SraVaani snapshot for {model_id} on {device}...")
        model_dir = snapshot_download(repo_id=model_id, token=hf_token)
        load_kwargs = {"trust_remote_code": True}
        if hf_token:
            load_kwargs["token"] = hf_token
        
        model = AutoModel.from_pretrained(model_dir, **load_kwargs)
        return model.to(device).eval()

    @staticmethod
    def extract_text(hypothesis) -> str:
        if hypothesis is None:
            return ""
        if isinstance(hypothesis, str):
            return hypothesis.strip()
        if hasattr(hypothesis, "text"):
            return str(hypothesis.text).strip()
        if isinstance(hypothesis, (list, tuple)) and hypothesis:
            return EmbeddedSraVaani.extract_text(hypothesis[0])
        return str(hypothesis).strip()

    @staticmethod
    def transcribe_batch(model, audio_list: list[np.ndarray], sample_rates: list[int], batch_size: int = 4) -> list[str]:
        if not audio_list:
            return []

        with tempfile.TemporaryDirectory(prefix="sravaani_run_") as temp_dir:
            temp_path = Path(temp_dir)
            wav_paths = []
            
            for idx, (audio, sr) in enumerate(zip(audio_list, sample_rates)):
                resampled = resample(audio, sr, 16000).astype(np.float32)
                p = temp_path / f"audio_{idx}.wav"
                sf.write(str(p), resampled, 16000, subtype="PCM_16")
                wav_paths.append(str(p))

            with torch.inference_mode():
                hypotheses = model.transcribe(
                    wav_paths,
                    batch_size=min(batch_size, len(wav_paths)),
                    return_hypotheses=True
                )

            if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
                hypotheses = hypotheses[0]
            if not isinstance(hypotheses, (list, tuple)):
                hypotheses = [hypotheses]

            return [EmbeddedSraVaani.extract_text(h) for h in hypotheses]


# ==============================================================================
# MAIN BENCHMARK ENGINE
# ==============================================================================
MODEL_METADATA = {
    "sraavani": {
        "model_id": "ARTPARK-IISc/SraVaani-1.0",
        "decoder": "hybrid TDT-CTC",
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

def _load_row_audio(row: dict[str, str]) -> tuple[np.ndarray, int]:
    loaded = load_audio(Path(row["audio_path"]))
    if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
        audio, sample_rate = loaded[0], loaded[1]
    else:
        raise ValueError("Unexpected return format from load_audio")

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
        
    model = EmbeddedSraVaani.build_model(MODEL_METADATA[key]["model_id"], device)
    
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    
    records_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_rows, batch_audios, batch_srs = [], [], []

    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file:
        for row in manifest:
            try:
                audio, sample_rate = _load_row_audio(row)
                batch_rows.append(row)
                batch_audios.append(audio)
                batch_srs.append(sample_rate)

                if len(batch_rows) >= batch_size:
                    _process_batch(model, batch_rows, batch_audios, batch_srs, key, device, raw_file, records_by_slice, batch_size)
                    batch_rows, batch_audios, batch_srs = [], [], []
            except Exception as row_exc:
                print(f"[{key}] SKIPPED row {row.get('utterance_id')}: {row_exc}", file=sys.stderr)

        if batch_rows:
            _process_batch(model, batch_rows, batch_audios, batch_srs, key, device, raw_file, records_by_slice, batch_size)

    results = {
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


def _process_batch(model, rows, audios, sample_rates, key, device, raw_file, records_by_slice, batch_size):
    _torch_memory_reset(device)
    started = time.perf_counter()
    
    hypotheses = EmbeddedSraVaani.transcribe_batch(model, audios, sample_rates, batch_size=batch_size)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = _peak_memory_mb(device)
    per_item = elapsed / len(rows)

    for row, audio, sr, hypothesis in zip(rows, audios, sample_rates, hypotheses):
        duration = len(audio) / sr if sr else 0.0
        record = {
            "utterance_id": row.get("utterance_id", ""),
            "audio": row.get("audio_path", ""),
            "slice": row.get("slice", "default"),
            "reference": row.get("reference", ""),
            "hypothesis": hypothesis,
            "duration_seconds": duration,
            "inference_seconds": per_item,
            "rtf": per_item / max(duration, 1e-9),
            "peak_process_memory_mb": peak_memory,
            "metrics": metric(row.get("reference", ""), hypothesis),
            "script_aware": script_aware_metrics(row.get("reference", ""), hypothesis),
        }
        raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        raw_file.flush()
        records_by_slice[row.get("slice", "default")].append(record)
        print(f"[{key}][{row.get('slice', 'default')}] {row.get('utterance_id')} -> {hypothesis[:30]}... ({per_item:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/rework_resultsAndArtifacts/frozen_eval_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default="C:/Users/nagab/Documents/salesagentProject/salesagent/modelbenchmark_day1/rework_resultsAndArtifacts")
    parser.add_argument("--models", nargs="+", choices=["sraavani"], default=["sraavani"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--language", default="hi")
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