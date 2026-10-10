from __future__ import annotations

"""Evaluate the six in-scope ASR models on evaluation/frozen_eval_manifest.csv.

Model downloads and GPU inference are deliberately opt-in via explicit CLI
invocation; this module preserves raw hypotheses and can be rerun independently
from scoring.
"""

import argparse
import csv
import gc
import json
import os
import re
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

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA = {
    "indicconformer": {
        "model_id": "ai4bharat/indic-conformer-600m-multilingual",
        "decoder": "CTC",
        "tokenizer_vocab_size": "checkpoint-defined; inspect model/tokenizer at runtime",
        "has_devanagari": "yes (Indic tokenizer)",
        "claims_hindi": "yes",
        "streaming": "chunked offline; 45-second windows",
        "long_audio": "45-second chunks, concatenated outputs",
    },
    "sraavani": {
        "model_id": "ARTPARK-IISc/SraVaani-1.0",
        "decoder": "hybrid TDT-CTC",
        "tokenizer_vocab_size": "checkpoint-defined; inspect model/tokenizer at runtime",
        "has_devanagari": "yes (Indic model)",
        "claims_hindi": "yes",
        "streaming": "chunked offline in this benchmark",
        "long_audio": "30-second windows with 1-second overlap",
    },
    "nemotron": {
        "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "decoder": "RNNT",
        "tokenizer_vocab_size": "checkpoint-defined; inspect processor tokenizer at runtime",
        "has_devanagari": "Hindi locale supported; Latin/Devanagari switching must be measured",
        "claims_hindi": "yes (hi-IN transcription-ready)",
        "streaming": "native cache-aware streaming; default lookahead 6 tokens / 560 ms",
        "long_audio": "streaming feature chunks with explicit terminal flush",
    },
    "parakeet_v2": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v2",
        "decoder": "TDT",
        "tokenizer_vocab_size": "1024 (English vocabulary per evaluation brief)",
        "has_devanagari": "no (expected; record actual emissions)",
        "claims_hindi": "no; English-only release",
        "streaming": "offline transcription in this benchmark; NeMo chunked streaming is separate",
        "long_audio": "full-attention input supported to about 24 minutes per model card",
    },
    "parakeet_v3": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "decoder": "TDT",
        "tokenizer_vocab_size": "8192 (multilingual vocabulary per evaluation brief)",
        "has_devanagari": "no (Hindi absent per evaluation brief)",
        "claims_hindi": "no; not tuned for code-switching",
        "streaming": "chunked streaming supported via NeMo RNNT script; this benchmark uses offline transcription",
        "long_audio": "full attention to about 24 minutes; local attention for longer recordings",
    },
    "mms": {
        "model_id": "facebook/mms-1b-all",
        "decoder": "wav2vec2 CTC",
        "tokenizer_vocab_size": "Hindi adapter tokenizer; record runtime vocabulary size",
        "has_devanagari": "Hindi adapter-dependent",
        "claims_hindi": "yes (Hindi adapter)",
        "streaming": "not native in this benchmark; offline CTC",
        "long_audio": "40-second windows with overlap and text merge",
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


def _model_name(model: Any, key: str) -> str:
    return str(getattr(model, "model_id", MODEL_METADATA[key]["model_id"]))


def _tokenizer_vocab_size(model: Any) -> int | str:
    candidates = [
        getattr(model, "tokenizer", None),
        getattr(getattr(model, "processor", None), "tokenizer", None),
    ]
    for tokenizer in candidates:
        if tokenizer is None:
            continue
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size
        try:
            return len(tokenizer)
        except (TypeError, AttributeError):
            pass
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    return vocab_size if isinstance(vocab_size, int) else "checkpoint-defined; runtime API did not expose size"


def _make_adapter(key: str, device: str, language: str, lookahead_tokens: int):
    """Load one model; imports model-specific heavyweight packages lazily."""
    if key == "indicconformer":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import indicconfor

        indicconfor.DEVICE = torch.device(device)
        model = indicconfor.build_model(MODEL_METADATA[key]["model_id"])
        def transcribe(audio: np.ndarray, sample_rate: int) -> str:
            audio = resample(audio, sample_rate, 16000)
            return indicconfor.transcribe_long_audio(model, audio, language, 45.0)
        return model, transcribe, {
            "sample_rate_hz": 16000, "batch_size": 1,
            "tokenizer_vocab_size": _tokenizer_vocab_size(model),
        }

    if key == "sraavani":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sraavani

        sraavani.DEVICE = torch.device(device)
        model = sraavani.build_model(MODEL_METADATA[key]["model_id"])
        def transcribe(audio: np.ndarray, sample_rate: int) -> str:
            audio = resample(audio, sample_rate, 16000)
            hypotheses, _, _ = sraavani.transcribe_batch(
                model=model,
                audio_list=[audio],
                model_batch_size=1,
                chunk_seconds=30.0,
                overlap_seconds=1.0,
                sample_rate=16000,
            )
            return hypotheses[0] if hypotheses else ""
        return model, transcribe, {
            "sample_rate_hz": 16000, "batch_size": 1,
            "chunk_seconds": 30.0, "overlap_seconds": 1.0,
            "tokenizer_vocab_size": _tokenizer_vocab_size(model),
        }

    if key == "nemotron":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from transcriber_nemotron import NemotronTranscriber

        model = NemotronTranscriber(
            model_id=MODEL_METADATA[key]["model_id"],
            language=language,
            lookahead_tokens=lookahead_tokens,
            device=device,
        )
        def transcribe(audio: np.ndarray, sample_rate: int) -> str:
            return model.transcribe(audio, sample_rate)[0]
        return model, transcribe, {
            "sample_rate_hz": model.sample_rate,
            "lookahead_tokens": lookahead_tokens,
            "streaming_latency_ms": getattr(model.processor, "streaming_latency_ms", None),
            "tokenizer_vocab_size": _tokenizer_vocab_size(model),
        }

    if key in {"parakeet_v2", "parakeet_v3"}:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("Parakeet runs require NVIDIA NeMo ASR (nemo_toolkit[asr]).") from exc
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_METADATA[key]["model_id"])
        model = model.to(device).eval()
        def transcribe(audio: np.ndarray, sample_rate: int) -> str:
            audio = resample(audio, sample_rate, 16000).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
            try:
                sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
                hypotheses = model.transcribe(
                    [str(wav_path)], batch_size=1, return_hypotheses=True
                )
            finally:
                wav_path.unlink(missing_ok=True)
            if isinstance(hypotheses, tuple):
                hypotheses = hypotheses[0]
            first = hypotheses[0] if hypotheses else ""
            return str(getattr(first, "text", first)).strip()
        return model, transcribe, {
            "sample_rate_hz": 16000, "batch_size": 1,
            "tokenizer_vocab_size": _tokenizer_vocab_size(model),
        }

    if key == "mms":
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        model_id = MODEL_METADATA[key]["model_id"]
        processor = AutoProcessor.from_pretrained(model_id, target_lang="hin")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang="hin", ignore_mismatched_sizes=True
        ).to(device).eval()
        chunk_seconds = 30.0
        overlap_seconds = 2.0
        sample_rate = int(processor.feature_extractor.sampling_rate)
        def transcribe(audio: np.ndarray, source_rate: int) -> str:
            audio = resample(audio, source_rate, sample_rate)
            chunk_size = int(chunk_seconds * sample_rate)
            step = int((chunk_seconds - overlap_seconds) * sample_rate)
            texts = []
            for start in range(0, len(audio), step):
                chunk = audio[start:start + chunk_size]
                if not len(chunk):
                    continue
                inputs = processor(
                    chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True
                )
                inputs = {name: value.to(device) for name, value in inputs.items()}
                with torch.inference_mode():
                    logits = model(**inputs).logits
                token_ids = torch.argmax(logits, dim=-1)
                texts.append(processor.batch_decode(token_ids)[0].strip())
                if start + chunk_size >= len(audio):
                    break
            return _merge_overlap_texts(texts)
        return model, transcribe, {
            "sample_rate_hz": sample_rate,
            "tokenizer_vocab_size": len(processor.tokenizer),
            "batch_size": 1,
            "adapter": "hin",
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
        }

    raise ValueError(f"Unknown model key: {key}")


def _merge_overlap_texts(texts: list[str], max_overlap_words: int = 12) -> str:
    merged: list[str] = []
    for text in texts:
        current = text.split()
        overlap = 0
        for size in range(min(max_overlap_words, len(merged), len(current)), 0, -1):
            if [word.casefold() for word in merged[-size:]] == [
                word.casefold() for word in current[:size]
            ]:
                overlap = size
                break
        merged.extend(current[overlap:])
    return " ".join(merged).strip()


def _load_row_audio(row: dict[str, str]) -> tuple[np.ndarray, int]:
    audio, sample_rate = load_audio(Path(row["audio_path"]))
    start = float(row["start_seconds"])
    end = float(row["end_seconds"])
    begin_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(len(audio), int(round(end * sample_rate)))
    audio = np.asarray(audio[begin_sample:end_sample], dtype=np.float32)
    if row["augmentation"] != "none":
        seed = int(row["augmentation_seed"])
        audio, sample_rate = make_telephony_audio(audio, sample_rate, seed)
    return audio, sample_rate


def evaluate_model(
    key: str,
    manifest_path: Path,
    output_path: Path,
    device: str,
    language: str,
    lookahead_tokens: int,
    threads: int | None,
) -> dict[str, Any]:
    if threads is not None:
        torch.set_num_threads(threads)
    model, transcribe, config = _make_adapter(key, device, language, lookahead_tokens)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    records_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_meta = runtime_metadata(beam_size=None)
    run_meta["device"] = str(device)
    if device == "cpu":
        run_meta["gpu_name"] = None
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file:
        for row in manifest:
            audio, sample_rate = _load_row_audio(row)
            duration = len(audio) / sample_rate if sample_rate else 0.0
            _torch_memory_reset(device)
            started = time.perf_counter()
            hypothesis = transcribe(audio, sample_rate)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_memory = _peak_memory_mb(device)
            script_metrics = script_aware_metrics(row["reference"], hypothesis)
            record = {
                "utterance_id": row["utterance_id"],
                "audio": row["audio_path"],
                "recording_id": row["recording_id"],
                "recording_group_id": row["recording_group_id"],
                "speaker_id": row["speaker_id"],
                "slice": row["slice"],
                "reference": row["reference"],
                "reference_source": row["reference_source"],
                "reference_review_status": row["reference_review_status"],
                "hypothesis": hypothesis,
                "duration_seconds": duration,
                "inference_seconds": elapsed,
                "rtf": elapsed / max(duration, 1e-9),
                "peak_process_memory_mb": peak_memory,
                "device": run_meta["device"],
                "gpu_name": run_meta["gpu_name"],
                "threads": run_meta["threads"],
                "cpu_count": run_meta["cpu_count"],
                "beam_size": None,
                "metrics": metric(row["reference"], hypothesis),
                "script_aware": script_metrics,
                "augmentation": row["augmentation"],
                "augmentation_seed": row["augmentation_seed"],
            }
            raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_file.flush()
            records_by_slice[row["slice"]].append(record)
            print(f"[{key}][{row['slice']}] {row['utterance_id']} {elapsed:.3f}s")

    results: dict[str, Any] = {
        "model_key": key,
        **MODEL_METADATA[key],
        "device": run_meta["device"],
        "gpu_name": run_meta["gpu_name"],
        "threads": run_meta["threads"],
        "cpu_count": run_meta["cpu_count"],
        "beam_size": None,
        "manifest": str(manifest_path.resolve()),
        "raw_output_jsonl": str(raw_path.resolve()),
        "manifest_version": "saleagent-stt-eval-v1",
        "normalization": NORMALIZATION_POLICY,
        "model_config": config,
        "tokenizer_vocab_size": config.get(
            "tokenizer_vocab_size", MODEL_METADATA[key]["tokenizer_vocab_size"]
        ),
        "slices": {},
    }
    for slice_name, records in sorted(records_by_slice.items()):
        results["slices"][slice_name] = {
            "count": len(records),
            "aggregate": aggregate_metrics(records),
            "records": records,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "evaluation" / "frozen_eval_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "modelbenchmark_day2")
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
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")

    errors = {}
    for key in args.models:
        output_path = args.output_dir / f"{key}_evaluation.json"
        try:
            evaluate_model(
                key, args.manifest, output_path, device, args.language,
                args.lookahead_tokens, args.threads,
            )
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"[{key}] FAILED: {errors[key]}", file=sys.stderr)
    if errors:
        (args.output_dir / "run_errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"{len(errors)} model(s) failed; see {args.output_dir / 'run_errors.json'}")


if __name__ == "__main__":
    main()
