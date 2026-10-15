from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "modelcodesday1"))

from audio_utils import ensure_16k, load_audio, save_wav, to_mono
from channel_separator import ChannelSeparator
from config import DIARIZATION_MODEL
from denoiser import DeepFilterDenoiser
from evaluate_asr import aggregate_metrics, script_aware_metrics
from experiment_runner import ExperimentRunner
from indicconformer_asr import IndicConformerASR
from mms_asr import MMSASR
from sraavani_asr import SraVaaniASR
from vad_processor import SileroVAD


VAD_SETTINGS = [
    ("vad_040_080_120_100", 0.40, 80, 120, 100),
    ("vad_040_100_200_150", 0.40, 100, 200, 150),
    ("vad_030_100_300_200", 0.30, 100, 300, 200),
    ("vad_050_250_100_030", 0.50, 250, 100, 30),
]
STEREO_LEFT_ID = "01-03791-01"
STEREO_RIGHT_ID = "01-07043-02"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_row_audio(row: dict[str, str]) -> tuple[np.ndarray, int]:
    audio, sample_rate = load_audio(Path(row["audio_path"]))
    start = int(round(float(row.get("start_seconds") or 0) * sample_rate))
    end_seconds = float(row.get("end_seconds") or len(audio) / sample_rate)
    end = min(len(audio), int(round(end_seconds * sample_rate)))
    audio = np.asarray(audio[max(start, 0):end], dtype=np.float32)
    audio = to_mono(audio)
    return ensure_16k(audio, sample_rate)


def _run_short_turn_vad(
    rows: list[dict[str, str]],
    asr: MMSASR,
    output_dir: Path,
) -> dict:
    vad = SileroVAD()
    records_by_technique: dict[str, list[dict]] = {"raw": []}
    records_by_technique.update({setting[0]: [] for setting in VAD_SETTINGS})

    if rows:
        warmup_audio, warmup_sr = _load_row_audio(rows[0])
        asr.transcribe(warmup_audio, warmup_sr)

    for row in rows:
        audio, sample_rate = _load_row_audio(row)
        variants = [("raw", audio, 0.0, 0, None)]

        for name, threshold, min_speech, min_silence, padding in VAD_SETTINGS:
            vad.threshold = threshold
            vad.min_speech_duration_ms = min_speech
            vad.min_silence_duration_ms = min_silence
            vad.speech_pad_ms = padding
            started = time.perf_counter()
            speech, timestamps = vad.process(audio, sample_rate)
            preprocess_seconds = time.perf_counter() - started
            variants.append((name, speech, preprocess_seconds, len(timestamps), {
                "threshold": threshold,
                "min_speech_duration_ms": min_speech,
                "min_silence_duration_ms": min_silence,
                "speech_pad_ms": padding,
            }))

        input_duration = len(audio) / sample_rate
        for name, processed, preprocess_seconds, segment_count, vad_config in variants:
            hypothesis, asr_seconds, peak_memory = asr.transcribe(processed, sample_rate)
            record = {
                "utterance_id": row["utterance_id"],
                "slice": row["slice"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "duration_seconds": input_duration,
                "processed_duration_seconds": len(processed) / sample_rate,
                "preprocessing_seconds": preprocess_seconds,
                "inference_seconds": preprocess_seconds + asr_seconds,
                "asr_seconds": asr_seconds,
                "vad_segments": segment_count,
                "dropped_completely": name != "raw" and len(processed) == 0,
                "vad_config": vad_config,
                "peak_process_memory_mb": peak_memory,
                "script_aware": script_aware_metrics(row["reference"], hypothesis),
            }
            records_by_technique[name].append(record)
            print(f"[{name}] {row['utterance_id']} {record['inference_seconds']:.3f}s")

    summary = []
    for name, records in records_by_technique.items():
        metrics = aggregate_metrics(records)
        processed_total = sum(record["processed_duration_seconds"] for record in records)
        summary.append({
            "technique": name,
            "count": len(records),
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "latency_seconds": metrics["inference_seconds"],
            "mean_latency_seconds": metrics["inference_seconds"] / len(records) if records else None,
            "input_duration_seconds": metrics["duration_seconds"],
            "processed_duration_seconds": processed_total,
            "mean_processed_duration_seconds": processed_total / len(records) if records else None,
            "dropped_completely": sum(record["dropped_completely"] for record in records),
            "mean_rtf": metrics["mean_rtf"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": asr.model_id,
        "adapter": asr.adapter,
        "runtime": getattr(asr, "runtime", "PyTorch"),
        "slice": "short_turn",
        "count": len(rows),
        "warmup": "first utterance, excluded from reported timing",
        "settings": [
            {"technique": name, "threshold": threshold, "min_speech_duration_ms": min_speech,
             "min_silence_duration_ms": min_silence, "speech_pad_ms": padding}
            for name, threshold, min_speech, min_silence, padding in VAD_SETTINGS
        ],
        "summary": summary,
        "records": records_by_technique,
    }
    (output_dir / "short_turn_vad_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "short_turn_vad_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    print("\nSHORT-TURN SUMMARY")
    print("Technique                         WER     Latency(s)  Processed(s)  Dropped")
    for item in summary:
        print(f"{item['technique']:32s} {item['wer']:.4f}  {item['latency_seconds']:10.2f}  "
              f"{item['processed_duration_seconds']:12.2f}  {item['dropped_completely']:7d}")
    return payload


def _run_stereo_pair(
    manifest_rows: list[dict[str, str]],
    asr: MMSASR,
    output_dir: Path,
) -> dict:
    by_id = {row["utterance_id"]: row for row in manifest_rows}
    left_row = by_id[STEREO_LEFT_ID]
    right_row = by_id[STEREO_RIGHT_ID]
    left, sample_rate = _load_row_audio(left_row)
    right, right_sample_rate = _load_row_audio(right_row)
    if right_sample_rate != sample_rate:
        raise ValueError("Stereo source sample rates must match after resampling.")

    length = max(len(left), len(right))
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[:len(left), 0] = left
    stereo[:len(right), 1] = right

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / f"synthetic_stereo_{STEREO_LEFT_ID}_{STEREO_RIGHT_ID}.wav"
    save_wav(synthetic_path, stereo, sample_rate)

    vad = SileroVAD(threshold=0.40, min_speech_duration_ms=100,
                    min_silence_duration_ms=200, speech_pad_ms=150)
    hf_token = os.environ.get("HF_TOKEN")
    diarizer = None
    diarization_status = "not run: HF_TOKEN is not set"
    if hf_token:
        from diarization_processor import SpeakerDiarizer
        try:
            diarizer = SpeakerDiarizer(hf_token, DIARIZATION_MODEL)
            diarization_status = "loaded"
        except Exception as exc:
            diarization_status = f"unavailable: {type(exc).__name__}: {exc}"

    stages = ["raw", "right_channel", "right_channel_vad"]
    asr.transcribe(right, sample_rate)
    denoiser = DeepFilterDenoiser()
    stages.append("right_channel_denoise")
    if diarizer is not None:
        stages.extend(["diarization", "diarization_vad"])

    runner = ExperimentRunner(
        asr=asr,
        channel_separator=ChannelSeparator(),
        vad=vad,
        diarizer=diarizer,
        denoiser=denoiser,
        target_speaker="longest",
        output_dir=output_dir / "stages",
        save_audio=True,
    )
    results = runner.run(
        stages=stages,
        audio=stereo,
        sample_rate=sample_rate,
        source_name=synthetic_path.name,
        reference=right_row["reference"],
    )
    payload = {
        "model": asr.model_id,
        "adapter": asr.adapter,
        "runtime": getattr(asr, "runtime", "PyTorch"),
        "construction": "left and right mono utterances from the frozen short_turn slice, aligned at t=0 and zero-padded to equal length",
        "left_utterance_id": STEREO_LEFT_ID,
        "left_reference": left_row["reference"],
        "right_utterance_id": STEREO_RIGHT_ID,
        "right_reference_scored": right_row["reference"],
        "synthetic_stereo_file": str(synthetic_path),
        "diarization_status": diarization_status,
        "target_speaker_policy": "longest accumulated duration; pyannote labels are not treated as stable identities",
        "results": [asdict(result) for result in results],
    }
    (output_dir / "stereo_channel_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR front-end comparisons on the frozen evaluation manifest.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "modelbenchmark_day1/rework_resultsAndArtifacts/frozen_eval_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "evaluation_results")
    parser.add_argument("--model", choices=("mms", "sraavani", "indicconformer"), default="mms")
    parser.add_argument("--mode", choices=("both", "short-turn", "stereo"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="Limit short-turn rows for a smoke test.")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    manifest_rows = _load_manifest(args.manifest)
    short_turn_rows = [row for row in manifest_rows if row["slice"] == "short_turn"]
    if args.limit is not None:
        short_turn_rows = short_turn_rows[:args.limit]

    asr_backends = {
        "mms": MMSASR,
        "sraavani": SraVaaniASR,
        "indicconformer": IndicConformerASR,
    }
    asr = asr_backends[args.model]()
    output_dir = args.output_dir if args.model == "mms" else args.output_dir / args.model
    if args.mode in {"both", "short-turn"}:
        _run_short_turn_vad(short_turn_rows, asr, output_dir)
    if args.mode in {"both", "stereo"}:
        _run_stereo_pair(manifest_rows, asr, output_dir / "stereo")


if __name__ == "__main__":
    main()