from __future__ import annotations

"""Build a single model-by-slice CSV from raw frozen-benchmark JSON outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_frozen_benchmark import MODEL_METADATA

SLICES = [
    "clean_hindi",
    "short_turn",
    "telephony_from_clean_hindi",
    "code_mixed_mucs",
    "long_form_separate",
]


def _number(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for key, metadata in MODEL_METADATA.items():
        result_path = results_dir / f"{key}_evaluation.json"
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        errors_path = results_dir / "run_errors.json"
        errors = json.loads(errors_path.read_text(encoding="utf-8")) if errors_path.is_file() else {}
        status = "completed" if payload else "failed" if key in errors else "not_run"
        for slice_name in SLICES:
            slice_result = payload.get("slices", {}).get(slice_name, {})
            agg = slice_result.get("aggregate", {})
            script = agg.get("script_aware", {})
            rows.append({
                "status": status,
                "model_key": key,
                "model_id": metadata["model_id"],
                "slice": slice_name,
                "n": slice_result.get("count", 0),
                "WER": _number(agg.get("wer")),
                "CER": _number(agg.get("cer")),
                "wrong_script_rate": _number(script.get("wrong_script_rate")),
                "english_word_accuracy": _number(script.get("english_word_accuracy")),
                "RTF_pooled": _number(agg.get("mean_rtf")),
                "peak_memory_MB": _number(agg.get("max_peak_process_memory_mb")),
                "device": payload.get("device", ""),
                "gpu_name": payload.get("gpu_name", ""),
                "threads": payload.get("threads", ""),
                "beam_size": payload.get("beam_size", ""),
                "reference_source": _reference_sources(slice_result.get("records", [])),
                "decoder": metadata["decoder"],
                "tokenizer_vocab_size": payload.get("tokenizer_vocab_size", metadata["tokenizer_vocab_size"]),
                "has_devanagari": metadata["has_devanagari"],
                "claims_hindi": metadata["claims_hindi"],
                "streaming": metadata["streaming"],
                "long_audio_handling": metadata["long_audio"],
                "failure": errors.get(key, ""),
            })
    return rows


def _reference_sources(records: list[dict[str, Any]]) -> str:
    sources = sorted({record.get("reference_source", "") for record in records if record.get("reference_source")})
    return "; ".join(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("modelbenchmark_day2"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = args.output or args.results_dir / "model_by_slice_results.csv"
    rows = build_rows(args.results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
