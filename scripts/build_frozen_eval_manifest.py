"""Build a deterministic, group-aware STT evaluation manifest.

The manifest references the original files in the external dataset folders; it
does not copy or modify audio. Run from the repository root with the project
Python environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


SEED = "saleagent-stt-eval-v1"
LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
BAD_GV_FLAGS = {"inaudible", "audio_break", "audio_jump"}
CSV_FIELDS = [
    "utterance_id",
    "split",
    "slice",
    "dataset",
    "speaker_id",
    "speaker_split",
    "recording_id",
    "recording_group_id",
    "audio_path",
    "start_seconds",
    "end_seconds",
    "sample_rate_hz",
    "duration_seconds",
    "reference",
    "reference_word_count",
    "reference_source",
    "reference_review_status",
    "source_quality_flags",
    "augmentation",
    "augmentation_seed",
    "derived_from_utterance_id",
]


def read_keyed_text(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            rows[parts[0]] = parts[1].strip()
    return rows


def read_keyed_tsv(
    path: Path, skip_header: bool = False, delimiter: str | None = None
) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if skip_header and lines:
        lines = lines[1:]
    for line in lines:
        parts = line.split(delimiter)
        if parts:
            rows[parts[0]] = parts[1:]
    return rows


def stable_order(value: str) -> str:
    return hashlib.sha256((SEED + ":" + value).encode("utf-8")).hexdigest()


def stable_seed(value: str) -> int:
    return int(hashlib.sha256((SEED + ":" + value).encode("utf-8")).hexdigest()[:8], 16)


def word_count(text: str) -> int:
    # Planned scoring normalization: punctuation (including danda) is not a
    # token, while Indic vowel signs and nukta are retained.
    text = text.casefold().replace("\u200c", "").replace("\u200d", "")
    text = text.replace("।", " ").replace("॥", " ")
    text = re.sub(r"[^\w\s\u0900-\u097f]", " ", text, flags=re.UNICODE)
    return len(text.split())


def group_for_gv(utterance_id: str) -> str:
    # Files with different final segment numbers can belong to one source
    # recording; keep that group in only one evaluation slice.
    return "-".join(utterance_id.split("-")[:-1])


def audio_metadata(path: Path) -> tuple[int, float]:
    info = sf.info(str(path))
    return int(info.samplerate), float(info.duration)


def make_row(
    *,
    utterance_id: str,
    slice_name: str,
    dataset: str,
    speaker_id: str,
    speaker_split: str,
    recording_id: str,
    recording_group_id: str,
    audio_path: Path,
    start: float,
    end: float,
    sample_rate: int,
    reference: str,
    reference_source: str,
    review_status: str,
    quality_flags: str = "",
    augmentation: str = "none",
    augmentation_seed: str = "",
    derived_from: str = "",
) -> dict[str, Any]:
    return {
        "utterance_id": utterance_id,
        "split": "test",
        "slice": slice_name,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "speaker_split": speaker_split,
        "recording_id": recording_id,
        "recording_group_id": recording_group_id,
        "audio_path": str(audio_path.resolve()).replace("\\", "/"),
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(end - start, 3),
        "reference": reference,
        "reference_word_count": word_count(reference),
        "reference_source": reference_source,
        "reference_review_status": review_status,
        "source_quality_flags": quality_flags,
        "augmentation": augmentation,
        "augmentation_seed": augmentation_seed,
        "derived_from_utterance_id": derived_from,
    }


def choose_by_hash(rows: list[dict[str, Any]], key: str, count: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_order(str(row[key])))[:count]


def build_gv_rows(gv_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_dir = gv_root / "Audio"
    references = read_keyed_text(gv_root / "text")
    labels = read_keyed_tsv(gv_root / "utt2labels", skip_header=True, delimiter="\t")
    all_rows: list[dict[str, Any]] = []

    for audio_path in sorted(audio_dir.glob("*.mp3")):
        utterance_id = audio_path.stem
        reference = references.get(utterance_id, "")
        metadata = labels.get(utterance_id, [])
        quality_flags = metadata[7].strip().lower() if len(metadata) > 7 else ""
        if not reference or quality_flags in BAD_GV_FLAGS:
            continue

        sample_rate, duration = audio_metadata(audio_path)
        n_words = word_count(reference)
        group_id = group_for_gv(utterance_id)
        row = make_row(
            utterance_id=utterance_id,
            slice_name="",
            dataset="GramVaani_GV_Eval_3h",
            # This release has no speaker-ID field. Do not infer speakers from
            # filenames or metadata such as accent/gender.
            speaker_id="unknown",
            speaker_split="test_only_speaker_identity_unavailable",
            recording_id=utterance_id,
            recording_group_id=group_id,
            audio_path=audio_path,
            start=0.0,
            end=duration,
            sample_rate=sample_rate,
            reference=reference,
            reference_source="GV_Eval_3h/text",
            review_status="dataset_reference_not_yet_human_audited",
            quality_flags=quality_flags,
        )
        row["_words"] = n_words
        row["_has_latin"] = bool(LATIN_RE.search(reference))
        all_rows.append(row)

    # The quality-filtered dataset contains too few <=5-word examples for 50;
    # use this explicit broader short-turn definition rather than fabricate data.
    short_candidates = [
        row for row in all_rows
        if row["_words"] <= 10 and row["duration_seconds"] <= 8.0
        and row["recording_group_id"] != "01-00159"
    ]
    if len(short_candidates) < 50:
        raise ValueError("Fewer than 50 quality-filtered GV short turns meet <=10 words and <=8 seconds.")
    short_rows = choose_by_hash(short_candidates, "utterance_id", 50)
    short_groups = {row["recording_group_id"] for row in short_rows}

    clean_candidates = [
        row for row in all_rows
        if row["_words"] >= 15 and not row["_has_latin"]
        and 3.0 <= row["duration_seconds"] <= 30.0
        and row["recording_group_id"] not in short_groups
        and row["recording_group_id"] != "01-00159"
    ]
    if len(clean_candidates) < 100:
        raise ValueError("Fewer than 100 clean Hindi utterances remain after group exclusion.")
    clean_rows = choose_by_hash(clean_candidates, "utterance_id", 100)

    for row in clean_rows:
        row["slice"] = "clean_hindi"
        row.pop("_words", None)
        row.pop("_has_latin", None)
    for row in short_rows:
        row["slice"] = "short_turn"
        row.pop("_words", None)
        row.pop("_has_latin", None)
    return clean_rows, short_rows


def build_mucs_mixed_rows(mucs_test_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    transcript_dir = mucs_test_root / "transcripts"
    references = read_keyed_text(transcript_dir / "text")
    utt2spk = read_keyed_tsv(transcript_dir / "utt2spk")
    segments = read_keyed_tsv(transcript_dir / "segments")

    speaker_ids = sorted({values[0] for values in utt2spk.values() if values}, key=stable_order)
    if len(speaker_ids) < 10:
        raise ValueError("MUCS speaker metadata is missing or too small for grouped splitting.")
    test_n = max(5, int(math.ceil(len(speaker_ids) * 0.20)))
    dev_n = max(3, int(math.ceil(len(speaker_ids) * 0.15)))
    # The retained long-form test recording belongs to this speaker, so reserve
    # the speaker in test rather than silently leaking it into train/dev.
    reserved_test_speakers = {"103085"} & set(speaker_ids)
    test_speakers = sorted(reserved_test_speakers, key=stable_order)
    test_speakers.extend(
        speaker for speaker in speaker_ids
        if speaker not in reserved_test_speakers
    )
    test_speakers = test_speakers[:test_n]
    assignments = {speaker: "train" for speaker in speaker_ids}
    for speaker in test_speakers:
        assignments[speaker] = "test"
    remaining_speakers = [speaker for speaker in speaker_ids if speaker not in test_speakers]
    for speaker in remaining_speakers[:dev_n]:
        assignments[speaker] = "dev"

    candidates: list[dict[str, Any]] = []
    recording_speakers: dict[str, set[str]] = defaultdict(set)
    for utterance_id, values in segments.items():
        if len(values) != 3:
            continue
        recording_id, raw_start, raw_end = values
        reference = references.get(utterance_id, "")
        speaker_values = utt2spk.get(utterance_id, [])
        if not reference or not speaker_values:
            continue
        speaker_id = speaker_values[0]
        recording_speakers[recording_id].add(speaker_id)
        start, end = float(raw_start), float(raw_end)
        duration = end - start
        is_mixed = bool(LATIN_RE.search(reference) and DEVANAGARI_RE.search(reference))
        # The 413-second lecture was overrepresented in Round 1. Keep its
        # recording as a separate long-form row, not in this independent slice.
        if (
            assignments.get(speaker_id) == "test"
            and recording_id != "w5Jyq3XMbb3WwiKQ"
            and is_mixed
            and 2 <= word_count(reference) <= 25
            and 1.0 <= duration <= 20.0
        ):
            audio_path = mucs_test_root / (recording_id + ".wav")
            if audio_path.is_file():
                sample_rate, _ = audio_metadata(audio_path)
                candidates.append(make_row(
                    utterance_id=utterance_id,
                    slice_name="code_mixed_mucs",
                    dataset="MUCS_Hindi_English_test",
                    speaker_id=speaker_id,
                    speaker_split="test",
                    recording_id=recording_id,
                    recording_group_id=recording_id,
                    audio_path=audio_path,
                    start=start,
                    end=end,
                    sample_rate=sample_rate,
                    reference=reference,
                    reference_source="MUCS Hindi-English_test/test/transcripts/text",
                    review_status="dataset_reference_script_style_needs_human_audit",
                ))

    if any(len(speakers) != 1 for speakers in recording_speakers.values()):
        raise ValueError("At least one MUCS recording maps to multiple speakers; group by recording too.")
    if len(candidates) < 100:
        raise ValueError("Fewer than 100 eligible mixed utterances in held-out MUCS speakers.")
    mixed_rows = choose_by_hash(candidates, "utterance_id", 100)
    return mixed_rows, assignments


def build_long_form(workspace: Path) -> dict[str, Any]:
    audio_path = workspace / "data" / "w5Jyq3XMbb3WwiKQ.wav"
    ground_truth = workspace / "data" / "gt.txt"
    prefix = "103085_w5Jyq3XMbb3WwiKQ_"
    matches: list[tuple[int, str]] = []
    for line in ground_truth.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].startswith(prefix):
            suffix = parts[0][len(prefix):]
            if suffix.isdigit():
                matches.append((int(suffix), parts[1].strip()))
    if not matches:
        raise ValueError("No segmented long-form reference found in data/gt.txt.")
    matches.sort(key=lambda item: item[0])
    reference = " ".join(text for _, text in matches)
    sample_rate, duration = audio_metadata(audio_path)
    return make_row(
        utterance_id="mucs_w5Jyq3XMbb3WwiKQ_longform",
        slice_name="long_form_separate",
        dataset="MUCS_Hindi_English_test",
        speaker_id="103085",
        speaker_split="test",
        recording_id="w5Jyq3XMbb3WwiKQ",
        recording_group_id="w5Jyq3XMbb3WwiKQ",
        audio_path=audio_path,
        start=0.0,
        end=duration,
        sample_rate=sample_rate,
        reference=reference,
        reference_source="repository data/gt.txt (segment references concatenated by index)",
        review_status="repo_reference_not_yet_human_audited; verify_not_model_hypothesis",
    )


def add_telephony_rows(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in clean_rows:
        row = dict(source)
        row.update({
            "utterance_id": source["utterance_id"] + "__telephony",
            "slice": "telephony_from_clean_hindi",
            "augmentation": "deterministic_8khz_mulaw_plus_noise_per_evaluate_asr_make_telephony_audio",
            "augmentation_seed": stable_seed(source["utterance_id"]),
            "derived_from_utterance_id": source["utterance_id"],
        })
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        slices[row["slice"]].append(row)
    summary: dict[str, Any] = {
        "manifest_version": SEED,
        "split_policy": "MUCS speaker/recording groups assigned train/dev/test; Gram Vaani recording groups disjoint but speaker IDs unavailable; test rows only; deterministic SHA-256 ordering",
        "slice_summary": {},
    }
    for slice_name, members in sorted(slices.items()):
        summary["slice_summary"][slice_name] = {
            "utterance_count": len(members),
            "total_duration_seconds": round(sum(row["duration_seconds"] for row in members), 3),
            "total_reference_words": sum(row["reference_word_count"] for row in members),
            "sample_rates_hz": dict(Counter(str(row["sample_rate_hz"]) for row in members)),
            "speaker_ids_known": len({row["speaker_id"] for row in members if row["speaker_id"] != "unknown"}),
            "recording_groups": len({row["recording_group_id"] for row in members}),
            "reference_review_status": dict(Counter(row["reference_review_status"] for row in members)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1],
        help="Saleagent repository root.",
    )
    parser.add_argument(
        "--mucs-root", type=Path,
        default=Path(r"C:\Users\91950\Downloads\mucs\Hindi-English_test\test"),
    )
    parser.add_argument(
        "--gv-root", type=Path,
        default=Path(r"C:\Users\91950\Downloads\GV_Eval_3h\GV_Eval_3h"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output_dir = (args.output_dir or workspace / "evaluation").resolve()
    if not (args.mucs_root / "transcripts" / "segments").is_file():
        raise SystemExit("MUCS root must contain transcripts/segments and transcripts/text.")
    if not (args.gv_root / "Audio").is_dir():
        raise SystemExit("GV root must contain Audio/ and text.")

    clean_rows, short_rows = build_gv_rows(args.gv_root)
    mixed_rows, speaker_assignments = build_mucs_mixed_rows(args.mucs_root)
    rows = clean_rows + add_telephony_rows(clean_rows) + short_rows + mixed_rows
    rows.append(build_long_form(workspace))

    ids = [row["utterance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate utterance IDs.")
    if len(clean_rows) != 100 or len(short_rows) != 50 or len(mixed_rows) != 100:
        raise ValueError("Requested slice sizes were not met.")
    short_groups = {row["recording_group_id"] for row in short_rows}
    clean_groups = {row["recording_group_id"] for row in clean_rows}
    if short_groups & clean_groups:
        raise ValueError("GV source recording groups overlap between clean and short-turn slices.")
    if any(row["split"] != "test" for row in rows):
        raise ValueError("Only held-out test rows should be written to this manifest.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frozen_eval_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metadata = summarize(rows)
    metadata.update({
        "selection_seed": SEED,
        "speaker_assignments_mucs": dict(sorted(speaker_assignments.items())),
        "excluded_gv_quality_flags": sorted(BAD_GV_FLAGS),
        "short_turn_rule": "<=10 normalized reference words and <=8 seconds; 50 utterances",
        "clean_rule": ">=15 normalized reference words, no Latin-script token, 3-30 seconds; 100 utterances",
        "code_mixed_rule": "reference contains both Latin and Devanagari, 2-25 words, 1-20 seconds; 100 utterances from held-out MUCS speakers and recordings",
        "telephony_rule": "one deterministic telephony degradation per clean Hindi utterance; same reference/audio identity",
        "long_form_rule": "one 413-second recording, scored and reported separately; never pool into short-form slices",
        "limitations": [
            "Gram Vaani metadata contains no speaker IDs; clean/short-turn slices are recording-group-disjoint, but speaker-disjointness within GV cannot be independently proven. Do not use these GV evaluation recordings for training until speaker identities are obtained.",
            "Dataset-provided references have not been manually checked against audio. The manifest records reference provenance and marks human audit as pending; particularly audit MUCS script choices and the concatenated long-form reference before publishing final scores.",
            "MUCS 16 kHz segments are represented by source recording path plus start/end offsets; evaluation code must crop those intervals before inference.",
            "The short-turn slice uses a <=10-word, <=8-second operational definition because the quality-filtered release has too few <=5-word samples to reach 50.",
            "The telephony slice is a deterministic augmentation recipe, not copied audio; apply the named degradation and recorded seed at evaluation time.",
        ],
    })
    (output_dir / "frozen_eval_manifest_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["slice_summary"], ensure_ascii=False, indent=2))
    print("Manifest:", manifest_path)
    print("Summary:", output_dir / "frozen_eval_manifest_summary.json")


if __name__ == "__main__":
    main()