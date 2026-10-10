from __future__ import annotations

import json
import os
import platform
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
NORMALIZATION_POLICY = (
    "casefold; strip zero-width characters; remove punctuation including danda; "
    "remove separators inside digit groups; collapse whitespace; preserve "
    "Devanagari combining marks and nukta distinctions"
)


@dataclass
class Example:
    audio: Path
    reference: str


def normalize_text(text: str) -> str:
    """Normalize text identically for all ASR scoring paths.

    Removes zero-width characters and punctuation (including danda), strips
    separators inside digit groups, case-folds Latin text, and preserves
    Devanagari combining marks and nukta distinctions.
    """
    text = text.casefold().replace("\u200c", "").replace("\u200d", "")
    text = NUMBER_RE.sub(lambda match: re.sub(r"[.,]", "", match.group()), text)
    normalized = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            normalized.append(" ")
        elif category.startswith(("P", "S")):
            normalized.append(" ")
        elif category.startswith("M") and "\u0900" <= char <= "\u097f":
            normalized.append(char)
        elif category[0] in {"L", "N"} or "\u0900" <= char <= "\u097f":
            normalized.append(char)
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


def words(text: str) -> list[str]:
    return normalize_text(text).split()


def characters(text: str) -> list[str]:
    return [char for char in normalize_text(text) if not char.isspace()]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    return None if not reference else edit_distance(reference, hypothesis) / len(reference)


def align_words(reference: list[str], hypothesis: list[str]) -> list[tuple[str | None, str | None]]:
    """Return a minimum-edit alignment, with deterministic tie handling."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], operations[i][0] = i, "delete"
    for j in range(1, columns):
        costs[0][j], operations[0][j] = j, "insert"

    for i in range(1, rows):
        for j in range(1, columns):
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            # Prefer diagonal (match/substitution), then deletion, then insertion.
            best = min(substitution, deletion, insertion)
            costs[i][j] = best
            operations[i][j] = (
                "equal" if reference[i - 1] == hypothesis[j - 1] else "substitute"
            ) if substitution == best else "delete" if deletion == best else "insert"

    aligned: list[tuple[str | None, str | None]] = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = operations[i][j]
        if operation in {"equal", "substitute"}:
            aligned.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif operation == "delete":
            aligned.append((reference[i - 1], None))
            i -= 1
        else:
            aligned.append((None, hypothesis[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def _is_latin_word(token: str) -> bool:
    return bool(LATIN_WORD_RE.fullmatch(token))


def _has_devanagari(token: str) -> bool:
    return bool(DEVANAGARI_RE.search(token))


def runtime_metadata(beam_size: int | str | None = None) -> dict[str, Any]:
    """Return comparable hardware/decoder metadata for an inference run."""
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_threads = torch.get_num_threads()
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    except (ImportError, RuntimeError):
        device, torch_threads, gpu_name = "cpu", None, None
    return {
        "device": device,
        "gpu_name": gpu_name,
        "threads": torch_threads or os.cpu_count(),
        "cpu_count": os.cpu_count(),
        "beam_size": beam_size,
        "platform": platform.platform(),
    }


def metric(reference: str, hypothesis: str) -> dict[str, float | int | None]:
    ref_words, hyp_words = words(reference), words(hypothesis)
    ref_chars, hyp_chars = characters(reference), characters(hypothesis)
    return {
        "wer": error_rate(ref_words, hyp_words),
        "cer": error_rate(ref_chars, hyp_chars),
        "reference_words": len(ref_words),
        "reference_characters": len(ref_chars),
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    word_errors = sum(edit_distance(words(item["reference"]), words(item["hypothesis"])) for item in records)
    character_errors = sum(
        edit_distance(characters(item["reference"]), characters(item["hypothesis"])) for item in records
    )
    reference_words = sum(len(words(item["reference"])) for item in records)
    reference_characters = sum(len(characters(item["reference"])) for item in records)
    duration_seconds = sum(float(item.get("duration_seconds", 0.0)) for item in records)
    elapsed_seconds = 0.0
    timed_record_count = 0
    batch_elapsed: dict[str, float] = {}
    for index, item in enumerate(records):
        inference_seconds = item.get("inference_seconds")
        if inference_seconds is not None:
            elapsed_seconds += float(inference_seconds)
            timed_record_count += 1
        elif item.get("batch_elapsed_seconds") is not None:
            batch_id = str(item.get("batch_id", index))
            batch_elapsed[batch_id] = float(item["batch_elapsed_seconds"])
        elif item.get("rtf") is not None:
            # Backward compatibility for archived raw outputs; do not use this
            # reconstructed timing for newly produced batched evaluations.
            elapsed_seconds += float(item["rtf"]) * float(item.get("duration_seconds", 0.0))
            timed_record_count += 1
    elapsed_seconds += sum(batch_elapsed.values())
    script_totals = {
        "english_reference_words": 0,
        "english_correct_words": 0,
        "aligned_reference_words": 0,
        "wrong_script_words": 0,
        "english_as_devanagari": 0,
        "hindi_as_latin": 0,
    }
    for item in records:
        script = item.get("script_aware", {})
        for key in script_totals:
            script_totals[key] += int(script.get(key, 0) or 0)
    return {
        "wer": word_errors / reference_words if reference_words else None,
        "cer": character_errors / reference_characters if reference_characters else None,
        "word_errors": word_errors,
        "character_errors": character_errors,
        "reference_words": reference_words,
        "reference_characters": reference_characters,
        "inference_seconds": elapsed_seconds,
        "duration_seconds": duration_seconds,
        "mean_rtf": elapsed_seconds / duration_seconds if duration_seconds else None,
        "script_aware": {
            **script_totals,
            "english_word_accuracy": (
                script_totals["english_correct_words"] / script_totals["english_reference_words"]
                if script_totals["english_reference_words"] else None
            ),
            "wrong_script_rate": (
                script_totals["wrong_script_words"] / script_totals["aligned_reference_words"]
                if script_totals["aligned_reference_words"] else None
            ),
        },
        "max_peak_process_memory_mb": max(
            (item["peak_process_memory_mb"] for item in records
             if item.get("peak_process_memory_mb") is not None),
            default=None,
        ),
        "timing_mode": (
            "batch_wall_time_pooled_once" if batch_elapsed else
            "per_utterance_sum" if timed_record_count else "unavailable"
        ),
    }


def script_aware_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_words = words(reference)
    hyp_words = words(hypothesis)
    aligned = align_words(ref_words, hyp_words)
    ref_devanagari = [char for char in characters(reference) if DEVANAGARI_RE.match(char)]
    hyp_devanagari = [char for char in characters(hypothesis) if DEVANAGARI_RE.match(char)]
    ref_latin = LATIN_WORD_RE.findall(normalize_text(reference))
    hyp_latin = LATIN_WORD_RE.findall(normalize_text(hypothesis))

    english_reference_words = 0
    english_correct_words = 0
    aligned_reference_words = 0
    english_as_devanagari = 0
    hindi_as_latin = 0
    for ref_token, hyp_token in aligned:
        if ref_token is None:
            continue
        is_english = _is_latin_word(ref_token)
        is_hindi = _has_devanagari(ref_token)
        if is_english:
            english_reference_words += 1
            if hyp_token == ref_token and hyp_token is not None and _is_latin_word(hyp_token):
                english_correct_words += 1
        if hyp_token is None:
            continue
        if is_english or is_hindi:
            aligned_reference_words += 1
        if is_english and _has_devanagari(hyp_token):
            english_as_devanagari += 1
        elif is_hindi and _is_latin_word(hyp_token):
            hindi_as_latin += 1
    wrong_script_words = english_as_devanagari + hindi_as_latin

    return {
        "is_code_mixed": bool(ref_devanagari and ref_latin),
        "mixed_error_rate": {
            "devanagari_cer": error_rate(ref_devanagari, hyp_devanagari),
            "latin_word_wer": error_rate(ref_latin, hyp_latin),
        },
        "english_word_accuracy": (
            english_correct_words / english_reference_words
            if english_reference_words
            else None
        ),
        "english_reference_words": english_reference_words,
        "english_correct_words": english_correct_words,
        "aligned_reference_words": aligned_reference_words,
        "wrong_script_words": wrong_script_words,
        "english_as_devanagari": english_as_devanagari,
        "hindi_as_latin": hindi_as_latin,
        "wrong_script_rate": (
            wrong_script_words / aligned_reference_words
            if aligned_reference_words else None
        ),
    }


def read_manifest(path: Path) -> list[Example]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Example(Path(item["audio"]), item["text"]) for item in records]


def read_ground_truth(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_audio_mappings(path: Path) -> dict[Path, str]:
    mappings = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        audio_path, identifier = line.split("\t", maxsplit=1)
        mappings[(path.parent / audio_path).resolve()] = identifier.strip()
    return mappings


def find_examples(data_dir: Path, ground_truth_path: Path) -> list[Example]:
    lines = read_ground_truth(ground_truth_path)
    mappings = read_audio_mappings(ground_truth_path)
    audio_files = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    examples = []
    for audio in audio_files:
        mapped_identifier = mappings.get(audio.resolve())
        candidates = [mapped_identifier] if mapped_identifier else [audio.stem, audio.parent.name]
        matching_segments = []
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            identifier = parts[0].rstrip(":")
            if any(
                identifier == candidate
                or (candidate.endswith("_") and identifier.startswith(candidate))
                or identifier.startswith(f"{candidate}_")
                or f"_{candidate}_" in identifier
                for candidate in candidates
            ):
                matching_segments.append(parts[1])
        reference = " ".join(matching_segments) if matching_segments else None
        if reference is None:
            reference = next((line for line in lines if line.startswith(f"{audio.stem}:")), None)
        if reference:
            examples.append(Example(audio, reference))
    return examples


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    old_positions = np.linspace(0, 1, len(audio), endpoint=False)
    new_positions = np.linspace(0, 1, target_length, endpoint=False)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def make_telephony_audio(audio: np.ndarray, sample_rate: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    narrowband = resample(audio, sample_rate, 8000)
    mu = 255.0
    encoded = np.sign(narrowband) * np.log1p(mu * np.abs(narrowband)) / np.log1p(mu)
    decoded = np.sign(encoded) * (np.expm1(np.abs(encoded) * np.log1p(mu)) / mu)
    noisy = decoded + rng.normal(0, 0.008, len(decoded)).astype(np.float32)
    return np.clip(resample(noisy, 8000, sample_rate), -1, 1), sample_rate


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sample_rate


