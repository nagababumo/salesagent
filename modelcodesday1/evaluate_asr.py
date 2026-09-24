from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class Example:
    audio: Path
    reference: str


def normalize_text(text: str) -> str:
    """Normalize without removing Indic vowel signs or changing scripts."""
    text = text.casefold().replace("\u200c", "").replace("\u200d", "")
    text = NUMBER_RE.sub(lambda match: re.sub(r"[.,]", "", match.group()), text)
    text = re.sub(r"[^\w\s\u0900-\u097f]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


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
    return {
        "wer": word_errors / reference_words if reference_words else None,
        "cer": character_errors / reference_characters if reference_characters else None,
        "reference_words": reference_words,
        "reference_characters": reference_characters,
        "mean_rtf": sum(item["rtf"] for item in records) / len(records) if records else None,
        "max_peak_process_memory_mb": max(
            (item["peak_process_memory_mb"] for item in records), default=None
        ),
    }


def script_aware_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_words = words(reference)
    hyp_words = words(hypothesis)
    ref_devanagari = [char for char in characters(reference) if DEVANAGARI_RE.match(char)]
    hyp_devanagari = [char for char in characters(hypothesis) if DEVANAGARI_RE.match(char)]
    ref_latin = LATIN_WORD_RE.findall(normalize_text(reference))
    hyp_latin = LATIN_WORD_RE.findall(normalize_text(hypothesis))
    english_reference = set(ref_latin)
    english_hypothesis = set(hyp_latin)
    ref_has_devanagari = bool(ref_devanagari)
    ref_has_latin = bool(ref_latin)

    wrong_script = 0
    for ref_word in ref_words:
        if DEVANAGARI_RE.search(ref_word) and LATIN_WORD_RE.fullmatch(ref_word):
            wrong_script += 1

    return {
        "is_code_mixed": ref_has_devanagari and ref_has_latin,
        "mixed_error_rate": {
            "devanagari_cer": error_rate(ref_devanagari, hyp_devanagari),
            "latin_word_wer": error_rate(ref_latin, hyp_latin),
        },
        "english_word_accuracy": (
            len(english_reference & english_hypothesis) / len(english_reference)
            if english_reference
            else None
        ),
        "wrong_script_rate": wrong_script / len(ref_words) if ref_words else None,
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


