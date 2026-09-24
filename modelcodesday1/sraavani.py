from __future__ import annotations

import argparse
import json
import os
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
from transformers import AutoModel

from evaluate_asr import (
    aggregate_metrics,
    find_examples,
    make_telephony_audio,
    metric,
    resample,
    script_aware_metrics,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "ARTPARK-IISc/SraVaani-1.0"
MODEL_SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# SraVaani-specific engineering defaults.
#
# These are NOT official maximum limits from ARTPARK.
# They are conservative defaults intended to keep long-utterance inference
# manageable while still allowing real batched inference.
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 1.0
DEFAULT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkJob:
    """
    One inference unit.

    file_index:
        Index of the original audio file in the current batch.

    chunk_index:
        Sequential chunk number for that file.

    audio:
        Float32 mono waveform at 16 kHz.
    """

    file_index: int
    chunk_index: int
    audio: np.ndarray


# ---------------------------------------------------------------------------
# File/audio helpers
# ---------------------------------------------------------------------------

def find_audio_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float32.
    """
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).flatten()

    return audio, sample_rate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(
    model_id: str,
):
    """
    Load SraVaani using the model's official custom-code implementation.

    We intentionally use model.transcribe() rather than calling the
    AutoProcessor directly. The released model documents transcribe()
    for one or more audio paths and exposes a batch_size argument.
    """
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    print(
        f"Downloading required repository files for: {model_id}"
    )

    model_dir = snapshot_download(
        repo_id=model_id,
        token=hf_token,
    )

    print(
        f"Loading model from local snapshot: {model_dir}"
    )

    load_kwargs = {
        "trust_remote_code": True,
    }

    if hf_token:
        load_kwargs["token"] = hf_token

    print(
        f"Loading model on {DEVICE}: {model_id}"
    )

    model = AutoModel.from_pretrained(
        model_dir,
        **load_kwargs,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def get_process_memory_mb() -> float:
    """
    Current process RSS, not OS-wide peak RSS.
    """
    return (
        psutil.Process().memory_info().rss
        / (1024 * 1024)
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_chunk_jobs(
    audio_list: list[np.ndarray],
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ChunkJob]:
    """
    Convert a list of utterances into inference chunks.

    Short utterances remain intact.

    Long utterances are split into overlapping windows.
    """
    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(
            f"Expected audio at {MODEL_SAMPLE_RATE} Hz, "
            f"got {sample_rate} Hz."
        )

    if chunk_seconds <= 0:
        raise ValueError(
            "chunk_seconds must be > 0"
        )

    if overlap_seconds < 0:
        raise ValueError(
            "overlap_seconds must be >= 0"
        )

    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            "overlap_seconds must be smaller than chunk_seconds"
        )

    chunk_samples = int(
        round(chunk_seconds * sample_rate)
    )

    overlap_samples = int(
        round(overlap_seconds * sample_rate)
    )

    step_samples = (
        chunk_samples - overlap_samples
    )

    jobs: list[ChunkJob] = []

    for file_index, audio in enumerate(audio_list):
        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).flatten()

        if audio.size == 0:
            jobs.append(
                ChunkJob(
                    file_index=file_index,
                    chunk_index=0,
                    audio=audio,
                )
            )
            continue

        # Keep short utterances intact.
        if audio.size <= chunk_samples:
            jobs.append(
                ChunkJob(
                    file_index=file_index,
                    chunk_index=0,
                    audio=audio,
                )
            )
            continue

        chunk_index = 0
        start = 0

        while start < audio.size:
            end = min(
                start + chunk_samples,
                audio.size,
            )

            chunk = np.asarray(
                audio[start:end],
                dtype=np.float32,
            )

            jobs.append(
                ChunkJob(
                    file_index=file_index,
                    chunk_index=chunk_index,
                    audio=chunk,
                )
            )

            chunk_index += 1

            if end >= audio.size:
                break

            start += step_samples

    return jobs


# ---------------------------------------------------------------------------
# SraVaani inference helpers
# ---------------------------------------------------------------------------

def extract_hypothesis_text(hypothesis) -> str:
    """
    Extract text from SraVaani/NeMo hypothesis objects.

    The custom model normally returns an object with a .text attribute,
    but this also tolerates plain strings and a few common container forms.
    """
    if hypothesis is None:
        return ""

    if isinstance(hypothesis, str):
        return hypothesis.strip()

    if hasattr(hypothesis, "text"):
        return str(hypothesis.text).strip()

    if isinstance(hypothesis, (list, tuple)) and hypothesis:
        return extract_hypothesis_text(hypothesis[0])

    return str(hypothesis).strip()


def write_batch_to_temp_wavs(
    audio_batch: list[np.ndarray],
    temp_dir: str | Path,
    prefix: str,
) -> list[str]:
    """
    Write already-normalized 16 kHz mono float32 arrays to temporary WAVs.

    SraVaani's documented batched API accepts a list of audio paths. Using
    temporary WAVs lets us retain our resampling/telephony transformations
    while still using the model's own batching/data-loader path.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []

    for index, audio in enumerate(audio_batch):
        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).flatten()

        path = temp_dir / f"{prefix}_{index:05d}.wav"

        sf.write(
            str(path),
            audio,
            MODEL_SAMPLE_RATE,
            subtype="PCM_16",
        )

        paths.append(str(path))

    return paths


def transcribe_paths(
    model,
    paths: list[str],
    batch_size: int,
) -> list[str]:
    """
    Run SraVaani's official batched transcription API.

    batch_size is passed directly to model.transcribe(), rather than
    attempting to batch variable-length waveforms through the custom
    processor ourselves.
    """
    if not paths:
        return []

    effective_batch_size = min(
        batch_size,
        len(paths),
    )

    with torch.inference_mode():
        hypotheses = model.transcribe(
            paths,
            batch_size=effective_batch_size,
            return_hypotheses=True,
        )

    # Be defensive because hybrid NeMo models can sometimes return
    # a tuple/list structure depending on the active decoding configuration.
    if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
        hypotheses = hypotheses[0]

    if hypotheses is None:
        return ["" for _ in paths]

    if not isinstance(hypotheses, (list, tuple)):
        hypotheses = [hypotheses]

    texts = [
        extract_hypothesis_text(hyp)
        for hyp in hypotheses
    ]

    if len(texts) != len(paths):
        raise RuntimeError(
            "SraVaani returned a different number of hypotheses "
            f"than input paths: {len(texts)} vs {len(paths)}"
        )

    return texts


def transcribe_chunk_jobs(
    model,
    jobs: list[ChunkJob],
    model_batch_size: int,
) -> dict[tuple[int, int], str]:
    """
    Transcribe chunk jobs in real model batches.

    Each batch is passed to model.transcribe([...], batch_size=N).
    """
    if not jobs:
        return {}

    results: dict[tuple[int, int], str] = {}

    with tempfile.TemporaryDirectory(
        prefix="sravaani_chunks_"
    ) as temp_dir:
        for start in range(
            0,
            len(jobs),
            model_batch_size,
        ):
            batch_jobs = jobs[
                start:start + model_batch_size
            ]

            batch_audio = [
                job.audio
                for job in batch_jobs
            ]

            durations = [
                len(audio) / MODEL_SAMPLE_RATE
                for audio in batch_audio
            ]

            print(
                "SraVaani batch: "
                f"{len(batch_jobs)} chunks | "
                "durations="
                + ", ".join(
                    f"{duration:.1f}s"
                    for duration in durations
                )
            )

            batch_paths = write_batch_to_temp_wavs(
                batch_audio,
                temp_dir=temp_dir,
                prefix=f"batch_{start:05d}",
            )

            try:
                batch_texts = transcribe_paths(
                    model=model,
                    paths=batch_paths,
                    batch_size=len(batch_paths),
                )

            except RuntimeError as error:
                # If the model/data-loader batch fails, retry progressively
                # with smaller actual model batches.
                if len(batch_jobs) == 1:
                    raise

                midpoint = len(batch_jobs) // 2

                print(
                    f"SraVaani batch of {len(batch_jobs)} failed; "
                    f"retrying as {midpoint} + "
                    f"{len(batch_jobs) - midpoint}."
                )

                left_results = transcribe_chunk_jobs(
                    model,
                    batch_jobs[:midpoint],
                    model_batch_size=min(
                        model_batch_size,
                        midpoint,
                    ),
                )

                right_results = transcribe_chunk_jobs(
                    model,
                    batch_jobs[midpoint:],
                    model_batch_size=min(
                        model_batch_size,
                        len(batch_jobs) - midpoint,
                    ),
                )

                results.update(left_results)
                results.update(right_results)
                continue

            for job, text in zip(
                batch_jobs,
                batch_texts,
            ):
                results[
                    (
                        job.file_index,
                        job.chunk_index,
                    )
                ] = text

    return results


def transcribe_batch(
    model,
    audio_list: list[np.ndarray],
    model_batch_size: int,
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int = MODEL_SAMPLE_RATE,
):
    """
    Transcribe multiple original utterances with real SraVaani batching.

    Long utterances are chunked first. The resulting chunks are then grouped
    into actual calls to model.transcribe(..., batch_size=N).

    This avoids the previous failure caused by passing a Python list of
    different-length NumPy arrays directly into processing_sravaani.py.
    """
    if not audio_list:
        return [], 0.0, 0.0

    if model_batch_size < 1:
        raise ValueError(
            "model_batch_size must be >= 1"
        )

    jobs = make_chunk_jobs(
        audio_list=audio_list,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    if not jobs:
        return (
            ["" for _ in audio_list],
            0.0,
            get_process_memory_mb(),
        )

    # Similar durations are placed close together so the underlying
    # batching/data-loader performs less padding/work.
    jobs = sorted(
        jobs,
        key=lambda job: len(job.audio),
    )

    started = time.perf_counter()

    chunk_results = transcribe_chunk_jobs(
        model=model,
        jobs=jobs,
        model_batch_size=model_batch_size,
    )

    final_texts: list[str] = []

    for file_index in range(len(audio_list)):
        file_chunks = [
            (
                job.chunk_index,
                chunk_results[
                    (
                        job.file_index,
                        job.chunk_index,
                    )
                ],
            )
            for job in jobs
            if job.file_index == file_index
        ]

        file_chunks.sort(
            key=lambda item: item[0]
        )

        merged = ""

        for _, text in file_chunks:
            merged = merge_texts(
                merged,
                text,
            )

        final_texts.append(
            merged.strip()
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    peak_memory = get_process_memory_mb()

    return (
        final_texts,
        elapsed,
        peak_memory,
    )


def transcribe_audio(
    model,
    audio: np.ndarray,
    model_batch_size: int,
    chunk_seconds: float,
    overlap_seconds: float,
):
    texts, elapsed, peak_memory = transcribe_batch(
        model=model,
        audio_list=[audio],
        model_batch_size=model_batch_size,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate=MODEL_SAMPLE_RATE,
    )

    return (
        texts[0] if texts else "",
        elapsed,
        peak_memory,
    )


# ---------------------------------------------------------------------------
# Text merging
# ---------------------------------------------------------------------------

def merge_texts(
    previous: str,
    current: str,
    max_overlap_words: int = 12,
) -> str:
    """
    Merge overlapping chunk transcriptions.

    Example:

        previous = "... आज बाजार जा रहे"
        current  = "जा रहे हैं कल"

    becomes:

        "... आज बाजार जा रहे हैं कल"

    This is intentionally conservative and only removes an exact
    suffix/prefix word overlap.
    """
    previous = previous.strip()
    current = current.strip()

    if not previous:
        return current

    if not current:
        return previous

    if previous == current:
        return previous

    previous_words = previous.split()
    current_words = current.split()

    max_overlap = min(
        max_overlap_words,
        len(previous_words),
        len(current_words),
    )

    overlap = 0

    for n in range(
        max_overlap,
        0,
        -1,
    ):
        if (
            previous_words[-n:]
            == current_words[:n]
        ):
            overlap = n
            break

    if overlap:
        current_words = current_words[
            overlap:
        ]

    if not current_words:
        return previous

    return " ".join(
        previous_words + current_words
    )


def merge_chunk_results(
    jobs: list[ChunkJob],
    chunk_texts: list[str],
) -> list[str]:
    """
    Reconstruct one final transcription per original file.
    """
    grouped: dict[int, list[tuple[int, str]]] = {}

    for job, text in zip(
        jobs,
        chunk_texts,
    ):
        grouped.setdefault(
            job.file_index,
            [],
        ).append(
            (job.chunk_index, text)
        )

    results: list[str] = []

    for file_index in range(
        len(grouped)
    ):
        pieces = sorted(
            grouped.get(file_index, []),
            key=lambda item: item[0],
        )

        merged = ""

        for _, text in pieces:
            merged = merge_texts(
                merged,
                text,
            )

        results.append(
            merged.strip()
        )

    return results


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    model_id: str,
    examples,
    language: str,
    output_path: Path,
    batch_size: int,
    chunk_seconds: float,
    overlap_seconds: float,
):
    results = {
        "model_id": model_id,
        "decoder": "SraVaani hybrid TDT-CTC via model.transcribe()",
        "language_argument": (
            f"{language} (evaluation metadata; "
            "SraVaani multilingual model handles language internally)"
        ),
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "batch_size": batch_size,
        "device": str(DEVICE),
        "normalization": (
            "casefold; remove ZWNJ/ZWJ; remove punctuation; "
            "collapse whitespace; remove number separators; "
            "preserve Devanagari vowel signs and scripts"
        ),
        "slices": {},
    }

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Pre-loading and resampling base audio samples..."
    )

    base_audios: list[np.ndarray] = []

    for ex in examples:
        audio, sample_rate = load_audio(
            ex.audio
        )

        audio = resample(
            audio,
            sample_rate,
            MODEL_SAMPLE_RATE,
        )

        base_audios.append(
            np.asarray(
                audio,
                dtype=np.float32,
            )
        )

    for slice_name in (
        "clean",
        "telephony",
        "code_mixed",
    ):
        records = []

        slice_examples = []
        slice_audios = []

        for index, example in enumerate(
            examples
        ):
            if (
                slice_name == "code_mixed"
                and not script_aware_metrics(
                    example.reference,
                    "",
                )["is_code_mixed"]
            ):
                continue

            audio = base_audios[index]

            if slice_name == "telephony":
                audio, _ = make_telephony_audio(
                    audio,
                    MODEL_SAMPLE_RATE,
                    index,
                )

                audio = resample(
                    audio,
                    MODEL_SAMPLE_RATE,
                    MODEL_SAMPLE_RATE,
                )

            slice_examples.append(
                example
            )

            slice_audios.append(
                np.asarray(
                    audio,
                    dtype=np.float32,
                )
            )

        # model_batch_size is the actual batch size passed to
        # SraVaani's model.transcribe() API.
        #
        # We process groups of original utterances here. A long utterance
        # can produce multiple chunk jobs internally.
        for i in range(
            0,
            len(slice_audios),
            batch_size,
        ):
            batch_audios = slice_audios[
                i:i + batch_size
            ]

            batch_examples = slice_examples[
                i:i + batch_size
            ]

            (
                hypotheses,
                elapsed,
                peak_memory,
            ) = transcribe_batch(
                model=model,
                    audio_list=batch_audios,
                model_batch_size=batch_size,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                sample_rate=MODEL_SAMPLE_RATE,
            )

            per_sample_elapsed = (
                elapsed / len(batch_audios)
            )

            for (
                example,
                audio,
                hypothesis,
            ) in zip(
                batch_examples,
                batch_audios,
                hypotheses,
            ):
                duration = (
                    len(audio)
                    / MODEL_SAMPLE_RATE
                )

                record = {
                    "audio": str(
                        example.audio
                    ),
                    "reference": (
                        example.reference
                    ),
                    "hypothesis": hypothesis,
                    "duration_seconds": duration,
                    "rtf": (
                        per_sample_elapsed
                        / max(duration, 1e-9)
                    ),
                    "peak_process_memory_mb": (
                        peak_memory
                    ),
                    "metrics": metric(
                        example.reference,
                        hypothesis,
                    ),
                    "script_aware": (
                        script_aware_metrics(
                            example.reference,
                            hypothesis,
                        )
                    ),
                }

                records.append(record)

                print(
                    f"[{slice_name}] "
                    f"{example.audio}: "
                    f"{hypothesis}"
                )

        results["slices"][slice_name] = {
            "count": len(records),
            "aggregate": (
                aggregate_metrics(records)
            ),
            "records": records,
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved evaluation results to "
        f"{output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SraVaani-1.0 ASR locally with "
            "true batched inference and long-audio chunking."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
    )

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--language",
        default="hi",
        help=(
            "Language metadata for evaluation. "
            "SraVaani transcribes multilingual audio directly; "
            "language is retained as evaluation metadata."
        ),
    )

    parser.add_argument(
        "--file",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).parent
            / "sravaani_transcriptions.jsonl"
        ),
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=(
            Path(__file__).parent
            / "data"
            / "gt.txt"
        ),
    )

    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=(
            Path(__file__).parent
            / "sravaani_evaluation.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Actual model inference batch size. "
            "Multiple audio paths/chunks are sent through model.transcribe() together."
        ),
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=(
            "Maximum duration of a single inference chunk. "
            "Longer files are split."
        ),
    )

    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=DEFAULT_OVERLAP_SECONDS,
        help=(
            "Overlap between adjacent long-audio chunks."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit(
            "--batch-size must be >= 1"
        )

    if args.chunk_seconds <= 0:
        raise SystemExit(
            "--chunk-seconds must be > 0"
        )

    if (
        args.overlap_seconds < 0
        or args.overlap_seconds >= args.chunk_seconds
    ):
        raise SystemExit(
            "--overlap-seconds must be >= 0 "
            "and smaller than --chunk-seconds"
        )

    audio_files = (
        [args.file]
        if args.file
        else find_audio_files(
            args.data_dir
        )
    )

    audio_files = [
        path
        for path in audio_files
        if path.is_file()
    ]

    if not audio_files:
        raise SystemExit(
            "No supported audio files were found."
        )

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model = build_model(
        args.model_id
    )

    print(
        f"SraVaani configuration: "
        f"batch_size={args.batch_size}, "
        f"chunk_seconds={args.chunk_seconds}, "
        f"overlap_seconds={args.overlap_seconds}"
    )

    # ---------------------------------------------------------------
    # Evaluation mode
    # ---------------------------------------------------------------

    if args.evaluate:
        examples = find_examples(
            args.data_dir,
            args.ground_truth,
        )

        if not examples:
            raise SystemExit(
                "No audio/reference pairs found "
                "for evaluation."
            )

        run_evaluation(
            model=model,
            model_id=args.model_id,
            examples=examples,
            language=args.language,
            output_path=args.evaluation_output,
            batch_size=args.batch_size,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )

        return

    # ---------------------------------------------------------------
    # Normal transcription mode
    # ---------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for i in range(
            0,
            len(audio_files),
            args.batch_size,
        ):
            batch_files = audio_files[
                i:i + args.batch_size
            ]

            batch_audios = []

            for audio_file in batch_files:
                audio, sample_rate = (
                    load_audio(audio_file)
                )

                audio = resample(
                    audio,
                    sample_rate,
                    MODEL_SAMPLE_RATE,
                )

                batch_audios.append(
                    np.asarray(
                        audio,
                        dtype=np.float32,
                    )
                )

            (
                texts,
                _,
                _,
            ) = transcribe_batch(
                model=model,
                    audio_list=batch_audios,
                model_batch_size=args.batch_size,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
                sample_rate=MODEL_SAMPLE_RATE,
            )

            for (
                audio_file,
                text,
            ) in zip(
                batch_files,
                texts,
            ):
                print(
                    f"{audio_file}: {text}"
                )

                output_file.write(
                    json.dumps(
                        {
                            "file": str(
                                audio_file
                            ),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()