from __future__ import annotations

import argparse
import json
import time
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

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
MODEL_SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

# Detect GPU hardware acceleration in Google Colab
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_audio_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def build_model(model_id: str):
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(DEVICE)
        model.eval()
        return model
    except OSError as error:
        message = str(error)
        if "gated repo" in message.lower() or "403" in message:
            raise SystemExit(
                f"Hugging Face access is required for '{model_id}'. "
                "Accept the model's access terms, then authenticate this environment "
                "with `huggingface-cli login` before running again."
            ) from error
        raise


def get_peak_memory_mb() -> float:
    """Get peak RSS RAM memory across Linux (Colab) environments."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


CHUNK_SECONDS = 45.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * MODEL_SAMPLE_RATE)


def transcribe_long_audio(
    model,
    audio: np.ndarray,
    language: str,
    chunk_seconds: float = CHUNK_SECONDS,
) -> str:
    audio = np.asarray(audio, dtype=np.float32).flatten()

    chunk_samples = int(chunk_seconds * MODEL_SAMPLE_RATE)

    parts = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]

        if len(chunk) == 0:
            continue

        wav = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE)

        result = model(wav, language, "ctc")

        if result is not None:
            parts.append(str(result).strip())

    return " ".join(part for part in parts if part)


def transcribe_batch(model, audio_list, language):
    if not audio_list:
        return [], 0.0, 0.0

    started = time.perf_counter()
    texts = []

    with torch.inference_mode():
        for audio in audio_list:
            text = transcribe_long_audio(
                model,
                audio,
                language,
                chunk_seconds=45.0,
            )
            texts.append(text)

    elapsed = time.perf_counter() - started
    peak_memory = get_peak_memory_mb()

    return texts, elapsed, peak_memory
    
def transcribe_audio(model, audio: np.ndarray, sample_rate: int, language: str):
    texts, elapsed, peak_memory = transcribe_batch(model, [audio], language)
    return texts[0], elapsed, peak_memory


def run_evaluation(model, model_id, examples, language, output_path, batch_size=1):
    results = {
        "model_id": model_id,
        "decoder": "IndicConformer direct CTC decoding",
        "normalization": "casefold; remove ZWNJ/ZWJ; remove punctuation; collapse whitespace; remove number separators; preserve Devanagari vowel signs and scripts",
        "slices": {},
    }

    # Pre-load and cache audio files once to prevent repeated disk I/O across slices
    print("Pre-loading and resampling base audio samples...")
    base_audios = []
    for ex in examples:
        audio, sample_rate = load_audio(ex.audio)
        audio = resample(audio, sample_rate, MODEL_SAMPLE_RATE)
        base_audios.append(audio)

    for slice_name in ("clean", "telephony", "code_mixed"):
        records = []
        slice_indices = []
        slice_examples = []
        slice_audios = []

        for index, example in enumerate(examples):
            if slice_name == "code_mixed" and not script_aware_metrics(example.reference, "")["is_code_mixed"]:
                continue

            audio = base_audios[index]
            if slice_name == "telephony":
                audio, _ = make_telephony_audio(audio, MODEL_SAMPLE_RATE, index)
                audio = resample(audio, MODEL_SAMPLE_RATE, MODEL_SAMPLE_RATE)

            slice_indices.append(index)
            slice_examples.append(example)
            slice_audios.append(audio)

        # Process slice in batches safely
        for i in range(0, len(slice_audios), batch_size):
            batch_audios = slice_audios[i : i + batch_size]
            batch_examples = slice_examples[i : i + batch_size]

            hypotheses, elapsed, peak_memory = transcribe_batch(
                model, batch_audios, language
            )
            per_sample_elapsed = elapsed / len(batch_audios)

            for example, audio, hypothesis in zip(batch_examples, batch_audios, hypotheses):
                duration = len(audio) / MODEL_SAMPLE_RATE
                record = {
                    "audio": str(example.audio),
                    "reference": example.reference,
                    "hypothesis": hypothesis,
                    "duration_seconds": duration,
                    "rtf": per_sample_elapsed / max(duration, 1e-9),
                    "peak_process_memory_mb": peak_memory,
                    "metrics": metric(example.reference, hypothesis),
                    "script_aware": script_aware_metrics(example.reference, hypothesis),
                }
                records.append(record)
                print(f"[{slice_name}] {example.audio}: {hypothesis}")

        results["slices"][slice_name] = {
            "count": len(records),
            "aggregate": aggregate_metrics(records),
            "records": records,
        }

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IndicConformer CTC ASR locally.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "indicconfor_transcriptions.jsonl")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ground-truth", type=Path, default=Path(__file__).parent / "data" / "gt.txt")
    parser.add_argument("--evaluation-output", type=Path, default=Path(__file__).parent / "indicconfor_evaluation.json")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()

    audio_files = [args.file] if args.file else find_audio_files(args.data_dir)
    audio_files = [path for path in audio_files if path.is_file()]
    if not audio_files:
        raise SystemExit("No supported audio files were found.")

    model = build_model(args.model_id)
    if args.evaluate:
        examples = find_examples(args.data_dir, args.ground_truth)
        if not examples:
            raise SystemExit("No audio/reference pairs found for evaluation.")
        run_evaluation(model, args.model_id, examples, args.language, args.evaluation_output, batch_size=args.batch_size)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for i in range(0, len(audio_files), args.batch_size):
            batch_files = audio_files[i : i + args.batch_size]
            batch_audios = []

            for audio_file in batch_files:
                audio, sample_rate = load_audio(audio_file)
                audio = resample(audio, sample_rate, MODEL_SAMPLE_RATE)
                batch_audios.append(audio)

            texts, _, _ = transcribe_batch(model, batch_audios, args.language)

            for audio_file, text in zip(batch_files, texts):
                print(f"{audio_file}: {text}")
                output_file.write(json.dumps({"file": str(audio_file), "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()