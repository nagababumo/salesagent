from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from evaluate_asr import (
    aggregate_metrics,
    find_examples,
    load_audio,
    make_telephony_audio,
    metric,
    resample,
    script_aware_metrics,
)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
MODEL_ID = "openai/whisper-large-v3-turbo"


def find_audio_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def build_pipeline(model_id, language="hi"):
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    if model.generation_config.suppress_tokens == []:
        model.generation_config.suppress_tokens = None

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if use_cuda else -1,
    )
    return transcriber, processor.feature_extractor.sampling_rate


def transcribe_audio(transcriber, audio, sample_rate, language):
    started = time.perf_counter()
    result = transcriber(
        {"raw": audio, "sampling_rate": sample_rate},
        chunk_length_s=30,
        stride_length_s=5,
    )
    elapsed = time.perf_counter() - started
    process = psutil.Process()
    peak_memory = process.memory_info().peak_wset / (1024 * 1024)
    return result["text"].strip(), elapsed, peak_memory


def run_evaluation(transcriber, model_id, model_sample_rate, examples, language, output_path):
    results = {
        "model_id": model_id,
        "normalization": "casefold; remove ZWNJ/ZWJ; remove punctuation; collapse whitespace; remove number separators; preserve Devanagari vowel signs and scripts",
        "slices": {},
    }
    for slice_name in ("clean", "telephony", "code_mixed"):
        records = []
        for index, example in enumerate(examples):
            if slice_name == "code_mixed" and not script_aware_metrics(example.reference, "")["is_code_mixed"]:
                continue
            audio, sample_rate = load_audio(example.audio)
            if slice_name == "telephony":
                audio, sample_rate = make_telephony_audio(audio, sample_rate, index)
            audio = resample(audio, sample_rate, model_sample_rate)
            hypothesis, elapsed, peak_memory = transcribe_audio(
                transcriber, audio, model_sample_rate, language
            )
            duration = len(audio) / model_sample_rate
            record = {
                "audio": str(example.audio),
                "reference": example.reference,
                "hypothesis": hypothesis,
                "duration_seconds": duration,
                "rtf": elapsed / max(duration, 1e-9),
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
    parser = argparse.ArgumentParser(description="Transcribe local audio with Whisper.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory searched recursively for audio files.",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--language", default="hi", help="Whisper language prompt.")
    parser.add_argument(
        "--file",
        type=Path,
        help="Transcribe one audio file instead of scanning --data-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "transcriptions.jsonl",
        help="JSONL file receiving one result per audio file.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run clean, telephony, and code-mixed evaluation.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent / "data" / "gt.txt",
        help="Ground-truth text used with --evaluate.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=Path(__file__).parent / "evaluation_results.json",
        help="Single JSON file containing transcriptions and metrics.",
    )
    args = parser.parse_args()

    audio_files = [args.file] if args.file else find_audio_files(args.data_dir)
    audio_files = [path for path in audio_files if path.is_file()]
    if not audio_files:
        raise SystemExit("No supported audio files were found.")

    transcriber, model_sample_rate = build_pipeline(args.model_id, args.language)
    if args.evaluate:
        examples = find_examples(args.data_dir, args.ground_truth)
        if not examples:
            raise SystemExit("No audio/reference pairs found for evaluation.")
        run_evaluation(
            transcriber,
            args.model_id,
            model_sample_rate,
            examples,
            args.language,
            args.evaluation_output,
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for audio_file in audio_files:
            audio, sample_rate = load_audio(audio_file)
            audio = resample(audio, sample_rate, model_sample_rate)
            text, _, _ = transcribe_audio(transcriber, audio, model_sample_rate, args.language)
            record = {"file": str(audio_file), "text": text}
            print(f"{audio_file}: {record['text']}")
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()