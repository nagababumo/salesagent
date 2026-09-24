from __future__ import annotations

import argparse
import json
from pathlib import Path

from whisperltv3 import (
	build_pipeline,
	find_audio_files,
	find_examples,
	run_evaluation,
	transcribe_audio,
)
from evaluate_asr import load_audio, resample


MODEL_ID = "vasista22/whisper-hindi-medium"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run the fine-tuned Hindi Whisper model on local audio."
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).parent / "data",
		help="Directory searched recursively for audio files.",
	)
	parser.add_argument(
		"--model-id",
		default=MODEL_ID,
		help="Hugging Face model ID; defaults to the fine-tuned Hindi model.",
	)
	parser.add_argument("--language", default="hi", help="Whisper language prompt.")
	parser.add_argument("--file", type=Path, help="Transcribe one local audio file.")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).parent / "whisper_fintuned_transcriptions.jsonl",
		help="JSONL output for normal transcription mode.",
	)
	parser.add_argument(
		"--evaluate",
		action="store_true",
		help="Run clean, telephony, and code-mixed evaluation.",
	)
	parser.add_argument(
		"--ground-truth",
		type=Path,
		default=Path(__file__).parent / "data" / "gt.txt",
		help="Manifest/ground truth used with --evaluate.",
	)
	parser.add_argument(
		"--evaluation-output",
		type=Path,
		default=Path(__file__).parent / "whisper_fintuned_evaluation.json",
		help="JSON output containing transcriptions and metrics.",
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
			text, _, _ = transcribe_audio(
				transcriber, audio, model_sample_rate, args.language
			)
			print(f"{audio_file}: {text}")
			output_file.write(
				json.dumps(
					{"file": str(audio_file), "text": text}, ensure_ascii=False
				)
				+ "\n"
			)


if __name__ == "__main__":
	main()
