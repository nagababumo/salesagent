from __future__ import annotations

import argparse
import os
from pathlib import Path

from audio_utils import ensure_16k, load_audio
from channel_separator import ChannelSeparator
from config import (
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_SPEECH_MS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEECH_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
    DIARIZATION_MODEL,
    MODEL_ID,
    STAGES,
)
from denoiser import DeepFilterDenoiser
from diarization_processor import SpeakerDiarizer
from experiment_runner import ExperimentRunner
from vad_processor import SileroVAD
from whisper_asr import WhisperASR


def parse_args():
    p = argparse.ArgumentParser(
        description="Modular speaker isolation/front-end benchmark with Whisper-large-v3-turbo."
    )
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--language", default="hi", help="Whisper language, e.g. hi, te, en, or auto")
    p.add_argument("--reference", default=None)
    p.add_argument("--stage", choices=STAGES + ["all"], default="raw")

    # VAD controls
    p.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD)
    p.add_argument("--vad-min-speech-ms", type=int, default=DEFAULT_MIN_SPEECH_MS)
    p.add_argument("--vad-min-silence-ms", type=int, default=DEFAULT_MIN_SILENCE_MS)
    p.add_argument("--vad-padding-ms", type=int, default=DEFAULT_SPEECH_PAD_MS)

    # Diarization controls
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--diarization-model", default=DIARIZATION_MODEL)
    p.add_argument("--target-speaker", default="longest")

    # Output
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-audio", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.file.is_file():
        raise SystemExit(f"Input file does not exist: {args.file}")

    stages = STAGES if args.stage == "all" else [args.stage]

    # Load once; all stages start from the same original waveform.
    audio, sr = load_audio(args.file)
    audio, sr = ensure_16k(audio, sr, 16000)

    print(f"Input        : {args.file}")
    print(f"Sample rate  : {sr}")
    print(f"Channels     : {1 if audio.ndim == 1 else audio.shape[1]}")
    print(f"Duration     : {audio.shape[0] / sr:.2f}s")

    # Load only modules needed by selected stages.
    need_vad = any("vad" in stage for stage in stages)
    need_diar = any("diarization" in stage for stage in stages)
    need_denoise = any("denoise" in stage for stage in stages)

    whisper = WhisperASR(args.model_id, args.language)
    channel_separator = ChannelSeparator()

    vad = None
    if need_vad:
        print("[Init] Silero VAD")
        vad = SileroVAD(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_padding_ms,
        )

    diarizer = None
    if need_diar:
        if not args.hf_token:
            raise SystemExit(
                "Diarization requested but no HF token was supplied. "
                "Use --hf-token or export HF_TOKEN=."
            )
        print(f"[Init] Pyannote: {args.diarization_model}")
        diarizer = SpeakerDiarizer(args.hf_token, args.diarization_model)

    denoiser = None
    if need_denoise:
        print("[Init] DeepFilterNet")
        denoiser = DeepFilterDenoiser()

    runner = ExperimentRunner(
        whisper=whisper,
        channel_separator=channel_separator,
        vad=vad,
        diarizer=diarizer,
        denoiser=denoiser,
        target_speaker=args.target_speaker,
        output_dir=args.output_dir,
        save_audio=args.save_audio,
    )

    runner.run(
        stages=stages,
        audio=audio,
        sample_rate=sr,
        source_name=args.file.name,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
