from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audio_utils import save_wav, ensure_16k
from config import MODEL_SR
from metrics import calculate_metrics


@dataclass
class ExperimentResult:
    stage: str
    file: str
    reference: str | None
    hypothesis: str
    original_duration_seconds: float
    processed_duration_seconds: float
    preprocessing_time_seconds: float
    whisper_time_seconds: float
    total_time_seconds: float
    rtf_whisper: float
    rtf_total: float
    wer: float | None
    cer: float | None
    vad_segments: int | None
    vad_speech_seconds: float | None
    selected_speaker: str | None
    speaker_durations: dict | None
    gpu_memory_mb: float | None


STAGE_CONFIG = {
    "raw": {"channel": "average", "vad": False, "diarization": False, "denoise": False},
    "right_channel": {"channel": "right", "vad": False, "diarization": False, "denoise": False},
    "vad": {"channel": "average", "vad": True, "diarization": False, "denoise": False},
    "right_channel_vad": {"channel": "right", "vad": True, "diarization": False, "denoise": False},
    "diarization": {"channel": "average", "vad": False, "diarization": True, "denoise": False},
    "diarization_vad": {"channel": "average", "vad": True, "diarization": True, "denoise": False},
    "right_channel_denoise": {"channel": "right", "vad": False, "diarization": False, "denoise": True},
    "right_channel_vad_denoise": {"channel": "right", "vad": True, "diarization": False, "denoise": True},
    "diarization_denoise": {"channel": "average", "vad": False, "diarization": True, "denoise": True},
    "diarization_vad_denoise": {"channel": "average", "vad": True, "diarization": True, "denoise": True},
}


class ExperimentRunner:
    def __init__(
        self,
        whisper,
        channel_separator=None,
        vad=None,
        diarizer=None,
        denoiser=None,
        target_speaker="longest",
        output_dir=Path("speaker_isolation_results"),
        save_audio=False,
    ):
        self.whisper = whisper
        self.channel_separator = channel_separator
        self.vad = vad
        self.diarizer = diarizer
        self.denoiser = denoiser
        self.target_speaker = target_speaker
        self.output_dir = Path(output_dir)
        self.save_audio = save_audio
        self.audio_dir = self.output_dir / "processed_audio"
        if save_audio:
            self.audio_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(
        self,
        stage: str,
        audio: np.ndarray,
        sample_rate: int,
        source_name: str,
        reference: str | None,
    ) -> ExperimentResult:
        cfg = STAGE_CONFIG[stage]
        original_duration = len(audio) / sample_rate
        work = audio
        selected_speaker = None
        speaker_durations = None
        vad_segments = None
        vad_speech_seconds = None

        prep_start = time.perf_counter()

        # 1) Channel separation
        if self.channel_separator is None:
            raise RuntimeError("Channel separator is not loaded.")
        work = self.channel_separator.process(work, cfg["channel"])

        # 2) Diarization
        if cfg["diarization"]:
            if self.diarizer is None:
                raise RuntimeError("Diarization requested but diarizer is not loaded.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)

            work, selected_speaker, _, speaker_durations = self.diarizer.process(
                work,
                sample_rate,
                target_speaker=self.target_speaker,
                temp_wav=temp_path,
            )

        # 3) VAD
        if cfg["vad"]:
            if self.vad is None:
                raise RuntimeError("VAD requested but VAD module is not loaded.")
            work, timestamps = self.vad.process(work, sample_rate)
            vad_segments = len(timestamps)
            vad_speech_seconds = len(work) / sample_rate

        # 4) Denoising
        if cfg["denoise"]:
            if self.denoiser is None:
                raise RuntimeError("Denoising requested but denoiser is not loaded.")
            work, processed_sr = self.denoiser.process(work, sample_rate)
            sample_rate = processed_sr

            # Whisper will receive 16k input; resampling is deliberately kept
            # outside denoiser.py so that that module only owns enhancement.
            if sample_rate != MODEL_SR:
                work, sample_rate = ensure_16k(work, sample_rate, MODEL_SR)

        prep_time = time.perf_counter() - prep_start
        processed_duration = len(work) / max(sample_rate, 1)

        if self.save_audio:
            save_wav(
                self.audio_dir / f"{Path(source_name).stem}__{stage}.wav",
                work,
                sample_rate,
            )

        hypothesis, whisper_time, gpu_memory = self.whisper.transcribe(
            work, sample_rate
        )

        total_time = prep_time + whisper_time
        wer_value, cer_value = calculate_metrics(reference, hypothesis)

        return ExperimentResult(
            stage=stage,
            file=source_name,
            reference=reference,
            hypothesis=hypothesis,
            original_duration_seconds=original_duration,
            processed_duration_seconds=processed_duration,
            preprocessing_time_seconds=prep_time,
            whisper_time_seconds=whisper_time,
            total_time_seconds=total_time,
            rtf_whisper=whisper_time / max(processed_duration, 1e-9),
            rtf_total=total_time / max(original_duration, 1e-9),
            wer=wer_value,
            cer=cer_value,
            vad_segments=vad_segments,
            vad_speech_seconds=vad_speech_seconds,
            selected_speaker=selected_speaker,
            speaker_durations=speaker_durations,
            gpu_memory_mb=gpu_memory,
        )

    def run(self, stages, audio, sample_rate, source_name, reference=None):
        results = []
        for stage in stages:
            print("\n" + "=" * 70)
            print(f"STAGE: {stage}")
            print("=" * 70)
            try:
                result = self.run_stage(
                    stage, audio, sample_rate, source_name, reference
                )
                results.append(result)
                print(f"Hypothesis : {result.hypothesis}")
                print(f"WER        : {result.wer if result.wer is not None else 'N/A'}")
                print(f"CER        : {result.cer if result.cer is not None else 'N/A'}")
                print(f"Total RTF  : {result.rtf_total:.4f}")
            except Exception as exc:
                print(f"[{stage}] FAILED: {exc}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / "experiment_results.json"
        payload = {"results": [asdict(r) for r in results]}
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("\nSUMMARY")
        print(f"{'Stage':35s} {'WER':>10s} {'CER':>10s} {'RTF':>10s}")
        print("-" * 70)
        for r in results:
            wer_s = f"{r.wer:.4f}" if r.wer is not None else "N/A"
            cer_s = f"{r.cer:.4f}" if r.cer is not None else "N/A"
            print(f"{r.stage:35s} {wer_s:>10s} {cer_s:>10s} {r.rtf_total:>10.4f}")

        print(f"\nSaved: {result_path}")
        return results
