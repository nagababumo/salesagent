from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pyannote.audio import Pipeline

from audio_utils import save_wav


class SpeakerDiarizer:
    """Mono-audio diarization and target-speaker extraction."""

    def __init__(
        self,
        hf_token: str,
        model_id: str = "pyannote/speaker-diarization-community-1",
    ):
        self.pipeline = Pipeline.from_pretrained(model_id, token=hf_token)

        try:
            import torch
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    def diarize_file(self, audio_path: Path):
        output = self.pipeline(str(audio_path))

        if hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization
        else:
            annotation = output.speaker_diarization

        segments = []
        speaker_durations = {}

        for turn, speaker in annotation:
            start = float(turn.start)
            end = float(turn.end)
            duration = max(0.0, end - start)

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(speaker),
                    "duration": duration,
                }
            )
            speaker_durations[str(speaker)] = (
                speaker_durations.get(str(speaker), 0.0) + duration
            )

        return segments, speaker_durations

    @staticmethod
    def choose_speaker(speaker_durations: dict, target_speaker: str = "longest") -> Optional[str]:
        if not speaker_durations:
            return None
        if target_speaker == "longest":
            return max(speaker_durations, key=speaker_durations.get)
        if target_speaker not in speaker_durations:
            raise ValueError(
                f"Speaker '{target_speaker}' not found. "
                f"Available speakers: {sorted(speaker_durations)}"
            )
        return target_speaker

    @staticmethod
    def extract_speaker(
        audio: np.ndarray,
        sample_rate: int,
        segments: list[dict],
        target_speaker: str,
    ) -> np.ndarray:
        pieces = []
        for seg in segments:
            if seg["speaker"] != target_speaker:
                continue
            start = max(0, int(round(seg["start"] * sample_rate)))
            end = min(len(audio), int(round(seg["end"] * sample_rate)))
            if end > start:
                pieces.append(audio[start:end])

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_speaker: str = "longest",
        temp_wav: Optional[Path] = None,
    ):
        if temp_wav is None:
            raise ValueError("Provide a temp_wav path for pyannote input audio.")

        save_wav(temp_wav, audio, sample_rate)
        try:
            segments, speaker_durations = self.diarize_file(temp_wav)
        finally:
            if temp_wav.exists():
                temp_wav.unlink()

        selected = self.choose_speaker(speaker_durations, target_speaker)
        if selected is None:
            return np.zeros(0, dtype=np.float32), selected, segments, speaker_durations

        extracted = self.extract_speaker(
            audio, sample_rate, segments, selected
        )
        return extracted, selected, segments, speaker_durations
