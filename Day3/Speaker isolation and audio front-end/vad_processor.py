from __future__ import annotations

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


class SileroVAD:
    """Silero VAD with all important short-utterance knobs exposed."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        self.model = load_silero_vad()
        if torch.cuda.is_available() and hasattr(self.model, "to"):
            self.model = self.model.to("cuda")

    def process(self, audio: np.ndarray, sample_rate: int = 16000):
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD expects 8 kHz or 16 kHz audio.")

        if audio.size == 0:
            return audio.astype(np.float32), []

        waveform = torch.from_numpy(audio.astype(np.float32))

        timestamps = get_speech_timestamps(
            waveform,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        chunks = [
            waveform[int(seg["start"]): int(seg["end"])]
            for seg in timestamps
            if int(seg["end"]) > int(seg["start"])
        ]

        if not chunks:
            return np.zeros(0, dtype=np.float32), timestamps

        speech = torch.cat(chunks).cpu().numpy().astype(np.float32)
        return speech, timestamps
