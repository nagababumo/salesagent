from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from audio_utils import ensure_16k


class SraVaaniASR:
    """ARTPARK SraVaani Hindi ASR backend."""

    def __init__(self, model_id: str = "ARTPARK-IISc/SraVaani-1.0"):
        self.model_id = model_id
        self.adapter = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        token = os.environ.get("HF_TOKEN")
        model_dir = snapshot_download(repo_id=model_id, token=token)
        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            token=token,
        ).to(self.device).eval()

    @staticmethod
    def _extract_text(hypothesis) -> str:
        if hypothesis is None:
            return ""
        if hasattr(hypothesis, "text"):
            return str(hypothesis.text).strip()
        return str(hypothesis).strip()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        if audio.size == 0:
            return "", 0.0, None

        audio, sample_rate = ensure_16k(audio.astype(np.float32), sample_rate)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
            with torch.inference_mode():
                hypotheses = self.model.transcribe(
                    [str(wav_path)], batch_size=1, return_hypotheses=True
                )
            if isinstance(hypotheses, tuple):
                hypotheses = hypotheses[0]
            if not isinstance(hypotheses, (list, tuple)):
                hypotheses = [hypotheses]
            hypothesis = self._extract_text(hypotheses[0]) if hypotheses else ""
        finally:
            wav_path.unlink(missing_ok=True)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory_mb = None

        return hypothesis, time.perf_counter() - started, peak_memory_mb