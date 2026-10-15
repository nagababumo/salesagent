from __future__ import annotations

import os
import time

import numpy as np
import torch
import huggingface_hub.file_download as hub_file_download
from transformers import AutoModel

from audio_utils import ensure_16k


class IndicConformerASR:
    """AI4Bharat IndicConformer Hindi CTC backend via ONNX Runtime."""

    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual"):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("IndicConformer requires HF_TOKEN from the repository .env.")

        self.model_id = model_id
        self.adapter = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.runtime = "ONNX Runtime CPUExecutionProvider"
        hub_file_download.are_symlinks_supported = lambda _path: False
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
        ).to(self.device).eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        if audio.size == 0:
            return "", 0.0, None

        audio, _ = ensure_16k(audio.astype(np.float32), sample_rate)
        waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        hypothesis = self.model(waveform, "hi", "ctc")
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return str(hypothesis).strip(), time.perf_counter() - started, None