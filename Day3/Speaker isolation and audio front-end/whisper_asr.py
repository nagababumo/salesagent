from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperASR:
    """Whisper-large-v3-turbo only. No audio-front-end logic belongs here."""

    def __init__(self, model_id: str, language: str = "auto"):
        use_cuda = torch.cuda.is_available()
        self.device = "cuda:0" if use_cuda else "cpu"
        self.dtype = torch.float16 if use_cuda else torch.float32
        self.language = None if language.lower() == "auto" else language

        print(f"[Whisper] loading {model_id}")
        print(f"[Whisper] device={self.device}")

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.dtype,
            device=0 if use_cuda else -1,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        if audio.size == 0:
            return "", 0.0, None

        audio = audio.astype(np.float32)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        generate_kwargs = {"task": "transcribe"}
        if self.language is not None:
            generate_kwargs["language"] = self.language

        result = self.pipe(
            {"raw": audio, "sampling_rate": sample_rate},
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs=generate_kwargs,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mb = None

        elapsed = time.perf_counter() - start
        return result["text"].strip(), elapsed, peak_mb
