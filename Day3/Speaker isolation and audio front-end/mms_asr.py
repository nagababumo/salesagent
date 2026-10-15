from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from audio_utils import ensure_16k


class MMSASR:
    """Hindi-adapted MMS ASR backend for the front-end experiments."""

    def __init__(self, model_id: str = "facebook/mms-1b-all", language: str = "hi"):
        if language.lower() not in {"hi", "hindi", "hin"}:
            raise ValueError("facebook/mms-1b-all is configured here for Hindi only.")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.adapter = "hin"
        self.sample_rate = 16000
        self.chunk_seconds = 30
        self.overlap_seconds = 2

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.set_target_lang("hin")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.load_adapter("hin")
        self.model.to(self.device).eval()

    @staticmethod
    def _merge_overlapping_text(texts: list[str]) -> str:
        merged: list[str] = []
        for text in texts:
            words = text.split()
            overlap = 0
            for size in range(min(12, len(merged), len(words)), 0, -1):
                if [word.casefold() for word in merged[-size:]] == [
                    word.casefold() for word in words[:size]
                ]:
                    overlap = size
                    break
            merged.extend(words[overlap:])
        return " ".join(merged).strip()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        if audio.size == 0:
            return "", 0.0, None

        audio, sample_rate = ensure_16k(audio.astype(np.float32), sample_rate)
        chunk_size = self.chunk_seconds * self.sample_rate
        step_size = (self.chunk_seconds - self.overlap_seconds) * self.sample_rate
        chunks = [audio[start:start + chunk_size] for start in range(0, len(audio), step_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 0:
            chunks.pop()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        texts = []
        with torch.inference_mode():
            for chunk in chunks:
                inputs = self.processor(
                    chunk,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                logits = self.model(**inputs).logits
                texts.append(self.processor.batch_decode(logits.argmax(dim=-1))[0].strip())

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory_mb = None

        return self._merge_overlapping_text(texts), time.perf_counter() - started, peak_memory_mb