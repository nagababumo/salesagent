from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: Path):
    audio, sample_rate = sf.read(path, always_2d=False)
    return audio.astype(np.float32), int(sample_rate)


def ensure_16k(audio: np.ndarray, sample_rate: int, target_sr: int = 16000):
    if sample_rate == target_sr:
        return audio.astype(np.float32), sample_rate

    gcd = np.gcd(sample_rate, target_sr)
    up = target_sr // gcd
    down = sample_rate // gcd

    if audio.ndim == 1:
        out = resample_poly(audio, up, down)
    else:
        out = np.stack(
            [resample_poly(audio[:, c], up, down) for c in range(audio.shape[1])],
            axis=1,
        )

    return out.astype(np.float32), target_sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def select_channel(audio: np.ndarray, channel: str) -> np.ndarray:
    """Select one channel or downmix stereo to mono for Whisper/front-end models."""
    if audio.ndim == 1:
        if channel == "right":
            raise ValueError("Right-channel isolation requested, but input is mono.")
        return audio.astype(np.float32)

    if audio.shape[1] < 2 and channel == "right":
        raise ValueError("Right-channel isolation requires at least 2 channels.")

    if channel == "left":
        return audio[:, 0].astype(np.float32)
    if channel == "right":
        return audio[:, 1].astype(np.float32)
    if channel == "average" or channel == "mono":
        return to_mono(audio)

    raise ValueError(f"Unknown channel mode: {channel}")


def save_wav(path: Path, audio: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)
