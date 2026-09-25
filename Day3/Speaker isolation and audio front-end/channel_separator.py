from __future__ import annotations

import numpy as np

from audio_utils import select_channel


class ChannelSeparator:
    """Stereo channel isolation. Keep this file limited to channel operations."""

    def process(self, audio: np.ndarray, channel: str = "right") -> np.ndarray:
        return select_channel(audio, channel)
