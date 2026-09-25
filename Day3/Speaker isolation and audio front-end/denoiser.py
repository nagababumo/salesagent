from __future__ import annotations

import numpy as np
import torch


class DeepFilterDenoiser:
    """DeepFilterNet wrapper. No VAD/diarization logic belongs here."""

    def __init__(self):
        from df.enhance import enhance, init_df

        self.enhance_fn = enhance
        self.model, self.df_state, _ = init_df()

    def process(self, audio: np.ndarray, sample_rate: int = 16000):
        from df.enhance import load_audio

        # DeepFilterNet's helper loads/resamples according to df_state.
        # We use a temporary WAV so the exact frontend behavior is simple to reproduce.
        import tempfile
        from pathlib import Path
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sf.write(temp_path, audio.astype(np.float32), sample_rate)
            waveform, _ = load_audio(temp_path, sr=self.df_state.sr())
            enhanced = self.enhance_fn(self.model, self.df_state, waveform)
        finally:
            temp_path.unlink(missing_ok=True)

        if torch.is_tensor(enhanced):
            enhanced = enhanced.detach().cpu().numpy()

        enhanced = np.asarray(enhanced).squeeze().astype(np.float32)
        return enhanced, int(self.df_state.sr())
