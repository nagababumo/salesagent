from pathlib import Path

MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_SR = 16000

# Stage names used by main.py
STAGES = [
    "raw",
    "right_channel",
    "vad",
    "right_channel_vad",
    "diarization",
    "diarization_vad",
    "right_channel_denoise",
    "right_channel_vad_denoise",
    "diarization_denoise",
    "diarization_vad_denoise",
]

DEFAULT_OUTPUT_DIR = Path("speaker_isolation_results")

DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 100
DEFAULT_SPEECH_PAD_MS = 30

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
