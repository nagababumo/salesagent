# Modular Speaker-Isolation + Whisper Benchmark

Each feature is isolated into its own Python file. `main.py` only connects the modules.

## Files

- `main.py` - CLI/orchestrator
- `config.py` - model/stage/default configuration
- `audio_utils.py` - loading, resampling, mono conversion, saving
- `channel_separator.py` - stereo channel isolation
- `vad_processor.py` - Silero VAD
- `diarization_processor.py` - pyannote diarization + target speaker extraction
- `denoiser.py` - DeepFilterNet enhancement
- `whisper_asr.py` - Whisper-large-v3-turbo STT
- `metrics.py` - WER/CER
- `experiment_runner.py` - connects selected modules and records measurements

## Examples

Baseline:

```bash
python main.py --file call.wav --language hi --stage raw --reference "हाँ"
```

Right channel:

```bash
python main.py --file stereo_call.wav --language hi --stage right_channel
```

Short-answer VAD:

```bash
python main.py \
  --file call.wav \
  --language hi \
  --stage vad \
  --vad-threshold 0.4 \
  --vad-min-speech-ms 80 \
  --vad-min-silence-ms 120 \
  --vad-padding-ms 100 \
  --reference "हाँ"
```

Diarization:

```bash
export HF_TOKEN="hf_xxx"
python main.py \
  --file mono_call.wav \
  --language hi \
  --stage diarization_vad \
  --target-speaker SPEAKER_01
```

Everything:

```bash
python main.py \
  --file call.wav \
  --language hi \
  --stage all \
  --target-speaker SPEAKER_01 \
  --vad-threshold 0.4 \
  --vad-min-speech-ms 80 \
  --vad-min-silence-ms 120 \
  --vad-padding-ms 100 \
  --save-audio
```
