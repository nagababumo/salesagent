from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForRNNT, AutoProcessor

from evaluate_asr import (
    aggregate_metrics,
    find_examples,
    load_audio,
    make_telephony_audio,
    metric,
    resample,
    script_aware_metrics,
)


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

DEFAULT_LANGUAGE = "hi-IN"

# Supported by the Nemotron processor.
# 0  -> 80 ms
# 3  -> 320 ms
# 6  -> 560 ms
# 13 -> 1120 ms
DEFAULT_LOOKAHEAD_TOKENS = 6

# Prevent the generation warning:
# "Using the model-agnostic default max_length..."
#
# This is a decoder-output limit, not the encoder's
# max_position_embeddings limit.
DEFAULT_MAX_NEW_TOKENS = 8192


def find_audio_files(data_dir: Path) -> list[Path]:
    """
    Recursively find supported audio files.
    """
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def get_process_memory_mb() -> float:
    """
    Return current process memory usage in MB.

    Colab/Linux generally exposes RSS.
    Windows may expose peak_wset.
    """

    process = psutil.Process()
    memory_info = process.memory_info()

    peak_wset = getattr(
        memory_info,
        "peak_wset",
        None,
    )

    if peak_wset is not None:
        return peak_wset / (1024 * 1024)

    return memory_info.rss / (1024 * 1024)


class NemotronTranscriber:
    """
    NVIDIA Nemotron 3.5 ASR Streaming 0.6B wrapper.

    Important:
        Long audio is processed through the model's streaming
        input-feature generator instead of sending the entire
        recording to model.generate() in one encoder pass.

    This prevents:
        ValueError:
        Sequence Length: XXXX has to be less or equal than
        config.max_position_embeddings 5000
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        language: str = DEFAULT_LANGUAGE,
        lookahead_tokens: int = DEFAULT_LOOKAHEAD_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:

        self.model_id = model_id
        self.language = language
        self.lookahead_tokens = lookahead_tokens
        self.max_new_tokens = max_new_tokens

        # ---------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        print()
        print("=" * 70)
        print("Loading NVIDIA Nemotron 3.5 ASR")
        print("=" * 70)
        print(f"Model              : {self.model_id}")
        print(f"Device             : {self.device}")
        print(f"Language           : {self.language}")
        print(f"Lookahead tokens   : {self.lookahead_tokens}")
        print(f"Max new tokens     : {self.max_new_tokens}")
        print()

        # ---------------------------------------------------------
        # PROCESSOR
        # ---------------------------------------------------------

        print("Loading processor...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id
        )

        # ---------------------------------------------------------
        # STREAMING CONFIGURATION
        # ---------------------------------------------------------
        #
        # NVIDIA's documented streaming API uses:
        #
        # processor.set_num_lookahead_tokens(...)
        #
        # and passes the same num_lookahead_tokens into
        # model.generate().
        # ---------------------------------------------------------

        supported_lookaheads = getattr(
            self.processor,
            "supported_num_lookahead_tokens",
            None,
        )

        if supported_lookaheads is not None:
            supported_lookaheads = list(
                supported_lookaheads
            )

            if (
                self.lookahead_tokens
                not in supported_lookaheads
            ):
                raise ValueError(
                    "Unsupported lookahead token value: "
                    f"{self.lookahead_tokens}. "
                    f"Supported values: {supported_lookaheads}"
                )

        self.processor.set_num_lookahead_tokens(
            self.lookahead_tokens
        )

        streaming_latency = getattr(
            self.processor,
            "streaming_latency_ms",
            None,
        )

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------

        print("Loading model...")

        self.model = AutoModelForRNNT.from_pretrained(
            self.model_id,
            device_map="auto",
        )

        self.model.eval()

        # ---------------------------------------------------------
        # SAMPLE RATE
        # ---------------------------------------------------------

        self.sample_rate = (
            self.processor
            .feature_extractor
            .sampling_rate
        )

        print(
            f"Model sampling rate: "
            f"{self.sample_rate} Hz"
        )

        if streaming_latency is not None:
            print(
                f"Streaming latency   : "
                f"{streaming_latency} ms"
            )

        # ---------------------------------------------------------
        # STREAMING CHUNK INFORMATION
        # ---------------------------------------------------------

        print()
        print(
            "Streaming first chunk samples:",
            self.processor.num_samples_first_audio_chunk,
        )

        print(
            "Streaming chunk samples:",
            self.processor.num_samples_per_audio_chunk,
        )

        print(
            "Streaming first mel frames:",
            self.processor.num_mel_frames_first_audio_chunk,
        )

        print(
            "Streaming mel frames/chunk:",
            self.processor.num_mel_frames_per_audio_chunk,
        )

        print()
        print("Model loaded successfully.")
        print("=" * 70)
        print()

    def _pad_audio(
        self,
        audio: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Zero-pad an audio chunk to the required number of samples.

        This is useful for the final partial streaming chunk.
        """

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if len(audio) >= target_length:
            return audio[:target_length]

        padding = target_length - len(audio)

        return np.pad(
            audio,
            (0, padding),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)

    def _streaming_feature_generator(
        self,
        audio: np.ndarray,
    ):
        """
        Generate model-sized streaming input feature chunks.

        This follows the streaming mechanism documented by
        Hugging Face/NVIDIA:

            first chunk
            ->
            subsequent fixed-size chunks
            ->
            reused cache inside model.generate()

        The entire recording therefore never becomes one giant
        encoder sequence.
        """

        first_chunk_samples = (
            self.processor.num_samples_first_audio_chunk
        )

        normal_chunk_samples = (
            self.processor.num_samples_per_audio_chunk
        )

        first_chunk = audio[
            :first_chunk_samples
        ]

        # Pad very short recordings so the first chunk has the
        # expected size.
        first_chunk = self._pad_audio(
            first_chunk,
            first_chunk_samples,
        )

        first_inputs = self.processor(
            first_chunk,
            sampling_rate=self.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self.language,
            return_tensors="pt",
        )

        first_inputs = first_inputs.to(
            self.model.device,
            dtype=self.model.dtype,
        )

        # ---------------------------------------------------------
        # FIRST CHUNK
        # ---------------------------------------------------------

        yield (
            first_inputs.input_features[
                :,
                :self.processor.num_mel_frames_first_audio_chunk,
                :,
            ]
        )

        # ---------------------------------------------------------
        # SUBSEQUENT CHUNKS
        # ---------------------------------------------------------

        mel_frame_idx = (
            self.processor.num_mel_frames_first_audio_chunk
        )

        hop_length = (
            self.processor
            .feature_extractor
            .hop_length
        )

        n_fft = (
            self.processor
            .feature_extractor
            .n_fft
        )

        # This is the same offset used in the official
        # streaming example.
        start_idx = (
            mel_frame_idx * hop_length
            - n_fft // 2
        )

        audio_length = len(audio)

        while start_idx < audio_length:

            end_idx = (
                start_idx
                + normal_chunk_samples
            )

            chunk = audio[
                start_idx:min(end_idx, audio_length)
            ]

            # -----------------------------------------------------
            # FINAL PARTIAL CHUNK
            # -----------------------------------------------------
            #
            # The model expects the normal streaming chunk shape.
            # Therefore pad the final short piece with silence.
            # -----------------------------------------------------

            chunk = self._pad_audio(
                chunk,
                normal_chunk_samples,
            )

            chunk_inputs = self.processor(
                chunk,
                sampling_rate=self.sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=self.language,
                return_tensors="pt",
            )

            chunk_inputs = chunk_inputs.to(
                self.model.device,
                dtype=self.model.dtype,
            )

            yield chunk_inputs.input_features

            mel_frame_idx += (
                self.processor
                .num_mel_frames_per_audio_chunk
            )

            start_idx = (
                mel_frame_idx * hop_length
                - n_fft // 2
            )

            # Once we've covered the original audio,
            # stop after the padded final chunk.
            if end_idx >= audio_length:
                break

    @torch.inference_mode()
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, float, float]:

        started = time.perf_counter()

        # ---------------------------------------------------------
        # NORMALIZE AUDIO ARRAY
        # ---------------------------------------------------------

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # RESAMPLE
        # ---------------------------------------------------------

        if sample_rate != self.sample_rate:
            audio = resample(
                audio,
                sample_rate,
                self.sample_rate,
            )

            sample_rate = self.sample_rate

        # ---------------------------------------------------------
        # BUILD STREAMING FEATURES
        # ---------------------------------------------------------

        feature_generator = (
            self._streaming_feature_generator(
                audio
            )
        )

        # We need the first chunk separately because NVIDIA's
        # streaming generate API takes the first chunk's metadata
        # and then receives subsequent input_features from a
        # generator.
        first_chunk_samples = (
            self.processor.num_samples_first_audio_chunk
        )

        first_chunk = audio[
            :first_chunk_samples
        ]

        first_chunk = self._pad_audio(
            first_chunk,
            first_chunk_samples,
        )

        first_chunk_inputs = self.processor(
            first_chunk,
            sampling_rate=self.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self.language,
            return_tensors="pt",
        )

        first_chunk_inputs = first_chunk_inputs.to(
            self.model.device,
            dtype=self.model.dtype,
        )

        # ---------------------------------------------------------
        # GENERATOR FOR SUBSEQUENT FEATURES
        # ---------------------------------------------------------

        mel_frame_idx = (
            self.processor
            .num_mel_frames_first_audio_chunk
        )

        hop_length = (
            self.processor
            .feature_extractor
            .hop_length
        )

        n_fft = (
            self.processor
            .feature_extractor
            .n_fft
        )

        normal_chunk_samples = (
            self.processor.num_samples_per_audio_chunk
        )

        start_idx = (
            mel_frame_idx * hop_length
            - n_fft // 2
        )

        audio_length = len(audio)

        def input_features_generator():

            # -----------------------------------------------------
            # FIRST CHUNK
            # -----------------------------------------------------

            yield (
                first_chunk_inputs
                .input_features[
                    :,
                    :self.processor.num_mel_frames_first_audio_chunk,
                    :,
                ]
            )

            # -----------------------------------------------------
            # REMAINING CHUNKS
            # -----------------------------------------------------

            nonlocal start_idx
            nonlocal mel_frame_idx

            while start_idx < audio_length:

                end_idx = (
                    start_idx
                    + normal_chunk_samples
                )

                chunk = audio[
                    start_idx:min(
                        end_idx,
                        audio_length,
                    )
                ]

                # Pad final partial chunk.
                chunk = self._pad_audio(
                    chunk,
                    normal_chunk_samples,
                )

                inputs = self.processor(
                    chunk,
                    sampling_rate=self.sample_rate,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    language=self.language,
                    return_tensors="pt",
                )

                inputs = inputs.to(
                    self.model.device,
                    dtype=self.model.dtype,
                )

                yield inputs.input_features

                mel_frame_idx += (
                    self.processor
                    .num_mel_frames_per_audio_chunk
                )

                start_idx = (
                    mel_frame_idx * hop_length
                    - n_fft // 2
                )

                if end_idx >= audio_length:
                    break

        # ---------------------------------------------------------
        # GENERATE KWARGS
        # ---------------------------------------------------------

        generate_kwargs = {
            **first_chunk_inputs,
            "input_features": input_features_generator(),
            "num_lookahead_tokens": self.lookahead_tokens,
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
        }

        # ---------------------------------------------------------
        # STREAMING GENERATION
        # ---------------------------------------------------------

        outputs = self.model.generate(
            **generate_kwargs
        )

        # ---------------------------------------------------------
        # DECODE
        # ---------------------------------------------------------

        decoded = self.processor.decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        if isinstance(decoded, list):
            text = decoded[0]
        else:
            text = decoded

        text = str(text).strip()

        # ---------------------------------------------------------
        # TIMING
        # ---------------------------------------------------------

        elapsed = (
            time.perf_counter()
            - started
        )

        # ---------------------------------------------------------
        # MEMORY
        # ---------------------------------------------------------

        peak_memory_mb = (
            get_process_memory_mb()
        )

        return (
            text,
            elapsed,
            peak_memory_mb,
        )


def run_evaluation(
    transcriber: NemotronTranscriber,
    model_id: str,
    examples,
    output_path: Path,
) -> None:

    results = {
        "model_id": model_id,
        "model_family": "NVIDIA Nemotron 3.5 ASR",
        "language": transcriber.language,
        "model_sample_rate": transcriber.sample_rate,
        "streaming": True,
        "lookahead_tokens": transcriber.lookahead_tokens,
        "streaming_latency_ms": getattr(
            transcriber.processor,
            "streaming_latency_ms",
            None,
        ),
        "max_new_tokens": transcriber.max_new_tokens,
        "normalization": (
            "casefold; remove ZWNJ/ZWJ; "
            "remove punctuation; "
            "collapse whitespace; "
            "remove number separators; "
            "preserve Devanagari vowel signs and scripts"
        ),
        "slices": {},
    }

    # -------------------------------------------------------------
    # EVALUATION SLICES
    # -------------------------------------------------------------

    for slice_name in (
        "clean",
        "telephony",
        "code_mixed",
    ):

        print()
        print("=" * 70)
        print(
            f"EVALUATION SLICE: {slice_name}"
        )
        print("=" * 70)

        records = []

        for index, example in enumerate(examples):

            # -----------------------------------------------------
            # CODE-MIXED FILTER
            # -----------------------------------------------------

            if slice_name == "code_mixed":

                reference_info = (
                    script_aware_metrics(
                        example.reference,
                        "",
                    )
                )

                if not reference_info[
                    "is_code_mixed"
                ]:
                    continue

            # -----------------------------------------------------
            # LOAD AUDIO
            # -----------------------------------------------------

            audio, sample_rate = load_audio(
                example.audio
            )

            # -----------------------------------------------------
            # TELEPHONY DEGRADATION
            # -----------------------------------------------------

            if slice_name == "telephony":

                audio, sample_rate = (
                    make_telephony_audio(
                        audio,
                        sample_rate,
                        index,
                    )
                )

            # -----------------------------------------------------
            # RESAMPLE
            # -----------------------------------------------------

            audio = resample(
                audio,
                sample_rate,
                transcriber.sample_rate,
            )

            duration = (
                len(audio)
                / transcriber.sample_rate
            )

            print()
            print(
                f"[{slice_name}] "
                f"{example.audio}"
            )
            print(
                f"  duration: "
                f"{duration:.2f} sec"
            )

            # -----------------------------------------------------
            # TRANSCRIBE
            # -----------------------------------------------------

            try:

                hypothesis, elapsed, peak_memory = (
                    transcriber.transcribe(
                        audio,
                        transcriber.sample_rate,
                    )
                )

            except Exception as exc:

                print(
                    f"  ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Continue with the next file instead of killing
                # the entire evaluation.
                continue

            # -----------------------------------------------------
            # RTF
            # -----------------------------------------------------

            rtf = (
                elapsed
                / max(duration, 1e-9)
            )

            # -----------------------------------------------------
            # METRICS
            # -----------------------------------------------------

            record = {
                "audio": str(example.audio),
                "reference": example.reference,
                "hypothesis": hypothesis,
                "duration_seconds": duration,
                "inference_seconds": elapsed,
                "rtf": rtf,
                "peak_process_memory_mb": peak_memory,
                "metrics": metric(
                    example.reference,
                    hypothesis,
                ),
                "script_aware": script_aware_metrics(
                    example.reference,
                    hypothesis,
                ),
            }

            records.append(record)

            print(
                f"  hypothesis: "
                f"{hypothesis}"
            )

            print(
                f"  RTF: "
                f"{rtf:.4f}"
            )

        # ---------------------------------------------------------
        # SLICE RESULT
        # ---------------------------------------------------------

        results["slices"][slice_name] = {
            "count": len(records),
            "aggregate": aggregate_metrics(
                records
            ),
            "records": records,
        }

        print()
        print(
            f"{slice_name} completed: "
            f"{len(records)} successful files"
        )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        f"Saved evaluation results to:"
    )
    print(output_path)
    print("=" * 70)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe/evaluate local audio "
            "with NVIDIA Nemotron 3.5 ASR."
        )
    )

    # -------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            Path(__file__).parent
            / "data"
        ),
        help=(
            "Directory searched recursively "
            "for audio files."
        ),
    )

    parser.add_argument(
        "--file",
        type=Path,
        help=(
            "Transcribe one audio file instead "
            "of scanning --data-dir."
        ),
    )

    # -------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model ID.",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            "Language locale. "
            "Use hi-IN for Hindi or auto "
            "for automatic detection."
        ),
    )

    # -------------------------------------------------------------
    # STREAMING
    # -------------------------------------------------------------

    parser.add_argument(
        "--lookahead-tokens",
        type=int,
        choices=[0, 3, 6, 13],
        default=DEFAULT_LOOKAHEAD_TOKENS,
        help=(
            "Streaming lookahead. "
            "0=80ms, 3=320ms, "
            "6=560ms, 13=1120ms."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum generated transcription "
            "tokens. Increase for exceptionally "
            "long recordings."
        ),
    )

    # -------------------------------------------------------------
    # NORMAL TRANSCRIPTION OUTPUT
    # -------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).parent
            / "transcriptions_nemotron.jsonl"
        ),
        help=(
            "JSONL file receiving one result "
            "per audio file."
        ),
    )

    # -------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Run clean, telephony, and "
            "code-mixed evaluation."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=(
            Path(__file__).parent
            / "data"
            / "gt.txt"
        ),
        help=(
            "Ground-truth text used with "
            "--evaluate."
        ),
    )

    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=(
            Path(__file__).parent
            / "evaluation_results_nemotron.json"
        ),
        help=(
            "JSON file containing "
            "transcriptions and metrics."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # FIND AUDIO
    # -------------------------------------------------------------

    audio_files = (
        [args.file]
        if args.file
        else find_audio_files(
            args.data_dir
        )
    )

    audio_files = [
        path
        for path in audio_files
        if path.is_file()
    ]

    if not audio_files:
        raise SystemExit(
            "No supported audio files were found."
        )

    # -------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------

    transcriber = NemotronTranscriber(
        model_id=args.model_id,
        language=args.language,
        lookahead_tokens=args.lookahead_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    # -------------------------------------------------------------
    # EVALUATION MODE
    # -------------------------------------------------------------

    if args.evaluate:

        examples = find_examples(
            args.data_dir,
            args.ground_truth,
        )

        if not examples:
            raise SystemExit(
                "No audio/reference pairs "
                "found for evaluation."
            )

        run_evaluation(
            transcriber=transcriber,
            model_id=args.model_id,
            examples=examples,
            output_path=args.evaluation_output,
        )

        return

    # -------------------------------------------------------------
    # NORMAL TRANSCRIPTION MODE
    # -------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for audio_file in audio_files:

            print()
            print(
                f"Processing: {audio_file}"
            )

            # Load audio.
            audio, sample_rate = load_audio(
                audio_file
            )

            # Resample.
            audio = resample(
                audio,
                sample_rate,
                transcriber.sample_rate,
            )

            # Transcribe using streaming path.
            text, elapsed, peak_memory = (
                transcriber.transcribe(
                    audio,
                    transcriber.sample_rate,
                )
            )

            record = {
                "file": str(audio_file),
                "text": text,
                "inference_seconds": elapsed,
                "duration_seconds": (
                    len(audio)
                    / transcriber.sample_rate
                ),
                "rtf": (
                    elapsed
                    / max(
                        len(audio)
                        / transcriber.sample_rate,
                        1e-9,
                    )
                ),
                "peak_process_memory_mb": peak_memory,
            }

            print(
                f"  text: {text}"
            )

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()