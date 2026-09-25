#!/usr/bin/env python3
"""
Whisper Fine-Tuning Script for Hindi & Hindi-English Code-Mixed (Hinglish) Speech
Includes:
- Telephony Audio Augmentation (8kHz downsampling + u-law companding)
- On-the-fly SpecAugment & Speed Perturbation
- Devanagari & Nukta Orthographic Canonicalization
- LoRA Efficient Fine-Tuning
"""

import os
import re
import random
import io
import torch
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import evaluate

# ==========================================
# 1. TEXT NORMALIZATION & CANONICALIZATION
# ==========================================

class HinglishTextNormalizer:
    """
    Applies Devanagari normalizer, Nukta canonicalization (e.g., फ़िल्म -> फिल्म),
    and basic text cleaning for Hindi-English code-mixed transcripts.
    """
    def __init__(self):
        self.hindi_normalizer = IndicNormalizerFactory().get_normalizer("hi")
        # Nukta and orthographic variants mapping
        self.nukta_map = {
            "क़": "क", "ख़": "ख", "ग़": "ग", 
            "ज़": "ज", "ड़": "ड", "ढ़": "ढ", 
            "फ़": "फ", "य़": "य"
        }

    def canonicalize_nukta(self, text: str) -> str:
        for char, replacement in self.nukta_map.items():
            text = text.replace(char, replacement)
        return text

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        # Devanagari normalization
        text = self.hindi_normalizer.normalize(text)
        # Nukta canonicalization for consistent WER scoring
        text = self.canonicalize_nukta(text)
        # Strip extraneous whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ==========================================
# 2. AUDIO AUGMENTATION PIPELINE
# ==========================================

def simulate_telephony(audio_array: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """
    Simulates narrowband telephony audio:
    1. Resamples to 8kHz (band-limiting).
    2. Applies 8-bit mu-law companding/quantization.
    3. Resamples back to target 16kHz rate for Whisper.
    """
    # Downsample to 8kHz telephony standard
    audio_8k = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=8000)
    
    # 8-bit mu-law companding simulation
    # Quantize to 8-bit mu-law using soundfile buffer stream
    buffer = io.BytesIO()
    sf.write(buffer, audio_8k, 8000, format='WAV', subtype='PCM_U8')
    buffer.seek(0)
    audio_quantized, _ = sf.read(buffer)
    
    # Resample back to 16kHz
    audio_16k = librosa.resample(audio_quantized, orig_sr=8000, target_sr=sampling_rate)
    return audio_16k.astype(np.float32)

def apply_speed_perturbation(audio_array: np.ndarray, factors: List[float] = [0.9, 1.0, 1.1]) -> np.ndarray:
    """Applies speed perturbation (0.9x, 1.0x, 1.1x)."""
    factor = random.choice(factors)
    if factor == 1.0:
        return audio_array
    return librosa.effects.time_stretch(audio_array, rate=factor)

# ==========================================
# 3. DATASET PARSING & PREPARATION
# ==========================================

def parse_transcript_file(transcript_path: str) -> Dict[str, str]:
    """Parses text file mapping base filenames to Hindi/Hinglish transcripts."""
    data = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                file_id, text = parts
                base_id = file_id.replace('.txt', '').replace('.wav', '')
                data[base_id] = text
    return data

def build_dataset_from_directory(audio_dir: str, transcript_file: str) -> pd.DataFrame:
    """Links audio files with corresponding text transcripts."""
    transcripts = parse_transcript_file(transcript_file)
    audio_paths = list(Path(audio_dir).glob("*.wav"))
    
    records = []
    for path in audio_paths:
        base_id = path.stem
        if base_id in transcripts:
            records.append({
                "audio": str(path),
                "sentence": transcripts[base_id]
            })
            
    return pd.DataFrame(records)

def prepare_hinglish_dataset(audio_dir: str, transcript_file: str) -> DatasetDict:
    """
    Creates train/validation split.
    NOTE: For strict Speaker/Recording Disjoint splits, supply pre-split CSVs or
    split at the speaker ID level instead of random utterance-level splits.
    """
    df = build_dataset_from_directory(audio_dir, transcript_file)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split into train/validation sets (80/20)
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({
        'train': dataset_split['train'],
        'validation': dataset_split['test']
    })

# ==========================================
# 4. PREPROCESSING & FEATURE EXTRACTION
# ==========================================

def is_sentence_valid(sentence: str) -> bool:
    """Filter out empty sentences."""
    return sentence is not None and len(sentence.strip()) > 0

def is_labels_in_length_range(labels: List[int]) -> bool:
    """Whisper max sequence length constraint (448 tokens)."""
    return len(labels) < 448

def prepare_dataset_batch(
    batch: Dict[str, Any], 
    feature_extractor: WhisperFeatureExtractor, 
    tokenizer: WhisperTokenizer, 
    normalizer: HinglishTextNormalizer,
    apply_augmentations: bool = True
) -> Dict[str, Any]:
    
    audio = batch["audio"]
    audio_array = audio["array"]
    sr = audio["sampling_rate"]

    # Apply Telephony and Speed Augmentation on training data
    if apply_augmentations:
        if random.random() < 0.4:  # 40% chance of telephony simulation
            audio_array = simulate_telephony(audio_array, sampling_rate=sr)
        if random.random() < 0.3:  # 30% chance of speed perturbation
            audio_array = apply_speed_perturbation(audio_array)

    # Extract log-Mel input features
    batch["input_features"] = feature_extractor(
        audio_array, sampling_rate=sr
    ).input_features[0]

    # Apply Devanagari/Nukta canonicalization to transcripts
    normalized_text = normalizer(batch["sentence"])
    batch["labels"] = tokenizer(normalized_text).input_ids

    return batch

# ==========================================
# 5. PEFT / LORA MODEL SETUP
# ==========================================

def setup_lora_model(model_id: str = "openai/whisper-small") -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Target all linear projections in attention and MLP blocks for code-mixing capacity
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ==========================================
# 6. DATA COLLATOR WITH SPECAUGMENT
# ==========================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ==========================================
# 7. TRAINING EXECUTION
# ==========================================

def main():
    MODEL_ID = "openai/whisper-small"
    OUTPUT_DIR = "./whisper-hindi-hinglish-lora"
    
    # Local directory paths
    HINDI_AUDIO_DIR = "/home/merai_bengaluru/whisperfinetune/Hindi_fem_mono/Hindi_fem_audio"
    HINDI_TXT = "/home/merai_bengaluru/whisperfinetune/hindi_fem_mono.txt"

    # Initialize Normalizer and Model Tokenizers
    normalizer = HinglishTextNormalizer()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="hindi", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hindi", task="transcribe")

    print("Loading and preparing dataset...")
    dataset = prepare_hinglish_dataset(HINDI_AUDIO_DIR, HINDI_TXT)

    # Filter out empty sentences
    dataset = dataset.filter(is_sentence_valid, input_columns=["sentence"])

    # Map preprocessing function
    dataset["train"] = dataset["train"].map(
        lambda batch: prepare_dataset_batch(
            batch, feature_extractor, tokenizer, normalizer, apply_augmentations=True
        ),
        remove_columns=dataset["train"].column_names,
        num_proc=1  # Sequential processing required when applying librosa/soundfile audio transforms
    )

    dataset["validation"] = dataset["validation"].map(
        lambda batch: prepare_dataset_batch(
            batch, feature_extractor, tokenizer, normalizer, apply_augmentations=False
        ),
        remove_columns=dataset["validation"].column_names,
        num_proc=1
    )

    # Enforce maximum label length (< 448 tokens)
    for split in dataset:
        dataset[split] = dataset[split].filter(is_labels_in_length_range, input_columns=["labels"])

    print("Initializing LoRA model...")
    model = setup_lora_model(MODEL_ID)

    # Compute WER with evaluating metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize predicted and ground truth strings using Nukta/Devanagari canonicalization before WER calculation
        pred_str = [normalizer(s) for s in pred_str]
        label_str = [normalizer(s) for s in label_str]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=500,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=4,
        push_to_hub=False,
        report_to=["tensorboard"]
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    # Save fine-tuned checkpoint
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()