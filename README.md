# Day 1 — Indian-Language ASR Benchmark

This project benchmarks open-source ASR models for Indian-language speech, with a focus on **Hindi, Hindi-English code-mixed speech, telephony audio, and long-form audio**.

The Day 1 work covers:

* Data source survey
* Open-source model survey and selection
* Evaluation dataset selection
* Long-audio inference investigation
* Benchmarking across clean, telephony, and code-mixed speech
* WER, CER, and RTF evaluation

---

# Part 1a — Data Sources

The following datasets were reviewed for Indian-language ASR training and evaluation.

| Dataset                                | Languages                     |               Approx. Hours | Speech Type                        | Code-Mixed                       | License / Access              | Purpose               |
| -------------------------------------- | ----------------------------- | --------------------------: | ---------------------------------- | -------------------------------- | ----------------------------- | --------------------- |
| **IndicVoices**                        | 22 Indian languages           |                     11,200+ | Diverse, prompted and natural      | No explicit code-switching       | CC BY 4.0 / gated             | Training              |
| **Kathbath**                           | 12 Indian languages           |                       1,684 | Crowdsourced read speech           | No explicit code-switching       | CC BY 4.0 / release dependent | Training / Evaluation |
| **Kathbath-Noisy / Hard**              | 12 Indian languages           |           Release dependent | Noisy read speech                  | No explicit code-switching       | Release dependent             | Evaluation            |
| **Shrutilipi**                         | 12 Indian languages           |                      6,400+ | Broadcast news                     | No                               | CC BY 4.0                     | Training              |
| **Lahaja**                             | Hindi                         |                        12.5 | Read + extempore                   | No explicit Hinglish split       | Dataset specific              | Evaluation            |
| **Svarah**                             | Indian-accented English       |                         9.6 | Indian-accented English            | No                               | CC BY 4.0                     | Evaluation            |
| **Vistaar**                            | 12 Indian languages           | 10,700+ training collection | Mixed domains, including telephony | Yes through constituent datasets | Dataset dependent             | Training / Evaluation |
| **Google FLEURS**                      | 102 languages                 |                ~12/language | Read speech                        | No                               | CC BY 4.0                     | Evaluation            |
| **Mozilla Common Voice**               | Multiple languages            |           Release dependent | Crowdsourced read speech           | Not specifically code-mixed      | CC0 current releases          | Training / Evaluation |
| **MUCS Hindi-English**                 | Hindi + English               |                       95.04 | Technical lectures                 | **Yes**                          | CC BY-SA 4.0                  | Training / Evaluation |
| **Gram Vaani**                         | Hindi                         |             ~1,100 released | Spontaneous telephone speech       | No explicit Hinglish split       | Academic-use access           | Evaluation            |
| **Navana Hindi / Indic ASR Benchmark** | Hindi + other Indic languages |      ~15.5 h Hindi held-out | Held-out evaluation                | No explicit code-mix split       | Dataset / repo dependent      | Evaluation            |

### Dataset observations

* Large datasets such as **IndicVoices, Kathbath, Shrutilipi, and Vistaar** are primarily useful for training and broad evaluation.
* **MUCS Hindi-English** provides an explicit Hindi-English code-switching condition.
* **Gram Vaani** provides natural, telephone-quality Hindi speech.
* **Lahaja** provides additional Hindi accent and regional variation.
* **Vistaar** combines multiple datasets and domains, making it useful for broader evaluation.

### Recommended evaluation sets

For evaluating Hindi ASR under realistic call-like and mixed-language conditions:

1. **MUCS Hindi-English** — explicit Hindi-English code-switching.
2. **Gram Vaani** — spontaneous telephone-quality Hindi with real-world acoustic variation.
3. **Lahaja** — Hindi accent and extempore speech diversity.

These datasets provide complementary evaluation conditions rather than representing one identical test set.

---

# Part 1b — Models

Five open-source ASR models were selected for the benchmark.

| Model                        | Architecture / Family   | Language Coverage              | Long-Audio Handling                |
| ---------------------------- | ----------------------- | ------------------------------ | ---------------------------------- |
| **Whisper large-v3-turbo**   | Whisper                 | Multilingual                   | 30-second chunks / stride          |
| **Fine-tuned Whisper Hindi** | Whisper fine-tuned      | Hindi                          | 30-second chunks                   |
| **IndicConformer 600M**      | Conformer / CTC         | 22 Indian languages            | 45-second chunks                   |
| **SraVaani 1.0**             | FastConformer / TDT-CTC | 65 Indian languages / dialects | 30-second chunks with overlap      |
| **NVIDIA Nemotron 3.5**      | FastConformer / RNNT    | Multilingual including Hindi   | Processor-defined streaming chunks |

Model cards were reviewed for:

* Languages used during training
* Tokenizer and output-script behavior
* Model architecture
* Supported input duration
* Long-audio/chunking behavior

All benchmark inference is performed locally. Customer audio should not be sent to external commercial APIs.

## Inference Problems and Solutions

### 1. Long-audio inference

Whisper and some Conformer-based models do not directly process arbitrarily long audio.

**Solution:** Audio is split into model-compatible chunks:

* Whisper: 30-second chunks with stride
* SraVaani: 30-second chunks with overlap
* IndicConformer: 45-second chunks
* Nemotron: processor-defined streaming chunks

### 2. ONNX long-sequence errors

Long Conformer audio caused attention / positional-encoding sequence-length mismatches.

**Solution:** Split long recordings into 45-second or smaller chunks while retaining the expected model input signature.

### 3. Incorrect model input

Passing additional `length` tensors to some model interfaces resulted in invalid or `None` inputs.

**Solution:** Use the model's expected call signature, for example:

```python
model(audio_tensor, language, "ctc")
```

### 4. Slow inference

Repeated model initialization significantly increased runtime.

**Solution:**

* Load each model once
* Reuse the model across evaluation slices
* Batch inference where supported
* Use optimized attention/runtime options where supported

---

# Part 1c — How to Evaluate

## Indian-Language ASR Benchmark

This project benchmarks five open-source ASR models on the audio in `data/`. It reports **word error rate (WER), character error rate (CER), and real-time factor (RTF)** for three slices:

* `clean`: original audio
* `telephony`: controlled telephone-bandwidth/noise simulation
* `code_mixed`: references containing both Devanagari and Latin-script words

The benchmark is intended for Hindi and Hindi-English speech, including long-form and telephony-style audio.

## Project layout

| Path                                     | Contents                                                                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `data/`                                  | Evaluation audio and `gt.txt` ground-truth references.                                                                             |
| `modelcodesday1/evaluate_asr.py`         | Shared audio loading, resampling, normalization, WER/CER, telephony augmentation, code-mixed detection, and aggregation utilities. |
| `modelcodesday1/whisperltv3.py`          | `openai/whisper-large-v3-turbo` runner; uses 30-second chunks with a 5-second stride.                                              |
| `modelcodesday1/whisper_fintuned.py`     | Fine-tuned Hindi Whisper runner for `vasista22/whisper-hindi-medium`.                                                              |
| `modelcodesday1/indicconfor.py`          | `ai4bharat/indic-conformer-600m-multilingual` CTC runner; splits long audio into 45-second chunks.                                 |
| `modelcodesday1/sraavani.py`             | `ARTPARK-IISc/SraVaani-1.0` runner; uses 30-second chunks, 1-second overlap, and batching.                                         |
| `modelcodesday1/transcriber_nemotron.py` | NVIDIA Nemotron 3.5 streaming runner with processor-defined streaming chunks.                                                      |
| `modelcodesday1/numberRescoring.py`      | Standalone Hindi number normalization/inverse-normalization example using NeMo Text Processing.                                    |
| `modelcodesday1/Welcome_To_Colab.ipynb`  | Notebook used for interactive setup and experimentation.                                                                           |
| `modelbenchmark_day1/`                   | Saved JSON evaluation outputs and the benchmark results reported below.                                                            |

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install numpy soundfile psutil torch huggingface_hub nemo_text_processing

pip install --upgrade transformers
```

The scripts download models from Hugging Face on first use. Authenticate when a gated model requests access:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

CUDA is used automatically when available; otherwise the scripts run on CPU.

The audio directory and ground truth default to:

```text
data/
data\gt.txt
```

The Nemotron runner uses `AutoModelForRNNT`; use an up-to-date Transformers release. If the import fails, rerun:

```powershell
pip install --upgrade transformers
```

before running the complete benchmark.

---

# Run the Complete Benchmark

The deliverable is one results table:

**model × slice → WER / CER / RTF**

All five evaluations can be reproduced with one command:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_benchmark.ps1
```

When PowerShell is already being used:

```powershell
.\run_benchmark.ps1
```

The wrapper runs every model with:

```text
--evaluate
--data-dir data
--ground-truth data\gt.txt
```

and writes fresh JSON files into:

```text
modelbenchmark_day1/
```

---

# Run Models Individually

Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe modelcodesday1\whisperltv3.py --data-dir data --evaluate --ground-truth data\gt.txt --evaluation-output modelbenchmark_day1\whisperevaluation_results.json

.\.venv\Scripts\python.exe modelcodesday1\whisper_fintuned.py --data-dir data --evaluate --ground-truth data\gt.txt --evaluation-output modelbenchmark_day1\ftwhisper_evaluation.json

.\.venv\Scripts\python.exe modelcodesday1\indicconfor.py --data-dir data --evaluate --ground-truth data\gt.txt --batch-size 1 --evaluation-output modelbenchmark_day1\indicconfor_evaluation.json

.\.venv\Scripts\python.exe modelcodesday1\sraavani.py --data-dir data --evaluate --ground-truth data\gt.txt --batch-size 4 --chunk-seconds 30 --overlap-seconds 1 --evaluation-output modelbenchmark_day1\sravaani_evaluation.json

.\.venv\Scripts\python.exe modelcodesday1\transcriber_nemotron.py --data-dir data --evaluate --ground-truth data\gt.txt --lookahead-tokens 6 --evaluation-output modelbenchmark_day1\evaluation_results_nemotron.json
```

For transcription without scoring, omit `--evaluate` and provide:

```text
--output path.jsonl
```

Model-specific options can be viewed with:

```powershell
python modelcodesday1\<script>.py --help
```

---

# Evaluation Slices

### Clean

Original evaluation audio without additional transformations.

### Telephony

The same audio is transformed to simulate telephone conditions:

* Downsample to 8 kHz
* Apply narrowband / μ-law characteristics
* Add controlled noise
* Upsample back to the model's expected sampling rate

### Code-Mixed

Evaluation uses references containing both:

* **Devanagari-script words**
* **Latin-script English words**

This slice is intended to measure Hindi-English code-mixed recognition and script behavior.

---

# Metrics

The benchmark reports:

* **WER** — Word Error Rate
* **CER** — Character Error Rate
* **RTF** — Real-Time Factor
* **Peak CPU memory**

For code-mixed speech, additional analysis includes:

* Mixed error rate
* CER on Devanagari
* WER on Latin-script words
* English-word accuracy
* Wrong-script rate

Normalization is applied consistently before scoring. Indic vowel signs are preserved, while punctuation and number-formatting differences are handled according to the documented normalization policy.

---

# Results

Values below are aggregate metrics read from the JSON files in `modelbenchmark_day1/`.

WER and CER are error rates. RTF is mean inference time divided by audio duration. Lower values indicate lower error or lower inference time.

`n` is the number of evaluated recordings.

| Model                    | Slice      |  n |    WER |    CER |    RTF |
| ------------------------ | ---------- | -: | -----: | -----: | -----: |
| NVIDIA Nemotron 3.5      | clean      |  3 | 0.4156 | 0.4212 | 0.1605 |
| NVIDIA Nemotron 3.5      | telephony  |  3 | 0.4166 | 0.4194 | 0.1046 |
| NVIDIA Nemotron 3.5      | code-mixed |  1 | 0.4226 | 0.4366 | 0.1037 |
| Fine-tuned Whisper Hindi | clean      |  3 | 0.4851 | 0.5016 | 1.9540 |
| Fine-tuned Whisper Hindi | telephony  |  3 | 0.5477 | 0.5519 | 1.7156 |
| Fine-tuned Whisper Hindi | code-mixed |  1 | 0.5005 | 0.5234 | 3.2042 |
| IndicConformer 600M      | clean      |  3 | 0.4446 | 0.4490 | 0.2525 |
| IndicConformer 600M      | telephony  |  3 | 0.4397 | 0.4567 | 0.2231 |
| IndicConformer 600M      | code-mixed |  1 | 0.4580 | 0.4678 | 0.2565 |
| SraVaani 1.0             | clean      |  3 | 0.3616 | 0.3902 | 0.2287 |
| SraVaani 1.0             | telephony  |  3 | 0.3433 | 0.3858 | 0.1603 |
| SraVaani 1.0             | code-mixed |  1 | 0.3741 | 0.4086 | 0.0121 |
| Whisper large-v3-turbo   | clean      |  3 | 0.4301 | 0.3721 | 2.3070 |
| Whisper large-v3-turbo   | telephony  |  3 | 0.4388 | 0.3964 | 3.2807 |
| Whisper large-v3-turbo   | code-mixed |  1 | 0.4247 | 0.3767 | 2.0353 |

## Result Files

| Result File                        | Script                    |
| ---------------------------------- | ------------------------- |
| `evaluation_results_nemotron.json` | `transcriber_nemotron.py` |
| `ftwhisper_evaluation.json`        | `whisper_fintuned.py`     |
| `indicconfor_evaluation.json`      | `indicconfor.py`          |
| `sravaani_evaluation.json`         | `sraavani.py`             |
| `whisperevaluation_results.json`   | `whisperltv3.py`          |

Each JSON file contains:

* Model configuration
* Normalization policy
* Per-slice aggregates
* Per-record hypotheses
* Audio duration
* RTF
* Memory measurements

---

# Reproducibility

The benchmark is designed to be re-runnable using the same:

* Dataset and ground-truth references
* Audio preprocessing
* Normalization policy
* Model configuration
* Inference settings
* Evaluation scripts

The complete benchmark can be reproduced through:

```powershell
.\run_benchmark.ps1
```

Hardware, Python/PyTorch versions, model revisions, and runtime device can affect the reported results.

---

# Notes

* Model downloads can require substantial disk space and RAM/VRAM.
* Results depend on model revisions, hardware, PyTorch version, and runtime device.
* CUDA is used automatically when available.
* Inference is performed locally.
* Customer audio should not be sent to external commercial APIs.


# Day 2 (Error Analysis)


## Overall Comparison

| Model | Avg WER | Avg CER | Avg RTF | Key Observation |
|---|---:|---:|---:|---|
| **ARTPARK-IISc/SraVaani-1.0** | **35.25%** | 38.80% | 0.195 | Best overall Hindi WER |
| **NVIDIA Nemotron 3.5 ASR 0.6B** | 41.61% | 42.03% | **0.133** | Fastest model |
| **OpenAI Whisper large-v3-turbo** | 43.44% | **38.43%** | 2.794 | Best English/code-switch preservation |
| **AI4Bharat IndicConformer-600M** | 44.21% | 45.29% | 0.238 | Good short Hindi recognition |
| **vasista22/whisper-hindi-medium** | 51.64% | 52.68% | 1.835 | Highest overall error |

*Avg = mean of clean and telephony aggregate results over the 3 samples.*

## Main Error Buckets

| Bucket | Main Finding |
|---|---|
| **Code-switch / script** | Largest problem. English words are often converted to Devanagari or lost at language boundaries. |
| **Long-audio / segmentation** | Long 413s tutorial causes deletions, compression and tail truncation. |
| **Named entities** | Names and technical terms are sometimes phonetically confused. |
| **Numbers / dates** | Digit vs spoken-number mismatch, e.g. `उन्नीस सौ तिरानवे` → `1993`. |
| **Accent / dialect** | Mostly visible in the Lahaja sample. |
| **Noise / telephony** | Model-dependent; SraVaani and IndicConformer remain relatively stable, while Vasista degrades. |
| **Normalization artefacts** | Variants such as `फ़िल्म` / `फिल्म` / `फ़िल्म` can be counted as errors. |
| **Hallucination** | No confirmed hallucination from the supplied results. |
| **Short utterances** | Not sufficiently tested in the current benchmark. |

## Representative Errors

| Type | Reference | Hypothesis |
|---|---|---|
| Code-switch | `document` | `डॉक्यूमेंट` |
| Code-switch | `gnu/linux` | `जेएनयू लिनक्स` |
| Number | `उन्नीस सौ तिरानवे` | `1993` |
| Named entity | `किम` | `कीम` / `टीम` |
| Long audio | Full closing section | Tail content truncated |

## Model Summary

**SraVaani**
- Best overall WER.
- Strongest on short Hindi samples.
- Fast.
- Main weakness: poor English/code-switch preservation.

**Whisper large-v3-turbo**
- Best CER.
- Strongest English/code-mixed preservation.
- Much slower than the other models.
- Weaker on the Lahaja sample.

**Nemotron 3.5 ASR**
- Fastest.
- Good quality/latency trade-off.
- Code-switch accuracy and long-audio truncation are concerns.

**IndicConformer**
- Good short Hindi recognition.
- Fast.
- Long mixed-language speech remains difficult.

**Vasista Hindi Medium**
- Good first short sample.
- Highest overall WER/CER.
- Larger telephony degradation.


## Error Analysis & Failure Hypotheses

| Error Bucket | Likely Cause |
|---|---|
| **Code-switch / Script** | Mainly **training-data and decoding bias** toward Hindi script; English technical words are often transliterated into Devanagari. |
| **Numbers / Dates** | **Decoding + normalization** differences between spoken numbers and digits (`उन्नीस सौ तिरानवे` ↔ `1993`). |
| **Named Entities** | **Limited training coverage + acoustic ambiguity** for names, places, brands, and technical terms. |
| **Noise / Telephony** | **Audio front-end + training robustness**; 8 kHz/noisy speech can increase substitutions and deletions. |
| **Accent / Dialect** | **Training-data coverage** of regional pronunciation and dialect variation. |
| **Long Audio / Segmentation** | **Chunking, context limits, and decoding** can cause deletions or truncation in long recordings. |
| **Normalization Artefacts** | **Scoring/normalization issue**, especially for Hindi orthographic variants such as `फ़िल्म` / `फिल्म` / `फ़िल्म`. |
| **Hallucination / Short Utterances** | Not confirmed or sufficiently represented in this benchmark. |

### Model-specific observations

| Model | Main Observed Weakness |
|---|---|
| **SraVaani** | Strong Hindi accuracy, but weak **English/code-switch preservation**. |
| **Whisper large-v3-turbo** | Better **code-switch handling**, but slower and less consistent on some Hindi/accent samples. |
| **Nemotron 3.5 ASR** | Very fast, but **code-switch and long-audio handling** need attention. |
| **IndicConformer-600M** | Good short Hindi performance, but more **long-audio deletions**. |
| **Vasista Hindi Medium** | More **telephony degradation** and weaker long mixed-language performance. |

## Recommended 2-Model Shortlist

### 1. ARTPARK-IISc/SraVaani-1.0
Best choice for **overall Hindi ASR accuracy and speed** in this benchmark.

### 2. OpenAI Whisper large-v3-turbo
Useful as a **complementary model for Hindi-English code-switched speech**, especially where preserving English terminology matters.

### Why These Two?

SraVaani has the lowest overall WER, while Whisper preserves English/code-mixed terminology better. The two therefore cover the main strengths and weaknesses observed in the benchmark.

