# Day 5 — Hinglish ASR experiments and voicebot pipeline

## Purpose and scope

Day 5 explores ways to improve Hindi/Hinglish call transcription and turn it into useful voicebot data: domain vocabulary, dialogue context, number handling, end-of-turn detection, audio-event tags, confidence flags, and LLM-based cleanup and extraction.

The experiments use eight short audio clips: `PC.m4a`, `kb1.m4a`, `LICS.m4a`, `PE.m4a`, `nbdid.m4a`, `EOT.m4a`, `DMF.m4a`, and `OOVB.m4a`. The notebooks run Faster-Whisper `large-v3`, generally with beam size 5; the Colab outputs show CUDA/float16. Most outputs are exploratory comparisons rather than a controlled accuracy benchmark.

**Important evaluation limitation:** the saved CSVs do not contain aligned human reference transcripts or labeled intents/entities. No WER/CER, slot accuracy, intent accuracy, or statistically controlled significance test was computed. The keyword experiment counts exact target-phrase substring matches only; it is a useful diagnostic, not a complete accuracy measure. Treat the results below as evidence for what to test next, not proof of production performance.

## At-a-glance decisions

| Idea | What the experiment showed | Build decision |
|---|---|---|
| Faster-Whisper baseline | All eight files transcribed faster than their audio duration on the recorded GPU run; several domain words were phonetic approximations. | Keep as the baseline; benchmark accuracy and repeatable latency on a larger labeled set. |
| Whisper hotwords | Exact target phrase hits increased from 3 to 5 across the eight clips; hotwords recovered the date phrase in `DMF` and “cashback claim” in `OOVB`. | Worth a small, configurable vocabulary-biasing feature and a proper precision/recall test. |
| Generic initial prompt | Three exact target hits, the same count as baseline; it did not recover the two extra phrases found with hotwords. | Do not build generic prompting as an accuracy feature without better prompt design and validation. |
| NeMo language filtering / ITN / TN | No segments were filtered; ITN outputs showed no demonstrated improvement; ordinary TN sometimes expanded digits or spelled English words letter-by-letter. | Do not ship this implementation. Revisit narrow, language-aware formatting only with reference-based tests. |
| Confidence + YAMNet + endpoint signal | All eight were marked confident and as finished speaking; YAMNet mostly returned “Speech” plus noise-like labels. | Confidence and endpoint flags merit a labeled validation pass; YAMNet is optional unless a product action needs its tags. |
| Silero VAD endpoint | All eight were also marked finished, with 1–2 speech chunks per clip. No decision changed versus the Whisper-tail heuristic. | Worth a streaming/latency prototype if endpointing matters; not yet shown to improve these clips. |
| Gemini transcript correction | The examples often corrected phonetic spellings, amounts, dates, and domain terms, but some corrections were guesses and there was no reference-scored evaluation. | Strong candidate for a guarded offline experiment; production needs confidence, validation, and human review for critical slots. |
| Combined ASR + LLM extraction | Produced readable transcripts, intent labels, and slot JSON for all eight; all `Confirm Slot?` values were `NO`. Accuracy of those fields was not measured. | Prototype the structured output behind schema checks and confirmation rules; do not treat extracted values as verified. |

---

# 1. [Day5.ipynb](Day5.ipynb) — ASR and audio-processing experiments


## 1.1 Standard Faster-Whisper baseline

**What was tried.** Each supported audio file was transcribed with `large-v3` on CUDA/float16 and beam size 5. The code recorded detected language and probability, duration, inference wall time, real-time factor (RTF), and transcript. It wrote [transcription_results.csv](results/transcription_results.csv).

**How it was measured.** Duration was read with Librosa. RTF is inference time divided by audio duration; below 1 means the recorded inference was faster than the clip's playback duration. The rounded durations in the CSV total about 72.6 seconds. Recorded inference ranged from 1.00 to 2.71 seconds per clip, with RTF from 0.114 to 0.392 (mean of the eight recorded RTF values: about 0.192).

**Result / what improved.** This established a fast GPU baseline. Whisper detected Hindi (`hi`) for all eight clips, but confidence varied substantially (0.36–0.95). Important terms were often phonetically transcribed rather than canonically spelled—for example, “Khatabook”/“bahi-khata,” “SraVaani Pro,” and “SurgeX” were not reliably represented as their intended terms. The run also emitted Librosa/PySoundFile warnings and fell back to audioread for the `.m4a` files.

**Worth building properly?** Yes as the comparison baseline, not as a finished ASR quality result. Preserve a fixed test set, add verified references, report WER/CER and critical-slot accuracy, and repeat latency measurements after warm-up on documented hardware.

## 1.2 Hotword biasing versus initial prompt

**What was tried.** The same clips were decoded three ways: no bias, Faster-Whisper `hotwords`, and an `initial_prompt` listing domain terms. The decode explicitly set `language="hi"` in all three configurations. The keyword list included product names, English/Hinglish support phrases, a date, amounts, and IDs. Results and full hypotheses were saved to [keyword_boosting_results.csv](results/keyword_boosting_results.csv).

**How it was measured.** For each transcript, the notebook counted which of the supplied target phrases appeared as exact, case-insensitive substrings. Across the eight files, baseline produced 3 target-phrase hits, hotwords produced 5, and initial prompt produced 3. These are phrase matches, not WER or complete keyword recall; the chosen phrase inventory and substring matching limit what the counts mean.

**Result / what improved.** Hotwords added two observed matches: `25th September 2026` for `DMF` and `cashback claim` for `OOVB`. The other matched terms (`refund process`, `984450`, `492810`) were already present in the baseline. The generic initial prompt did not add hits beyond baseline. Hotwords also changed some transcript spellings and punctuation, but the table does not establish that those changes were more accurate overall.

**Worth building properly?** Yes, as a narrowly scoped and configurable domain vocabulary option. Evaluate phrase-level precision and recall against references, include out-of-domain and negative examples to detect forced hallucinations, and test vocabulary size and per-customer customization. Do not assume the hit-count gain generalizes beyond these eight examples.

## 1.3 Segment language filtering with NeMo ITN

**What was tried.** A later experiment combined hotwords and an initial prompt with an allowlist of Hindi and English (`hi`, `en`). It attempted to discard non-target-language segments and apply NeMo inverse text normalization (ITN) per segment. Its saved table appears in the notebook output; this version did not write one of the checked-in result CSVs.

**How it was measured.** The output reported primary language, language probability, elapsed time, number of filtered segments, and transcript. All eight primary languages were `hi`; all eight had **0 filtered segments**. The code checks `seg.language` when present, otherwise falls back to the clip-wide `info.language`, so the fallback is not segment-level language identification. No reference comparison or before/after numeric accuracy score was produced.

**Result / what improved.** The table showed no evidence that filtering removed hallucinations: it removed no segment. The resulting transcripts were broadly similar to the baseline/boosted outputs. A zero filter count is not proof that no non-Hindi hallucinations existed; it only shows this run did not flag any.

**Worth building properly?** Not in this form. If mixed-language segment routing is needed, use a verified segment-level language signal and evaluate false removals and missed language switches on labeled code-switched audio before filtering user speech.

## 1.4 Unconstrained NeMo ITN and NeMo text normalization

**What was tried.** The next cells applied NeMo ITN to every segment without filtering, then tried NeMo *text normalization* (TN) instead. Both retained the raw transcript as a fallback if normalization raised an error. The notebook markdown itself notes that ITN showed no effect and that TN turns numbers into words.

**How it was measured.** The displayed before/after transcripts and elapsed times were inspected; no reference-based score was recorded. The ITN output was largely unchanged from the Whisper transcript. TN visibly changed representation—for example, digit strings were verbalized in Hindi and English phrases could be spelled as individual letters (the `nbdid` output rendered “order id” letter-by-letter and read the ID aloud).

**Result / what improved.** The observed TN behavior is the opposite of the likely call-center need when IDs and amounts should remain stable, machine-readable digits. ITN did not demonstrate an improvement on these samples. The NeMo grammar initialization also adds setup cost, and the output was not used to establish a numeric quality gain.

**Worth building properly?** No for the current broad, every-segment normalization. If needed, build targeted post-processing for specific entity types (IDs, dates, currency) and preserve raw text. Score exact entity values and spoken-vs-written formatting separately before adoption.

## 1.5 Whisper confidence, YAMNet audio events, and Whisper-tail endpoint heuristic

**What was tried.** The notebook extracted Faster-Whisper segment average log probabilities and no-speech probabilities, set `STT Unsure` when mean log probability was below `-0.8` or maximum no-speech probability exceeded `0.6`, estimated trailing silence from the final Whisper segment timestamp, and used YAMNet to return the top three audio-event labels. Results were saved in [stt_confidence_and_events.csv](results/stt_confidence_and_events.csv).

**How it was measured.** The eight saved rows show mean log probability from `-0.108` to `-0.302`; maximum no-speech probability was at most `0.171`. Consequently, **all eight** received `STT Unsure = NO`. Whisper-timestamp trailing silence ranged from 1.30 to 2.80 seconds, so all eight were labeled `Turn Finished? = YES` at the 0.8-second threshold. YAMNet's top label was Speech for all eight; the other displayed labels were mostly noise-like/background classes.

**Result / what improved.** The pipeline adds useful observability columns, but on this test set the confidence threshold did not separate any difficult examples and the endpoint rule returned the same answer for every clip. No ground truth says whether those confidence flags are calibrated or whether the audio-event labels are useful. YAMNet event scores are model tags, not validated emotion or intent labels.

**Worth building properly?** Confidence and endpointing are worth validating with deliberately difficult audio and human labels, then calibrating thresholds against transcription errors and actual turn boundaries. Build YAMNet only if an identified downstream feature uses its classes; emotion should not be inferred from the current noise/event output.

## 1.6 Silero VAD endpointing

**What was tried.** The last `Day5.ipynb` experiment replaced the Whisper timestamp-based endpoint estimate with Silero VAD at 16 kHz. It used a minimum silence duration of 800 ms and threshold 0.5, while retaining Whisper confidence and YAMNet tags. Results were saved in [silero_vad_and_events_results.csv](results/silero_vad_and_events_results.csv).

**How it was measured.** It recorded speech-chunk count, final detected speech end, trailing silence, end-of-turn label, log probability, and event tags. Silero found 1–2 speech chunks per clip. Trailing silence ranged from 1.11 to 2.68 seconds and all eight were labeled finished. This agrees with the prior Whisper-tail method, which also labeled all eight finished. Recorded integrated runtimes ranged from 1.56 to 7.97 seconds; they include processing beyond ASR and are not directly comparable to the baseline transcription-only timings.

**Result / what improved.** Silero provides a speech-activity-based endpoint estimate and chunk count, but did not change any end-of-turn decision on these clips. A model-download/trust prompt appeared in the Colab output, and Librosa emitted `.m4a` decoding warnings.

**Worth building properly?** A prototype is reasonable for streaming voice interaction, where endpoint timing and false cut-offs matter. This batch test does not establish superiority. Compare both detectors against hand-labeled speech boundaries, especially internal pauses, short turns, background noise, and clips with no trailing silence.

---

# 2. [asrWithLLM.ipynb](asrWithLLM.ipynb) — Gemini correction and structured extraction


## 2.1 Whisper context prompt followed by Gemini transcript correction

**What was tried.** For each clip, the cell ran a no-context Whisper baseline, then a Whisper decode with an `initial_prompt` derived from the bot's previous question, then asked Gemini to correct the context-prompt transcript. The correction instructions asked Gemini to preserve Hindi/Hinglish style while repairing phonetic words, numbers, and domain terms. Output was saved in [whisper_gemini_context_results.csv](results/whisper_gemini_context_results.csv).

**How it was measured.** The CSV contains all three transcript stages, bot context, and total elapsed time. The saved per-file total ranged from 3.83 to 10.68 seconds. No separate Whisper-versus-API timing, token/cost report, or reference-scored accuracy metric was recorded.

**Result / what improved.** The examples show useful normalization: `kb1`'s garbled plan phrase became “subscription plan active”; `DMF` retained the date and 3500 amount in readable form; `EOT` standardized “account number”; and `LICS` improved punctuation and “pending.” However, `OOVB` became “service card” rather than the target SurgeX brand, illustrating that context alone can choose a plausible but wrong term. `PC` was normalized to “खाता बुक,” but this is not by itself proof that the product spelling or meaning is correct.

**Worth building properly?** Worth a guarded offline prototype. Add ground-truth transcript scoring, preserve raw and corrected text, constrain edits to acoustically plausible repairs, and explicitly test whether context causes invented content. Never overwrite the original transcript or automatically trust critical numbers.

## 2.2 Combined hotwords + bot context + Gemini

**What was tried.** A second version supplied both a domain hotword list and the bot question to Whisper, then gave the boosted transcript, question, and vocabulary to Gemini for correction. All eight files received scenario contexts. It wrote [whisper_hotwords_gemini_results.csv](results/whisper_hotwords_gemini_results.csv).

**How it was measured.** The saved CSV compares raw baseline, combined Whisper output, final Gemini output, and total per-file time (3.14–15.78 seconds). It was reviewed by comparing transcript examples; there is no automatic ground-truth score.

**Result / what improved.** The final stage recovered target spellings in useful examples: `PC` included “Khatabook”/“bahi-khata”; `kb1` included “SraVaani Pro” and “UPI”; `OOVB` included “SurgeX” and “cashback claim”; `DMF` retained its date and amount. But prompting caused a notable failure: for `nbdid` and `EOT`, the raw combined Whisper result echoed the bot instruction/context instead of the customer's full utterance; Gemini then returned only the requested ID. That may satisfy a slot-only task, but loses conversational content and shows how easily a prompted decode can be contaminated by its prompt. `PE` also changed the wording substantially, so semantic preservation needs explicit evaluation.

**Worth building properly?** The combination is promising for domain terms and slot-focused flows, but only with safeguards. Detect prompt echo, compare against unprompted hypotheses, preserve a full transcript alongside extracted slot values, and validate every corrected number/brand. Run a reference-based ablation so the separate effects of hotwords, context, and Gemini can be identified.

## 2.3 Comprehensive ASR + LLM voicebot pipeline

**What was tried.** The final cell combined Whisper with a bot-context prompt, Silero VAD, log-probability confidence, YAMNet event tags, and a Gemini call requesting JSON: cleaned transcript, intent, amount, order/account ID, brand/product, and whether confirmation is required. It saved [comprehensive_asr_llm_pipeline_results.csv](results/comprehensive_asr_llm_pipeline_results.csv).

**How it was measured.** Eight rows were produced. The table reports raw and cleaned transcripts, intent, entities, turn-finished status, confidence flag, confirmation flag, and elapsed time. All eight were labeled `Turn Finished? = YES`, `STT Unsure Flag = NO`, and `Confirm Slot? = NO`. Total measured time ranged from 3.84 to 50.39 seconds; the first `PC` row is a large outlier. The timer includes VAD/audio-event processing and the Gemini request, but the CSV does not separate these costs. There is no human-labeled intent/entity comparison.

**Result / what improved.** Gemini produced plausible structured outcomes in the examples: `kb1` → `payment_status` with SraVaani Pro; `LICS` and `OOVB` → `refund_issue`; `DMF` extracted amount `3500`; and `nbdid` extracted `984450`. It also normalized product names in the cleaned transcript. The confidence rule never requested confirmation, and the model returned `Confirm Slot? = NO` for every row, even though numeric IDs are high-impact fields. `EOT`'s account number is put in the schema's `order_id` slot, suggesting the current entity schema does not distinguish account IDs from order IDs. Several labels look reasonable, but none is validated against a reference label.

**Worth building properly?** Yes as an application prototype, not yet as an autonomous production decision-maker. Use a strict JSON schema and deterministic validation, separate order ID from account number, add a policy that requires confirmation for uncertain or high-impact values, and test intent/entity precision and recall on labeled calls. Track API failures, retries, per-stage latency, and cost; keep transcript provenance and raw ASR available for review.

---

## Result files and reproducibility notes

The CSVs document the experiments as follows:

- [transcription_results.csv](results/transcription_results.csv) — baseline language, duration, inference time, RTF, and transcript.
- [keyword_boosting_results.csv](results/keyword_boosting_results.csv) — three Whisper configurations, phrase-match counts, and hypotheses.
- [stt_confidence_and_events.csv](results/stt_confidence_and_events.csv) — Whisper confidence, silence-tail heuristic, YAMNet tags, and transcript.
- [silero_vad_and_events_results.csv](results/silero_vad_and_events_results.csv) — Silero speech chunks/endpoints plus confidence and YAMNet fields.
- [whisper_gemini_context_results.csv](results/whisper_gemini_context_results.csv) — baseline, context-prompt transcript, Gemini correction, and total time.
- [whisper_hotwords_gemini_results.csv](results/whisper_hotwords_gemini_results.csv) — baseline, combined boosted transcript, Gemini correction, and total time.
- [comprehensive_asr_llm_pipeline_results.csv](results/comprehensive_asr_llm_pipeline_results.csv) — integrated transcript, intent/entities, flags, and time.
- [transcriptionsOfRecordings.json](transcriptionsOfRecordings.json) — intended scripted test cases and expected text/target vocabulary; these were not joined to the CSVs to calculate accuracy.
- [intandlangresults.json](intandlangresults.json) — saved language/ITN experiment output plus a note that inverse normalization did not improve these audios and took extra time.


## Recommended next evaluation pass

1. Recover or create a fixed audio set with verified reference transcripts and labels for every target term, intent, amount, date, and ID.
2. Re-run the same clips with warm-up and repeated timing measurements; record hardware, package/model versions, and Gemini model/version.
3. Score baseline, hotwords, context, and LLM as separate ablations with WER/CER plus exact-match metrics for business-critical slots.
4. Label speech boundaries and ambiguous examples to compare Whisper timestamps with Silero; evaluate false turn endings and delay, not just the final YES/NO decision.
5. For an application prototype, retain raw hypotheses, validate model output against a schema, require confirmation for sensitive/ambiguous slots, and measure end-to-end cost and latency.
