# Research: a light, good-quality Basque TTS that runs in real time on CPU

Date: 2026-10-14. Scope: pick a base model, a data strategy (real + synthetic) and a
training/evaluation recipe for a Basque (euskara) TTS that runs faster than real time on a
laptop CPU with quality clearly above Piper. Everything below was checked against primary
sources or measured on a single data-centre GPU (46 GB VRAM).
Uncertain claims are marked as such.

---

## 1. Executive summary

- **Base model: Kyutai Pocket TTS.** It is the only candidate that is ~100M params, measured
  faster than real time on 2 CPU cores, permissively licensed (MIT code, CC-BY-4.0 weights),
  and ships official training code whose headline use case is "train a new language".
  Eleven community languages have been released with it since the training code came out in
  Aug 2026 (Czech, Korean, Hindi, Estonian, Welsh, Polish, Turkish, Indonesian, Greek,
  Persian, Japanese, Mooré/Dyula), several at Basque-like data scales (70–500 h).
- **The previous attempt on the VM (`~/polttsiko`) is a good starting point but used the
  wrong recipe.** It planned to fine-tune the 6-layer Spanish *student* directly. The
  official recipe (and every community model that worked) is: fine-tune a **24-layer
  teacher** with a fresh Basque tokenizer (`finetune_language.yaml`), then **depth-distill**
  to the 6-layer student (`depth_distill.yaml`). The data prep, text normalizer and
  dataset policy in `polttsiko` can be reused.
- **OmniVoice (k2-fsa) is an excellent Basque teacher technically.** Measured on the GPU:
  intelligible Basque in auto-voice and voice-design modes (CER 0–5 % with a Basque
  Whisper), RTF 0.05–0.13 on GPU at 32 steps, ~2 GB VRAM in fp16 (not 3 GB). It is **not**
  real time on CPU (RTF 2.1–4.2 with 8 threads, 7.7–14.9 with 2 threads), which confirms the
  need for a small student.
- **The blocker on OmniVoice is licensing, not quality.** Code is Apache-2.0 but the
  **weights are CC-BY-NC**, and its Higgs audio tokenizer carries a Boson community license
  clause forbidding use of outputs "to improve any other large language model". A student
  trained on OmniVoice output would at minimum have to be released non-commercially, and the
  Boson clause is a grey area. This is a decision for the project owners (see §7).
- **Real data is enough to get started without synthetic data.** Open Basque speech:
  HiTZ-Aholab studio corpus (~34 h, 2 speakers, CC-BY-4.0), OpenSLR 76 (14 h, 52 speakers,
  CC-BY-SA-4.0), Common Voice eu (~300 h validated, >10k speakers, CC0), Basque Parliament
  (~410 h Basque, CC0, broadcast quality). That is ~760 h total, above Pocket TTS's "100 h
  minimum, 1000 h for a strong model" guidance. Quality of the crowdsourced part is the risk.
- **Recommended plan:** (1) Pocket TTS teacher fine-tune on the real ~760 h (clean studio
  data upweighted) + distill to 6 layers; (2) in parallel, build the OmniVoice synthetic
  corpus pipeline (random voices + Basque reference prompts, ASR-filtered) and train a second
  model on real+synthetic to measure the gain; (3) decide the release license once the
  numbers are in. Evaluation: WER with `xezpeleta/whisper-large-v3-eu` (primary) and
  `HiTZ/stt_eu_conformer_transducer_large_v2` (second opinion), UTMOS, speaker similarity.

---

## 2. Starting point: what already exists

### 2.1 Current systems

| System | Quality | CPU | Notes |
|---|---|---|---|
| Piper (VITS, ~20–60M) | basic, robotic prosody | RTF ~0.1–0.4 on 2 ARM cores | espeak-ng has `eu`; no official `eu_ES` voice; community Antton/Maider ports (itzune) exist; GPL-3.0 since the OHF-Voice fork |
| HiTZ/TTS-eu_antton, TTS-eu_maider (VITS/aHoTTS, Apache-2.0) | the reference open Basque voices | yes (ONNX) | 2 studio speakers, no cloning |
| OmniVoice (k2-fsa) | very good Basque | **no** (see §4) | 0.8B incl. codec; GPU only in practice |
| facebook/mms-tts-eus (VITS 36M) | intelligible, monotone | yes | CC-BY-NC-4.0 |
| Commercial | Azure AinhoaNeural/AnderNeural, Google eu-ES-Standard-B, Elhuyar neural TTS | — | ElevenLabs has no Basque |

### 2.2 The earlier `polttsiko` attempt (May–Sep 2026)

State: data policy, HiTZ composite data prep script, Basque text
normalizer + tests, tokenizer recipe, fine-tune and inference configs written; **forced
alignment, tokenizer training and the fine-tune run were never executed**. Findings worth
keeping:

- Baseline (Pocket TTS 2.1.0, laptop CPU): Spanish 6L model on Basque text runs at ~1.9x
  real time and is "not far from the desired result": rhythm and voice quality fine, clear
  Spanish accent, mispronunciations (trilled /r/, affricates, stress), poor tokenization of
  agglutinative words.
- An earlier hand-rolled training loop collapsed after ~4k steps because it lacked EMA, CFG
  dropout, the `lsd` objective, `flow_batch_multiplier` and batch ≥ 64. The official
  `training/` package has all of these.
- Decisions to revisit: it starts from the **6L Spanish student** (`pocket_tts/config/spanish.yaml`)
  and trains a vocab-4000 tokenizer to keep the embedding table size. The official path is
  24L teacher → distill (§5).

### 2.3 Measured on the VM today

Pocket TTS 3.1.0, CPU build, `torch.set_num_threads(2)`, Spanish models on Basque text:

| Model | audio (s) | gen (s) | x real time |
|---|---|---|---|
| spanish (6L, ~100M) | 4.0 / 6.3 | 1.16 / 1.72 | **3.5 / 3.7** |
| spanish_24l (~336M) | 5.1 / 7.8 | 4.88 / 7.63 | 1.05 / 1.03 |

So on this (slow, server-class) CPU the 6L student is comfortably real time and the 24L
teacher is borderline. On a MacBook Air M4 Kyutai reports ~6x; a community Polish 6L model
reports RTF 0.21 fp32 / 0.11 int8 on a desktop CPU.

---

## 3. Base-model survey (Oct 2026)

Full survey criteria: < ~350M params, real-time on CPU, official training/fine-tuning code,
new-language precedent, license, community momentum.

| Model | Params | License (code/weights) | Official train code | New-language recipe | Text frontend | CPU RTF (lower = faster) | Cloning | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Pocket TTS** (Kyutai) | 100M student / 336M teacher | MIT / CC-BY-4.0 | yes (Aug 2026) | yes, Czech worked example | SentencePiece on raw text | 0.17–0.43 (2 cores); int8 ~0.11 | yes | **primary** |
| Piper (OHF-Voice piper1-gpl) | 20–60M VITS | GPL-3.0 | yes | yes (`--data.espeak_voice eu`, warm-start from any-language ckpt) | espeak-ng IPA (`eu` exists) | 0.12–0.37 | no | fallback / baseline |
| MOSS-TTS-Nano (OpenMOSS) | 100M + 20M codec | Apache-2.0 | yes (Apr 2026), ~3.2 GiB VRAM | not documented (community Persian) | LLM tokens | unpublished | yes | runner-up, worth a parallel try |
| ZipVoice (k2-fsa) | 123M flow-matching | Apache-2.0 | yes (k2 recipes) | Catalan community FT exists (WER 7–11 %) | espeak phonemes / chars | ONNX CPU, ~RT (unverified) | yes | option |
| Kokoro-82M | 82M StyleTTS2 | Apache-2.0 | **no** (unofficial kikiri-tts German recipe, 51 h) | — | misaki/espeak-ng, no `eu` path | 0.2–1.9 depending on CPU | no | not as base |
| Supertonic 3 | 99M | MIT / OpenRAIL-M | **no**, repo archived 2026-09-09 | — | raw text | ~0.015 (M4 Pro) | no | no |
| KittenTTS (legacy) | 15–80M | Apache-2.0 | **no** | — | espeak-ng | 0.4–0.7 | no | no |
| NeuTTS Nano/Air | 229M / 552M | NeuTTS Open License / Apache | yes (English-oriented) | not documented | espeak-ng | real-time claimed; slow in Picovoice bench | yes | no |
| Chatterbox Nano / Multilingual | 110M / 500M | MIT | no (community) | community vocab extension | chars | Nano ~0.33 on 8 cores; 500M not RT | yes | no Basque, no training code |
| Matcha-TTS | 18M | MIT | yes | via espeak-ng | espeak-ng | 0.14–0.40 | no | research-grade, low activity |
| IMS-Toucan | tens of M | Apache-2.0 | yes, `lang="eus"` supported | yes | articulatory features | unpublished | limited | very-low-resource niche |
| OmniVoice | 0.8B | Apache-2.0 / **CC-BY-NC** | yes (full + LoRA) | 646 langs incl. **eu** | Qwen3 byte-BPE | **2–15** (measured, §4) | yes + voice design | teacher / eval only |
| Qwen3-TTS 0.6B, CosyVoice3 0.5B, VoxCPM, F5-TTS, Fish S2, Kyutai TTS 1.6B, Orpheus, Dia, Zonos | ≥0.5B | mixed (F5, Fish: NC) | mostly yes | — | — | not RT | — | too big for CPU, and no Basque except Fish (research license) |

Why Pocket TTS wins for this project:

- **Architecture fits the goal.** CALM-style continuous-latent LM (6 transformer layers,
  d=1024) + a 20M Mimi codec at 12.5 Hz, ~200 ms first audio, streaming, 2 CPU cores. Many
  ports: ONNX/sherpa-onnx (12 language bindings, WebAssembly), Rust/candle, MLX, audio.cpp
  GGUF, Wyoming/Home Assistant.
- **The training code targets exactly our case.** `training/README.md` walks through
  adding Czech: forced alignment with a language-specific wav2vec2-CTC, SentencePiece
  tokenizer, `finetune_language.yaml` (reset text embeddings, keep lr 2e-4), then
  `depth_distill.yaml`. On 976 h of Czech the language fine-tune hit 12 % WER by 10k steps
  where a scratch run was at 46 %.
- **Data guidance is explicit.** "100 h minimum, 200 h for a model that speaks, 1000 h+ for
  one that sounds good." Community data points: Welsh 173 h from scratch (works, but ~39 %
  of long sentences fail voicing); Mooré/Dyula ~70 h each (works, early overfitting, CER
  13–15 %); Turkish 262 h **of synthetic voices** fine-tuned from the EN 24L teacher in 1.5 h +
  4.4 h distill, WER 11.9 % (same as real audio); Polish 425 h, teacher 2.4 h on 1×H100,
  student 62 min, WER 8.5 %.
- **Permissive license** lets the result be published as a community model listed in the
  upstream README (Kyutai invites this).

Known weaknesses to plan for: raw-text tokenizer means **numbers, dates and abbreviations
must be verbalised** by our own Basque normalizer (no `num2words` Basque, no NeMo `eu`);
cloning of unseen voices degrades when training has few speakers; training needs a GPU with
~16 GB (batch 16 × accum 4) to ~56 GB (batch 64).

---

## 4. OmniVoice: what it is and why its Basque is good

Source: paper arXiv:2604.00688 (Zhu, Ye, Kang, Yao, Guo, Kuang, Han, Zhuang, Lin, Povey;
Xiaomi / k2-fsa), GitHub `k2-fsa/OmniVoice` (14.6k stars), HF `k2-fsa/OmniVoice`.

**Architecture.** Single-stage *masked discrete diffusion* ("diffusion language model") TTS:
a bidirectional transformer initialised from **Qwen3-0.6B** (28 layers, d=1024) predicts
8 RVQ codebooks of the **Higgs Audio V2 tokenizer** (24 kHz, 25 Hz, 1024 entries/codebook)
by iterative unmasking (32 steps default, 16 for speed, CFG scale 2.0). Total ~0.8B params
(we measured 812.9M loaded) including codec; 3.3 GB on disk (fp32 safetensors 2.45 GB +
tokenizer 0.8 GB). Text goes in as **raw text through the Qwen3 byte-level BPE** with a
language tag (`<|lang_start|>eu<|lang_end|>`), no G2P, no phonemes.

**Training data.** 581k h across 646 languages from ~50 open datasets (Emilia, Emilia-YODAS,
Common Voice, YODAS-Granary, Meta Omnilingual ASR corpus, FLEURS, EuroSpeech, CML-TTS, …).
**Basque: 479.86 h** (vs Spanish 27,560 h, Catalan 3,359 h, Galician 209 h). The per-dataset
composition is not published; given the sources, it is most likely Common Voice eu +
YouTube-derived (YODAS) + possibly Meta's corpus, i.e. crowd/web speech, not studio data.
No synthetic data was used. Trained 2M steps on 8×H800 in ~10 days.

**Why Basque pronunciation and prosody come out well (paper facts + our interpretation):**

1. ~480 h of real Basque is a lot for a TTS language (most Pocket TTS community models used
   less), and the paper upsamples low-resource languages (repeat factor
   `max(1, round((D_max/D_i)^(1-β)))`, β=0.8).
2. Qwen3-0.6B was pretrained on 36T tokens in 119 languages **including Basque**, so the
   backbone already models Basque word forms; the paper credits LLM init with a large WER
   gain (1.57 vs ≥2.52 random init).
3. Byte-level BPE plus Basque's near-phonemic orthography means no front-end is needed and
   no G2P errors are introduced.
4. Bidirectional (non-autoregressive) decoding sees the whole sentence at once, which the
   authors argue helps intelligibility and, in our reading, global prosody.
5. Massive Spanish/Catalan data shares most of the phone inventory (interpretation; the
   paper does not claim this).
6. "Full-codebook random masking" training (all 8 codebooks masked independently) is the
   paper's main modelling contribution (WER 1.57 vs 3.00 SoundStorm-style).

There is **no official Basque evaluation**: FLEURS-102 and the MiniMax-24 set do not include
Basque.

**Smoke test on the GPU (fp16, `language_id="eu"`, files in `samples/omnivoice_smoke/`):**

| sample | mode | steps | audio (s) | gen (s) | RTF | peak VRAM |
|---|---|---|---|---|---|---|
| auto1 | auto voice | 32 | 2.65 | 1.09 | 0.41 (first call, warm-up) | 1.98 GB |
| auto1 | auto voice | 16 | 2.54 | 0.28 | 0.11 | 1.98 GB |
| auto2 | auto voice | 32 | 4.50 | 0.58 | 0.13 | 2.0 GB |
| design_f | "female, middle-aged, moderate pitch" | 32 | 4.80 | 0.56 | 0.12 | 2.0 GB |
| design_m | "male, young adult, low pitch" | 32 | 5.08 | 0.57 | 0.11 | 2.0 GB |
| long (3 sentences) | auto voice | 32 | 11.63 | 0.61 | 0.05 | 2.1 GB |

Intelligibility, transcribed with `xezpeleta/whisper-large-v3-eu` (WER / CER):
auto1@16 0.125 / 0.053; auto1@32 **0 / 0**; auto2 **0 / 0**; design_f **0 / 0**;
design_m 0.20 / 0.026 ("Tknikako" → "Teknikako", "ikastetxeentzat" → "ikastetxentzat");
long 0.04 / 0.006. Stock `openai/whisper-large-v3` gave WER 0.20–0.40 on the same files, all
from its own errors (merged words, dropped "h"), so **stock Whisper must not be used as a
Basque filter** (it saw ~21 h of Basque; 38.9 % WER on CV13 vs 10.5 % fine-tuned).

Observations: voice design is documented as zh/en-only but produced stable Basque here;
VRAM is ~2 GB, not 3 GB; a 20 s reference clip reportedly spikes to ~8 GB (GitHub issue
#199). Streaming is not supported by the architecture (issue #6).

**CPU timing (PyTorch fp32, same VM, 4.5 s sentence):**

| threads | steps | gen (s) | RTF |
|---|---|---|---|
| 8 | 32 | 18.9 | 4.2 |
| 8 | 16 | 9.3 | 2.1 |
| 2 | 32 | 64.4 | 14.9 |
| 2 | 16 | 32.1 | 7.7 |

A third-party Strix Halo (16 cores) report gives RTF 1.5–5.5. audio.cpp ships OmniVoice as
GGUF Q8 but publishes only CUDA numbers (7–24x RT on an RTX 4090). Conclusion: OmniVoice
cannot be the deployed model; it is a teacher and a reference.

**Fine-tuning.** Full pipeline in `examples/` (data → Higgs tokens → `accelerate launch`),
full FT (lr 5e-5, 5k steps default) and LoRA (r=16, 19M trainable, lr 1e-4). One could
fine-tune OmniVoice on the HiTZ studio corpus to get an even better Basque teacher, but the
license problem (§7) stays.

---

## 5. Recommended recipe: Basque Pocket TTS

### 5.1 Model path

```
EN 24L teacher (english_2026-04_24l)   or   Spanish 24L (spanish_24l)
        │  finetune_language.yaml: reset_text_embedding, new Basque SentencePiece, lr 2e-4
        ▼
Basque 24L teacher  ──depth_distill.yaml (CFG baked in, lr 4e-4)──▶  Basque 6L student (~100M)
```

- Start from the **English 24L** teacher as the official recipe does (it is the most
  trained checkpoint). Run a second teacher from **Spanish 24L** as an experiment: the
  polttsiko baseline suggests Spanish transfers well to Basque. Keep whichever has the
  lower WER at 15k steps.
- Do **not** fine-tune the 6L student directly: Kyutai states two-stage training beats a
  6L model trained directly, and the distilled student matches the teacher (WER 0.76 vs 0.82
  on English).
- Tokenizer: `train_tokenizer.py` default vocab (matches `n_bins`, no override needed when
  `reset_text_embedding: true`). Check tokenization of long agglutinative words; the polttsiko
  concern about vocab 4000 only applied to warm-starting the embedding table, which the
  official recipe does not do.
- Mimi codec stays frozen (language-agnostic); confirmed by all community models.

### 5.2 Data

Real, open Basque speech (license-compatible with a CC-BY release):

| Corpus | Hours (Basque) | Speakers | Quality | License |
|---|---|---|---|---|
| HiTZ-Aholab studio corpus (Zenodo 17952596) | ~34 (Maider 17.5 + Antton 16.8) | 2 pro | studio 48 kHz | CC-BY-4.0 |
| HiTZ-Aholab emotional + re-enacted spontaneous sets (Zenodo 18804769, 20638013) | 16k + 2.5k utt/speaker | same 2 | studio | check (likely CC-BY) |
| OpenSLR SLR76 | 14 | 52 | near-studio (KM184, insulated room) | CC-BY-SA-4.0 |
| Common Voice eu (CC0; v20 mirror `xezpeleta/mozilla-common-voice-20-eu`) | ~300 validated | >10k | crowd, MP3 | CC0 |
| Basque Parliament (`gttsehu/basque_parliament_1`) | ~410 Basque (+1019 Spanish) | hundreds | broadcast | CC0 |
| `HiTZ/composite_corpus_eu_v2.1` (= CV18 300 h + Parliament 370 h + SLR76 6 h) | 676 train | — | mixed | CC-BY-4.0 |
| YODAS eu000 | 20 | ? | YouTube | CC-BY-3.0 |

Total ~760 h, of which ~50 h is TTS-grade. Strategy:

- Use the composite corpus (already scripted in polttsiko) **plus the two Zenodo studio sets
  and the full SLR76** (composite only includes 6 h of it).
- Filter crowd/parliament audio: DNSMOS/UTMOS threshold, duration 1–30 s, ASR agreement
  with the transcript (both Basque ASRs below), drop clips with music/overlap.
  Pocket TTS's own advice: noisy prompts make noisy output, so keep the clean studio data
  for the voice prompts and for evaluation.
- Upweight studio data (repeat 3–5x in the manifest) so acoustic quality is learned from it.
- Forced alignment: `training/scripts/align_data.py` with a Basque CTC model
  (`cahya/wav2vec2-large-xlsr-basque` or `ivangtorre/wav2vec2-large-xlsr-53-basque`;
  polttsiko planned `deepdml/wav2vec2-large-xls-r-300m-basque`). Alignment is the blocking
  step in polttsiko; validate on 100 files before running the full set.
- Text normalization: Pocket TTS consumes raw text, so numbers (vigesimal: hogei, berrogei…,
  with case endings), dates, abbreviations and Spanish names must be verbalised in the
  training transcripts and at inference. Extend `polttsiko_basque_text_norm.py`; no open
  Basque normalizer exists (AhoTTS has a legacy C++ one).

### 5.3 Synthetic data with OmniVoice (if licensing is accepted, §7)

Evidence that this works for small students:

- "Training TTS with Purely Synthetic Data" (arXiv 2512.17356): a 21M Matcha-TTS trained on
  ~100 h of XTTS-synthesised speech matched or beat the real-data model (WER 3.49 vs 4.43,
  UTMOS 3.75 vs 3.74). Sensitivities: ≥50 speakers for WER/UTMOS, ~500 for speaker
  similarity; keep text diverse (phoneme coverage collapses with reused text); synthetic set
  must be nearly noise-free (>0.1 % noisy clips hurts); read style beats conversational.
- Pocket TTS Turkish community model: 262 h of synthetic voices, WER equal to real audio.
- Thai (arXiv 2609.03502): OmniVoice as teacher → 82M on-device student, CER 3.7 %; teacher
  errors become targets, so filter.
- Kokoro-82M itself was trained largely on synthetic audio from closed TTS APIs.

Proposed pipeline (all on the GPU):

1. **Text pool** (~300k sentences): Common Voice eu sentence pool (~160k CC0 sentences),
   Basque Wikipedia / Euskal Wikipedia sentences, HiTZ corpus sentence design as a template
   (declarative / interrogative / exclamative / Spanish names / numbers), plus
   normalizer-generated number/date sentences. De-duplicate; maximise diphone coverage.
2. **Voices**: mix of (a) OmniVoice **auto voice** (random), (b) **voice design** prompts
   (gender × age × pitch grid; worked in Basque in the smoke test), (c) **cloning from real
   Basque reference clips** (SLR76's 52 speakers, CV top speakers, the 2 HiTZ speakers) so
   voices are native-Basque-accented. Target ≥200 distinct voices; de-duplicate by ECAPA
   cosine < 0.8.
3. **Generation**: `omnivoice-infer-batch` with FlashInfer, batch 8, 32 steps, fp16,
   `language_id="eu"`, `denoise=True`. Measured single-stream RTF ≈0.1, so 500 h ≈ 50 GPU-h
   single-stream; batched with FlashInfer (0.0115 RTF on H100 at batch 8) it should be well
   under a day on the GPU.
4. **Filtering**: ASR round-trip with `xezpeleta/whisper-large-v3-eu` **and**
   `HiTZ/stt_eu_conformer_transducer_large_v2` (keep clips where both agree, CER ≤ 5 %);
   duration/text-length ratio outliers (drop bottom 1 % / top 5 %); UTMOS ≥ 3.5; speaker
   embedding check against the intended prompt; trailing-silence trim.
5. **Alignment and training** as in §5.2; synthetic data is clean so alignment is easy.

Expected gain: speaker diversity (cloning of unseen voices) and prosody on hard text
(numbers, names). Risk: the student inherits OmniVoice's artifacts and its slight
"Spanish-flavoured" rendering of some words; keep the studio data in the mix.

### 5.4 Compute (single 46 GB GPU)

- Latent precompute: ~6 GB per 1000 h, once.
- Teacher fine-tune: batch 32 × grad_accum 2 (effective 64). polttsiko estimated ~0.77
  steps/s at batch 64 on a comparable GPU; WER plateaus by ~15k steps (≈6 h), acoustic quality keeps
  improving to 50–100k (1–2 days). Stop early by watching `progress.jsonl`.
- Distillation: 9.7 GB VRAM at batch 64 on H100; WER parity by ~50k steps, UTMOS to ~150k
  (≈1–2 days on the GPU).
- Disk: ~60 GB per 1000 h of prepared 24 kHz audio; the VM has 47 GB free, so prepare data
  on an attached volume or prune `~/work` (cmix experiments, ~0.5 GB) and HF caches.

### 5.5 Evaluation

- **WER/CER**: `xezpeleta/whisper-large-v3-eu` (WER 4.84 on CV18; CT2 variants for speed)
  as primary; `HiTZ/stt_eu_conformer_transducer_large_v2` (NeMo, WER 2.50 on CV18, 11.9 on
  SLR76) as the independent second scorer. Report both; adapt
  `training/eval/librispeech.py` to a Basque test list (SLR76 test + HiTZ held-out
  sentences + the polttsiko curated list in `scripts/generate_samples.py`).
- **UTMOS** and **speaker similarity** (WavLM-base-plus-sv) from the official eval script.
- **Listening test** against: Piper Antton/Maider, HiTZ VITS, OmniVoice (teacher
  ceiling), Azure AinhoaNeural.
- **CPU benchmark**: RTF, first-audio latency and RAM at 2 threads, fp32 and int8
  (`quantize=True`), on a laptop; target ≥3x real time (the Spanish 6L model does 3.5x on the
  VM's CPU).

---

## 6. Alternatives and fallbacks

- **Piper fine-tune from a Spanish/Catalan checkpoint** with espeak-ng `eu`: cheapest, works
  on tiny data, fastest CPU. Keep as a baseline and as an emergency deliverable. GPL-3.0 and
  no cloning.
- **MOSS-TTS-Nano**: same size class, Apache-2.0, 3.2 GiB VRAM fine-tune, ONNX CPU path;
  unproven for a new language. Good second experiment if Pocket TTS results disappoint.
- **ZipVoice**: Catalan precedent (WER 7–11 %), Apache-2.0, k2 training stack; CPU speed
  claims unverified.
- **Running OmniVoice on CPU via audio.cpp GGUF Q8**: no CPU numbers published; our PyTorch
  CPU measurement (RTF 2–15) makes real time implausible without a much smaller model.
- **Kokoro**: no training code; would require the unofficial StyleTTS2 recipe and hand-wiring
  espeak-ng `eu`. Not recommended.

---

## 7. Licensing: the decision the owners must make

| Asset                                           | License                                                                                                                                                                           | Consequence                                                                                                            |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Pocket TTS code / weights                       | MIT / CC-BY-4.0                                                                                                                                                                   | student can be released CC-BY-4.0                                                                                      |
| HiTZ-Aholab studio corpora                      | CC-BY-4.0                                                                                                                                                                         | attribution                                                                                                            |
| SLR76                                           | CC-BY-SA-4.0                                                                                                                                                                      | share-alike on the *data*; model release under CC-BY is common practice but note it                                    |
| Common Voice, Basque Parliament                 | CC0                                                                                                                                                                               | none                                                                                                                   |
| **OmniVoice weights**                           | **CC-BY-NC-4.0** ("due to constraints from its training data, e.g. Emilia")                                                                                                       | synthetic audio is arguably an NC derivative; a model trained on it should be released NC unless k2-fsa says otherwise |
| **Higgs Audio V2 tokenizer** (inside OmniVoice) | Boson Higgs Audio 2 Community License: "will not use the Higgs Materials or any output … to improve any other large language model"; >100k annual users needs an expanded license | grey area for a 100M TTS student; legal review or a written OK from Boson/k2-fsa                                       |
| HiTZ VITS voices (Antton/Maider)                | Apache-2.0, data CC-BY-4.0                                                                                                                                                        | synthetic audio from them is clean (itzune already did this), but only 2 voices                                        |
| Fish Audio S2 (has Basque)                      | research license, forbids training other models on outputs                                                                                                                        | unusable as teacher                                                                                                    |
| Azure / Google Basque voices                    | commercial terms restrict training competing models                                                                                                                               | unusable as teacher                                                                                                    |

Options, in order of preference:

1. **Ask k2-fsa** (GitHub issue) whether Basque synthetic data for an open, non-profit
   community model is acceptable and whether the Boson clause applies; precedent: Thai paper
   distilled OmniVoice into an 82M student and published it.
2. **Two releases**: a CC-BY-4.0 model trained on real data only, and an NC "plus" model
   trained on real + OmniVoice-synthetic if it is measurably better.
3. **Real data only** (760 h is above Kyutai's minimum; the Polish/Estonian community
   models reached WER 5–8 % with 400–1200 h of real speech).

---

## 8. Proposed next steps

1. Create the new repo from `polttsiko`'s reusable parts (data prep, normalizer, tests,
   dataset policy) and switch the configs to the teacher→distill recipe.
2. Data: download HiTZ Zenodo sets + SLR76 full + composite corpus; run the quality filters;
   validate forced alignment on a 100-file sample with two CTC models; pick one.
3. Train the Basque SentencePiece tokenizer; inspect splits on long words and numbers.
4. Teacher fine-tune from EN 24L (and Spanish 24L in parallel if VRAM allows); evaluate WER
   with both Basque ASRs at 5k/10k/15k steps; keep the better init.
5. Distill to 6L; CPU benchmark; listening test vs Piper/HiTZ/OmniVoice.
6. In parallel, implement the OmniVoice synthetic pipeline (text pool, voice grid,
   batch generation, dual-ASR filtering) and open the license question with k2-fsa.
7. If synthetic data is cleared, train the real+synthetic variant and compare.
8. Package: `model.safetensors` + `tokenizer.model` + yaml on Hugging Face; submit to the
   Pocket TTS community-models list.

---

## 9. Artifacts from this research

- `samples/omnivoice_smoke/*.wav` — OmniVoice Basque outputs (auto voice, voice design,
  long text) with `results.json` (timing) and `asr_eu_results.json` (Basque Whisper WER/CER).
- VM: `~/omnivoice-smoke/` (venv with `omnivoice` 0.2.x, torch 2.8 cu128; scripts
  `smoke_omnivoice.py`, `asr_cpu.py`, `asr_eu.py`); `~/pockettts-smoke/` (CPU-only Pocket TTS
  3.1.0, `bench.py`); `~/pocket-tts` (upstream checkout at v3.1.0, Sep 4); `~/polttsiko`
  (previous attempt).

## 10. Sources

- Pocket TTS: https://github.com/kyutai-labs/pocket-tts (training/README.md,
  training/configs/finetune_language.yaml, depth_distill.yaml); CALM paper
  https://arxiv.org/abs/2509.06926; community models listed in the README and on HF
  (vvolhejn/pocket-tts-czech, seastar105/pocket-tts-korean-300m, saryps-labs/pocket-tts-hindi,
  cbentes/pocket-tts-estonian, EryriLabs/pocket-tts-cymraeg, shefowl/pocket-tts-polish-6l,
  alperiox/pocket-tts-turkish, anak10thn/pocket-tts-indonesian, myned-ai/pocket-tts-greek,
  Minervus00/ft_pocket_tts).
- OmniVoice: https://github.com/k2-fsa/OmniVoice (README, docs/languages.md,
  docs/training.md, docs/lora_finetuning.md, docs/generation-parameters.md,
  examples/config/*.json); paper https://arxiv.org/abs/2604.00688; HF
  https://huggingface.co/k2-fsa/OmniVoice (license section); Higgs tokenizer license
  https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/blob/main/LICENSE;
  issues #6 (streaming), #7 (RTF, Malay FT), #199 (VRAM).
- Basque data: https://zenodo.org/records/17952596, https://zenodo.org/records/18804769,
  https://zenodo.org/records/20638013, https://www.openslr.org/76/,
  https://huggingface.co/datasets/HiTZ/composite_corpus_eu_v2.1,
  https://huggingface.co/datasets/gttsehu/basque_parliament_1,
  https://huggingface.co/datasets/xezpeleta/mozilla-common-voice-20-eu,
  https://huggingface.co/datasets/espnet/yodas.
- Basque ASR: https://huggingface.co/xezpeleta/whisper-large-v3-eu,
  https://huggingface.co/HiTZ/stt_eu_conformer_transducer_large_v2,
  https://huggingface.co/cahya/wav2vec2-large-xlsr-basque; stock-Whisper Basque weakness
  https://arxiv.org/abs/2503.23542.
- Existing Basque TTS: https://huggingface.co/HiTZ/TTS-eu_antton,
  https://github.com/itzune/basque-piper-tts, https://huggingface.co/facebook/mms-tts-eus,
  https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md.
- Synthetic-data TTS: https://arxiv.org/abs/2512.17356 (purely synthetic data),
  https://arxiv.org/abs/2609.03502 (OmniVoice → 82M Thai student),
  https://aclanthology.org/2025.emnlp-main.629.pdf (speech back-translation, filtering),
  https://huggingface.co/hexgrad/Kokoro-82M (synthetic training data statement),
  https://arxiv.org/abs/2505.17589 (CosyVoice 3 data pipeline).
- Other candidates: https://github.com/OpenMOSS/MOSS-TTS-Nano, https://github.com/k2-fsa/ZipVoice,
  https://github.com/ErikUPV/ZipVoice-CA, https://github.com/semidark/kikiri-tts,
  https://github.com/0xShug0/audio.cpp, https://picovoice.ai/blog/on-device-tts/.

## 7. In-browser inference (added 2026-10-20)

Kyutai's web demo calls a backend (`pocket-tts-api.kyutai.org`); it is not WebAssembly. The in-browser
engines are community projects, the mature one being gradium-ai's `xn-ptts` (npm `phonon-tts`, Rust →
WASM, CPU SIMD/threads or WebGPU). The released Basque 6-layer model runs in it unchanged apart from
dropping the 32-dim speaker projection (unused with precomputed voice states), a tokenizer conversion and
the engine's Pocket TTS config: 3.8× real time on 3 browser threads with f32 weights, first audio in
~180 ms, WER 0 on the test sentences. Details, limits (native loader, q8, no cloning) and tooling in
`scripts/phonon/README.md`.
