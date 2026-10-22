---
language:
- eu
license: cc-by-4.0
library_name: pocket-tts
pipeline_tag: text-to-speech
tags:
- text-to-speech
- pocket-tts
- kyutai
- basque
- euskara
- on-device
base_model: itzune/pocket-tts-basque-24l
datasets:
- itzune/bssd
---

# Pocket TTS Basque small (euskarazko ahots-sintesi arina)

A smaller, faster version of [itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque),
the Basque (euskara) text-to-speech model based on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).
Its backbone is 8 layers × 512 wide (14.7 M parameters with the text table) instead of 6 layers × 1024
(~80 M), and it runs about **1.6-1.9× faster on CPU** than the base model, at close intelligibility and
naturalness. It ships with five ready-made Basque voices and supports voice cloning from a short clip.

This release shrinks the **backbone only**, following the first stage of Kyutai's
[small-model recipe](https://kyutai.org/blog/2026-10-20-pocket-tts-small/). The sampling head (9.3 M) and the
Mimi audio codec (20.1 M, of which the 10 M encoder is used only for voice cloning) are the standard ones, so
the whole file is ~44 M parameters (176 MB, float32). Smaller heads and decoders are future work.

## Usage

With [`uv`](https://docs.astral.sh/uv/) installed, no setup is needed (Pocket TTS ≥ 4.0):

```bash
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque-small/basque_small.yaml \
  --voice hf://itzune/pocket-tts-basque-small/voices/Maider.safetensors \
  --text "Kaixo! Hau euskarazko ahots sintetiko bat da." --output kaixo.wav
```

Python:

```python
from pocket_tts import TTSModel
import soundfile as sf

model = TTSModel.load_model(config="hf://itzune/pocket-tts-basque-small/basque_small.yaml")
voice = model.get_state_for_audio_prompt("hf://itzune/pocket-tts-basque-small/voices/Antton.safetensors")
audio = model.generate_audio(voice, "Egun on, zer moduz zaude gaur?")
sf.write("egun_on.wav", audio.squeeze().numpy(), model.sample_rate)  # 24 kHz
```

Bundled voices (`voices/`): **Maider** (female) and **Antton** (male), the two professional speakers of the
[HiTZ-Aholab studio corpus](https://zenodo.org/records/17952596) (CC BY 4.0); and three synthetic voices,
**Nahia**, **Ane** (female) and **Aitor** (male). Voice state files are specific to this model; the base
model's voice files do not work here and vice versa.

Notes:

- Use the provided config: it sets `default_eos_threshold: 0.0`. With Pocket TTS's default (−4) this model
  ends some sentences early.
- Write the text as it should be read: numbers, dates, abbreviations and symbols spelled out in words
  (e.g. `hogeita lau` rather than `24`). The dataset's normaliser
  ([`bssd.text_norm`](https://huggingface.co/datasets/itzune/bssd)) does this for standard Basque.
- Only clone voices you have the right to use.

## Speed

Generation only (no model load), three sentences of 2-8 s, Maider voice, PyTorch CPU, Pocket TTS 4.0.1, as a
multiple of real time (higher is faster):

| model | 1 thread | 2 threads |
|---|---|---|
| **this model** | 8.1× | 13.0× |
| [pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque) | 5.0× | 6.7× |

Measured on a laptop x86 CPU; absolute numbers vary by machine, the ratio is the relevant part.

## Evaluation

Held-out BSSD test sentences, 40 per voice, scored with two independent Basque ASR systems
(`xezpeleta/whisper-large-v3-eu` and `HiTZ/stt_eu_conformer_transducer_large_v2`), UTMOS naturalness and
WavLM speaker similarity to the reference clip. "Failures" are sentences with WER > 0.5.

| model | voices | WER (Whisper) | WER (Conformer) | failure rate | UTMOS | speaker sim. |
|---|---|---|---|---|---|---|
| **this model** | 8 named prompts | 0.054 | 0.036 | 0.0 % | 3.46 | 0.958 |
| | 5 unseen voices | 0.039 | 0.033 | 0.5 % | 3.48 | 0.949 |
| [pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque) (6 layers × 1024) | 8 named prompts | 0.049 | 0.040 | 1.3 % | 3.48 | 0.960 |
| | 5 unseen voices | 0.029 | 0.030 | 0.5 % | 3.50 | 0.957 |

With the end-of-speech threshold at Pocket TTS's default (−4) the same model scores WER 0.11 and a
7-8 % failure rate, all from sentences cut short; use the provided config.

## Training

1. **Teacher**: [itzune/pocket-tts-basque-24l](https://huggingface.co/itzune/pocket-tts-basque-24l)
   (24 layers, fine-tuned from Kyutai's English 24-layer model on BSSD), further fine-tuned 10 k steps with a
   64-dimensional interface between backbone and heads (`z_dim: 64`), initialised from the top-64 principal
   components of the backbone output and folded into the heads. This narrow interface is what lets a small
   student copy the teacher.
2. **Student**: Kyutai's depth distillation with classifier-free guidance baked in, into an 8-layer × 512
   backbone (feed-forward 512, 8 heads), seeded from the teacher's 4 bottom and 4 top layers cut to width 512
   along the principal components of the residual stream. 100 k steps, batch 64, learning rate 4e-4 with
   cosine decay; the teacher's sampling and end-of-speech heads are kept frozen. EMA weights released.
3. **Inference**: the end-of-speech threshold is raised to 0 (from −4): the frozen teacher end-of-speech head
   fired early on the student's outputs, cutting ~7 % of sentences short; at 0 those sentences are read in
   full with no runaway generation.

Data: BSSD train split (about 350 k utterances, 450 h of synthetic Basque speech, 2,400+ voices).
Training code, configs and evaluation: https://github.com/itzune/pocket-tts-basque
(`TRAINING_PLAN_SMALL.md` has the full run log).

## Limitations

- Basque only; code-switched Spanish/French/English words are read with Basque phonetics.
- Slightly lower speaker similarity than the base model, mostly on voices it has not seen.
- Trained on synthetic speech, so artefacts of the dataset's teacher system may be inherited.

## License and attribution

Model weights: CC-BY-4.0 (as the upstream Pocket TTS weights). Please cite Kyutai's Pocket TTS and its small-model
recipe, the HiTZ-Aholab studio corpus for the Maider and Antton voices, and the BSSD dataset
([itzune/bssd](https://huggingface.co/datasets/itzune/bssd), CC-BY-SA-4.0).
