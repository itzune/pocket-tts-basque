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
- voice-cloning
- on-device
base_model: itzune/pocket-tts-basque-24l
datasets:
- itzune/bssd
---

# Pocket TTS Basque (euskarazko ahots-sintesia, denbora errealean CPUan)

A lightweight Basque (euskara) text-to-speech model that runs in real time on a laptop CPU, based on
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai (6-layer backbone, about 100 M
parameters). It was distilled from the 24-layer Basque teacher
[itzune/pocket-tts-basque-24l](https://huggingface.co/itzune/pocket-tts-basque-24l), itself fine-tuned on
the [BSSD](https://huggingface.co/datasets/itzune/bssd) dataset. It clones a voice from a short reference
clip and ships with five ready-made Basque voices.

## Usage

With [`uv`](https://docs.astral.sh/uv/) installed, no setup is needed:

```bash
# one of the bundled voices
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque/basque.yaml \
  --voice hf://itzune/pocket-tts-basque/voices/Maider.safetensors \
  --text "Kaixo! Hau euskarazko ahots sintetiko bat da." --output kaixo.wav

# your own voice: 5-10 s of clean Basque speech
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque/basque.yaml \
  --voice my_voice.wav --text "Kaixo!" --output kaixo.wav
```

Python:

```python
from pocket_tts import TTSModel
import soundfile as sf

model = TTSModel.load_model(config="hf://itzune/pocket-tts-basque/basque.yaml")
voice = model.get_state_for_audio_prompt("hf://itzune/pocket-tts-basque/voices/Antton.safetensors")
audio = model.generate_audio(voice, "Egun on, zer moduz zaude gaur?")
sf.write("egun_on.wav", audio.squeeze().numpy(), model.sample_rate)  # 24 kHz
```

Bundled voices (`voices/`): **Maider** (female) and **Antton** (male), the two professional speakers of the
[HiTZ-Aholab studio corpus](https://zenodo.org/records/17952596) (CC BY 4.0); and three synthetic voices,
**Nahia**, **Ane** (female) and **Aitor** (male). Voice state files are specific to this model.

Notes:

- Write the text as it should be read: numbers, dates, abbreviations and symbols must be spelled out in
  words (e.g. `hogeita lau` rather than `24`). The dataset's normaliser
  ([`bssd.text_norm`](https://huggingface.co/datasets/itzune/bssd)) does this for standard Basque.
- Only clone voices you have the right to use.

## Speed

Measured with the bundled benchmark (`scripts/bench_cpu.py`: three sentences of 2-8 s, Maider voice) on a
x86 server CPU (4 threads available), generation only (no model load), as a multiple of real time (higher is faster):

| threads | fp32 | int8 (`--quantize`) |
|---|---|---|
| 2 | 3.5x | 6.9x |
| 4 | 3.4x | 6.7x |

The model is ~100 M parameters (438 MB fp32 weights including the Mimi codec). Streaming synthesis makes
the first audio available well before a sentence is finished.

## Evaluation

Held-out BSSD test sentences, 40 per voice, scored with two independent Basque ASR systems
(`xezpeleta/whisper-large-v3-eu` and `HiTZ/stt_eu_conformer_transducer_large_v2`), UTMOS naturalness and
WavLM speaker similarity to the reference clip. "Failures" are sentences with WER > 0.5 (the model
continuing the reference clip instead of reading the text).

| model | voices | WER (Whisper) | WER (Conformer) | failure rate | WER excl. failures | UTMOS | speaker sim. |
|---|---|---|---|---|---|---|---|
| **this model** (6 layers, 150 k) | 8 named prompts | 0.049 | 0.040 | 1.3 % | 0.040 | 3.48 | 0.960 |
| | 5 unseen voices | 0.029 | 0.029 | 0.5 % | 0.026 | 3.50 | 0.957 |
| teacher (24 layers) | 8 named prompts | 0.095 | 0.091 | 5.3 % | 0.042 | 3.59 | 0.958 |
| | 5 unseen voices | 0.073 | 0.069 | 3.5 % | 0.033 | 3.61 | 0.961 |

The distilled model is more robust than its teacher (fewer prompt-continuation failures, lower WER) at a
slightly lower naturalness score, and about 2x faster on GPU and far faster on CPU.

## Training

- Teacher: [itzune/pocket-tts-basque-24l](https://huggingface.co/itzune/pocket-tts-basque-24l)
  (24 layers, fine-tuned from Kyutai's English 24-layer model on BSSD).
- Student: Kyutai's `depth_distill` recipe. A 6-layer backbone learns the teacher's activations with the
  classifier-free guidance baked in (`distill_cfg_coef 2.0`); the flow head and all non-backbone weights
  are copied from the teacher. 150 k steps, batch 64, learning rate 4e-4 with cosine decay, EMA weights
  released.
- Data: BSSD train split (about 350 k utterances, 450 h of synthetic Basque speech, 2,400+ voices).
- Tokenizer: SentencePiece, 4000 tokens, trained on the BSSD transcripts (shared with the teacher).

Training code, configs and evaluation scripts: https://github.com/xezpeleta/pocket-tts-eu

## Limitations

- Basque only; code-switched Spanish/French/English words are read with Basque phonetics.
- A small share of sentences (about 1-2 % in our tests, more with some reference clips) comes out as a
  continuation of the reference clip instead of the input text; always check generated audio.
- Trained on synthetic speech, so artefacts of the dataset's teacher system may be inherited. See the
  BSSD dataset card for the provenance of voices and texts.

## License and attribution

Model weights: CC-BY-4.0 (as the upstream Pocket TTS weights). Please cite Kyutai's Pocket TTS, the
HiTZ-Aholab studio corpus for the Maider and Antton voices, and the BSSD dataset
([itzune/bssd](https://huggingface.co/datasets/itzune/bssd), CC-BY-SA-4.0).
