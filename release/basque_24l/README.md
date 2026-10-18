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
base_model: kyutai/pocket-tts
datasets:
- itzune/bssd
---

# Pocket TTS Basque — 24-layer model (euskarazko ahots-sintesia)

A Basque (euskara) text-to-speech model based on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
by Kyutai. This is the **24-layer "teacher" model** (~340 M parameters), obtained by fine-tuning Kyutai's
24-layer English model for Basque with the official new-language recipe, on the
[BSSD](https://huggingface.co/datasets/itzune/bssd) speech dataset. It supports zero-shot voice cloning
from a short reference clip.

For real-time use on CPU, see the lighter 6-layer model distilled from this one:
[itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque).

## Usage

With [`uv`](https://docs.astral.sh/uv/) installed, no setup is needed:

```bash
uvx pocket-tts generate \
  --config hf://itzune/pocket-tts-basque-24l/basque_24l.yaml \
  --voice my_voice.wav \
  --text "Kaixo! Hau euskarazko ahots sintetiko bat da." \
  --output kaixo.wav
```

Python:

```python
from pocket_tts import TTSModel
import soundfile as sf

model = TTSModel.load_model(config="hf://itzune/pocket-tts-basque-24l/basque_24l.yaml")
voice = model.get_state_for_audio_prompt("my_voice.wav")      # 5-10 s of clean Basque speech
audio = model.generate_audio(voice, "Kaixo! Hau euskarazko ahots sintetiko bat da.")
sf.write("kaixo.wav", audio.squeeze().numpy(), model.sample_rate)  # 24 kHz
```

Notes:

- Write the text as it should be read: numbers, dates, abbreviations and symbols must be spelled out
  in words (e.g. `hogeita lau` rather than `24`). The dataset's normaliser
  ([`bssd.text_norm`](https://huggingface.co/datasets/itzune/bssd)) does this for standard Basque.
- The reference clip defines the voice. Only clone voices you have the right to use.
- `--config` also accepts a local copy of `basque_24l.yaml`; the weights and tokenizer are fetched from
  this repository.

## Evaluation

Held-out BSSD test sentences, 40 sentences per voice, scored with two independent Basque ASR systems
(`xezpeleta/whisper-large-v3-eu` and `HiTZ/stt_eu_conformer_transducer_large_v2`), UTMOS naturalness and
WavLM speaker similarity to the reference clip.

| voices | WER (Whisper) | WER (Conformer) | failure rate | WER excl. failures | UTMOS | speaker sim. |
|---|---|---|---|---|---|---|
| 8 named prompts (dataset default voices) | 0.095 | 0.091 | 5.3 % | 0.042 | 3.59 | 0.958 |
| 5 unseen voices | 0.073 | 0.069 | 3.5 % | 0.033 | 3.61 | 0.961 |

The median sentence is read with no errors. "Failures" are sentences with WER > 0.5: in nearly all of
them the model continues the reference clip's own speech instead of reading the input text, a known
failure mode of prompt-conditioned models. It occurs more often with some reference clips than others
(from 0 % to about 20 % per voice in our tests); if a voice misbehaves, try a different or shorter clip.

## Training

- Base: Kyutai's 24-layer English Pocket TTS model (text embedding re-initialised for the new tokenizer).
- Tokenizer: SentencePiece, 4000 tokens, trained on the BSSD transcripts.
- Data: BSSD train split (about 350 k utterances, 450 h of synthetic Basque speech, 2,400+ voices,
  word-level alignments from `deepdml/wav2vec2-large-xls-r-300m-basque`).
- Recipe: official `finetune_language` configuration (learning rate 2e-4, effective batch 64,
  classifier-free-guidance dropout as upstream); the released weights are the EMA weights at step 40 k,
  chosen among the 40 k-80 k checkpoints by the evaluation above (later checkpoints had higher failure
  rates on unseen voices).

Training code, configs and evaluation scripts: https://github.com/xezpeleta/pocket-tts-eu

## Limitations

- Basque only; code-switched Spanish/French/English words are read with Basque phonetics.
- Prompt-continuation failures as described above; always check generated audio.
- Trained on synthetic speech: the voices in the dataset are themselves synthetic or re-synthesised,
  so the model may inherit artefacts of the dataset's teacher system. See the BSSD dataset card for the
  provenance of the voices and texts.

## License and attribution

Model weights: CC-BY-4.0 (as the upstream Pocket TTS weights). Please cite Kyutai's Pocket TTS and the
BSSD dataset. Dataset: [itzune/bssd](https://huggingface.co/datasets/itzune/bssd) (CC-BY-SA-4.0).
