---
language:
- eu
license: cc-by-4.0
library_name: onnx
pipeline_tag: text-to-speech
tags:
- text-to-speech
- onnx
- onnxruntime
- pocket-tts
- kyutai
- basque
- euskara
- voice-cloning
- on-device
base_model: itzune/pocket-tts-basque
datasets:
- itzune/bssd
---

# Pocket TTS Basque — ONNX (euskarazko ahots-sintesia, PyTorch gabe)

ONNX export of [itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque), the 6-layer
Basque (euskara) Pocket TTS model, for **streaming text-to-speech on a CPU with onnxruntime only — no
PyTorch**. Same weights, same five bundled voices, same zero-shot voice cloning; fp32 and int8 graphs.

The bundle follows the format of [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx)
(the community ONNX export of Kyutai's Pocket TTS), whose reference runtime is included here as
`pocket_tts_onnx.py`.

## Files

| file | what it is |
|---|---|
| `flow_lm_main[_int8].onnx` | transformer backbone (text + voice conditioning, one latent frame per step) |
| `flow_lm_flow[_int8].onnx` | flow-matching head (latent sampling) |
| `mimi_encoder[_int8].onnx` | Mimi encoder, for cloning a voice from a clip |
| `mimi_decoder[_int8].onnx` | Mimi decoder, streaming (latent frames → 24 kHz audio) |
| `text_conditioner[_int8].onnx` | token embeddings |
| `bundle.json`, `bos_before_voice.npy` | state layout and metadata for the runtime; learned BOS vector placed before the voice |
| `tokenizer.model`, `vocab.json`, `token_scores.json` | Basque SentencePiece tokenizer (4000 tokens), also as JSON |
| `voices/*.safetensors` | the five bundled voices (Maider, Antton, Nahia, Ane, Aitor; see `voices/voices.json`) |
| `pocket_tts_onnx.py` | reference streaming runtime by KevinAHM (Apache-2.0) |

The int8 graphs use dynamic MatMul quantization. The reference runtime always runs the Mimi encoder and the
text conditioner in fp32 and uses int8 for the other three, so `precision="int8"` needs about 160 MB of
files and `precision="fp32"` about 440 MB; the `mimi_encoder_int8` and `text_conditioner_int8` graphs are
there for other runtimes.

## Usage

```bash
pip install onnxruntime numpy sentencepiece safetensors huggingface_hub soundfile
```

```python
import sys
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

bundle = snapshot_download("itzune/pocket-tts-basque-onnx", allow_patterns=[
    "*int8.onnx", "mimi_encoder.onnx", "text_conditioner.onnx",  # the runtime runs these two in fp32
    "*.json", "*.npy", "*.model", "*.py", "voices/*"])
sys.path.insert(0, bundle)
from pocket_tts_onnx import PocketTTSOnnx

# temperature 0.3 is this model's default (the runtime's own default, 0.7, is for Kyutai's models)
tts = PocketTTSOnnx(models_dir=bundle, precision="int8", temperature=0.3)

voice = f"{bundle}/voices/Maider.safetensors"   # or a 5-10 s .wav of clean Basque speech to clone
audio = np.concatenate(list(tts.stream("Kaixo! Hau euskarazko ahots sintetiko bat da.", voice)))
sf.write("kaixo.wav", audio, tts.sample_rate)  # 24 kHz
```

`tts.stream()` yields audio as it is generated (the first chunk after two latent frames), so playback can
start long before the sentence is finished; `precision="fp32"` uses the full-precision graphs.

Notes:

- Write the text as it should be read: numbers, dates, abbreviations and symbols must be spelled out in
  words (e.g. `hogeita lau` rather than `24`). The dataset's normaliser
  ([`bssd.text_norm`](https://huggingface.co/datasets/itzune/bssd)) does this for standard Basque.
- **Not compatible with sherpa-onnx as is**: this model inserts a learned BOS vector before the voice
  (`bos_before_voice.npy`), which sherpa-onnx's Pocket TTS implementation does not apply.
- Only clone voices you have the right to use.

## Speed

`scripts/bench_onnx.py` (three sentences of 1-7 s, Maider voice, streaming, generation only, after a
warm-up) on an 8-core x86 laptop CPU, with the PyTorch model on the same machine for comparison:

| runtime | threads | first audio | speed (× real time) |
|---|---|---|---|
| **ONNX int8** | 2 | 0.07 s | 4.4× |
| ONNX fp32 | 2 | 0.09 s | 3.5× |
| PyTorch fp32 | 2 | 0.07 s | 6-7× |
| PyTorch int8 (`--quantize`) | 4 | 0.03-0.05 s | 8× |

The ONNX runtime is slower than PyTorch but still several times faster than real time, with first audio in
about 0.1 s, at a fraction of the install size (onnxruntime and friends: ~150 MB, against several GB for
PyTorch).

## Export

Exported with the community exporter [lomotron/pocket-tts-onnx-export](https://github.com/lomotron/pocket-tts-onnx-export)
(pinned commit, with a small fix for exporting from a config file) from the released `basque.yaml`; every
graph was verified against the PyTorch model within tolerance before quantization. Reproduce with
`scripts/export_onnx.sh` in https://github.com/itzune/pocket-tts-basque.

## Limitations

As for [itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque): Basque only
(code-switched words are read with Basque phonetics); a small share of sentences (about 1-2 %) comes out as
a continuation of the reference clip instead of the input text; trained on synthetic speech. int8
quantization was checked by ear and with ASR on the benchmark sentences, not with the full evaluation of
the PyTorch model.

## License and attribution

Model weights: CC-BY-4.0 (as [itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque)
and the upstream Pocket TTS weights). `pocket_tts_onnx.py`: Apache-2.0, from
[KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx). Please cite Kyutai's Pocket
TTS, the HiTZ-Aholab studio corpus for the Maider and Antton voices, and the BSSD dataset
([itzune/bssd](https://huggingface.co/datasets/itzune/bssd), CC-BY-SA-4.0).
