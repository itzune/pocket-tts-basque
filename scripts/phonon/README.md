# In-browser inference with phonon-tts (research notes, 2026-10-20)

[gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts) is a Rust re-implementation of the Pocket TTS
architecture with a WebAssembly build published on npm as `phonon-tts`. It runs the model in a web worker
on the CPU (SIMD, 3 threads on a cross-origin-isolated page) or on WebGPU, with no server. Kyutai's own
online demo is **not** in-browser: it posts to `pocket-tts-api.kyutai.org`.

## Result: the Basque model runs in the browser as is

Tested on a laptop with Chrome (headless, via Playwright) and the released `itzune/pocket-tts-basque`
files, prepared with `prepare_phonon_model.py`:

| | threads | 5.3 s sentence | 11.4 s, two sentences |
|---|---|---|---|
| f32 weights, CPU | 1 (page not isolated) | 1.8× real time, first audio 273 ms | |
| f32 weights, CPU | 3 (isolated page) | 3.8× real time, first audio 180 ms | 3.1× real time |

Both outputs were transcribed by `xezpeleta/whisper-large-v3-eu` with **WER 0**. Voice states exported by
`scripts/export_voices.py` load unchanged (`addVoice` / `voices:` in `ModelSpec`).

## What had to change

- `flow_lm.speaker_proj_weight` is removed from the weights: Pocket TTS *training* checkpoints project the
  32-dim Mimi latents, the engine expects a 512-dim speaker projection. It is only used to clone a voice
  from audio, which the browser build cannot do anyway (no Mimi encoder); precomputed KV-cache voice
  states do not need it.
- `tokenizer.json`: our SentencePiece model converted with xn-ptts's `scripts/convert-tokenizer.py`
  (round-trip checked by the script).
- `config.json`: the engine's own Pocket TTS 6-layer config (`ptts/tests/fixtures/config.json`); the
  24-layer teacher would need `num_layers: 24`.
- `lang: 'none'`: the engine's text normalisation covers en/fr/de/es/pt only, so numbers etc. must be
  written out, as with the Python package.

## Limits found

- The **native** `ptts` CLI/Python/servers (same engine, strict loader) reject the checkpoint: the Mimi
  encoder's 512→32 downsample does not match the engine's channel-wise encoder, and KV-cache voice files
  are only understood by the browser engine. For native use keep the Python package or Kyutai's own Rust
  binary (`pocket-tts/rust`, built on a fork of xn-ptts that adds these cases).
- `q8` GGUF weights (3× smaller download, faster) are produced by `ptts/examples/quantize.rs` (Rust
  toolchain needed); not built yet. The f32 file is 438 MB, cached by the browser after the first load.
- No voice cloning in the browser; only the shipped voice states.

## Files

- `prepare_phonon_model.py`: builds the folder (optionally `--native` for the strict loader).
- `config.json`: engine config for the 6-layer model.
- `test_page.html`: minimal page using the npm package from `./phonon-tts/` and the model from `./model/`;
  serve the folder with COOP/COEP headers to get threads.
