# Pocket TTS Basque

Open Basque (euskara) text-to-speech that runs in real time on a CPU, built on
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai and trained on the synthetic
[BSSD](https://huggingface.co/datasets/itzune/bssd) dataset.

**Demo page with audio samples and a comparison with Piper and OmniVoice:
https://itzune.eus/pocket-tts-basque/**

| | Hugging Face | What it is |
|---|---|---|
| **Pocket TTS Basque** | [itzune/pocket-tts-basque](https://huggingface.co/itzune/pocket-tts-basque) | 6-layer model (~100 M parameters) for real-time CPU use, five bundled voices, zero-shot cloning |
| ONNX export | [itzune/pocket-tts-basque-onnx](https://huggingface.co/itzune/pocket-tts-basque-onnx) | the same model for onnxruntime (fp32 + int8), streaming on CPU without PyTorch |
| Teacher | [itzune/pocket-tts-basque-24l](https://huggingface.co/itzune/pocket-tts-basque-24l) | 24-layer model it was distilled from |
| Dataset | [itzune/bssd](https://huggingface.co/datasets/itzune/bssd) | 470 h of synthetic Basque speech, CC BY-SA 4.0 |

```bash
# with uv installed, nothing else to set up
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque/basque.yaml \
  --voice hf://itzune/pocket-tts-basque/voices/Maider.safetensors \
  --text "Kaixo! Hau euskarazko ahots sintetiko bat da." --output kaixo.wav
```

## Results

Held-out BSSD test sentences (40 per voice), two independent Basque ASR systems
(`xezpeleta/whisper-large-v3-eu`, `HiTZ/stt_eu_conformer_transducer_large_v2`), UTMOS and WavLM speaker
similarity. Failures are sentences with WER > 0.5 (the model continuing the prompt instead of reading).

| model | voices | WER Whisper / Conformer | failure rate | UTMOS | speaker sim. |
|---|---|---|---|---|---|
| Pocket TTS Basque (6 L, 150 k) | 8 named prompts | 0.049 / 0.040 | 1.3 % | 3.48 | 0.960 |
| | 5 unseen voices | 0.029 / 0.029 | 0.5 % | 3.50 | 0.957 |
| teacher (24 L, 40 k) | 8 named prompts | 0.095 / 0.091 | 5.3 % | 3.59 | 0.958 |
| | 5 unseen voices | 0.073 / 0.069 | 3.5 % | 3.61 | 0.961 |

CPU speed of the 6-layer model (generation only, x86 server CPU): 3.5× real time in fp32 with 2 threads,
6.9× with int8 dynamic quantization (`--quantize`). Full tables: `reports/`.

## Recipe

1. **Data**: BSSD train split pulled from the Hub (`scripts/prepare_bssd.py`), word-aligned with
   `deepdml/wav2vec2-large-xls-r-300m-basque` (`scripts/align_bssd.sh`, `scripts/validate_alignment.py`),
   Basque SentencePiece tokenizer (4000 tokens, `scripts/train_tokenizer.py`).
2. **Teacher**: Kyutai's 24-layer English model fine-tuned for Basque with the official new-language recipe
   (`configs/basque_24l_finetune.yaml`, text embedding reset). Checkpoint 40 k chosen by the comparison in
   `reports/teacher_compare.md` (later checkpoints drifted under the constant learning rate).
3. **Student**: official depth distillation into the 6-layer backbone with CFG baked in
   (`configs/basque_6l_distill.yaml`, 150 k steps); 150 k EMA weights released
   (`reports/student_compare.md`).
4. **Evaluation and release**: `scripts/eval_basque.py` (both ASRs via `scripts/score_conformer.py`, UTMOS,
   speaker similarity, failure rate), `scripts/compare_checkpoints.sh`, `scripts/bench_cpu.py`,
   `scripts/export_teacher.py`, `scripts/export_voices.py`, `scripts/upload_hf.py`; release sources in
   `release/`.
5. **ONNX export**: `scripts/export_onnx.sh` exports the released model with the community
   [pocket-tts-onnx-export](https://github.com/lomotron/pocket-tts-onnx-export) exporter (pinned, with
   `scripts/onnx_exporter.patch`), verifies every graph against PyTorch, quantizes to int8 and assembles
   the release folder; `scripts/bench_onnx.py` measures it without PyTorch.

The training code is the official `kyutai-labs/pocket-tts` repository (v3.3.0, `training/`), checked out
next to this one; the configs here extend its `finetune_language.yaml` and `depth_distill.yaml`. Run helpers
for a small single-GPU box (resume after crashes, evaluation loop, W&B sidecar, memory guard) are in
`scripts/` as well. `RESEARCH.md` is the initial survey (why Pocket TTS, why OmniVoice as the data teacher)
and `TRAINING_PLAN.md` the plan plus the complete run log with every number and incident; `TRAINING_PLAN_SMALL.md` is the first attempt at the 18M model with Kyutai's shrinking recipe and `TRAINING_PLAN_V2.md` the plan to redo it properly from the data up.

## Demo page

`docs/` is the static site served by GitHub Pages. `scripts/make_demo_audio.py` generates the clips with
the released model on CPU, `scripts/make_demo_compare.sh` the Piper and OmniVoice clips for the same
sentences and voices, and `scripts/build_demo_page.py` renders `docs/index.html`.

## Licences

Code in this repository: MIT. Model weights: CC BY 4.0 (as upstream Pocket TTS). Dataset: CC BY-SA 4.0.
The Maider and Antton voices come from the
[HiTZ-Aholab studio corpus](https://zenodo.org/records/17952596) (CC BY 4.0).
