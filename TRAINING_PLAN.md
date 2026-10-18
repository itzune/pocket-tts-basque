# Training plan: Basque Pocket TTS from BSSD

Goal: a Pocket TTS model for Basque that runs faster than real time on 2 CPU cores, with the eight
BSSD named voices as presets (Maider and Antton as defaults) and zero-shot cloning from a clip.
Base: Kyutai Pocket TTS (MIT code, CC-BY-4.0 weights), official `training/` code (v3.3+).
Data: [itzune/bssd](https://huggingface.co/datasets/itzune/bssd) (CC-BY-SA-4.0): train 448 h /
dev 10.6 h / test 12.1 h, 24 kHz FLAC, verbalised transcripts (`text_norm`).

## Recipe (official new-language path)

```
english_2026-04_24l (EN teacher)
   │ finetune_language.yaml: reset_text_embedding, Basque SentencePiece, lr 2e-4, batch 64 eff.
   ▼
Basque 24L teacher ──depth_distill.yaml (CFG baked in)──▶ Basque 6L student (~100M, the release)
```

A second teacher warm-started from `spanish_24l` runs as a comparison if GPU time allows; keep the
one with lower dev WER at 15k steps.

## Steps

| # | step | script / config | output |
|---|---|---|---|
| 1 | pull BSSD, keep FLAC, write manifests (`path, duration, transcript, voice_id`) | `scripts/prepare_bssd.py` | `/root/data/bssd/{train,dev,test}.jsonl` |
| 2 | forced alignment with a Basque wav2vec2-CTC; validate on 300 utterances first | `scripts/align_bssd.sh`, `scripts/validate_alignment.py` | `*_aligned.jsonl` |
| 3 | Basque SentencePiece tokenizer on train transcripts (default vocab = n_bins) | pocket-tts `training/scripts/train_tokenizer.py` | `tokenizer/basque.model` |
| 4 | teacher fine-tune | `configs/basque_24l_finetune.yaml` | `runs/basque_24l/model.safetensors` (EMA) |
| 5 | distill to 6 layers | `configs/basque_6l_distill.yaml` | `runs/basque_6l/model.safetensors` |
| 6 | evaluate on BSSD test (named voices and held-out voices): WER with `xezpeleta/whisper-large-v3-eu` and `HiTZ/stt_eu_conformer_transducer_large_v2`, WavLM speaker similarity, UTMOS; CPU RTF at 2 threads | `scripts/eval_basque.py`, `scripts/bench_cpu.py` | `reports/` |
| 7 | export the 8 named-voice states, package yaml + safetensors + tokenizer, model card, upload | `scripts/package_model.py` | HF `itzune/pocket-tts-eu` |

## Compute and disk (single 46 GB GPU, 4 CPU cores, 15 GB RAM)

- Data: ~46 GB FLAC + ~3 GB Mimi latents (precompute, once).
- Teacher: batch 32 × grad_accum 2 (effective 64). Expect WER to plateau by ~15k steps, acoustic
  quality to keep improving to ~60k; stop early by dev WER/UTMOS. Roughly 1 step/s →
  ~17 h for 60k steps.
- Distillation: ~2 steps/s; WER parity by ~50k, UTMOS to ~150k; ~1 day.
- Alignment: ~10 h of audio per minute on the GPU → ~1 h for 450 h.

## Evaluation protocol

- Test prompts: for each named voice its own prompt; for held-out voices the bank/real prompt clip;
  texts: BSSD test sentences (unseen). Baselines: OmniVoice (teacher ceiling), Piper Basque
  community voices, HiTZ VITS Antton/Maider; also the Spanish Pocket TTS on Basque text (old baseline).
- Report WER (both ASRs), speaker sim, UTMOS, RTF and first-audio latency on CPU, model size.

## Release

`model.safetensors` + `tokenizer.model` + `basque.yaml` (same schema as the official configs) +
`voices/<Name>.safetensors` states; usable as
`uvx pocket-tts generate --config hf://itzune/pocket-tts-eu/basque.yaml --voice Maider ...`.
Model license CC-BY-4.0 (as upstream); card documents BSSD provenance (teacher CC-BY-NC).

## Log

- 2026-10-17 12:30 UTC: VM2 set up (Pocket TTS 3.3.0, torch 2.14 cu130). Alignment validated on 300
  dev utterances with `deepdml/wav2vec2-large-xls-r-300m-basque` and `cahya/wav2vec2-large-xlsr-basque`:
  300/300 aligned, word counts match, monotonic, word duration p50 0.30 s / p95 0.62 s / max 1.1 s,
  first-word start p50 0.16 s, tail slack p50 0.20 s; both ~9 s per hour of audio on the GPU. Using
  the xls-r-300m model. `scripts/train_pipeline.sh` chains data → alignment → tokenizer → teacher.
- 2026-10-17 13:57 UTC: teacher fine-tune started (latent precompute 20 min; ~3 steps/s at batch 32 x 2,
  19-22 GB VRAM). Step-5k samples already intelligible. Eval @10k (20 sentences/voice, Basque Whisper):
  named WER 0.162 / UTMOS 3.62 / sim 0.958; held-out 0.155 / 3.60 / 0.963. @20k: named 0.136 / 3.63;
  held-out 0.181 / 3.62 (per-voice variance is high at n=20; deep male voices Jon and bank0070 converge
  slowest). 15:56 UTC: trainer hung after the step-20k checkpoint ("Shared memory manager connection
  has timed out" in the loader subprocesses) while the step-20k evaluation ran concurrently on the
  4-core VM; killed at 16:13, auto-resumed from checkpoint_00020000 (lost ~17 min). Mitigation: the
  eval loop now runs with nice 19, 1 CPU thread and 16 sentences/voice; watcher flags stalls > 15 min.
- 2026-10-17 17:41 UTC: eval @30k (16 sentences/voice): named WER 0.099 / CER 0.064 / UTMOS 3.60 /
  sim 0.958; held-out WER 0.090 / UTMOS 3.61 / sim 0.961. Per-voice WER 10k→20k→30k: Maider 0.14→0.01→0.00,
  Aitor 0.04→0.04→0.00, bank0070 0.39→0.09→0.04, Jon 0.62→0.50→0.36 (still the worst), euf_05679
  0.06→0.41→0.16 (noisy at n=16). UTMOS flat at ~3.6 since 10k; valid flow_loss still improving
  (-0.057 @40k). Decision at the 40k checkpoint: continue to 80k (ETA ~22:20 UTC at ~2.8 steps/s).
  Housekeeping: the trainer keeps only the last 3 checkpoints, so 10k/20k/30k are gone; the eval loop
  now copies every evaluated checkpoint to `<run_dir>/keep/` (40k saved by hand). W&B sidecar: evals and
  samples arrive after the training stream has moved on and were silently dropped ("step less than
  current step"); both now log on their own axes (`eval/step`, `samples/step`), dropped points re-logged.
- 2026-10-17 18:37 UTC: eval @40k: named WER 0.093 / UTMOS 3.63 / sim 0.959; held-out WER 0.048 /
  UTMOS 3.67 / sim 0.961. Per-voice 30k→40k: Mikel 0.12→0.04, Antton 0.09→0.01, Nahia 0.08→0.01,
  euf_05679 0.16→0.00, Jon 0.36→0.33, Unai 0.06→0.25 (single bad sentence at n=16). Continuing to 80k.
- 2026-10-17 19:35 UTC: eval @50k: named WER 0.115 / UTMOS 3.59; held-out 0.121 / 3.67. The mean WER is
  dominated by catastrophic outputs, not by pronunciation: median WER is 0.0 at every step and the mean
  over non-failed sentences is 0.02-0.04. Failures (WER > 0.5) per eval: 32/260 @10k, 27/260 @20k,
  14/208 @30k, 9/208 @40k, 18/208 @50k (sampling noise is a few units). Two failure modes: (a) prompt
  continuation: the model speaks the tail of the voice prompt's own text instead of the input (Ane:
  "oso gustura egoten naiz", 1.6 s outputs); (b) hallucinated unrelated sentence on very short inputs
  ("Bi azoka, hilabete honetan." fails for Jon/Mikel/Ane/bank0083). Track failure rate, not only WER.
  19:29 UTC: trainer OOM-killed (RAM 15 GB + 4 GB swap exhausted with the eval running; anon RSS 6.2 GB).
  Orphaned loader children kept the resume script's ps|grep check "alive" (self-match race) → replaced by
  pgrep; added an 8 GB swap file; relaunched from checkpoint_00050000 (~1,700 steps lost).
- 2026-10-17 21:02 UTC: eval @60k (no OOM this time, swap 12 GB): named WER 0.037 / UTMOS 3.61 / sim 0.961;
  held-out 0.094 / 3.62 / 0.956. Failures 8/208 (lowest so far); Jon 0.28→0.05, Ane 0.24→0.02,
  bank0083 0.23→0.00; remaining failures on Unai, bank0027, bank0070, euf_05679. Valid flow_loss
  plateauing around -0.055 since 45k. ETA 80k ~22:35 UTC; disk 94 % (13 GB free).
- 2026-10-17 21:48 UTC: eval @70k: named WER 0.114 / UTMOS 3.63; held-out 0.213 / 3.55; failures 22/208
  (Jon 7, bank0070 6, euf_05679 4). Confirmed the hallucination mode: the model re-speaks the *voice
  prompt's own text* (Jon's prompt: "...lore eta fruituak ditu, zuhaitz bat irudikatuz"; Mikel's:
  "...historia testuinguruetan aztertzen dira, bestetik"), i.e. prompt continuation instead of reading
  the input. 21:41 UTC: second loader crash ("Shared memory manager connection has timed out") right
  after the 70k checkpoint while the eval ran (swap 10/11 GB used); main process hung in wait. Killed at
  21:53 and relaunched from checkpoint_00070000 (no steps lost). Evaluations at 10k intervals already use the EMA weights
  stored in the checkpoint (load_model does that); the swings are sampling noise at 16 sentences/voice,
  so the final pick needs a larger eval (40+/voice, both ASRs, failure rate).
- 2026-10-17 22:52 UTC: teacher run finished (80,000 steps, ~9 h wall incl. two restarts; peak 19.9 GB VRAM).
  Eval @80k: named WER 0.118 / UTMOS 3.58; held-out 0.191 / 3.55; failures 23/208. Failure counts per
  10k eval (n=208 from 30k): 32, 27, 14, 9, 18, 8, 22, 23 — best at 40k/60k; 70k/80k worse, which at
  16 sentences/voice may be noise or a real drift under the constant 2e-4 LR (no decay in the official
  fine-tune recipe). Kept checkpoints 40k-80k (`keep/`), run marked `.finished`, W&B run closed.
- 2026-10-18 06:30 UTC: final comparison started (`scripts/compare_checkpoints.sh`): 40 sentences/voice,
  fixed sampling seed, Basque Whisper + HiTZ Conformer (`scripts/score_conformer.py`, NeMo venv),
  failure rate / median / non-failed mean per checkpoint → `eval_final/compare.md`
  (copied to `reports/teacher_compare.md`). Per-step summaries archived in `reports/teacher_evals/`.
- 2026-10-18 07:40 UTC: comparison (40 sentences/voice, seed 0; Whisper-eu / Conformer WER, failure rate):
  40k named 0.095/0.091, 5.3 % — held-out 0.073/0.069, 3.5 %; 50k 0.152/0.145, 9.7 % — 0.128/0.115, 9.0 %;
  60k 0.093/0.087, 4.1 % — 0.142/0.143, 9.0 %; 70k 0.155/0.148, 10.3 % — 0.215/0.211, 15 %. UTMOS 3.57-3.61
  throughout. Both ASRs agree within 0.01. The run drifts after 60k under the constant LR; 40k generalises
  best to unseen voices. **Decision (user, matches their listening impression): teacher = checkpoint 40k**
  (EMA weights; `distill_teacher_use_ema` is on by default). Student: `configs/basque_6l_distill.yaml`.
- 2026-10-18 07:45 UTC: student run started (`configs/basque_6l_distill.yaml`, teacher = keep/checkpoint_00040000.pt,
  EMA): ~4.1 steps/s at batch 64, 8.6 GB VRAM → 150k steps ≈ 10 h (ETA ~18:00 UTC). Eval loop every 25k
  (`configs/basque_6l_infer.yaml`), W&B run `basque_6l`. Trainer processes shielded from the OOM killer
  (`choom -1000`); side jobs run under `systemd-run --scope -p MemoryMax=...` with `choom 1000`.
- 2026-10-18 07:55 UTC: **teacher released**: https://huggingface.co/itzune/pocket-tts-basque-24l (public;
  `model.safetensors` 1.34 GB = 40k EMA flow_lm + Mimi, `tokenizer.model`, `basque_24l.yaml`, README).
  Export via `scripts/export_teacher.py` (mmap, no model build); verified by loading
  `hf://itzune/pocket-tts-basque-24l/basque_24l.yaml` from the Hub and synthesising. Release files kept in
  `release/basque_24l/`. VM cleanup: kept checkpoints 40k + 60k only; comparison wavs deleted (22 GB free).
- 2026-10-18 09:44 UTC: student eval @25k (16 sentences/voice): named WER 0.093 / fail 2.3 % / WER-ok 0.065 /
  UTMOS 3.43 / sim 0.960; held-out 0.110 / 5.0 % / 0.068 / 3.42 / 0.953; GPU RTF 0.42 (teacher 0.9). Failure
  rate already at teacher level; non-failed WER and UTMOS still below the teacher (0.04 / 3.6), as upstream
  predicts for < 50k. Memory: swap was 11/11 GB after the eval → added /swapfile2 (8 GB, total 19 GB).
  Disk: deleted the 60k teacher checkpoint (13 GB free; the 21.5 GB dataset-phase FLAC folders remain).
- 2026-10-18 11:32 UTC: student eval @50k: named WER 0.108 / fail 7.8 % (Jon alone: 6 of 16) / WER-ok 0.042 /
  UTMOS 3.44 / sim 0.960; held-out WER 0.035 / fail 1.3 % / WER-ok 0.025 / UTMOS 3.47 / sim 0.954. Non-failed
  WER now at teacher level; UTMOS still ~0.15 below the teacher. 12:05 UTC: found the trainer's main process
  at 16 GB RSS+swap (slow host-memory leak, ~0.25 GB/1k steps; the teacher run showed the same growth before
  its OOM kill) plus 6 orphaned loader processes from yesterday's teacher runs holding 4.5 GB swap (killed).
  Added `scripts/mem_guard.sh` (restart from the next checkpoint when the trainer exceeds 11 GB; also
  launched by resume_training.sh).
- 2026-10-18 13:20 UTC: student eval @75k: named WER 0.083 / fail 3.9 % (Jon 3, Antton 1, Unai 1) / WER-ok 0.042 /
  UTMOS 3.47 / sim 0.958; held-out WER 0.034 / fail 0 % / UTMOS 3.49 / sim 0.956. Intelligibility at or above
  the teacher; UTMOS climbing slowly (3.42 → 3.47 → 3.49 on held-out) but still ~0.1 below the teacher's 3.6.
  mem_guard restarted the trainer once (60k); swap back to 7 GB at 77k, next restart expected ~90k.
- 2026-10-18 15:00 UTC: student eval @100k: named WER 0.068 / fail 3.1 % (Antton 2, Jon 2) / WER-ok 0.043 /
  UTMOS 3.51 / sim 0.958; held-out WER 0.032 / fail 0 % / UTMOS 3.53 / sim 0.956. UTMOS trend on held-out:
  3.42 → 3.47 → 3.49 → 3.53 (teacher 3.61). Second mem_guard restart at the 100k checkpoint (14:58 UTC).
- 2026-10-18 16:57 UTC: student eval @125k: named WER 0.035 / UTMOS 3.53 / sim 0.960; held-out WER 0.035 /
  UTMOS 3.56 / sim 0.956. Third mem_guard restart at 135k. 18:10 UTC: loader crash ("Shared memory manager
  connection has timed out") right after the 145k checkpoint, main process hung; killed and relaunched at
  18:13 from checkpoint_00145000 (no eval was running this time, so the loader's shared-memory manager
  itself is fragile under memory pressure, not only CPU contention).
- 2026-10-18 18:35 UTC: student run finished (150,000 steps, ~10.8 h wall incl. 3 guard restarts and 1 crash;
  peak 20.8 GB VRAM). Next: quick eval @150k, then `STEPS="100000 125000 150000" RUN=/root/runs/basque_6l
  CFG=configs/basque_6l_infer.yaml OUT=/root/runs/basque_6l/eval_final scripts/compare_checkpoints.sh`
  (40 sentences/voice, both ASRs), CPU benchmark, release.
- 2026-10-18 18:40 UTC: student eval @150k (16/voice): named WER 0.030 / fail 1.6 % / WER-ok 0.019 / UTMOS 3.55 /
  sim 0.961; held-out WER 0.020 / fail 1.3 % / WER-ok 0.013 / UTMOS 3.55 / sim 0.957. Jon 0.04 with 0 failures.
  Best point of the run on every metric; UTMOS within 0.05 of the teacher. Final comparison (100k/125k/150k,
  40/voice, both ASRs) and the CPU benchmark running next; voices exported for the 150k model.
- 2026-10-18 19:04 UTC: student comparison (40/voice, both ASRs): 150k wins — named WER 0.049 / 0.040 (Whisper /
  Conformer), fail 1.3 % / 0.9 %, UTMOS 3.48, sim 0.960; held-out 0.029 / 0.029, fail 0.5 % / 0 %, UTMOS 3.50,
  sim 0.957 (125k: 0.054 / 0.036; 100k: 0.080 / 0.050). Student beats the teacher on WER and failures, ~0.1 lower
  UTMOS. CPU benchmark (training VM, 4 CPU cores): fp32 3.5x real time at 2 threads; int8 dynamic quant 6.9x. Named voice
  states exported for the 150k model (8 x 2.5-4.2 MB). Release staged in /root/export/basque_6l (model 438 MB,
  tokenizer, basque.yaml, voices/, README) and smoke-tested with the Maider and Jon states. Upload to
  itzune/pocket-tts-basque pending user confirmation (public repo).
- 2026-10-18 19:20 UTC: **student released**: https://huggingface.co/itzune/pocket-tts-basque (public): 150k EMA
  weights (438 MB incl. Mimi), tokenizer, `basque.yaml`, five voice states (Maider, Antton, Nahia, Ane, Aitor;
  user decision: only these "stable" voices are published/mentioned for now — Mikel, Jon, Unai states are
  staged in /root/export/basque_6l/voices and `release/` but not uploaded), README with the comparison and CPU
  benchmark. Verified by loading config + Maider state from the Hub. Teacher card now links the student.
  VM2 cleanup: student keep/ reduced to 125k + 150k; 13 GB free.
- 2026-10-18 20:05 UTC: code published at https://github.com/itzune/pocket-tts-basque (MIT; fresh history, docs
  scrubbed of IPs / hardware names / local LLM server mentions; `.env`, `logs/` and weights ignored). Demo page
  (GitHub Pages from `docs/`, https://itzune.eus/pocket-tts-basque/): 5 voices x 4 sentences generated on 2 CPU
  threads (3.4-3.7x real time), plus the same 3 sentences for Maider and Antton with Piper (itzune eu-*-medium
  ONNX voices) and OmniVoice (clone from the same prompt clips). ASR check of every demo clip: all content
  sentences WER 0 for our model and OmniVoice; Piper 0-0.25.
