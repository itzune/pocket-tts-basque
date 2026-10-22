# Plan 2: small Basque Pocket TTS models (18M and 7M)

Status: planned 2026-10-20, not started. Follows Kyutai's recipe from
["The tiny TTS was actually 10x too big"](https://kyutai.org/blog/2026-10-20-pocket-tts-small/)
(code: pocket-tts v4.0.0). Phase 1 (teacher + 6-layer student) is in `TRAINING_PLAN.md`.

## Goal

| model | backbone | params (incl. decoder) | CPU speed, 1 thread (Kyutai, English) | browser f32 download |
|---|---|---|---|---|
| current `itzune/pocket-tts-basque` | 6 L × 1024 | ~100 M | 2.8× (ours: 3.5× on 2 threads) | 438 MB |
| **target A: 18M** | 8 L × 512, z 64 | 18.2 M | 9.9× | ~70 MB |
| target B: 7M | 8 L × 224, GQA 2 kv heads, rank-64 text table | 6.9 M | 12.6× | ~28 MB |

Quality bar: match the current model on our own metrics (both ASRs, failure rate, UTMOS, speaker
similarity on the 40-sentence/voice comparison set). Kyutai's English numbers say 18M and 7M hold the
100M quality; we stop at the size where ours does not. The 1.9M model is out of scope (loses quality even
in English).

Starting assets: Basque teacher `basque_24l` step 40k (`/root/runs/basque_24l/keep/checkpoint_00040000.pt`,
EMA), BSSD train/dev/test manifests + alignments + latents, the phase-1 evaluation harness.

## Steps

### 0. Upgrade and smoke-test the training code (½ day)

- `~/pocket-tts` on the GPU VM from v3.3.0 to v4.0.0; re-run `training/tests` and a 200-step
  `basque_6l_distill.yaml` resume to confirm old checkpoints still load ("every option defaults to the
  current behaviour").
- Re-check the host-memory leak and the loader shared-memory crash on v4; keep `mem_guard.sh`.
- Read the final form of: `training/scripts/shrink_checkpoint.py` (`--config/--override/--cov`),
  `residual_cov.py`, `training/args.py` (`z_dim`, `student_init_from`, `head_distill*`, `freeze_head`,
  `refl`), `training/refl.py` (which CTC ASR it expects), `english_18m_2026-10.yaml` /
  `english_7m_2026-10.yaml` (released architectures).

### 1. Free win: Kyutai's distilled 2.6 M decoder (1 evening, no training)

The Mimi decoder maps latents to audio and is language-independent; our backbone emits the same 32-dim
latents. Build `configs/basque_6l_smalldec_infer.yaml` = our 6-layer config with the 18m config's
`mimi.outer_dim: 128`, `quantizer.output_dimension: 128`, `decoder_seanet`/decoder transformer dims, and
the decoder tensors taken from `kyutai/pocket-tts/languages/english_18m_2026-10/model.safetensors`
(`mimi.*`; v4 supports separate Mimi weights). Evaluate with `scripts/compare_checkpoints.sh` logic on the
same 40 sentences/voice: keep it only if UTMOS/WER/sim stay within noise of the current decoder. If it
holds, the released model gets a 4× faster decoder immediately (and the browser build shrinks by ~30 MB).

### 2. Narrow the teacher's interface to 64 dims (~3 h GPU)

Blog: project the teacher's 1024-dim backbone output onto its top 64 principal components, fold the
projection into the heads, fine-tune 30k steps; the teacher loses almost nothing (UTMOS 4.24 vs 4.27) and
becomes distillable by small students. Ablation: 64 ≈ 128 > 32; a full-width teacher yields the worst
student.

- `residual_cov.py` on the teacher run (32 batches of dev) → `residual_cov.pt`.
- `shrink_checkpoint.py` with `--config` = teacher config, `--override flow_lm.z_dim=64`, `--cov` →
  `teacher_z64_init.pt` (exact flags to confirm in v4; the blog's "fold into the heads" is what
  `shrink_width` + `z_dim` are for).
- `configs/basque_24l_z64.yaml`: `finetune.yaml`-style continuation from that init, same data, 30k steps,
  LR as the official fine-tune, `freeze_head: false`. Evaluate at 30k (with CFG 2, as a teacher is
  scored): must match the 40k teacher's numbers.

### 3. Distil the 18M student (~6-8 h GPU)

`configs/basque_18m_distill.yaml` from `depth_distill.yaml` + the `english_18m_2026-10.yaml`
architecture: `transformer: {num_layers: 8, d_model: 512, hidden_scale: 1, num_heads: 8}`,
`lookup_table: {dim: 512, n_bins: 4000, tokenizer: sentencepiece}`, `flow_lm.z_dim: 64`; teacher =
step-2 checkpoint (`distill_teacher_*`, `distill_cfg_coef: 2.0`), `student_init_from` = the teacher cut
by `shrink_checkpoint` (bottom+top layers, PCA width cut), batch 64, lr 4e-4 cosine, 100k steps (Kyutai's
stage length; our 6-layer run converged by ~125k at the same batch). Head and decoder stay the teacher's
at this point (the blog distils backbone first, heads frozen and shared).

Gate: our comparison set, failure rate ≤ current model's 1.3 %, WER-ok ≤ 0.045, UTMOS ≥ 3.45, sim ≥ 0.955.

### 4. Small sampling head (~2-4 h GPU)

`head_distill: true`, `head_distill_from` = step-3 checkpoint, new head `flow: {type: drifting, depth: 4,
dim: 192}`, `head_distill_loss: drifting`, `head_distill_samples: 64`, backbone frozen (the run trains only
the new head). This replaces the 9.8 M LSD head with 0.84 M at no quality cost in their ablation. If
drifting is unstable, fall back to `head_distill_loss: mse` (−0.05 UTMOS in their table).

### 5. Decoder (reuse step 1, or distil our own)

If Kyutai's 2.6 M decoder passed step 1, ship it. Otherwise distilling our own needs the decoder training
loop (adversarial + feature-matching + spectral + L1 on 100k of our model's own outputs); check whether v4
includes it before planning it.

### 6. Optional: reward fine-tuning (~1 h GPU)

`refl` with UTMOS + CTC log-likelihood, 3k steps, head and decoder frozen, distillation loss as anchor.
Needs a Basque character-level CTC model for the intelligibility term: `deepdml/wav2vec2-large-xls-r-300m-basque`
(our aligner) if `refl.py` accepts a custom CTC model; otherwise skip. Watch for pitch-range narrowing
(Kyutai saw −0.4 semitones on an internal model); compare F0 range on the comparison set before/after.

### 7. Optional: 7M (3 more cut-and-retrain stages, ~1 day GPU)

From the 18M backbone: width 320 → 224 → (2 kv heads, rank-64 text table, FFN 224), each stage distilled
to convergence (Kyutai: ~100k steps each; shorter stages lost 0.4 UTMOS when skipping steps). Then redo
step 4 for the smaller head. Only if 18M passed its gate with margin.

### 8. Evaluation and release

Same protocol as phase 1 (`scripts/eval_basque.py`, `score_conformer.py`, `compare_checkpoints.sh`,
`bench_cpu.py` at 1/2 threads fp32 and int8, phonon-tts browser test), plus a listening pass on the five
published voices. Release as `itzune/pocket-tts-basque-18m` (and `-7m`), export voice states per model
(`scripts/export_voices.py`), update the demo page with speed/size per model and the in-browser box.

## Compute and risk

- GPU: ~2 days for 18M (steps 0-4, 6), +1 day for 7M; VRAM is not a constraint (students are tiny); the
  teacher forward dominates distillation as in phase 1 (~4 steps/s then).
- Known hazards from phase 1: host-memory leak (guard restarts at 11 GB), loader shared-memory crash
  after checkpoints under memory pressure, evaluation OOM when run alongside training; swap 19 GB, disk
  ~13 GB free (clean before starting; the 60k teacher checkpoint is already gone).
- Quality risks specific to us: prompt-continuation failures (teacher 3.5-5 %, student 0.5-1.3 %) may
  rise in a smaller student; the failure rate is the gating metric. Voice similarity on real speakers
  (Maider, Antton) is the second thing to watch.
- Kyutai's numbers are English, 11 M utterances, 3 seeds; ours is 350 k utterances of synthetic speech and
  1 seed per stage. Expect noisier gates; use the 40-sentence/voice set, not the 16-sentence loop evals.

## Deliverables

`configs/basque_24l_z64.yaml`, `configs/basque_18m_distill.yaml`, `configs/basque_18m_head.yaml`,
(`configs/basque_7m_*`), `scripts/run_stage.sh` (one launcher per stage reusing resume/eval/sidecar/guard),
evaluation reports under `reports/small/`, release sources under `release/basque_18m/`, and the run log
appended to this file.

## Log

- 2026-10-20: plan written after reading the blog post and the v4.0.0 commit (`#360`: z_dim, attention
  options, low-rank text table, one-step heads, decoder dims; `shrink_checkpoint.py`, `residual_cov.py`,
  `refl.py`). GPU VM unreachable from the laptop at the time (VPN); nothing started.
- 2026-10-20 21:10 UTC: VM2 back. **Step 0 done**: `~/pocket-tts` → v4.0.0 in place (deps identical to
  v3.3.0). `uv sync` pruned the manually added packages (pyarrow, wandb, faster-whisper, librosa,
  transformers, nvidia cu12 libs) and left a hollow `nvidia-cudnn-cu13` (metadata without the .so):
  reinstalled; 18/18 unit tests pass; the released 6-layer model generates under v4; the teacher
  checkpoint opens.
- 2026-10-20 21:25 UTC: **Step 1 result (Kyutai 2.6 M decoder on our 6-layer model)**: quality neutral
  (40/voice, same seed) — named WER 0.042/0.043 vs 0.049/0.040 (Whisper/Conformer), held-out 0.033/0.034 vs
  0.029/0.030, failure rates identical, UTMOS +0.02/+0.01, sim −0.001. But **no CPU speed gain in the
  Python runtime**: laptop 1 thread 2.6× vs 3.2× real time (2 threads 3.8× vs 4.1×) — the 12-layer
  128-wide decoder transformer run frame by frame is overhead-bound in PyTorch; Kyutai's 14.8× decoder
  figure comes from their optimised runtime. Parked: revisit in the Rust/WASM engines. Config kept as
  `configs/basque_6l_smalldec_infer.yaml`, hybrid weights `/root/export/hybrid_smalldec/`.
- 2026-10-20 21:25 UTC: **Step 2 started**: residual covariance (49 norms, 190k tokens), seed checkpoint by
  `scripts/small/narrow_teacher_z64.py` (top-64 PCs keep 84 % of head-input variance; z_proj + folded
  cond_embed/out_eos; strict load OK), fine-tune `configs/basque_24l_z64.yaml` (lr 5e-5 const, batch 32×2,
  30k steps) running at ~3.3 steps/s → ETA ~00:00 UTC; evals at 10k/20k/30k with CFG 2
  (`configs/basque_24l_z64_infer.yaml`). Initial flow_loss 0.47 (teacher at 40k was ~0.10).
- 2026-10-21: **HOLD** (user): do not start the 18M distillation (step 3) yet. The z64 teacher fine-tune
  (step 2) runs to its 30k steps and is evaluated; nothing further is launched without a go-ahead.
- 2026-10-21 (overnight): the z64 fine-tune **crashed at step 12,500 (22:43 UTC) on a full disk** while
  writing optimizer state (24 L run: 2.4 GB checkpoint + 2.4 GB optimizer per save, 3 rotating + kept
  copies, plus the v4 package reinstall and the 8 GB second swap file). Freed 14 GB: partial optimizer
  file, superseded rotating checkpoints, the finished run's optimizer state, uv cache, the teacher export
  already on the Hub, the parked decoder experiment, and `/swapfile2` (swap back to 12 GB). Run marked
  `.finished`. Lesson: budget ~5 GB per 24 L checkpoint save and check free space before 24 L runs.
- 2026-10-21: z64 teacher eval @10k (16/voice, CFG 2) already matches the original teacher: named WER
  0.061 / UTMOS 3.65 / sim 0.960, held-out 0.028 / 3.67 / 0.961 (original 40k: 0.093 / 3.63 / 0.959 and
  0.048 / 3.67 / 0.961; z64 step 0 before fine-tuning: 0.232 / 3.60 and 0.171 / 3.53). **Teacher for step 3
  = z64 step 10k (EMA)**; not resumed to 30k (phase 1 showed drift with more constant-LR steps).
- 2026-10-21 16:16 UTC: user go-ahead → **step 3 started**: residual covariance on the z64 teacher,
  student seed by `shrink_checkpoint --config ... --cov` (teacher layers 0-3 + 20-23, PCA width cut to
  512, all tensors seeded), `configs/basque_18m_distill.yaml` (8 L × 512, FFN 512, z64, teacher LSD head
  frozen, lr 4e-4 cosine, batch 64, 100k steps). 5.1 steps/s, 5.7 GB VRAM → ETA ~21:45 UTC. Evals every
  25k with `configs/basque_18m_infer.yaml`. Initial distill_mse 0.51 → 0.44 at step 600.
- 2026-10-21 19:30 UTC: 18M @25k eval (EMA, 16/voice): WER 0.53 named and held-out, UTMOS 3.19/3.28,
  sim 0.947 — mumbled, words merged/dropped (6-layer student was at 0.09 at 25k). Diagnosis on 47.5k,
  12 sentences × 2 voices: EMA weights WER 0.379 (7/24 fails), raw weights 0.196 (4/24). Both factors:
  the width cut 1024→512 makes the 18M student slower to converge, and EMA 0.9999 still carries the cut
  init this early. No change for now: cosine decay to 100k should close the EMA gap; judge on the 75k and
  100k evals (and evaluate raw weights too if EMA still lags). mem_guard restarted the trainer once (42.5k).
- 2026-10-21 19:33 UTC: 18M @50k eval (EMA): named WER 0.235 / UTMOS 3.27 / sim 0.943, held-out 0.274 /
  3.22 / 0.940; median WER 0.5 → 0.1, failures 92 → 47 of 208. Improving fast, still far from the
  6-layer student (0.05 / 3.48). Next evals 75k (~20:55 UTC) and 100k (~22:20 UTC).
- 2026-10-21 21:01 UTC: 18M @75k eval (EMA): named WER 0.127 (non-failed 0.061) / UTMOS 3.50 / sim 0.951,
  held-out 0.135 (0.065) / 3.55 / 0.942; median WER 0.0, failures 21/208 (47 at 50k, 92 at 25k). UTMOS
  now at the 6-layer student's level (3.48/3.50); failures and speaker similarity still behind.
- 2026-10-21 21:30 UTC: checked Kyutai's repo for 18M training details: no args/config published (only
  the blog: 100k-step stages, batch 64; #382 lists sizes only). Fixes since v4.0.0 checked against our data:
  #396 (no weight decay on the text table: unseen rows collapse) — our 216 unseen tokens are only the
  4 control pieces and 212 byte-fallback pieces, every real word piece is seen, and the 18M unseen rows
  are at 88 % of the seen-row norm, so no impact on normalised Basque text; #394/#397 (unaligned or
  punctuation-only last word cuts the target) — 15 of 200k utterances, negligible. #398 retrained their
  small models on audiobook + conversational data (no change to the recipe itself). Nothing to change in
  our run; pull v4.0.1 before the next stage.
- 2026-10-21 22:16 UTC: 18M distillation finished (100k). Quick eval @100k: named 0.119 / UTMOS 3.50 /
  sim 0.951, held-out 0.133 / 3.55 / 0.935, failures 23/208 — plateau vs 75k.
- 2026-10-21 23:20 UTC: 40/voice comparison (Whisper/Conformer):
  | model | voices | WER | failure rate | WER-ok | UTMOS | sim |
  |---|---|---|---|---|---|---|
  | 6 L 150k (released) | named | 0.049 / 0.040 | 1.3 % | 0.040 | 3.48 | 0.960 |
  | | held-out | 0.029 / 0.030 | 0.5 % | 0.026 | 3.50 | 0.957 |
  | 18M 100k EMA | named | 0.112 / 0.097 | 6.9 % | 0.066 | 3.46 | 0.952 |
  | | held-out | 0.113 / 0.106 | 8.0 % | 0.063 | 3.49 | 0.942 |
  | 18M 100k raw | named | 0.110 / 0.092 | 6.9 % | 0.065 | 3.47 | 0.952 |
  | | held-out | 0.115 / 0.117 | 10.0 % | 0.050 | 3.48 | 0.941 |
  EMA and raw are now equivalent. **Failure analysis: 30 of 38 failures are early stops** (the start of
  the sentence read correctly, then end of speech, e.g. "Hala ere," for a full sentence), 0 prompt
  continuations, 8 other. The EOS head is the teacher's, frozen; it reads the student's z, which is
  slightly off. Testing a higher inference eos_threshold (default −4) on the failed set + 40 control
  sentences. Reports in `reports/small/`.
- 2026-10-22: **eos_threshold fix**. Sweep on the 38 failed + 40 control sentences (seed 0): −4 → 20/38
  fails (control 1/40), −2 → 3/38, 0 → 0/38 (WER 0.047; control WER 0.046, 0 fails), +2 identical to 0; no
  runaway (max 10.7 s). Full 40/voice comparison with `default_eos_threshold: 0.0`: named WER 0.054 /
  0.036 (Whisper/Conformer), failures 0.0 %, UTMOS 3.46, sim 0.958; held-out 0.039 / 0.033, 0.5 %, 3.48,
  0.949 — within ~0.01 WER of the 6-layer model (0.049/0.040, 1.3 %; 0.029/0.030, 0.5 %), sim −0.002/−0.008.
- 2026-10-22: release staged locally in `release/basque_small/` (weights git-ignored): model.safetensors
  (176 MB: backbone+text 14.7 M, teacher head 9.3 M, Mimi 20.1 M), tokenizer, `basque_small.yaml`
  (hf://itzune/pocket-tts-basque-small, eos 0), 5 voice states (Maider, Antton, Nahia, Ane, Aitor) + voices.json,
  README. Loaded with pocket-tts 4.0.1 from the local files; laptop CPU 8.1×/13.0× real time (1/2 threads)
  vs 5.0×/6.7× for the 6-layer model; three clips transcribed correctly. **Not uploaded** (user).
