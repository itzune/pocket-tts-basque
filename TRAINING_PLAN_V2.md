# Plan 3: doing the small Basque model properly (restart from the data up)

Written 2026-10-22 after the first 8 L × 512 student ("small", `TRAINING_PLAN_SMALL.md`) reached WER parity
on paper but audibly more pronunciation slips and dropped words than the 6-layer model. Decision (user):
do it well, restarting where needed, not the cheapest fix.

## Diagnosis of the first attempt

| Kyutai's recipe | what we did | gap |
|---|---|---|
| ~300 M teacher trained from scratch, 2,000 h / 11 M utterances of real speech | 24 L English model fine-tuned 40 k steps at constant LR on 448 h / 350 k synthetic utterances; later checkpoints drifted | **teacher is the ceiling** of any student; ours has 3-5 % failures and was chosen by early stop |
| narrow the teacher to z 64, fine-tune 30 k steps | 10 k steps (disk crash), never fully re-evaluated | targets noisier than needed |
| each stage distilled "to convergence" | fixed 100 k steps; validation loss trend not checked at the end | may have stopped early |
| student sees 11 M distinct utterances | student sees 350 k utterances ~18 times | **generalisation on unseen text** (the "bihar → bidar" class of errors) |
| evaluation: 1,127 sentences × 3 seeds | 520 sentences × 1 seed, WER only | our metric missed what the ear hears |
| EOS head stays calibrated | EOS head fired early; patched with an inference threshold | symptom of z-space mismatch |

The student architecture, seeding and guided targets matched the recipe. The inputs did not.

## Principles

1. Measurement before training: a paired, seed-matched protocol with character-level metrics and a
   listening protocol; every stage has a gate on it, decided before the run.
2. Strengthen the teacher before shrinking anything: more and more diverse data, a proper schedule,
   full evaluation of the narrowed teacher.
3. Run every stage to convergence by validation loss, not to a step count.
4. Fix root causes (EOS calibration, teacher drift) rather than thresholds.
5. Use both VMs: VM2 for backbone runs, VM1 for data generation, scoring and evaluation, so evaluation
   never shares a box with training again.

## Stage 0 — Evaluation protocol (VM1, 1 day, no training)

- `scripts/paired_eval.py`: N ≥ 400 sentences from the test split (fixed list), 13 voices, same seed per
  (voice, sentence) for every model; metrics per sentence: WER and **CER** with both Basque ASRs,
  UTMOS, speaker sim, duration ratio, failure class (early stop / prompt continuation / other);
  **paired** differences model A − B with bootstrap 95 % CIs and per-sentence win/tie/loss; 3 seeds for
  release decisions.
- A hard subset (rare word pieces, proper names, numbers in words, long sentences, questions) to track
  the slips the ear catches; built once from the dev split, frozen.
- Listening protocol: the 5 voices × 20 sentences page, A/B blind order, you score "A better / same / B
  better"; a release needs ≥ 85 % "same or better" against the current model.
- Re-score the released 6 L model and the first small model with it, so the baseline is on the new scale.

## Stage 1 — Data: BSSD v2 (VM1 + VM2 in parallel, ~4-5 days wall)

Target ≥ 1,000 h train (from 448 h), with the diversity the student lacked:
- Text: broaden sources and reduce repetition — more EusCrawl/news/wiki sentences with a per-source cap,
  more generated numbers/dates/acronyms, questions and exclamations, long sentences (15-30 words), and
  a coverage pass so every SentencePiece piece appears ≥ 100 times.
- Voices: same policy (no Common Voice cloning, no politicians); more bank voices (more design briefs) so
  unseen-voice similarity improves; keep the 8 named voices' shares.
- Generation: OmniVoice at 15-35 audio-hours per GPU-hour → 600 new hours ≈ 20-40 GPU-hours on two GPUs,
  then the same scoring (both ASRs, UTMOS, speaker checks) and filtering; publish as BSSD v2 on the Hub
  (same card constraints). Prepare + align + precompute latents.
- Also: regenerate the 8 named-voice prompts' **long-form** material (so the teacher sees them more in
  context) — small share, measurable on named-voice sim.
Disk: VM2 is at 95 %; move the dataset-phase FLAC folders (21.5 GB) off or delete after confirming they
are in the Hub release; a 24 L run needs ~15 GB headroom for checkpoints.

## Stage 2 — Teacher v2 (VM2, ~1.5 days)

- From the English 24 L model with `finetune_language.yaml` (reset text embedding; tokenizer retrained on
  the v2 corpus) on BSSD v2, **cosine schedule to a known length** (120-150 k steps at batch 64) instead
  of constant LR with early stopping; EMA 0.9999; evaluations every 10 k on the paired protocol with
  CFG 2. Keep the checkpoint with the best paired score, not the last.
- Then narrow to z 64 (PCA init as now) and **fine-tune the full 30 k steps**; gate: paired score within
  noise of the un-narrowed teacher (Kyutai: UTMOS 4.24 vs 4.27). If 64 fails the gate, try 128 (their
  ablation shows 128 ≈ 64).
- Compute: ~3 steps/s → 150 k steps ≈ 14 h + 30 k ≈ 3 h. Disk ~5 GB per save (keep 2 + kept evals).

## Stage 3 — Student v2, 8 L × 512 (VM2, ~1-2 days)

- As before (seeded from the teacher along principal components, guided targets, heads frozen), but
  trained until the validation distillation loss is flat for 25 k steps (plan 200 k, cosine to that),
  evaluations every 25 k on the paired protocol with the hard subset.
- Gate: paired CER/WER within CI of the 6 L model on both ASRs, failure rate ≤ 1 %, named-voice sim
  within 0.005, listening ≥ 85 % same-or-better. If it fails on the hard subset only, go to Stage 3b.
- 3b (only if needed): larger student, 8 L × 768 or 10 L × 640 (~30 M backbone), same procedure. Still
  ~1.4× faster than 6 L; an honest size for our teacher/data if 512 cannot reach parity.

## Stage 4 — Heads (VM1 while Stage 3 evaluates)

- **EOS head recalibration**: fine-tune only `out_eos` on the student's frozen backbone (minutes), then
  evaluate with the default threshold; no inference threshold hacks in the release config.
- Sampler head: distil the 0.84 M drifting head (Kyutai: no quality cost); gate on the paired protocol.
- Decoder: keep the standard Mimi decoder unless the 2.6 M one is faster in the target runtime (it is
  not in PyTorch); re-test it inside the WASM engine only for the browser build.

## Stage 5 — Reward fine-tuning (VM2, ½ day)

ReFL with UTMOS and the Basque wav2vec2 CTC likelihood as rewards, distillation loss as anchor,
3 k steps; monitor pitch range (F0 std on the listening set) and reject if it narrows by > 0.3 semitones.

## Stage 6 — Release

Paired report (3 seeds), listening pass, CPU benchmarks (PyTorch and WASM), voice states, card, Hub repo
`itzune/pocket-tts-basque-small` (replacing the staged candidate), demo page update. Then, optionally,
the gradual cuts towards 7 M with the same gates.

## Timeline and sequencing

| days | VM1 | VM2 |
|---|---|---|
| 1 | Stage 0 protocol + baselines | BSSD v2 text pool, voice bank extension |
| 2-5 | generation + scoring (half the shards) | generation + scoring (other half) |
| 6 | align, latents, publish v2 | teacher v2 starts |
| 7 | — | teacher v2 (cont.) + narrowing |
| 8-9 | student evaluations as they land | student v2 |
| 10 | EOS/head work, benchmarks | (3b if needed) |
| 11 | ReFL, release | — |

About two weeks of calendar time with the 8:00 shutdowns, mostly unattended. The stage gates are the
places to stop and reconsider; nothing after Stage 1 is started without the previous gate passing.

## Open decisions (user)

1. Data target: 1,000 h (plan) or more; and whether BSSD v2 is published as a new version.
2. Teacher: fine-tune from English 24 L as now (plan) vs. from scratch (Kyutai's path; needs far more data
   and ~4× the compute; not recommended at 1,000 h).
3. Whether the current staged "small" model should be uploaded as an interim release or dropped. Plan: drop.

## Log

- 2026-10-22 14:00 UTC: both VMs up. Disk freed (local copies of the published release and the dataset-phase
  audio deleted: VM1 89 GB free, VM2 50 GB). Exclusion list of 468k already-synthesised sentences built.
  VM1: Pocket TTS v4.0.1 env, test split pulled, frozen eval sets (main 400 test sentences; hard 169 dev
  sentences tagged palatal/g_j/names/numbers/question/long), baseline paired evals of the 6 L and small
  models started (each sentence read by one named + one held-out voice, seed-matched). Voice bank round 3
  (2,000 candidates) and the v3 text pool (1.3 M target, all-new sentences, ≤ 40 words) building on VM1.
- 2026-10-22 14:40 UTC: **pronunciation probe of the data teacher** (`scripts/small/pron_probe.py`, curated
  native words, Maider/Antton prompts, Whisper-eu check): OmniVoice reads "Antton" as "Anton" 16/16,
  "pittin" as "pitin", "onddo" as "ondo" 6/8, "Pello"/"ollo" with Spanish ll (→ "tejo", "oiho"); ñ, j
  (initial/medial), ge/gi, x and tz/ts/tx are fine. So BSSD v1 contains no correct palatal "tt/dd/ll" and
  the released model cannot have learned them (user's observation; Piper, phoneme-based, gets them right).
  Two fixes being tested: (1) generation-side respellings that make OmniVoice produce the palatal while the
  transcript keeps the standard spelling (probe 3 running); (2) **real studio speech as training data**: the
  HiTZ-Aholab corpus (Zenodo 17952596, CC BY 4.0, two speakers, 16 GB WAV, explicitly covers "tt" and other
  Basque-specific spellings) + OpenSLR 76 (7,136 utterances, CC BY-SA 4.0), both already licensed for this
  use; downloading HiTZ to VM1. Listening page: `samples/pron_probe/index.html`.
- 2026-10-22 15:10 UTC: respelling probe (6 palatal words × 4-7 spellings × Maider/Antton, ASR hint): "dd" can
  be coaxed ("ondyo"/"onddyo"/"ondjo" → heard as "onddo"); "tt" cannot (Antyon → "antzio/antxo", Anttyon,
  Antxon, Anchon → "antson/antxon": affricates, never the palatal stop); "ll" and "Maddi" unchanged.
  Conclusion: OmniVoice has no /c/, so generation-side respelling cannot fix "tt"; the fix is real speech
  (HiTZ-Aholab explicitly covers "tt") in the training mix, with sentences containing palatals
  over-sampled, plus the palatal tag in the hard set as the gate. Clips: `samples/pron_probe/index.html`.
- 2026-10-22 16:00 UTC: bank round 3 done (30 new voices, 164 total; merge-path bug fixed). Pool v3 rebuilt
  after a wrong corpus path cost an hour. Queue env + reboot cron prepared on both VMs (finalize/upload
  disabled). Baseline paired evals slower than planned on the shared VM1 (CPU-bound): 6 L ~⅓ done.
  HiTZ download at 5.6/16 GB; ingestion (`bssd/scripts/ingest_real.py`) auto-starts when it lands.
- 2026-10-22 (VM clock 14:39 UTC): **BSSD v2 generation launched** on both VMs: pool v3 = 1,092,969 new
  sentences (euscrawl 650k, euwiki 295k, cv 44k, numbers 104k); bank3 = 164 voices; palatal sentences
  (tt/dd/ll between vowels, ~1 % of text) are skipped at job creation after the user confirmed by ear that
  OmniVoice's palatals are unusable even with respellings — real speech will be their only source.
  VM1 runs v12-v16 (offsets 0-200k), VM2 v17-v20 (250k-400k), ~49.5k jobs per run, 10 shards each;
  score loops and Conformer loops running; reboot cron on `~/bssd/scripts/resume_all.sh` (finalize/upload
  disabled). Expected ≈ 450k utterances ≈ 580 h; ~3 h per run → VM1 ~15 h, VM2 ~12 h, with the 08:00
  shutdowns in between (resumable). Baseline paired evals continue on VM1 (chain adds Conformer + report).
- 2026-10-22 ~17:00 UTC: real speech ingested on VM1 (`/root/data/real_v2`, 24 kHz FLAC + rows.jsonl):
  HiTZ-Aholab Maider 17.6 h + Antton 16.7 h (26,909 rows; the two archives share file names and had to be
  extracted into separate folders — first pass had pointed every Maider row at Antton's audio) and
  OpenSLR 76 12.6 h (6,548 rows) with the 10 dev/test held-out speakers excluded; 46.9 h total, 1,493 rows
  with palatals, 10 normaliser rejects, 91 clips < 0.8 s dropped.
- 2026-10-22 ~17:30 UTC: **Stage 0 baseline on the new paired protocol** (1,138 seed-matched utterances,
  `reports/paired/report_6l_vs_small.md`): small vs 6 L — WER/CER ties within CI on both ASRs (Conformer
  CER even slightly better: −0.002, CI excludes 0), UTMOS −0.025 and speaker sim −0.004 (both significant),
  failures 3 vs 6. So the ASR metrics do not see the slips the ear catches; UTMOS does, faintly. What the
  per-sentence view shows: the small model garbles the **first word** of the time-table sentences
  ("Autobusa" → "Aupuru/Aukudu") — an utterance-onset problem, not a general one; and the 6 L model has a
  question-sentence weakness (hard-set "question" CER 0.048 vs 0.004 for small). Both are new, specific,
  measurable targets for v2. Protocol note: the ASR-based gate must be complemented by the blind listening
  protocol; CER on the hard-set tags (palatal, numbers, names, question) is the most sensitive automatic signal.
- 2026-10-22 ~17:40 UTC: user listened to 24 real palatal clips (`samples/real_palatal/`): Maider/Antton
  pronounce tt/dd/ll correctly but with flat, Piper-like prosody and lower perceived quality (UTMOS says
  otherwise — Maider 3.72, Antton 3.51 vs synthetic p50 3.48 — so UTMOS does not capture prosody; SLR76
  crowd audio is 3.19, p10 2.62). **Decision**: real speech is a pronunciation source only, not a style
  source — v2 training uses just the HiTZ rows that contain a palatal (1,493 rows ≈ 2 h, repeated ×2),
  no other real rows, no SLR76 audio (`build_v2_manifest.py` defaults). The palatal tag of the hard set
  tells whether that is enough; if not, raise the repeat before adding non-palatal real rows. The real
  corpora stay packaged for the BSSD v2 release as a separate `real` config.
- 2026-10-22 17:00 UTC: tokenizer v2 (`/root/data/bssd_v2/tokenizer/basque_v2.model`, 4000 pieces, same
  spec as the official script) trained on 1.46 M lines (v1 + pool v3 texts without palatals, real
  transcripts with them). A first attempt split the palatal digraphs across pieces ("An"+"t"+"ton"), so
  it is retrained with `user_defined_symbols=["tt","dd","ll"]` (`scripts/small/train_tokenizer_v2.py`) so
  every palatal is one dedicated piece the model can tie to the real-speech examples. dev/test manifests and
  the eval sets stay as they are (they only use text).
- 2026-10-22 18:00 UTC: tokenizer v2 verified on VM2: "An|tt|on", "Ma|dd|i", "Pe|ll|o" — palatal digraphs are
  single pieces (`/root/data/bssd_v2/tokenizer/basque_v2.model`, 4000 pieces). Day-1 state: generation
  v12/v17 at shard ~7/10 on both VMs; nothing else running. Tomorrow: wait for generation + scoring →
  package v2 synthetic (per VM) → pull to VM2 → `build_v2_manifest.py` → align + latents → teacher v2.
- 2026-10-23: VMs were off from ~08:00 to ~21:15 local (13 h lost). Reboot resume worked unattended on both:
  gen_queue skipped the finished runs and resumed v15 at shard 4 (VM1) and v20 at shard 5 (VM2), scoring
  and Conformer loops restarted. Done before the shutdown: v12-v14 (VM1), v17-v19 (VM2) = ~297k rows.
  Remaining: v15 (7 shards), v20 (6) + v16 (10). Primary scoring lags (~2 shards/h while sharing the GPU):
  22 shards left on VM1, 31 on VM2 → scoring finishes about a day after generation. Schedule slips ~1 day.
- 2026-10-24 12:00 local: VM1 parts ready: `synth_vm1_aligned_latents.jsonl` (160,037 rows, 213.5 h, latents
  1.5 GB) and `real_pal_aligned_latents.jsonl` (716 rows, 1.0 h); v1 assets copied in (`/root/data/bssd`,
  Mimi hash 29335a05 matches). VM1's runs packaged for the Hub release (`/root/data/release_v2/vm1`: 81
  parquet files, 160,037 rows, 6,550 voices, 20 GB, not uploaded); their FLACs deleted afterwards (the
  parquet holds the audio, the latents the training representation) to make room for 24 L checkpoints.
  Waiting for VM2's prep chain (starts when v16 scoring ends), then merge + teacher v2 launch.
