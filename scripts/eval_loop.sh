#!/usr/bin/env bash
# Evaluate every new checkpoint that is a multiple of EVAL_EVERY steps (default 10000) with
# scripts/eval_basque.py (named + held-out voices, N sentences each); results go to
# <run_dir>/eval/step<N>/summary.json, which the W&B sidecar picks up as eval/* metrics.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-/root/runs/basque_24l}; CFG=${CFG:-configs/basque_24l_infer.yaml}; EVERY=${EVAL_EVERY:-10000}; N=${N:-16}
. logs/ct2.env 2>/dev/null || true
log(){ echo "[$(date -Is)] $*"; }
while true; do
  for ck in $(ls $RUN/checkpoint_*.pt 2>/dev/null | sort); do
    step=$(basename $ck .pt | sed 's/checkpoint_0*//'); step=${step:-0}
    [ $((step % EVERY)) -eq 0 ] || continue
    out=$RUN/eval/step$step; mkdir -p $RUN/eval
    [ -f $out/summary.json ] && continue
    log "evaluating step $step"
    # the trainer keeps only the last 3 checkpoints; keep every evaluated one for the final pick
    mkdir -p $RUN/keep; [ -f $RUN/keep/$(basename $ck) ] || nice cp $ck $RUN/keep/
    # low priority + 1 CPU thread: the trainer's loader subprocesses must keep the GPU fed (only a few CPU cores on the training VM)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LD_LIBRARY_PATH=${CT2_LIBS:-} PYTHONPATH=$HOME/bssd nice -n 19 $HOME/pocket-tts/.venv/bin/python scripts/eval_basque.py --config $CFG --checkpoint $ck \
      --test /root/data/bssd/test.jsonl --named /root/data/named/selected.jsonl --n $N --out $out > $out.log 2>&1 || log "eval failed for step $step (see $out.log)"
    [ -f $out/summary.json ] && log "step $step: $(python3 -c "import json; s=json.load(open('$out/summary.json'))['summary']; print({k: (v['wer'], v['utmos'], v['spk_sim']) for k, v in s.items()})")"
  done
  [ -f $RUN/.finished ] && exit 0
  sleep 600
done
