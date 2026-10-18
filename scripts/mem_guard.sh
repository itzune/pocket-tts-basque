#!/usr/bin/env bash
# The trainer's main process leaks host memory slowly (~0.25 GB per 1k steps on the distillation run).
# On a 15 GB VM that ends in an OOM kill, so: when its RSS+swap exceeds LIMIT_GB, wait for the next
# checkpoint to land, kill the trainer tree and let resume_training.sh relaunch it (train.py resumes
# from the last checkpoint; cost ≈ 2 min + the steps since that checkpoint).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-/root/runs/basque_6l}; CFG_NAME=${CFG_NAME:-basque_6l_distill.yaml}; LIMIT_GB=${LIMIT_GB:-11}
log(){ echo "[$(date -Is)] $*"; }
mem_gb(){ awk '/^VmRSS:/{r=$2} /^VmSwap:/{w=$2} END{printf "%.1f", (r+w)/1e6}' /proc/$1/status 2>/dev/null; }
log "mem_guard start (limit ${LIMIT_GB} GB)"
while true; do
  [ -f $RUN/.finished ] && { log "run finished, exiting"; exit 0; }
  pid=$(pgrep -of "training/train\.py .*$CFG_NAME" || true)
  if [ -n "$pid" ]; then
    m=$(mem_gb $pid)
    if awk -v m="$m" -v l="$LIMIT_GB" 'BEGIN{exit !(m>l)}'; then
      last=$(ls -t $RUN/checkpoint_*.pt 2>/dev/null | head -1)
      log "trainer pid $pid at ${m} GB > ${LIMIT_GB} GB; waiting for the next checkpoint after $(basename "${last:-none}")"
      while [ "$(ls -t $RUN/checkpoint_*.pt 2>/dev/null | head -1)" = "$last" ]; do sleep 20; [ -f $RUN/.finished ] && exit 0; done
      sleep 30   # let the checkpoint + model.safetensors export complete
      new=$(ls -t $RUN/checkpoint_*.pt | head -1)
      log "checkpoint $(basename $new) written; restarting trainer"
      pkill -9 -f "training/train\.py .*$CFG_NAME"; pkill -9 -f "multiprocessing.spawn" ; sleep 5
      bash scripts/resume_training.sh
      log "relaunched: $(tail -1 logs/resume_training.log)"
      sleep 300
    fi
  fi
  sleep 120
done
