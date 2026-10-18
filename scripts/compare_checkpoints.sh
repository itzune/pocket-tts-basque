#!/usr/bin/env bash
# Final teacher comparison: evaluate every kept checkpoint with 40 sentences/voice and a fixed seed,
# score with Basque Whisper (in eval_basque.py) and the HiTZ Conformer (score_conformer.py, NeMo venv),
# then write the comparison table. Run on VM2 after training has finished (GPU idle).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-/root/runs/basque_24l}; CFG=${CFG:-configs/basque_24l_infer.yaml}; N=${N:-40}
OUT=${OUT:-$RUN/eval_final}; NEMO_PY=${NEMO_PY:-$HOME/nemo-venv/.venv/bin/python}
. logs/ct2.env 2>/dev/null || true
log(){ echo "[$(date -Is)] $*"; }
mkdir -p $OUT
STEPS=${STEPS:-}   # optional space-separated list of steps to evaluate (default: every kept checkpoint)
for ck in $(ls $RUN/keep/checkpoint_*.pt | sort); do
  step=$(basename $ck .pt | sed 's/checkpoint_0*//')
  if [ -n "$STEPS" ]; then case " $STEPS " in *" $step "*) ;; *) continue ;; esac; fi
  out=$OUT/step$step
  if [ ! -f $out/summary.json ]; then
    log "evaluating $ck"
    OMP_NUM_THREADS=4 LD_LIBRARY_PATH=${CT2_LIBS:-} PYTHONPATH=$HOME/bssd $HOME/pocket-tts/.venv/bin/python scripts/eval_basque.py \
      --config $CFG --checkpoint $ck --test /root/data/bssd/test.jsonl --named /root/data/named/selected.jsonl \
      --n $N --seed 0 --out $out > $out.log 2>&1 || { log "eval failed for step $step"; continue; }
  fi
  if [ ! -f $out/eval_conformer.jsonl ]; then
    log "conformer scoring step $step"
    $NEMO_PY scripts/score_conformer.py --dir $out > $out.conformer.log 2>&1 || log "conformer failed for step $step"
  fi
  log "step $step: $(python3 -c "import json; s=json.load(open('$out/summary.json'))['summary']; print({k: (v['wer'], v.get('wer_conformer'), v['fail_rate'], v['utmos']) for k, v in s.items()})")"
done
python3 scripts/eval_table.py $OUT/step* > $OUT/compare.md && log "table written to $OUT/compare.md"
log "COMPARE DONE"
