#!/usr/bin/env bash
# Per-VM preparation of the BSSD v2 synthetic runs for training (TRAINING_PLAN_V2.md stage 1 → 2):
#   reflag scores with the Conformer merged → manifest of kept rows → forced alignment → Mimi latents.
# Usage: RUNS=v12,v13,v14,v15 TAG=vm1 bash scripts/v2_prepare_vm.sh
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:?}; TAG=${TAG:?}; D=/root/data; OUT=$D/bssd_v2; mkdir -p $OUT
PT=$HOME/pocket-tts; PY=$PT/.venv/bin/python; BSSD=$HOME/bssd
log(){ echo "[$(date -Is)] $*"; }
for run in ${RUNS//,/ }; do
  log "reflag $run (merge secondary)"
  (cd $BSSD && PYTHONPATH=. .venv/bin/python scripts/reflag.py --scores $D/scores_$run --flac $D/flac_$run --config configs/dataset_v1.yaml --merge-secondary 2>&1 | tail -2)
done
log "manifest"
$PY scripts/v2_manifest_from_scores.py --data $D --runs $RUNS --out $OUT/synth_$TAG.jsonl || exit 1
log "alignment"
(cd $PT && $PY training/scripts/align_data.py $OUT/synth_$TAG.jsonl $OUT/synth_${TAG}_aligned.jsonl --model deepdml/wav2vec2-large-xls-r-300m-basque --resume 2>&1 | grep -vE "it/s\]" | tail -3)
log "latents"
cat > $OUT/precompute_$TAG.yaml <<YAML
model_config: pocket_tts/config/english_2026-04_24l.yaml
model_overrides:
  flow_lm.lookup_table.tokenizer: sentencepiece
  flow_lm.lookup_table.tokenizer_path: $OUT/tokenizer/basque_v2.model
start_from_pretrained: false
data:
  train_jsonl: $OUT/synth_${TAG}_aligned.jsonl
  valid_jsonl: $OUT/synth_${TAG}_aligned.jsonl
run_dir: /root/runs/_precompute_$TAG
YAML
(cd $PT && $PY training/scripts/precompute_latents.py $OUT/precompute_$TAG.yaml --batch-size 16 2>&1 | grep -vE "it/s\]" | tail -3)
ls -la $OUT | grep -E "synth_$TAG"
log "PREPARE $TAG DONE"
