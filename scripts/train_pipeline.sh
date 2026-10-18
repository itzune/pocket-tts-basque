#!/usr/bin/env bash
# Resumable driver for the teacher stage on the training VM:
# wait for prepare_bssd -> align train -> tokenizer -> teacher fine-tune (nohup, resumable via checkpoints).
# Usage: bash scripts/train_pipeline.sh   (from ~/pocket-tts-eu; expects ~/pocket-tts checkout)
set -uo pipefail
cd "$(dirname "$0")/.."
PT=$HOME/pocket-tts; D=/root/data/bssd; PY=$PT/.venv/bin/python
ALIGNER=${ALIGNER:-deepdml/wav2vec2-large-xls-r-300m-basque}
log(){ echo "[$(date -Is)] $*"; }
until grep -qE "^train: [0-9]+ rows" logs/prepare_bssd.log 2>/dev/null; do sleep 120; done
log "data ready: $(grep -E '^(train|dev|test):' logs/prepare_bssd.log | tr '\n' ' ')"
for sp in dev test train; do
  [ -s $D/${sp}_aligned.jsonl ] && [ "$(wc -l < $D/${sp}_aligned.jsonl)" -ge "$(wc -l < $D/$sp.jsonl)" ] && continue
  log "aligning $sp"
  (cd $PT && $PY training/scripts/align_data.py $D/$sp.jsonl $D/${sp}_aligned.jsonl --model $ALIGNER --resume 2>&1 | grep -E "done:|Error|Traceback" | tail -2)
done
if [ ! -s $D/tokenizer/basque.model ]; then
  log "training tokenizer"
  mkdir -p $D/tokenizer
  (cd $PT && $PY training/scripts/train_tokenizer.py $D/tokenizer/basque $D/train_aligned.jsonl 2>&1 | tail -3)
  ls -la $D/tokenizer/
fi
# the EN teacher weights are a gated HF repo: wait until this token is authorised
until $PY - <<'PYEOF'
from huggingface_hub import hf_hub_download
hf_hub_download("kyutai/pocket-tts", "languages/english_2026-04_24l/model.safetensors", revision="492522650173a0653b7575cdc25ae09810e5d741")
PYEOF
do log "waiting for access to gated kyutai/pocket-tts (accept at https://huggingface.co/kyutai/pocket-tts)"; sleep 600; done
log "teacher weights available"
if [ ! -f /root/runs/basque_24l/.started ]; then
  mkdir -p /root/runs/basque_24l && touch /root/runs/basque_24l/.started
  log "launching teacher fine-tune (see logs/train_24l.log)"
fi
cd $PT
nohup $PY training/train.py $HOME/pocket-tts-eu/configs/basque_24l_finetune.yaml >> $HOME/pocket-tts-eu/logs/train_24l.log 2>&1 < /dev/null &
log "train.py pid $!"
