#!/usr/bin/env bash
# Forced alignment of the BSSD manifests with a Basque wav2vec2-CTC model, via the official
# Pocket TTS aligner. Usage: scripts/align_bssd.sh /root/data/bssd [model] [split ...]
set -euo pipefail
DATA=${1:-/root/data/bssd}; MODEL=${2:-deepdml/wav2vec2-large-xls-r-300m-basque}; shift 2 || true
SPLITS=${@:-dev test train}
PT=${POCKET_TTS:-$HOME/pocket-tts}
cd "$PT"
for sp in $SPLITS; do
  echo "=== aligning $sp with $MODEL $(date -Is)"
  .venv/bin/python training/scripts/align_data.py "$DATA/$sp.jsonl" "$DATA/${sp}_aligned.jsonl" --model "$MODEL" --resume
done
