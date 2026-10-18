#!/usr/bin/env bash
# Start the student (depth distillation) run on VM2 once configs/basque_6l_distill.yaml points at the
# chosen teacher checkpoint. Marks the run as started and lets resume_training.sh launch everything.
set -euo pipefail
cd "$(dirname "$0")/.."
grep -q "checkpoint_best.pt" configs/basque_6l_distill.yaml && { echo "set distill_teacher_weights first"; exit 1; }
mkdir -p /root/runs/basque_6l && touch /root/runs/basque_6l/.started
bash scripts/resume_training.sh
tail -3 logs/resume_training.log
