#!/usr/bin/env bash
# After a reboot (or a crash): relaunch whichever run is started but not finished — the teacher
# (basque_24l) and/or the student (basque_6l) — plus its eval loop and W&B sidecar. train.py resumes from
# the last checkpoint in run_dir. Idempotent. Installed as @reboot cron on the training VM.
# NOTE: liveness checks use pgrep (never ps|grep, which can match itself).
cd "$(dirname "$0")/.."
PT=$HOME/pocket-tts; PY=$PT/.venv/bin/python
mkdir -p logs; log(){ echo "[$(date -Is)] $*" >> logs/resume_training.log; }
log "resume_training start"

resume_run(){  # name train_cfg infer_cfg eval_every
  local name=$1 cfg=$PWD/configs/$2 infer=configs/$3 every=$4 run=/root/runs/$1
  [ -f $run/.started ] && [ ! -f $run/.finished ] || return 0
  pgrep -f "training/train\.py .*$2" >/dev/null || { log "relaunching train.py ($name)"; (cd $PT && nohup $PY training/train.py $cfg >> $HOME/pocket-tts-eu/logs/train_$name.log 2>&1 < /dev/null &); }
  pgrep -f "eval_loop\.sh" >/dev/null \
    || { log "relaunching eval loop ($name)"; (RUN=$run CFG=$infer EVAL_EVERY=$every nohup bash scripts/eval_loop.sh >> logs/eval_loop_$name.log 2>&1 < /dev/null &); }
  pgrep -f "wandb_sidecar\.py --run-dir $run" >/dev/null || { log "relaunching sidecar ($name)"; (set -a; . logs/wandb.env; set +a; nohup $PY scripts/wandb_sidecar.py --run-dir $run --config $cfg --project pocket-tts-eu --name $name >> logs/wandb_sidecar_$name.log 2>&1 < /dev/null &); }
}

resume_run basque_24l basque_24l_finetune.yaml basque_24l_infer.yaml 10000
resume_run basque_6l basque_6l_distill.yaml basque_6l_infer.yaml 25000
resume_run basque_24l_z64 basque_24l_z64.yaml basque_24l_z64_infer.yaml 10000
resume_run basque_18m basque_18m_distill.yaml basque_18m_infer.yaml 25000
resume_run basque_24l_v2 basque_24l_v2.yaml basque_24l_v2_infer.yaml 10000
# memory guard for the leaking trainer process (restarts it from the last checkpoint when needed)
for spec in "basque_6l basque_6l_distill.yaml" "basque_24l_z64 basque_24l_z64.yaml" "basque_18m basque_18m_distill.yaml" "basque_24l_v2 basque_24l_v2.yaml"; do set -- $spec
  [ -f /root/runs/$1/.started ] && [ ! -f /root/runs/$1/.finished ] && ! pgrep -f "mem_guard\.sh" >/dev/null \
    && { log "relaunching mem_guard ($1)"; (RUN=/root/runs/$1 CFG_NAME=$2 setsid -f nohup bash scripts/mem_guard.sh >> logs/mem_guard_$1.log 2>&1 < /dev/null); }
done
