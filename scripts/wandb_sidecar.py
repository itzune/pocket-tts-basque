#!/usr/bin/env python3
"""Stream a Pocket TTS training run to Weights & Biases without touching the trainer.

Tails <run_dir>/progress.jsonl (one JSON event per line; every numeric field is logged against
its step) and uploads new files from <run_dir>/samples/ as wandb.Audio, captioned with the
sentence from the training config. Resumable: remembers the last line and files already sent.

  WANDB_API_KEY=... python scripts/wandb_sidecar.py --run-dir /root/runs/basque_24l \
      --config configs/basque_24l_finetune.yaml --project pocket-tts-eu --name basque_24l
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config")
    ap.add_argument("--project", default="pocket-tts-eu")
    ap.add_argument("--name")
    ap.add_argument("--entity")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    import wandb

    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    sentences = cfg.get("sample_sentences", [])
    os.makedirs(args.run_dir, exist_ok=True)
    state_path = os.path.join(args.run_dir, ".wandb_sidecar.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"lines": 0, "samples": [], "run_id": None}
    run = wandb.init(project=args.project, entity=args.entity, name=args.name or os.path.basename(args.run_dir.rstrip("/")),
                     id=state.get("run_id"), resume="allow", config=cfg, dir="/tmp")
    state["run_id"] = run.id
    # evaluations arrive later than the training stream; give them their own x-axis so W&B accepts them
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    # same for the audio samples: the trainer writes them at step N but the sidecar sees them after N+50
    wandb.define_metric("samples/step")
    wandb.define_metric("samples/*", step_metric="samples/step")
    progress = os.path.join(args.run_dir, "progress.jsonl")
    sample_re = re.compile(r"step(\d+)_(\d+)\.wav$")
    print("sidecar started for", args.run_dir, "run", run.id, flush=True)
    while True:
        # 1. scalars
        if os.path.exists(progress):
            with open(progress) as fh:
                lines = fh.readlines()
            for line in lines[state["lines"]:]:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = ev.get("step")
                nums = {}

                def flat(d, prefix=""):
                    for k, v in d.items():
                        if k == "step":
                            continue
                        if isinstance(v, bool):
                            continue
                        if isinstance(v, (int, float)):
                            nums[prefix + k] = v
                        elif isinstance(v, dict):
                            flat(v, prefix + k + "/")

                flat(ev)
                tag = ev.get("event") or ev.get("kind") or ev.get("type")
                if tag:
                    nums = {f"{tag}/{k}": v for k, v in nums.items()}
                if nums:
                    wandb.log(nums, step=int(step) if step is not None else None)
            state["lines"] = len(lines)
        # 2. samples
        new = []
        for p in sorted(glob.glob(os.path.join(args.run_dir, "samples", "*.wav"))):
            if p in state["samples"]:
                continue
            m = sample_re.search(p)
            step, idx = (int(m.group(1)), int(m.group(2))) if m else (None, 0)
            cap = sentences[idx] if idx < len(sentences) else os.path.basename(p)
            new.append((step, idx, p, cap))
        for step, idx, p, cap in new:
            try:
                # commit=False: do not advance W&B's internal step counter, or the next valid/checkpoint event
                # (logged with an explicit, now smaller step) would be dropped
                wandb.log({"samples/step": step, f"samples/sentence_{idx}": wandb.Audio(p, caption=f"step {step}: {cap}")}, commit=False)
                state["samples"].append(p)
            except Exception as e:  # noqa: BLE001
                print("sample upload failed", p, e, flush=True)
        # 3. periodic evaluations written by scripts/eval_loop.sh
        state.setdefault("evals", [])
        for sp in sorted(glob.glob(os.path.join(args.run_dir, "eval", "step*", "summary.json"))):
            if sp in state["evals"]:
                continue
            try:
                step = int(re.search(r"step(\d+)", sp).group(1))
                summ = json.load(open(sp))["summary"]
                wandb.log({"eval/step": step, **{f"eval/{kind}/{k}": v for kind, d in summ.items() for k, v in d.items() if k != "n"}}, commit=False)
                state["evals"].append(sp)
            except Exception as e:  # noqa: BLE001
                print("eval log failed", sp, e, flush=True)
        json.dump(state, open(state_path, "w"))
        if os.path.exists(os.path.join(args.run_dir, ".finished")):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
