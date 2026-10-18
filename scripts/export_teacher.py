#!/usr/bin/env python3
"""Export a training checkpoint (.pt) to a Pocket TTS `model.safetensors` without building the model.

Memory-light (mmap): takes the flow_lm EMA tensors from the checkpoint and the frozen Mimi tensors from a
previous trainer export (`<run_dir>/model.safetensors`, written at every checkpoint), so it can run next
to a training job. Same key layout as training.checkpointing.export_pocket_safetensors.

  python scripts/export_teacher.py --checkpoint /root/runs/basque_24l/keep/checkpoint_00040000.pt \
      --base /root/runs/basque_24l/model.safetensors --out /root/export/basque_24l/model.safetensors
"""
import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True, help="trainer model.safetensors (for the mimi.* tensors and the key list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model_state = payload["model"]
    # EMA.state_dict() is the shadow dict itself: {"flow_lm.<k>": tensor, "flow.<k>": tensor}
    ema = None if args.no_ema else payload.get("ema")
    print("checkpoint step", payload.get("step"), "| model keys", len(model_state), "| ema keys", len(ema) if ema else 0)
    # the trainer wraps flow_lm + the lsd flow head ("flow.*", training-only); only flow_lm.* is exported
    flow = {k: v.detach().float().cpu() for k, v in model_state.items() if k.startswith("flow_lm.")}
    n_ema = 0
    if ema:
        for k, v in ema.items():
            if k in flow:
                flow[k] = v.detach().float().cpu()
                n_ema += 1
    print("ema tensors applied", n_ema, "of", len(flow))
    out = {}
    with safe_open(args.base, "pt") as f:
        base_keys = set(f.keys())
        for k in f.keys():
            if k.startswith("mimi."):
                out[k] = f.get_tensor(k)
    missing = {k for k in base_keys if k.startswith("flow_lm.")} - set(flow)
    extra = set(flow) - base_keys
    assert not missing and not extra, f"key mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
    for k in base_keys:
        if k.startswith("flow_lm."):
            out[k] = flow[k].contiguous()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(out, args.out, metadata={"format": "pt", "source_checkpoint": os.path.basename(args.checkpoint), "step": str(payload.get("step"))})
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e9, 3), "GB,", len(out), "tensors")


if __name__ == "__main__":
    main()
