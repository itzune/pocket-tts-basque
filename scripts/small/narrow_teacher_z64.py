#!/usr/bin/env python3
"""Seed a z_dim-narrowed copy of a trained teacher (Kyutai's "narrow interface" step).

The teacher's backbone output h (d_model wide, after out_norm) feeds the sampler head and the EOS
head. We add z_proj = top-k principal components of h (from training/scripts/residual_cov.py: the
second moment of out_norm's normalized input, with the norm gain folded in) and fold the projection
into the linears that read h (flow_net.cond_embed, out_eos), so the narrowed teacher computes nearly
the same function. Written as a step-0 run_dir checkpoint for a fine-tune with flow_lm.z_dim: <k>.

  python scripts/small/narrow_teacher_z64.py --checkpoint keep/checkpoint_00040000.pt \
      --cov residual_cov.pt --z-dim 64 --out /root/runs/basque_24l_z64/checkpoint_00000000.pt
"""
import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cov", required=True, help="residual_cov.py output")
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    if not args.no_ema and payload.get("ema"):
        state.update({k: v for k, v in payload["ema"].items() if k in state})
        print("using EMA weights")
    cov = torch.load(args.cov, weights_only=True)["cov"]
    key = [k for k in cov if k.endswith("out_norm")]
    assert key, f"no out_norm covariance in {list(cov)[:5]}"
    c = cov[key[0]].to(torch.float64)
    g = state["flow_lm.out_norm.weight"].to(torch.float64)
    b = state["flow_lm.out_norm.bias"].to(torch.float64)
    # covariance of what the heads actually read: h = n * g + b  (bias dropped for the basis)
    c_h = (g[:, None] * c) * g[None, :]
    evals, evecs = torch.linalg.eigh(c_h)
    order = torch.argsort(evals, descending=True)[: args.z_dim]
    U = evecs[:, order]  # [d_model, z]
    kept = float(evals[order].sum() / evals.sum())
    print(f"top-{args.z_dim} PCs keep {kept:.4f} of the head-input variance")
    d_model = U.shape[0]
    state["flow_lm.z_proj.weight"] = U.T.contiguous().to(torch.float32)  # z = U^T h
    folded = 0
    for name in ("flow_lm.flow_net.cond_embed.weight", "flow_lm.out_eos.weight"):
        w = state[name].to(torch.float64)
        assert w.shape[1] == d_model, (name, tuple(w.shape))
        # w h ≈ w (U U^T h) + w (I - U U^T) b-ish terms; fold the mean of the dropped part into the bias
        state[name] = (w @ U).to(torch.float32)
        bias_name = name.replace(".weight", ".bias")
        dropped = (torch.eye(d_model, dtype=torch.float64) - U @ U.T) @ b
        if bias_name in state:
            state[bias_name] = (state[bias_name].to(torch.float64) + w @ dropped).to(torch.float32)
        folded += 1
    print("folded", folded, "head inputs; z_proj", tuple(state["flow_lm.z_proj.weight"].shape))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"step": 0, "model": state, "optimizer": None, "ema": None}, args.out)
    print("->", args.out)


if __name__ == "__main__":
    main()
