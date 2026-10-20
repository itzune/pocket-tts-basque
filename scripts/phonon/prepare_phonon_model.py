#!/usr/bin/env python3
"""Turn a released Pocket TTS Basque folder (model.safetensors, tokenizer.model, voices/) into a checkpoint
folder the phonon-tts browser engine (gradium-ai/xn-ptts, npm `phonon-tts`) loads:

  config.json      engine config for the Pocket TTS 6-layer architecture (scripts/phonon/config.json)
  tokenizer.json   tokenizers-JSON conversion of the SentencePiece model (xn-ptts scripts/convert-tokenizer.py)
  model.safetensors  our weights minus `flow_lm.speaker_proj_weight` (training checkpoints project 32-dim
                   latents, the engine expects 512; the browser uses precomputed KV-cache voices, so the
                   projection is never run) and optionally minus the Mimi encoder + `bos_before_voice`
                   (--native: what the strict native loader needs; the browser build ignores them)
  voices/          our voice states as they are: `transformer.layers.{i}.self_attn/cache` is exactly the
                   KV-cache voice format the engine reads

  uv run --with safetensors --with torch --with numpy scripts/phonon/prepare_phonon_model.py \
      --src release_dir --out phonon_dir
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = "https://raw.githubusercontent.com/gradium-ai/xn-ptts/main/scripts/convert-tokenizer.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with model.safetensors, tokenizer.model, voices/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--native", action="store_true", help="also drop the Mimi encoder and bos_before_voice (native ptts loader)")
    args = ap.parse_args()
    from safetensors.torch import load_file, save_file

    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)
    t = load_file(os.path.join(args.src, "model.safetensors"))
    dropped = [k for k in t if k == "flow_lm.speaker_proj_weight"]
    if args.native:
        dropped += [k for k in t if k.startswith(("mimi.encoder", "mimi.encoder_transformer", "mimi.downsample")) or k == "flow_lm.bos_before_voice"]
    for k in dropped:
        del t[k]
    save_file(t, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt", "note": "prepared for phonon-tts; dropped " + ",".join(dropped)})
    print("weights:", len(t), "tensors, dropped", len(dropped))
    shutil.copy(os.path.join(HERE, "config.json"), os.path.join(args.out, "config.json"))
    conv = os.path.join(args.out, "_convert-tokenizer.py")
    if not os.path.exists(conv):
        subprocess.run(["curl", "-sL", CONVERTER, "-o", conv], check=True)
    subprocess.run(["uv", "run", "-q", "--no-project", conv, os.path.join(args.src, "tokenizer.model"), "-o", os.path.join(args.out, "tokenizer.json")], check=True)
    os.remove(conv)
    for f in os.listdir(os.path.join(args.src, "voices")):
        if f.endswith(".safetensors"):
            shutil.copy(os.path.join(args.src, "voices", f), os.path.join(args.out, "voices", f))
    print("ready:", sorted(os.listdir(args.out)), "voices:", sorted(os.listdir(os.path.join(args.out, "voices"))))


if __name__ == "__main__":
    main()
