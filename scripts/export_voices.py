#!/usr/bin/env python3
"""Export the named voices as Pocket TTS voice states (.safetensors) for a given model.

Voice states are model-specific (they are the model's state after reading the prompt), so re-run this
for every released model. Also writes voices/voices.json with gender/source metadata.

  python scripts/export_voices.py --config configs/basque_6l_infer.yaml [--checkpoint ...] \
      --named /root/data/named/selected.jsonl --out /root/export/basque_6l/voices
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--named", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pocket_tts import TTSModel, export_model_state

    model = TTSModel.load_model(config=args.config, checkpoint=args.checkpoint)
    os.makedirs(args.out, exist_ok=True)
    meta = {}
    for line in open(args.named):
        v = json.loads(line)
        name = v["voice_id"]
        state = model.get_state_for_audio_prompt(v["path"])
        dest = os.path.join(args.out, f"{name}.safetensors")
        export_model_state(state, dest)
        meta[name] = {"gender": v.get("gender"), "source": v.get("source_candidate"), "brief": v.get("brief"),
                      "prompt_duration_s": v.get("duration"), "size_mb": round(os.path.getsize(dest) / 1e6, 2)}
        print(name, meta[name], flush=True)
    json.dump(meta, open(os.path.join(args.out, "voices.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
