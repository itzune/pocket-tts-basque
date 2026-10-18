#!/usr/bin/env python3
"""Generate the demo clips for docs/ with the released Pocket TTS Basque model, on CPU (2 threads) so the
timings shown on the page are real CPU numbers. Writes <out>/pocket/<Voice>_<id>.wav and timings.json.

  python scripts/make_demo_audio.py --config hf://itzune/pocket-tts-basque/basque.yaml \
      --voices-dir /root/export/basque_6l/voices --sentences docs/demo/sentences.json --out /root/demo
"""
import argparse
import json
import os
import time

import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--voices-dir", required=True, help="folder with <Voice>.safetensors states")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pocket_tts import TTSModel

    cfg = json.load(open(args.sentences))
    model = TTSModel.load_model(config=args.config)
    out = os.path.join(args.out, "pocket")
    os.makedirs(out, exist_ok=True)
    timings = {}
    for voice in cfg["voices"]:
        state_path = os.path.join(args.voices_dir, f"{voice}.safetensors")
        items = [("intro", cfg["intro"].format(name=voice))] + [(c["id"], c["text"]) for c in cfg["common"]]
        for sid, text in items:
            torch.manual_seed(0)
            state = model.get_state_for_audio_prompt(state_path)  # fresh state per sentence
            t0 = time.time()
            audio = model.generate_audio(state, text)
            dt = time.time() - t0
            a = audio.detach().cpu().numpy().squeeze()
            sf.write(os.path.join(out, f"{voice}_{sid}.wav"), a, model.sample_rate)
            timings[f"{voice}_{sid}"] = {"audio_s": round(len(a) / model.sample_rate, 2), "gen_s": round(dt, 2),
                                         "x_realtime": round(len(a) / model.sample_rate / dt, 2)}
            print(voice, sid, timings[f"{voice}_{sid}"], flush=True)
    json.dump({"threads": args.threads, "clips": timings}, open(os.path.join(out, "timings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
