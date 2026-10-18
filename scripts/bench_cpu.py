#!/usr/bin/env python3
"""CPU benchmark: real-time factor and first-audio latency at a given thread count.

  python scripts/bench_cpu.py --config configs/basque_infer.yaml [--checkpoint ...] --voice samples/Maider.wav --threads 2
"""
import argparse, json, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint")
ap.add_argument("--voice", required=True)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--quantize", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
from pocket_tts import TTSModel

kw = {"quantize": True} if a.quantize else {}
m = TTSModel.load_model(config=a.config, checkpoint=a.checkpoint, **kw)
st = m.get_state_for_audio_prompt(a.voice)
texts = ["Kaixo, nola zaude gaur?", "Euskara munduko hizkuntzarik zaharrenetarikoa da, eta Euskal Herrian hitz egiten da.",
         "Bihar goizeko hamarretan hasiko da bilera, Donostiako udaletxean, eta arratsaldean mendira joango gara denok elkarrekin."]
res = []
for t in texts:
    t0 = time.time(); au = m.generate_audio(st, t); dt = time.time() - t0
    dur = au.shape[-1] / m.sample_rate
    res.append({"chars": len(t), "audio_s": round(dur, 2), "gen_s": round(dt, 2), "x_realtime": round(dur / dt, 2)})
    print(json.dumps(res[-1]))
print("threads", a.threads, "quantize", a.quantize, "mean x realtime", round(sum(r["x_realtime"] for r in res) / len(res), 2))
