#!/usr/bin/env python3
"""ONNX CPU benchmark (no PyTorch): first-audio latency and real-time factor of the streaming runtime.

Runs the reference runtime shipped in the release folder (pocket_tts_onnx.py) on the exported bundle; needs
only onnxruntime, numpy, sentencepiece, safetensors and huggingface_hub.

  python scripts/bench_onnx.py --bundle /tmp/pocket-onnx-export/release --voice voices/Maider.safetensors --threads 2
"""
import argparse, json, os, sys, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True, help="release folder (bundle.json + pocket_tts_onnx.py)")
ap.add_argument("--voice", required=True, help=".wav to clone, or a voices/*.safetensors state (relative to --bundle)")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
a = ap.parse_args()

sys.path.insert(0, a.bundle)
import onnxruntime as ort
import pocket_tts_onnx as P


def session_options(_self):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.inter_op_num_threads = 1
    return opts


P.PocketTTSOnnx._make_session_options = session_options
voice = a.voice if os.path.exists(a.voice) else os.path.join(a.bundle, a.voice)
TEXTS = [
    "Kaixo, zer moduz?",
    "Donostian hamahiru gradu eta zirimiria, hezetasuna ehuneko hirurogeita hamasei, haizea sei koma zazpi kilometro orduko.",
    "Nahi al duzu hurrengo egunetako iragarpena?",
]

t = time.monotonic()
tts = P.PocketTTSOnnx(models_dir=a.bundle, precision=a.precision, device="cpu", temperature=0.3, lsd_steps=1)
load_s = time.monotonic() - t
tts.prepare_voice_state(voice)
list(tts.stream("Kaixo.", voice))  # warm-up

rows = []
for text in TEXTS:
    t = time.monotonic(); first = None; chunks = []
    for chunk in tts.stream(text, voice):
        first = first or time.monotonic() - t
        chunks.append(chunk)
    total = time.monotonic() - t
    dur = sum(len(c) for c in chunks) / tts.sample_rate
    rows.append({"audio_s": round(dur, 2), "first_audio_s": round(first, 3), "rtf_x": round(dur / total, 2)})
print(json.dumps({"precision": a.precision, "threads": a.threads, "load_s": round(load_s, 2),
                  "first_audio_s": max(r["first_audio_s"] for r in rows),
                  "realtime_x": round(sum(r["audio_s"] for r in rows) / sum(r["audio_s"] / r["rtf_x"] for r in rows), 2),
                  "sentences": rows}, indent=1))
