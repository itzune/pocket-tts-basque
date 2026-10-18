#!/usr/bin/env python3
"""Evaluate a Basque Pocket TTS checkpoint on BSSD test sentences.

For each prompt voice (named voices + held-out voices), synthesise N unseen test sentences, then
score: WER/CER with the Basque Whisper (faster-whisper CT2), optional Conformer CER via a jsonl
produced separately, WavLM speaker similarity to the prompt, UTMOS, and generation RTF.

  python scripts/eval_basque.py --config configs/basque_24l_infer.yaml --checkpoint /root/runs/basque_24l/model.safetensors \
      --test /root/data/bssd/test.jsonl --named /root/data/named/selected.jsonl --out reports/eval_24l --n 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for cand in (Path.home() / "bssd", Path(__file__).resolve().parents[2] / "itzune" / "bssd"):
    if (cand / "bssd").is_dir():
        sys.path.insert(0, str(cand))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--test", required=True, help="BSSD test manifest (path, duration, transcript, voice_id)")
    ap.add_argument("--named", help="named/selected.jsonl (voice_id, path, text)")
    ap.add_argument("--held-out", default="bank0027,bank0083,bank0070,euf_05679,eum_08886")
    ap.add_argument("--n", type=int, default=40, help="sentences per voice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="fixed sampling seed per (voice, sentence) so checkpoints are comparable")
    args = ap.parse_args()
    import torch
    from pocket_tts import TTSModel

    from bssd.scoring import UTMOS, BasqueASR, SpeakerEmbedder, cer, cosine, wer
    from bssd.audio import load
    import librosa

    rng = random.Random(0)
    test = [json.loads(l) for l in open(args.test)]
    sents = [r["transcript"] for r in test]
    rng.shuffle(sents)
    # prompts: named voices (clip + text) and one clip per held-out voice from the test manifest
    prompts = []
    if args.named:
        for l in open(args.named):
            v = json.loads(l)
            prompts.append((v["voice_id"], v["path"], "named"))
    for vid in args.held_out.split(","):
        rows = [r for r in test if r["voice_id"] == vid and 4 <= r["duration"] <= 9]
        if rows:
            prompts.append((vid, rows[0]["path"], "held_out"))
    model = TTSModel.load_model(config=args.config, checkpoint=args.checkpoint)
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    asr, utmos, spk = BasqueASR(), UTMOS(args.device), SpeakerEmbedder(args.device)
    os.makedirs(args.out, exist_ok=True)
    rows_out = []
    for pi, (vid, ppath, kind) in enumerate(prompts):
        state = model.get_state_for_audio_prompt(ppath)
        pa, _ = load(ppath, 16000)
        pemb = spk.embed(pa)
        mine = [s for s in sents if s][: args.n]
        for k, text in enumerate(mine):
            torch.manual_seed(args.seed * 1_000_003 + pi * 1000 + k)
            t0 = time.time()
            audio = model.generate_audio(state, text, temp=args.temp, cfg_coef=args.cfg) if "temp" in model.generate_audio.__code__.co_varnames else model.generate_audio(state, text)
            dt = time.time() - t0
            a = audio.detach().cpu().numpy().astype(np.float32).squeeze()
            sr = model.sample_rate
            wav = os.path.join(args.out, f"{vid}_{k:03d}.wav")
            sf.write(wav, a, sr)
            a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
            hyp = asr.transcribe(a16)
            rows_out.append({"voice_id": vid, "kind": kind, "text": text, "hyp": hyp, "wer": round(wer(text, hyp), 4), "cer": round(cer(text, hyp), 4),
                             "utmos": round(utmos.score(a16), 3), "spk_sim": round(cosine(pemb, spk.embed(a16)), 3),
                             "duration": round(len(a) / sr, 2), "gen_s": round(dt, 2), "rtf": round(dt / max(0.1, len(a) / sr), 3), "wav": wav})
        print(vid, kind, "done", flush=True)
    with open(os.path.join(args.out, "eval.jsonl"), "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {}
    for kind in ("named", "held_out"):
        rs = [r for r in rows_out if r["kind"] == kind]
        if rs:
            summ[kind] = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in ("wer", "cer", "utmos", "spk_sim", "rtf")} | {"n": len(rs)}
            # the mean WER is dominated by catastrophic outputs (prompt continuation / hallucination);
            # report them separately: failure rate (WER > 0.5), median, and the mean over non-failed sentences
            ws = [r["wer"] for r in rs]
            ok = [w for w in ws if w <= 0.5]
            summ[kind] |= {"fail_rate": round(sum(w > 0.5 for w in ws) / len(ws), 4), "wer_median": round(float(np.median(ws)), 4),
                           "wer_ok": round(float(np.mean(ok)), 4) if ok else None}
    per_voice = {v: {k: round(float(np.mean([r[k] for r in rows_out if r["voice_id"] == v])), 3) for k in ("wer", "cer", "utmos", "spk_sim")}
                 | {"fails": sum(r["wer"] > 0.5 for r in rows_out if r["voice_id"] == v)} for v in {r["voice_id"] for r in rows_out}}
    json.dump({"summary": summ, "per_voice": per_voice, "config": args.config, "checkpoint": args.checkpoint}, open(os.path.join(args.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
