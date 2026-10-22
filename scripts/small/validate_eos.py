#!/usr/bin/env python3
"""Validate the inference eos_threshold of a student on sentences NOT used to pick it:
dev-split sentences (sweep), a second sampling seed on test sentences, and long multi-sentence texts
(runaway check). Writes <out>/validate_eos.json and prints a summary.

  python scripts/small/validate_eos.py --config configs/basque_18m_infer.yaml --checkpoint ... --out /root/runs/basque_18m/eval_validate
"""
import argparse, json, os, random, sys
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.expanduser("~/bssd"))
from pocket_tts import TTSModel
from bssd.scoring import BasqueASR, wer


def prompts():
    p = {json.loads(l)["voice_id"]: json.loads(l)["path"] for l in open("/root/data/named/selected.jsonl")}
    test = [json.loads(l) for l in open("/root/data/bssd/test.jsonl")]
    for vid in "bank0027,bank0083,bank0070,euf_05679,eum_08886".split(","):
        p[vid] = [r for r in test if r["voice_id"] == vid and 4 <= r["duration"] <= 9][0]["path"]
    return p, test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    P, test = prompts(); asr = BasqueASR(); res = {}
    dev = [json.loads(l)["transcript"] for l in open("/root/data/bssd/dev.jsonl")]
    rng = random.Random(1); rng.shuffle(dev); dev_s = dev[: a.n]
    tsent = [r["transcript"] for r in test]; rng0 = random.Random(0); rng0.shuffle(tsent); test_s = tsent[: a.n]
    long_texts = [" ".join(dev[100 + i * 4: 100 + i * 4 + 4]) for i in range(6)]

    def run(m, voices, sents, seed, tag):
        ws, durs, exp = [], [], []
        states = {v: m.get_state_for_audio_prompt(P[v]) for v in voices}
        for v in voices:
            for i, s in enumerate(sents):
                torch.manual_seed(seed * 100003 + i)
                au = m.generate_audio(states[v], s).detach().cpu().numpy().squeeze()
                h = asr.transcribe(librosa.resample(au, orig_sr=m.sample_rate, target_sr=16000))
                ws.append(wer(s, h)); durs.append(len(au) / m.sample_rate); exp.append(len(s.split()) * 0.42)
        r = {"n": len(ws), "wer": float(np.mean(ws)), "fails": int(sum(w > 0.5 for w in ws)), "median": float(np.median(ws)),
             "max_dur": float(max(durs)), "dur_over_expected": float(np.mean(np.array(durs) / np.array(exp))),
             "runaway": int(sum(d > 2.0 * e + 2 for d, e in zip(durs, exp)))}
        print(tag, json.dumps(r), flush=True); return r

    all_v = list(P)
    for thr in (-4.0, 0.0, 2.0):
        m = TTSModel.load_model(config=a.config, checkpoint=a.checkpoint, eos_threshold=thr).to("cuda")
        res[f"dev_seed1_eos{thr:+.0f}"] = run(m, all_v, dev_s, 1, f"dev split, seed 1, eos {thr:+.0f}:")
        if thr == 0.0:
            res["test_seed1_eos+0"] = run(m, all_v, test_s, 1, "test sentences, seed 1, eos +0:")
        if thr in (0.0, 2.0):
            res[f"long_eos{thr:+.0f}"] = run(m, ["Maider", "Antton"], long_texts, 0, f"long 4-sentence texts, eos {thr:+.0f}:")
    json.dump(res, open(os.path.join(a.out, "validate_eos.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
