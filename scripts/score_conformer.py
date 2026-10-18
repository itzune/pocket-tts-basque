#!/usr/bin/env python3
"""Secondary ASR pass over an eval_basque.py output folder with HiTZ/stt_eu_conformer_transducer_large_v2.

Runs in the NeMo venv (no bssd imports). Reads <dir>/eval.jsonl, transcribes every wav (resampled to
16 kHz), writes <dir>/eval_conformer.jsonl with {wav, hyp_conformer, wer_conformer, cer_conformer} and adds
*_conformer aggregates (+ fail_rate_conformer) to <dir>/summary.json.

  /root/nemo-venv/bin/python scripts/score_conformer.py --dir /root/runs/basque_24l/eval_final/step60000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

import numpy as np

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s.lower())).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edit_distance(r, h) / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="HiTZ/stt_eu_conformer_transducer_large_v2")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    import librosa
    import soundfile as sf
    import nemo.collections.asr as nemo_asr

    rows = [json.loads(l) for l in open(os.path.join(args.dir, "eval.jsonl"))]
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()
    tmp = tempfile.mkdtemp()
    out_rows = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        wavs = []
        for r in batch:
            a, sr = sf.read(r["wav"], dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            if sr != 16000:
                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            if len(a) < 1600:
                a = np.pad(a, (0, 1600 - len(a)))
            p = os.path.join(tmp, os.path.basename(r["wav"]))
            sf.write(p, a, 16000)
            wavs.append(p)
        hyps = model.transcribe(wavs, batch_size=len(wavs), verbose=False)
        if hyps and not isinstance(hyps[0], str):
            hyps = [h.text if hasattr(h, "text") else str(h) for h in hyps]
        for r, h in zip(batch, hyps):
            out_rows.append({"wav": r["wav"], "voice_id": r["voice_id"], "kind": r["kind"], "hyp_conformer": h,
                             "wer_conformer": round(wer(r["text"], h), 4), "cer_conformer": round(cer(r["text"], h), 4)})
        for p in wavs:
            os.remove(p)
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}", flush=True)
    with open(os.path.join(args.dir, "eval_conformer.jsonl"), "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    sp = os.path.join(args.dir, "summary.json")
    summ = json.load(open(sp))
    for kind in ("named", "held_out"):
        rs = [r for r in out_rows if r["kind"] == kind]
        if rs and kind in summ["summary"]:
            ws = [r["wer_conformer"] for r in rs]
            summ["summary"][kind] |= {"wer_conformer": round(float(np.mean(ws)), 4), "cer_conformer": round(float(np.mean([r["cer_conformer"] for r in rs])), 4),
                                      "fail_rate_conformer": round(sum(w > 0.5 for w in ws) / len(ws), 4)}
    for v, d in summ.get("per_voice", {}).items():
        rs = [r for r in out_rows if r["voice_id"] == v]
        if rs:
            d["wer_conformer"] = round(float(np.mean([r["wer_conformer"] for r in rs])), 3)
            d["fails_conformer"] = sum(r["wer_conformer"] > 0.5 for r in rs)
    json.dump(summ, open(sp, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ["summary"], indent=1))


if __name__ == "__main__":
    main()
