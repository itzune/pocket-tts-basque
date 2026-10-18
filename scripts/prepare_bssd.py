#!/usr/bin/env python3
"""Pull itzune/bssd from the Hub and lay it out for Pocket TTS training.

Keeps the FLAC bytes as files (Pocket TTS reads audio with sphn, which decodes FLAC), writes one
manifest per split with {path, duration, transcript, voice_id, voice_mode, text_src} where
transcript = text_norm (verbalised). Downloads one parquet file at a time and deletes it after
extraction, so peak disk use is the audio plus one file. Resumable (skips files already extracted).

  python scripts/prepare_bssd.py --out /root/data/bssd [--splits train,dev,test] [--prefix vm1,vm2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "itzune/bssd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/data/bssd")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--prefix", default="", help="comma-separated batch prefixes to include (a,b); empty = all")
    ap.add_argument("--cache", default="/root/data/bssd/_dl")
    args = ap.parse_args()
    splits = args.splits.split(",")
    prefixes = {p for p in args.prefix.split(",") if p}
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))
    sel = []
    for f in files:
        name = os.path.basename(f)
        sp, pfx = name.split("-")[0], name.split("-")[1]
        if sp in splits and (not prefixes or pfx in prefixes):
            sel.append((f, sp))
    print(f"{len(sel)} parquet files to process", flush=True)
    os.makedirs(args.cache, exist_ok=True)
    done_path = os.path.join(args.out, "_done_files.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    manifests = {sp: open(os.path.join(args.out, f"{sp}.jsonl"), "a") for sp in splits}
    t0, n_rows = time.time(), 0
    for i, (f, sp) in enumerate(sel):
        if f in done:
            continue
        local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.cache)
        os.makedirs(os.path.join(args.out, sp), exist_ok=True)
        t = pq.read_table(local, columns=["id", "audio", "text_norm", "duration", "voice_id", "voice_mode", "text_src"])
        for uid, audio, text, dur, vid, vmode, src in zip(*(t.column(c).to_pylist() for c in ("id", "audio", "text_norm", "duration", "voice_id", "voice_mode", "text_src"))):
            path = os.path.join(args.out, sp, uid + ".flac")
            if not os.path.exists(path):
                with open(path, "wb") as fh:
                    fh.write(audio["bytes"])
            manifests[sp].write(json.dumps({"path": path, "duration": float(dur), "transcript": text, "voice_id": vid,
                                            "voice_mode": vmode, "text_src": src}, ensure_ascii=False) + "\n")
            n_rows += 1
        manifests[sp].flush()
        os.remove(local)
        with open(done_path, "a") as fh:
            fh.write(f + "\n")
        print(f"{i+1}/{len(sel)} {f} rows so far {n_rows} ({time.time()-t0:.0f}s)", flush=True)
    for m in manifests.values():
        m.close()
    for sp in splits:
        p = os.path.join(args.out, f"{sp}.jsonl")
        rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        print(f"{sp}: {len(rows)} rows, {sum(r['duration'] for r in rows)/3600:.1f} h")


if __name__ == "__main__":
    main()
