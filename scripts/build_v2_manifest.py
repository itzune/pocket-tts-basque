#!/usr/bin/env python3
"""Assemble the BSSD v2 training manifest for Pocket TTS (TRAINING_PLAN_V2.md, stage 1 → 2).

Inputs are Pocket TTS manifests ({path, duration, transcript} per line):
  --v1        v1 train manifest (synthetic; rows whose transcript contains a palatal are dropped, the
              data teacher mispronounces them)
  --v2        v2 synthetic manifests from package.py (already palatal-free at job creation)
  --real      real-speech manifest from package_real.py. Decision 2026-10-22 (user, by ear): the studio
              recordings pronounce the palatals correctly but have flat read-speech prosody, so real
              speech is used ONLY for what it is needed for: by default just the HiTZ rows containing a
              palatal (--real-sources hitz, --real-repeat 0), each repeated --palatal-repeat times.
Writes <out>/train.jsonl (shuffled, seed 0) and <out>/stats.json. dev/test stay v1's.

  python scripts/build_v2_manifest.py --v1 /root/data/bssd/train.jsonl --v2 /root/data/bssd_v2/synth/manifest-*.jsonl \
      --real /root/data/bssd_v2/real/pocket_tts_manifest-train-real.jsonl --out /root/data/bssd_v2
"""
import argparse, glob, json, os, random, re

PALATAL = re.compile(r"[a-zñ](tt|dd|ll)[aeiouáéíóú]")


def read(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True); ap.add_argument("--v2", nargs="*", default=[]); ap.add_argument("--real", required=True)
    ap.add_argument("--real-repeat", type=int, default=0, help="repeats for real rows WITHOUT palatals (0 = only palatal rows are used)")
    ap.add_argument("--palatal-repeat", type=int, default=2)
    ap.add_argument("--real-sources", default="hitz", help="comma-separated text_src values of real rows to use (e.g. hitz,slr76)")
    ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    stats = {}
    v1 = read(a.v1); keep = [r for r in v1 if not PALATAL.search(r["transcript"].lower())]
    stats["v1"] = {"rows": len(v1), "kept": len(keep), "dropped_palatal": len(v1) - len(keep), "hours": round(sum(r["duration"] for r in keep) / 3600, 1)}
    rows = [dict(r, src="v1") for r in keep]
    v2 = [r for pat in a.v2 for p in sorted(glob.glob(pat)) for r in read(p)]
    stats["v2"] = {"rows": len(v2), "hours": round(sum(r["duration"] for r in v2) / 3600, 1)}
    rows += [dict(r, src="v2") for r in v2]
    srcs = {x.strip() for x in a.real_sources.split(",") if x.strip()}
    real = [r for r in read(a.real) if (r.get("text_src") or ("hitz" if "hitz_" in r["path"] else "slr76")) in srcs]
    n_pal = 0
    for r in real:
        rep = a.real_repeat
        if PALATAL.search(r["transcript"].lower()):
            rep = a.palatal_repeat; n_pal += 1
        rows += [dict(r, src="real")] * rep
    stats["real"] = {"rows": len(real), "palatal_rows": n_pal, "hours": round(sum(r["duration"] for r in real) / 3600, 1),
                     "repeat": a.real_repeat, "palatal_repeat": a.palatal_repeat}
    random.Random(0).shuffle(rows)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"path": r["path"], "duration": r["duration"], "transcript": r["transcript"]}, ensure_ascii=False) + "\n")
    stats["train"] = {"rows": len(rows), "hours_with_repeats": round(sum(r["duration"] for r in rows) / 3600, 1)}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
