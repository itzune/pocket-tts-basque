#!/usr/bin/env python3
"""Pocket TTS manifest of the KEPT rows of BSSD v2 generation runs, straight from the pipeline folders
(no parquet): for each run NN, jobs_vNN/meta.jsonl (text_norm), flac_vNN/shard_*/post.jsonl (duration) and
scores_vNN/*.scores.jsonl (status after reflag --merge-secondary). Rows whose transcript contains a palatal
are dropped as well (belt and braces; job creation already skipped them).

  python scripts/v2_manifest_from_scores.py --data /root/data --runs v12,v13,v14,v15 --out /root/data/bssd_v2/synth_vm1.jsonl
"""
import argparse, collections, glob, json, os, re
PALATAL = re.compile(r"[a-zñ](tt|dd|ll)[aeiouáéíóú]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/root/data"); ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--status", default="keep"); a = ap.parse_args()
    ok = set(a.status.split(",")); tot = collections.Counter(); hours = 0.0; n = 0
    with open(a.out, "w") as out:
        for run in a.runs.split(","):
            meta = {m["id"]: m for m in map(json.loads, open(os.path.join(a.data, f"jobs_{run}", "meta.jsonl")))}
            post = {}
            for p in glob.glob(os.path.join(a.data, f"flac_{run}", "shard_*", "post.jsonl")):
                for r in map(json.loads, open(p)):
                    post[r["id"]] = r
            for sp in glob.glob(os.path.join(a.data, f"scores_{run}", "*.scores.jsonl")):
                for s in map(json.loads, open(sp)):
                    tot[(run, s["status"])] += 1
                    if s["status"] not in ok:
                        continue
                    m, p = meta.get(s["id"]), post.get(s["id"])
                    if not m or not p:
                        tot[(run, "missing")] += 1; continue
                    if PALATAL.search(m["text_norm"].lower()):
                        tot[(run, "palatal_dropped")] += 1; continue
                    path = os.path.join(a.data, f"flac_{run}", f"shard_{m['shard']:04d}", s["id"] + ".flac")
                    out.write(json.dumps({"path": path, "duration": p["duration"], "transcript": m["text_norm"]}, ensure_ascii=False) + "\n")
                    hours += p["duration"] / 3600; n += 1
    print({f"{k[0]}:{k[1]}": v for k, v in sorted(tot.items())}); print(f"wrote {n} rows, {hours:.1f} h -> {a.out}")


if __name__ == "__main__":
    main()
