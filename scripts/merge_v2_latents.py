#!/usr/bin/env python3
"""Merge precomputed-latents manifests of the BSSD v2 parts into one training manifest, without moving
any audio. Each part is a `<stem>_latents.jsonl` produced by training/scripts/precompute_latents.py; its
`latents_file` entries are relative to the part's own directory, and all parts must live under --out
(same `latents/<mimi-hash>/` store, files are named by part stem so they never collide).

  python scripts/merge_v2_latents.py --out /root/data/bssd_v2 --name train_v2_aligned \
      --part /root/data/bssd/train_aligned_latents.jsonl:1:drop_palatal \
      --part /root/data/bssd_v2/synth_vm1_aligned_latents.jsonl:1 \
      --part /root/data/bssd_v2/synth_vm2_aligned_latents.jsonl:1 \
      --part /root/data/bssd_v2/real_pal_aligned_latents.jsonl:3
A part spec is path:repeat[:drop_palatal]. Parts outside --out get their latents_file rewritten to a
relative path from --out (e.g. ../bssd/latents/...).
"""
import argparse, json, os, random, re
PALATAL = re.compile(r"[a-zñ](tt|dd|ll)[aeiouáéíóú]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--name", default="train_v2_aligned")
    ap.add_argument("--part", action="append", required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    out = os.path.abspath(a.out); rows = []; metas = []; stats = {}
    for spec in a.part:
        bits = spec.split(":"); path = os.path.abspath(bits[0]); rep = int(bits[1]) if len(bits) > 1 else 1; drop = len(bits) > 2 and bits[2] == "drop_palatal"
        pdir = os.path.dirname(path); meta = json.load(open(path.replace("_latents.jsonl", "_latents.meta.json"))); metas.append(meta)
        n = kept = missing = 0; hours = 0.0
        for line in open(path):
            r = json.loads(line); n += 1
            if drop and PALATAL.search(r["transcript"].lower()):
                continue
            lf = os.path.join(pdir, r["latents_file"])
            if not os.path.exists(lf):
                missing += 1; continue
            r["latents_file"] = os.path.relpath(lf, out); kept += 1; hours += r["duration"] / 3600
            rows += [r] * rep
        stats[os.path.basename(path)] = {"rows": n, "kept": kept, "missing_latents": missing, "repeat": rep, "hours": round(hours, 1)}
    hashes = {m["mimi_hash"] for m in metas}; stitch = {m["stitch_frames"] for m in metas}
    assert len(hashes) == 1 and len(stitch) == 1, f"parts disagree: hashes {hashes} stitch {stitch}"
    random.Random(a.seed).shuffle(rows)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, a.name + "_latents.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, a.name + ".jsonl"), "w") as f:  # audio manifest, only consulted if latents go stale
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "latents_file"}, ensure_ascii=False) + "\n")
    json.dump(metas[0], open(os.path.join(out, a.name + "_latents.meta.json"), "w"), indent=2)
    stats["merged"] = {"rows": len(rows), "hours_with_repeats": round(sum(r["duration"] for r in rows) / 3600, 1)}
    json.dump(stats, open(os.path.join(out, a.name + "_stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
