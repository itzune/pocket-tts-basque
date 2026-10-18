#!/usr/bin/env python3
"""Sanity-check a sample of forced alignments: coverage, word count match, monotonicity,
word durations, start/end slack. Prints a summary and the worst cases.

  python scripts/validate_alignment.py /root/data/bssd/dev_aligned.jsonl [--n 300]
"""
import argparse, json, random, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("aligned")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.aligned)]
rng = random.Random(0)
sample = rng.sample(rows, min(a.n, len(rows)))
n_words_ok = n_mono = 0
durs, starts, ends, bad = [], [], [], []
for r in sample:
    words = [(w["word"], w["start"], w["end"]) for w in (r.get("words") or [])]
    toks = r["transcript"].split()
    if len(words) == len(toks):
        n_words_ok += 1
    mono = all(words[i][2] <= words[i + 1][1] + 1e-3 for i in range(len(words) - 1)) if words else False
    n_mono += mono
    if words:
        durs += [w[2] - w[1] for w in words]
        starts.append(words[0][1]); ends.append(r["duration"] - words[-1][2])
        if not mono or len(words) != len(toks) or max(w[2] - w[1] for w in words) > 2.5:
            bad.append((r["path"], len(toks), len(words), round(max(w[2] - w[1] for w in words), 2)))
    else:
        bad.append((r["path"], len(toks), 0, None))
print(f"sampled {len(sample)}; with words: {sum(1 for r in sample if r.get('words'))}; word-count match: {n_words_ok}; monotonic: {n_mono}")
if durs:
    q = lambda v, p: round(sorted(v)[int(p * (len(v) - 1))], 3)
    print(f"word duration p50 {q(durs,.5)} p95 {q(durs,.95)} max {max(durs):.2f}s | first-word start p50 {q(starts,.5)} p95 {q(starts,.95)} | tail slack p50 {q(ends,.5)} p95 {q(ends,.95)}")
print("worst cases:", bad[:10])
if rows and rows[0].get("words"):
    print("example:", rows[0]["transcript"][:80], rows[0]["words"][:3])
