#!/usr/bin/env python3
"""Paired comparison of two paired_eval.py outputs (A vs B) with bootstrap CIs and failure classes.

  python scripts/eval/paired_report.py --a /root/evals/basque_6l --b /root/evals/basque_small --names 6L small
"""
import argparse, collections, json, os, re
import numpy as np

_P = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s): return re.sub(r"\s+", " ", _P.sub(" ", s.lower())).strip()


def load(d):
    rows = {}
    for r in map(json.loads, open(os.path.join(d, "eval.jsonl"))):
        rows[(r["voice_id"], r["set"], r["idx"], r["seed"])] = r
    conf = os.path.join(d, "eval_conformer.jsonl")
    if os.path.exists(conf):
        by_wav = {r["wav"]: r for r in map(json.loads, open(conf))}
        for r in rows.values():
            c = by_wav.get(r["wav"])
            if c:
                r["wer_conformer"] = c["wer_conformer"]; r["cer_conformer"] = c["cer_conformer"]
    return rows


def fail_class(r):
    if r["wer"] <= 0.5:
        return None
    exp = len(r["text"].split()) * 0.42
    pw = set(norm(r.get("prompt_text") or "").split()); hw = set(norm(r["hyp"]).split())
    if pw and len(pw & hw) >= max(2, len(hw) // 2):
        return "prompt_continuation"
    if r["duration"] < 0.6 * exp:
        return "early_stop"
    return "other"


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--names", nargs=2, default=["A", "B"]); ap.add_argument("--out")
    a = ap.parse_args(); A, B = load(a.a), load(a.b); keys = sorted(set(A) & set(B))
    na, nb = a.names
    lines = [f"# Paired evaluation: {na} vs {nb}", f"{len(keys)} paired utterances ({len(set(A)^set(B))} unmatched dropped)", ""]
    metrics = ["wer", "cer", "wer_conformer", "cer_conformer", "utmos", "spk_sim"]
    for scope, sel in (("all", keys), ("main set", [k for k in keys if k[1] == "main"]), ("hard set", [k for k in keys if k[1] == "hard"]),
                       ("named voices", [k for k in keys if A[k]["kind"] == "named"]), ("held-out voices", [k for k in keys if A[k]["kind"] == "held_out"])):
        if not sel:
            continue
        lines += [f"## {scope} (n={len(sel)})", "", f"| metric | {na} | {nb} | diff ({nb}−{na}) | 95 % CI | {nb} better / tie / worse |", "|---|---|---|---|---|---|"]
        for m in metrics:
            if any(m not in A[k] or m not in B[k] for k in sel):
                continue
            x = np.array([A[k][m] for k in sel]); y = np.array([B[k][m] for k in sel]); d = y - x
            mean, lo, hi = boot_ci(d)
            better = (d < 0).sum() if m not in ("utmos", "spk_sim") else (d > 0).sum()
            worse = (d > 0).sum() if m not in ("utmos", "spk_sim") else (d < 0).sum()
            flag = " **" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {m} | {x.mean():.4f} | {y.mean():.4f} | {mean:+.4f}{flag} | [{lo:+.4f}, {hi:+.4f}] | {better} / {len(sel)-better-worse} / {worse} |")
        fa = collections.Counter(fail_class(A[k]) for k in sel); fb = collections.Counter(fail_class(B[k]) for k in sel)
        lines += ["", f"failures (WER > 0.5): {na} {sum(v for c,v in fa.items() if c)} ({dict((c,v) for c,v in fa.items() if c)}), "
                  f"{nb} {sum(v for c,v in fb.items() if c)} ({dict((c,v) for c,v in fb.items() if c)})", ""]
    tagged = [k for k in keys if A[k].get("tag")]
    if tagged:
        lines += ["## hard set by tag (mean CER)", "", f"| tag | n | {na} | {nb} |", "|---|---|---|---|"]
        for t in sorted({A[k]["tag"] for k in tagged}):
            ks = [k for k in tagged if A[k]["tag"] == t]
            lines.append(f"| {t} | {len(ks)} | {np.mean([A[k]['cer'] for k in ks]):.4f} | {np.mean([B[k]['cer'] for k in ks]):.4f} |")
    lines += ["", "## worst regressions (by CER increase)", ""]
    for k in sorted(keys, key=lambda k: B[k]["cer"] - A[k]["cer"], reverse=True)[:12]:
        lines.append(f"- {k[0]} [{k[1]}{k[2]}] CER {A[k]['cer']:.2f}→{B[k]['cer']:.2f}: {B[k]['text'][:70]} | {nb}: {B[k]['hyp'][:70]}")
    out = "\n".join(lines)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
