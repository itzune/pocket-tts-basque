#!/usr/bin/env python3
"""Pronunciation probe of the data teacher (OmniVoice) on Basque phenomena the ear flagged:
palatals (tt, dd, ll, ñ), the g/j distinction (j = /x/ or /j/, g before e/i = /g/), and sibilants.

--stage jobs:  pick sentences from a pool containing target words, 4 named prompts -> jobs.jsonl
--stage score: for each generated wav, ASR (Whisper-eu + optional Conformer hyps) and check whether the
               target word is read correctly (exact match of the normalised word in the hypothesis, or
               the closest hypothesis word by edit distance); per-phenomenon error rates + examples.

  python scripts/small/pron_probe.py --stage jobs --pool /root/data/text/pool_v2.jsonl --named /root/data/named/selected.jsonl --out /root/probe
  omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list /root/probe/jobs.jsonl --res_dir /root/probe/wav --batch_size 16 --num_step 32 --guidance_scale 2.0 --enable_flashinfer true
  python scripts/small/pron_probe.py --stage score --out /root/probe
"""
import argparse, collections, difflib, glob, json, os, random, re, sys
sys.path.insert(0, os.path.expanduser("~/bssd"))

PHEN = {
    "tt": r"\b\w*[aeiou]tt[aeiou]\w*\b",
    "dd": r"\b\w*[aeiou]dd[aeiou]\w*\b",
    "ll": r"\b\w*[aeiou]ll[aeiou]\w*\b",
    "ñ": r"\b\w*ñ\w*\b",
    "j_initial": r"\bj[aeiou]\w*\b",
    "j_medial": r"\b\w+[aeiou]j[aeiou]\w*\b",
    "ge_gi": r"\b\w*g[ei]\w*\b",
    "x": r"\b\w*[aeiou]x[aeiou]\w*\b",
    "tz_ts_tx": r"\b\w*(tz|ts|tx)[aeiou]\w*\b",
}
_P = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s): return re.sub(r"\s+", " ", _P.sub(" ", s.lower())).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["jobs", "score"], required=True)
    ap.add_argument("--pool"); ap.add_argument("--named"); ap.add_argument("--out", required=True); ap.add_argument("--per", type=int, default=40)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.stage == "jobs":
        rng = random.Random(11)
        rows = [json.loads(l) for l in open(a.pool)]
        rng.shuffle(rows)
        named = {json.loads(l)["voice_id"]: json.loads(l) for l in open(a.named)}
        voices = [named[v] for v in ("Maider", "Antton", "Nahia", "Aitor")]
        jobs, used = [], set()
        for ph, rx in PHEN.items():
            n = 0
            for r in rows:
                t = r["text_norm"]
                if r["id"] in used or not (5 <= len(t.split()) <= 18):
                    continue
                m = re.search(rx, t.lower())
                if not m:
                    continue
                used.add(r["id"]); n += 1
                v = voices[n % 4]
                jobs.append({"id": f"{ph}_{n:03d}", "text": t, "ref_audio": v["path"], "ref_text": v["text"], "language_id": "eu",
                             "phenomenon": ph, "target": m.group(0), "voice": v["voice_id"]})
                if n >= a.per:
                    break
        with open(os.path.join(a.out, "jobs.jsonl"), "w") as f:
            for j in jobs:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        print("jobs", len(jobs), collections.Counter(j["phenomenon"] for j in jobs))
        return
    # score
    import librosa, soundfile as sf
    from bssd.scoring import BasqueASR
    asr = BasqueASR()
    jobs = {j["id"]: j for j in map(json.loads, open(os.path.join(a.out, "jobs.jsonl")))}
    wavs = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(a.out, "wav", "**", "*.wav"), recursive=True)}
    res = collections.defaultdict(list); out = open(os.path.join(a.out, "probe_scores.jsonl"), "w")
    for jid, j in jobs.items():
        p = wavs.get(jid)
        if not p:
            continue
        au, sr = sf.read(p, dtype="float32"); au = au.mean(1) if au.ndim > 1 else au
        if sr != 16000:
            au = librosa.resample(au, orig_sr=sr, target_sr=16000)
        hyp = asr.transcribe(au)
        tw = norm(j["target"]); hw = norm(hyp).split()
        exact = tw in hw
        close = difflib.get_close_matches(tw, hw, n=1, cutoff=0.0)
        closest = close[0] if close else ""
        ratio = difflib.SequenceMatcher(a=tw, b=closest).ratio() if closest else 0.0
        rec = {"id": jid, "phenomenon": j["phenomenon"], "voice": j["voice"], "target": j["target"], "closest": closest, "exact": exact, "ratio": round(ratio, 3), "text": j["text"], "hyp": hyp}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); res[j["phenomenon"]].append(rec)
    print(f"{'phenomenon':10s} n   target word exact   mean ratio   examples of misses")
    for ph, rs in res.items():
        miss = [r for r in rs if not r["exact"]]
        ex = "; ".join(f"{r['target']}→{r['closest']}" for r in miss[:4])
        print(f"{ph:10s} {len(rs):3d} {sum(r['exact'] for r in rs)/len(rs):18.2%} {sum(r['ratio'] for r in rs)/len(rs):12.3f}   {ex}")


if __name__ == "__main__":
    main()
