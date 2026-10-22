#!/usr/bin/env python3
"""Freeze the evaluation sentence sets for the paired protocol (TRAINING_PLAN_V2.md, stage 0).

  main: N sentences sampled from the test split (seed 7), stratified by length.
  hard: sentences from the dev split that stress what the ear catches: rare SentencePiece pieces,
        proper names inside the sentence, numbers written in words, long sentences, questions.

  python scripts/eval/make_eval_sets.py --test /root/data/bssd/test.jsonl --dev /root/data/bssd/dev.jsonl \
      --train /root/data/bssd/train.jsonl --tokenizer /root/data/bssd/tokenizer/basque.model --out reports/eval_sets
"""
import argparse, collections, json, os, random, re
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-main", type=int, default=400); ap.add_argument("--n-hard", type=int, default=210)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    rng = random.Random(7)
    test = [json.loads(l)["transcript"] for l in open(a.test)]
    test = sorted(set(test)); rng.shuffle(test)
    short = [s for s in test if len(s.split()) <= 7]; mid = [s for s in test if 8 <= len(s.split()) <= 15]; long_ = [s for s in test if len(s.split()) > 15]
    main_set = short[: a.n_main // 4] + mid[: a.n_main // 2] + long_[: a.n_main - a.n_main // 4 - a.n_main // 2]
    rng.shuffle(main_set)
    # piece frequencies on the train transcripts
    cnt = collections.Counter()
    for l in open(a.train):
        cnt.update(sp.encode(json.loads(l)["transcript"]))
    dev = [json.loads(l) for l in open(a.dev)]
    seen = set(main_set); hard = []
    tags = collections.Counter()
    for r in dev:
        s = r["transcript"]
        if s in seen or not (4 <= len(s.split()) <= 40):
            continue
        ids = sp.encode(s); rare = sum(1 for i in ids if cnt[i] < 50)
        words = s.split()
        names = sum(1 for w in words[1:] if w[:1].isupper() and w.strip(".,;:!?").isalpha())
        tag = None
        low = " " + s.lower() + " "
        if re.search(r"[a-zñ](tt|dd|ll|ñ)[aeiou]", low): tag = "palatal"          # Antton, onddo, llabur, andereño
        elif re.search(r"\bj[aeiou]|[aeiou]j[aeiou]|g[ei]", low): tag = "g_j"      # Julen, ijito, gero/gizon
        elif rare >= 2: tag = "rare_pieces"
        elif r.get("text_src") == "gen_numbers": tag = "numbers"
        elif names >= 2: tag = "names"
        elif len(words) >= 22: tag = "long"
        elif s.rstrip().endswith("?"): tag = "question"
        if tag and tags[tag] < a.n_hard // 7:
            tags[tag] += 1; hard.append({"text": s, "tag": tag}); seen.add(s)
        if len(hard) >= a.n_hard:
            break
    json.dump({"seed": 7, "sentences": main_set}, open(os.path.join(a.out, "main.json"), "w"), ensure_ascii=False, indent=0)
    json.dump({"sentences": hard}, open(os.path.join(a.out, "hard.json"), "w"), ensure_ascii=False, indent=0)
    print("main", len(main_set), "hard", len(hard), dict(tags))


if __name__ == "__main__":
    main()
