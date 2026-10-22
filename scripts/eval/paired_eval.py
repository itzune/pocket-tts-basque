#!/usr/bin/env python3
"""Seed-matched synthesis of frozen sentence sets for one model, in the eval.jsonl schema that
scripts/score_conformer.py and scripts/eval/paired_report.py read.

Every (voice, sentence, seed) gets the same torch seed for every model, so two runs of this script
with different models can be compared pairwise. Voice prompts: the named voices (selected.jsonl, audio
prompts) and the held-out test voices (one clip each), as in eval_basque.py.

  python scripts/eval/paired_eval.py --config configs/basque_6l_infer.yaml [--checkpoint ...] \
      --sets reports/eval_sets --named /root/data/named/selected.jsonl --test /root/data/bssd/test.jsonl \
      --seeds 1 --out /root/evals/basque_6l
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, librosa
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for cand in (Path.home() / "bssd", Path(__file__).resolve().parents[3] / "itzune" / "bssd"):
    if (cand / "bssd").is_dir():
        sys.path.insert(0, str(cand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint")
    ap.add_argument("--sets", required=True); ap.add_argument("--named", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--held-out", default="bank0027,bank0083,bank0070,euf_05679,eum_08886")
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--eos-threshold", type=float, default=None)
    ap.add_argument("--all-voices", action="store_true", help="every voice reads every sentence (default: each sentence is read by one named and one held-out voice, chosen deterministically)")
    a = ap.parse_args()
    from pocket_tts import TTSModel
    from bssd.scoring import UTMOS, BasqueASR, SpeakerEmbedder, cer, cosine, wer
    from bssd.audio import load
    sets = {"main": json.load(open(os.path.join(a.sets, "main.json")))["sentences"],
            "hard": [h["text"] for h in json.load(open(os.path.join(a.sets, "hard.json")))["sentences"]]}
    tags = {h["text"]: h["tag"] for h in json.load(open(os.path.join(a.sets, "hard.json")))["sentences"]}
    test = [json.loads(l) for l in open(a.test)]
    prompts = [(json.loads(l)["voice_id"], json.loads(l)["path"], "named") for l in open(a.named)]
    for vid in a.held_out.split(","):
        rows = [r for r in test if r["voice_id"] == vid and 4 <= r["duration"] <= 9]
        prompts.append((vid, rows[0]["path"], "held_out"))
    kw = {"eos_threshold": a.eos_threshold} if a.eos_threshold is not None else {}
    model = TTSModel.load_model(config=a.config, checkpoint=a.checkpoint, **kw)
    if torch.cuda.is_available():
        model = model.to("cuda")
    asr, utmos, spk = BasqueASR(), UTMOS("cuda"), SpeakerEmbedder("cuda")
    os.makedirs(a.out, exist_ok=True)
    done = set()
    outp = os.path.join(a.out, "eval.jsonl")
    if os.path.exists(outp):
        done = {(r["voice_id"], r["set"], r["idx"], r["seed"]) for r in map(json.loads, open(outp))}
    fh = open(outp, "a")
    t_all = time.time()
    named_ids = [p[0] for p in prompts if p[2] == "named"]; held_ids = [p[0] for p in prompts if p[2] == "held_out"]

    def reads(vid, kind, sname, i):
        if a.all_voices:
            return True
        off = 0 if sname == "main" else 7
        pool = named_ids if kind == "named" else held_ids
        return pool[(i + off) % len(pool)] == vid

    for pi, (vid, ppath, kind) in enumerate(prompts):
        state = model.get_state_for_audio_prompt(ppath)
        pa, _ = load(ppath, 16000); pemb = spk.embed(pa)
        ptext = next((json.loads(l).get("text", "") for l in open(a.named) if json.loads(l)["voice_id"] == vid), "")
        for sname, sents in sets.items():
            for i, text in enumerate(sents):
                if not reads(vid, kind, sname, i):
                    continue
                for seed in range(a.seeds):
                    if (vid, sname, i, seed) in done:
                        continue
                    torch.manual_seed(seed * 1_000_003 + (0 if sname == "main" else 500_000) + i)  # same for every model
                    t0 = time.time()
                    audio = model.generate_audio(state, text)
                    dt = time.time() - t0
                    au = audio.detach().cpu().numpy().astype(np.float32).squeeze(); sr = model.sample_rate
                    wav = os.path.join(a.out, f"{vid}_{sname}{i:03d}_s{seed}.wav"); sf.write(wav, au, sr)
                    a16 = librosa.resample(au, orig_sr=sr, target_sr=16000)
                    hyp = asr.transcribe(a16)
                    fh.write(json.dumps({"voice_id": vid, "kind": kind, "set": sname, "idx": i, "seed": seed, "tag": tags.get(text),
                                         "text": text, "hyp": hyp, "wer": round(wer(text, hyp), 4), "cer": round(cer(text, hyp), 4),
                                         "utmos": round(utmos.score(a16), 3), "spk_sim": round(cosine(pemb, spk.embed(a16)), 3),
                                         "duration": round(len(au) / sr, 2), "gen_s": round(dt, 2), "prompt_text": ptext, "wav": wav},
                                        ensure_ascii=False) + "\n"); fh.flush()
        print(vid, kind, "done", round(time.time() - t_all), "s", flush=True)
    fh.close()
    json.dump({"config": a.config, "checkpoint": a.checkpoint, "seeds": a.seeds, "eos_threshold": a.eos_threshold},
              open(os.path.join(a.out, "run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
