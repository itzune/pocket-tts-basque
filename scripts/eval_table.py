#!/usr/bin/env python3
"""Markdown comparison table over eval folders (eval_basque.py + score_conformer.py outputs).

  python scripts/eval_table.py /root/runs/basque_24l/eval_final/step*  > reports/teacher_compare.md
"""
import json
import os
import re
import sys


def main() -> None:
    dirs = sorted((d.rstrip("/") for d in sys.argv[1:] if os.path.isfile(os.path.join(d, "summary.json"))),
                  key=lambda d: int(re.search(r"(\d+)", d.split("/")[-1]).group(1)))
    cols = ["wer", "wer_conformer", "fail_rate", "fail_rate_conformer", "wer_ok", "utmos", "spk_sim"]
    print("| step | " + " | ".join(f"{k} {c}" for k in ("named", "held") for c in cols) + " |")
    print("|---|" + "---|" * (2 * len(cols)))
    pv = {}
    for d in dirs:
        s = json.load(open(d + "/summary.json"))
        step = re.search(r"(\d+)", d.split("/")[-1]).group(1)
        cells = []
        for kind in ("named", "held_out"):
            for c in cols:
                v = s["summary"].get(kind, {}).get(c)
                cells.append("" if v is None else f"{v:.3f}")
        print(f"| {step} | " + " | ".join(cells) + " |")
        pv[step] = s.get("per_voice", {})
    print("\nFailures (WER > 0.5, Whisper / Conformer) per voice:\n")
    voices = sorted({v for d in pv.values() for v in d})
    print("| voice | " + " | ".join(pv) + " |")
    print("|---|" + "---|" * len(pv))
    for v in voices:
        print(f"| {v} | " + " | ".join(f"{pv[s].get(v, {}).get('fails', '')}/{pv[s].get(v, {}).get('fails_conformer', '')}" for s in pv) + " |")


if __name__ == "__main__":
    main()
