#!/usr/bin/env python3
"""Build docs/index.html (static demo page, GitHub Pages) from docs/demo/sentences.json, the generated clips in
docs/audio/{pocket,piper,omnivoice}/ and their timings. No external assets.

  python scripts/build_demo_page.py
"""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
cfg = json.load(open(os.path.join(DOCS, "demo", "sentences.json")))
tim_pocket = json.load(open(os.path.join(DOCS, "audio", "pocket", "timings.json"))) if os.path.exists(os.path.join(DOCS, "audio", "pocket", "timings.json")) else {"clips": {}}
HF = "https://huggingface.co/itzune/pocket-tts-basque"
HF24 = "https://huggingface.co/itzune/pocket-tts-basque-24l"
GH = "https://github.com/itzune/pocket-tts-basque"
DS = "https://huggingface.co/datasets/itzune/bssd"

VOICE_INFO = {"Maider": ("female", "HiTZ-Aholab studio speaker"), "Antton": ("male", "HiTZ-Aholab studio speaker"),
              "Nahia": ("female", "synthetic voice"), "Ane": ("female", "synthetic voice"), "Aitor": ("male", "synthetic voice")}


def audio(path):
    if not os.path.exists(os.path.join(DOCS, path)):
        return '<span class="missing">clip missing</span>'
    return f'<audio controls preload="none" src="{path}"></audio>'


def esc(s):
    return html.escape(s)


voices_html = []
for v in cfg["voices"]:
    gender, origin = VOICE_INFO[v]
    rows = []
    items = [("intro", cfg["intro"].format(name=v))] + [(c["id"], c["text"]) for c in cfg["common"]]
    for sid, text in items:
        t = tim_pocket["clips"].get(f"{v}_{sid}")
        speed = f'<span class="speed">{t["x_realtime"]}× real time on CPU</span>' if t else ""
        rows.append(f'<li><p class="eu">{esc(text)}</p>{audio(f"audio/pocket/{v}_{sid}.mp3")}{speed}</li>')
    voices_html.append(f'''<article class="voice"><h3>{v} <small>{gender} · {origin}</small></h3><ul>{"".join(rows)}</ul>
<p class="use">Use it: <code>--voice hf://itzune/pocket-tts-basque/voices/{v}.safetensors</code></p></article>''')

cmp_rows = []
for v in cfg["compare_voices"]:
    for c in cfg["common"]:
        cmp_rows.append(f'''<tr><td class="sent"><span class="vname">{v}</span><p class="eu">{esc(c["text"])}</p></td>
<td>{audio(f"audio/pocket/{v}_{c['id']}.mp3")}</td><td>{audio(f"audio/piper/{v}_{c['id']}.mp3")}</td><td>{audio(f"audio/omnivoice/{v}_{c['id']}.mp3")}</td></tr>''')

speeds = [t["x_realtime"] for t in tim_pocket["clips"].values()]
mean_speed = f"{sum(speeds)/len(speeds):.1f}" if speeds else "—"

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pocket TTS Basque</title>
<meta name="description" content="Open Basque (euskara) text-to-speech that runs in real time on a CPU: demos, voices and a comparison with Piper and OmniVoice.">
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66635c;--accent:#b4452c;--card:#ffffff;--line:#e6e2d9;--code:#f1eee7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ecebe6;--muted:#a09d95;--accent:#f08a6a;--card:#1f1f25;--line:#2e2e36;--code:#26262d}}}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:980px;margin:0 auto;padding:0 16px 64px}}header{{padding:56px 0 24px}}h1{{font-size:2.4rem;margin:0 0 8px;letter-spacing:-.02em}}
h1 small{{display:block;font-size:1.1rem;font-weight:400;color:var(--muted)}}h2{{margin:48px 0 12px;font-size:1.5rem}}h3{{margin:0 0 6px;font-size:1.15rem}}h3 small{{font-weight:400;color:var(--muted);font-size:.9rem}}
.links a{{display:inline-block;margin:8px 10px 0 0;padding:8px 14px;border:1px solid var(--line);border-radius:999px;text-decoration:none;color:var(--fg);background:var(--card)}}.links a.primary{{background:var(--accent);border-color:var(--accent);color:#fff}}
.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}.facts div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}}.facts b{{display:block;font-size:1.3rem}}.facts span{{color:var(--muted);font-size:.9rem}}
pre{{background:var(--code);padding:14px;border-radius:10px;overflow:auto;font-size:.88rem}}code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.9em}}pre code{{padding:0;background:none}}
.voice{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:12px 0}}.voice ul{{list-style:none;margin:0;padding:0}}.voice li{{display:grid;grid-template-columns:1fr auto;gap:6px 16px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}}.voice li:first-child{{border-top:0}}
.eu{{margin:0;font-style:italic}}.speed{{grid-column:1/-1;color:var(--muted);font-size:.85rem}}.use{{margin:10px 0 0;color:var(--muted);font-size:.9rem}}
audio{{width:260px;max-width:100%;height:36px}}.missing{{color:var(--muted);font-size:.85rem}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}}th,td{{padding:10px 12px;border-top:1px solid var(--line);vertical-align:middle;text-align:left}}th{{background:var(--code);font-weight:600}}td audio{{width:200px}}.sent{{min-width:220px}}.vname{{display:inline-block;font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);font-weight:700}}
.sys{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:14px 0}}.sys div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:.92rem}}.sys b{{display:block;margin-bottom:4px}}
footer{{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:.88rem}}a{{color:var(--accent)}}
@media (max-width:640px){{.voice li{{grid-template-columns:1fr}}table,thead,tbody,tr,td,th{{display:block}}thead{{display:none}}td{{border-top:0;padding:4px 12px}}tr{{border-top:1px solid var(--line);padding:8px 0}}td audio{{width:100%}}td:not(.sent)::before{{content:attr(data-sys);display:block;font-size:.75rem;color:var(--muted)}}}}
</style>
</head>
<body><main>
<header>
<h1>Pocket TTS Basque <small>Euskarazko ahots-sintesia, denbora errealean zure ordenagailuan · Open Basque text-to-speech that runs in real time on a CPU</small></h1>
<div class="links"><a class="primary" href="{HF}">Model on Hugging Face</a><a href="{GH}">Code on GitHub</a><a href="{DS}">BSSD dataset</a><a href="{HF24}">24-layer teacher</a></div>
<div class="facts"><div><b>~100 M</b><span>parameters, 6-layer backbone</span></div><div><b>{mean_speed}×</b><span>real time on 2 CPU threads (fp32, this page's clips)</span></div><div><b>5</b><span>bundled voices + zero-shot cloning</span></div><div><b>CC BY 4.0</b><span>model weights</span></div></div>
<p>Pocket TTS Basque is a Basque (euskara) adaptation of Kyutai's <a href="https://github.com/kyutai-labs/pocket-tts">Pocket TTS</a>, trained on the synthetic <a href="{DS}">BSSD</a> dataset (470 h, CC BY-SA 4.0). A 24-layer teacher was fine-tuned first and then distilled into this 6-layer model, which keeps the teacher's pronunciation and prosody while running several times faster than real time on an ordinary laptop CPU. All clips on this page were generated by the released model on 2 CPU threads; nothing was hand-picked.</p>
</header>

<h2>Try it</h2>
<pre><code># with uv installed, no setup needed
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque/basque.yaml \\
  --voice hf://itzune/pocket-tts-basque/voices/Maider.safetensors \\
  --text "Kaixo! Hau euskarazko ahots sintetiko bat da." --output kaixo.wav

# your own voice: 5-10 s of clean Basque speech as the prompt
uvx pocket-tts generate --config hf://itzune/pocket-tts-basque/basque.yaml --voice my_voice.wav --text "Kaixo!" --output kaixo.wav</code></pre>
<p>Write the text as it should be read (numbers, dates and abbreviations in words). Python usage, evaluation details and licences are in the <a href="{HF}">model card</a>.</p>

<h2>Voices</h2>
<p>Maider and Antton are the two professional speakers of the <a href="https://zenodo.org/records/17952596">HiTZ-Aholab studio corpus</a> (CC BY 4.0). Nahia, Ane and Aitor are synthetic voices from the BSSD voice bank. Each reads the same three sentences: a news item, a narrative line and a question with numbers.</p>
{"".join(voices_html)}

<h2>Comparison with other Basque TTS</h2>
<p>The same sentences and the same two voices, synthesised with three systems. Piper uses the community <a href="https://huggingface.co/itzune/maider-tts">Maider</a> and <a href="https://huggingface.co/itzune/antton-tts">Antton</a> voices trained on the HiTZ-Aholab recordings; OmniVoice clones the voice from the very same 5-7 s prompt clip that our voice states were built from (it is also the system that generated the BSSD training data, i.e. our teacher's teacher).</p>
<div class="sys"><div><b>Pocket TTS Basque (this model)</b>~100 M parameters, 438 MB fp32 (int8 available) · CPU, several times faster than real time · zero-shot voice cloning · CC BY 4.0</div>
<div><b>Piper (VITS)</b>~20 M parameters per voice, ~60 MB ONNX · CPU, very fast · fixed voices, espeak-ng phonemes · voices Apache-2.0</div>
<div><b>OmniVoice (k2-fsa)</b>~0.6 B parameters · needs a GPU for practical use · zero-shot cloning, 100+ languages · CC BY-NC (weights)</div></div>
<table><thead><tr><th>Sentence</th><th>Pocket TTS Basque</th><th>Piper</th><th>OmniVoice</th></tr></thead><tbody>
{"".join(cmp_rows)}
</tbody></table>

<h2>How it was built</h2>
<ol>
<li><b>Data.</b> <a href="{DS}">BSSD</a>: 470 h of synthetic Basque speech generated with OmniVoice from curated Basque text, 2,400+ voices (a synthetic voice bank plus consented real prompts), filtered with two Basque ASR systems, UTMOS and speaker checks. Word alignments from a Basque wav2vec2 model.</li>
<li><b>Teacher.</b> Kyutai's 24-layer English Pocket TTS fine-tuned for Basque with a new 4000-token SentencePiece tokenizer (official new-language recipe). Released as <a href="{HF24}">pocket-tts-basque-24l</a>.</li>
<li><b>Student.</b> Depth distillation of the teacher into a 6-layer backbone with classifier-free guidance baked in (150 k steps). On held-out sentences and unseen voices it reaches WER 0.03 with both ASR systems, below the teacher, at a slightly lower naturalness score.</li>
</ol>
<p>Everything needed to reproduce the training and evaluation is in the <a href="{GH}">GitHub repository</a>; the dataset pipeline is documented with the dataset.</p>

<footer>Model weights CC BY 4.0 · dataset CC BY-SA 4.0 · Maider and Antton voices from the HiTZ-Aholab studio corpus (CC BY 4.0) · built on <a href="https://github.com/kyutai-labs/pocket-tts">Pocket TTS</a> by Kyutai · an <a href="https://huggingface.co/itzune">Itzune</a> project.</footer>
</main></body></html>
'''
open(os.path.join(DOCS, "index.html"), "w").write(page)
print("wrote docs/index.html", len(page), "bytes; pocket clips:", len(tim_pocket["clips"]))
