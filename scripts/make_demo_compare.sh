#!/usr/bin/env bash
# Comparison clips for the demo page: the same sentences with Piper (eu-maider/antton-medium ONNX voices)
# and OmniVoice (voice cloning from the same Maider/Antton prompt clips used for our voice states).
# Run on the GPU VM after make_demo_audio.py. Needs /root/piper/.venv (piper-tts) and the dataset venv.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-/root/demo}; SENT=${SENT:-docs/demo/sentences.json}
NAMED=${NAMED:-/root/data/named/selected.jsonl}
PIPER_PY=${PIPER_PY:-/root/piper/.venv/bin/python}; OMNI_VENV=${OMNI_VENV:-/root/pocket-tts-eu/.venv}
mkdir -p $OUT/piper $OUT/omnivoice
python3 - "$SENT" "$NAMED" "$OUT" <<'PY'
import json, sys, os, time, subprocess
sent, named, out = sys.argv[1:4]
cfg = json.load(open(sent)); prompts = {json.loads(l)["voice_id"]: json.loads(l) for l in open(named)}
tim = {}
jobs = open(os.path.join(out, "omnivoice_jobs.jsonl"), "w")
for voice in cfg["compare_voices"]:
    for c in cfg["common"]:
        # Piper
        wav = os.path.join(out, "piper", f"{voice}_{c['id']}.wav")
        t0 = time.time()
        subprocess.run([os.environ.get("PIPER_PY", "/root/piper/.venv/bin/python"), "-m", "piper", "-m",
                        f"/root/piper/voices/eu-{voice.lower()}-medium.onnx", "-f", wav], input=c["text"].encode(), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tim[f"piper/{voice}_{c['id']}"] = round(time.time() - t0, 2)
        # OmniVoice job (clone from the same prompt clip)
        p = prompts[voice]
        jobs.write(json.dumps({"id": f"{voice}_{c['id']}", "text": c["text"], "ref_audio": p["path"], "ref_text": p["text"], "language_id": "eu"}, ensure_ascii=False) + "\n")
jobs.close()
json.dump(tim, open(os.path.join(out, "piper", "timings.json"), "w"), indent=1)
print("piper done", tim)
PY
MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$HOME/bssd/configs/dataset_v1.yaml'))['teacher']['model'])")
PATH=$OMNI_VENV/bin:$PATH omnivoice-infer-batch --model "$MODEL" --test_list $OUT/omnivoice_jobs.jsonl --res_dir $OUT/omnivoice/raw \
    --batch_size 6 --num_step 32 --guidance_scale 2.0 --enable_flashinfer true 2>&1 | grep -vE "NNPACK|it/s\]" | tail -3
find $OUT/omnivoice/raw -name "*.wav" -exec cp {} $OUT/omnivoice/ \;
ls $OUT/omnivoice/*.wav | wc -l
# mp3 versions for the web page
for d in pocket piper omnivoice; do for f in $OUT/$d/*.wav; do ffmpeg -loglevel error -y -i "$f" -codec:a libmp3lame -b:a 96k "${f%.wav}.mp3"; done; done
echo "COMPARE DEMO DONE"
