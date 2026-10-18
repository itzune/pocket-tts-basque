#!/usr/bin/env bash
# Export the 6-layer model to ONNX (fp32 + int8) for PyTorch-free inference with onnxruntime, in the bundle
# format of KevinAHM/pocket-tts-onnx, and assemble the itzune/pocket-tts-basque-onnx release folder:
# bundle + bundled voices + the reference runtime (pocket_tts_onnx.py, Apache-2.0) + model card.
#
# Uses the community exporter (lomotron/pocket-tts-onnx-export) pinned at EXPORTER_REV with
# scripts/onnx_exporter.patch: upstream always passes the default --language along with --config, which
# TTSModel.load_model rejects. PY needs torch plus the exporter's requirements.txt (CPU is enough).
#
#   PY=~/pocket-tts/.venv/bin/python scripts/export_onnx.sh
#   python scripts/upload_hf.py --folder /tmp/pocket-onnx-export/release --repo itzune/pocket-tts-basque-onnx
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}; WORK=${WORK:-/tmp/pocket-onnx-export}
EXPORTER=${EXPORTER:-https://github.com/lomotron/pocket-tts-onnx-export}
EXPORTER_REV=${EXPORTER_REV:-8f52199533098963fe1cbd90bed35bb20d871b5e}
RUNTIME_REV=${RUNTIME_REV:-58a6d00cf13d239b6748cb0769f35c580a8f606c}   # KevinAHM/pocket-tts-onnx
log(){ echo "[$(date -Is)] $*"; }

if [ ! -d $WORK/exporter ]; then
  log "cloning exporter at $EXPORTER_REV"
  git clone -q $EXPORTER $WORK/exporter
  git -C $WORK/exporter checkout -q $EXPORTER_REV
  git -C $WORK/exporter apply "$PWD/scripts/onnx_exporter.patch"
fi

# The exporter's vendored pocket_tts predates `default_temperature` (a sampling default, not a weight)
# and rejects it: drop it here; runtimes must sample at 0.3 themselves. The file stem names the bundle.
grep -v '^default_temperature:' release/basque_6l/basque.yaml > $WORK/exporter/basque.yaml
log "exporting (fp32 + int8, verified against torch)"
rm -rf $WORK/out
(cd $WORK/exporter && $PY export.py --config basque.yaml --output_dir $WORK/out --quantize)

log "assembling $WORK/release"
REL=$WORK/release
rm -rf $REL; mkdir -p $REL/voices
find $WORK/out/basque -maxdepth 1 -type f -exec cp {} $REL/ \;
$PY - "$REL" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
rel = sys.argv[1]
for name in ["Maider", "Antton", "Nahia", "Ane", "Aitor"]:
    shutil.copy(hf_hub_download("itzune/pocket-tts-basque", f"voices/{name}.safetensors"), f"{rel}/voices/")
shutil.copy(hf_hub_download("itzune/pocket-tts-basque", "voices/voices.json"), f"{rel}/voices/")
EOF
curl -sfL -o $REL/pocket_tts_onnx.py \
  https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/$RUNTIME_REV/pocket_tts_onnx.py
cp release/basque_6l_onnx/README.md $REL/README.md
log "done: $(du -sh $REL | cut -f1) in $REL"
