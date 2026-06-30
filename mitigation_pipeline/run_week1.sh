#!/usr/bin/env bash
# Week-1 spike on a plain A100 box (SSH / cloud VM / HPC). Same steps as
# week1_spike.ipynb, no Colab/Drive. Outputs persist on the server's local disk.
#
# One-time: scp the licensed RAF zip to the box (not in git):
#     scp RAF_DB.zip user@server:~/cambridge/RAF_DB.zip
#
# Run from inside the repo's mitigation_pipeline/ dir:
#     export WORK=$HOME/cambridge          # data + outputs root (has RAF_DB.zip)
#     # export HF_TOKEN=hf_xxx             # optional: faster HF downloads
#     bash run_week1.sh
set -euo pipefail
cd "$(dirname "$0")"                         # mitigation_pipeline/ (sibling imports need this)

WORK="${WORK:-$HOME/cambridge}"
RAF_ZIP="${RAF_ZIP:-$WORK/RAF_DB.zip}"
RAF_ROOT="$WORK/RAF_DB"
PROBE="$WORK/week1_spike/raf_probe"
ANNO="$PROBE/anno"
ADAPTER="$WORK/week1_spike/adapters/bite_llm_only"
HAIR_MODEL="$WORK/hair_segmenter.tflite"
PER_CLASS="${PER_CLASS:-40}"
PLACEMENT="${PLACEMENT:-llm_only}"

echo "== preflight: torch + CUDA =="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible'; print('CUDA:', torch.cuda.get_device_name(0))"

echo "== deps (torch assumed already installed via your env / module load) =="
pip install -q "transformers>=4.48" peft accelerate grad-cam retina-face mediapipe opencv-python tqdm

echo "== MediaPipe hair model =="
[ -f "$HAIR_MODEL" ] || wget -qO "$HAIR_MODEL" \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite

echo "== RAF: unzip + reorg =="
[ -d "$RAF_ROOT" ] || unzip -q "$RAF_ZIP" -d "$RAF_ROOT"
[ -d "$RAF_ROOT/DATASET/train/1" ] || python reorg_raf.py --raf-root "$RAF_ROOT" --out "$RAF_ROOT/DATASET"
RAF_TRAIN="$RAF_ROOT/DATASET/train"

echo "== probe subset =="
python make_probe.py --split "$RAF_ROOT/DATASET/test" --out "$PROBE/images" --per-class "$PER_CLASS"

echo "== annotations (RetinaFace -> hair seg -> merge) =="
python make_annotations.py --images "$PROBE/images" --work-dir "$ANNO" --hair-model "$HAIR_MODEL"

echo "== BASELINE Grad-CAM (stock) =="
python gradcam_lora.py --images "$PROBE/images" --annotations "$ANNO/annotations.json" --hair-dir "$ANNO/hair_masks"

echo "== bite-tune ($PLACEMENT) =="
python bite_tune.py --data-dir "$RAF_TRAIN" --placement "$PLACEMENT" --steps 300 --subset 350 --out "$ADAPTER"

echo "== GATE: stock vs adapted =="
python gradcam_lora.py --images "$PROBE/images" --annotations "$ANNO/annotations.json" --hair-dir "$ANNO/hair_masks" --adapter "$ADAPTER" --compare
