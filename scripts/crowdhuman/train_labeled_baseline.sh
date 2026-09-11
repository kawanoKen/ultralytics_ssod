#!/usr/bin/env bash
# Supervised-only convergence baseline on CrowdHuman, before SSOD.
# Usage: train_labeled_baseline.sh <pct>   where <pct> in {1p, 5p, 10p}
#   labeled   = <pct> of train (used for training)
#   unlabeled = the rest (listed as 'ssod_train' but NOT used here)
#   eval      = official val split (4370 images, used as 'val')
#
# Batch size picked from scripts/crowdhuman/bench_batch_size.sh: unlike VOC,
# CrowdHuman's dense scenes (~23 persons/image) make batch=128/GPU noticeably
# SLOWER (not OOM, but ~2.5x fewer img/s) than batch=64/GPU, so this uses a
# smaller total batch than the VOC baseline script.
set -euo pipefail

PCT=${1:?usage: train_labeled_baseline.sh <pct e.g. 1p|5p|10p>}

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA="crowdhuman_${PCT}.yaml"
DEVICES="0,1,2,3"       # 4x RTX 3090, DDP
EPOCHS=100
IMGSZ=640
BATCH=256               # total batch size across all GPUs (64/gpu, per bench_batch_size.sh)
SAVE_PERIOD=10
PROJECT="runs/crowdhuman_labeled_baseline_${PCT}"

run() {
  local model=$1
  local name=$2
  echo "==> Training ${model} (labeled-only baseline, ${PCT}) -> ${PROJECT}/${name}"
  uv run yolo detect train \
    model="${model}" \
    data="${DATA}" \
    epochs="${EPOCHS}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICES}" \
    save_period="${SAVE_PERIOD}" \
    project="${PROJECT}" \
    name="${name}" \
    exist_ok=True
}

run yolov8n.pt yolov8n_crowdhuman_labeled
run yolo11n.pt yolo11n_crowdhuman_labeled
