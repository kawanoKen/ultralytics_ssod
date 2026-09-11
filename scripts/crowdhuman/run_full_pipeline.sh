#!/usr/bin/env bash
# Full 6-experiment CrowdHuman pipeline for one labeled percentage:
#   1) labeled-only baseline (yolov8n, yolo11n), 4-GPU DDP
#   2) SSOD comparison (yolov8n/yolo11n x baseline/dfl), 2+2 GPU, burn_in_epochs=0
#      starting from each arch's step-1 best.pt
# Usage: run_full_pipeline.sh <pct>   where <pct> in {1p, 5p, 10p}
set -euo pipefail

PCT=${1:?usage: run_full_pipeline.sh <pct e.g. 1p|5p|10p>}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== [${PCT}] Step 1/2: labeled-only baseline ==="
bash scripts/crowdhuman/train_labeled_baseline.sh "${PCT}"

echo "=== [${PCT}] Step 2/2: SSOD comparison ==="
DATA="crowdhuman_${PCT}.yaml"
PROJECT="runs/crowdhuman_ssod_${PCT}"
BATCH=128       # total across 2 GPUs = 64/GPU, per bench_ssod_batch_size.sh
BATCH_SSOD=128

for arch in yolov8n yolo11n; do
  BASE_MODEL="runs/crowdhuman_labeled_baseline_${PCT}/${arch}_crowdhuman_labeled/weights/best.pt"
  uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch "${arch}" --model "${BASE_MODEL}" \
    --data "${DATA}" --project "${PROJECT}" --device "0,1" --batch "${BATCH}" --batch_ssod "${BATCH_SSOD}" \
    > "scripts/crowdhuman/logs/${PCT}_${arch}_baseline.log" 2>&1 &
  p1=$!
  uv run python scripts/voc_ssod/train_ssod.py --variant dfl --arch "${arch}" --model "${BASE_MODEL}" \
    --data "${DATA}" --project "${PROJECT}" --device "2,3" --batch "${BATCH}" --batch_ssod "${BATCH_SSOD}" \
    > "scripts/crowdhuman/logs/${PCT}_${arch}_dfl.log" 2>&1 &
  p2=$!
  echo "[${PCT}] ${arch} baseline pid ${p1}, dfl pid ${p2}"
  wait "${p1}" "${p2}"
  echo "[${PCT}] ${arch} SSOD runs finished."
done

echo "=== [${PCT}] All 6 experiments finished. ==="
