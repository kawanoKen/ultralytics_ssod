#!/usr/bin/env bash
# Waits for the CrowdHuman labeled-only baseline (both yolov8n and yolo11n) to finish,
# then launches the 4 SSOD comparison runs (yolov8n/yolo11n x baseline/dfl), 2+2 GPU,
# starting directly from each arch's converged best.pt (burn_in_epochs=0, same recipe
# as scripts/voc_ssod/train_ssod.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BASELINE_SH_PID=$1
echo "Waiting for CrowdHuman labeled baseline (pid ${BASELINE_SH_PID}) to finish..."
while kill -0 "${BASELINE_SH_PID}" 2>/dev/null; do
  sleep 15
done
echo "CrowdHuman labeled baseline finished. Launching SSOD comparison runs."

DATA="crowdhuman_10p.yaml"
PROJECT="runs/crowdhuman_ssod"
BATCH=128       # total across 2 GPUs = 64/GPU, per bench_ssod_batch_size.sh
BATCH_SSOD=128

for arch in yolov8n yolo11n; do
  BASE_MODEL="runs/crowdhuman_labeled_baseline/${arch}_crowdhuman_labeled/weights/best.pt"
  uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch "${arch}" --model "${BASE_MODEL}" \
    --data "${DATA}" --project "${PROJECT}" --device "0,1" --batch "${BATCH}" --batch_ssod "${BATCH_SSOD}" \
    > "scripts/crowdhuman/logs/${arch}_baseline.log" 2>&1 &
  p1=$!
  uv run python scripts/voc_ssod/train_ssod.py --variant dfl --arch "${arch}" --model "${BASE_MODEL}" \
    --data "${DATA}" --project "${PROJECT}" --device "2,3" --batch "${BATCH}" --batch_ssod "${BATCH_SSOD}" \
    > "scripts/crowdhuman/logs/${arch}_dfl.log" 2>&1 &
  p2=$!
  echo "${arch} baseline pid ${p1}, dfl pid ${p2}"
  wait "${p1}" "${p2}"
  echo "${arch} SSOD runs finished."
done

echo "All CrowdHuman SSOD runs finished."
