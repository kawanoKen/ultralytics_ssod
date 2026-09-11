#!/usr/bin/env bash
# Waits for the currently-running yolov8n baseline/dfl SSOD runs to finish, then launches the
# same baseline/dfl comparison for yolo11n from its own converged VOC07 checkpoint.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BASELINE_PID=$1
DFL_PID=$2

echo "Waiting for yolov8n baseline (pid ${BASELINE_PID}) and dfl (pid ${DFL_PID}) to finish..."
while kill -0 "${BASELINE_PID}" 2>/dev/null || kill -0 "${DFL_PID}" 2>/dev/null; do
  sleep 15
done
echo "yolov8n runs finished. Launching yolo11n baseline/dfl."

BASE_MODEL="runs/voc_labeled_baseline/yolo11n_voc07_labeled/weights/best.pt"
uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolo11n --model "${BASE_MODEL}" --device "0,1" --batch 128 --batch_ssod 128 \
  > scripts/voc_ssod/logs/yolo11n_baseline.log 2>&1 &
p1=$!
uv run python scripts/voc_ssod/train_ssod.py --variant dfl --arch yolo11n --model "${BASE_MODEL}" --device "2,3" --batch 128 --batch_ssod 128 \
  > scripts/voc_ssod/logs/yolo11n_dfl.log 2>&1 &
p2=$!
echo "yolo11n baseline pid ${p1}, dfl pid ${p2}"
wait "${p1}" "${p2}"
echo "yolo11n runs finished."
