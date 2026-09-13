#!/usr/bin/env bash
# CrowdHuman Experiment 1: fixed-quality oracle instance selection.
#
# The script first creates the immutable candidate/selection artifacts, runs a
# 200-update smoke test for every method, and only then launches the 2,000-update
# x 3-seed full runs.  BATCH is the total batch size across the four GPUs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICES="${DEVICES:-0,1,2,3}"
GEN_DEVICE="${GEN_DEVICE:-0}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
PCT="${PCT:-1p}"
ROOT="$(pwd)"
DATA_ROOT="${ROOT}/datasets/crowdhuman"
DATA_YAML="${ROOT}/ultralytics/cfg/datasets/crowdhuman_${PCT}.yaml"
MODEL="${ROOT}/runs/crowdhuman_labeled_baseline_${PCT}/yolov8n_crowdhuman_labeled/weights/best.pt"
if [[ "${PCT}" == "10p" ]]; then
  MODEL="${ROOT}/runs/crowdhuman_labeled_baseline/yolov8n_crowdhuman_labeled/weights/best.pt"
fi

ARTIFACT_ROOT="${ROOT}/runs/crowdhuman_experiment1_${PCT}"
CANDIDATES="${ARTIFACT_ROOT}/candidates.json"
SELECTION_DIR="${ARTIFACT_ROOT}/selection"
SMOKE_ROOT="${ROOT}/runs/crowdhuman_experiment1_${PCT}_smoke"
FULL_ROOT="${ROOT}/runs/crowdhuman_experiment1_${PCT}"
METHODS=(CONF RANDOM COVERAGE NEAR)

mkdir -p "${ARTIFACT_ROOT}"

if [[ ! -s "${CANDIDATES}" ]]; then
  "${PYTHON_BIN}" scripts/crowdhuman/experiment1.py generate \
    --model "${MODEL}" \
    --dataset-root "${DATA_ROOT}" \
    --unlabeled-list "${DATA_ROOT}/train_unlabeled_${PCT}.txt" \
    --annotation "${DATA_ROOT}/annotation_train.odgt" \
    --out "${CANDIDATES}" \
    --imgsz 640 --nms-iou 0.65 --candidate-conf 0.05 \
    --batch 16 --device "${GEN_DEVICE}"
fi

if [[ ! -s "${SELECTION_DIR}/selection_summary.json" ]]; then
  "${PYTHON_BIN}" scripts/crowdhuman/experiment1.py select \
    --candidates "${CANDIDATES}" \
    --dataset-root "${DATA_ROOT}" \
    --annotation "${DATA_ROOT}/annotation_train.odgt" \
    --labeled-list "${DATA_ROOT}/train_labeled_${PCT}.txt" \
    --out-dir "${SELECTION_DIR}" \
    --methods "${METHODS[@]}" --seed 0
fi

for method in "${METHODS[@]}"; do
  echo "=== smoke: ${method} ==="
  "${PYTHON_BIN}" scripts/crowdhuman/experiment1.py train \
    --data "${DATA_YAML}" \
    --selection "${SELECTION_DIR}/${method}.json" \
    --model "${MODEL}" \
    --method "${method}" --seed 0 --updates 200 --val-interval 0 \
    --batch "${BATCH}" --workers "${WORKERS}" --device "${DEVICES}" \
    --project "${SMOKE_ROOT}" --name "${method}/seed0" \
    --ssod-weight 1.0
done

for method in "${METHODS[@]}"; do
  for seed in 0 1 2; do
    echo "=== full: ${method} seed${seed} ==="
    "${PYTHON_BIN}" scripts/crowdhuman/experiment1.py train \
      --data "${DATA_YAML}" \
      --selection "${SELECTION_DIR}/${method}.json" \
      --model "${MODEL}" \
      --method "${method}" --seed "${seed}" --updates 2000 --val-interval 500 --val \
      --batch "${BATCH}" --workers "${WORKERS}" --device "${DEVICES}" \
      --project "${FULL_ROOT}" --name "${method}/seed${seed}" \
      --ssod-weight 1.0
  done
done

"${PYTHON_BIN}" scripts/crowdhuman/experiment1.py report \
  --selection-dir "${SELECTION_DIR}" \
  --runs-dir "${FULL_ROOT}" \
  --out "${ARTIFACT_ROOT}/report.md" \
  --methods "${METHODS[@]}" --seeds 0 1 2

echo "Experiment 1 completed: ${ARTIFACT_ROOT}/report.md"
