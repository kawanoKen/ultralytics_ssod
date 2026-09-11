#!/usr/bin/env bash
set -euo pipefail

# Run the two frozen R1 experiments sequentially. Override the first argument only
# when the baseline was run on a different CUDA device set.
R1_DEVICE="${1:-0,1}"
MODEL="runs/crowdhuman_labeled_baseline_5p/yolov8n_crowdhuman_labeled/weights/best.pt"
DATA="ultralytics/cfg/datasets/crowdhuman_5p.yaml"

.venv/bin/python scripts/dfl_assignment/train_assignment_stability.py \
  --perturbation fixed \
  --model "$MODEL" \
  --data "$DATA" \
  --device "$R1_DEVICE" \
  --project runs/r1_preliminary \
  --name r1_fixed

.venv/bin/python scripts/dfl_assignment/train_assignment_stability.py \
  --perturbation width_matched \
  --model "$MODEL" \
  --data "$DATA" \
  --device "$R1_DEVICE" \
  --project runs/r1_preliminary \
  --name r1_width
