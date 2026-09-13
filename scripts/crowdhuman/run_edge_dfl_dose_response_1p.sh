#!/usr/bin/env bash
# CrowdHuman 1% edge-wise DFL supervision dose-response launcher.
#
# One run on any machine:
#   bash scripts/crowdhuman/run_edge_dfl_dose_response_1p.sh oracle 5 0 0,1,2,3
#
# One selector's full grid (w={0,1,5,20}, seed={0,1}):
#   bash scripts/crowdhuman/run_edge_dfl_dose_response_1p.sh --all oracle 0,1,2,3
# Both selectors' full grid:
#   bash scripts/crowdhuman/run_edge_dfl_dose_response_1p.sh --all-both 0,1,2,3
#
# w=1 is an exact baseline-equivalence control. It is included by --all only
# when RUN_W1_CONTROLS=1 is set, so remote execution can omit duplicate runs.
# Before an Oracle run on a fresh dataset copy, generate CrowdHuman labels once
# with scripts/crowdhuman/convert_odgt_to_yolo.py from this revision so fboxes
# are clipped before augmentation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"

MODEL="runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt"
PROJECT="runs/crowdhuman_ssod_1p_zero_pseudo_ablation"
BATCH="${BATCH:-128}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
DIAG_INTERVAL="${DIAG_INTERVAL:-100}"

run_one() {
  local selector="$1" weight="$2" seed="$3" devices="$4"
  local weight_tag="${weight/./p}"
  local name="yolov8n_1p_loss_balance_edge_dfl_${selector}_w${weight_tag}_seed${seed}"
  echo "=== ${name}; devices=${devices} ==="
  uv run python scripts/voc_ssod/train_ssod.py \
    --variant baseline --arch yolov8n --model "${MODEL}" --data crowdhuman_1p.yaml \
    --project "${PROJECT}" --name "${name}" --device "${devices}" \
    --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
    --diag-interval "${DIAG_INTERVAL}" --seed "${seed}" \
    --skip-zero-pseudo-cls-loss --loss-balancing-mode ema_scale --loss-balance-beta 0.9 \
    --edge-dfl-reweight --edge-dfl-selector "${selector}" --edge-dfl-weight "${weight}" \
    --oracle-edge-error-threshold 0.10
}

run_grid() {
  local selector="$1" devices="$2"
  for weight in 0 1 5 20; do
    if [[ "${weight}" == "1" && "${RUN_W1_CONTROLS:-0}" != "1" ]]; then
      echo "Skipping ${selector} w=1 baseline-equivalence control (set RUN_W1_CONTROLS=1 to run it)."
      continue
    fi
    for seed in 0 1; do
      run_one "${selector}" "${weight}" "${seed}" "${devices}"
    done
  done
}

if [[ "${1:-}" == "--all" ]]; then
  selector="${2:?usage: $0 --all SELECTOR{oracle|dfl} DEVICES}"
  devices="${3:?usage: $0 --all SELECTOR{oracle|dfl} DEVICES}"
  if [[ "${selector}" != "oracle" && "${selector}" != "dfl" ]]; then
    echo "selector must be oracle or dfl" >&2
    exit 2
  fi
  run_grid "${selector}" "${devices}"
  exit 0
fi

if [[ "${1:-}" == "--all-both" ]]; then
  devices="${2:?usage: $0 --all-both DEVICES}"
  run_grid oracle "${devices}"
  run_grid dfl "${devices}"
  exit 0
fi

selector="${1:?usage: $0 SELECTOR{oracle|dfl} WEIGHT{0|1|5|20} [SEED] [DEVICES]}"
weight="${2:?usage: $0 SELECTOR{oracle|dfl} WEIGHT{0|1|5|20} [SEED] [DEVICES]}"
seed="${3:-0}"
devices="${4:-0}"
if [[ "${selector}" != "oracle" && "${selector}" != "dfl" ]]; then
  echo "selector must be oracle or dfl" >&2
  exit 2
fi
case "${weight}" in 0|1|5|20|0.0|1.0|5.0|20.0) ;; *) echo "weight must be one of 0, 1, 5, 20" >&2; exit 2;; esac
run_one "${selector}" "${weight}" "${seed}" "${devices}"
