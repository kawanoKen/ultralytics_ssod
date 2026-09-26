#!/usr/bin/env bash
# Safe class-confidence-only SSOD (LabelMatch ACT, no DFL/localization gate) with full epoch-1
# per-sample logging. Usage: bash scripts/crowdhuman/run_safe_clsconf_act.sh 1p   (or 5p / 10p)
#
# Frozen settings (all seeds identical; only --seed differs):
#   selection : ACT class-wise thresholds on classification confidence only
#               (t_c = K_c-th score with K_c = rho_L * N_probe; t_c^r = top 20% of candidates;
#                [t_c, t_c^r) ignored for cls; below t_c discarded)
#   stability : skip_zero_pseudo_cls_loss, ssod_weight=0.25 (best 1%/10% in the B ablation)
#   optimizer : AdamW lr0=0.001 momentum=0.9 warmup_bias_lr=0 warmup 3 epochs -- set EXPLICITLY,
#               because optimizer=auto ignores lr0 and flips to SGD lr=0.01 when iterations>10000
#   data      : crowdhuman_<ratio>_local.yaml (resolved via this machine's datasets_dir; the
#               committed crowdhuman_<ratio>.yaml hard-codes a path whose labels are missing here)
#   logging   : all existing diagnostics every step (--diag-interval 1) + epoch_samples/ for epoch 1
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RATIO="${1:?usage: $0 1p|5p|10p}"
case "${RATIO}" in
  1p)  MODEL="runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt" ;;
  5p)  MODEL="runs/crowdhuman_labeled_baseline_5p/yolov8n_crowdhuman_labeled/weights/best.pt" ;;
  10p) MODEL="runs/crowdhuman_labeled_baseline/yolov8n_crowdhuman_labeled/weights/best.pt" ;;
  *) echo "unknown ratio ${RATIO}"; exit 1 ;;
esac
DEVICES="${DEVICES:-0,1,2,3}"
BATCH="${BATCH:-128}"
PROJECT="runs/crowdhuman_ssod_${RATIO}_safe_clsconf_act"

for SEED in 0 1 2; do
  NAME="yolov8n_${RATIO}_act_w025_adamw1e-3_seed${SEED}"
  echo "=== ${NAME} ==="
  uv run python scripts/voc_ssod/train_ssod.py --variant labelmatch_act --arch yolov8n \
    --model "${MODEL}" --data "crowdhuman_${RATIO}_local.yaml" --project "${PROJECT}" --name "${NAME}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --seed "${SEED}" --save_period 10 \
    --optimizer AdamW --lr0 0.001 --momentum 0.9 --warmup-bias-lr 0.0 --warmup-epochs 3.0 \
    --skip-zero-pseudo-cls-loss --ssod-weight 0.25 \
    --diag-interval 1 --sample-log-epochs 1
  echo "=== ${NAME} done (exit $?) ==="
done
