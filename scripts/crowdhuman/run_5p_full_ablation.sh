#!/usr/bin/env bash
# Full A/B/C/D/E normalization ablation for CrowdHuman 5% labeled, mirroring what was already
# run for 1% and 10% (see run_zero_pseudo_ablation.sh for A, run_b_weight_ablation.sh /
# run_cd_ablation.sh for B/C/D, and the standalone yolov8n_{1p,10p}_fixed_denom runs for E).
#
# All conditions share: same init checkpoint (runs/crowdhuman_labeled_baseline_5p/.../best.pt),
# same seed, same LR/update count, same thresholds, same skip_zero_pseudo_cls_loss=True, same
# gradient clipping. Only the one variable under test differs per run:
#
#   A  yolov8n_5p_skip          : current normalization (target_score_sum), ssod_weight=0.5
#   B1 yolov8n_5p_weight010     : ssod_weight=0.1 (current normalization)
#   B2 yolov8n_5p_weight025     : ssod_weight=0.25 (current normalization)
#   C  yolov8n_5p_ema_denom     : cls_loss_denom=ema, ssod_weight=0.5
#   D  yolov8n_5p_loss_balance  : loss_balancing_mode=ema_scale, ssod_weight=0.5
#   E  yolov8n_5p_fixed_denom   : cls_loss_denom=fixed, ssod_weight=0.5
#
# Does NOT use `set -e` -- a crash in one run (e.g. condition A is expected to behave like the
# 10% case and may be unstable) must not abort the rest of the queue.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DEVICES="0,1,2,3"
BATCH=256
SAVE_PERIOD=10
DIAG_INTERVAL=100
MODEL="runs/crowdhuman_labeled_baseline_5p/yolov8n_crowdhuman_labeled/weights/best.pt"
DATA="crowdhuman_5p.yaml"
PROJECT="runs/crowdhuman_ssod_5p_zero_pseudo_ablation"

run() {
  local name=$1
  shift
  echo "=== ${name} ==="
  uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolov8n \
    --model "${MODEL}" --data "${DATA}" --project "${PROJECT}" --name "${name}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
    --diag-interval "${DIAG_INTERVAL}" --skip-zero-pseudo-cls-loss "$@"
  echo "=== ${name} done (exit $?) ==="
}

# A: current normalization (baseline for comparison)
run yolov8n_5p_skip

# B: fixed lower ssod_weight
run yolov8n_5p_weight010 --ssod-weight 0.1
run yolov8n_5p_weight025 --ssod-weight 0.25

# C: EMA denominator
run yolov8n_5p_ema_denom --cls-loss-denom ema

# D: EMA loss-scale balancing
run yolov8n_5p_loss_balance --loss-balancing-mode ema_scale

# E: fixed/reference denominator
run yolov8n_5p_fixed_denom --cls-loss-denom fixed

echo "=== 5% full A/B/C/D/E ablation finished. ==="
