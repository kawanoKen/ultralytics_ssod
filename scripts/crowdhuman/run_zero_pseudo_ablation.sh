#!/usr/bin/env bash
# Zero-pseudo-label-batch ablation: isolates whether treating a zero-adopted-pseudo-label batch
# as "train the whole anchor grid as background, unnormalized" is the trigger for the
# Detect-head/BN collapse documented in scripts/crowdhuman/bn_mismatch_diagnosis2.py.
#
# Three runs, one variable changed at a time:
#   1. yolov8n_1p_baseline_noskip : current behavior (no fix) on 1% labeled -- reproduces the
#                                    original collapse for direct comparison
#   2. yolov8n_1p_skip            : identical to (1) except skip_zero_pseudo_cls_loss=True
#   3. yolov8n_10p_skip           : same fix applied to the 10% labeled setting
#
# All three log full ssod_diagnostics.py output (loss decomposition, assignment stats, EMA
# health, grad norm pre/post clip, BN mismatch probe on model.21/model.22) at diag_interval=1
# (every step) so the pre/post-collapse dynamics are fully reconstructable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DEVICES="0,1,2,3"
BATCH=256
SAVE_PERIOD=10
DIAG_INTERVAL=1

echo "=== [1/3] 1% baseline, no fix (reproduces collapse) ==="
uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolov8n \
  --model runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt \
  --data crowdhuman_1p.yaml --project runs/crowdhuman_ssod_1p_zero_pseudo_ablation \
  --name yolov8n_1p_baseline_noskip \
  --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
  --diag-interval "${DIAG_INTERVAL}"
echo "=== [1/3] done ==="

echo "=== [2/3] 1% with skip_zero_pseudo_cls_loss ==="
uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolov8n \
  --model runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt \
  --data crowdhuman_1p.yaml --project runs/crowdhuman_ssod_1p_zero_pseudo_ablation \
  --name yolov8n_1p_skip \
  --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
  --diag-interval "${DIAG_INTERVAL}" --skip-zero-pseudo-cls-loss
echo "=== [2/3] done ==="

echo "=== [3/3] 10% with skip_zero_pseudo_cls_loss ==="
uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolov8n \
  --model runs/crowdhuman_labeled_baseline/yolov8n_crowdhuman_labeled/weights/best.pt \
  --data crowdhuman_10p.yaml --project runs/crowdhuman_ssod_10p_zero_pseudo_ablation \
  --name yolov8n_10p_skip \
  --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
  --diag-interval "${DIAG_INTERVAL}" --skip-zero-pseudo-cls-loss
echo "=== [3/3] done ==="

echo "=== All zero-pseudo ablation runs finished. ==="
