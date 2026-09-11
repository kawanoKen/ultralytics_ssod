#!/usr/bin/env bash
# Supervised-only convergence baseline on VOC, before adding SSOD.
#
#   labeled   = VOC07 trainval (~5011 images, used for training)
#   unlabeled = VOC12 trainval (~11540 images, listed in VOC_ssod.yaml as
#               'ssod_train' but NOT used here -- this run is plain supervised
#               training via the standard trainer, not SSODTrainer)
#   eval      = VOC07 test (~4952 images, used as 'val')
#
# Trains YOLOv8n and YOLOv11n sequentially, each DDP-distributed across all 4
# GPUs, fine-tuning from COCO-pretrained weights. Adjust DEVICES/BATCH below
# to match the GPUs actually free on the machine.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA="VOC_ssod.yaml"
DEVICES="0,1,2,3"       # 4x RTX 3090, DDP
EPOCHS=100
IMGSZ=640
BATCH=384               # total batch size across all GPUs (96/gpu, per bench_batch_size.sh sweep: 128/gpu peaks but leaves no OOM margin)
SAVE_PERIOD=10          # save a checkpoint every 10 epochs
PROJECT="runs/voc_labeled_baseline"

run() {
  local model=$1
  local name=$2
  echo "==> Training ${model} (labeled-only baseline) -> ${PROJECT}/${name}"
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

run yolov8n.pt yolov8n_voc07_labeled
run yolo11n.pt yolo11n_voc07_labeled
