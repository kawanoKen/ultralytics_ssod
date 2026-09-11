# Step 0 — Existing Implementation Audit

Audit date: 2026-09-09. No training was run during this audit. Both local `yolov8n.pt` and `yolo11n.pt` were
loaded on CPU to verify their actual detection-head interface.

## 1. Teacher → pseudo-label → assigner → loss

The shared SSOD path is `SSODTrainer._do_train` in
`ultralytics/models/yolo/detect/ssod_train.py`.

1. The unlabeled weak/geometric view is passed through `self.teacher.ema` under `torch.no_grad()` around lines
   905–906. The returned tuple contains decoded inference predictions and the raw multi-scale feature-head outputs.
2. Raw outputs are flattened scale-by-scale and split into `4 * reg_max` regression logits and `nc`
   classification logits around lines 913–916.
3. `non_max_suppression(..., return_idxs=True)` around lines 921–923 returns each surviving box and its original
   dense prediction index.
4. The raw DFL row at that index is reshaped to `(N, 4, reg_max)` around line 939. Ordering is
   left/top/right/bottom, confirmed by `v8DetectionLoss.bbox_decode` and `dist2bbox`.
5. Pseudo boxes are converted from pixel XYXY to normalized XYWH around lines 966–969.
6. The student receives `img_strong` when available; weak and strong views share geometry, so no extra box transform
   is needed. `EfficientTeacherLoss` is called around lines 993–1003.
7. The supervised and unlabeled losses are combined as
   `supervised + ssod_weight * unlabeled` around line 1028. The EMA teacher is updated after optimizer steps.

YOLOv8n and YOLO11n both use the same `Detect` interface here: `reg_max=16`, strides 8/16/32, and, for a 64×64
input, feature shapes `(B, 144, 8, 8)`, `(B, 144, 4, 4)`, `(B, 144, 2, 2)`. Their backbones/necks are not assumed
identical; only this verified head contract is shared.

## 2. Raw DFL logits ↔ NMS boxes

The key invariant is that concatenation of raw head outputs uses the same scale/grid ordering as decoded inference
output. `return_idxs=True` preserves this dense row through NMS. The selected index is reused for DFL confidence,
anchor point, stride, and deterministic perturbations. This avoids geometric nearest-neighbor matching after NMS.

## 3. Existing DFL confidence

`ultralytics/utils/dfl_confidence.py::localization_confidence` applies softmax over the DFL-bin dimension. For each
edge it sums the argmax-bin mass and the larger valid adjacent-bin mass. Box confidence is the minimum of the four
edge confidences. In `EfficientTeacherLoss._get_reliable_and_unreliable_mask`, `use_loc_conf=True` adds the fixed
box-confidence threshold to the classification-confidence reliable mask. Independently, `use_edge_conf=True` can
mask low-confidence edges from DFL loss without dropping the IoU term or the object.

## 4. Assigner

The actual assigner is `ultralytics/utils/tal.py::TaskAlignedAssigner`, constructed by
`v8DetectionLoss.__init__` with `topk=13`, `alpha=0.5`, `beta=6.0` around `ultralytics/utils/loss.py:215`.

Inputs are sigmoid classification scores `(B,A,C)`, decoded student boxes `(B,A,4)` in pixels, anchor/grid points
in pixels, padded object labels/boxes, and a valid-object mask. Candidate anchors must lie inside an object box.
The task-aligned metric is class score^alpha × CIoU^beta; top-k candidates are retained. If one prediction is a
candidate for multiple objects, `select_highest_overlaps` resolves it by maximum overlap. Outputs include target
labels, target boxes, normalized target scores, foreground mask, and a per-image local target-object index.

## 5. Loss and normalization

Classification is elementwise BCE over positive and negative anchors. Anchors assigned only to the intermediate
confidence (`unreliable`) set are ignored. Its divisor is reliable target-score sum, clamped to at least one.
Standard IoU and DFL terms use `target_scores.sum(-1)` as per-positive weights and the same target-score sum as
divisor. Gains are `box`, `cls`, and `dfl`, then all three terms are multiplied by batch size. The outer trainer
applies `ssod_weight`.

For R1/R2, localization is multiplied by same-object stability and divided by the resulting effective weight sum.
Thus stability does not merely lower the whole regression loss scale. Classification targets remain unchanged.

## 6. Minimal insertion points and sharing

The minimal teacher-side insertion is immediately after NMS indexing, where raw DFL distributions, selected anchor
points, and stride are all available. The minimal loss-side insertion is immediately after baseline assignment:
run the same assigner for eight additional target-box sets, compare local object identity, then alter localization
responsibility (R1) and optionally the negative BCE mask (R2). Both locations are shared by the verified YOLOv8 and
YOLO11 head contract.

## 7. Concerns

- Nine assigner calls replace one for the reliable set; runtime/VRAM must be measured on the GPU host.
- Discrete quantiles use the first bin reaching the CDF threshold, deliberately without claiming calibrated
  boundary coverage or sampling probability.
- Current execution container has PyTorch CUDA 12.8 but zero visible CUDA devices; `/dev/nvidia*` is absent and
  `nvidia-smi` cannot communicate with the driver.
- Dataset caches and checkpoints exist, but Gate A requires real checkpoint analysis before any full R1/R2 run.
- DDP and full-resolution crowded batches remain unverified for the new path until a GPU smoke test is possible.
