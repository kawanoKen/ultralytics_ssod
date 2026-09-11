# Perturbation Ablation — DFL Shape vs Generic Assignment Sensitivity

Date: 2026-09-10

## Question

Does assignment instability predict GT assignment error because its perturbations are specifically supported by
the DFL distribution, or would generic geometric box perturbations provide the same signal?

No training was performed. All three conditions use the same existing YOLOv8n CrowdHuman 5% supervised checkpoint,
500 deterministically evenly spaced validation images, 640 px inference, confidence threshold 0.6, GT matching,
TaskAlignedAssigner configuration, and M=9 candidate count.

## Controls

| Condition | Definition |
|---|---|
| Fixed 5% | Baseline plus each x edge moved by ±5% of box width and each y edge by ±5% of box height |
| Width-matched | Each DFL distance moved symmetrically by ±(Q90−Q10)/2 around its expectation; clipped to valid DFL distance range |
| DFL-supported | Per-edge discrete Q10 and Q90 locations from the complete DFL distribution |

The baseline candidate is the identical NMS box in every condition. Width-matched preserves the DFL-derived
per-object/per-edge perturbation magnitude but removes quantile asymmetry and locations. Fixed 5% removes DFL from
the perturbation definition entirely.

## Fair evaluation universe

AUROC comparison uses exactly the same 70,305 dense predictions and the same 10,248 GT assignment-error labels in
all conditions. A prediction is included when either baseline pseudo assignment or its matched GT assignment is
foreground. Perturbation-only positives are deliberately not used to select the evaluation population, because
otherwise each perturbation method changes its own test set.

Unmatched pseudo/GT objects are excluded from this primary localization-assignment analysis to avoid conflating
object-existence error with assignment uncertainty.

## Results

| Perturbation | Mean object instability | Identity-switch events | AUROC | AUPRC | ΔAUROC vs fixed |
|---|---:|---:|---:|---:|---:|
| Fixed 5% | 0.0386 | 177 | 0.6960 | 0.4076 | — |
| DFL-supported | 0.1148 | 1,021 | 0.7246 | 0.4108 | +0.0286 |
| Width-matched | 0.0833 | 697 | **0.7734** | **0.4555** | +0.0774 |

Differences from DFL-supported:

- Width-matched − DFL-supported AUROC: **+0.0488**
- Width-matched − DFL-supported AUPRC: **+0.0448**
- DFL-supported − fixed 5% AUROC: **+0.0286**
- DFL-supported − fixed 5% AUPRC: **+0.0032**

Assignment error rate rises monotonically with instability for every condition:

| Instability bin | Fixed 5% | DFL-supported | Width-matched |
|---|---:|---:|---:|
| 0.0–0.2 | 11.7% | 9.8% | 9.9% |
| 0.2–0.4 | 63.4% | 45.6% | 56.0% |
| 0.4–0.6 | 69.0% | 69.3% | 65.7% |
| 0.6–0.8 | 80.9% | 77.1% | 81.2% |
| 0.8–1.0 | 100% (n=1) | 100% (n=1) | 100% (n=1) |

## Interpretation

Generic assignment sensitivity is real: fixed 5% perturbations already reach AUROC 0.6960. Therefore the result
cannot be attributed solely to detailed DFL distribution shape.

The DFL-supported method improves AUROC modestly over fixed 5%, but width-matched symmetric perturbation performs
best. This indicates that the useful DFL contribution in this experiment is primarily its adaptive, per-edge
perturbation magnitude. The asymmetric Q10/expectation/Q90 locations and other distribution-shape details provide no
additional benefit here and may make the sensitivity estimate noisier.

The defensible current claim is therefore:

> Assignment sensitivity predicts localization-assignment error, and adapting perturbation magnitude using DFL
> spread is more effective than a fixed 5% geometric perturbation in this CrowdHuman setting.

The current evidence does **not** support the stronger claim that DFL-supported quantile shape is uniquely required.

## Consequence for the method

Before full training, add a width-matched R1/R2 variant. Training only the original DFL-supported R1/R2 would test a
weaker analysis method than the best control. The primary method comparison should include:

1. Existing confidence-only SSOD
2. Existing DFL-selection SSOD
3. R1/R2 with fixed 5% sensitivity
4. R1/R2 with width-matched sensitivity
5. R1/R2 with DFL-supported sensitivity

## Limitations and next checks

- Fixed 5% is one predeclared scale, not a scale sweep. Avoid tuning it on these GT results.
- Point estimates are reported here; paired image-level bootstrap confidence intervals should be added for a paper.
- The ablation currently covers YOLOv8n, CrowdHuman 5%, and 500 validation images. Repeat the frozen comparison on
  CrowdHuman 1%/10%, YOLO11, and VOC before claiming generality.
- Width-matched still uses DFL spread. It isolates magnitude from detailed shape, but is not a completely DFL-free
  adaptive control.

## Artifacts

- DFL-supported: `results/dfl_assignment/gate_a_crowdhuman5_yolov8_640/`
- Fixed 5%: `results/dfl_assignment/ablation_fixed5_crowdhuman5_yolov8_640/`
- Width-matched: `results/dfl_assignment/ablation_width_matched_crowdhuman5_yolov8_640/`
- Reproducible analyzer: `scripts/dfl_assignment/analyze_gate_a.py`
