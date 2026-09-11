# DFL Assignment Stability — Work Summary

## 1. Executive summary

Step 0 and the reusable Step 1/2/R1/R2 implementation are complete. Unit tests and CPU real-model loss smoke tests
pass for YOLOv8n and YOLO11n. A 14-step real CrowdHuman trainer smoke also completed for both R1 and R2.
Dataset-scale analysis, Gate A, GPU profiling, and full training are pending because the current container exposes
no NVIDIA device. No full training was started without satisfying Gate A.

## 2. Existing implementation audit

See `step0_implementation_audit.md`.

## 3. Deterministic perturbation sanity check

`ultralytics/utils/assignment_stability.py` implements expectation plus eight one-edge Q10/Q90 boxes using the exact
DFL anchor/stride decode. Synthetic sharp distributions have zero Q90−Q10 width; uniform 16-bin distributions have
width 13 bins. Image-level 30-instance visualization remains pending.

## 4. Does DFL ambiguity produce assignment instability?

Measured at 640 px on 500 deterministically spaced CrowdHuman validation images using the existing YOLOv8n 5%
supervised checkpoint. DFL confidence itself was unrelated to object instability (rho=0.0289), while mean Q90-Q10
width had a statistically clear but weak positive relationship (rho=0.1732, p=7.44e-38). This did not pass the
predeclared Condition A effect-size threshold of 0.20.

## 5. Does assignment instability predict GT assignment error?

On 5,183 IoU/class-matched pseudo objects and 89,620 relevant dense predictions, instability predicted GT assignment
error with AUROC 0.6897 and AUPRC 0.3230. Error rate rose from 10.2% in the 0.0–0.2 instability bin to 34.5%, 57.9%,
67.0%, and 100% in successively higher bins (the last bin contains only one prediction). Condition B passed.

## 6. CrowdHuman vs VOC / ratios

The 5% CrowdHuman primary Gate A analysis is complete. The 1%/10% and VOC control analyses remain pending.

## 7. New method implementation

- R1: same-object stability weights IoU/DFL responsibility; effective weights are renormalized.
- R2: R1 plus baseline-negative/perturbed-positive anchors ignored in classification BCE.
- Classification target is never replaced by stability.
- Foreground/background flips and object-identity switches are logged separately.
- Per-batch JSONL includes counts, mean/median/histogram, effective regression weight, assigner time, iteration time,
  and CUDA memory when available.

## 8. Verification

Four deterministic unit tests pass. CPU forwards through both local YOLOv8n and YOLO11n checkpoints produce finite
R1/R2 losses. Real YOLOv8n/CrowdHuman 5% runs at 64 px completed 14 forward/backward updates for each method without
NaN/inf. Only two pseudo-label-bearing steps occurred at this deliberately tiny resolution, yielding 20 baseline
positives, 9 ambiguous negatives, and mean active-step stability 0.8722. This confirms code-path behavior but is far
too small and distribution-shifted to support Gate A. DDP, mixed precision, and GPU memory remain unverified.

The CPU smoke exposed and fixed two boundary issues: the assigner's zero-object mask is float rather than bool, and
the pseudo-label image loop previously shadowed the outer batch index used by running-loss/logging code. Ultralytics
still launches final validation even with `val=False`; it was interrupted after training completed because validating
all 4,370 CrowdHuman images was outside the smoke-test scope.

## 9. Go / No-Go conclusion

**GO by Condition B.** Condition A did not pass, but assignment instability predicted matched-object GT assignment
error above the frozen AUROC threshold. Full training is scientifically authorized, but cannot be launched here
because no GPU device is visible.

## 10. Recommended next experiment

On a GPU-visible host, first run checkpoint-only CrowdHuman 5% assignment analysis and the required sanity plots.
Only if Condition A or B holds, run the short YOLOv8 CrowdHuman 5% R1/R2 smoke commands documented by the launcher,
then profile M=9 overhead before full training.
