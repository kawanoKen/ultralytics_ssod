# Gate A Report — CrowdHuman 5% / YOLOv8 / 640px / object_average

This is checkpoint-only analysis: no optimizer, backward pass, EMA update, or training was performed.

## Scope

- Checkpoint: `runs/crowdhuman_labeled_baseline_5p/yolov8n_crowdhuman_labeled/weights/best.pt`
- Perturbation: `object_average`
- Images: 500 deterministically evenly spaced validation-list images (no result-driven selection)
- Pseudo-object confidence threshold: 0.6
- Pseudo objects: 5437
- Relevant dense predictions: 70333

## Step 2 — Deterministic assignment sensitivity

Each pseudo object was decoded into M=9 fixed candidates: expectation plus one-edge Q10/Q90 changes. The same
TaskAlignedAssigner (`topk=13`, `alpha=0.5`, `beta=6.0`) was rerun for every candidate. Object identity was retained;
identity switches were not merged with foreground/background flips.

- Mean object Jaccard: 0.9157
- Mean object instability: 0.0843
- Mean ambiguous-prediction ratio: 0.3630
- Object-level identity-switch events: 668
- Matched-domain ambiguous-negative predictions: 2687
- Matched-domain identity-switch predictions: 633

## Step 3 — DFL shape relation

- DFL confidence vs instability: Spearman rho=-0.1590,
  p=3.914e-32
- Mean Q90-Q10 width vs instability: Spearman rho=0.2977,
  p=1.03e-111

Condition A was operationalized before inspection as |rho|≥0.20 in the expected direction, p<0.01, with at least
30 pseudo objects. Result: **PASS**.

## Step 4 — GT assignment error relation

Pseudo objects were class-aware one-to-one matched to GT at IoU≥0.5. The GT boxes were passed through the same
assigner. Baseline pseudo identity (mapped to GT) and foreground/background were compared per relevant dense
prediction. Unmatched pseudo/GT objects were excluded from this primary H2 metric so object-existence errors were
not conflated with localization-assignment errors.

- Assignment errors: 10263 / 70333
- Matched pseudo objects: 5185 / 5437
- Instability detecting assignment error: AUROC=0.7768, AUPRC=0.4601
- Error rate by instability bin: `{"0.0-0.2": {"n": 63334, "error_rate": 0.09775160261470932}, "0.2-0.4": {"n": 5403, "error_rate": 0.5595039792707754}, "0.4-0.6": {"n": 1444, "error_rate": 0.6447368421052632}, "0.6-0.8": {"n": 148, "error_rate": 0.7702702702702703}, "0.8-1.0": {"n": 4, "error_rate": 1.0}}`

Condition B was operationalized as AUROC≥0.60 with at least 100 relevant predictions. Result:
**PASS**.

## Gate A decision

**GO**. Full training is authorized by this report only when Condition A or B passes. This report
does not interpret DFL bins as calibrated boundary probabilities or assignment stability as object confidence.
