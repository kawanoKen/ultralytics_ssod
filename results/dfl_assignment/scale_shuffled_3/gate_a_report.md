# Gate A Report — CrowdHuman 5% / YOLOv8 / 640px / shuffled_width

This is checkpoint-only analysis: no optimizer, backward pass, EMA update, or training was performed.

## Scope

- Checkpoint: `runs/crowdhuman_labeled_baseline_5p/yolov8n_crowdhuman_labeled/weights/best.pt`
- Perturbation: `shuffled_width`
- Images: 500 deterministically evenly spaced validation-list images (no result-driven selection)
- Pseudo-object confidence threshold: 0.6
- Pseudo objects: 5437
- Relevant dense predictions: 70333

## Step 2 — Deterministic assignment sensitivity

Each pseudo object was decoded into M=9 fixed candidates: expectation plus one-edge Q10/Q90 changes. The same
TaskAlignedAssigner (`topk=13`, `alpha=0.5`, `beta=6.0`) was rerun for every candidate. Object identity was retained;
identity switches were not merged with foreground/background flips.

- Mean object Jaccard: 0.9179
- Mean object instability: 0.0821
- Mean ambiguous-prediction ratio: 0.3769
- Object-level identity-switch events: 613
- Matched-domain ambiguous-negative predictions: 2592
- Matched-domain identity-switch predictions: 589

## Step 3 — DFL shape relation

- DFL confidence vs instability: Spearman rho=0.0351,
  p=0.009554
- Mean Q90-Q10 width vs instability: Spearman rho=-0.0061,
  p=0.6515

Condition A was operationalized before inspection as |rho|≥0.20 in the expected direction, p<0.01, with at least
30 pseudo objects. Result: **FAIL**.

## Step 4 — GT assignment error relation

Pseudo objects were class-aware one-to-one matched to GT at IoU≥0.5. The GT boxes were passed through the same
assigner. Baseline pseudo identity (mapped to GT) and foreground/background were compared per relevant dense
prediction. Unmatched pseudo/GT objects were excluded from this primary H2 metric so object-existence errors were
not conflated with localization-assignment errors.

- Assignment errors: 10263 / 70333
- Matched pseudo objects: 5185 / 5437
- Instability detecting assignment error: AUROC=0.7572, AUPRC=0.4316
- Error rate by instability bin: `{"0.0-0.2": {"n": 63761, "error_rate": 0.10247643543859099}, "0.2-0.4": {"n": 5322, "error_rate": 0.5450958286358512}, "0.4-0.6": {"n": 1161, "error_rate": 0.6537467700258398}, "0.6-0.8": {"n": 87, "error_rate": 0.7701149425287356}, "0.8-1.0": {"n": 2, "error_rate": 1.0}}`

Condition B was operationalized as AUROC≥0.60 with at least 100 relevant predictions. Result:
**PASS**.

## Gate A decision

**GO**. Full training is authorized by this report only when Condition A or B passes. This report
does not interpret DFL bins as calibrated boundary probabilities or assignment stability as object confidence.
