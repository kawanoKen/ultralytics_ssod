# R2 pre-check: ambiguous negatives

## Setup

- Model: YOLOv8n supervised CrowdHuman 5% checkpoint
- Evaluation: the same 500 deterministically spaced CrowdHuman validation images used by the R1 analysis
- Image size: 640 px
- Pseudo-label confidence threshold: 0.6
- Assigner: the same `TaskAlignedAssigner(topk=13, alpha=0.5, beta=6.0)` for pseudo and GT boxes
- Perturbation: the shared Width-matched implementation (`Q90-Q10`, symmetric half-width in DFL distance space)
- Evaluation universe: all 8,400 dense predictions per image, or 4,200,000 predictions total

No R2 training or threshold tuning was performed.

## 1. Assignment mismatch type breakdown

Pseudo-object identities were class-aware one-to-one mapped to GT at IoU >= 0.5 before testing Type 3 identity.

| Assignment outcome | Count | Fraction of all predictions |
|---|---:|---:|
| All evaluated dense predictions | 4,200,000 | 100% |
| All mismatches (Type 1 + 2 + 3) | 56,185 | 1.338% |
| Type 1: pseudo positive -> GT negative | 6,803 | 0.162% |
| Type 2: pseudo negative -> GT positive | 46,986 | 1.119% |
| Type 3: pseudo object A -> GT object B | 2,396 | 0.057% |
| Other agreement | 4,143,815 | 98.662% |

Type 2 accounts for 83.63% of all assignment mismatches, so missed positive responsibility is the dominant mismatch
mode in this dense-assignment evaluation.

## 2. R2 ambiguous-negative precision and recall

The mask was evaluated exactly as proposed:

`M = {j: A_base(j) is negative and at least one Width-matched candidate assigns j positive}`.

| Metric | Result |
|---|---:|
| Ambiguous-negative mask size, `|M|` | 16,887 |
| Members of `M` that are GT-positive | 4,535 |
| Precision, `GT-positive / |M|` | **0.26855 (26.85%)** |
| Recall over all Type 2 errors | **0.09652 (9.65%)** |

Thus the mask finds 4,535 genuine pseudo-negative/GT-positive predictions, but it also contains 12,352 GT-negative
predictions. As an unconditional classification-ignore mask, most affected negatives would therefore be correct
negatives, while more than 90% of Type 2 errors remain uncovered.

## 3. GT-positive rate by instability

For a baseline-negative prediction, instability is its foreground frequency across the baseline plus eight candidates.
The final interval includes 1.0.

| Instability | n | GT-positive | GT-positive rate |
|---|---:|---:|---:|
| 0.0-0.2 | 12,005 | 2,532 | 21.09% |
| 0.2-0.4 | 4,484 | 1,776 | 39.61% |
| 0.4-0.6 | 379 | 213 | 56.20% |
| 0.6-0.8 | 18 | 13 | 72.22% |
| 0.8-1.0 | 1 | 1 | 100.00% |

GT-positive rate increases monotonically with instability. However, 97.64% of the mask (16,489/16,887) lies below
0.4; the apparently high-purity bins at or above 0.4 contain only 398 predictions in total.

## 4. Is a one-seed R2 run worthwhile?

The exact, unthresholded R2 mask is **not strongly supported for a full one-seed run** by this pre-check. Its precision
is only 26.85% and its Type 2 recall is 9.65%, so blindly ignoring classification loss for all members of `M` would
mostly ignore GT-negative predictions while correcting only a small portion of the target error mode. The monotonic
instability trend is scientifically interesting and suggests that a separately predeclared selective R2 hypothesis
could be investigated later, but choosing a cutoff from these GT results would be threshold tuning and was not done.

Therefore the recommendation for the currently specified R2 definition is **NO-GO**. R2 training was not started.

## Artifacts and reproducibility note

- Detailed rerun: `results/dfl_assignment/r2_precheck_width_matched_640/`
- Machine-readable counts: `results/dfl_assignment/r2_precheck_width_matched_640/summary.json` under `r2_precheck`
- Analyzer: `scripts/dfl_assignment/analyze_gate_a.py`

The full-dense rerun produced 5,437 pseudo objects and 70,333 matched-domain predictions, versus 5,436 and 70,305
in the earlier saved Width-matched run. The resulting Gate-A AUROC changed only from 0.773388 to 0.773446. This tiny
GPU/NMS boundary variation does not affect the R2 conclusion; all numbers above come from the single internally
consistent full-dense rerun.
