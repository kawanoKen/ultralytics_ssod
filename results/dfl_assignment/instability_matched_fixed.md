# Instability-matched Fixed control

## 1. Ratio decision

GT AUROC/AUPRCを参照せず、既存のFixed 10% (`I=0.06774268`) とFixed 15% (`I=0.09890330`) から、
Object-average Widthのtarget `I=0.08432009`へ線形補間した。

`r0 = 0.10 + (0.08432009 - 0.06774268) / (0.09890330 - 0.06774268) * 0.05 = 0.12659994`

初回候補12.65999%のinstabilityは0.08380034（target差−0.00051975）だった。performance metricを見ず、
10%–初回候補間の局所instability傾きだけで1回修正し、最終ratioを固定した。許可された2回のうち、追加調整は
1回のみ使用した。

## 2. Final Fixed ratio

**12.74609%** (`0.127460909793`)

- Mean perturbation: 14.6590 px
- Mean perturbation / box size: 0.127461
- Instability histogram: `[62,980, 5,348, 1,767, 230, 8]` for bins
  `[0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0]`

## 3. Mean instability comparison

| Method | Mean instability | Difference (Object-average − Fixed) |
|---|---:|---:|
| Object-average Width | 0.08432009 | — |
| Instability-matched Fixed 12.74609% | 0.08434117 | −0.00002108 |

Absolute difference is `0.00002108`; the control is effectively instability-matched.

## 4. AUROC / AUPRC

Tie-aware metrics:

| Method | AUROC | AUPRC |
|---|---:|---:|
| Object-average Width | 0.76544 | 0.40179 |
| Instability-matched Fixed | **0.77097** | **0.41236** |

## 5. Image-paired bootstrap

The same 500 images were resampled as clusters 10,000 times (seed 20260910); prediction-level bootstrap was not used.
Differences are Object-average Width minus instability-matched Fixed.

| Comparison | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |
|---|---:|---:|---:|---:|
| Object-average − Instability-matched Fixed | **−0.00554** | **[−0.01100, −0.00027]** | **−0.01057** | **[−0.01698, −0.00421]** |

Both intervals are below zero: instability-matched Fixed is better on both metrics (Case 3).

## 6. Decision

**DFL-derived adaptive scale should not remain the central claim based on this setting.** Once mean assignment
instability is matched without looking at GT performance, Fixed perturbation significantly outperforms Object-average
Width. Together with the monotonic Fixed sweep, this indicates that the main signal is generic assignment sensitivity
at an appropriate perturbation scale, not a demonstrated advantage of DFL-derived object adaptation.

The earlier Width > Shuffled result still shows that object/scale correspondence changes the score, but it is not
sufficient to establish DFL adaptation as beneficial because an instability-matched non-adaptive Fixed control is
better. No training or additional variant was started.

## Artifacts

- Final prediction output: `results/dfl_assignment/instability_matched_fixed_final/`
- Initial interpolation check: `results/dfl_assignment/instability_matched_fixed_candidate/`
- Paired statistics: `results/dfl_assignment/perturbation_scale_ablation.json`
