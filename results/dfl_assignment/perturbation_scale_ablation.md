# Perturbation-scale ablation for assignment sensitivity

## Setup and metric correction

YOLOv8n CrowdHuman 5%、supervised `best.pt`、640 px、confidence 0.6、NMS IoU 0.65、同じ500 validation
画像、同じTaskAlignedAssignerとGT matchingを用いた。全methodでbaseline/GTにより固定された70,333 dense
predictionsと10,263 mismatch（error rate 14.592%）を評価した。M=9で、trainingとthreshold tuningは行っていない。

解析中、旧`binary_auc`が9段階の同点instabilityを入力順のまま積分しており、標準的なtie-aware AUROC/
AUPRCではないことを発見した。本レポートの主表とpaired bootstrapは、AUROCで同点を0.5、AUPRCで同一
thresholdを一括処理した修正版を使用する。したがって旧レポートのWidth AUROC 0.7734/AUPRC 0.4553に
対応する修正値は0.7623/0.3955である。全methodを同じ保存predictionから再計算したため比較は一貫している。

## Table 1: Fixed sweep

Mean perturbationは8つのone-edge candidateで実際に動いた座標の平均絶対pixel変位。括弧内は対応するbox
width/heightに対する平均比率である。

| Method | Mean perturbation | Mean instability | AUROC | AUPRC | Mismatch rate |
|---|---:|---:|---:|---:|---:|
| Fixed 2.5% | 2.875 px (0.0250) | 0.02181 | 0.61530 | 0.27881 | 0.14592 |
| Fixed 5% | 5.750 px (0.0500) | 0.03864 | 0.68036 | 0.34454 | 0.14592 |
| Fixed 7.5% | 8.626 px (0.0750) | 0.05340 | 0.72129 | 0.38194 | 0.14592 |
| Fixed 10% | 11.501 px (0.1000) | 0.06774 | 0.74919 | 0.40157 | 0.14592 |
| Fixed 15% | 17.251 px (0.1500) | 0.09890 | **0.78079** | **0.41161** | 0.14592 |
| Width-matched | 9.235 px (0.1098) | 0.08338 | 0.76229 | 0.39552 | 0.14592 |

摂動量、mean instability、AUROCは概ね共に増加した。特にFixed 15%はWidth-matchedを上回るため、単に
大きく揺らす効果は強い。一方、平均pixel変位が最も近い事前指定点はFixed 7.5%（差0.609 px）であり、
WidthはこれをAUROC +0.04100、AUPRC +0.01357上回った。Fixed 10%はWidthより2.266 px大きいが、AUROCは
なお0.01310低い。

### Prediction-instability histogram

各セルは該当する70,333 predictionsの件数。

| Method | 0.0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| Fixed 2.5% | 68,080 | 1,809 | 437 | 7 | 0 |
| Fixed 5% | 66,494 | 2,973 | 816 | 49 | 1 |
| Fixed 7.5% | 65,217 | 3,866 | 1,153 | 96 | 1 |
| Fixed 10% | 64,149 | 4,572 | 1,442 | 164 | 6 |
| Fixed 15% | 61,972 | 6,051 | 2,012 | 289 | 9 |
| Width-matched | 63,524 | 5,484 | 1,225 | 99 | 1 |

## Table 2: DFL adaptation controls

Shuffled-widthはseed 0–9の平均±sample SD。L/T/R/Bを混ぜず、各FPN stride内で辺ごとに独立shuffleした。
これによりobject対応を破壊しつつDFL幅とnominal pixel変位の分布を維持した。clipping後の実変位は
9.212±0.003 pxで、Widthの9.235 pxとの差は0.25%だった。

| Method | Mean instability | AUROC | AUPRC |
|---|---:|---:|---:|
| Width-matched edge-specific | 0.08338 | 0.76229 | 0.39552 |
| Shuffled-width mean | 0.08277 ± 0.00021 | 0.74565 ± 0.00132 | 0.37358 ± 0.00202 |
| Object-average width | 0.08432 | **0.76544** | **0.40179** |
| Magnitude-matched Fixed 7.5% | 0.05340 | 0.72129 | 0.38194 |

10 seedsに対する平均のt-based 95% CIは、shuffle AUROC [0.74471, 0.74660]、AUPRC
[0.37213, 0.37502]、mean instability [0.08262, 0.08292]だった。全seedを使用し、選別していない。

## Table 3: image-paired bootstrap

同じ500画像をcluster単位で10,000回復元抽出した（bootstrap seed 20260910）。Shuffled比較では各bootstrap
sampleについて10 shuffle seedsのmetricを平均した。prediction単位bootstrapは使用していない。

| Comparison | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |
|---|---:|---:|---:|---:|
| Width − Fixed 5% | +0.08193 | [+0.07501, +0.08848] | +0.05098 | [+0.04190, +0.05968] |
| Width − magnitude-matched Fixed 7.5% | +0.04100 | [+0.03480, +0.04718] | +0.01357 | [+0.00544, +0.02139] |
| Width − Shuffled-width | +0.01663 | [+0.01246, +0.02089] | +0.02194 | [+0.01714, +0.02665] |
| Width − Object-average | −0.00315 | [−0.00758, +0.00134] | −0.00627 | [−0.01111, −0.00148] |

## Interpretation

1. **Magnitude confoundは実在する。** Fixed sweepでは摂動量とinstabilityを増やすだけでAUROCが大きく
   上昇し、Fixed 15%はWidthを上回った。元のFixed 5%との比較だけではDFL adaptationを主張できない。
2. **それでもobject-specific DFL correspondenceには追加情報がある（Case C）。** Widthとほぼ同じ
   instability・変位分布を持つShuffled-widthに対し、WidthのAUROC/AUPRC差は画像paired CIで共に正。
3. **Magnitude-comparable FixedにもWidthが勝る。** pixel変位が最も近いFixed 7.5%との差も両metricで
   CIが0を跨がない。ただしFixed sweepの最大値との比較ではない点を明示する必要がある。
4. **Edge-specific spreadの追加価値は支持されない（Case Dではない）。** Object-averageはWidthより
   AUROCがわずかに高く（差CIは0を跨ぐ）、AUPRCは有意に高い。少なくともこの設定ではobject-level平均幅で
   十分で、辺別DFL spreadが必要とは言えない。

したがって最も妥当な結論は、assignment sensitivityの性能には摂動scaleが大きく効くが、同程度の幅分布を
無作為に割り当てるよりobject-specific DFL spreadを保つ方が良い、というもの。一方、edge-specific shape
までは不要であり、研究の中核候補は「DFL-derived object-level adaptive scaleを用いたassignment sensitivity」
である。DFLのQ10/Q90は真のconfidence intervalとは解釈しない。

## Legacy metric reconciliation

旧実装値（同点未補正）は再現され、Fixed 2.5/5/7.5/10/15%のAUROCはそれぞれ
0.6320/0.6961/0.7353/0.7633/0.7912、Widthは0.7734、Object-averageは0.7768、Shuffled平均は0.7579だった。
順位と主結論はtie-aware修正後も同じだが、今後の報告には本レポートの修正値を使用する。

## Artifacts

- Machine-readable aggregate: `results/dfl_assignment/perturbation_scale_ablation.json`
- Per-method outputs: `results/dfl_assignment/scale_fixed_*`, `scale_width_matched`, `scale_object_average`
- Primary shuffled outputs: `results/dfl_assignment/scale_shuffled_stride_0` through `_9`
- Analyzer: `scripts/dfl_assignment/analyze_gate_a.py`
- Paired statistics: `scripts/dfl_assignment/summarize_scale_ablation.py`

初回のunrestricted cross-stride shuffleは、stride変更により平均pixel変位が9.23から9.47 pxへ変わることが
判明したため主解析から除外した。これは結果選択ではなく、事前指定されたmagnitude-preserving controlを満たす
ための実装修正である。
