# CrowdHuman GT clip後のDFL edge-confidence再解析

既存checkpointのみを用いて、GT依存の全解析を再実行した。追加学習は行っていない。CrowdHumanの `fbox` と `vbox` は画像外へ伸びる場合があるため、IoU matching、edge error、object size、edge occlusionを計算する前に、XYXY座標の各辺を `[0, image width]` / `[0, image height]` へclipした。prediction側は従来から同じ座標範囲へclipされている。

旧出力は保持し、clip版を `results/dfl_edge_effect_analysis_clipped/` に保存した。対象は1%・10%のpseudo-label群統計、edge occlusionとの関係、final checkpointのedge localization、および1% random-ablationのvalidation pseudo edge errorである。

## 最優先: 1% final strong-bottom

`delta = DFL edge error - confidence-only edge error` であり、負値がDFL版の改善を表す。各modelで同一GTに両方がone-to-one matchしたedgeだけを比較している。

| DFL seed | n | mean delta (px) | median delta (px) | improved rate |
|---|---:|---:|---:|---:|
| 0 | 12,013 | -2.04 | -0.31 | 51.7% |
| 1 | 12,568 | -2.00 | -0.53 | 52.2% |
| 2 | 12,456 | -1.47 | -0.15 | 50.9% |
| 3-seed mean ± SD | — | **-1.84 ± 0.32** | — | — |

結論として、**1%のfinal strongly-occluded bottom edge改善はGT clip後にも3 seedすべてで残る**。旧結果の3-seed平均は `-1.73 ± 0.26 px` であり、clip後も効果の符号・規模は維持された。

ただしclipによりstrong-bottom GT edge数は37,747から32,599へ変化し、absolute errorも境界外GTを含んだ旧値より小さくなった。従って、absolute errorの旧表は置き換える必要があるが、pairedなDFL対baseline比較の結論は変わらない。

| model | matched strong-bottom edges | mean edge error (px) | median edge error (px) |
|---|---:|---:|---:|
| confidence-only | 14,659 | 34.49 | 18.21 |
| DFL seed 0 | 15,204 | 33.17 | 17.96 |
| DFL seed 1 | 16,439 | 33.92 | 17.85 |
| DFL seed 2 | 16,442 | 33.05 | 17.52 |

10%のstrong-bottom paired deltaはseed 0/1/2でそれぞれ `+0.03`, `+0.20`, `+0.52 px` であり、1%と異なり改善は支持されない。

## pseudo edgeのGT error

1% validationで、DFL-mask判定を適用したbaseline teacher pseudo edgeをGTと比較した。clip後も、DFL-masked edgeはnon-masked edgeより明確に大きなpseudo-to-GT errorを持つ。

| visibility | mask status | n | mean edge error (px) | mean DFL confidence |
|---|---|---:|---:|---:|
| visible | DFL-masked | 131 | 77.07 | 0.515 |
| visible | non-masked | 160,271 | 9.36 | 0.988 |
| occluded | DFL-masked | 295 | 52.64 | 0.507 |
| occluded | non-masked | 44,111 | 19.57 | 0.968 |

occluded bottomだけを見ても、DFL-maskedは49.14 px、non-maskedは33.69 pxである。旧解析で見えた逆転は未clip GTが原因であり、clip後には残らない。

## 遮蔽とDFL confidence

edge occlusion量とDFL confidenceのPearson相関は、1%でL/T/R/B = `-0.093 / -0.067 / -0.095 / -0.302`、10%で `-0.136 / -0.089 / -0.144 / -0.364` だった。bottomで最も強い負の関係はclip後も維持された。

## 出力

- [1% final edge summary](dfl_edge_effect_analysis_clipped/1p_final_edge_error_summary.csv)
- [1% paired delta summary](dfl_edge_effect_analysis_clipped/1p_final_paired_delta_summary.csv)
- [10% final edge summary](dfl_edge_effect_analysis_clipped/10p_final_edge_error_summary.csv)
- [10% paired delta summary](dfl_edge_effect_analysis_clipped/10p_final_paired_delta_summary.csv)
- [1% pseudo group statistics](dfl_edge_effect_analysis_clipped/1p_pseudo_group_stats.csv)
- [10% pseudo group statistics](dfl_edge_effect_analysis_clipped/10p_pseudo_group_stats.csv)
- [1% pseudo masked-edge error](dfl_edge_mask_random_ablation/pseudo_masked_edge_error_1p_clipped.csv)

実装は [analyze_dfl_edge_effect.py](../scripts/crowdhuman/analyze_dfl_edge_effect.py) の `gt_features()` に集約しており、[analyze_masked_pseudo_edge_error.py](../scripts/crowdhuman/analyze_masked_pseudo_edge_error.py) も同じ関数を利用する。
