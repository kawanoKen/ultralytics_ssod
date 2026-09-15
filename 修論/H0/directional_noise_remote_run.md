# H0 directional boundary-noise study

## 目的

clean GTの各boxに1辺だけnoiseを加えるH0で、edge noiseの方向性を操作する。外向きはleft/topを負方向、right/bottomを正方向へ動かしてboxを広げる向き、内向きはその反対である。

DFL targetの効果を切り分けるため、各新規distributionについてNoisy DFL（noisy boxのDFL target）とClean DFL（IoU/assignmentはnoisy box、DFL targetだけclean GT）の両方をseed 0/1で実行する。既存High symmetric（alpha=0.20、外向き確率0.5）は再利用する。

| tag | alpha | P(outward) | 目的 | 新規run数 |
|---|---:|---:|---|---:|
| a20_out100 | 0.20 | 1.00 | 外向き100%の最大効果 | 4 |
| a20_out0 | 0.20 | 0.00 | 内向き100%の非対称性 | 4 |
| a20_out80 | 0.20 | 0.80 | 外向きbiasの用量反応 | 4 |
| a40_out50 | 0.40 | 0.50 | 対称noiseの上限 | 4 |
| a20_out50 | 0.20 | 0.50 | 既存High symmetric control | 0 |

新規は計16 runである。noise絶対値は各対象box辺の `Uniform[0.5, 1.5] × alpha`、対象edgeは一様、noise map seedは全条件で20260914に固定する。

## 実行

```bash
git pull --ff-only
mkdir -p scripts/crowdhuman/logs
bash scripts/crowdhuman/run_h0_directional_noise.sh 0,1,2,3 |& tee scripts/crowdhuman/logs/h0_directional_noise.log
```

終了後、標準validationの`results.csv`からmAP50:95 / mAP50 / P / Rを取得する。さらに次を実行する。

```bash
bash scripts/crowdhuman/run_h0_directional_noise_eval.sh 0 |& tee scripts/crowdhuman/logs/h0_directional_noise_eval.log
```

`修論/H0/directional_noise_eval/*.csv`には、clean GTとIoU≥0.5でmatchしたpredictionについて次を保存する。

- AP50 / AP75（同一custom evaluator内で比較する補助指標）
- absolute edge error
- signed L/T/R/B edge error（pxとGT辺長正規化）
- prediction area / matched clean-GT areaのmean / median

符号は `prediction edge − clean GT edge` である。従ってL/Tの正はinward、R/Bの正はoutwardを表す。
