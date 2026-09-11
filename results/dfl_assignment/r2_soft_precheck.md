# R2-soft pre-check

## Setup

- YOLOv8n, CrowdHuman 5%, 640 px
- R2 pre-checkと同じ、validation list上で決定論的に等間隔抽出した500画像
- 同一supervised checkpoint、pseudo-label threshold 0.6
- 同一`TaskAlignedAssigner(topk=13, alpha=0.5, beta=6.0)`
- 解析と学習で共通化したWidth-matched perturbation
- 全4,200,000 dense predictionsのうち、baseline-negative 4,130,731件を評価
- `L_neg = BCEWithLogits(logit, 0) = softplus(logit)`、gradient proxyは`p = sigmoid(logit)`

R2-soft trainingおよびthreshold tuningは実施していない。

## 1. Loss enrichment

R2対象領域は、指定どおり`s_j > 0`、すなわちbaseline-negativeだが9候補中少なくとも1候補でpositiveに
なるpredictionとした。

| 集合 | n | Negative loss合計 | 平均negative loss |
|---|---:|---:|---:|
| 全baseline-negative: GT-positive | 46,986 | 10,200.662 | 0.217100 |
| 全baseline-negative: GT-negative | 4,083,745 | 9,495.734 | 0.002325 |
| R2領域 (`s>0`): GT-positive | 4,535 | **945.817** | **0.208559** |
| R2領域 (`s>0`): GT-negative | 12,352 | **740.122** | **0.059919** |

比較結果：

- R2領域の平均negative lossは0.099837で、全baseline-negativeの0.004768に対して**20.94倍**。
- R2領域のGT-positive率26.85%は、全baseline-negativeの1.1375%に対して**23.61倍**。
- R2領域内ではGT-positiveは件数の26.85%だが、negative loss総量の**56.10%**を占める。
- R2領域内のGT-positive平均lossはGT-negativeの**3.48倍**。
- GT-positiveだけを見ると、R2領域の平均lossは全GT-positive baseline-negativeの0.961倍であり、
  `s>0`はGT-positive内で特に高lossな例を選ぶわけではない。一方、GT-negative側は全体平均の25.77倍で、
  assignment境界上のhard negativeも強く拾う。

したがって、`s>0`は誤ったnegative supervisionを強く濃縮するが、同時に高lossな正しいnegativeも含む。
これはhard ignoreよりsoft weightingを検討する理由になる。

## 2. Gradient enrichment

Negative BCEのlogit gradient magnitude `p_j` を集計した。

| 集合 | `sum p_j` | Mean `p_j` |
|---|---:|---:|
| 全baseline-negative: GT-positive | 8,302.866 | 0.176709 |
| 全baseline-negative: GT-negative | 8,701.031 | 0.002131 |
| R2領域 (`s>0`): GT-positive | **756.456** | **0.166804** |
| R2領域 (`s>0`): GT-negative | **659.616** | **0.053402** |

- R2領域のmean `p`は0.083856で、全baseline-negativeの0.004116に対して**20.37倍**。
- R2領域内でGT-positiveはgradient proxy総量の**53.42%**を占める。
- R2領域内のGT-positive mean `p`はGT-negativeの**3.12倍**。
- GT-positive側のmean `p`は全GT-positive baseline-negativeの0.944倍、GT-negative側は全GT-negativeの
  25.06倍。lossと同様に、maskは誤supervisionと正しいhard negativeの両方を濃縮している。

## 3. Instability別結果

Baseline-negativeについて、`s_j = positiveになった候補数 / 9`とした。最終binは1.0を含む。

| Instability | n | GT-positive率 | Mean neg loss | Mean `p` |
|---|---:|---:|---:|---:|
| 0.0-0.2 | 4,125,849 | 1.09% | 0.004604 | 0.003980 |
| 0.2-0.4 | 4,484 | 39.61% | 0.139424 | 0.115737 |
| 0.4-0.6 | 379 | 56.20% | 0.183902 | 0.154043 |
| 0.6-0.8 | 18 | 72.22% | 0.319120 | 0.252637 |
| 0.8-1.0 | 1 | 100.00% | 0.461500 | 0.369663 |

Instabilityの上昇に伴い、GT-positive率、mean negative loss、mean `p`がすべて単調増加した。ただし
`s=1/9`は0.0–0.2 binに含まれるため、このbinは安定negativeと弱いambiguous negativeを混在させている。
また`s>=0.4`は398件しかなく、高instability側の推定値はサンプル数が少ない。

## 4. R2-softを1 seed試す価値があるか

**R2-softは1 seedの事前登録済み確認実験を行う価値がある（GO）**と判断する。

根拠は、R2領域が全baseline-negativeよりGT-positive率を23.61倍濃縮し、誤ったnegative supervisionが
領域内negative lossの56.10%、gradient proxyの53.42%を占めること、さらにinstabilityとGT-positive率・
loss・gradientが一貫して単調関係を示すことである。これはassignment instabilityが、誤ったnegative
classification gradientの位置を一定程度捉えていることを支持する。

ただしR2領域の73.15%はGT-negativeであるため、hard ignoreは支持されない。試す場合もtargetは0のまま、
事前指定された

`L_neg,j^R2 = (1 - s_j) * BCE(p_j, 0)`

のみとし、今回のGT bin結果からthresholdを選ばない。これは探索的な1 seedを支持する判定であり、性能改善を
保証するものではない。今回、trainingは開始していない。

## Artifacts

- Machine-readable result: `results/dfl_assignment/r2_soft_precheck_width_matched_640/summary.json`
- Detailed prediction/object output: `results/dfl_assignment/r2_soft_precheck_width_matched_640/`
- Shared analyzer: `scripts/dfl_assignment/analyze_gate_a.py`
