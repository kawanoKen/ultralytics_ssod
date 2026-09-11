# R2 final pre-check: object identity

## Question

R2領域 (`s>0`) かつGT-positiveである4,535 dense predictionsについて、Width-matched候補でpositiveに
変わった際のpseudo objectが、事前のclass-aware pseudo↔GT matching（IoU >= 0.5）で、そのpredictionを
GT-positiveにしたGT objectと同一かを調べた。

複数の摂動候補で異なるobjectへ割り当てられる場合があるため、prediction単位では以下の排他的3分類を用いた。

- same-object only: positive化した候補がすべて対応GTと同一object
- different-object only: positive化した候補がすべて別GTまたは未対応pseudo object
- mixed: 候補間でsame-objectとdifferent-objectの両方が存在

## Result

| Identity outcome | n | Rate among 4,535 |
|---|---:|---:|
| Same-object only | 2,554 | 56.32% |
| Mixed same/different | 74 | 1.63% |
| Different/unmatched object only | 1,907 | 42.05% |
| Any same-object candidate | **2,628** | **57.95%** |

候補遷移単位でも確認した。

| Positive candidate transition | n | Rate |
|---|---:|---:|
| All positive transitions | 7,612 | 100% |
| Same-object transitions | 4,681 | 61.50% |
| Different/unmatched-object transitions | 2,931 | 38.50% |

## Interpretation

過半数（57.95%）では、assignment instabilityが意図したGT objectに対応するpseudo boxのassignment
supportを検出している。したがって信号は単なる「近傍の何らかのobjectに反応した」ものだけではない。

一方、42.05%は対応GTと同一objectへのpositive化を一度も含まない。この部分では、R2-softがnegative
supervisionを弱める根拠は「正しいobjectのassignment回復」ではなく、別objectとのassignment競合である。
したがって、4,535件すべてをsame-object rescueとして数えることはできない。全R2 mask 16,887件に対する
明確なsame-object GT rescueは2,628件、すなわち15.56%である。

## Decision

R2-softの判断は **weak GO** とする。Hard ignoreを支持する結果ではない。`(1-s_j)`による連続的で比較的
弱い減衰を1 seedで探索する根拠は残るが、期待する機構は約58%のGT-positive maskにしか直接整合しない。
改善した場合も、「GT-positiveを与えた同一objectのassignment supportを常に回復した」とは主張せず、
same-object recoveryとcross-object competitionの混合効果として解釈する必要がある。

この診断ではthreshold tuningおよびtrainingを実施していない。

## Artifact

- Machine-readable result: `results/dfl_assignment/r2_final_precheck_width_matched_640/summary.json`
  (`r2_precheck.gt_positive_identity`)
- Analyzer: `scripts/dfl_assignment/analyze_gate_a.py`
