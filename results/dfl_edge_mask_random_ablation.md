# CrowdHuman 1%: count-matched random edge masking ablation

## 結論

既存の `loss_balance + DFL edge-confidence` と同じ「loss balance」を使い、DFLが決めたedge別の個数だけをランダム位置に適用する3-seed実験を完了した。

best mAP50:95 の平均は次の順だった。

| Method | seeds | best mAP50:95 | best mAP50 |
|---|---:|---:|---:|
| No mask / `loss_balance` | 1 existing | 0.37668 | 0.67590 |
| Random count-matched mask | 3 | 0.37810 ± 0.00589 | 0.67909 ± 0.00405 |
| DFL-selected mask | 3 existing | **0.38175 ± 0.00662** | **0.68590 ± 0.00426** |

この結果は平均値だけでは `DFL-selected > Random > No mask` で、DFL edge-confidenceが選択したedgeに追加の価値がある可能性を支持する。ただし、No maskはseed 0のみであり、DFL-selectedとRandomの差もseedごとには一貫していないため、強い結論ではなく「小さいが正のselection効果が示唆された」と解釈する。

## 1. 実装方法

Random variantは `loss_balance + use_edge_conf=True` のまま、以下だけを変更した。

1. reliable pseudo boxに対応するforeground anchor集合を取得する。
2. DFL-selected maskからedge別の採用数 `k_L, k_T, k_R, k_B` を数える。
3. 各edgeについて、confidenceの順位を使わず、eligible foreground anchorから `randperm` で `k_edge` 個を選ぶ。
4. classification loss、box loss、pseudo-label selection、confidence threshold、reliable box集合は変更しない。

したがって、RandomでDFL confidenceが使われるのは「各edgeの個数を決める」箇所だけで、どのanchor/edge位置をmaskするかには使われない。

共通条件は、CrowdHuman 1%、100 epoch、4 GPU、batch/batch_ssod=128、imgsz=640、`edge_conf_threshold=0.6`、`ssod_weight=0.5`、`loss_balancing_mode=ema_scale`、beta=0.9、`skip_zero_pseudo_cls_loss=True` である。Randomはseed 0/1/2を実行した。

実装:

- [loss_ssod.py](/work/kawano/LA/ultralytics_ssod/ultralytics/utils/loss_ssod.py)
- [train_ssod.py](/work/kawano/LA/ultralytics_ssod/scripts/voc_ssod/train_ssod.py)
- [random ablation launcher](/work/kawano/LA/ultralytics_ssod/scripts/crowdhuman/run_loss_balance_random_edge_mask_1p.sh)

## 2. mask量の一致

Random runでは、各batchについてDFL-selected相当の `target` countと、randomに実際に適用した `applied` countを記録した。全seedで11,600 batch rowsを確認し、edge別・totalとも不一致は0件だった。

ここで「kept」はDFL lossに残したedge数、「masked」はeligible edge数からkeptを引いた除外数である。

| seed | eligible edges | reliable pseudo boxes | masked L | masked T | masked R | masked B | masked total | mask rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 231,159,040 | 6,410,487 | 49,350 | 124,906 | 46,110 | 660,793 | 881,159 | 0.3812% |
| 1 | 246,038,676 | 6,711,463 | 59,283 | 383,837 | 53,365 | 832,611 | 1,329,096 | 0.5402% |
| 2 | 239,471,112 | 6,652,209 | 28,656 | 248,975 | 24,887 | 654,467 | 956,985 | 0.3996% |

この一致は、Random run内での同一batchにおけるcounterfactual DFL-selected countとrandom applied countの一致である。既存DFL-selected runには今回のmask診断列が保存されていなかったため、異なるcheckpoint trajectory間の集計mask総数を直接同一と主張してはいない。実装上は同じthreshold判定からcountを作り、そのcountをそのままrandomに適用している。

診断ログ:

- [seed0 assignment dynamics](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed0/logs/assignment_dynamics.csv)
- [seed1 assignment dynamics](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed1/logs/assignment_dynamics.csv)
- [seed2 assignment dynamics](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed2/logs/assignment_dynamics.csv)

## 3. mAP結果

### Best / final

| Method | seed | best mAP50 | best mAP50:95 | final mAP50 | final mAP50:95 |
|---|---:|---:|---:|---:|---:|
| No mask | existing 0 | 0.67590 | 0.37668 | 0.67365 | 0.37483 |
| DFL-selected | 0 | 0.69053 | 0.38867 | 0.68438 | 0.38502 |
| DFL-selected | 1 | 0.68502 | 0.38110 | 0.67696 | 0.37569 |
| DFL-selected | 2 | 0.68214 | 0.37548 | 0.67130 | 0.36888 |
| Random | 0 | 0.68119 | 0.37827 | 0.67558 | 0.37420 |
| Random | 1 | 0.68166 | 0.38391 | 0.66979 | 0.37627 |
| Random | 2 | 0.67443 | 0.37213 | 0.66746 | 0.36629 |

集計:

| Method | best mAP50 mean ± SD | best mAP50:95 mean ± SD | final mAP50 mean ± SD | final mAP50:95 mean ± SD |
|---|---:|---:|---:|---:|
| No mask | 0.67590 | 0.37668 | 0.67365 | 0.37483 |
| Random | 0.67909 ± 0.00405 | 0.37810 ± 0.00589 | 0.67094 ± 0.00418 | 0.37225 ± 0.00527 |
| DFL-selected | **0.68590 ± 0.00426** | **0.38175 ± 0.00662** | **0.67755 ± 0.00656** | **0.37653 ± 0.00810** |

DFL-selected minus Randomのbest mAP50:95差は、seed 0で `+0.01040`、seed 1で `-0.00281`、seed 2で `+0.00335`、平均で `+0.00365` だった。

RandomはNo maskのsingle existing seedよりbest mAP50:95で `+0.00142`、DFL-selectedは `+0.00507` 高い。ただしNo maskのseed数が1なので、Random > No maskの差はseed variationと区別できない。

各runのraw結果:

- [No mask](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance/results.csv)
- [DFL-selected seed0](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_edge_conf/results.csv)
- [DFL-selected seed1](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_edge_conf_seed1_valid/results.csv)
- [DFL-selected seed2](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_edge_conf_seed2_valid/results.csv)
- [Random seed0](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed0/results.csv)
- [Random seed1](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed1/results.csv)
- [Random seed2](../runs/crowdhuman_ssod_1p_zero_pseudo_ablation/yolov8n_1p_loss_balance_random_edge_mask_seed2/results.csv)

## 4. validation pseudo edgeのGT error

追加学習なしで、既存のNo mask teacher checkpointのvalidation predictionを同一NMS (`conf=0.01`, `IoU=0.65`) で評価した。classification confidenceが0.5以上で、full-body GT IoUが0.5以上のmatched pseudo boxについて、edge confidenceが0.6未満をDFL-maskedとした。CrowdHumanのfull-body / visible GT boxは、predictionと同じ有効画像座標系で比較できるよう、IoU・edge error・occlusion算出の前に画像境界へclipした。

全edgeをまとめたweighted meanでは次のようになった。medianはedge side間の単純な集約ができないため、下表のside別medianを参照する。

| GT visibility | DFL mask status | n | mean pseudo-to-GT edge error (px) | mean DFL confidence | mean occlusion |
|---|---|---:|---:|---:|---:|
| visible | DFL-masked | 131 | 77.07 | 0.515 | 0.009 |
| visible | non-masked | 160,271 | 9.36 | 0.988 | 0.006 |
| occluded | DFL-masked | 295 | 52.64 | 0.507 | 0.426 |
| occluded | non-masked | 44,111 | 19.57 | 0.968 | 0.220 |

side別の詳細:

| visibility | status | side | n | mean error px | median error px | mean DFL conf | mean occlusion |
|---|---|---|---:|---:|---:|---:|---:|
| visible | masked | L | 23 | 110.65 | 110.92 | 0.522 | 0.009 |
| visible | masked | T | 12 | 58.36 | 56.99 | 0.525 | 0.009 |
| visible | masked | R | 30 | 100.82 | 75.75 | 0.524 | 0.012 |
| visible | masked | B | 66 | 57.98 | 38.10 | 0.506 | 0.008 |
| visible | non-masked | L | 35,333 | 11.51 | 4.76 | 0.990 | 0.007 |
| visible | non-masked | T | 50,514 | 7.47 | 3.34 | 0.988 | 0.005 |
| visible | non-masked | R | 37,254 | 10.83 | 4.35 | 0.990 | 0.006 |
| visible | non-masked | B | 37,170 | 8.41 | 3.00 | 0.982 | 0.003 |
| occluded | masked | L | 18 | 101.80 | 90.49 | 0.507 | 0.186 |
| occluded | masked | T | 3 | 65.90 | 69.11 | 0.543 | 0.105 |
| occluded | masked | R | 16 | 51.35 | 40.86 | 0.541 | 0.201 |
| occluded | masked | B | 258 | 49.14 | 34.30 | 0.504 | 0.460 |
| occluded | non-masked | L | 15,828 | 12.13 | 5.60 | 0.986 | 0.145 |
| occluded | non-masked | T | 673 | 16.53 | 7.06 | 0.971 | 0.103 |
| occluded | non-masked | R | 13,902 | 14.28 | 6.61 | 0.986 | 0.152 |
| occluded | non-masked | B | 13,708 | 33.69 | 16.64 | 0.928 | 0.382 |

DFL-masked edgeはvisible/occludedの両方でnon-masked edgeより低confidenceかつ平均GT errorが大きい。特にoccluded bottomではmasked 49.14 px、non-masked 33.69 pxである。したがって「低DFL-confidence edgeは難しいpseudo edgeである」という前提は概ね支持される。旧集計にあったbottomでの逆転は、画像外へ伸びるGT boxを未clipで比較したことによるアーティファクトだった。

詳細CSV:

- [pseudo masked/non-masked edge error, GT clipped](/work/kawano/LA/ultralytics_ssod/results/dfl_edge_mask_random_ablation/pseudo_masked_edge_error_1p_clipped.csv)

## 5. 保存したloss / pseudo-label情報

Randomのfinal epochにおける loss は以下の通り。`train/*`はlabeled loss、`ssod/*`はunlabeled lossである。

| seed | train box | train cls | train DFL | SSOD box | SSOD cls | SSOD DFL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.43856 | 0.28639 | 0.83749 | 0.60786 | 0.41548 | 0.92826 |
| 1 | 0.38554 | 0.25958 | 0.82049 | 0.59498 | 0.41290 | 0.91690 |
| 2 | 0.42435 | 0.27921 | 0.83260 | 0.61759 | 0.42372 | 0.92491 |

各runの `results.csv`、`logs/training_dynamics.csv`、`logs/assignment_dynamics.csv` に、mAP、loss、pseudo-label count、reliable pseudo box count、edge別mask countを保存している。

## 6. 解釈

今回の平均best mAP50:95は、指定された分類では **Case Aに近い**。

```text
DFL-selected > Random count-matched > No mask
0.38175      > 0.37810                 > 0.37668
```

したがって、DFL supervisionを減らすregularization効果は一部あり得るが、DFL confidenceでedgeを選ぶことにも平均で `+0.00365` の追加差が残った。ただし、RandomとDFL-selectedのseed別結果は混在し、No maskは1 seedのみなので、現時点では「DFL固有のselection効果を支持するが、効果量は小さくseed variationも大きい」という結論が妥当である。

今回のvalidation pseudo解析は、DFL-masked edgeが実際に難しいedgeを多く含むことを確認した。DFL confidenceを遮蔽edgeの完全なoracleとはみなせないが、GT clip後にもbottomを含む全sideで「masked edgeほどpseudo-to-GT errorが大きい」という方向は維持された。
