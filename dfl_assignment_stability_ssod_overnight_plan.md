# DFL Shape × Assignment Stability for SSOD
## Overnight Implementation / Analysis / Training Plan

### 0. 目的

既存の DFL-based pseudo-label selection をさらに発展させ、**DFL が曖昧な pseudo box を除外するのではなく、その box によって生じる assignment の不安定性を検出し、localization loss の responsibility に反映する**。

今回の中心仮説は以下。

\[
	ext{DFL ambiguity}
ightarrow
	ext{boundary ambiguity}
ightarrow
	ext{assignment instability}
\]

ただし、

\[
	ext{assignment instability}

eq
	ext{object existence uncertainty}
\]

と考える。

したがって、classification target を soft にするのではなく、**localization をどの prediction が担当するかの安定性だけを loss weight / mask に利用する**。

---

# 1. 既に完了しているもの

以下は **再実装・再学習しない**。既存コード、checkpoint、log、result を再利用する。

- YOLOv8 の supervised detection
- YOLOv11 の supervised detection
- YOLOv8 / YOLOv11 の SSOD
- confidence threshold のみを用いる baseline SSOD
- DFL confidence を導入した既存 SSOD
- 論文で用いた DFL localization-confidence algorithm
- VOC
  - labeled ratio: 1%, 5%, 10%
  - supervised 学習済み
  - baseline SSOD 学習済み
  - DFL-based SSOD 学習済み
- CrowdHuman
  - labeled ratio: 1%, 5%, 10%
  - supervised 学習済み
  - baseline SSOD 学習済み
  - DFL-based SSOD 学習済み

既存結果を baseline として固定する。

今回、新規に行うのは

1. 現行実装の確認
2. DFL distribution と assigner の接続確認
3. deterministic DFL perturbation による assignment sensitivity 解析
4. GT を用いた assignment error 解析
5. assignment-stability-aware SSOD の実装
6. smoke test
7. 必要最小限の full training
8. 層別解析

である。

---

# 2. Design freeze

既存 baseline との比較可能性を維持するため、以下は変更しない。

- train / val split
- labeled / unlabeled split
- augmentation
- optimizer
- learning-rate schedule
- EMA teacher update
- batch size
- image size
- pseudo-label confidence threshold
- burn-in
- training epochs
- NMS
- model size
- seed の定義
- evaluation code

新しい自由度は原則として、

- DFL-supported deterministic perturbation
- assignment stability score
- regression loss weighting
- ambiguous negative masking

だけに限定する。

**結果を見てから threshold を調整しない。**

---

# Step 0 — 現行実装の完全確認

## 0.1 最初にコードを読む

まだ新しい学習は実行しない。

YOLOv8 / YOLOv11 について、以下の実装経路を実際のコードから特定する。

```text
unlabeled image
    ↓
teacher forward
    ↓
raw dense predictions
    ├─ classification logits
    └─ DFL regression logits
    ↓
DFL softmax
    ↓
expectation decode
    ↓
decoded boxes
    ↓
NMS
    ↓
pseudo labels
    ↓
student augmentation / coordinate transform
    ↓
assigner
    ↓
foreground mask / target assignment
    ↓
classification loss
box / IoU loss
DFL loss
```

## 0.2 必ず確認するもの

### Teacher output

- raw DFL logits の shape
- `reg_max`
- 4 辺の並び
- stride
- anchor point / grid point
- expectation decode の実装
- decoded box の座標系
- NMS 前後の index 対応

### DFL confidence implementation

既存の論文実装について、

- edge-level DFL confidence の実装場所
- max-adjacent probability sum の計算
- box-level `min(edge confidence)` の計算
- pseudo-label selection に適用する場所
- NMS 後 box と元 DFL logits の紐付け方法

を特定する。

**すでに DFL-based SSOD が動いているため、新しく似た処理を作らず、既存の対応関係を最大限再利用する。**

### Assigner

YOLOv8 / YOLOv11 それぞれについて、

- 実際に使用されている assigner の class / function
- assigner input
- assigner output
- positive / negative の決定方法
- 1 prediction が複数 object の候補になった場合の解決
- top-k / IoU / classification score 等の利用
- foreground mask
- target score
- target box
- target GT / pseudo-object index

を確認する。

**TaskAlignedAssigner 等を前提に決め打ちしない。実コードを確認して記録する。**

### Loss

以下を正確に追跡する。

- positive classification loss
- negative classification loss
- box / IoU loss
- DFL loss
- normalization
- foreground 数による normalization の有無
- `target_scores_sum` 等による normalization の有無

新しい weight を入れた際に、単純な loss scale の低下が起きないようにするため、ここは特に重要。

---

## 0.3 Step 0 の成果物

`results/dfl_assignment/step0_implementation_audit.md`

最低限、以下を書く。

```text
1. YOLOv8 teacher → pseudo-label → assigner → loss の経路
2. YOLOv11 teacher → pseudo-label → assigner → loss の経路
3. raw DFL logits を NMS 後 pseudo box に対応付ける方法
4. existing DFL confidence implementation
5. assigner の正確な実装
6. cls / box / DFL loss の normalization
7. 新規処理を挿入すべき最小箇所
8. YOLOv8 と YOLOv11 で共通化できる箇所
9. 実装上の懸念点
```

コードの **file path / class / function 名 / line 周辺** を必ず残す。

---

# Step 1 — Deterministic DFL-supported box perturbation

## 1.1 Sampling は行わない

通常の prediction と同様、baseline pseudo box は DFL expectation から作る。

各 edge \(e\) について、

\[
p_e(k)=\mathrm{softmax}(z_e)_k
\]

\[
\mu_e=\sum_k k p_e(k)
\]

が通常の DFL expectation。

今回、DFL を真の確率分布と仮定して Monte Carlo sampling はしない。

代わりに、**DFL shape が支持する deterministic な境界変動を作る。**

---

## 1.2 Primary perturbation definition

各 edge の CDF から固定 quantile を求める。

Primary:

\[
d_e^- = Q_{0.10}(p_e)
\]

\[
d_e^0 = \mu_e
\]

\[
d_e^+ = Q_{0.90}(p_e)
\]

ここで \(Q\) は DFL bin 上の deterministic quantile。

これは「真の 80% confidence interval」とは呼ばない。

あくまで、

> DFL distribution shape に基づく deterministic perturbation range

として扱う。

必要なら bin 内 interpolation を実装するが、まずは現行 DFL decode と矛盾しない最も単純な方法を使う。

---

## 1.3 Candidate boxes

combinatorial な全組合せは作らない。

baseline expectation box:

\[
B^0 =
(d_L^0,d_T^0,d_R^0,d_B^0)
\]

に加えて、1 edge だけを perturb した 8 個を作る。

\[
B_L^-, B_L^+,
B_T^-, B_T^+,
B_R^-, B_R^+,
B_B^-, B_B^+
\]

合計：

\[
M=9
\]

candidate sets。

```text
B0        : E[L], E[T], E[R], E[B]

B_L_low   : Q10[L], E[T], E[R], E[B]
B_L_high  : Q90[L], E[T], E[R], E[B]

B_T_low   : E[L], Q10[T], E[R], E[B]
B_T_high  : E[L], Q90[T], E[R], E[B]

B_R_low   : E[L], E[T], Q10[R], E[B]
B_R_high  : E[L], E[T], Q90[R], E[B]

B_B_low   : E[L], E[T], E[R], Q10[B]
B_B_high  : E[L], E[T], E[R], Q90[B]
```

DFL が distance representation の場合は、**必ず現行 decode と同じ anchor point / stride / coordinate conversion を使用する。**

---

## 1.4 Sanity check

sharp distribution では candidate boxes が expectation box の近傍に集まること。

flat / multimodal distribution では perturbation 幅が広くなること。

以下を可視化。

- image
- expected pseudo box
- 8 perturbed boxes
- 4-edge DFL distribution
- existing edge confidence
- perturbation width

出力：

`results/dfl_assignment/step1_sanity/`

最低 30 instance を保存。

- high DFL confidence
- low DFL confidence
- visible
- occluded
- crowded overlap

を含める。

---

# Step 2 — Assignment sensitivity analysis

ここが最重要。

新しい学習を行う前に、

\[
	ext{DFL ambiguity}
ightarrow
	ext{assignment instability}
\]

が本当に存在するかを確認する。

---

## 2.1 Assigner を M=9 回適用

同じ teacher prediction に対して、

- expected pseudo boxes
- 8 deterministic perturbation sets

を使って assigner を再実行する。

各 candidate set について、

prediction \(j\) が pseudo object \(i\) に assign されたかを記録。

---

## 2.2 Assignment stability score

sampling probability とは呼ばない。

各 prediction \(j\)、pseudo object \(i\) について、

\[
s_{ij}
=
rac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}
[
j 	ext{ is assigned to object } i
	ext{ under perturbation } m
]
\]

と定義する。

これは、

> **DFL-supported deterministic perturbations に対する same-object assignment stability**

である。

確率とは解釈しない。

### 重要

単なる foreground / background だけでなく、

- object A → object A
- object A → object B
- foreground → background
- background → foreground

を区別する。

CrowdHuman では特に **object identity の入れ替わり** を記録する。

---

## 2.3 Object-level metrics

各 pseudo object について最低限、

### A. Assignment Jaccard stability

baseline positive set \(P_i^0\) と perturbation positive set \(P_i^m\) について、

\[
J_i
=
rac{1}{M-1}
\sum_{m=1}^{M-1}
rac{|P_i^0\cap P_i^m|}
{|P_i^0\cup P_i^m|}
\]

### B. Instability

\[
U_i=1-J_i
\]

### C. Ambiguous prediction ratio

\[
A_i
=
rac{
|\{j:0<s_{ij}<1\}|
}{
|\{j:s_{ij}>0\}|
}
\]

### D. Identity-switch count

prediction が perturbation により別 pseudo object に assign される回数。

---

# Step 3 — DFL shape と assignment instability の関係

## 3.1 既存 DFL 指標

既存実装をそのまま利用する。

- edge confidence
- box confidence = minimum over 4 edges

さらに解析用として、

- edge entropy
- edge variance
- Q90-Q10 width
- bimodality / secondary peak indicator

を保存してよい。

ただし、**新しい学習ルールにはまだ使わない。**

---

## 3.2 Dataset / ratio

解析はまず既存 checkpoint を利用する。

### Priority 1

CrowdHuman

- 1%
- 5%
- 10%

### Priority 2

VOC

- 1%
- 5%
- 10%

VOC は CrowdHuman より crowd / occlusion が弱い control として扱う。

---

## 3.3 見る関係

最低限：

\[
c^{DFL}_{box}
\leftrightarrow
U_i
\]

\[
	ext{Q90-Q10 width}
\leftrightarrow
U_i
\]

\[
	ext{occlusion}
\leftrightarrow
U_i
\]

CrowdHuman では可能なら、

- visible / occluded
- overlap IoU
- crowd density
- object size

で stratify する。

---

## 3.4 必須 plot

1. DFL confidence vs assignment instability
2. DFL width vs assignment instability
3. occlusion bin vs assignment instability
4. object size vs assignment instability
5. CrowdHuman vs VOC
6. 1% / 5% / 10%
7. identity-switch rate vs DFL confidence

出力：

`results/dfl_assignment/step3_analysis/`

---

# Step 4 — GT-based assignment error analysis

単に assignment が揺れるだけでは不十分。

> その instability が hard pseudo assignment の誤りと関係しているか

を確認する。

labeled / validation data についてのみ GT を解析に使う。

---

## 4.1 GT assignment reference

同じ assigner に GT box を入力して、

\[
A^{GT}
\]

を得る。

teacher expected pseudo box による assignment:

\[
A^{pseudo}
\]

DFL perturbation による stability:

\[
s_{ij}
\]

を比較する。

---

## 4.2 見るもの

### Hard assignment error

baseline expected pseudo box により、

- GT では positive なのに pseudo では negative
- GT では negative なのに pseudo では positive
- GT object A なのに pseudo object B に assign

されたものを記録。

### 主要検証

assignment stability が低い prediction ほど、

\[
A^{pseudo}
eq A^{GT}
\]

になりやすいか。

---

## 4.3 評価

- error rate by stability bin
- precision / recall of instability for detecting assignment error
- AUROC
- AUPRC
- object identity mismatch
- occlusion stratification

---

# Gate A — 学習へ進む条件

以下の少なくともどちらかが明確に成立した場合、Step 5 へ進む。

### Condition A

DFL ambiguity が assignment instability と明確に関連する。

### Condition B

assignment instability が GT assignment error を予測する。

特に CrowdHuman で強く、VOC で弱いなら非常に興味深い。

## No-Go

以下なら full training は行わない。

- DFL shape と assignment instability がほぼ無関係
- instability が GT assignment error と無関係
- perturbation が decode artifact しか反映していない
- candidate range が極端で現実的でない

---

# Step 5 — Assignment-stability-aware SSOD

学習へ進む場合のみ実装。

## 5.1 基本方針

**classification target を stability score にしない。**

つまり、

\[
\mathrm{BCE}(p,s)
\]

のような soft classification target は使わない。

object existence uncertainty と assignment uncertainty を分離する。

---

# Variant R1 — Regression Responsibility

baseline expectation box による通常 assignment を \(A^0\) とする。

baseline で object \(i\) の positive に assign された prediction \(j\) に対して、

\[
s_{ij}
\]

を localization responsibility として利用。

classification は既存のまま。

\[
L_{cls}^{new}=L_{cls}^{base}
\]

box / DFL は stability weight を掛ける。

\[
L_{box}^{new}
=
rac{
\sum_{(i,j)\in A^0}
s_{ij}L_{box}^{ij}
}{
\sum_{(i,j)\in A^0}s_{ij}+\epsilon
}
\]

\[
L_{DFL}^{new}
=
rac{
\sum_{(i,j)\in A^0}
s_{ij}L_{DFL}^{ij}
}{
\sum_{(i,j)\in A^0}s_{ij}+\epsilon
}
\]

### 重要

単純に \(\sum sL\) として loss magnitude を落とさない。

**sum of weights で normalize** して、

> gradient scale が小さくなっただけ

という confound を避ける。

---

# Variant R2 — Regression Responsibility + Ambiguous Negative Ignore

R1 に加えて、

baseline expectation box では negative だが、

\[
s_{ij}>0
\]

となる prediction は hard negative として扱わない。

### classification

- baseline positive: existing positive classification loss
- stable negative: existing negative loss
- ambiguous negative: negative classification loss を mask

ただし、

\[
s_{ij}
\]

を classification target にはしない。

---

## 5.2 まず threshold-free で実装

Primary experiment は、

\[
s_{ij}>0
\]

を ambiguous とする。

0.1, 0.2, 0.3 などを結果を見ながら探索しない。

threshold tuning は後続実験に分離する。

---

# Step 6 — Computational overhead measurement

新方式では assigner を複数回実行する。

学習前に、

- baseline iteration time
- R1 iteration time
- R2 iteration time
- GPU memory
- CPU time
- assigner time

を計測。

forward / backward ではなく assigner overhead が主なら、その旨を記録。

もし M=9 が過度に重い場合、**勝手に候補数を変更せず、まず profiling を報告する。**

---

# Step 7 — Smoke test

いきなり full training しない。

## Primary smoke

YOLOv8 + CrowdHuman 5%

理由：

- CrowdHuman は occlusion / overlap 仮説に最も直接的
- 1% より teacher が安定しており、新手法そのものの挙動を確認しやすい

短時間学習で、

- R1
- R2

を確認。

## 必須 logging

- total loss
- cls loss
- box loss
- DFL loss
- pseudo-label count
- positive count
- negative count
- ambiguous-negative count
- mean stability
- median stability
- stability histogram
- effective regression weight sum
- assignment identity switches
- iteration time
- VRAM
- NaN / inf

を保存。

---

# Step 8 — Full training priority

既存 baseline は再実行しない。

新規 method のみ。

## Priority 1

YOLOv8 + CrowdHuman 5%

- R1
- R2

## Priority 2

YOLOv8 + CrowdHuman 1%

- Priority 1 で良かった method
- 1% は stress test

## Priority 3

YOLOv11 + CrowdHuman 5%

- best method の portability check

## Priority 4

YOLOv8 + VOC 5%

- less-occluded control

時間が残れば 10% および追加 architecture / dataset へ進む。

---

# Step 9 — 学習結果の解析

mAP だけで結論を出さない。

## 9.1 Main metrics

- mAP50
- mAP50:95
- pseudo-label precision
- pseudo-label recall
- final epoch
- best epoch
- training stability

既存の

- confidence-only SSOD
- existing DFL-selection SSOD

と比較する。

## 9.2 Stratified analysis

### DFL confidence

- high
- middle
- low

### Assignment stability

- stable
- moderately unstable
- highly unstable

### Occlusion

CrowdHuman で利用可能な visibility / occlusion 情報に基づく。

### Object size

- small
- medium
- large

### Overlap

- isolated
- overlapping
- strongly overlapping

---

# Step 10 — Ablation は最小限

初日に大量の ablation は行わない。

まず：

| Method | Description |
|---|---|
| Existing baseline | confidence-only SSOD |
| Existing DFL selection | current paper method |
| R1 | regression responsibility |
| R2 | R1 + ambiguous-negative ignore |

だけ。

---

# Step 11 — 翌朝の成果物

最終的に、

`results/dfl_assignment/overnight_summary.md`

を作成。

構成：

```text
# 1. Executive summary
# 2. Existing implementation audit
# 3. Deterministic perturbation sanity check
# 4. Does DFL ambiguity produce assignment instability?
# 5. Does assignment instability predict GT assignment error?
# 6. CrowdHuman vs VOC
# 7. 1% / 5% / 10% comparison
# 8. New method implementation
# 9. Smoke test
# 10. Full training results
# 11. Stratified results
# 12. Runtime overhead
# 13. Go / No-Go conclusion
# 14. Recommended next experiment
```

---

# 実装時の重要な禁止事項

1. DFL distribution から random sampling しない。
2. DFL bin probability を真の GT boundary probability と断定しない。
3. assignment stability score を object confidence と解釈しない。
4. `BCE(p, s)` の soft classification target にしない。
5. ambiguous assignment を単純に negative にしない。
6. loss weight を入れたことで総 gradient scale が落ちる confound を放置しない。
7. 新しい threshold を validation result を見ながら調整しない。
8. 既存 baseline を無駄に再学習しない。
9. YOLOv8 と YOLOv11 の実装が同じだと仮定しない。
10. CrowdHuman の object identity switch を foreground/background flip と一緒に集約しない。

---

# 最終的に検証したい研究仮説

## H1

DFL が曖昧な pseudo box ほど、境界を DFL-supported range 内で変化させた際に assignment が不安定になる。

## H2

その assignment instability は、GT assignment との mismatch を予測する。

## H3

この instability は CrowdHuman の occluded / overlapping objects で強い。

## H4

不確かな pseudo object を除外するより、

> object は利用し続け、localization responsibility のみ assignment stability に応じて制御する

方が SSOD に有効である。

## H5

assignment uncertainty を classification uncertainty として扱う必要はなく、

\[
L_{cls}
\]

を維持し、

\[
L_{box},L_{DFL}
\]

だけを制御する方が自然である。

---

# 一夜の最優先順位

```text
Step 0  implementation audit
        ↓
Step 1  deterministic DFL perturbation
        ↓
Step 2  assignment sensitivity
        ↓
Step 3  DFL / occlusion relation
        ↓
Step 4  GT assignment error relation
        ↓
       Gate A
        ↓
Step 5  R1 / R2 implementation
        ↓
Step 7  smoke
        ↓
Step 8  CrowdHuman full training
        ↓
Step 9  analysis
        ↓
overnight_summary.md
```

**解析で仮説が成立しない場合は、full training を回すこと自体を成果とみなさず、Gate A で止める。**
