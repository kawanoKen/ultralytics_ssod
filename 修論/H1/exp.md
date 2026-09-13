# Implementation task: edge-wise DFL supervision dose-response
## Oracle selector + DFL-confidence selector

## 目的

今回は **実装のみ** 行う。

Full trainingはこのマシンでは実行しない。
実装・unit test・短いsmoke test・launcher/config作成までを行い、
別マシンへコードを移して本実験を実行できる状態にする。

実験では、特定edgeに対するDFL supervisionの相対weightを

\[
w \in \{0,1,5,20\}
\]

に変更する。

対象edgeの選び方を2種類用意する。

1. **Oracle selector**
   - GTから実際に大きくずれているpseudo-edgeを選ぶ
2. **DFL selector**
   - 既存のDFL edge-confidenceが低いpseudo-edgeを選ぶ

重要：

> selectorだけを交換し、DFL weighting処理・normalization・loss経路は完全に共通化すること。

---

# 1. 実験モード

新しい設定として、例えば以下を追加する。

```yaml
edge_dfl_reweight: true

edge_dfl_selector: oracle   # oracle | dfl

edge_dfl_weight: 5.0

edge_dfl_normalize: true

oracle_edge_error_threshold: 0.10

edge_conf_threshold: 0.60

名前は既存コード設計に合わせて変更してよい。

ただし、

Oracle用weightingコード
DFL用weightingコード

を別々に作らないこと。

共通処理：

selector
   ↓
selected_edge_mask [positive predictions, 4]
   ↓
raw edge weights
   ↓
normalization
   ↓
edge-wise DFL loss

とする。

2. 共通weighting

4辺reduce前のDFL loss

$$ \ell_{j,e} $$

に対し、

$$ a_{j,e}= \begin{cases} w & \text{selected edge}\\ 1 & \text{otherwise} \end{cases} $$

とする。

使用するweight：

$$ w\in\{0,1,5,20\} $$

意味：

w=0: selected edgeのDFL supervisionを抑制
w=1: baseline
w=5: selected edgeを中程度に増幅
w=20: selected edgeを強く増幅
3. 総DFL量のnormalization

Primary dose-responseでは、単純なloss-scale増加を避けるため、
eligible edge全体で平均weightが1になるようにする。

$$ \tilde a_{j,e} = a_{j,e} \frac{N_{\rm edge}} {\sum a_{j,e}} $$

DFL denominator自体は既存baselineから変更しない。

したがって測りたいのは、

DFL supervision総量ではなく、
selected edgeへsupervisionをどれだけ相対配分するか

である。

Edge case
selected edgeが0件 → 全weight=1
w=0かつeligible edge全てがselectedになり sum(a)=0 になる場合
division by zeroを起こさない
そのbatchのDFL contributionを0として扱い、専用counterをlog
勝手に別weightへ置換しない

このケースが何回発生したか必ず保存する。

4. Oracle selector
4.1 GT clipping

CrowdHumanのfbox / vboxは画像外へ伸びるため、
GTは必ず画像範囲へclipする。

$$ x\in[0,W], \qquad y\in[0,H] $$

matching、edge error、wrong-edge判定のすべてで
clip済みGTを使用する。

既に修正済みのGT clipping処理がある場合は、
新しい別実装を作らず共通関数を再利用する。

4.2 Student coordinate system

Oracle判定は、

実際にstudent lossへ入るpseudo boxとGT boxが
同じ座標系になった後

に行う。

teacher側でL/R/T/Bを判定してmaskだけstudent側へ持ち込まない。

特に確認：

horizontal flip
crop
mosaic
resize / letterbox

horizontal flip時にLeft/Rightを取り違えないこと。

4.3 Pseudo–GT matching

trainingで実際に利用されるpseudo boxとGTを、

same class
IoU >= 0.5
deterministic one-to-one matching

で対応付ける。

既存の解析matcherがあれば再利用する。

unmatched pseudo boxはOracle介入しない。

4.4 Wrong-edge definition

clip済みGTに対して、

$$ e_L= \frac{|\hat l-l^{GT}|}{w_{GT}}, \qquad e_R= \frac{|\hat r-r^{GT}|}{w_{GT}} $$ $$ e_T= \frac{|\hat t-t^{GT}|}{h_{GT}}, \qquad e_B= \frac{|\hat b-b^{GT}|}{h_{GT}} $$

を計算する。

$$ e_e \ge 0.10 $$

をOracle-selected edgeとする。

thresholdは固定。

GTはedge選択以外には絶対に使わない。

禁止：

GT targetへの置換
GTによるpseudo box修正
GTによるassignment変更
GTによるclassification変更
GTによるpseudo-label追加・削除
5. DFL-confidence selector

こちらではGTを一切使用しない。

既存のDFL edge-confidence実装をそのまま再利用する。

現在使っているedge confidence：

max bin probability + larger adjacent-bin probability

の既存関数を使用する。

新しく似た計算を再実装しない。

selected edge：

$$ c^{edge}<0.60 $$

とする。

つまり、

edge confidence < 0.60
→ selected

のみ。

重要

既存の

use_edge_conf=True

によるlegacy maskingと、
新しいdose-response weightingが二重に適用されないこと。

新しいdose-response modeでは、

DFL selector
↓
selected mask
↓
common weighting

だけを使用する。

旧mask pathはdisableするか、共通処理へ統合する。

6. Assigner mapping

pseudo object単位の4-edge maskを、
そのobjectを担当するpositive predictionへ伝播する。

必ずTaskAlignedAssignerが返した

target_gt_idx

等の実際のobject identityを利用する。

後からIoUでpredictionとpseudo boxを再matchingしない。

つまり、

pseudo object i
    ↓
selected edges [L,T,R,B]
    ↓
TaskAlignedAssigner
    ↓
object i にassignedされた全positive predictions
    ↓
同じedge maskを適用

とする。

7. DFL lossへの適用位置

weightは必ず4辺reduce前に掛ける。

正しい形：

$$ \ell_{j,L}, \ell_{j,T}, \ell_{j,R}, \ell_{j,B} $$

↓

edge weight

↓

sum / mean

既に4辺をreduceしたDFL scalarにweightを掛ける実装は禁止。

8. 変更してはいけないもの

edge weighting以外はbaselineと完全に同じ。

変更禁止：

pseudo-label selection
classification loss
negative classification loss
box / CIoU loss
TaskAlignedAssigner
foreground mask
EMA teacher
augmentation
confidence threshold
loss_balance設定

特に、

Oracle/DFL selectorによってassignmentを変更しない。

9. loss_balance / ema_scaleとの関係

既存のloss_balance / ema_scaleが、
新しいDFL weighting効果を後段で打ち消していないか確認する。

最低限以下をlog可能にする。

raw DFL loss
weighted DFL loss
normalized weighted DFL loss
loss_balance後の最終DFL contribution
selected edge count
eligible edge count
selected edge rate
mean raw weight
mean normalized weight
selected-edge mean normalized weight
non-selected-edge mean normalized weight

Oracle / DFL両モードで同じloggingを使う。

10. Legacy DFL maskingを再現可能にしておく

注意：

今回のPrimary dose-responseではweight normalizationを行うため、

DFL selector + w=0 + normalize=true

は、既存DFL-selected maskingと完全には同じではない可能性がある。

そのため、実装上は可能なら

edge_dfl_normalize: true | false

を持たせる。

normalize=true
dose-response用
normalize=false
legacy DFL mask再現用

とする。

ただし、このマシンでは追加実験を実行しない。

11. 必須unit tests

Full trainingはしないが、以下は必ず通す。

Test 1: w=1 baseline equivalence

同一batch・同一model weightsで、

original baseline
Oracle selector + w=1
DFL selector + w=1

について、

total loss
cls loss
box loss
DFL loss

が数値誤差範囲で一致すること。

これが最重要。

Test 2: Oracle one-edge synthetic test

1 objectでbottomだけOracle-selectedにする。

w=0で、

bottom DFL contributionだけ抑制
L/T/Rは残る
cls不変
box loss不変
assignment不変

を確認。

Test 3: DFL one-edge synthetic test

DFL confidenceを

L = high
T = high
R = high
B = low

としたsynthetic predictionで、
bottomだけselectedになること。

Test 4: w=5 / 20 normalization

例としてselected edge 1個、non-selected 3個の場合、

raw weight:

w=5:
[1,1,1,5]

が、normalization後に平均1になること。

w=20も同様。

Test 5: GT clip

画像外へ伸びるCrowdHuman fboxを使い、

clip後GT
edge error
Oracle wrong mask

が期待どおりになること。

Test 6: horizontal flip

Leftだけwrongなobjectをhorizontal flipした際、

student coordinateではRight側の対応が正しく判定されること。

Test 7: empty cases

以下でNaN/infが出ないこと。

pseudo label 0件
matched pseudo object 0件
selected edge 0件
foreground 0件
w=0
12. Smoke test

このマシンではfull trainingはしない。

可能な範囲で短いsmokeだけ実行する。

最低限：

Oracle
w=0
w=1
w=20
DFL
w=0
w=1
w=20

数iterationで、

forward
backward
finite loss
selected edge count
normalized weight
no NaN / inf

を確認する。

GPUが無ければCPU smokeでよい。

13. 別マシン用launcherを作成

Full training用launcherを作成するが、
このマシンでは実行しない。

実験条件：

CrowdHuman 1%
YOLOv8n
640 px
100 epochs
seeds: 0, 1
Oracle
w = 0
w = 1
w = 5
w = 20

計8 runs。

DFL
w = 0
w = 1
w = 5
w = 20

計8 runs。

合計16 runs。

ただし w=1 が完全に同一baselineになるため、
実際の別マシン実行時に重複runを省略できるようコメントしておく。

14. Portability

今回は別マシンへコードを移すため、以下を守る。

absolute pathを新規に埋め込まない
GPU番号をhard-codeしない
dataset pathは既存configを使用
launcherはdeviceを引数で変更可能にする
random seedを明示
config keyをDDPへ正しくforward
resume時にもselector / weight設定が保持されること
15. 実装完了時の検証

以下を実行する。

unit tests
compileall
git diff --check

可能ならCPU/GPU smoke。

16. 成果物

以下を作成する。

results/edge_dfl_reweight_implementation.md

記載：

変更したfile / function
共通weightingの実装
Oracle selector
DFL selector
GT clipping処理
student座標変換
assigner identity mapping
normalization
loss_balanceとの接続
unit test結果
smoke結果
known limitations
別マシンで実行するcommand

さらに、

Oracle launcher
DFL launcher
必要なconfig
test code

を保存する。

重要な禁止事項

このマシンでは、

100 epoch full trainingを開始しない
thresholdを変更しない
weightを追加しない
mAPを見てvariantを増やさない
Oracle GTをedge選択以外へ漏らさない
Oracle版とDFL版でweighting implementationを分岐させない

実装と検証だけを完了し、
full experimentは別マシンで実行できる状態にして終了する。