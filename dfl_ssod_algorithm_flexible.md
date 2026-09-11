# DFL Localization Confidence + SSOD 実装方針
## — 最小拘束・高自由度版 —

## 0. 目的

YOLO 系 detector の DFL（Distribution Focal Loss）回帰分布から
**localization confidence** を取り出し、半教師あり物体検出（SSOD）の
pseudo-label 利用に組み込む。

この実装では、論文のアイデアを再現可能な最小構成だけ固定し、
それ以外の SSOD 設計は実装者が自由に選べるものとする。

---

# 1. 固定するコアアイデア

論文由来として固定するのは、基本的に以下だけとする。

## 1.1 DFL 出力を localization uncertainty / confidence として利用する

各 bounding box の各辺について DFL の離散分布

```text
p(edge, bin)
```

を取得する。

---

## 1.2 Edge confidence

各辺について最大確率 bin を `k_max` とし、

\[
c^{edge}
=
p(k_{\max})
+
\max
\left(
p(k_{\max}-1),
p(k_{\max}+1)
\right)
\]

とする。

範囲外の bin は 0 とする。

この指標は、

- 1 bin に強く集中した分布
- 隣接 2 bin に確率が集中した分布

の両方を高 confidence と評価する。

---

## 1.3 Box confidence

4 辺の localization confidence を

\[
c^{box}
=
\min
\left(
c_l^{edge},
c_t^{edge},
c_r^{edge},
c_b^{edge}
\right)
\]

で box-level confidence にまとめる。

論文では「1 辺でも不確実なら box 全体の localization が不安定」
という考え方を採用している。

---

## 1.4 SSOD で利用する

Teacher が生成した pseudo-label に

```text
classification confidence
localization confidence
```

を持たせ、pseudo-label の採用・重み付け・分類などに利用する。

ここまでが実装上のコア。

---

# 2. 最小インターフェース

SSOD の具体的な構造には依存させず、
pseudo-label を以下のような形式で扱えるようにする。

```python
PseudoLabel(
    box,
    class_id,
    cls_conf,
    loc_conf,
    edge_conf=None,
)
```

最低限必要なのは

```text
box
class_id
cls_conf
loc_conf
```

だけ。

---

# 3. DFL confidence モジュール

```python
def localization_confidence(dfl_logits):
    """
    Input:
        dfl_logits: [..., 4, B]

    Output:
        edge_conf: [..., 4]
        box_conf:  [...]
    """
```

概念実装:

```python
p = softmax(dfl_logits, dim=-1)

p_max, k_max = p.max(dim=-1)

left_prob  = probability_at(k_max - 1)
right_prob = probability_at(k_max + 1)

edge_conf = p_max + max(left_prob, right_prob)

box_conf = min(edge_conf over 4 edges)
```

このモジュールは detector / trainer から独立させる。

---

# 4. SSOD 全体構造

最低限、

```text
Student detector
Teacher detector
Unlabeled image
Pseudo-label generation
Student learning
Teacher update
```

があればよい。

典型的には、

```text
Labeled data --------------------> Student
                                     |
                                     +--> supervised loss

Unlabeled data --> Teacher
                     |
                     +--> pseudo-label
                           |
                           + cls_conf
                           + loc_conf
                           |
                           v
                       selection /
                       weighting /
                       filtering
                           |
                           v
                        Student
```

とする。

Teacher は EMA が自然だが、実装上は固定しない。

---

# 5. Localization confidence の使い方

ここは **自由設計領域** とする。

論文では classification confidence と localization confidence の
2 種類の閾値によって pseudo-label を分類したが、
実装では以下のいずれを選んでもよい。

---

## Option A: 論文に近い hard selection

```text
high cls + high loc
    -> reliable

low cls + low loc
    -> background

otherwise
    -> uncertain
```

例:

```python
reliable = (cls_conf >= tau_high) & (loc_conf >= pi_high)
background = (cls_conf <= tau_low) & (loc_conf <= pi_low)
uncertain = ~(reliable | background)
```

論文再現を優先する場合に使う。

---

## Option B: reliable pseudo-label のみ localization で制限

```python
reliable =
    cls_conf >= tau_cls
    and
    loc_conf >= tau_loc
```

それ以外は ignore。

最も単純。

---

## Option C: confidence weighting

pseudo-label を捨てず、

\[
w_i = f(c_i^{cls}, c_i^{box})
\]

として loss weight を変える。

例:

```python
weight = cls_conf * loc_conf
```

または

```python
weight = min(cls_conf, loc_conf)
```

など。

この部分は論文そのものではないため、研究拡張として扱う。

---

## Option D: localization confidence のみ regression loss に利用

```text
classification loss:
    cls_conf で制御

box / DFL loss:
    loc_conf で制御
```

という分離も可能。

ただし論文では loss-level selection を試した結果、
学習が不安定になったと記述されているため、
採用する場合は別実験として扱う。

---

## Option E: threshold を固定しない

```text
fixed threshold
quantile
class-wise threshold
epoch-wise schedule
EMA statistics
LabelMatch-style adaptive threshold
```

などを選べる。

論文再現でなければ `0.6 / 0.8` に固定する必要はない。

---

# 6. NMS との関係

localization confidence をどの段階で計算・利用するかも実装上は選択可能。

## 案 1

```text
raw prediction
    ↓
NMS
    ↓
残った prediction に loc_conf を付与
    ↓
pseudo-label selection
```

最も論文に近い。

---

## 案 2

```text
raw prediction
    ↓
cls_conf + loc_conf で candidate filtering
    ↓
NMS
```

pseudo-label 候補そのものを変える。

---

## 案 3

```text
raw prediction
    ↓
loc_conf を NMS score に利用
    ↓
NMS
```

これは論文の future work に近い拡張であり、
SSOD 本体とは分けて検証した方がよい。

---

# 7. Teacher / Student の自由度

以下は固定しない。

```text
Teacher:
    EMA teacher
    frozen teacher
    periodically updated teacher

Student:
    YOLOv8
    YOLOv11 / YOLOv12 / YOLOv13
    その他 DFL 系 one-stage detector
```

ただし DFL localization confidence を使うためには、
**box regression の離散分布を取得できること**が必要。

---

# 8. Augmentation の自由度

Teacher / Student で

```text
weak / strong augmentation
```

を使うのが一般的だが、具体的な augmentation は固定しない。

候補:

```text
flip
scale
crop
color jitter
mosaic
mixup
cutout
autoaugment
```

必要条件は、

```text
Teacher が生成した pseudo-box
        ↓
Student 側画像座標
```

への変換が正しく追跡できること。

---

# 9. Loss の自由度

SSOD loss は特定の形式に固定しない。

全体として

\[
L =
L_{sup}
+
\lambda_u L_{unsup}
\]

程度を基本形とする。

`L_unsup` の中身は採用する detector / SSOD framework に合わせる。

例えば、

```text
classification loss
box regression loss
DFL loss
quality/objectness loss
```

など。

重要なのは、

> DFL localization confidence を pseudo-label の品質情報として利用する

ことであり、loss architecture そのものは本手法の必須要素ではない。

---

# 10. Teacher update の自由度

論文では EMA teacher を使用している。

典型形:

\[
\theta_T
\leftarrow
\alpha \theta_T +
(1-\alpha)\theta_S
\]

ただし、

```text
decay
schedule
update frequency
burn-in
```

は実装依存としてよい。

---

# 11. Dataset / split

論文再現では KITTI を使用するが、
手法実装自体は dataset に依存しない。

利用可能な例:

```text
KITTI
COCO
VOC
CrowdHuman
BDD100K
custom dataset
```

labeled / unlabeled ratio も固定しない。

論文再現をしたい場合のみ、

```text
train / val = 90 / 10
train 内 labeled / unlabeled = 10 / 90
```

を使う。

---

# 12. 最小 training loop

アルゴリズムとして必要なのは以下程度。

```python
for labeled_batch, unlabeled_batch in loader:

    # supervised
    pred_l = student(labeled_batch.image)
    loss_sup = supervised_loss(pred_l, labeled_batch.target)

    # pseudo-label generation
    with torch.no_grad():
        teacher_pred = teacher(unlabeled_batch.teacher_image)

        pseudo = decode_predictions(teacher_pred)

        edge_conf, box_conf = localization_confidence(
            teacher_pred.dfl_logits
        )

        pseudo.loc_conf = box_conf

        pseudo = pseudo_label_policy(pseudo)

    # unsupervised
    pred_u = student(unlabeled_batch.student_image)

    loss_unsup = unsupervised_loss(
        pred_u,
        pseudo,
    )

    loss = loss_sup + lambda_u * loss_unsup

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_teacher(student, teacher)
```

`pseudo_label_policy()` の中身は実験ごとに交換できるようにする。

---

# 13. 推奨モジュール分割

実装は以下程度に疎結合にする。

```text
detector/
    model

ssod/
    teacher
    pseudo_label_generator
    pseudo_label_policy
    unsupervised_loss

dfl_confidence/
    confidence.py

evaluation/
    detection_metrics
    pseudo_label_metrics
```

特に

```text
localization_confidence()
```

と

```text
pseudo_label_policy()
```

を分離する。

これにより、

```text
confidence は同じ
policy だけ変更
```

という比較が容易になる。

---

# 14. 最低限の比較実験

実装確認として最低限必要なのは 2 本。

## Baseline

```text
classification confidence のみで pseudo-label を選択
```

## DFL-aware

```text
classification confidence
+
DFL localization confidence
```

これで改善が見えた後に、

```text
threshold
weighting
adaptive policy
NMS
loss-level use
```

などを広げればよい。

---

# 15. 実装時に記録したいもの

最低限:

```text
mAP
pseudo-label 数
cls_conf
loc_conf
```

研究解析をするなら追加で、

```text
pseudo-label precision
pseudo-label recall
reliable / uncertain の数
edge_conf distribution
GT IoU と loc_conf の相関
occlusion と loc_conf の関係
box size と loc_conf の関係
class ごとの loc_conf
```

を保存するとよい。

---

# 16. 実装上の注意点

## DFL logits と detection の対応

localization confidence は raw DFL distribution から計算するため、

```text
最終 detection
    ↔
元 candidate
    ↔
DFL logits
```

の対応を失わないようにする。

具体的な実装方法は detector codebase に依存する。

---

## DFL confidence の計算タイミング

```text
NMS 前に全 candidate に計算
```

でも

```text
NMS で残る candidate index を取得してから計算
```

でもよい。

計算量とコード変更量で選択する。

---

## 境界 bin

`k_max = 0` または `B-1` の場合、
存在しない隣接 bin の probability は 0 とする。

---

# 17. 論文再現と研究実装を分ける

実装は最初から完全再現に縛らない。

## Mode A: Paper reproduction

可能な範囲で論文設定を合わせる。

```text
YOLOv8-s
Efficient Teacher
KITTI
10% labeled
tau_low  = 0.3
tau_high = 0.6
pi_low   = 0.6
pi_high  = 0.8
```

---

## Mode B: Research implementation

固定するのは

```text
DFL edge confidence
box confidence = min(edge confidence)
SSOD で localization confidence を使う
```

のみ。

その他は自由。

---

# 18. 最初の実装ゴール

最初から Efficient Teacher 全仕様や論文の全実験を再現する必要はない。

まず、

```text
1. YOLO の DFL logits を取得できる
2. loc_conf を計算できる
3. Teacher prediction に loc_conf を付与できる
4. classification-only SSOD baseline が動く
5. loc_conf を pseudo-label policy に追加できる
```

までを完成させる。

その後、

```text
どの pseudo-label policy がよいか
どの threshold がよいか
uncertain をどう使うか
adaptive selection にするか
NMS に使うか
```

を実験で決める。

---

# 19. 最小拘束事項まとめ

この研究実装で必須とするのは次の 3 点だけ。

```text
① DFL 分布から edge-level localization confidence を計算する

② 4 edge の minimum を box-level localization confidence とする

③ その localization confidence を SSOD の pseudo-label 品質判断に利用する
```

それ以外の

```text
SSOD framework
loss
threshold
augmentation
teacher update
dataset
NMS
pseudo-label policy
uncertain label の扱い
training schedule
```

は、目的に応じて変更可能とする。
