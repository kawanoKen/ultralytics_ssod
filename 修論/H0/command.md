# Artificial Boundary Noise × DFL Mechanism Experiment

## 1. 目的

人工的に boundary target を誤らせた場合に、

> **誤った edge target を DFL supervision に使用すること自体が検出性能・localization 性能を悪化させるか**

を controlled experiment で検証する。

今回は SSOD の pseudo-label quality や teacher confidence などの要因を除外するため、**100% labeled data を用いた fully-supervised setting** で実験する。

---

# 2. 検証したい仮説

主仮説：

$$
\boxed{
\text{wrong edge を DFL に使用すると localization performance が悪化する}
}
$$

具体的には、同じ noisy box を IoU loss に使用したまま、

* noisy edge を DFL に使用する
* noisy edge を DFL から除外する

の2条件を比較する。

予想：

$$
\text{DFL-OFF} > \text{DFL-ON}
$$

特に noise が大きいほど、

$$
\Delta =
Performance_{\mathrm{OFF}}
-
Performance_{\mathrm{ON}}
$$

が大きくなることを期待する。

---

# 3. 実験条件

Noise level は3段階とする。

* Noise-0: noiseなし
* Noise-L: 過去に取得した edge error 統計に基づく低い noise scale
* Noise-H: 過去に取得した edge error 統計に基づく高い noise scale

Noise-L / Noise-H の具体値は、**既に取得済みの pseudo-edge error 統計から事前に固定する**。

結果を見て noise scale を調整しないこと。

### 今回固定する値

既存のclip済みpseudo-edge統計では、reliable clean edgeの典型的な誤差はbox sideの数%である一方、partial / low-confidence edgeはおよそ10--25%に達する。これに基づき、対象edgeの対応box dimensionに対して以下を固定する。

* Noise-L: `0.05 × {box width (left/right), box height (top/bottom)}`
* Noise-H: `0.20 × {box width (left/right), box height (top/bottom)}`

各objectの実際のabsolute magnitudeは、このscaleの `Uniform[0.5, 1.5]` とし、符号は正負をdeterministicに均等選択する。この値と分布は本実験中に変更しない。

---

## Main experiments

Noise-L / Noise-H について、それぞれ以下を実施する。

| Noise | Corrupted edge の DFL | IoU loss  |             Seeds |
| ----- | -------------------- | --------- | ----------------: |
| 0     | clean                | clean     | existing baseline |
| Low   | ON                   | noisy box |                 2 |
| Low   | OFF                  | noisy box |                 2 |
| High  | ON                   | noisy box |                 2 |
| High  | OFF                  | noisy box |                 2 |

したがって、新規実験は

$$
2\text{ noise levels}
\times
2\text{ DFL conditions}
\times
2\text{ seeds}
=
\boxed{8\text{ runs}}
$$

とする。

Noise-0 baseline は既存結果を利用してよい。

ただし、

* 同一コード
* 同一 dataset split
* 同一 augmentation
* 同一 optimizer / scheduler
* 同一 epoch 数
* 同一 seed

で比較できない baseline の場合は再実行する。

---

# 4. Noise の生成

GT box

$$
b=(l,t,r,b)
$$

を基準とする。

各 object について、**1つの edge のみを corruption 対象にする**。

対象 edge は

$$
e\in\{l,t,r,b\}
$$

から選択する。

edge の選択は deterministic にし、同じ sample については、

* DFL-ON
* DFL-OFF

で必ず同じ edge を corruption すること。

Seed 間でも noise generation と model initialization の影響を分離できるよう、可能なら **corruption map 自体は固定**する。

---

## Noise

対象 edge を

$$
e' = e + \epsilon
$$

とする。

Noise-L:

$$
\epsilon \sim D_{\mathrm{low}}
$$

Noise-H:

$$
\epsilon \sim D_{\mathrm{high}}
$$

とする。

\(D_{\mathrm{low}}\), \(D_{\mathrm{high}}\) の scale は、これまでに測定した実際の pseudo-label boundary error 統計に基づいて決定する。

可能であれば、単純な任意 Gaussian よりも、**観測した edge error の empirical distribution またはその scale** を使用する。

正方向・負方向の両方の error を含める。

---

# 5. Box validity

noise injection 後も、

$$
l' < r', \qquad t' < b'
$$

を必ず満たすようにする。

画像境界外に出た場合も clamp する。

ただし clamp によって実際の noise magnitude が大幅に変化した sample が大量に発生しないことを確認する。

---

# 6. 最重要：loss の実装

通常の regression loss を

$$
L_{\mathrm{reg}}
=
\lambda_{\mathrm{IoU}}L_{\mathrm{IoU}}
+
\lambda_{\mathrm{DFL}}
\sum_e L_{\mathrm{DFL},e}
$$

とする。

---

## DFL-ON

corrupted edge についても noisy target をそのまま DFL に使用する。

$$
L_{\mathrm{DFL}}
=
L_l+L_t+L_r+L_b
$$

corrupted edge の target は \(e'\)。

---

## DFL-OFF

corrupted edge に対する DFL loss のみ 0 にする。

例えば right edge が corrupted された場合、

$$
L_{\mathrm{DFL}}
=
L_l+L_t+0\cdot L_r+L_b
$$

とする。

### 注意

**IoU loss は OFF にしない。**

DFL-ON と DFL-OFF の両方で、

$$
L_{\mathrm{IoU}}
(\hat b,b_{\mathrm{noisy}})
$$

を使用する。

つまり両条件の違いは、

$$
\boxed{
\text{corrupted edge を DFL に入れるかどうかだけ}
}
$$

にする。

classification loss その他の loss も完全に同一とする。

---

# 7. DFL loss normalization

DFL edge を OFF にしたことで、残り3 edge の実効 weight が増加しないようにする。

例えば通常が

$$
L_{\mathrm{DFL}}
=
\frac{L_l+L_t+L_r+L_b}{4}
$$

なら、right edge OFF の場合も

$$
L_{\mathrm{DFL}}
=
\frac{L_l+L_t+0+L_b}{4}
$$

とする。

以下にはしない。

$$
\frac{L_l+L_t+L_b}{3}
$$

後者では、残った edge の DFL weight が \(4/3\) 倍になってしまい、DFL removal の効果と混ざる。

---

# 8. 固定するもの

以下は全条件で完全に固定する。

* dataset
* train / val split
* architecture
* pretrained checkpoint
* batch size
* optimizer
* learning rate
* scheduler
* augmentation
* epochs
* IoU loss weight
* DFL global weight
* classification loss
* evaluation settings
* corruption 対象 object
* corruption 対象 edge
* noise sample

特に DFL-ON / OFF の paired comparison では、

$$
\boxed{\text{DFL mask 以外を変えない}}
$$

こと。

---

# 9. Seed

2 seeds 実施する。

できるだけ paired design とし、

* Low-ON seed1
* Low-OFF seed1

では model initialization / dataloader seed を同一にする。

同様に、

* Low-ON seed2
* Low-OFF seed2
* High-ON seed1
* High-OFF seed1
* High-ON seed2
* High-OFF seed2

も対応させる。

これにより seed variance よりも DFL intervention の影響を見やすくする。

---

# 10. 必須 evaluation

最低限以下を記録する。

### Detection metrics

* mAP50-95
* mAP50
* AP75
* Precision
* Recall

特に今回は localization が目的なので、

$$
\boxed{\mathrm{mAP50-95},\ AP75}
$$

を重視する。

---

## Boundary-specific metric

可能なら validation GT に対し、matched detection について corrupted した方向の boundary error を測定する。

例えば right edge なら、

$$
E_r=
|\hat r-r_{\mathrm{GT}}|
$$

を計算する。

4方向を統合して、

$$
E_{\mathrm{edge}}
=
|\hat e-e_{\mathrm{GT}}|
$$

として報告する。

可能なら pixel 単位ではなく object size で normalize した値も出す。

$$
E_{\mathrm{norm}}
=
\frac{|\hat e-e_{\mathrm{GT}}|}
{\text{box width or height}}
$$

これは global mAP より今回の仮説に直接対応する。

---

# 11. 学習中に最低限確認する項目

追加の大量 logging は不要。

各 run について少なくとも、

* total box / regression loss
* IoU loss
* DFL loss

が正常であることを確認する。

さらに最初の数 iteration だけでよいので、

DFL-ON では corrupted edge の DFL loss が存在し、

DFL-OFF では

$$
L_{\mathrm{DFL, corrupted}}=0
$$

になっていることを確認する。

---

# 12. Sanity check

full training の前に short smoke test を行う。

確認項目：

1. Noise-0 では元 GT と完全一致する
2. Noise-L < Noise-H になっている
3. noisy box が valid
4. DFL-ON/OFF で同一 noisy box を使用
5. OFF では対象 edge の DFL gradient が 0
6. IoU loss は ON/OFF 両方で同一 noisy target を使用
7. 他3 edge の DFL weight が変化していない

---

# 13. 結果表

最終的に以下の形式でまとめる。

| Noise | DFL noisy edge | Seed | mAP50-95 | AP75 | mAP50 |  P |  R | Edge Error |
| ----- | -------------- | ---: | -------: | ---: | ----: | -: | -: | ---------: |
| 0     | clean          |  ... |          |      |       |    |    |            |
| Low   | ON             |    1 |          |      |       |    |    |            |
| Low   | ON             |    2 |          |      |       |    |    |            |
| Low   | OFF            |    1 |          |      |       |    |    |            |
| Low   | OFF            |    2 |          |      |       |    |    |            |
| High  | ON             |    1 |          |      |       |    |    |            |
| High  | ON             |    2 |          |      |       |    |    |            |
| High  | OFF            |    1 |          |      |       |    |    |            |
| High  | OFF            |    2 |          |      |       |    |    |            |

さらに paired difference を出す。

$$
\Delta_{\mathrm{Low}}
=
Metric_{\mathrm{OFF}}
-
Metric_{\mathrm{ON}}
$$

$$
\Delta_{\mathrm{High}}
=
Metric_{\mathrm{OFF}}
-
Metric_{\mathrm{ON}}
$$

---

# 14. 結果の解釈

## Case A

$$
OFF > ON
$$

かつ High noise で差が大きくなる。

→ 主仮説を支持。

**wrong boundary に対する DFL supervision 自体が有害である可能性が高い。**

---

## Case B

Low では差がないが、High では

$$
OFF > ON
$$

→ 通常の pseudo-label error が弱すぎて影響が観測されていなかった可能性。

DFL の害には error severity threshold が存在する可能性がある。

---

## Case C

ON/OFF に差がないが、Noise-0 と noisy conditions の差は大きい。

→ noisy regression supervision は有害だが、

$$
\boxed{\text{主因は DFL ではない}}
$$

可能性が高い。

次の実験として IoU loss の removal を検討する。

---

## Case D

Noise-H でも Noise-0 とほぼ差がない。

→ regression target corruption 自体の影響が予想以上に小さい、または corruption が学習に十分作用していない。

この場合は、

* corruption rate
* noise magnitude
* regression gradient contribution

を確認する。

---

# 15. 今回はやらないこと

この実験では以下を追加しない。

* confidence filtering
* teacher model
* EMA
* pseudo-label selection
* uncertainty estimation
* DFL confidence
* adaptive weighting
* IoU loss removal

まず、

$$
\boxed{
\text{人工的 wrong edge}
\times
\text{DFL ON/OFF}
}
$$

だけを検証する。

IoU removal は、この結果を確認した後の次段階とする。
