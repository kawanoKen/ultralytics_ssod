## SSOD (Semi-Supervised Object Detection)
ultralyticsに半教師あり学習を実装。Efficient Teacherを参考に、YOLOv8以降にも対応可能に編集。

- 何をするか: ラベルあり + ラベルなしで半教師あり学習（擬似ラベル）を行います。

### YAML（ラベル有/無の分離）
```yaml
path: <root>
train: <labeled.txt or images dir>        # ラベルあり
ssod_train: <unlabeled.txt or images dir> # ラベルなし
val: <images dir>
names: {0: cls0, 1: cls1}
```

### 最短実行例（Python）
```python
from ultralytics.models.yolo.detect.ssod_train import SSODTrainer
SSODTrainer(overrides=dict(
    model="yolov8n.pt",
    data="path/to.yaml",
    cfg="ultralytics/cfg/default_ssod.yaml",
    epochs=100,
)).train()
```

### 実装ポイント（本SSOD）
- 疑似ラベル生成＋Consistency（一体運用）: burn-in終了後にTeacher(EMA)の出力を`conf_threshold_high/low`で擬似ラベル化し、Studentの検出損失（box/cls/DFL）で一致させる（= 一貫性）。強度は`ssod_weight`、可視化は`pseudo_label_plots`。

### 実装箇所とON/OFF
- 疑似ラベル生成＋Consistency（同一フェーズ）
  - どこ: `ultralytics/models/yolo/detect/ssod_train.py` の `SSODTrainer._do_train()`（burn-in後にTeacher=EMA → `non_max_suppression` → 擬似ラベル化 → `EfficientTeacherLoss`で学習）
  - 切替/調整:
    - ON/OFFの主制御: `burn_in_epochs`（このepoch以降で両方有効。0で最初からON、非常に大きくすると実質OFF）
    - 追加調整: `ssod: True`、`ssod_weight`、`conf_threshold_high/low`、`pseudo_label_plots`
- Domain Adaptation（任意の補助正則化）
  - どこ: `ultralytics/models/yolo/detect/ssod_train.py` の `_do_train()` DAブロック（`DomainAdversarialNet`）
  - 切替/調整: `domain_adaptation`（ON/OFF）、`da_loss_weights`（重み）
