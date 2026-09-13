# H1 Oracle selector: remote-machine execution instructions

この手順は、edge-wise DFL supervision dose-responseのうちOracle selector側を、4 GPUの別マシンで実行するためのものです。こちらのマシンでは同じ条件でDFL selectorを実行するため、selector以外の条件は変更しません。

## 実行前

リポジトリをH1実装を含むcommitへ更新する。

```bash
git pull
git rev-parse --short HEAD
uv sync
```

Oracleは変換済みのunlabeled GT labelをstudent座標系で使う。CrowdHuman labelがこのrevisionの変換器で作られていない場合、実験前にdataset rootを明示して再生成する。これは既存labelを上書きするため、必要なら先にdataset側をバックアップする。

```bash
uv run python scripts/crowdhuman/convert_odgt_to_yolo.py --root /path/to/crowdhuman
```

`annotation_train.odgt`、`images/train`、`labels/train` と、validation側にも同じroot内でアクセスできることを確認する。この変換器は画像外に伸びるCrowdHuman fboxを画像境界へclipする。

## 4 GPUで全8 runを起動

launcherはw=1をbaseline等価controlとして既定では省略する。今回の表を完全に埋めるため、環境変数でw=1も含める。

```bash
RUN_W1_CONTROLS=1 \
bash scripts/crowdhuman/run_edge_dfl_dose_response_1p.sh --all oracle 0,1,2,3 \
  > scripts/crowdhuman/logs/h1_oracle_all.log 2>&1
```

実行条件は以下の8本。

| selector | weight | seeds |
|---|---|---|
| oracle | 0, 1, 5, 20 | 0, 1 |

各runはCrowdHuman 1%、YOLOv8n、640 px、100 epochs、`batch=batch_ssod=128`、`loss_balancing_mode=ema_scale`、`skip_zero_pseudo_cls_loss=True` を使用する。device番号はlauncher末尾の引数で変更できる。

## 起動直後の確認

各runの`args.yaml`に以下があることを確認する。

```yaml
edge_dfl_reweight: true
edge_dfl_selector: oracle
edge_dfl_weight: 0.0  # runごとに 0/1/5/20
edge_dfl_normalize: true
oracle_edge_error_threshold: 0.1
use_edge_conf: false
use_loc_conf: false
```

`logs/assignment_dynamics.csv`の`unsup_edge_dfl_*`列を確認する。最低限、以下を保存する。

- `selected_edges`, `eligible_edges`, `selected_edge_rate`
- raw / normalized weightの平均
- raw / weighted / normalized DFL loss
- `normalization_zero_sum_batch(es)`
- `loss_balance_factor`, `final_contribution`

`w=1`ではselectorに関わらず既存baselineと同じloss経路になるよう実装している。`w=0`でselected edgeが全eligible edgeを占めるbatchは、DFL contributionが0となり、zero-sum counterへ記録される。

## 出力場所

各runは次の形式で保存される。

```text
runs/crowdhuman_ssod_1p_zero_pseudo_ablation/
  yolov8n_1p_loss_balance_edge_dfl_oracle_w{0|1|5|20}_seed{0|1}/
```

終了後は各runの`results.csv`、`weights/best.pt`、`weights/last.pt`、`logs/assignment_dynamics.csv`を保持する。
