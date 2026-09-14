# H0 follow-up: Clean DFL target vs Noisy DFL target

## 目的と固定条件

既存H0 DFL-ONをNoisy DFL controlとして再利用し、IoU targetは同一のnoisy boxに固定したまま、corrupted edgeのDFL targetだけをclean GTへ戻す。新規runはCrowdHuman 100%、YOLOv8n、640 px、100 epoch、global batch 256、seed 0/1、Noise-L=0.05 / Noise-H=0.20の4本だけである。

`h0_dfl_mode=clean`では、assignmentとIoU lossはnoisy boxを使う。DFLは同じforeground anchor・同じassigned GT objectに対してclean boxから距離targetを再計算する。edge weight、mask、reweight、normalizationは使用しない（全edge weight=1）。

## 実行

```bash
git pull --ff-only
mkdir -p scripts/crowdhuman/logs
bash scripts/crowdhuman/run_h0_clean_dfl_target.sh 0,1,2,3 |& tee scripts/crowdhuman/logs/h0_clean_dfl_target.log
```

出力先は以下である。

```text
runs/crowdhuman_h0_boundary_noise/
  yolov8n_full_h0_low_dfl_clean_seed0/
  yolov8n_full_h0_low_dfl_clean_seed1/
  yolov8n_full_h0_high_dfl_clean_seed0/
  yolov8n_full_h0_high_dfl_clean_seed1/
```

各runの`args.yaml`で以下を確認する。

```yaml
h0_boundary_noise: true
h0_noise_fraction: 0.05 # highでは0.20
h0_dfl_mode: clean
h0_corruption_seed: 20260914
```

学習終了後、best checkpointのAP75とclean-GT normalized edge errorを取得する。

```bash
bash scripts/crowdhuman/run_h0_clean_dfl_target_eval.sh 0 |& tee scripts/crowdhuman/logs/h0_clean_dfl_target_eval.log
```

## sanity check

学習開始時に、paired Noisy DFL（既存`*_dfl_on_seed{0,1}`）とClean DFLで、同一noise/seedの`h0_corruption_seed`、noise fraction、model seed、batch、epochが一致することを確認する。`clean`条件のみがDFL targetをclean GTに戻し、IoU targetを戻していないことが実装上の差分である。
