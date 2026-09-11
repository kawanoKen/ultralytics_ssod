"""Plot the frozen CrowdHuman 5% baseline and R1 learning curves."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "confidence-only": Path("runs/crowdhuman_ssod_5p/yolov8n_voc_ssod_baseline/results.csv"),
    "DFL-selection": Path("runs/crowdhuman_ssod_5p/yolov8n_voc_ssod_dfl/results.csv"),
    "R1-Fixed": Path("runs/r1_preliminary/r1_fixed/results.csv"),
    "R1-Width": Path("runs/r1_preliminary/r1_width/results.csv"),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "ssod/box_loss", "ssod/dfl_loss")
titles = ("Validation mAP50", "Validation mAP50:95", "SSOD box loss", "SSOD DFL loss")
for name, path in RUNS.items():
    rows = list(csv.DictReader(path.open()))
    epochs = [int(row["epoch"]) for row in rows]
    for axis, key in zip(axes.flat, keys):
        axis.plot(epochs, [float(row[key]) for row in rows], label=name, linewidth=1.5)
for axis, title in zip(axes.flat, titles):
    axis.set(title=title, xlabel="Epoch")
    axis.grid(alpha=0.25)
axes[0, 0].legend(fontsize=8)
out = Path("results/r1_preliminary/r1_learning_curves.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
