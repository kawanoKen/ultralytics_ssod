"""Create tie-aware point estimates and 10k image-paired bootstrap for the scale ablation."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path("results/dfl_assignment")
METHODS = {
    "fixed_2.5": ROOT / "scale_fixed_0p025",
    "fixed_5": ROOT / "scale_fixed_0p05",
    "fixed_7.5": ROOT / "scale_fixed_0p075",
    "fixed_10": ROOT / "scale_fixed_0p10",
    "fixed_15": ROOT / "scale_fixed_0p15",
    "width": ROOT / "scale_width_matched",
    "object_average": ROOT / "scale_object_average",
    "instability_matched_fixed": ROOT / "instability_matched_fixed_final",
    **{f"shuffled_{seed}": ROOT / f"scale_shuffled_stride_{seed}" for seed in range(10)},
}


def load_counts(directory, image_index):
    counts = np.zeros((len(image_index), 10, 2), dtype=np.int64)
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            score_bin = int(round(float(row["instability"]) * 9))
            counts[image_index[row["image"]], score_bin, int(row["assignment_error"])] += 1
    return counts


def metrics(counts):
    """Accept (..., 10, 2) low-to-high score-bin counts; return tie-aware AUROC and AP."""
    neg, pos = counts[..., :, 0], counts[..., :, 1]
    total_pos, total_neg = pos.sum(-1), neg.sum(-1)
    lower_neg = np.cumsum(neg, axis=-1) - neg
    auc = (pos * (lower_neg + 0.5 * neg)).sum(-1) / (total_pos * total_neg)
    pos_desc, neg_desc = pos[..., ::-1], neg[..., ::-1]
    tp, fp = np.cumsum(pos_desc, axis=-1), np.cumsum(neg_desc, axis=-1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    ap = ((pos_desc / total_pos[..., None]) * precision).sum(-1)
    return np.stack((auc, ap), axis=-1)


with (METHODS["width"] / "images.csv").open(newline="", encoding="utf-8") as file:
    images = [row["image"] for row in csv.DictReader(file)]
image_index = {name: index for index, name in enumerate(images)}
counts = {name: load_counts(path, image_index) for name, path in METHODS.items()}
summaries = {name: json.load((path / "summary.json").open()) for name, path in METHODS.items()}
points = {name: metrics(value.sum(0)).tolist() for name, value in counts.items()}

rng = np.random.default_rng(20260910)
weights = rng.multinomial(len(images), np.full(len(images), 1 / len(images)), size=10_000)
boot = {name: metrics(np.einsum("bi,isk->bsk", weights, value)) for name, value in counts.items()}
shuffle_names = [f"shuffled_{seed}" for seed in range(10)]
shuffle_points = np.array([points[name] for name in shuffle_names])
shuffle_boot = np.stack([boot[name] for name in shuffle_names]).mean(0)

comparisons = {}
for label, other, other_boot in (
    ("width_minus_fixed_5", "fixed_5", boot["fixed_5"]),
    ("width_minus_fixed_7.5", "fixed_7.5", boot["fixed_7.5"]),
    ("width_minus_shuffled", None, shuffle_boot),
    ("width_minus_object_average", "object_average", boot["object_average"]),
    (
        "object_average_minus_instability_matched_fixed",
        "instability_matched_fixed",
        boot["instability_matched_fixed"],
    ),
):
    reference = "object_average" if label == "object_average_minus_instability_matched_fixed" else "width"
    difference = boot[reference] - other_boot
    point_other = np.array(points[other]) if other else shuffle_points.mean(0)
    comparisons[label] = {
        "delta_auroc": float(np.array(points[reference])[0] - point_other[0]),
        "auroc_ci": np.quantile(difference[:, 0], (0.025, 0.975)).tolist(),
        "delta_auprc": float(np.array(points[reference])[1] - point_other[1]),
        "auprc_ci": np.quantile(difference[:, 1], (0.025, 0.975)).tolist(),
    }

shuffle_sd = shuffle_points.std(0, ddof=1)
shuffle_ci_half = t.ppf(0.975, 9) * shuffle_sd / np.sqrt(10)
output = {
    "bootstrap": {"unit": "image", "replicates": 10_000, "seed": 20260910},
    "tie_aware_points": points,
    "shuffled": {
        "mean": shuffle_points.mean(0).tolist(),
        "sd": shuffle_sd.tolist(),
        "mean_95_ci": np.stack((shuffle_points.mean(0) - shuffle_ci_half, shuffle_points.mean(0) + shuffle_ci_half), 1).tolist(),
        "per_seed": shuffle_points.tolist(),
    },
    "comparisons": comparisons,
    "summaries": summaries,
}
(ROOT / "perturbation_scale_ablation.json").write_text(json.dumps(output, indent=2) + "\n")
