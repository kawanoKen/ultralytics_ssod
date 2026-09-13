"""CrowdHuman Experiment 1: fixed-quality pseudo-instance selection.

This file owns the offline oracle pipeline (candidate generation, GT matching,
descriptor calculation and selection) and the small command-line wrapper around
the fixed-candidate trainer.

Typical 1% flow::

  python scripts/crowdhuman/experiment1.py generate ...
  python scripts/crowdhuman/experiment1.py select ...
  python scripts/crowdhuman/experiment1.py train ...

The selected artifact contains only pseudo boxes needed by training.  GT and the
oracle descriptors remain in the offline candidate/selection artifact and are not
loaded by the student dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.experiment1_train import Experiment1Trainer
from ultralytics.utils import LOGGER, YAML


QUALITY_BINS = (
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90-1.00"),
)


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_image_path(raw: str, dataset_root: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (dataset_root / raw.lstrip("./")).resolve()


def image_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def box_area(box: list[float] | np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_one(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def load_annotations(path: Path) -> dict[str, list[dict]]:
    annotations = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            gts = []
            for gt_index, box in enumerate(rec.get("gtboxes", [])):
                if box.get("tag") != "person" or "fbox" not in box:
                    continue
                gts.append(
                    {
                        "gt_index": gt_index,
                        "fbox": xywh_to_xyxy(box["fbox"]),
                        "vbox": xywh_to_xyxy(box.get("vbox", box["fbox"])),
                    }
                )
            annotations[rec["ID"]] = gts
    return annotations


def descriptors_for_gts(gts: list[dict], image_hw: tuple[int, int]) -> list[list[float]]:
    h, w = image_hw
    image_area = max(float(h * w), 1.0)
    descriptors = []
    for i, gt in enumerate(gts):
        fbox = gt["fbox"]
        vbox = gt["vbox"]
        area = max(box_area(fbox), 1.0)
        visibility = min(max(box_area(vbox) / area, 0.0), 1.0)
        neighbor_overlap = max((iou_one(fbox, other["fbox"]) for j, other in enumerate(gts) if j != i), default=0.0)
        cx = (fbox[0] + fbox[2]) / 2.0
        cy = (fbox[1] + fbox[3]) / 2.0
        radius = 2.0 * math.sqrt(area)
        local_crowding = sum(
            1
            for j, other in enumerate(gts)
            if j != i
            and math.hypot(
                cx - (other["fbox"][0] + other["fbox"][2]) / 2.0,
                cy - (other["fbox"][1] + other["fbox"][3]) / 2.0,
            )
            <= radius
        )
        descriptors.append(
            [
                math.log(area / image_area),
                visibility,
                neighbor_overlap,
                float(local_crowding),
            ]
        )
    return descriptors


def image_hw(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return int(image.shape[0]), int(image.shape[1])


def match_predictions(predictions: list[dict], gts: list[dict]) -> list[dict]:
    """Confidence-descending, one-to-one prediction-to-GT matching."""
    unmatched = set(range(len(gts)))
    matched = []
    for pred_index, pred in sorted(
        enumerate(predictions), key=lambda x: (-float(x[1]["teacher_confidence"]), x[0])
    ):
        best_gt = None
        best_iou = 0.0
        for gt_position in sorted(unmatched):
            score = iou_one(pred["box_xyxy"], gts[gt_position]["fbox"])
            if score > best_iou:
                best_iou = score
                best_gt = gt_position
        if best_gt is not None and best_iou >= 0.5:
            unmatched.remove(best_gt)
            pred["matched_gt_position"] = best_gt
            pred["matched_gt_id"] = gts[best_gt]["gt_index"]
            pred["iou_to_gt"] = float(best_iou)
            pred["is_true_positive"] = True
        else:
            pred["matched_gt_position"] = None
            pred["matched_gt_id"] = None
            pred["iou_to_gt"] = float(best_iou)
            pred["is_true_positive"] = False
        pred["prediction_index"] = pred_index
        matched.append(pred)
    return matched


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def generate_candidates(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    unlabeled_list = Path(args.unlabeled_list).resolve()
    annotation_path = Path(args.annotation).resolve()
    output = Path(args.out).resolve()
    annotations = load_annotations(annotation_path)
    image_paths = [resolve_image_path(x, dataset_root) for x in read_list(unlabeled_list)]
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    model = YOLO(args.model)
    model.model.eval()

    records = []
    for start in range(0, len(image_paths), args.batch):
        paths = image_paths[start : start + args.batch]
        results = model.predict(
            source=[str(x) for x in paths],
            imgsz=args.imgsz,
            conf=0.01,
            iou=args.nms_iou,
            max_det=args.max_det,
            batch=args.batch,
            device=args.device,
            augment=False,
            verbose=False,
            stream=False,
        )
        for path, result in zip(paths, results):
            iid = image_id_from_path(path)
            h, w = result.orig_shape
            gts = annotations.get(iid, [])
            gt_descriptors = descriptors_for_gts(gts, (h, w))
            predictions = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                for box, conf, cls in zip(boxes, confs, classes):
                    if float(conf) < args.candidate_conf:
                        continue
                    predictions.append(
                        {
                            "box_xyxy": [float(x) for x in box],
                            "teacher_confidence": float(conf),
                            "cls": int(cls),
                        }
                    )
            matched = match_predictions(predictions, gts)
            for pred in matched:
                gt_pos = pred["matched_gt_position"]
                pred.pop("matched_gt_position", None)
                if gt_pos is not None:
                    pred["descriptor_raw"] = gt_descriptors[gt_pos]
            records.append(
                {
                    "image_id": iid,
                    "image_path": str(path),
                    "image_hw": [int(h), int(w)],
                    "gt_count": len(gts),
                    "gt_descriptors_raw": gt_descriptors,
                    "candidates": matched,
                }
            )
        if (start // args.batch + 1) % 25 == 0 or start + args.batch >= len(image_paths):
            LOGGER.info(f"candidate generation: {min(start + args.batch, len(image_paths))}/{len(image_paths)} images")
            write_json(
                output.with_suffix(".partial.json"),
                {
                    "metadata": {
                        "model": str(Path(args.model).resolve()),
                        "dataset_root": str(dataset_root),
                        "unlabeled_list": str(unlabeled_list),
                        "annotation": str(annotation_path),
                        "imgsz": args.imgsz,
                        "nms_iou": args.nms_iou,
                        "candidate_conf": args.candidate_conf,
                    },
                    "records": records,
                },
            )
    payload = {
        "metadata": {
            "model": str(Path(args.model).resolve()),
            "dataset_root": str(dataset_root),
            "unlabeled_list": str(unlabeled_list),
            "annotation": str(annotation_path),
            "imgsz": args.imgsz,
            "nms_iou": args.nms_iou,
            "candidate_conf": args.candidate_conf,
            "matching": "confidence-descending one-to-one, IoU >= 0.5",
            "descriptor": "[log_fbox_area/image_area, vbox_area/fbox_area, max_neighbor_fbox_IoU, center_crowding_count]",
            "crowding_radius": "2 * sqrt(fbox_area), in original image pixels",
        },
        "records": records,
    }
    write_json(output, payload)
    output.with_suffix(".partial.json").unlink(missing_ok=True)
    valid = sum(
        int(c["is_true_positive"])
        for rec in records
        for c in rec["candidates"]
    )
    LOGGER.info(f"saved {len(records)} images and {valid} oracle-correct candidates to {output}")


def quality_bin(iou: float) -> str | None:
    for low, high, label in QUALITY_BINS:
        if low <= iou < high:
            return label
    return None


def standardizer(candidate_payload: dict) -> tuple[np.ndarray, np.ndarray]:
    values = [d for rec in candidate_payload["records"] for d in rec.get("gt_descriptors_raw", [])]
    if not values:
        raise RuntimeError("D_U_diag contains no GT descriptors")
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def labeled_support(annotation_path: Path, labeled_list: Path, dataset_root: Path, mean, std) -> np.ndarray:
    annotations = load_annotations(annotation_path)
    support = []
    for raw in read_list(labeled_list):
        path = resolve_image_path(raw, dataset_root)
        iid = image_id_from_path(path)
        gts = annotations.get(iid, [])
        desc = descriptors_for_gts(gts, image_hw(path))
        support.extend((np.asarray(x, dtype=np.float64) - mean) / std for x in desc)
    if not support:
        raise RuntimeError("D_L contains no GT support descriptors")
    return np.asarray(support, dtype=np.float64)


def selection_sort_key(item: dict) -> tuple:
    return (-float(item["teacher_confidence"]), item["image_id"], int(item["prediction_index"]))


def pairwise_min_distance(descriptors: np.ndarray, support: np.ndarray) -> np.ndarray:
    if len(support) == 0:
        return np.full(len(descriptors), np.inf, dtype=np.float64)
    # Keep the offline selector bounded when D_U has many candidates and D_L has
    # many GT boxes.  A full (N_candidates, N_support, 4) float64 tensor is both
    # unnecessary and can consume multiple GB.
    result = np.empty(len(descriptors), dtype=np.float64)
    for start in range(0, len(descriptors), 4096):
        block = descriptors[start : start + 4096]
        result[start : start + len(block)] = np.sqrt(
            ((block[:, None, :] - support[None, :, :]) ** 2).sum(axis=2)
        ).min(axis=1)
    return result


def farthest_first(items: list[dict], count: int, support: np.ndarray) -> list[dict]:
    if count >= len(items):
        return list(items)
    descriptors = np.asarray([x["descriptor"] for x in items], dtype=np.float64)
    distances = pairwise_min_distance(descriptors, support)
    available = np.ones(len(items), dtype=bool)
    selected = []
    for _ in range(count):
        scores = np.where(available, distances, -np.inf)
        best_score = float(scores.max())
        candidates = np.flatnonzero(available & np.isclose(scores, best_score))
        best = min(candidates, key=lambda i: (items[i]["image_id"], int(items[i]["prediction_index"])))
        selected.append(items[best])
        available[best] = False
        # Incrementally update each candidate's distance to the growing support;
        # recomputing the complete candidate-vs-support matrix at every step is
        # needlessly quadratic in the selected budget.
        distances = np.minimum(
            distances,
            np.sqrt(((descriptors - descriptors[best]) ** 2).sum(axis=1)),
        )
    return selected


def draw_previews(selected: list[dict], out_path: Path) -> None:
    grouped = defaultdict(list)
    for item in selected:
        grouped[item["image_id"]].append(item)
    canvas_images = []
    for iid in sorted(grouped)[:12]:
        image = cv2.imread(grouped[iid][0]["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        for item in grouped[iid]:
            x1, y1, x2, y2 = map(int, item["box_xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                image,
                f"{item['quality_bin']} c={item['teacher_confidence']:.2f}",
                (x1, max(15, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
        scale = min(1.0, 640.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas_images.append(image)
    if not canvas_images:
        return
    width = max(x.shape[1] for x in canvas_images)
    height = sum(x.shape[0] for x in canvas_images)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for image in canvas_images:
        canvas[y : y + image.shape[0], : image.shape[1]] = image
        y += image.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def select_candidates(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidates).resolve()
    payload = json.loads(candidate_path.read_text())
    mean, std = standardizer(payload)
    support = labeled_support(
        Path(args.annotation).resolve(),
        Path(args.labeled_list).resolve(),
        Path(args.dataset_root).resolve(),
        mean,
        std,
    )
    valid = []
    baseline_count = 0
    for rec in payload["records"]:
        for candidate in rec["candidates"]:
            if float(candidate["teacher_confidence"]) >= 0.5:
                baseline_count += 1
            if not candidate.get("is_true_positive") or float(candidate["iou_to_gt"]) < 0.5:
                continue
            qbin = quality_bin(float(candidate["iou_to_gt"]))
            if qbin is None:
                continue
            descriptor = (np.asarray(candidate["descriptor_raw"], dtype=np.float64) - mean) / std
            item = {
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "box_xyxy": candidate["box_xyxy"],
                "box_xywh_norm": [
                    (candidate["box_xyxy"][0] + candidate["box_xyxy"][2]) / (2 * rec["image_hw"][1]),
                    (candidate["box_xyxy"][1] + candidate["box_xyxy"][3]) / (2 * rec["image_hw"][0]),
                    (candidate["box_xyxy"][2] - candidate["box_xyxy"][0]) / rec["image_hw"][1],
                    (candidate["box_xyxy"][3] - candidate["box_xyxy"][1]) / rec["image_hw"][0],
                ],
                "teacher_confidence": candidate["teacher_confidence"],
                "cls": candidate.get("cls", 0),
                "matched_gt_id": candidate["matched_gt_id"],
                "iou_to_gt": candidate["iou_to_gt"],
                "quality_bin": qbin,
                "descriptor": descriptor.tolist(),
                "distance_from_labeled": float(pairwise_min_distance(descriptor[None], support)[0]),
                "prediction_index": candidate["prediction_index"],
            }
            valid.append(item)

    by_bin = {label: [] for _, _, label in QUALITY_BINS}
    for item in valid:
        by_bin[item["quality_bin"]].append(item)
    counts = {key: len(value) for key, value in by_bin.items()}
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"At least one quality bin is empty: {counts}")
    if args.per_bin is None:
        per_bin = min(min(counts.values()), max(1, baseline_count // len(QUALITY_BINS)))
    else:
        per_bin = int(args.per_bin)
    if per_bin < 1 or any(count < per_bin for count in counts.values()):
        raise RuntimeError(f"per_bin={per_bin} exceeds candidates: {counts}")
    LOGGER.info(f"candidate counts by IoU bin={counts}; baseline conf>=0.5={baseline_count}; fixed per-bin budget={per_bin}")

    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for method in args.methods:
        selected = []
        for bin_index, (_, _, label) in enumerate(QUALITY_BINS):
            items = sorted(by_bin[label], key=lambda x: (x["image_id"], int(x["prediction_index"])))
            if method == "CONF":
                chosen = sorted(items, key=selection_sort_key)[:per_bin]
            elif method == "RANDOM":
                # Use the bin index rather than Python's process-randomized hash so
                # the selected RANDOM artifact is reproducible across invocations.
                rng = random.Random(args.seed + 1009 * bin_index)
                chosen = list(items)
                rng.shuffle(chosen)
                chosen = chosen[:per_bin]
            elif method == "COVERAGE":
                chosen = farthest_first(items, per_bin, support)
            elif method == "NEAR":
                chosen = sorted(items, key=lambda x: (x["distance_from_labeled"], x["image_id"], int(x["prediction_index"])))[:per_bin]
            else:
                raise ValueError(f"unknown method {method}")
            selected.extend(chosen)

        result = {
            "metadata": {
                "method": method,
                "selection_seed": args.seed,
                "candidate_file": str(candidate_path),
                "labeled_list": str(Path(args.labeled_list).resolve()),
                "annotation": str(Path(args.annotation).resolve()),
                "quality_bins": [label for _, _, label in QUALITY_BINS],
                "per_bin": per_bin,
                "budget": len(selected),
                "candidate_counts_by_bin": counts,
                "descriptor_mean": mean.tolist(),
                "descriptor_std": std.tolist(),
                "baseline_conf_ge_0.5_count": baseline_count,
            },
            "selected": selected,
        }
        out = output_dir / f"{method}.json"
        write_json(out, result)
        draw_previews(selected, output_dir / f"preview_{method}.jpg")
        summaries[method] = {
            "budget": len(selected),
            "per_bin": per_bin,
            "mean_iou": statistics.mean(x["iou_to_gt"] for x in selected),
            "mean_conf": statistics.mean(x["teacher_confidence"] for x in selected),
            "mean_distance_from_labeled": statistics.mean(x["distance_from_labeled"] for x in selected),
            "bin_counts": {label: sum(x["quality_bin"] == label for x in selected) for _, _, label in QUALITY_BINS},
        }
    write_json(
        output_dir / "selection_summary.json",
        {"metadata": {"per_bin": per_bin, "budget": per_bin * len(QUALITY_BINS), "counts": counts}, "methods": summaries},
    )


def make_train_yaml(args: argparse.Namespace) -> Path:
    base = YAML.load(args.data)
    base["e1_selection"] = str(Path(args.selection).resolve())
    base["e1_method"] = args.method
    base["e1_seed"] = int(args.seed)
    base["e1_max_updates"] = int(args.updates)
    base["e1_val_interval"] = int(args.val_interval)
    # BaseTrainer removes the run directory before launching its DDP worker
    # subprocess.  Keep this descriptor outside that directory so the worker can
    # still resolve the fixed selection artifact.
    out = Path(args.project).resolve() / "_data" / f"{args.method}_seed{args.seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(out, base)
    return out


def train_experiment(args: argparse.Namespace) -> None:
    data_yaml = make_train_yaml(args)
    data_cfg = YAML.load(data_yaml)
    dataset_root = Path(data_cfg["path"])
    pseudo_list = Path(data_cfg["ssod_train"])
    if not pseudo_list.is_absolute():
        pseudo_list = dataset_root / pseudo_list
    n_images = len(read_list(pseudo_list))
    steps_per_epoch = max(1, n_images // args.batch)
    epochs = max(1, math.ceil(args.updates / steps_per_epoch) + 1)
    run_dir = (Path(args.project).resolve() / args.method / f"seed{args.seed}").resolve()
    overrides = {
        "model": str(Path(args.model).resolve()),
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": 640,
        "batch": args.batch,
        "batch_ssod": args.batch,
        "device": args.device,
        # Keep the method in the project component and use only the seed as the
        # run name.  Ultralytics normalizes args.name to its basename before
        # creating the DDP worker command, so a slash in name would be lost.
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "workers": args.workers,
        "val": bool(args.val),
        "plots": False,
        "amp": True,
        "deterministic": True,
        "seed": int(args.seed),
        "optimizer": "auto",
        "close_mosaic": 0,
        "ssod": True,
        "burn_in_epochs": 0,
        "domain_adaptation": False,
        "mosaic_ssod": 0.0,
        "mixup_ssod": 0.0,
        "cutmix_ssod": 0.0,
        "ssod_weight": args.ssod_weight,
        "nbs": args.batch,
        "resume": False,
    }
    LOGGER.info(
        f"starting Experiment 1 train: method={args.method}, seed={args.seed}, "
        f"updates={args.updates}, batch={args.batch}, epochs={epochs}, data={data_yaml}"
    )
    trainer = Experiment1Trainer(overrides=overrides)
    trainer.train()


def report_experiment(args: argparse.Namespace) -> None:
    selection_summary = json.loads((Path(args.selection_dir) / "selection_summary.json").read_text())
    rows = []
    root = Path(args.runs_dir)
    for method in args.methods:
        values = []
        for seed in args.seeds:
            path = root / method / f"seed{seed}" / "learning_curve.csv"
            if not path.exists():
                continue
            with path.open() as f:
                curve = list(csv.DictReader(f))
            if not curve:
                continue
            final = curve[-1]
            values.append(
                {
                    "seed": seed,
                    "mAP50-95": float(final["mAP50-95"]),
                    "mAP50": float(final["mAP50"]),
                    "precision": float(final["precision"]),
                    "recall": float(final["recall"]),
                }
            )
        row = {"method": method, "n_seeds": len(values)}
        for key in ("mAP50-95", "mAP50", "precision", "recall"):
            series = [x[key] for x in values if math.isfinite(x[key])]
            row[key] = f"{statistics.mean(series):.4f} +/- {statistics.stdev(series):.4f}" if len(series) > 1 else (f"{series[0]:.4f}" if series else "NA")
        rows.append(row)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CrowdHuman Experiment 1 report",
        "",
        "This is an oracle diagnostic: only matched IoU>=0.5 pseudo boxes were eligible for selection.",
        "",
        "## Selected pseudo-labels",
        "",
        "| Method | #PL | Mean IoU | Mean confidence | Mean distance from labeled |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, summary in selection_summary["methods"].items():
        lines.append(
            f"| {method} | {summary['budget']} | {summary['mean_iou']:.4f} | "
            f"{summary['mean_conf']:.4f} | {summary['mean_distance_from_labeled']:.4f} |"
        )
    lines += [
        "",
        "## Detection performance",
        "",
        "| Method | Seeds | mAP50-95 | mAP50 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['n_seeds']} | {row['mAP50-95']} | {row['mAP50']} | "
            f"{row['precision']} | {row['recall']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Interpret only after confirming equal #PL and equal IoU-bin counts. "
        "Use `INCONCLUSIVE` if fewer than three seeds or validation metrics are missing.",
    ]
    out.write_text("\n".join(lines) + "\n")
    LOGGER.info(f"wrote report to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True)
    generate.add_argument("--dataset-root", required=True)
    generate.add_argument("--unlabeled-list", required=True)
    generate.add_argument("--annotation", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--imgsz", type=int, default=640)
    generate.add_argument("--nms-iou", type=float, default=0.65)
    generate.add_argument("--candidate-conf", type=float, default=0.05)
    generate.add_argument("--max-det", type=int, default=300)
    generate.add_argument("--batch", type=int, default=16)
    generate.add_argument("--device", default="0")
    generate.add_argument("--max-images", type=int, default=0, help="debug limit; 0 means all images")

    select = sub.add_parser("select")
    select.add_argument("--candidates", required=True)
    select.add_argument("--dataset-root", required=True)
    select.add_argument("--annotation", required=True)
    select.add_argument("--labeled-list", required=True)
    select.add_argument("--out-dir", required=True)
    select.add_argument("--methods", nargs="+", default=["CONF", "RANDOM", "COVERAGE", "NEAR"])
    select.add_argument("--per-bin", type=int, default=None)
    select.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--selection", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--method", required=True, choices=["CONF", "RANDOM", "COVERAGE", "NEAR"])
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--updates", type=int, default=2000)
    train.add_argument("--val-interval", type=int, default=500)
    train.add_argument("--batch", type=int, default=64)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--device", default="0,1,2,3")
    train.add_argument("--project", required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--ssod-weight", type=float, default=1.0)
    train.add_argument("--val", action="store_true")

    report = sub.add_parser("report")
    report.add_argument("--selection-dir", required=True)
    report.add_argument("--runs-dir", required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--methods", nargs="+", default=["CONF", "RANDOM", "COVERAGE", "NEAR"])
    report.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_candidates(args)
    elif args.command == "select":
        select_candidates(args)
    elif args.command == "train":
        train_experiment(args)
    elif args.command == "report":
        report_experiment(args)


if __name__ == "__main__":
    main()
