"""Checkpoint-only Step 2--4 analysis at a fixed image size; this script never trains the model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from ultralytics import YOLO
from ultralytics.utils.assignment_stability import (
    assignment_stability,
    dfl_distribution_statistics,
    dfl_supported_box_candidates,
    geometric_box_candidates,
    object_assignment_metrics,
)
from ultralytics.utils.dfl_confidence import localization_confidence
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


def letterbox(image: np.ndarray, size: int) -> tuple[torch.Tensor, float, float, float]:
    """Resize/pad an image to a square and return tensor, ratio, left pad and top pad."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return tensor, ratio, float(left), float(top)


def load_gt(label_path: Path, original_shape: tuple[int, int], ratio: float, left: float, top: float):
    """Load YOLO labels and transform normalized original-image XYWH to letterboxed pixel XYXY."""
    if not label_path.exists() or not label_path.stat().st_size:
        return torch.empty((0,), dtype=torch.long), torch.empty((0, 4))
    values = torch.tensor([[float(x) for x in line.split()] for line in label_path.read_text().splitlines()])
    classes, xywh = values[:, 0].long(), values[:, 1:5]
    height, width = original_shape
    xywh *= torch.tensor([width, height, width, height])
    xyxy = torch.empty_like(xywh)
    xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * ratio + left
    xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * ratio + top
    xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * ratio + left
    xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * ratio + top
    return classes, xyxy


def run_assigner(assigner, scores, boxes, anchors, labels, targets) -> torch.Tensor:
    """Return local object identity per dense prediction, using -1 for background."""
    labels = labels.view(1, -1, 1).to(scores.device)
    targets = targets.view(1, -1, 4).to(scores.device)
    valid = targets.sum(2, keepdim=True).gt(0)
    _, _, _, foreground, target_idx = assigner(scores, boxes, anchors, labels, targets, valid)
    return torch.where(foreground.bool(), target_idx.long(), -1)[0]


def match_pseudo_to_gt(pseudo_boxes, pseudo_classes, gt_boxes, gt_classes, threshold=0.5):
    """Class-aware one-to-one IoU matching; unmatched pseudo objects map to -2."""
    mapping = torch.full((len(pseudo_boxes),), -2, dtype=torch.long)
    if not len(pseudo_boxes) or not len(gt_boxes):
        return mapping
    iou = box_iou(pseudo_boxes, gt_boxes)
    iou[pseudo_classes[:, None] != gt_classes[None, :]] = -1
    pseudo_idx, gt_idx = linear_sum_assignment((-iou).cpu().numpy())
    for p, g in zip(pseudo_idx, gt_idx):
        if iou[p, g] >= threshold:
            mapping[p] = int(g)
    return mapping


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute tie-aware AUROC and threshold-grouped average precision."""
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan, math.nan
    levels = np.unique(scores)[::-1]
    positive_counts = np.array([labels[scores == level].sum() for level in levels], dtype=float)
    negative_counts = np.array([(scores == level).sum() for level in levels], dtype=float) - positive_counts
    # Probability that a random positive scores higher than a random negative, with ties worth 0.5.
    auroc = sum(
        positive_counts[i] * (negative_counts[i + 1 :].sum() + 0.5 * negative_counts[i])
        for i in range(len(levels))
    ) / (positives * negatives)
    tp, fp = np.cumsum(positive_counts), np.cumsum(negative_counts)
    precision = tp / (tp + fp)
    auprc = np.sum((positive_counts / positives) * precision)
    return float(auroc), float(auprc)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_r2_soft_stats(stats, bins, baseline_negative, gt_positive, instability, logits):
    """Accumulate negative BCE and its logit-gradient proxy for baseline-negative anchors."""
    probability = logits.sigmoid()
    negative_loss = torch.nn.functional.softplus(logits)
    for region_name, region in (("all", baseline_negative), ("unstable", baseline_negative & (instability > 0))):
        for gt_name, gt_mask in (("gt_positive", gt_positive), ("gt_negative", ~gt_positive)):
            mask = region & gt_mask
            values = stats[region_name][gt_name]
            values["n"] += int(mask.sum())
            values["loss_sum"] += float(negative_loss[mask].sum())
            values["p_sum"] += float(probability[mask].sum())
    for low, high in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.000001)):
        mask = baseline_negative & (instability >= low) & (instability < high)
        values = bins[f"{low:.1f}-{min(high, 1):.1f}"]
        values["n"] += int(mask.sum())
        values["gt_positive"] += int((mask & gt_positive).sum())
        values["loss_sum"] += float(negative_loss[mask].sum())
        values["p_sum"] += float(probability[mask].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--perturbation", choices=("dfl", "fixed", "width_matched", "shuffled_width", "object_average"), default="dfl"
    )
    parser.add_argument("--fixed-ratio", type=float, default=0.05)
    parser.add_argument("--width-pool", type=Path)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = YOLO(args.model).model.to(device).eval()
    head = model.model[-1]
    reg_max, nc = head.reg_max, head.nc
    assigner = TaskAlignedAssigner(topk=13, num_classes=nc, alpha=0.5, beta=6.0)
    project = torch.arange(reg_max, device=device, dtype=torch.float32)
    image_rows, object_rows, prediction_rows = [], [], []
    r2_counts = {"predictions": 0, "type1": 0, "type2": 0, "type3": 0, "mask": 0, "mask_gt_positive": 0}
    r2_identity = {
        "gt_positive_mask": 0,
        "same_object_only": 0,
        "different_object_only": 0,
        "mixed_same_and_different": 0,
        "positive_candidate_transitions": 0,
        "same_object_candidate_transitions": 0,
    }
    r2_bins = {f"{low:.1f}-{high:.1f}": [0, 0] for low, high in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.0))}
    empty_soft_group = lambda: {"n": 0, "loss_sum": 0.0, "p_sum": 0.0}
    r2_soft = {
        region: {gt: empty_soft_group() for gt in ("gt_positive", "gt_negative")}
        for region in ("all", "unstable")
    }
    r2_soft_bins = {
        f"{low:.1f}-{high:.1f}": {"n": 0, "gt_positive": 0, "loss_sum": 0.0, "p_sum": 0.0}
        for low, high in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.0))
    }
    all_relative_paths = args.image_list.read_text().splitlines()
    sample_count = min(args.max_images, len(all_relative_paths))
    # Deterministic coverage of the complete validation-list order without result-driven image selection.
    sample_indices = np.linspace(0, len(all_relative_paths) - 1, sample_count, dtype=int)
    paths = [args.dataset_root / all_relative_paths[index].strip() for index in sample_indices]
    started = time.perf_counter()
    shuffled_widths, width_cursor = None, 0
    if args.perturbation == "shuffled_width":
        if not args.width_pool:
            raise ValueError("shuffled_width requires --width-pool from a width_matched run")
        with args.width_pool.open(newline="", encoding="utf-8") as file:
            pool_rows = list(csv.DictReader(file))
            pool = torch.tensor(
                [[float(row[f"width_{edge}"]) for edge in ("left", "top", "right", "bottom")] for row in pool_rows]
            )
            pool_stride = torch.tensor([float(row["stride"]) for row in pool_rows])
        generator = torch.Generator().manual_seed(args.shuffle_seed)
        shuffled_widths = pool.clone()
        # Keep the FPN stride (and therefore the nominal pixel displacement distribution) fixed while
        # breaking object-specific width correspondence independently for each edge type.
        for stride_value in pool_stride.unique():
            indices = (pool_stride == stride_value).nonzero().flatten()
            for edge in range(4):
                shuffled_widths[indices, edge] = pool[indices[torch.randperm(len(indices), generator=generator)], edge]
    perturbation_abs_sum = perturbation_ratio_sum = perturbation_edge_count = 0.0

    with torch.inference_mode():
        for image_index, image_path in enumerate(paths):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            tensor, ratio, left, top = letterbox(image, args.imgsz)
            tensor = tensor.to(device)
            inference, features = model(tensor)
            raw_distribution, raw_scores = torch.cat(
                [feature.view(1, nc + reg_max * 4, -1) for feature in features], 2
            ).split((reg_max * 4, nc), 1)
            raw_distribution = raw_distribution.permute(0, 2, 1).contiguous()
            raw_scores = raw_scores.permute(0, 2, 1).contiguous()
            anchor_points, stride = make_anchors(features, head.stride, 0.5)
            distances = raw_distribution.view(1, -1, 4, reg_max).softmax(-1).matmul(project)
            dense_boxes = dist2bbox(distances, anchor_points, xywh=False) * stride
            nms, kept = non_max_suppression(inference, conf_thres=0.01, iou_thres=0.65, return_idxs=True)
            selected = nms[0]
            keep = kept[0]
            confidence_mask = selected[:, 4] >= args.conf
            selected, keep = selected[confidence_mask], keep[confidence_mask]

            relative = image_path.relative_to(args.dataset_root)
            label_path = args.dataset_root / "labels" / relative.relative_to("images").with_suffix(".txt")
            gt_classes, gt_boxes = load_gt(label_path, image.shape[:2], ratio, left, top)
            gt_classes, gt_boxes = gt_classes.to(device), gt_boxes.to(device)
            gt_assignment = run_assigner(
                assigner, raw_scores.sigmoid(), dense_boxes, anchor_points * stride, gt_classes, gt_boxes
            )

            if not len(selected):
                r2_counts["predictions"] += gt_assignment.numel()
                r2_counts["type2"] += int((gt_assignment >= 0).sum())
                update_r2_soft_stats(
                    r2_soft,
                    r2_soft_bins,
                    torch.ones_like(gt_assignment, dtype=torch.bool),
                    gt_assignment >= 0,
                    torch.zeros_like(gt_assignment, dtype=torch.float32),
                    raw_scores[0, :, 0],
                )
                image_rows.append(
                    {"image": str(relative), "pseudo_objects": 0, "gt_objects": len(gt_boxes), "matched_objects": 0}
                )
                continue
            pseudo_boxes, pseudo_classes = selected[:, :4], selected[:, 5].long()
            selected_logits = raw_distribution[0, keep].view(-1, 4, reg_max)
            edge_confidence, box_confidence = localization_confidence(selected_logits, reg_max)
            statistics = dfl_distribution_statistics(selected_logits)
            if args.perturbation == "dfl":
                candidates = dfl_supported_box_candidates(selected_logits, anchor_points[keep], stride[keep])
                candidates[0] = pseudo_boxes  # identical baseline across all ablations
            else:
                candidate_width = statistics["width"]
                if args.perturbation == "object_average":
                    candidate_width = candidate_width.mean(1, keepdim=True).expand(-1, 4)
                elif args.perturbation == "shuffled_width":
                    candidate_width = shuffled_widths[width_cursor : width_cursor + len(selected)].to(device)
                    width_cursor += len(selected)
                candidates = geometric_box_candidates(
                    pseudo_boxes,
                    "fixed" if args.perturbation == "fixed" else "width_matched",
                    dfl_width=candidate_width,
                    dfl_expectation=statistics["expectation"],
                    anchor_points=anchor_points[keep],
                    stride_tensor=stride[keep],
                    reg_max=reg_max,
                    fixed_ratio=args.fixed_ratio,
                )
            changed_edges = torch.stack(
                [torch.abs(candidates[1 + edge * 2 : 3 + edge * 2, :, edge] - pseudo_boxes[:, edge]) for edge in range(4)]
            )
            object_sizes = torch.stack(
                (pseudo_boxes[:, 2] - pseudo_boxes[:, 0], pseudo_boxes[:, 3] - pseudo_boxes[:, 1])
            )
            edge_sizes = torch.stack((object_sizes[0], object_sizes[1], object_sizes[0], object_sizes[1]))[:, None, :]
            perturbation_abs_sum += float(changed_edges.sum())
            perturbation_ratio_sum += float((changed_edges / edge_sizes.clamp_min(1e-9)).sum())
            perturbation_edge_count += changed_edges.numel()
            assignments = torch.stack(
                [
                    run_assigner(
                        assigner,
                        raw_scores.sigmoid(),
                        dense_boxes,
                        anchor_points * stride,
                        pseudo_classes,
                        candidate_boxes,
                    )
                    for candidate_boxes in candidates
                ]
            )
            stability, ambiguous_negative, identity_switch, foreground_frequency = assignment_stability(
                assignments.unsqueeze(1)
            )
            stability, ambiguous_negative = stability[0], ambiguous_negative[0]
            identity_switch, foreground_frequency = identity_switch[0], foreground_frequency[0]
            object_metrics = object_assignment_metrics(assignments, len(pseudo_boxes))
            mapping = match_pseudo_to_gt(pseudo_boxes, pseudo_classes, gt_boxes, gt_classes).to(device)
            baseline = assignments[0]
            # For a baseline foreground, stability means retaining the same object. For a baseline
            # background, it means remaining background; using zero same-object score for stable
            # negatives would spuriously label every stable negative as maximally unstable.
            state_stability = torch.where(baseline >= 0, stability, 1.0 - foreground_frequency)
            mapped_baseline = torch.full_like(baseline, -1)
            mapped_baseline[baseline >= 0] = mapping[baseline[baseline >= 0]]
            assignment_error = mapped_baseline != gt_assignment
            baseline_positive, gt_positive = baseline >= 0, gt_assignment >= 0
            type1 = baseline_positive & ~gt_positive
            type2 = ~baseline_positive & gt_positive
            type3 = baseline_positive & gt_positive & (mapped_baseline != gt_assignment)
            r2_mask = ambiguous_negative
            r2_gt_positive = r2_mask & gt_positive
            r2_instability = foreground_frequency  # baseline is negative on every member of M
            candidate_positive = assignments[1:] >= 0
            mapped_candidates = torch.full_like(assignments[1:], -1)
            mapped_candidates[candidate_positive] = mapping[assignments[1:][candidate_positive]]
            same_candidate_object = candidate_positive & (mapped_candidates == gt_assignment.unsqueeze(0))
            has_same = same_candidate_object.any(0)
            has_different = (candidate_positive & ~same_candidate_object).any(0)
            r2_identity["gt_positive_mask"] += int(r2_gt_positive.sum())
            r2_identity["same_object_only"] += int((r2_gt_positive & has_same & ~has_different).sum())
            r2_identity["different_object_only"] += int((r2_gt_positive & ~has_same & has_different).sum())
            r2_identity["mixed_same_and_different"] += int((r2_gt_positive & has_same & has_different).sum())
            r2_identity["positive_candidate_transitions"] += int(candidate_positive[:, r2_gt_positive].sum())
            r2_identity["same_object_candidate_transitions"] += int(same_candidate_object[:, r2_gt_positive].sum())
            update_r2_soft_stats(
                r2_soft, r2_soft_bins, ~baseline_positive, gt_positive, foreground_frequency, raw_scores[0, :, 0]
            )
            r2_counts["predictions"] += baseline.numel()
            r2_counts["type1"] += int(type1.sum())
            r2_counts["type2"] += int(type2.sum())
            r2_counts["type3"] += int(type3.sum())
            r2_counts["mask"] += int(r2_mask.sum())
            r2_counts["mask_gt_positive"] += int(r2_gt_positive.sum())
            for low, high in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.000001)):
                in_bin = r2_mask & (r2_instability >= low) & (r2_instability < high)
                values = r2_bins[f"{low:.1f}-{min(high, 1):.1f}"]
                values[0] += int(in_bin.sum())
                values[1] += int((in_bin & gt_positive).sum())
            # H2 concerns localization assignment, not missed/spurious object existence. Restrict its
            # primary comparison to one-to-one matched pseudo/GT identities. Unmatched counts remain
            # in objects.csv and images.csv but must not manufacture assignment errors here.
            matched_pseudo = mapping >= 0
            matched_gt = mapping[matched_pseudo]
            gt_is_matched = (gt_assignment[:, None] == matched_gt[None, :]).any(1) if len(matched_gt) else torch.zeros_like(gt_assignment, dtype=torch.bool)
            baseline_is_matched = torch.zeros_like(baseline, dtype=torch.bool)
            baseline_is_matched[baseline >= 0] = matched_pseudo[baseline[baseline >= 0]]
            # Keep the evaluation universe identical across perturbation ablations. Including
            # perturbation-only positives here would let larger perturbations change their own test set.
            relevant = baseline_is_matched | gt_is_matched

            for object_index in range(len(pseudo_boxes)):
                object_rows.append(
                    {
                        "image": str(relative),
                        "object": object_index,
                        "cls_conf": float(selected[object_index, 4]),
                        "dfl_conf": float(box_confidence[object_index]),
                        "mean_edge_conf": float(edge_confidence[object_index].mean()),
                        "mean_width": float(statistics["width"][object_index].mean()),
                        "max_width": float(statistics["width"][object_index].max()),
                        "width_left": float(statistics["width"][object_index, 0]),
                        "width_top": float(statistics["width"][object_index, 1]),
                        "width_right": float(statistics["width"][object_index, 2]),
                        "width_bottom": float(statistics["width"][object_index, 3]),
                        "stride": float(stride[keep][object_index]),
                        "mean_entropy": float(statistics["entropy"][object_index].mean()),
                        "mean_variance": float(statistics["variance"][object_index].mean()),
                        "jaccard": float(object_metrics["jaccard"][object_index]),
                        "instability": float(object_metrics["instability"][object_index]),
                        "ambiguous_ratio": float(object_metrics["ambiguous_ratio"][object_index]),
                        "identity_switches": float(object_metrics["identity_switches"][object_index]),
                        "matched_gt": int(mapping[object_index]),
                    }
                )
            for prediction_index in relevant.nonzero().flatten().tolist():
                prediction_rows.append(
                    {
                        "image": str(relative),
                        "prediction": prediction_index,
                        "stability": float(state_stability[prediction_index]),
                        "instability": float(1 - state_stability[prediction_index]),
                        "foreground_frequency": float(foreground_frequency[prediction_index]),
                        "baseline_object": int(baseline[prediction_index]),
                        "gt_object": int(gt_assignment[prediction_index]),
                        "assignment_error": int(assignment_error[prediction_index]),
                        "ambiguous_negative": int(ambiguous_negative[prediction_index]),
                        "identity_switch": int(identity_switch[prediction_index]),
                    }
                )
            image_rows.append(
                {
                    "image": str(relative),
                    "pseudo_objects": len(pseudo_boxes),
                    "gt_objects": len(gt_boxes),
                    "matched_objects": int(matched_pseudo.sum()),
                }
            )
            if (image_index + 1) % 10 == 0:
                print(f"processed {image_index + 1}/{len(paths)} images, pseudo objects={len(object_rows)}")

    write_csv(args.output / "images.csv", image_rows)
    write_csv(args.output / "objects.csv", object_rows)
    write_csv(args.output / "predictions.csv", prediction_rows)
    dfl_conf = np.array([row["dfl_conf"] for row in object_rows])
    width = np.array([row["mean_width"] for row in object_rows])
    instability = np.array([row["instability"] for row in object_rows])
    errors = np.array([row["assignment_error"] for row in prediction_rows], dtype=int)
    prediction_instability = np.array([row["instability"] for row in prediction_rows])
    rho_conf = spearmanr(dfl_conf, instability) if len(object_rows) >= 3 else (math.nan, math.nan)
    rho_width = spearmanr(width, instability) if len(object_rows) >= 3 else (math.nan, math.nan)
    auroc, auprc = binary_auc(errors, prediction_instability) if len(errors) else (math.nan, math.nan)
    error_by_bin = {}
    for low, high in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.000001)):
        mask = (prediction_instability >= low) & (prediction_instability < high)
        error_by_bin[f"{low:.1f}-{min(high, 1):.1f}"] = {
            "n": int(mask.sum()), "error_rate": float(errors[mask].mean()) if mask.any() else None
        }
    condition_a = bool(
        len(object_rows) >= 30
        and ((rho_conf.statistic <= -0.2 and rho_conf.pvalue < 0.01) or (rho_width.statistic >= 0.2 and rho_width.pvalue < 0.01))
    )
    condition_b = bool(len(errors) >= 100 and np.isfinite(auroc) and auroc >= 0.60)
    summary = {
        "model": args.model,
        "perturbation": args.perturbation,
        "fixed_ratio": args.fixed_ratio if args.perturbation == "fixed" else None,
        "shuffle_seed": args.shuffle_seed if args.perturbation == "shuffled_width" else None,
        "imgsz": args.imgsz,
        "images_requested": args.max_images,
        "images_processed": len(image_rows),
        "pseudo_objects": len(object_rows),
        "matched_pseudo_objects": sum(row["matched_objects"] for row in image_rows),
        "mean_object_jaccard": float(np.mean([row["jaccard"] for row in object_rows])) if object_rows else None,
        "mean_object_instability": float(np.mean(instability)) if len(instability) else None,
        "mean_absolute_perturbation_pixels": perturbation_abs_sum / perturbation_edge_count
        if perturbation_edge_count
        else None,
        "mean_perturbation_box_ratio": perturbation_ratio_sum / perturbation_edge_count
        if perturbation_edge_count
        else None,
        "mean_object_ambiguous_ratio": float(np.mean([row["ambiguous_ratio"] for row in object_rows])) if object_rows else None,
        "object_identity_switches": int(sum(row["identity_switches"] for row in object_rows)),
        "relevant_predictions": len(prediction_rows),
        "ambiguous_negative_predictions": int(sum(row["ambiguous_negative"] for row in prediction_rows)),
        "identity_switch_predictions": int(sum(row["identity_switch"] for row in prediction_rows)),
        "assignment_errors": int(errors.sum()) if len(errors) else 0,
        "spearman_dfl_conf_vs_instability": {"rho": float(rho_conf.statistic), "p": float(rho_conf.pvalue)},
        "spearman_width_vs_instability": {"rho": float(rho_width.statistic), "p": float(rho_width.pvalue)},
        "instability_error_auroc": auroc,
        "instability_error_auprc": auprc,
        "error_by_instability_bin": error_by_bin,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "gate_a": "GO" if condition_a or condition_b else "NO-GO",
        "elapsed_seconds": time.perf_counter() - started,
        "r2_precheck": {
            **r2_counts,
            "mismatches": r2_counts["type1"] + r2_counts["type2"] + r2_counts["type3"],
            "mask_precision": r2_counts["mask_gt_positive"] / r2_counts["mask"] if r2_counts["mask"] else None,
            "type2_recall": r2_counts["mask_gt_positive"] / r2_counts["type2"] if r2_counts["type2"] else None,
            "instability_bins": {
                key: {"n": value[0], "gt_positive": value[1], "gt_positive_rate": value[1] / value[0] if value[0] else None}
                for key, value in r2_bins.items()
            },
            "gt_positive_identity": {
                **r2_identity,
                "any_same_object": r2_identity["same_object_only"]
                + r2_identity["mixed_same_and_different"],
                "any_same_object_rate": (
                    (r2_identity["same_object_only"] + r2_identity["mixed_same_and_different"])
                    / r2_identity["gt_positive_mask"]
                    if r2_identity["gt_positive_mask"]
                    else None
                ),
                "same_object_transition_rate": (
                    r2_identity["same_object_candidate_transitions"]
                    / r2_identity["positive_candidate_transitions"]
                    if r2_identity["positive_candidate_transitions"]
                    else None
                ),
            },
        },
        "r2_soft_precheck": {
            region: {
                gt: {
                    **values,
                    "mean_loss": values["loss_sum"] / values["n"] if values["n"] else None,
                    "mean_p": values["p_sum"] / values["n"] if values["n"] else None,
                }
                for gt, values in groups.items()
            }
            for region, groups in r2_soft.items()
        }
        | {
            "instability_bins": {
                key: {
                    **values,
                    "gt_positive_rate": values["gt_positive"] / values["n"] if values["n"] else None,
                    "mean_loss": values["loss_sum"] / values["n"] if values["n"] else None,
                    "mean_p": values["p_sum"] / values["n"] if values["n"] else None,
                }
                for key, values in r2_soft_bins.items()
            }
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if len(object_rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(dfl_conf, instability, s=10, alpha=0.5)
        axes[0].set(xlabel="DFL box confidence", ylabel="Assignment instability")
        axes[1].scatter(width, instability, s=10, alpha=0.5)
        axes[1].set(xlabel="Mean Q90-Q10 width (bins)", ylabel="Assignment instability")
        fig.tight_layout()
        fig.savefig(args.output / "dfl_vs_instability.png", dpi=160)
        plt.close(fig)

    report = f"""# Gate A Report — CrowdHuman 5% / YOLOv8 / {args.imgsz}px / {args.perturbation}

This is checkpoint-only analysis: no optimizer, backward pass, EMA update, or training was performed.

## Scope

- Checkpoint: `{args.model}`
- Perturbation: `{args.perturbation}`
- Images: {summary['images_processed']} deterministically evenly spaced validation-list images (no result-driven selection)
- Pseudo-object confidence threshold: {args.conf}
- Pseudo objects: {len(object_rows)}
- Relevant dense predictions: {len(prediction_rows)}

## Step 2 — Deterministic assignment sensitivity

Each pseudo object was decoded into M=9 fixed candidates: expectation plus one-edge Q10/Q90 changes. The same
TaskAlignedAssigner (`topk=13`, `alpha=0.5`, `beta=6.0`) was rerun for every candidate. Object identity was retained;
identity switches were not merged with foreground/background flips.

- Mean object Jaccard: {summary['mean_object_jaccard']:.4f}
- Mean object instability: {summary['mean_object_instability']:.4f}
- Mean ambiguous-prediction ratio: {summary['mean_object_ambiguous_ratio']:.4f}
- Object-level identity-switch events: {summary['object_identity_switches']}
- Matched-domain ambiguous-negative predictions: {summary['ambiguous_negative_predictions']}
- Matched-domain identity-switch predictions: {summary['identity_switch_predictions']}

## Step 3 — DFL shape relation

- DFL confidence vs instability: Spearman rho={summary['spearman_dfl_conf_vs_instability']['rho']:.4f},
  p={summary['spearman_dfl_conf_vs_instability']['p']:.4g}
- Mean Q90-Q10 width vs instability: Spearman rho={summary['spearman_width_vs_instability']['rho']:.4f},
  p={summary['spearman_width_vs_instability']['p']:.4g}

Condition A was operationalized before inspection as |rho|≥0.20 in the expected direction, p<0.01, with at least
30 pseudo objects. Result: **{'PASS' if condition_a else 'FAIL'}**.

## Step 4 — GT assignment error relation

Pseudo objects were class-aware one-to-one matched to GT at IoU≥0.5. The GT boxes were passed through the same
assigner. Baseline pseudo identity (mapped to GT) and foreground/background were compared per relevant dense
prediction. Unmatched pseudo/GT objects were excluded from this primary H2 metric so object-existence errors were
not conflated with localization-assignment errors.

- Assignment errors: {summary['assignment_errors']} / {len(prediction_rows)}
- Matched pseudo objects: {summary['matched_pseudo_objects']} / {len(object_rows)}
- Instability detecting assignment error: AUROC={auroc:.4f}, AUPRC={auprc:.4f}
- Error rate by instability bin: `{json.dumps(error_by_bin)}`

Condition B was operationalized as AUROC≥0.60 with at least 100 relevant predictions. Result:
**{'PASS' if condition_b else 'FAIL'}**.

## Gate A decision

**{summary['gate_a']}**. Full training is authorized by this report only when Condition A or B passes. This report
does not interpret DFL bins as calibrated boundary probabilities or assignment stability as object confidence.
"""
    (args.output / "gate_a_report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
