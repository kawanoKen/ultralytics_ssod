"""Offline analysis of the DFL edge-confidence effect on CrowdHuman.

This script does not train. It reconstructs teacher pseudo-label inference and final
validation inference from saved checkpoints. The training implementation of
``use_edge_conf`` does not change the reliable/unreliable pseudo-label mask; it only
masks low-confidence edges inside the DFL loss for already reliable boxes. The output
therefore reports both the actual selection transition matrix and the edge-level
counterfactual gate that is really used by the loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.ssod_train import xyxy_to_xywh
from ultralytics.utils.dfl_confidence import localization_confidence
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors


CONF_LOW = 0.3
CONF_HIGH = 0.5
NMS_CONF = 0.01
NMS_IOU = 0.65
EDGE_THRESHOLD = 0.6
MATCH_IOU = 0.5
SIDES = ("L", "T", "R", "B")


def load_odgt(path: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            records[rec["ID"]] = [x for x in rec.get("gtboxes", []) if x.get("tag") == "person"]
    return records


def list_images(root: Path, list_path: Path) -> list[Path]:
    return [(list_path.parent / line.strip().lstrip("./")).resolve() for line in list_path.read_text().splitlines() if line.strip()]


def image_id(path: Path) -> str:
    return path.stem


def fbox_ltrb(box: dict[str, Any]) -> np.ndarray:
    x, y, w, h = np.asarray(box["fbox"], dtype=np.float32)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def vbox_ltrb(box: dict[str, Any]) -> np.ndarray | None:
    if "vbox" not in box:
        return None
    x, y, w, h = np.asarray(box["vbox"], dtype=np.float32)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def gt_features(box: dict[str, Any], image_w: int, image_h: int) -> dict[str, Any]:
    full = fbox_ltrb(box)
    visible = vbox_ltrb(box)
    if visible is None:
        occ = np.full(4, np.nan, dtype=np.float32)
        visible_ratio = np.nan
    else:
        fw = max(float(full[2] - full[0]), 1.0)
        fh = max(float(full[3] - full[1]), 1.0)
        occ = np.asarray(
            [
                (visible[0] - full[0]) / fw,
                (visible[1] - full[1]) / fh,
                (full[2] - visible[2]) / fw,
                (full[3] - visible[3]) / fh,
            ],
            dtype=np.float32,
        )
        occ = np.clip(occ, 0.0, 1.0)
        visible_ratio = float(np.clip((visible[2] - visible[0]) * (visible[3] - visible[1]), 0, None) /
                              max((full[2] - full[0]) * (full[3] - full[1]), 1.0))
    extra_occ = box.get("extra", {}).get("occ", np.nan)
    return {
        "full": full,
        "occ": occ,
        "visible_ratio": visible_ratio,
        "occ_flag": float(extra_occ) if extra_occ is not None else np.nan,
        "area_frac": float((full[2] - full[0]) * (full[3] - full[1]) / max(image_w * image_h, 1)),
        "width_frac": float((full[2] - full[0]) / max(image_w, 1)),
        "height_frac": float((full[3] - full[1]) / max(image_h, 1)),
    }


def side_severity(value: float) -> str:
    if not math.isfinite(value) or value <= 0.05:
        return "visible"
    if value <= 0.25:
        return "mild"
    return "strong"


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.clip(a + b - inter, 1e-9, None)


def prepare_image(path: Path, imgsz: int) -> tuple[torch.Tensor, float, float, float, tuple[int, int]]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False)
    image = letterbox(image=image)
    r = min(imgsz / h, imgsz / w)
    pad_w = (imgsz - w * r) / 2
    pad_h = (imgsz - h * r) / 2
    tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return tensor, r, pad_w, pad_h, (w, h)


def to_original(boxes: np.ndarray, r: float, pad_w: float, pad_h: float, w: int, h: int) -> np.ndarray:
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_w) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_h) / r
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def load_module(checkpoint: Path, device: torch.device, key: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module = ckpt.get(key) or ckpt.get("ema") or ckpt.get("model")
    if module is None:
        raise KeyError(f"No {key}/ema/model module in {checkpoint}")
    module = module.float().to(device).eval()
    if hasattr(module, "criterion"):
        module.criterion = None
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def raw_predictions(module: torch.nn.Module, images: torch.Tensor):
    output = module(images)
    if not isinstance(output, tuple) or len(output) < 2:
        raise RuntimeError("Checkpoint did not return decoded predictions and raw detection features")
    return output[0], output[1]


def edge_conf_from_features(pred_feats, keep: torch.Tensor, batch_index: int, reg_max: int, stride) -> tuple[np.ndarray, np.ndarray]:
    no = reg_max * 4 + 1
    pred_distri, _ = torch.cat(
        [x.view(pred_feats[0].shape[0], no, -1) for x in pred_feats], 2
    ).split((reg_max * 4, 1), 1)
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    selected = pred_distri[batch_index, keep].view(-1, 4, reg_max)
    edge, box = localization_confidence(selected, reg_max)
    return edge.cpu().numpy(), box.cpu().numpy()


class GroupStats:
    NUMERIC = (
        "cls_conf", "box_dfl_conf", "edge_L", "edge_T", "edge_R", "edge_B", "gt_iou",
        "gt_err_L", "gt_err_T", "gt_err_R", "gt_err_B", "area_frac", "width_frac", "height_frac",
        "occ_L", "occ_T", "occ_R", "occ_B", "visible_ratio", "occ_flag",
        "low_edge_L", "low_edge_T", "low_edge_R", "low_edge_B", "low_edge_any",
        "lowest_edge_is_most_occluded",
    )

    def __init__(self) -> None:
        self.n = 0
        self.values = defaultdict(list)

    def add(self, values: dict[str, float]) -> None:
        self.n += 1
        for key, value in values.items():
            value = float(value)
            if math.isfinite(value):
                self.values[key].append(value)

    def row(self, group: str) -> dict[str, Any]:
        row: dict[str, Any] = {"group": group, "n": self.n}
        for key in self.NUMERIC:
            vals = self.values.get(key, [])
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"{key}_median"] = float(np.median(vals)) if vals else np.nan
            row[f"{key}_n"] = len(vals)
        return row


def pseudo_status(conf: float) -> str:
    if conf >= CONF_HIGH:
        return "reliable"
    if conf >= CONF_LOW:
        return "uncertain"
    return "ignore_background"


def pseudo_group(status: str, edge_conf: np.ndarray) -> str:
    if status != "reliable":
        return status
    n_pass = int((edge_conf >= EDGE_THRESHOLD).sum())
    if n_pass == 4:
        return "reliable_edge_clean"
    if n_pass == 0:
        return "reliable_edge_all_low"
    return "reliable_edge_partial"


def pseudo_analysis(
    module: torch.nn.Module,
    image_paths: list[Path],
    annotations: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    imgsz: int,
    ratio: str,
) -> None:
    stats: dict[str, GroupStats] = defaultdict(GroupStats)
    transition = defaultdict(int)
    total_images = 0
    total_dets = 0
    total_reliable = 0
    total_edge_partial = 0
    total_edge_all_low = 0
    relation = {side: defaultdict(float) for side in SIDES}
    reg_max = int(module.model[-1].reg_max)
    stride = module.model[-1].stride
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start:start + batch_size]
        tensors, meta = [], []
        for path in paths:
            tensor, r, pw, ph, shape = prepare_image(path, imgsz)
            tensors.append(tensor)
            meta.append((path, r, pw, ph, shape))
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            pred, feats = raw_predictions(module, images)
            labels, keeps = non_max_suppression(pred, conf_thres=NMS_CONF, iou_thres=NMS_IOU, return_idxs=True)
        for bi, (path, r, pw, ph, (w, h)) in enumerate(meta):
            total_images += 1
            dets = labels[bi]
            if dets.numel() == 0:
                continue
            dets_np = dets.cpu().numpy()
            boxes = to_original(dets_np[:, :4], r, pw, ph, w, h)
            edge_conf, box_conf = edge_conf_from_features(feats, keeps[bi], bi, reg_max, stride)
            gts = [gt_features(x, w, h) for x in annotations.get(image_id(path), [])]
            gt_boxes = np.asarray([x["full"] for x in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
            for di, det in enumerate(dets_np):
                total_dets += 1
                conf = float(det[4])
                status = pseudo_status(conf)
                # The actual implementation leaves this box-level status unchanged when use_edge_conf=True.
                transition[(status, status)] += 1
                if status == "reliable":
                    total_reliable += 1
                    n_pass = int((edge_conf[di] >= EDGE_THRESHOLD).sum())
                    total_edge_partial += int(0 < n_pass < 4)
                    total_edge_all_low += int(n_pass == 0)
                best_gt = -1
                best_iou = 0.0
                if gt_boxes.shape[0]:
                    ious = box_iou_one(boxes[di], gt_boxes)
                    best_gt = int(np.argmax(ious))
                    best_iou = float(ious[best_gt])
                vals: dict[str, float] = {
                    "cls_conf": conf,
                    "box_dfl_conf": float(box_conf[di]),
                    "edge_L": float(edge_conf[di, 0]), "edge_T": float(edge_conf[di, 1]),
                    "edge_R": float(edge_conf[di, 2]), "edge_B": float(edge_conf[di, 3]),
                    "gt_iou": best_iou,
                    "low_edge_L": float(edge_conf[di, 0] < EDGE_THRESHOLD),
                    "low_edge_T": float(edge_conf[di, 1] < EDGE_THRESHOLD),
                    "low_edge_R": float(edge_conf[di, 2] < EDGE_THRESHOLD),
                    "low_edge_B": float(edge_conf[di, 3] < EDGE_THRESHOLD),
                    "low_edge_any": float((edge_conf[di] < EDGE_THRESHOLD).any()),
                }
                if best_gt >= 0:
                    gt = gts[best_gt]
                    occ = gt["occ"]
                    if np.isfinite(occ).all():
                        vals["lowest_edge_is_most_occluded"] = float(np.argmin(edge_conf[di]) == np.argmax(occ))
                        for si, side in enumerate(SIDES):
                            x = float(occ[si]); y = float(edge_conf[di, si])
                            relation[side]["n"] += 1
                            relation[side]["sx"] += x; relation[side]["sy"] += y
                            relation[side]["sxx"] += x * x; relation[side]["syy"] += y * y; relation[side]["sxy"] += x * y
                    vals.update({
                        "gt_err_L": abs(float(boxes[di, 0] - gt["full"][0])),
                        "gt_err_T": abs(float(boxes[di, 1] - gt["full"][1])),
                        "gt_err_R": abs(float(boxes[di, 2] - gt["full"][2])),
                        "gt_err_B": abs(float(boxes[di, 3] - gt["full"][3])),
                        "area_frac": gt["area_frac"], "width_frac": gt["width_frac"],
                        "height_frac": gt["height_frac"], "occ_L": gt["occ"][0],
                        "occ_T": gt["occ"][1], "occ_R": gt["occ"][2], "occ_B": gt["occ"][3],
                        "visible_ratio": gt["visible_ratio"], "occ_flag": gt["occ_flag"],
                    })
                stats[pseudo_group(status, edge_conf[di])].add(vals)
        if (start // batch_size) % 20 == 0:
            print(f"[{ratio}] pseudo {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
    rows = [stats[key].row(key) for key in sorted(stats)]
    with (out_dir / f"{ratio}_pseudo_group_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["group", "n"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{ratio}_selection_transition.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["confidence_only_status", "edge_status_actual", "count"])
        writer.writeheader()
        for (a, b), count in sorted(transition.items()):
            writer.writerow({"confidence_only_status": a, "edge_status_actual": b, "count": count})
    relation_rows = []
    for side in SIDES:
        a = relation[side]; n = a["n"]
        cov = a["sxy"] / n - (a["sx"] / n) * (a["sy"] / n) if n else np.nan
        vx = a["sxx"] / n - (a["sx"] / n) ** 2 if n else np.nan
        vy = a["syy"] / n - (a["sy"] / n) ** 2 if n else np.nan
        relation_rows.append({"side": side, "n": int(n), "mean_occlusion": a["sx"] / n if n else np.nan,
                              "mean_edge_conf": a["sy"] / n if n else np.nan,
                              "pearson_occlusion_vs_edge_conf": cov / math.sqrt(max(vx * vy, 1e-12)) if n > 1 else np.nan})
    with (out_dir / f"{ratio}_pseudo_edge_occlusion_relation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(relation_rows[0])); writer.writeheader(); writer.writerows(relation_rows)
    with (out_dir / f"{ratio}_pseudo_meta.json").open("w") as f:
        json.dump({
            "images": total_images, "nms_survivor_boxes": total_dets, "reliable_boxes": total_reliable,
            "reliable_edge_partial_boxes": total_edge_partial, "reliable_edge_all_low_boxes": total_edge_all_low,
            "conf_low": CONF_LOW, "conf_high": CONF_HIGH, "nms_conf": NMS_CONF, "nms_iou": NMS_IOU,
            "edge_threshold": EDGE_THRESHOLD,
            "note": "Actual use_edge_conf does not change reliable/unreliable selection; it only masks DFL edges.",
        }, f, indent=2)


def greedy_match(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> dict[int, int]:
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return {}
    iou = np.stack([box_iou_one(box, gt_boxes) for box in pred_boxes])
    pairs = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1]) if iou[i, j] >= MATCH_IOU]
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), {}
    for _, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); matches[gi] = pi
    return matches


def final_prediction_analysis(
    models: dict[str, torch.nn.Module],
    image_paths: list[Path],
    annotations: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    imgsz: int,
    ratio: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, module in models.items():
        print(f"[{ratio}] final model {model_name}", flush=True)
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start:start + batch_size]
            tensors, meta = [], []
            for path in paths:
                tensor, r, pw, ph, shape = prepare_image(path, imgsz)
                tensors.append(tensor); meta.append((path, r, pw, ph, shape))
            images = torch.stack(tensors).to(device)
            with torch.inference_mode():
                pred, _ = raw_predictions(module, images)
                labels = non_max_suppression(pred, conf_thres=NMS_CONF, iou_thres=NMS_IOU)
            for bi, (path, r, pw, ph, (w, h)) in enumerate(meta):
                dets = labels[bi]
                det_np = dets.cpu().numpy() if dets.numel() else np.zeros((0, 6), dtype=np.float32)
                pred_boxes = to_original(det_np[:, :4], r, pw, ph, w, h)
                gts = [gt_features(x, w, h) for x in annotations.get(image_id(path), [])]
                gt_boxes = np.asarray([x["full"] for x in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
                matches = greedy_match(pred_boxes, gt_boxes)
                for gi, gt in enumerate(gts):
                    pred_idx = matches.get(gi)
                    pred_box = pred_boxes[pred_idx] if pred_idx is not None else None
                    errors = [abs(float(pred_box[k] - gt["full"][k])) for k in range(4)] if pred_box is not None else [np.nan] * 4
                    for si, side in enumerate(SIDES):
                        rows.append({
                            "ratio": ratio, "model": model_name, "image_id": image_id(path), "gt_index": gi,
                            "side": side, "severity": side_severity(float(gt["occ"][si])),
                            "occlusion": float(gt["occ"][si]), "visible_ratio": gt["visible_ratio"],
                            "occ_flag": gt["occ_flag"], "matched": int(pred_idx is not None),
                            "edge_error_px": errors[si], "edge_error_norm_full_side": errors[si] / max(float(gt["full"][2] - gt["full"][0]) if si in (0, 2) else float(gt["full"][3] - gt["full"][1]), 1.0),
                            "gt_width": float(gt["full"][2] - gt["full"][0]),
                            "gt_height": float(gt["full"][3] - gt["full"][1]),
                        })
            if (start // batch_size) % 20 == 0:
                print(f"[{ratio}] {model_name} final {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fieldnames = list(rows[0]) if rows else ["ratio", "model"]
    with (out_dir / f"{ratio}_final_edge_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    summary = []
    for (model, severity, side), group in _group_rows(rows, "model", "severity", "side"):
        errors = [float(x["edge_error_px"]) for x in group if math.isfinite(float(x["edge_error_px"]))]
        summary.append({
            "ratio": ratio, "model": model, "severity": severity, "side": side,
            "gt_edges": len(group), "matched_edges": len(errors),
            "match_rate": len(errors) / max(len(group), 1),
            "mean_error_px": float(np.mean(errors)) if errors else np.nan,
            "median_error_px": float(np.median(errors)) if errors else np.nan,
            "mean_error_norm": float(np.mean([float(x["edge_error_norm_full_side"]) for x in group if math.isfinite(float(x["edge_error_norm_full_side"]))])) if errors else np.nan,
        })
    with (out_dir / f"{ratio}_final_edge_error_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["ratio", "model"]); writer.writeheader(); writer.writerows(summary)
    baseline_name = "confidence_only"
    pair_rows = []
    base = {(x["image_id"], int(x["gt_index"]), x["side"]): x for x in rows if x["model"] == baseline_name}
    for dfl_name in sorted({x["model"] for x in rows if x["model"] != baseline_name}):
        dfl = {(x["image_id"], int(x["gt_index"]), x["side"]): x for x in rows if x["model"] == dfl_name}
        for key, bx in base.items():
            dx = dfl.get(key)
            if dx is None: continue
            bv, dv = float(bx["edge_error_px"]), float(dx["edge_error_px"])
            if math.isfinite(bv) and math.isfinite(dv):
                pair_rows.append({"ratio": ratio, "dfl_model": dfl_name, "image_id": key[0], "gt_index": key[1], "side": key[2], "severity": bx["severity"], "baseline_error_px": bv, "dfl_error_px": dv, "delta_dfl_minus_baseline_px": dv - bv})
    if pair_rows:
        with (out_dir / f"{ratio}_final_paired_deltas.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
        pair_summary = []
        for (dfl_model, severity, side), group in _group_rows(pair_rows, "dfl_model", "severity", "side"):
            deltas = [float(x["delta_dfl_minus_baseline_px"]) for x in group]
            pair_summary.append({"ratio": ratio, "dfl_model": dfl_model, "severity": severity, "side": side, "n": len(deltas), "mean_delta_px": float(np.mean(deltas)), "median_delta_px": float(np.median(deltas)), "improved_rate": float(np.mean(np.asarray(deltas) < 0))})
        with (out_dir / f"{ratio}_final_paired_delta_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_summary[0])); writer.writeheader(); writer.writerows(pair_summary)


def _group_rows(rows: list[dict[str, Any]], *keys: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped.items()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/work/kawano/LA/datasets/crowdhuman"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dfl_edge_effect_analysis"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--ratio", choices=["1p", "10p"], required=True)
    parser.add_argument("--max-unlabeled", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--pseudo-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratio = args.ratio
    run_root = Path(f"runs/crowdhuman_ssod_{ratio}_zero_pseudo_ablation")
    base_dir = run_root / f"yolov8n_{ratio}_loss_balance"
    dfl_dirs = {
        "dfl_seed0": run_root / f"yolov8n_{ratio}_loss_balance_edge_conf",
        "dfl_seed1": run_root / f"yolov8n_{ratio}_loss_balance_edge_conf_seed1_valid",
        "dfl_seed2": run_root / f"yolov8n_{ratio}_loss_balance_edge_conf_seed2_valid",
    }
    val_list = args.root / "val.txt"
    unlabeled_list = args.root / f"train_unlabeled_{'90p' if ratio == '10p' else '1p'}.txt"
    odgt = load_odgt(args.root / "annotation_train.odgt")
    val_odgt = load_odgt(args.root / "annotation_val.odgt")
    unlabeled_images = list_images(args.root, unlabeled_list)
    val_images = list_images(args.root, val_list)
    if args.max_unlabeled:
        unlabeled_images = unlabeled_images[:args.max_unlabeled]
    if args.max_val:
        val_images = val_images[:args.max_val]
    print(f"{ratio}: unlabeled={len(unlabeled_images)} val={len(val_images)} device={device}", flush=True)

    # Same baseline teacher prediction is used for the actual selection transition analysis.
    baseline_ckpt = base_dir / "weights" / "best.pt"
    pseudo_teacher = load_module(baseline_ckpt, device, "teacher")
    pseudo_analysis(pseudo_teacher, unlabeled_images, odgt, args.out_dir, device, args.batch_size, args.imgsz, ratio)
    del pseudo_teacher
    if device.type == "cuda": torch.cuda.empty_cache()

    if not args.pseudo_only:
        final_models = {"confidence_only": load_module(baseline_ckpt, device, "ema")}
        for name, d in dfl_dirs.items():
            final_models[name] = load_module(d / "weights" / "best.pt", device, "ema")
        final_prediction_analysis(final_models, val_images, val_odgt, args.out_dir, device, args.batch_size, args.imgsz, ratio)
    print(f"{ratio}: analysis finished; outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
