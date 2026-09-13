"""Training support for the CrowdHuman Experiment 1 oracle selection study.

The normal SSOD trainer is intentionally not used for the unlabeled branch here:
it refreshes its teacher and regenerates pseudo labels on every batch.  This module
loads a frozen, pre-selected candidate file and trains only on its selected boxes.
"""

from __future__ import annotations

import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset_ssod import YOLODataset_ssod
from ultralytics.models.yolo.detect.ssod_train import SSODTrainer
from ultralytics.utils import LOGGER, RANK, SSOD_DEFAULT_CFG_PATH, TQDM, colorstr
from ultralytics.utils.loss_ssod import EfficientTeacherLoss
from ultralytics.utils.torch_utils import autocast, torch_distributed_zero_first, unwrap_model


class SelectedPseudoDataset(YOLODataset_ssod):
    """SSOD image dataset whose labels come only from an offline selection artifact.

    It deliberately never reads the CrowdHuman YOLO label files.  This keeps the GT
    used for the oracle matching/descriptor calculation out of teacher/student data
    loading; training sees only the selected candidate boxes.
    """

    def __init__(self, *args, selection_path: str | Path, **kwargs):
        selection = json.loads(Path(selection_path).read_text())
        self._selected_by_image = {}
        for item in selection["selected"]:
            self._selected_by_image.setdefault(item["image_id"], []).append(item)
        super().__init__(*args, **kwargs)

    def get_labels(self) -> list[dict]:
        labels = []
        for im_file in self.im_files:
            image_id = Path(im_file).stem
            selected = self._selected_by_image.get(image_id, [])
            if selected:
                bboxes = np.asarray([x["box_xywh_norm"] for x in selected], dtype=np.float32).reshape(-1, 4)
                cls = np.asarray([[int(x.get("cls", 0))] for x in selected], dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0, 1), dtype=np.float32)
            labels.append(
                {
                    "im_file": im_file,
                    # The selected boxes are already normalized.  Training uses
                    # rect=False, so BaseDataset only needs a placeholder here;
                    # avoiding an image read for every rank keeps DDP startup fast.
                    "shape": (0, 0),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels


class Experiment1Trainer(SSODTrainer):
    """Fixed-candidate, selected-positive-only trainer for Experiment 1."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg or SSOD_DEFAULT_CFG_PATH, overrides, _callbacks)
        selection_path = self.data.get("e1_selection")
        if not selection_path:
            raise ValueError("Experiment 1 data YAML must contain e1_selection")
        self.e1_selection_path = Path(selection_path)
        self.e1_method = str(self.data.get("e1_method", "unknown"))
        self.e1_seed = int(self.data.get("e1_seed", self.args.seed))
        self.e1_max_updates = int(self.data.get("e1_max_updates", 200))
        self.e1_val_interval = int(self.data.get("e1_val_interval", 0))
        self.e1_curve_path = self.save_dir / "learning_curve.csv"
        self.e1_train_log_path = self.save_dir / "train_stats.jsonl"

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None, ssod: bool = False):
        if not ssod:
            return build_yolo_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=mode == "val",
                stride=max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32),
            )

        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return SelectedPseudoDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            selection_path=self.e1_selection_path,
        )

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train", ssod: bool = False
    ):
        """Build a loader whose stochastic order/augmentation depends on the experiment seed."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size, ssod)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        drop_last = self.args.compile and mode == "train"
        if ssod and mode == "train":
            drop_last = True
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=drop_last,
            seed=self.e1_seed,
        )

    @staticmethod
    def _metric(metrics: dict, needle: str) -> float:
        for key, value in (metrics or {}).items():
            if needle in str(key):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("nan")
        return float("nan")

    def _write_curve(self, update: int, metrics: dict | None, fitness: float | None) -> None:
        if RANK not in {-1, 0}:
            return
        row = {
            "update": update,
            "mAP50-95": self._metric(metrics, "mAP50-95"),
            "mAP50": self._metric(metrics, "mAP50(B)"),
            "precision": self._metric(metrics, "precision(B)"),
            "recall": self._metric(metrics, "recall(B)"),
            "fitness": float(fitness) if fitness is not None else float("nan"),
        }
        exists = self.e1_curve_path.exists()
        with self.e1_curve_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_train_log(self, record: dict) -> None:
        if RANK in {-1, 0}:
            with self.e1_train_log_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def _validate_at(self, update: int) -> None:
        metrics, fitness = self.validate()
        if RANK in {-1, 0}:
            self._write_curve(update, metrics, fitness)
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            self.save_model()
        self._clear_memory(threshold=0.5)

    def _do_train(self):
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb_labeled = len(self.train_loader)
        nb_pseudo = len(self.ssod_train_loader)
        if nb_pseudo < 1:
            raise RuntimeError("Experiment 1 pseudo dataset produced no batches")
        warmup_iterations = self._warmup_iterations(self.args.warmup_epochs, 0, nb_labeled, nb_pseudo)
        self.loss_func_ssod = EfficientTeacherLoss(
            unwrap_model(self.model),
            conf_threshold_high=0.5,
            conf_threshold_low=0.0,
            use_loc_conf=False,
            assignment_stability_method="off",
            cls_loss_denom="target_score_sum",
            positive_only_cls_loss=True,
        )

        if RANK in {-1, 0}:
            LOGGER.info(
                f"Experiment 1: method={self.e1_method}, seed={self.e1_seed}, "
                f"fixed candidates={self.e1_selection_path}, max_updates={self.e1_max_updates}, "
                f"positive-only unlabeled loss, no teacher EMA"
            )
            self.e1_curve_path.unlink(missing_ok=True)
            self.e1_train_log_path.unlink(missing_ok=True)

        self.optimizer.zero_grad()
        update = 0
        last_opt_step = -1
        self.train_time_start = time.time()
        self.epoch_time_start = self.train_time_start
        self.run_callbacks("on_train_start")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                self.ssod_train_loader.sampler.set_epoch(epoch)
            labeled_iter = iter(self.train_loader)
            pseudo_iter = iter(self.ssod_train_loader)
            pbar = range(nb_pseudo)
            if RANK in {-1, 0}:
                pbar = TQDM(pbar, total=min(nb_pseudo, self.e1_max_updates - update))

            for batch_idx in pbar:
                if update >= self.e1_max_updates:
                    break
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.train_loader)
                    labeled_batch = next(labeled_iter)
                try:
                    pseudo_batch = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(self.ssod_train_loader)
                    pseudo_batch = next(pseudo_iter)

                ni = update
                if ni <= warmup_iterations:
                    xi = [0, warmup_iterations]
                    self.accumulate = max(
                        1,
                        int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size_ssod]).round()),
                    )
                    for j, group in enumerate(self.optimizer.param_groups):
                        group["lr"] = np.interp(
                            ni,
                            xi,
                            [self.args.warmup_bias_lr if j == 0 else 0.0, group["initial_lr"] * self.lf(epoch)],
                        )
                        if "momentum" in group:
                            group["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )

                with autocast(self.amp):
                    labeled_batch = self.preprocess_batch(labeled_batch)
                    preds_labeled = self.model(labeled_batch["img"])
                    loss_labeled, self.loss_items = unwrap_model(self.model).loss(labeled_batch, preds_labeled)

                    pseudo_batch = self.preprocess_batch(pseudo_batch)
                    pseudo_img = pseudo_batch.get("img_strong", pseudo_batch["img"])
                    preds_pseudo = unwrap_model(self.model)(pseudo_img)
                    pseudo_boxes = pseudo_batch["bboxes"]
                    pseudo_cls = pseudo_batch["cls"]
                    pseudo_batch_idx = pseudo_batch["batch_idx"].view(-1, 1).long()
                    pseudo_conf = torch.ones(
                        (pseudo_boxes.shape[0], 1), device=pseudo_boxes.device, dtype=pseudo_boxes.dtype
                    )
                    loss_pseudo, self.loss_items_unlabeled, reliable, _ = self.loss_func_ssod(
                        preds_pseudo,
                        pseudo_boxes,
                        pseudo_cls,
                        pseudo_conf,
                        pseudo_batch_idx,
                    )
                    self.loss = loss_labeled.sum() + self.ssod_weight * loss_pseudo.sum()
                    if RANK != -1:
                        self.loss *= self.world_size

                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    update += 1

                if RANK in {-1, 0}:
                    sup_items = [float(x) for x in self.loss_items.detach().cpu()]
                    unsup_items = [float(x) for x in self.loss_items_unlabeled.detach().cpu()]
                    record = {
                        "update": update,
                        "epoch": epoch,
                        "method": self.e1_method,
                        "seed": self.e1_seed,
                        "selected_boxes": int(pseudo_boxes.shape[0]),
                        "positive_assigned": int(reliable.sum()),
                        "sup_loss": float(loss_labeled.sum().detach().cpu()),
                        "unsup_loss": float(loss_pseudo.sum().detach().cpu()),
                        "sup_items": sup_items,
                        "unsup_items": unsup_items,
                        "total_loss": float(self.loss.detach().cpu()),
                    }
                    self._write_train_log(record)
                    if update == 1 or update % 20 == 0:
                        pbar.set_description(
                            f"{self.e1_method} u={update}/{self.e1_max_updates} "
                            f"loss={record['total_loss']:.3g} pl={record['selected_boxes']} "
                            f"pos={record['positive_assigned']}"
                        )

                if self.e1_val_interval and update % self.e1_val_interval == 0:
                    self._validate_at(update)

            if update >= self.e1_max_updates:
                break

        if self.args.val:
            self._validate_at(update)
        else:
            if RANK in {-1, 0}:
                self.best_fitness = self.fitness = 0.0
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                self.save_model()
                self._write_curve(update, {}, self.fitness)

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"Experiment 1 {self.e1_method}/seed{self.e1_seed} finished: {update} updates in {seconds:.1f}s")
            self.run_callbacks("on_train_end")
        self._clear_memory()
