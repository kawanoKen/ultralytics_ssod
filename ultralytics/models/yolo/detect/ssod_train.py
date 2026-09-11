# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import gc
import json
import math
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
import torch.nn.functional as F

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, check_det_dataset_ssod
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.loss_ssod import EfficientTeacherLoss, DomainAdversarialNet
from ultralytics.utils.dfl_confidence import localization_confidence
from ultralytics.utils.ssod_diagnostics import (
    BNMismatchProbe,
    SSODDiagnosticsLogger,
    compute_supervised_assignment_stats,
    dfl_entropy,
)
from ultralytics.utils.assignment_stability import (
    dfl_distribution_statistics,
    dfl_supported_box_candidates,
    geometric_box_candidates,
)
from ultralytics.utils.loss_spike_lab import (
    SpikeCaptured,
    apply_intervention,
    compute_split_gradients,
    log_iteration_stats,
    save_snapshot,
    snapshot_trainer_state,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.nms import non_max_suppression

import random
from typing import Any

from ultralytics.data import build_dataloader, build_yolo_dataset, build_yolo_dataset_ssod
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import SSOD_DEFAULT_CFG_PATH, LOGGER
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)
from ultralytics.utils.tal import make_anchors


class SSODTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=SSOD_DEFAULT_CFG_PATH, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a DetectionTrainer object for training YOLO object detection model training.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.batch_size_ssod = self.args.batch_ssod
        self.burn_in_epochs = self.args.burn_in_epochs
        self.conf_threshold_high = self.args.conf_threshold_high
        self.conf_threshold_low = self.args.conf_threshold_low
        self.use_loc_conf = self.args.use_loc_conf
        self.loc_conf_threshold = self.args.loc_conf_threshold
        self.use_edge_conf = self.args.use_edge_conf
        self.edge_conf_threshold = self.args.edge_conf_threshold
        self.spike_diag_enabled = self.args.spike_diag_enabled
        self.spike_diag_threshold = self.args.spike_diag_threshold
        self.spike_diag_stop_after_capture = self.args.spike_diag_stop_after_capture
        self.spike_diag_intervention = self.args.spike_diag_intervention
        self.spike_diag_intervention_iter = self.args.spike_diag_intervention_iter
        self.spike_diag_g_ref = self.args.spike_diag_g_ref
        self.spike_diag_log_path = self.args.spike_diag_log_path or str(self.save_dir / "spike_diag.jsonl")
        self.spike_diag_snapshot_path = self.args.spike_diag_snapshot_path or str(self.save_dir / "spike_snapshot.pt")
        self._spike_captured = False
        self.assignment_stability_method = self.args.assignment_stability_method
        self.assignment_perturbation = self.args.assignment_perturbation
        if self.assignment_perturbation not in {"dfl", "fixed", "width_matched"}:
            raise ValueError(f"assignment_perturbation must be dfl, fixed, or width_matched, got {self.assignment_perturbation!r}")
        self.ssod_weight = self.args.ssod_weight
        self.domain_adaptation = self.args.domain_adaptation
        self.skip_zero_pseudo_cls_loss = self.args.skip_zero_pseudo_cls_loss
        self.cls_loss_denom = self.args.cls_loss_denom
        self.loss_balancing_mode = self.args.loss_balancing_mode
        self.loss_balance_beta = self.args.loss_balance_beta
        self.ema_loss_sup = None
        self.ema_loss_unsup = None
        self.supervised_only_control = self.args.supervised_only_control



    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None, ssod: bool = False):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if ssod:
            return build_yolo_dataset_ssod(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train", ssod: bool = False):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size, ssod)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        # Drop the final undersized batch for the *unlabeled* SSOD loader: with drop_last=False
        # (the normal default), each epoch's leftover remainder batch can be as small as a couple
        # of images. In training mode BatchNorm uses per-batch statistics, and for such a tiny,
        # possibly heavily-crowded batch those statistics can swing wildly, destabilizing logits
        # across the whole anchor grid and producing large, spurious ssod/cls_loss spikes -- see
        # scripts/loss_spike_lab/. The labeled loader is left untouched since it never showed this.
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
            )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if "img_strong" in batch:
            batch["img_strong"] = batch["img_strong"].float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                if "img_strong" in batch:
                    batch["img_strong"] = nn.functional.interpolate(
                        batch["img_strong"], size=ns, mode="bilinear", align_corners=False
                    )
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        """Save training checkpoints, temporarily detaching the BN mismatch probe's forward hooks
        first. BaseTrainer.save_model() deepcopies self.teacher.ema and torch.save()s (pickles)
        the result; the probe's hooks are nested closures with no module-level import path, so
        pickling them raises AttributeError. The probe is a live, training-time-only instrument
        (see ultralytics.utils.ssod_diagnostics.BNMismatchProbe) that has no business being part
        of a saved checkpoint anyway, so detach-save-reattach is the correct fix, not a workaround.
        """
        probe = getattr(self, "bn_mismatch_probe", None)
        if probe is not None:
            probe.remove()
        try:
            super().save_model()
        finally:
            if probe is not None:
                self.bn_mismatch_probe = BNMismatchProbe(unwrap_model(self.teacher.ema))

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self, include_ssod: bool = False, include_da: bool = False):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        names_to_show = list(self.loss_names)
        if include_ssod:
            names_to_show += [" ssod_bloss ", " ssod_closs ", " ssod_dloss "]
        if include_da:
            names_to_show += [" da/loss_s ", " da/loss_t ", " da/loss "]
        return ("\n" + "%11s" * (4 + len(names_to_show))) % (
            "Epoch",
            "GPU_mem",
            *names_to_show,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_pseudo_samples(self, batch: dict[str, Any], epoch, ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        
        # 出力先ディレクトリ runs/*/epoch{epoch} を作成
        out_dir = self.save_dir / f"epoch{epoch}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=out_dir / f"pseudo_batch{ni}.jpg",
            on_plot=self.on_plot,
            threaded=False,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def plot_metrics(self):
        """Plot metrics using SSOD-safe plotting to handle ragged CSV lines."""
        from ultralytics.utils.plotting_ssod import plot_results as plot_results_ssod
        plot_results_ssod(file=self.csv, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
        # compile 有効時は multi_scale を無効化してグラフ膨張とメモリ増大を回避
        if self.args.compile and getattr(self.args, "multi_scale", False):
            LOGGER.warning("Disabling multi_scale under torch.compile to prevent graph bloat and memory growth.")
            self.args.multi_scale = False

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        batch_size_ssod = self.batch_size_ssod // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train", ssod=False
        )
        self.ssod_train_loader = self.get_dataloader(
            self.data["ssod_train"], batch_size=batch_size_ssod, rank=LOCAL_RANK, mode="train", ssod=True
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.burn_in_epochs
            + math.ceil(len(self.ssod_train_loader.dataset) / max(self.batch_size_ssod, self.args.nbs))
            * (self.epochs - self.burn_in_epochs)
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self._resume_teacher_state = ckpt.get("teacher") if ckpt and self.resume else None
        self._resume_teacher_updates = ckpt.get("teacher_updates") if ckpt and self.resume else None
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    @staticmethod
    def _warmup_iterations(warmup_epochs, burn_in_epochs, nb, nb_ssod):
        """Convert epoch-based warmup to iterations across the supervised/SSOD phase boundary."""
        if warmup_epochs <= 0:
            return -1
        supervised_epochs = min(warmup_epochs, burn_in_epochs)
        ssod_epochs = max(warmup_epochs - burn_in_epochs, 0.0)
        return max(round(supervised_epochs * nb + ssod_epochs * nb_ssod), 100)

    def _restore_teacher_after_resume(self):
        """Restore the pseudo-label EMA exactly when resuming after burn-in."""
        self.teacher = ModelEMA(self.model)
        if RANK in {-1, 0}:
            self.bn_mismatch_probe = BNMismatchProbe(unwrap_model(self.teacher.ema))
        if self._resume_teacher_state is not None:
            self.teacher.ema.load_state_dict(self._resume_teacher_state.float().state_dict())
            self.teacher.updates = self._resume_teacher_updates or 0
            LOGGER.info("Restored SSOD teacher EMA from the resume checkpoint")
            return

        raise RuntimeError(
            "This legacy checkpoint has no serialized SSOD teacher and cannot be resumed exactly after burn-in. "
            "Restart from a pre-burn-in checkpoint or a checkpoint written by the fixed trainer."
        )

    def _close_dataloader_mosaic(self):
        """Disable mix augmentations for both labeled and unlabeled training datasets."""
        for loader_name in ("train_loader", "ssod_train_loader"):
            loader = getattr(self, loader_name, None)
            dataset = getattr(loader, "dataset", None)
            if dataset is None:
                continue
            if hasattr(dataset, "mosaic"):
                dataset.mosaic = False
            if hasattr(dataset, "close_mosaic"):
                LOGGER.info(f"Closing mosaic for {loader_name}")
                dataset.close_mosaic(hyp=copy(self.args))

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] == "ndjson":
                # Convert NDJSON to YOLO format
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo

                yaml_path = asyncio.run(convert_ndjson_to_yolo(self.args.data))
                self.args.data = str(yaml_path)
                data = check_det_dataset_ssod(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset_ssod(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def _do_train_supervised_only_control(self):
        """
        Compute/update-matched supervised-only control run.

        Reuses exactly the same setup as a real SSOD run (_setup_train: same starting checkpoint,
        same labeled dataset, same optimizer/hyperparameters, same LR schedule construction) and
        the same per-step iteration/warmup/scheduler arithmetic as the real pseudo-label phase
        (burn_in_epochs=0, ni = i + nb_ssod*epoch, identical warmup formula), so that a run
        launched with the same --epochs/--batch/--batch_ssod/--device as an SSOD run performs
        the exact same number of optimizer updates on the exact same step-indexed LR schedule.

        The only difference: unlabeled images are never loaded or forwarded through the model
        (no BN update from them, no pseudo-label loss of any kind) -- `self.ssod_train_loader` is
        built (by the shared _setup_train()) only so `len(self.ssod_train_loader)` gives the same
        per-epoch step count as the real run; it is never iterated. The 150 labeled images are
        cycled repeatedly via the same try/except StopIteration pattern the real run's labeled
        side already uses, so batch composition/reshuffling behavior matches exactly too.
        """
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb_ssod = len(self.ssod_train_loader)  # iteration-count reference only; never iterated
        nb = len(self.train_loader)
        nw = self._warmup_iterations(self.args.warmup_epochs, 0, nb, nb_ssod)
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(
            f"[supervised_only_control] Image sizes {self.args.imgsz}\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting {self.epochs} epochs x {nb_ssod} steps (compute-matched to the SSOD run), "
            f"labeled-only, no unlabeled forward..."
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb_ssod
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        if RANK in {-1, 0}:
            self.ssod_diag = SSODDiagnosticsLogger(
                self.save_dir, ema_interval=self.args.diag_interval, dist_interval=self.args.diag_interval
            )
        unsup_stats_zero = {
            "num_pseudo_boxes": 0,
            "num_unsup_assigned": 0,
            "num_unsup_positive": 0,
            "sum_target_scores_unsup": 0.0,
            "assigned_anchors_per_pseudo_box_mean": 0.0,
            "assigned_anchors_per_pseudo_box_std": 0.0,
            "target_score_mean_unsup": 0.0,
            "target_score_std_unsup": 0.0,
            "cls_loss_skipped": False,
        }
        zero_loss_items = np.zeros(3, dtype=np.float32)

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(range(nb_ssod), total=nb_ssod)
            else:
                pbar = range(nb_ssod)
            labeled_iter = iter(self.train_loader)
            self.tloss = None
            for i in pbar:
                self.run_callbacks("on_train_batch_start")
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.train_loader)
                    labeled_batch = next(labeled_iter)
                ni = i + nb_ssod * epoch  # matches the real run's ni with burn_in_epochs=0
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(
                        1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size_ssod]).round())
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    labeled_batch = self.preprocess_batch(labeled_batch)
                    preds = self.model(labeled_batch["img"])
                    loss, self.loss_items = unwrap_model(self.model).loss(labeled_batch, preds)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    loss_items_det = self.loss_items.detach().cpu().numpy()
                    self.tloss = loss_items_det if self.tloss is None else (self.tloss * i + loss_items_det) / (i + 1)

                    if RANK in {-1, 0}:
                        sup_stats = compute_supervised_assignment_stats(
                            unwrap_model(self.model).criterion, preds, labeled_batch
                        )
                        self.ssod_diag.log_training_dynamics(
                            ni,
                            epoch,
                            loss_items_det,
                            zero_loss_items,
                            sup_stats,
                            unsup_stats_zero,
                            total_loss=float(self.loss.item()),
                            teacher_conf_max=None,
                            teacher_conf_mean=None,
                            num_teacher_predictions_pre_filter=None,
                            num_teacher_predictions_post_filter=None,
                        )
                        self.ssod_diag.log_assignment_dynamics(ni, epoch, sup_stats, unsup_stats_zero)

                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if RANK in {-1, 0}:
                        self.ssod_diag.log_grad_norm(ni, epoch, self.last_grad_norm, total_loss=float(self.loss.item()))
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_values = list(self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + len(loss_values)))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *loss_values,
                            labeled_batch["cls"].shape[0],
                            labeled_batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            final_epoch = epoch + 1 >= self.epochs  # unconditional: read outside any RANK guard below
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                nan = float("nan")
                train_keys = [f"train/{x}" for x in self.loss_names]
                train_vals = [
                    round(float(x), 5) for x in (self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))
                ]
                log_metrics = dict(zip(train_keys, train_vals))
                ssod_keys = [f"ssod/{x}" for x in self.loss_names]
                log_metrics.update(dict(zip(ssod_keys, [nan, nan, nan])))
                self.save_metrics(metrics={**log_metrics, **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            if getattr(self, "ssod_diag", None) is not None:
                self.ssod_diag.close()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.supervised_only_control:
            return self._do_train_supervised_only_control()
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nb_ssod = len(self.ssod_train_loader)  # number of batches for SSOD training
        nw = self._warmup_iterations(self.args.warmup_epochs, self.burn_in_epochs, nb, nb_ssod)
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        # self.nc = self.data["nc"]
        # self.n_labeled_per_cls = torch.zeros(self.nc, device= self.device, dtype= torch.long)
        # for l in self.train_loader.dataset.labels:  # 各画像の label dict
        #     cls_ids = l["cls"]
        #     # クラスごとに出現数をカウント
        #     self.n_labeled_per_cls.index_add_(0, cls_ids, torch.ones_like(cls_ids, dtype=torch.long))
        # self.N_labeled_images = len(self.train_loader.dataset)
        # self.N_unlabeled_images = len(self.ssod_train_loader.dataset)

        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch


        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start

        self.loss_func_ssod = EfficientTeacherLoss(
            unwrap_model(self.model),
            conf_threshold_high=self.conf_threshold_high,
            conf_threshold_low=self.conf_threshold_low,
            use_loc_conf=self.use_loc_conf,
            loc_conf_threshold=self.loc_conf_threshold,
            use_edge_conf=self.use_edge_conf,
            edge_conf_threshold=self.edge_conf_threshold,
            assignment_stability_method=self.assignment_stability_method,
            skip_zero_pseudo_cls_loss=self.skip_zero_pseudo_cls_loss,
            cls_loss_denom=self.cls_loss_denom,
            ema_denom_beta=self.args.ema_denom_beta,
        )
        if RANK in {-1, 0}:
            self.ssod_diag = SSODDiagnosticsLogger(self.save_dir, ema_interval=self.args.diag_interval, dist_interval=self.args.diag_interval)
        if self.start_epoch > self.burn_in_epochs:
            self._restore_teacher_after_resume()
            if RANK != -1 and self.world_size > 1 and isinstance(self.model, nn.parallel.DistributedDataParallel):
                self.model = nn.parallel.DistributedDataParallel(
                    unwrap_model(self.model), device_ids=[RANK], find_unused_parameters=True, static_graph=True
                )
        if self.domain_adaptation:
            # === forward hook で neck 出力を取る ===
            self.neck_feats_labeled = []
            self.neck_feats_unlabeled = []
            self.da_loss_weights = self.args.da_loss_weights

            def make_neck_hook(store_list):
                def hook(module, inp, out):
                    store_list.append(out)  # out: Tensor or list of Tensors
                return hook

            # Detect layer は model.model[-1]
            detect_module = self.model.model[-1]
            neck_ids = detect_module.f  # 例: [17, 20, 23] = neck(P3,P4,P5) 出力の index

            # ラベル付き用 hook
            self.neck_hooks_l = []
            for idx in neck_ids:
                h = self.model.model[idx].register_forward_hook(make_neck_hook(self.neck_feats_labeled))
                self.neck_hooks_l.append(h)

            # ラベルなし用 hook
            self.neck_hooks_u = []
            for idx in neck_ids:
                h = self.model.model[idx].register_forward_hook(make_neck_hook(self.neck_feats_unlabeled))
                self.neck_hooks_u.append(h)

            self.domain_nets = None 

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if epoch < self.burn_in_epochs:
                    
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string(include_ssod=True, include_da=getattr(self, "domain_adaptation", False)))
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                self.tloss_da = None
                if self.domain_adaptation:
                    unlabeled_iter = iter(self.ssod_train_loader)
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            if x.get("is_domain", False):
                                continue
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if not self.domain_adaptation:
                            if self.args.compile:
                                # Decouple inference and loss calculations for improved compile performance
                                preds = self.model(batch["img"])
                                loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                            else:
                                loss, self.loss_items = self.model(batch)
                        
                                self.loss = loss.sum()
                        # domain adaptation loss
                        if self.domain_adaptation:
                            try:
                                unlabeled_batch = next(unlabeled_iter)
                            except StopIteration:
                                unlabeled_iter = iter(self.ssod_train_loader)
                                unlabeled_batch = next(unlabeled_iter)
                            unlabeled_batch = self.preprocess_batch(unlabeled_batch)
                            if self.args.compile:
                                # Decouple inference and loss calculations for improved compile performance
                                self.neck_feats_unlabeled.clear()
                                preds_unlabeled = self.model(unlabeled_batch["img"])
                                neck_feats_unlabeled = [x for x in self.neck_feats_unlabeled]

                                self.neck_feats_labeled.clear()
                                preds = self.model(batch["img"])
                                neck_feats_labeled = [x for x in self.neck_feats_labeled]

                                loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                            else:
                                self.neck_feats_unlabeled.clear()   
                                preds_unlabeled = self.model(unlabeled_batch["img"])
                                neck_feats_unlabeled = [x for x in self.neck_feats_unlabeled]

                                self.neck_feats_labeled.clear()
                                preds = self.model(batch["img"])
                                neck_feats_labeled = [x for x in self.neck_feats_labeled]

                                loss, self.loss_items = self.model.loss(batch, preds)
                            
                            

                            if self.domain_nets is None:
                                self.domain_nets = nn.ModuleList()
                                for f in neck_feats_labeled:
                                    c = f.shape[1]  # channel数
                                    net = DomainAdversarialNet(in_dim=c, num_classes=2).to(self.device)
                                    self.domain_nets.append(net)
                                    # optimizer に domain_nets のパラメータを追加
                                self.optimizer.add_param_group({
                                            "params": self.domain_nets.parameters(),
                                            "lr": self.args.lr0,
                                            "initial_lr": self.args.lr0,
                                            "weight_decay": 0.0,
                                            "is_domain": True,
                                        })
                            loss_da_s_list = []
                            loss_da_t_list = []

                            for f_s, f_t, net in zip(neck_feats_labeled, neck_feats_unlabeled, self.domain_nets):
                                feat_s = flatten_multi_scale_feats(f_s)  # [Ns, Ck]
                                feat_t = flatten_multi_scale_feats(f_t)  # [Nt, Ck]

                                logits_s = net(feat_s)
                                logits_t = net(feat_t)

                                labels_s = torch.zeros(logits_s.size(0), dtype=torch.long, device=self.device)
                                labels_t = torch.ones (logits_t.size(0), dtype=torch.long, device=self.device)

                                loss_da_s = F.cross_entropy(logits_s, labels_s)
                                loss_da_t = F.cross_entropy(logits_t, labels_t)

                                loss_da_s_list.append(loss_da_s)
                                loss_da_t_list.append(loss_da_t)

                            loss_da_s = sum(loss_da_s_list) * 0.5
                            loss_da_t = sum(loss_da_t_list) * 0.5
                            loss_da   = loss_da_s + loss_da_t

                            self.loss = loss.sum() + loss_da * self.da_loss_weights

                            # moving avg for logging (3要素: source, target, total)
                            curr_da = np.stack([loss_da_s.detach().cpu().numpy(), loss_da_t.detach().cpu().numpy(), loss_da.detach().cpu().numpy()])
                            self.tloss_da = curr_da if self.tloss_da is None else (self.tloss_da * i + curr_da) / (i + 1)
                        loss_items_det = self.loss_items.detach().cpu().numpy()

                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = loss_items_det if self.tloss is None else (self.tloss * i + loss_items_det) / (i + 1)
                    

                    # Backward
                    self.scaler.scale(self.loss).backward()
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni
                        # BUG FIX: self.teacher (the pseudo-label-generating EMA) was previously
                        # never updated after being created at burn-in end -- only its non-weight
                        # attributes were refreshed via update_attr() once per epoch. It therefore
                        # stayed frozen at the burn-in snapshot for the entire pseudo-label phase,
                        # rather than actually tracking the student as a Mean-Teacher EMA is meant
                        # to. `self.ema` (the separate, checkpoint-saved EMA) was already updated
                        # correctly inside optimizer_step() above; self.teacher needs the same call.
                        # (self.teacher does not exist yet during burn-in, hence the guard.)
                        if getattr(self, "teacher", None) is not None:
                            self.teacher.update(self.model)
                        if RANK in {-1, 0}:
                            self.ssod_diag.log_grad_norm(ni, epoch, self.last_grad_norm, total_loss=float(self.loss.item()))

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                    # Log
                    if RANK in {-1, 0}:
                        # 固定順序: train(3) -> ssod(3) -> [da(3) if enabled]
                        nan = float("nan")
                        loss_values = []
                        # train
                        train_vals = list(self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))
                        loss_values += train_vals
                        # ssod (burn-inではNaN埋め)
                        loss_values += [nan, nan, nan]
                        # da（有効時のみ出力）
                        if self.domain_adaptation:
                            da_vals = list(self.tloss_da if len(self.tloss_da.shape) > 0 else torch.unsqueeze(self.tloss_da, 0))
                            da_vals = (da_vals + [nan, nan, nan])[:3]
                            loss_values += da_vals
                        num_vals = 2 + len(loss_values)  # gpu_mem + losses... + instances & size
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * num_vals)
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *loss_values,  # losses (train, ssod, da)
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

                self.run_callbacks("on_train_epoch_end")
                # Computed unconditionally: `final_epoch` is read below outside any RANK guard
                # (`if self.args.val or final_epoch or ...`), so leaving this inside `if RANK in
                # {-1, 0}` left it unbound on other ranks -- masked whenever self.args.val is
                # truthy (Python's `or` short-circuits before ever evaluating final_epoch), but a
                # real UnboundLocalError on non-zero ranks under DDP with val=False.
                final_epoch = epoch + 1 >= self.epochs
                if RANK in {-1, 0}:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()

                # NaN recovery
                if self._handle_nan_recovery(epoch):
                    continue

                self.nan_recovery_attempts = 0
                if RANK in {-1, 0}:
                    # 常に同一キーでCSVに保存（値が無い場合は NaN）
                    nan = float("nan")
                    # train losses (always 3 keys)
                    train_keys = [f"train/{x}" for x in self.loss_names]
                    if self.tloss is not None:
                        train_vals = [round(float(x), 5) for x in (self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))]
                    else:
                        train_vals = [nan, nan, nan]
                    log_metrics = dict(zip(train_keys, train_vals))
                    # ssod losses placeholder (not used in burn-in)
                    ssod_keys = [f"ssod/{x}" for x in self.loss_names]
                    ssod_vals = [nan, nan, nan]
                    log_metrics.update(dict(zip(ssod_keys, ssod_vals)))
                    # da losses (3 keys)
                    da_keys = ["da/loss_s", "da/loss_t", "da/loss"]

                    if self.domain_adaptation:
                        if getattr(self, "tloss_da", None) is not None:
                            da_vals = [round(float(x), 5) for x in (self.tloss_da if len(self.tloss_da.shape) > 0 else torch.unsqueeze(self.tloss_da, 0))]
                            log_metrics.update(dict(zip(da_keys, da_vals)))
                        else:
                            log_metrics.update(dict(zip(da_keys, [nan, nan, nan])))
                    # save with validator metrics and lr
                    self.save_metrics(metrics={**log_metrics, **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1
            
            else:
                if epoch == self.burn_in_epochs:
                    print("psud-labeling start!! epoch: ", epoch)

                    # ===== ここで DA を完全に終了させる =====
                    if self.domain_adaptation:
                        # 1) optimizer から domain 用の param_group を外す
                        for g in self.optimizer.param_groups:
                            if g.get("is_domain", False):
                                g["lr"] = 0.0
                                for p in g["params"]:
                                    p.requires_grad = False

                        # 2) forward hook を解除（登録しているなら）
                        for h in getattr(self, "neck_hooks_l", []):
                            h.remove()
                        for h in getattr(self, "neck_hooks_u", []):
                            h.remove()
                        self.neck_hooks_l = []
                        self.neck_hooks_u = []

                        # 4) 特徴マップ用バッファも空にする
                        if hasattr(self, "neck_feats_labeled"):
                            self.neck_feats_labeled.clear()
                        if hasattr(self, "neck_feats_unlabeled"):
                            self.neck_feats_unlabeled.clear()
                        torch.cuda.empty_cache()
                        LOGGER.info("Domain Adaptation disabled after burn-in.")
                    try:
                        self.teacher = ModelEMA(self.model)
                        LOGGER.info("Teacher EMA at burn-in end.")
                        if RANK in {-1, 0}:
                            self.bn_mismatch_probe = BNMismatchProbe(unwrap_model(self.teacher.ema))
                    except Exception as e:
                        LOGGER.warning(f"Teacher EMA reset failed: {e}")

                    # From here on every step runs two forward passes (labeled, then unlabeled)
                    # through self.model before a single combined backward(). Plain DDP's Reducer
                    # assumes one forward per backward and corrupts autograd's in-place version
                    # tracking across the two calls, raising "modified by an inplace operation"
                    # RuntimeErrors. Re-wrapping with static_graph=True (the officially documented
                    # fix for multiple-forward-per-iteration DDP use) is safe here since burn-in
                    # already finished and no parameters are frozen/unfrozen afterward.
                    if RANK != -1 and self.world_size > 1 and isinstance(self.model, nn.parallel.DistributedDataParallel):
                        self.model = nn.parallel.DistributedDataParallel(
                            unwrap_model(self.model), device_ids=[RANK], find_unused_parameters=True, static_graph=True
                        )

                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                    self.ssod_train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.ssod_train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()
                    self.ssod_train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string(include_ssod=True, include_da=getattr(self, "domain_adaptation", False)))
                    pbar = TQDM(enumerate(self.ssod_train_loader), total=nb_ssod)
                labeled_iter = iter(self.train_loader)
                self.tloss = None
                self.tloss_unlabeled = None
                for i, unlabeled_batch in pbar:
                    stability_step_start = time.perf_counter()
                    self.run_callbacks("on_train_batch_start")
                    #labeledデータ取得. なくなったら作り直す
                    try:
                        labeled_batch = next(labeled_iter)
                    except StopIteration:
                        labeled_iter = iter(self.train_loader)
                        labeled_batch = next(labeled_iter)
                    # Warmup
                    ni = i + nb_ssod * (epoch-self.burn_in_epochs) + nb * self.burn_in_epochs
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size_ssod]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        labeled_batch = self.preprocess_batch(labeled_batch)
                        unlabeled_batch = self.preprocess_batch(unlabeled_batch)

                        # self.ema.ema.train()でモデルの出力をそのまま得られる. しかし特徴点ごとなので整形必要.
                        # self.ema.ema.train()
                        with torch.no_grad():
                            unlabeled_preds_teacher, unlabeled_feats_teacher = self.teacher.ema(unlabeled_batch["img"]) # (preds, feature)の形式で出力 boxはxyXY
                        unlabeled_preds_teacher = unlabeled_preds_teacher.detach()
                        if RANK in {-1, 0}:
                            # BN mismatch probe hooks fired during the forward call just above --
                            # read them now, before the next teacher forward overwrites them.
                            if getattr(self, "bn_mismatch_probe", None) is not None:
                                self.ssod_diag.maybe_log_bn_mismatch(ni, epoch, self.bn_mismatch_probe)
                            _teacher_conf_all = unlabeled_preds_teacher[:, 4 : 4 + self.loss_func_ssod.nc, :]
                            teacher_conf_max = float(_teacher_conf_all.max().item())
                            teacher_conf_mean = float(_teacher_conf_all.mean().item())

                        # DFL localization confidence: recover the raw per-edge distributions in the
                        # same anchor ordering `non_max_suppression` indexes into (see
                        # ultralytics/utils/dfl_confidence.py), so surviving pseudo-boxes can be traced
                        # back to their DFL logits after NMS.
                        pred_distri_teacher, _ = torch.cat(
                            [xi.view(unlabeled_feats_teacher[0].shape[0], self.loss_func_ssod.no, -1) for xi in unlabeled_feats_teacher], 2
                        ).split((self.loss_func_ssod.reg_max * 4, self.loss_func_ssod.nc), 1)
                        pred_distri_teacher = pred_distri_teacher.permute(0, 2, 1).contiguous().detach()  # (B, N, 4*reg_max)
                        teacher_anchor_points, teacher_stride_tensor = make_anchors(
                            unlabeled_feats_teacher, self.loss_func_ssod.stride, 0.5
                        )

                        unlabeled_labels, keep_idxs = non_max_suppression(
                            unlabeled_preds_teacher, conf_thres=0.01, iou_thres=0.65, return_idxs=True
                        )
                        # Diagnostics only (see ultralytics.utils.ssod_diagnostics section 3): how many
                        # raw teacher candidates cleared NMS's own conf_thres=0.01 gate before NMS's IoU
                        # suppression ever ran, so num_removed_by_nms below is isolated from
                        # num_removed_by_confidence (our separate, much higher SSOD adoption threshold).
                        num_candidates_prenms = int(
                            (unlabeled_preds_teacher[:, 4 : 4 + self.loss_func_ssod.nc, :].amax(1) >= 0.01).sum()
                        )
                        n_post_nms = sum(len(labels) for labels in unlabeled_labels)
                        batch_idx_list = []
                        loc_conf_list = []
                        edge_conf_list = []
                        candidate_box_list = []
                        entropy_list = []
                        # Do not shadow the outer dataloader batch index `i`; it is used below for
                        # running-loss averaging, logging and checkpoint visualization cadence.
                        for image_idx, labels in enumerate(unlabeled_labels):
                            if labels.numel() == 0:
                                continue  # その画像にラベルが無い場合はスキップ
                            batch_idx_list.append(
                                torch.full(
                                    (labels.shape[0], 1),  # その画像の物体数ぶん
                                    image_idx,              # その画像のバッチインデックス
                                    device=labels.device,
                                    dtype=torch.long,
                                )
                            )
                            box_distri = pred_distri_teacher[image_idx, keep_idxs[image_idx]].view(
                                -1, 4, self.loss_func_ssod.reg_max
                            )
                            edge_conf, box_conf = localization_confidence(box_distri, self.loss_func_ssod.reg_max)
                            loc_conf_list.append(box_conf.unsqueeze(-1))
                            edge_conf_list.append(edge_conf)
                            entropy_list.append(dfl_entropy(box_distri))
                            selected_idx = keep_idxs[image_idx]
                            selected_anchors = teacher_anchor_points[selected_idx]
                            selected_strides = teacher_stride_tensor[selected_idx]
                            if self.assignment_perturbation == "dfl":
                                candidate_xyxy = dfl_supported_box_candidates(
                                    box_distri, selected_anchors, selected_strides
                                )
                                candidate_xyxy[0] = labels[:, :4]
                            else:
                                dfl_stats = dfl_distribution_statistics(box_distri)
                                candidate_xyxy = geometric_box_candidates(
                                    labels[:, :4],
                                    self.assignment_perturbation,
                                    dfl_width=dfl_stats["width"],
                                    dfl_expectation=dfl_stats["expectation"],
                                    anchor_points=selected_anchors,
                                    stride_tensor=selected_strides,
                                    reg_max=self.loss_func_ssod.reg_max,
                                    fixed_ratio=0.05,
                                )
                            candidate_xywh = xyxy_to_xywh(candidate_xyxy.reshape(-1, 4)).view(9, -1, 4)
                            candidate_box_list.append(candidate_xywh.permute(1, 0, 2) / unlabeled_batch["img"].shape[2])
                        if batch_idx_list:
                            unlabeled_batch_idx = torch.cat(batch_idx_list, dim=0)  # shape: [total_num_labels, 1]
                            unlabeled_loc_conf = torch.cat(loc_conf_list, dim=0)  # shape: [total_num_labels, 1]
                            unlabeled_edge_conf = torch.cat(edge_conf_list, dim=0)  # shape: [total_num_labels, 4]
                            unlabeled_candidate_bboxes = torch.cat(candidate_box_list, dim=0)  # (N, 9, 4), normalized xywh
                            unlabeled_dfl_entropy = torch.cat(entropy_list, dim=0)  # shape: [total_num_labels]
                        else:
                            # ラベルが1つも無いケース
                            unlabeled_batch_idx = torch.empty((0, 1), device=unlabeled_labels[0].device, dtype=torch.long)
                            unlabeled_loc_conf = torch.empty((0, 1), device=unlabeled_labels[0].device, dtype=torch.float)
                            unlabeled_edge_conf = torch.empty((0, 4), device=unlabeled_labels[0].device, dtype=torch.float)
                            unlabeled_candidate_bboxes = torch.empty(
                                (0, 9, 4), device=unlabeled_labels[0].device, dtype=torch.float
                            )
                            unlabeled_dfl_entropy = torch.empty((0,), device=unlabeled_labels[0].device, dtype=torch.float)


                        unlabeled_labels = torch.cat((unlabeled_labels))
                        unlabeled_bboxes = xyxy_to_xywh(unlabeled_labels[:, :4]) / unlabeled_batch["img"].shape[2]
                        unlabeled_cls = unlabeled_labels[:, -1].unsqueeze(1)
                        unlabeled_conf = unlabeled_labels[:, -2].unsqueeze(1)

                        # Always decouple inference and loss (mathematically identical to
                        # `self.model(labeled_batch)` -- see DetectionModel.forward/.loss in
                        # ultralytics/nn/tasks.py, same single forward pass either way) so the
                        # raw `preds` are available below for compute_supervised_assignment_stats
                        # without an extra model forward call (which would double-update
                        # BatchNorm's running stats and change training dynamics).
                        preds = self.model(labeled_batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(labeled_batch, preds)


                        #SSOD Loss 計算
                        # Student trains on the strongly augmented view; the teacher (above) only ever
                        # sees the weak view. Both share identical geometry/labels, so the teacher's
                        # pseudo-boxes need no remapping to be used against the student's predictions.
                        student_unlabeled_img = unlabeled_batch.get("img_strong", unlabeled_batch["img"])
                        # Under DDP, calling self.model(...) a second time in the same iteration (the
                        # labeled forward above already went through it) re-triggers DDP's
                        # prepare-for-backward bookkeeping and corrupts autograd's in-place version
                        # tracking on shared buffers ("modified by an inplace operation" RuntimeError).
                        # Forwarding through the unwrapped module for this second call avoids that:
                        # gradients still flow into the same shared parameters and get reduced by the
                        # single combined backward() below, since DDP's hooks are attached to the
                        # parameters themselves, not to which module reference issued the forward.
                        preds_unlabeled = unwrap_model(self.model)(student_unlabeled_img)
                        _diag_now = RANK in {-1, 0} and self.ssod_diag.should_log_dist(ni)
                        loss_unlabeled, self.loss_items_unlabeled, reliable_mask, unreliable_mask = self.loss_func_ssod(
                            preds_unlabeled,
                            unlabeled_bboxes,
                            unlabeled_cls,
                            unlabeled_conf,
                            unlabeled_batch_idx,
                            unlabeled_loc_conf,
                            unlabeled_edge_conf,
                            unlabeled_candidate_bboxes,
                            compute_extra_diag=_diag_now,
                        )
                        if self.assignment_stability_method != "off" and RANK in {-1, 0}:
                            stability_record = {
                                "epoch": epoch,
                                "batch": i,
                                "iteration": ni,
                                "method": self.assignment_stability_method,
                                "perturbation": self.assignment_perturbation,
                                **self.loss_func_ssod.last_stability_stats,
                                "iteration_seconds_to_loss": time.perf_counter() - stability_step_start,
                                "vram_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                                "vram_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                            }
                            stability_log = self.save_dir / "assignment_stability.jsonl"
                            with stability_log.open("a", encoding="utf-8") as file:
                                file.write(json.dumps(stability_record) + "\n")



                        loss_items_det = self.loss_items.detach().cpu().numpy()
                        loss_items_unlabeled_det = self.loss_items_unlabeled.detach().cpu().numpy()

                        if self.loss_balancing_mode == "ema_scale":
                            # Zoph et al.-style loss-scale matching: instead of a fixed
                            # ssod_weight, scale the unsupervised loss so its EMA magnitude tracks
                            # the supervised loss's EMA magnitude, then still apply ssod_weight on
                            # top. Orthogonal to EfficientTeacherLoss's own cls_loss_denom -- this
                            # rescales the already-computed total unsupervised loss, it does not
                            # change how that loss was normalized internally.
                            l_sup_val = float(loss.sum().item())
                            l_unsup_val = float(loss_unlabeled.sum().item())
                            self.ema_loss_sup = (
                                l_sup_val if self.ema_loss_sup is None
                                else self.loss_balance_beta * self.ema_loss_sup + (1 - self.loss_balance_beta) * l_sup_val
                            )
                            self.ema_loss_unsup = (
                                l_unsup_val if self.ema_loss_unsup is None
                                else self.loss_balance_beta * self.ema_loss_unsup + (1 - self.loss_balance_beta) * l_unsup_val
                            )
                            balance_factor = self.ema_loss_sup / max(self.ema_loss_unsup, 1e-6)
                            self.loss = loss.sum() + self.ssod_weight * balance_factor * loss_unlabeled.sum()
                        else:
                            self.loss = loss.sum() + self.ssod_weight * loss_unlabeled.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = loss_items_det if self.tloss is None else (self.tloss * i + loss_items_det) / (i + 1)
                        self.tloss_unlabeled = (
                            loss_items_unlabeled_det if self.tloss_unlabeled is None
                            else (self.tloss_unlabeled * i + loss_items_unlabeled_det) / (i + 1)
                        )

                        # ===== SSOD training-dynamics diagnostics (see ssod_diagnostics.py) =====
                        # Everything logged here only exists during this live forward/assignment
                        # pass -- it cannot be recomputed later from a checkpoint or from running
                        # inference, unlike validation metrics or PR curves.
                        if RANK in {-1, 0}:
                            unsup_stats = self.loss_func_ssod.last_assignment_stats
                            sup_stats = compute_supervised_assignment_stats(
                                unwrap_model(self.model).criterion, preds, labeled_batch
                            )
                            self.ssod_diag.log_training_dynamics(
                                ni,
                                epoch,
                                loss_items_det,
                                loss_items_unlabeled_det,
                                sup_stats,
                                unsup_stats,
                                total_loss=float(self.loss.item()),
                                teacher_conf_max=teacher_conf_max,
                                teacher_conf_mean=teacher_conf_mean,
                                num_teacher_predictions_pre_filter=num_candidates_prenms,
                                num_teacher_predictions_post_filter=n_post_nms,
                            )
                            self.ssod_diag.log_assignment_dynamics(ni, epoch, sup_stats, unsup_stats)
                            self.ssod_diag.maybe_log_ema(ni, epoch, self.teacher, unwrap_model(self.model))
                            if _diag_now:
                                reliable_entropy = unlabeled_dfl_entropy[reliable_mask]
                                rejected_entropy = unlabeled_dfl_entropy[~reliable_mask]
                                n_reliable = int(reliable_mask.sum())
                                self.ssod_diag.maybe_log_pseudo_label_dynamics(
                                    ni,
                                    epoch,
                                    unlabeled_batch["img"].shape[0],
                                    n_post_nms,  # NMS survivors, before our reliable-confidence gate
                                    n_reliable,  # adopted as pseudo-labels
                                    n_post_nms - n_reliable,  # dropped by our confidence/loc-conf gate
                                    num_candidates_prenms - n_post_nms,  # dropped by NMS's own IoU suppression
                                    unlabeled_conf.squeeze(-1)[reliable_mask],
                                    reliable_entropy,
                                    rejected_entropy,
                                    self.loss_func_ssod.last_diag,
                                )

                        # ===== Loss-spike causal diagnosis (no-op unless spike_diag_enabled) =====
                        if self.spike_diag_enabled and RANK in {-1, 0}:
                            ssod_cls_loss_value = float(loss_items_unlabeled_det[1])  # order: box, cls, dfl
                            is_trigger = self.spike_diag_intervention_iter == ni or ssod_cls_loss_value > self.spike_diag_threshold
                            norm_s = norm_u = cos_su = r_t = None
                            if is_trigger:
                                # Only pay for the extra backward passes when something is actually
                                # interesting (a spike, or the pre-designated intervention iteration).
                                norm_s, norm_u, cos_su, r_t = compute_split_gradients(
                                    unwrap_model(self.model), loss.sum(), self.ssod_weight * loss_unlabeled.sum()
                                )

                            n_pseudo = int(unlabeled_bboxes.shape[0])
                            n_imgs_pseudo = int(unlabeled_batch_idx.unique().numel()) if unlabeled_batch_idx.numel() else 0
                            log_iteration_stats(
                                self.spike_diag_log_path,
                                {
                                    "epoch": epoch,
                                    "iteration": ni,
                                    "ssod_box_loss": float(loss_items_unlabeled_det[0]),
                                    "ssod_cls_loss": ssod_cls_loss_value,
                                    "ssod_dfl_loss": float(loss_items_unlabeled_det[2]),
                                    "sup_box_loss": float(loss_items_det[0]),
                                    "sup_cls_loss": float(loss_items_det[1]),
                                    "sup_dfl_loss": float(loss_items_det[2]),
                                    "lr": self.optimizer.param_groups[0]["lr"],
                                    "num_pseudo_boxes": n_pseudo,
                                    "num_images_with_pseudo": n_imgs_pseudo,
                                    "pseudo_boxes_per_image": n_pseudo / max(n_imgs_pseudo, 1),
                                    "teacher_conf_mean": float(unlabeled_conf.mean().item()) if unlabeled_conf.numel() else None,
                                    "teacher_conf_min": float(unlabeled_conf.min().item()) if unlabeled_conf.numel() else None,
                                    "teacher_conf_max": float(unlabeled_conf.max().item()) if unlabeled_conf.numel() else None,
                                    "grad_norm_s": norm_s,
                                    "grad_norm_u": norm_u,
                                    "cos_su": cos_su,
                                    "r_t": r_t,
                                    "scaler_scale": self.scaler.get_scale() if self.amp else None,
                                },
                            )

                            if (not self._spike_captured) and ssod_cls_loss_value > self.spike_diag_threshold:
                                snapshot = snapshot_trainer_state(self, labeled_batch, unlabeled_batch, epoch, ni)
                                save_snapshot(snapshot, self.spike_diag_snapshot_path)
                                self._spike_captured = True
                                LOGGER.info(
                                    f"[spike_diag] captured pre-spike state at epoch={epoch} ni={ni} "
                                    f"ssod_cls_loss={ssod_cls_loss_value:.3f} -> {self.spike_diag_snapshot_path}"
                                )
                                if self.spike_diag_stop_after_capture:
                                    raise SpikeCaptured(
                                        f"epoch={epoch} ni={ni} ssod_cls_loss={ssod_cls_loss_value:.3f} "
                                        f"snapshot={self.spike_diag_snapshot_path}"
                                    )

                            if self.spike_diag_intervention_iter == ni and self.spike_diag_intervention != "normal":
                                self.loss = apply_intervention(
                                    self.spike_diag_intervention,
                                    loss.sum(),
                                    loss_unlabeled.sum(),
                                    self.ssod_weight,
                                    self.spike_diag_g_ref,
                                    norm_u,
                                )
                                if RANK != -1:
                                    self.loss *= self.world_size

                    # Backward
                    self.scaler.scale(self.loss).backward()
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni
                        # BUG FIX: self.teacher (the pseudo-label-generating EMA) was previously
                        # never updated after being created at burn-in end -- only its non-weight
                        # attributes were refreshed via update_attr() once per epoch. It therefore
                        # stayed frozen at the burn-in snapshot for the entire pseudo-label phase,
                        # rather than actually tracking the student as a Mean-Teacher EMA is meant
                        # to. `self.ema` (the separate, checkpoint-saved EMA) was already updated
                        # correctly inside optimizer_step() above; self.teacher needs the same call.
                        # (self.teacher does not exist yet during burn-in, hence the guard.)
                        if getattr(self, "teacher", None) is not None:
                            self.teacher.update(self.model)
                        if RANK in {-1, 0}:
                            self.ssod_diag.log_grad_norm(ni, epoch, self.last_grad_norm, total_loss=float(self.loss.item()))

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                    # Log
                    if RANK in {-1, 0}:
                        # 固定順序: train(3) -> ssod(3) -> da(3)
                        nan = float("nan")
                        loss_values = []
                        # train
                        if self.tloss is not None:
                            loss_values += list(self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))
                        else:
                            loss_values += [nan, nan, nan]
                        # ssod
                        if getattr(self, "tloss_unlabeled", None) is not None:
                            loss_values += list(
                                self.tloss_unlabeled
                                if len(self.tloss_unlabeled.shape) > 0
                                else torch.unsqueeze(self.tloss_unlabeled, 0)
                            )
                        else:
                            loss_values += [nan, nan, nan]
                        # da
                        if self.domain_adaptation:
                            loss_values += [nan, nan, nan]
                        num_vals = 2 + len(loss_values)  # (gpu_mem, then losses..., then instances & size)
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * num_vals)
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *loss_values,  # losses (train, ssod, da)
                                labeled_batch["cls"].shape[0],  # batch size, i.e. 8
                                labeled_batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(labeled_batch, ni)
                        # Only dump pseudo-label visualizations at checkpoint epochs (matching
                        # save_period) and just the first few batches -- plotting every batch of
                        # every epoch (the previous behavior) writes tens of thousands of images
                        # over a full run and synchronously stalls the training loop each time.
                        if (
                            self.args.pseudo_label_plots
                            and epoch % max(self.args.save_period, 1) == 0
                            and ni < 4
                        ):
                            pseudo_batch = {}
                            pseudo_batch["img"] = unlabeled_batch["img"]
                            pseudo_batch["cls"] = unlabeled_cls[reliable_mask].squeeze(-1)
                            pseudo_batch["bboxes"] = unlabeled_bboxes[reliable_mask]
                            pseudo_batch["im_file"] = unlabeled_batch["im_file"]
                            pseudo_batch["batch_idx"] = unlabeled_batch_idx[reliable_mask].squeeze(-1)
                            pseudo_batch["resized_shape"] = unlabeled_batch["resized_shape"]
                            pseudo_batch["ori_shape"] = unlabeled_batch["ori_shape"]
                            self.plot_pseudo_samples(pseudo_batch, epoch, ni)
                    self.run_callbacks("on_train_batch_end")

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

                self.run_callbacks("on_train_epoch_end")
                # See the matching comment in the burn-in branch above: must be set unconditionally.
                final_epoch = epoch + 1 >= self.epochs
                if RANK in {-1, 0}:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                    self.teacher.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()

                # NaN recovery
                if self._handle_nan_recovery(epoch):
                    continue

                self.nan_recovery_attempts = 0
                if RANK in {-1, 0}:
                    # 常に同一キーでCSVに保存（値が無い場合は NaN）
                    nan = float("nan")
                    # train losses
                    train_keys = [f"train/{x}" for x in self.loss_names]
                    if self.tloss is not None:
                        train_vals = [round(float(x), 5) for x in (self.tloss if len(self.tloss.shape) > 0 else torch.unsqueeze(self.tloss, 0))]
                    else:
                        train_vals = [nan, nan, nan]
                    log_metrics = dict(zip(train_keys, train_vals))
                    # ssod losses
                    ssod_keys = [f"ssod/{x}" for x in self.loss_names]
                    if getattr(self, "tloss_unlabeled", None) is not None:
                        ssod_vals = [round(float(x), 5) for x in (self.tloss_unlabeled if len(self.tloss_unlabeled.shape) > 0 else torch.unsqueeze(self.tloss_unlabeled, 0))]
                    else:
                        ssod_vals = [nan, nan, nan]
                    log_metrics.update(dict(zip(ssod_keys, ssod_vals)))
                    # da losses（有効時のみ出力）
                    if self.domain_adaptation:
                        da_keys = ["da/loss_s", "da/loss_t", "da/loss"]
                        if getattr(self, "tloss_da", None) is not None:
                            da_vals = [round(float(x), 5) for x in (self.tloss_da if len(self.tloss_da.shape) > 0 else torch.unsqueeze(self.tloss_da, 0))]
                            da_vals = (da_vals + [nan, nan, nan])[:3]
                        else:
                            da_vals = [nan, nan, nan]
                        log_metrics.update(dict(zip(da_keys, da_vals)))
                    self.save_metrics(metrics={**log_metrics, **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)  # clear if memory utilization > 50%

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1
                
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            if getattr(self, "ssod_diag", None) is not None:
                self.ssod_diag.close()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

def flatten_multi_scale_feats(f):
    # f: [N, C, H, W] → [N*H*W, C]
    n, c, h, w = f.shape
    return f.view(n, c, h * w).permute(0, 2, 1).reshape(-1, c)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (..., 4) 形式の tensor [x1, y1, x2, y2]
    return: (..., 4) 形式の tensor [cx, cy, w, h]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack((cx, cy, w, h), dim=-1)
