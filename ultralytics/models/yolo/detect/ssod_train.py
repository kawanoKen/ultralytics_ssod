# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import gc
import math
import os
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

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, check_det_dataset_ssod
from ultralytics.nn.tasks import load_checkpoint
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
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results

import random
from typing import Any

from ultralytics.data import build_dataloader, build_yolo_dataset
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

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
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
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
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
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
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

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
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

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

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
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        self.ssod_train_loader = self.get_dataloader(
            self.data["ssod_train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
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
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
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
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

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

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
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
        burn_in_epoch = self.args.burn_in_epochs


        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if epoch < burn_in_epoch:
                    
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                    # Backward
                    self.scaler.scale(self.loss).backward()
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

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
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
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
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
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
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
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
                if epoch == burn_in_epoch:
                    print("SSOD training start!!")
            
            else:
                                
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                # Update dataloader attributes (optional)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                    # Backward
                    self.scaler.scale(self.loss).backward()
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

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
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
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
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
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
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
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
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")