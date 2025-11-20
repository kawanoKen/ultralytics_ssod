from __future__ import annotations

import os

import torch
from torch.utils.data import distributed
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.dataset_ssod import YOLODataset_ssod
from ultralytics.utils import RANK, colorstr

def build_dataloader_ssod(
    dataset: YOLODataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader_ssod(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader_ssod(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )

def build_yolo_dataset_ssod(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset_ssod
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )