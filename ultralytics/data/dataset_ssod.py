import random
from copy import copy

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
    Mosaic,
    RandomPerspective,
    RandomFlip,
)
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr

class YOLODataset_ssod(YOLODataset):
    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        # Do not overwrite the trainer's shared configuration while mapping the
        # SSOD-specific augmentation probabilities onto the standard pipeline.
        hyp = copy(hyp)
        if self.augment:
            hyp.mosaic = hyp.mosaic_ssod if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup_ssod if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix_ssod if self.augment and not self.rect else 0.0
            transforms = weak_geo_augmentations(self, self.imgsz, hyp)
            # Student sees a strongly, appearance-only augmented copy of the same
            # geometry/labels the teacher (weak view, "img") uses for pseudo-labels.
            transforms.append(StrongAugment(hyp))
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """Disable every mix augmentation used by the unlabeled SSOD pipeline."""
        hyp = copy(hyp)
        hyp.mosaic_ssod = 0.0
        hyp.mixup_ssod = 0.0
        hyp.cutmix_ssod = 0.0
        super().close_mosaic(hyp)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Same as YOLODataset.collate_fn but also stacks the extra 'img_strong' tensor."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "img_strong", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class StrongAugment:
    """
    Appearance-only "strong" augmentation for the student branch of SSOD.

    Applied after all geometric transforms (Mosaic/RandomPerspective/RandomFlip) have
    finalized ``labels["img"]`` (the teacher/weak view) and its matching ``instances``.
    Builds a second, heavily perturbed view of the *same* pixels under
    ``labels["img_strong"]`` so the student and teacher share identical geometry/labels
    while only the student's input appearance is strongly perturbed (STAC / Unbiased
    Teacher style weak-teacher / strong-student decoupling).

    Only color jitter, grayscale, blur and cutout are used since none of them move
    pixels around, so the bboxes/instances computed for the weak view stay valid.
    """

    def __init__(self, hyp: IterableSimpleNamespace) -> None:
        self.hgain = getattr(hyp, "strong_hsv_h", 0.05)
        self.sgain = getattr(hyp, "strong_hsv_s", 0.9)
        self.vgain = getattr(hyp, "strong_hsv_v", 0.9)
        self.contrast_gain = getattr(hyp, "strong_contrast", 0.5)
        self.grayscale_p = getattr(hyp, "strong_grayscale_p", 0.2)
        self.blur_p = getattr(hyp, "strong_blur_p", 0.5)
        self.cutout_p = getattr(hyp, "strong_cutout_p", 0.7)
        self.cutout_n = getattr(hyp, "strong_cutout_n", 5)
        self.cutout_size = getattr(hyp, "strong_cutout_size", 0.1)

    def __call__(self, labels: dict) -> dict:
        img = labels["img"].copy()
        img = self._color_jitter(img)
        if random.random() < self.grayscale_p:
            img = self._grayscale(img)
        if random.random() < self.blur_p:
            img = self._blur(img)
        if random.random() < self.cutout_p:
            img = self._cutout(img)
        # Match Format._format_img's HWC -> CHW conversion; Format() itself never
        # touches "img_strong", so it must already be a tensor by the time it reaches
        # collate_fn.
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        labels["img_strong"] = torch.from_numpy(img)
        return labels

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] != 3:
            return img
        dtype = img.dtype
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        if self.contrast_gain:
            alpha = 1.0 + np.random.uniform(-self.contrast_gain, self.contrast_gain)
            beta = np.random.uniform(-32, 32)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    @staticmethod
    def _grayscale(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _blur(img: np.ndarray) -> np.ndarray:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)

    def _cutout(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        mean = img.mean(axis=(0, 1))
        for _ in range(random.randint(1, self.cutout_n)):
            cut_h = int(h * random.uniform(0.02, self.cutout_size))
            cut_w = int(w * random.uniform(0.02, self.cutout_size))
            y = random.randint(0, max(h - cut_h, 0))
            x = random.randint(0, max(w - cut_w, 0))
            img[y : y + cut_h, x : x + cut_w] = mean
        return img


def weak_geo_augmentations(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """幾何系の変形をweak augmentationとして採用"""
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic_ssod)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])

    flip_idx = dataset.data.get("flip_idx", [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0
            LOGGER.warning(
                "No 'flip_idx' array defined in data.yaml, "
                "disabling 'fliplr' and 'flipud' augmentations."
            )
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )
