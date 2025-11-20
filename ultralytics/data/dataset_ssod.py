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
        if self.augment:
            hyp.mosaic = hyp.mosaic_ssod if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup_ssod if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix_ssod if self.augment and not self.rect else 0.0
            transforms = weak_geo_augmentations(self, self.imgsz, hyp)
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
