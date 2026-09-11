"""Focused regression tests for SSOD phase transitions and resume state."""

import unittest
from types import SimpleNamespace

import torch

from ultralytics.data.dataset_ssod import YOLODataset_ssod
from ultralytics.models.yolo.detect.ssod_train import SSODTrainer
from ultralytics.utils.torch_utils import ModelEMA


class _DatasetStub:
    def __init__(self):
        self.mosaic = True
        self.closed_with = None

    def close_mosaic(self, hyp):
        self.closed_with = hyp


class SSODLifecycleTests(unittest.TestCase):
    def test_warmup_uses_ssod_epoch_length_when_burn_in_is_zero(self):
        self.assertEqual(SSODTrainer._warmup_iterations(3.0, 0, 7, 112), 336)
        self.assertEqual(SSODTrainer._warmup_iterations(0.0, 0, 7, 112), -1)

    def test_warmup_can_cross_phase_boundary(self):
        self.assertEqual(SSODTrainer._warmup_iterations(3.0, 2, 10, 80), 100)

    def test_close_mosaic_reaches_both_loaders(self):
        trainer = SSODTrainer.__new__(SSODTrainer)
        trainer.args = SimpleNamespace(marker="copied")
        labeled, unlabeled = _DatasetStub(), _DatasetStub()
        trainer.train_loader = SimpleNamespace(dataset=labeled)
        trainer.ssod_train_loader = SimpleNamespace(dataset=unlabeled)

        trainer._close_dataloader_mosaic()

        self.assertFalse(labeled.mosaic)
        self.assertFalse(unlabeled.mosaic)
        self.assertEqual(labeled.closed_with.marker, "copied")
        self.assertEqual(unlabeled.closed_with.marker, "copied")

    def test_ssod_dataset_close_zeros_ssod_probabilities(self):
        dataset = YOLODataset_ssod.__new__(YOLODataset_ssod)
        captured = {}
        dataset.build_transforms = lambda hyp: captured.update(vars(hyp)) or "transforms"
        hyp = SimpleNamespace(
            mosaic=1.0,
            mixup=1.0,
            cutmix=1.0,
            copy_paste=1.0,
            mosaic_ssod=1.0,
            mixup_ssod=1.0,
            cutmix_ssod=1.0,
        )

        dataset.close_mosaic(hyp)

        for key in ("mosaic", "mixup", "cutmix", "copy_paste", "mosaic_ssod", "mixup_ssod", "cutmix_ssod"):
            self.assertEqual(captured[key], 0.0)
        self.assertEqual(hyp.mosaic_ssod, 1.0)  # caller configuration was not mutated

    def test_teacher_resume_restores_independent_state(self):
        trainer = SSODTrainer.__new__(SSODTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        saved_teacher = ModelEMA(torch.nn.Linear(2, 1))
        with torch.no_grad():
            for parameter in saved_teacher.ema.parameters():
                parameter.fill_(3.0)
        trainer._resume_teacher_state = saved_teacher.ema
        trainer._resume_teacher_updates = 17
        trainer.burn_in_epochs = 1
        trainer.ema = ModelEMA(trainer.model)

        trainer._restore_teacher_after_resume()

        self.assertEqual(trainer.teacher.updates, 17)
        for parameter in trainer.teacher.ema.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 3.0)))

    def test_post_burn_legacy_resume_is_rejected(self):
        trainer = SSODTrainer.__new__(SSODTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        trainer._resume_teacher_state = None
        trainer._resume_teacher_updates = None
        trainer.burn_in_epochs = 0
        trainer.ema = ModelEMA(trainer.model)

        with self.assertRaisesRegex(RuntimeError, "no serialized SSOD teacher"):
            trainer._restore_teacher_after_resume()

    def test_checkpoint_state_reload_restores_teacher_for_nan_recovery(self):
        trainer = SSODTrainer.__new__(SSODTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        trainer.ema = None
        trainer.teacher = ModelEMA(trainer.model)
        saved_teacher = ModelEMA(torch.nn.Linear(2, 1))
        with torch.no_grad():
            for parameter in saved_teacher.ema.parameters():
                parameter.fill_(5.0)
        checkpoint = {
            "optimizer": None,
            "scaler": None,
            "teacher": saved_teacher.ema,
            "teacher_updates": 23,
            "best_fitness": 0.4,
        }

        trainer._load_checkpoint_state(checkpoint)

        self.assertEqual(trainer.teacher.updates, 23)
        self.assertEqual(trainer.best_fitness, 0.4)
        for parameter in trainer.teacher.ema.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 5.0)))


if __name__ == "__main__":
    unittest.main()
