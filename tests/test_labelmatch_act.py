"""Regression tests for LabelMatch Adaptive Confidence Threshold (ACT)."""

import unittest

import numpy as np
import torch

from ultralytics.utils.labelmatch_act import ACTManager, act_threshold, labeled_boxes_per_image
from ultralytics.utils.loss_ssod import EfficientTeacherLoss


def selector(thresholds):
    loss = EfficientTeacherLoss.__new__(EfficientTeacherLoss)
    loss.conf_threshold_low = 0.3
    loss.conf_threshold_high = 0.6
    loss.use_loc_conf = False
    loss.two_axis_selection = False
    loss.class_conf_thresholds = thresholds
    return loss


class ACTThresholdTests(unittest.TestCase):
    def test_kth_highest_is_one_indexed(self):
        scores = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.3])
        self.assertAlmostEqual(act_threshold(scores, 1, 0.001), 0.9, places=6)
        self.assertAlmostEqual(act_threshold(scores, 3, 0.001), 0.5, places=6)
        self.assertAlmostEqual(act_threshold(scores, 5, 0.001), 0.1, places=6)

    def test_boundary_conditions(self):
        scores = torch.tensor([0.8, 0.4])
        self.assertEqual(act_threshold(scores, 0, 0.001), 1.0)
        self.assertEqual(act_threshold(scores, 3, 0.001), 0.001)
        self.assertEqual(act_threshold(torch.empty(0), 2, 0.001), 0.001)
        with_nan = torch.tensor([float("nan"), 0.8, float("inf"), 0.4])
        self.assertAlmostEqual(act_threshold(with_nan, 2, 0.001), 0.4, places=6)

    def test_rho_counts_background_images(self):
        labels = [{"cls": np.array([[0], [0], [1]])}, {"cls": np.zeros((0, 1))}]
        np.testing.assert_allclose(labeled_boxes_per_image(labels, 3), [1.0, 0.5, 0.0])

    def test_manager_matches_target_count(self):
        manager = ACTManager(np.array([2.0, 0.0]))
        scores = [torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), torch.tensor([0.95])]
        manager.update_from_scores(scores, [6, 1], num_images=2, iteration=0, epoch=0)
        self.assertAlmostEqual(float(manager.thresholds[0]), 0.6, places=6)  # K = round(2.0 * 2) = 4
        self.assertEqual(float(manager.thresholds[1]), 1.0)  # K = 0
        self.assertTrue(manager.should_update(1000))
        self.assertFalse(manager.should_update(500))


class ACTLossMaskTests(unittest.TestCase):
    def test_class_wise_threshold_without_ignore_band(self):
        loss = selector(torch.tensor([0.5, 0.007]))
        conf = torch.tensor([[0.55], [0.45], [0.25], [0.008], [0.006]])
        cls = torch.tensor([[0], [0], [0], [1], [1]])
        reliable, unreliable = loss._get_reliable_and_unreliable_mask(conf, None, cls)
        self.assertEqual(reliable.tolist(), [True, False, False, True, False])
        self.assertEqual(unreliable.tolist(), [False] * 5)  # tau_low=0.3 plays no role under ACT

    def test_fixed_threshold_path_unchanged(self):
        loss = selector(None)
        conf = torch.tensor([[0.7], [0.4], [0.2]])
        reliable, unreliable = loss._get_reliable_and_unreliable_mask(conf, None, torch.zeros(3, 1))
        self.assertEqual(reliable.tolist(), [True, False, False])
        self.assertEqual(unreliable.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
