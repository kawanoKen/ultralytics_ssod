"""Regression tests for LabelMatch Adaptive Confidence Threshold (ACT)."""

import unittest

import numpy as np
import torch

from ultralytics.utils.labelmatch_act import ACTManager, act_threshold, labeled_boxes_per_image
from ultralytics.utils.loss_ssod import EfficientTeacherLoss


def selector(reliable, candidate=None):
    loss = EfficientTeacherLoss.__new__(EfficientTeacherLoss)
    loss.conf_threshold_low = 0.3
    loss.conf_threshold_high = 0.6
    loss.use_loc_conf = False
    loss.two_axis_selection = False
    loss.class_conf_thresholds = reliable
    loss.class_candidate_thresholds = candidate
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

    def test_manager_candidate_and_reliable_thresholds(self):
        manager = ACTManager(np.array([2.5, 0.0]), reliable_ratio=0.2)
        scores = [torch.linspace(1.0, 0.01, 100), torch.tensor([0.95])]
        t_c, t_r = manager.update_from_scores(scores, [100, 1], num_images=20, iteration=0, epoch=0)
        # class 0: K = round(2.5 * 20) = 50 candidates, K_r = round(0.2 * 50) = 10 reliable.
        self.assertAlmostEqual(float(t_c[0]), float(scores[0][49]), places=6)
        self.assertAlmostEqual(float(t_r[0]), float(scores[0][9]), places=6)
        self.assertEqual(float(t_c[1]), 1.0)  # K = 0
        self.assertTrue(manager.should_update(1000))
        self.assertFalse(manager.should_update(500))

    def test_ratio_one_collapses_to_single_threshold(self):
        manager = ACTManager(np.array([1.0]), reliable_ratio=1.0)
        t_c, t_r = manager.update_from_scores([torch.tensor([0.9, 0.5, 0.2])], [3], 2, 0, 0)
        self.assertEqual(float(t_c[0]), float(t_r[0]))


class ACTLossMaskTests(unittest.TestCase):
    def test_reliable_uncertain_discard_bands(self):
        loss = selector(reliable=torch.tensor([0.5, 0.02]), candidate=torch.tensor([0.2, 0.007]))
        conf = torch.tensor([[0.55], [0.45], [0.15], [0.03], [0.01], [0.005]])
        cls = torch.tensor([[0], [0], [0], [1], [1], [1]])
        reliable, uncertain = loss._get_reliable_and_unreliable_mask(conf, None, cls)
        self.assertEqual(reliable.tolist(), [True, False, False, True, False, False])
        # [t_c, t_c^r) is ignored; tau_low=0.3 plays no role under ACT.
        self.assertEqual(uncertain.tolist(), [False, True, False, False, True, False])

    def test_fixed_threshold_path_unchanged(self):
        loss = selector(None)
        conf = torch.tensor([[0.7], [0.4], [0.2]])
        reliable, unreliable = loss._get_reliable_and_unreliable_mask(conf, None, torch.zeros(3, 1))
        self.assertEqual(reliable.tolist(), [True, False, False])
        self.assertEqual(unreliable.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
