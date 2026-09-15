"""Regression tests for paper-style four-threshold pseudo-label selection."""

import unittest

import torch

from ultralytics.utils.loss_ssod import EfficientTeacherLoss


def selector():
    """Construct only the state required by the pure mask helper."""
    loss = EfficientTeacherLoss.__new__(EfficientTeacherLoss)
    loss.conf_threshold_low = 0.3
    loss.conf_threshold_high = 0.6
    loss.two_axis_selection = True
    loss.loc_conf_threshold_low = 0.6
    loss.loc_conf_threshold_high = 0.8
    return loss


class TwoAxisSelectionTests(unittest.TestCase):
    def test_kitti_four_threshold_partition(self):
        # reliable, background, low-cls/high-loc uncertain, high-cls/mid-loc uncertain,
        # and a fully intermediate point, respectively.
        cls = torch.tensor([[0.60], [0.30], [0.20], [0.70], [0.40]])
        loc = torch.tensor([[0.80], [0.60], [0.90], [0.70], [0.70]])

        reliable, uncertain = selector()._get_reliable_and_unreliable_mask(cls, loc)

        self.assertEqual(reliable.tolist(), [True, False, False, False, False])
        self.assertEqual(uncertain.tolist(), [False, False, True, True, True])

    def test_two_axis_requires_localization_confidence(self):
        cls = torch.tensor([[0.8]])
        with self.assertRaisesRegex(ValueError, "requires unlabeled_loc_conf"):
            selector()._get_reliable_and_unreliable_mask(cls, None)

    def test_legacy_mode_is_unchanged(self):
        loss = selector()
        loss.two_axis_selection = False
        loss.use_loc_conf = False
        cls = torch.tensor([[0.7], [0.4], [0.2]])
        reliable, unreliable = loss._get_reliable_and_unreliable_mask(cls, None)
        self.assertEqual(reliable.tolist(), [True, False, False])
        self.assertEqual(unreliable.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
