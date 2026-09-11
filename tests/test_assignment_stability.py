"""Unit tests for deterministic DFL perturbation and assignment identity handling."""

import unittest

import torch

from ultralytics.utils.assignment_stability import (
    assignment_stability,
    dfl_distribution_statistics,
    dfl_supported_box_candidates,
    geometric_box_candidates,
    object_assignment_metrics,
)


class AssignmentStabilityTest(unittest.TestCase):
    def test_sharp_and_flat_distribution_width(self):
        logits = torch.full((2, 4, 16), -12.0)
        logits[0, :, 3] = 12.0
        logits[1] = 0.0
        stats = dfl_distribution_statistics(logits)
        self.assertTrue(torch.equal(stats["width"][0], torch.zeros(4)))
        self.assertTrue(torch.equal(stats["width"][1], torch.full((4,), 13.0)))

    def test_candidate_shape_and_one_edge_changes(self):
        logits = torch.zeros((1, 4, 16))
        boxes = dfl_supported_box_candidates(logits, torch.tensor([[10.0, 10.0]]), torch.tensor([[8.0]]))
        self.assertEqual(boxes.shape, (9, 1, 4))
        baseline = boxes[0, 0]
        for edge in range(4):
            for offset in (1, 2):
                candidate = boxes[edge * 2 + offset, 0]
                changed = ~torch.isclose(candidate, baseline)
                self.assertEqual(int(changed.sum()), 1)
                self.assertTrue(bool(changed[edge]))

    def test_fixed_candidates_use_five_percent_of_object_size(self):
        baseline = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
        boxes = geometric_box_candidates(baseline, "fixed")
        expected_edges = torch.tensor([9.0, 11.0, 18.0, 22.0, 29.0, 31.0, 58.0, 62.0])
        observed = torch.stack([boxes[i, 0, (i - 1) // 2] for i in range(1, 9)])
        self.assertTrue(torch.equal(boxes[0], baseline))
        self.assertTrue(torch.equal(observed, expected_edges))

    def test_width_matched_is_symmetric_in_dfl_distance_space(self):
        expectation = torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        width = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
        anchor = torch.tensor([[10.0, 10.0]])
        stride = torch.tensor([[2.0]])
        baseline = torch.tensor([[12.0, 10.0, 32.0, 34.0]])
        boxes = geometric_box_candidates(
            baseline,
            "width_matched",
            dfl_width=width,
            dfl_expectation=expectation,
            anchor_points=anchor,
            stride_tensor=stride,
            reg_max=16,
        )
        self.assertTrue(torch.equal(boxes[0], baseline))
        # Each pair differs only on its selected decoded edge and is separated by width * stride.
        for edge in range(4):
            low, high = boxes[1 + 2 * edge, 0], boxes[2 + 2 * edge, 0]
            changed = ~torch.isclose(low, high)
            self.assertEqual(int(changed.sum()), 1)
            self.assertTrue(bool(changed[edge]))
            self.assertAlmostEqual(abs(float(high[edge] - low[edge])), float(width[0, edge] * stride[0, 0]))

    def test_background_flip_and_identity_switch_are_distinct(self):
        ids = torch.tensor([[[0, -1, 1]], [[0, 0, 2]], [[-1, -1, 1]]])
        stability, ambiguous, switches, foreground_frequency = assignment_stability(ids)
        self.assertTrue(torch.allclose(stability, torch.tensor([[2 / 3, 0.0, 2 / 3]])))
        self.assertEqual(ambiguous.tolist(), [[False, True, False]])
        self.assertEqual(switches.tolist(), [[False, False, True]])
        self.assertTrue(torch.allclose(foreground_frequency, torch.tensor([[2 / 3, 1 / 3, 1.0]])))

    def test_object_metrics_do_not_count_foreground_to_background_as_identity_switch(self):
        ids = torch.tensor([[[0, 0]], [[0, -1]], [[1, 0]]])
        metrics = object_assignment_metrics(ids, 2)
        self.assertEqual(metrics["identity_switches"].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
