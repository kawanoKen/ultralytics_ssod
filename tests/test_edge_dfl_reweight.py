"""Unit tests for selector-independent edge-wise DFL reweighting."""

from __future__ import annotations

import torch

from ultralytics.utils.edge_dfl_reweight import (
    clip_xyxy,
    dfl_per_edge_loss,
    make_edge_dfl_weights,
    oracle_selected_edges,
    reduce_weighted_dfl,
    select_dfl_low_confidence_edges,
)


def _oracle_inputs(pseudo_box, gt_box, width=10, height=10):
    pseudo = torch.tensor([[pseudo_box]], dtype=torch.float32)
    gt = torch.tensor([[gt_box]], dtype=torch.float32)
    cls = torch.zeros((1, 1, 1), dtype=torch.float32)
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    return pseudo, cls, valid, gt, cls.clone(), valid.clone(), height, width


def test_weight_one_is_baseline_equivalent_for_both_selectors():
    torch.manual_seed(0)
    reg_max = 16
    pred = torch.randn(3, 4, reg_max, requires_grad=True)
    target = torch.tensor([[1.2, 2.4, 3.6, 4.8], [0.2, 6.4, 7.1, 2.0], [5.3, 1.8, 4.2, 6.6]])
    per_edge = dfl_per_edge_loss(pred, target, reg_max)
    target_weight = torch.tensor([[0.3], [0.7], [0.5]])
    baseline = reduce_weighted_dfl(per_edge, target_weight, 1.5, torch.ones_like(per_edge))
    for selected in (
        torch.tensor([[False, False, False, True], [True, False, True, False], [False, True, False, False]]),
        select_dfl_low_confidence_edges(torch.tensor([[0.9, 0.9, 0.9, 0.2]]).expand(3, -1), 0.6),
    ):
        weights = make_edge_dfl_weights(selected, selected_weight=1.0, normalize=True)
        actual = reduce_weighted_dfl(per_edge, target_weight, 1.5, weights.normalized)
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_oracle_selects_only_wrong_bottom_edge():
    args = _oracle_inputs([1, 2, 9, 10], [1, 2, 9, 8])
    selected, matched = oracle_selected_edges(*args, error_threshold=0.10)
    assert matched.item()
    assert selected.tolist() == [[[False, False, False, True]]]
    weights = make_edge_dfl_weights(selected[matched.squeeze(-1)], selected_weight=0.0, normalize=False)
    assert weights.normalized.tolist() == [[1.0, 1.0, 1.0, 0.0]]


def test_dfl_selector_selects_only_low_bottom_edge():
    confidence = torch.tensor([[0.9, 0.8, 0.7, 0.2]])
    assert select_dfl_low_confidence_edges(confidence, 0.6).tolist() == [[False, False, False, True]]


def test_weight_normalization_for_five_and_twenty():
    selected = torch.tensor([[False, False, False, True]])
    for weight, expected in ((5.0, [0.5, 0.5, 0.5, 2.5]), (20.0, [4 / 23, 4 / 23, 4 / 23, 80 / 23])):
        result = make_edge_dfl_weights(selected, weight, normalize=True)
        torch.testing.assert_close(result.normalized, torch.tensor([expected]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(result.normalized.mean(), torch.tensor(1.0))


def test_gt_clip_and_horizontal_flip_swap_left_right_selection():
    clipped = clip_xyxy(torch.tensor([[-2.0, -3.0, 12.0, 15.0]]), 10, 10)
    assert clipped.tolist() == [[0.0, 0.0, 10.0, 10.0]]
    original = oracle_selected_edges(*_oracle_inputs([2, 0, 10, 10], [0, 0, 10, 10]), error_threshold=0.10)[0]
    flipped = oracle_selected_edges(*_oracle_inputs([0, 0, 8, 10], [0, 0, 10, 10]), error_threshold=0.10)[0]
    assert original.tolist() == [[[True, False, False, False]]]
    assert flipped.tolist() == [[[False, False, True, False]]]


def test_empty_and_all_zero_cases_are_finite():
    empty = make_edge_dfl_weights(torch.zeros((0, 4), dtype=torch.bool), 20.0, normalize=True)
    assert empty.normalized.numel() == 0
    all_selected = make_edge_dfl_weights(torch.ones((2, 4), dtype=torch.bool), 0.0, normalize=True)
    assert all_selected.normalization_zero_sum
    assert torch.count_nonzero(all_selected.normalized) == 0
    pseudo = torch.zeros((1, 0, 4))
    cls = torch.zeros((1, 0, 1))
    valid = torch.zeros((1, 0, 1), dtype=torch.bool)
    selected, matched = oracle_selected_edges(pseudo, cls, valid, pseudo, cls, valid, 10, 10, 0.1)
    assert selected.numel() == matched.numel() == 0
