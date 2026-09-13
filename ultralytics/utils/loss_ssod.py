from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors, bbox2dist
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.assignment_stability import assignment_stability
from ultralytics.utils.edge_dfl_reweight import (
    dfl_per_edge_loss,
    make_edge_dfl_weights,
    oracle_selected_edges,
    reduce_weighted_dfl,
    select_dfl_low_confidence_edges,
)
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

# Every negative-supervision bin dict (see EfficientTeacherLoss._compute_negative_supervision_bins)
# must carry the same keys every call, empty bins included -- the diagnostics CSV writer infers its
# header from the first row it ever sees, so a row with fewer keys than an earlier one raises a
# ValueError instead of just leaving cells blank.
_EMPTY_NEGSUP_BIN = {
    "count": 0,
    "teacher_score_mean": None,
    "selected_as_pseudo_label_rate": None,
    "student_score_mean": None,
    "student_target_score_mean": None,
    "student_is_positive_rate": None,
    "student_is_negative_rate": None,
    "classification_loss_mean": None,
}


class EfficientTeacherLoss(v8DetectionLoss):
    def __init__(
        self,
        model,
        conf_threshold_high=0.6,
        conf_threshold_low=0.1,
        use_loc_conf=True,
        loc_conf_threshold=0.6,
        use_edge_conf=False,
        edge_conf_threshold=0.6,
        edge_conf_mask_mode="selected",
        edge_dfl_reweight=False,
        edge_dfl_selector="dfl",
        edge_dfl_weight=1.0,
        edge_dfl_normalize=True,
        oracle_edge_error_threshold=0.10,
        assignment_stability_method="off",
        skip_zero_pseudo_cls_loss=False,
        cls_loss_denom="target_score_sum",
        ema_denom_beta=0.9,
        positive_only_cls_loss=False,
    ):
        super().__init__(model)
        self.conf_threshold_high = conf_threshold_high
        self.conf_threshold_low = conf_threshold_low
        self.use_loc_conf = use_loc_conf
        self.loc_conf_threshold = loc_conf_threshold
        self.use_edge_conf = use_edge_conf
        self.edge_conf_threshold = edge_conf_threshold
        if edge_conf_mask_mode not in {"selected", "random"}:
            raise ValueError(f"edge_conf_mask_mode must be selected or random, got {edge_conf_mask_mode!r}")
        self.edge_conf_mask_mode = edge_conf_mask_mode
        self.edge_dfl_reweight = edge_dfl_reweight
        if edge_dfl_selector not in {"oracle", "dfl"}:
            raise ValueError(f"edge_dfl_selector must be oracle or dfl, got {edge_dfl_selector!r}")
        if edge_dfl_weight < 0:
            raise ValueError(f"edge_dfl_weight must be non-negative, got {edge_dfl_weight}")
        if oracle_edge_error_threshold < 0:
            raise ValueError(f"oracle_edge_error_threshold must be non-negative, got {oracle_edge_error_threshold}")
        self.edge_dfl_selector = edge_dfl_selector
        self.edge_dfl_weight = float(edge_dfl_weight)
        self.edge_dfl_normalize = edge_dfl_normalize
        self.oracle_edge_error_threshold = float(oracle_edge_error_threshold)
        self.edge_dfl_zero_sum_batches = 0
        if assignment_stability_method not in {"off", "r1", "r2"}:
            raise ValueError(f"assignment_stability_method must be off, r1, or r2, got {assignment_stability_method!r}")
        self.assignment_stability_method = assignment_stability_method
        # Single isolated experimental switch (see scripts/crowdhuman/*_zero_pseudo_skip*): when a
        # batch has zero adopted pseudo-labels, target_scores_sum_reliable falls to its max(...,1)
        # floor and the classification BCE -- summed over EVERY anchor against an all-zero target
        # -- gets divided by 1 instead of a real normalizer, i.e. the whole anchor grid is trained
        # as a hard negative with no normalization. This flag skips that update instead, to test
        # whether it is the trigger for the observed Detect-head/BN collapse. Everything else
        # (box/dfl loss, which already naturally skip via `if fg_mask_reliable.sum():`) is
        # unaffected.
        self.skip_zero_pseudo_cls_loss = skip_zero_pseudo_cls_loss
        # "target_score_sum" (default): divide the unsupervised classification BCE sum by
        # max(sum_target_scores_unsup, 1), as in the original design. "fixed": divide by a
        # content-independent reference (batch_size * anchors_per_image) instead -- see
        # scripts/crowdhuman/normalization_ablation.py, which found this content-dependent
        # denominator inflates ||g_u|| by 3-5x whenever few pseudo-labels are adopted (small
        # denominator), and that capping the resulting gradient scale (not its direction) mostly
        # restores alignment between the combined update and the supervised gradient. "ema":
        # Detectron2-style smoothing -- divide by an exponential moving average of
        # sum_target_scores_unsup across steps (D_t = beta*D_{t-1} + (1-beta)*S_t) instead of the
        # raw current-step value, so a single low-pseudo-label batch doesn't spike the divisor
        # down to near zero. Box/DFL loss normalization is unaffected in every mode.
        if cls_loss_denom not in {"target_score_sum", "fixed", "ema"}:
            raise ValueError(f"cls_loss_denom must be target_score_sum, fixed, or ema, got {cls_loss_denom!r}")
        self.cls_loss_denom = cls_loss_denom
        self.ema_denom_beta = ema_denom_beta
        self.ema_target_score_sum = None
        # Experiment 1 uses an oracle-selected subset of boxes.  In that diagnostic
        # setting, anchors which are not assigned to one of the selected boxes must
        # contribute neither a positive nor a background classification target.
        self.positive_only_cls_loss = positive_only_cls_loss
        self.last_stability_stats = {}

    def __call__(
        self,
        preds,
        unlabeled_bboxes,
        unlabeled_cls,
        unlabeled_conf,
        unlabeled_batch_idx,
        unlabeled_loc_conf=None,
        unlabeled_edge_conf=None,
        unlabeled_candidate_bboxes=None,
        oracle_gt_bboxes=None,
        oracle_gt_cls=None,
        oracle_gt_batch_idx=None,
        compute_extra_diag=False,
    ):
        """
        Args:
            preds:
            unlabeled_bboxes: [num_unlabeled_boxes, 4]
            unlabeled_cls: [num_unlabeled_boxes, 1]
            unlabeled_conf: [num_unlabeled_boxes, 1]
            unlabeled_batch_idx: [num_unlabeled_boxes, 1]
            unlabeled_loc_conf: [num_unlabeled_boxes, 1], optional DFL box-level localization
                confidence (see ultralytics.utils.dfl_confidence.localization_confidence). Only
                used to further restrict the reliable set when `self.use_loc_conf` is True.
            unlabeled_edge_conf: [num_unlabeled_boxes, 4], optional per-edge (l,t,r,b) DFL
                confidence. When `self.use_edge_conf` is True, the DFL loss for a reliable box's
                individual edge is skipped if that edge's confidence is below
                `self.edge_conf_threshold` -- unlike `use_loc_conf` (which gates the whole box),
                this keeps training the box's other, more confident edges instead of dropping it.
                With ``edge_conf_mask_mode="random"``, the same per-edge selected counts are
                sampled at random among eligible foreground anchors.
            unlabeled_candidate_bboxes: [num_unlabeled_boxes, 9, 4] normalized XYWH boxes. Candidate
                zero is the expectation and the other eight are deterministic one-edge Q10/Q90 boxes.
            oracle_gt_bboxes / oracle_gt_cls / oracle_gt_batch_idx: transformed dataset GT in the
                student loss coordinate system. Used exclusively by the Oracle edge selector when
                ``edge_dfl_reweight=True`` and ``edge_dfl_selector="oracle"``.
        """

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        reliable_mask, unreliable_mask = self._get_reliable_and_unreliable_mask(unlabeled_conf, unlabeled_loc_conf)

        reliable_targets = torch.cat((unlabeled_batch_idx[reliable_mask], unlabeled_cls[reliable_mask], unlabeled_bboxes[reliable_mask]), 1)
        reliable_targets = self.preprocess(reliable_targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        reliable_gt_labels, reliable_gt_bboxes = reliable_targets.split((1, 4), 2)  # cls, xyxy
        reliable_mask_gt = reliable_gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        unreliable_targets = torch.cat((unlabeled_batch_idx[unreliable_mask], unlabeled_cls[unreliable_mask], unlabeled_bboxes[unreliable_mask]), 1)
        unreliable_targets = self.preprocess(unreliable_targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        unreliable_gt_labels, unreliable_gt_bboxes = unreliable_targets.split((1, 4), 2)  # cls, xyxy
        unreliable_mask_gt = unreliable_gt_bboxes.sum(2, keepdim=True).gt_(0.0)


        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        assigner_start = time.perf_counter()
        _, target_bboxes_reliable, target_scores_reliable, fg_mask_reliable, target_gt_idx_reliable = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(reliable_gt_bboxes.dtype),
            anchor_points * stride_tensor,
            reliable_gt_labels,
            reliable_gt_bboxes,
            reliable_mask_gt,
        )
        # TaskAlignedAssigner's no-object fast path inherits the score dtype for this mask.
        fg_mask_reliable = fg_mask_reliable.bool()
        stability = torch.ones_like(fg_mask_reliable, dtype=pred_scores.dtype)
        ambiguous_negative = torch.zeros_like(fg_mask_reliable, dtype=torch.bool)
        identity_switch = torch.zeros_like(fg_mask_reliable, dtype=torch.bool)
        if self.assignment_stability_method != "off":
            if unlabeled_candidate_bboxes is None:
                raise ValueError("assignment stability requires unlabeled_candidate_bboxes")
            candidate_assignments = [torch.where(fg_mask_reliable, target_gt_idx_reliable, -1)]
            reliable_candidates = unlabeled_candidate_bboxes[reliable_mask]
            for candidate_idx in range(1, reliable_candidates.shape[1]):
                candidate_targets = torch.cat(
                    (
                        unlabeled_batch_idx[reliable_mask],
                        unlabeled_cls[reliable_mask],
                        reliable_candidates[:, candidate_idx],
                    ),
                    1,
                )
                candidate_targets = self.preprocess(candidate_targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                candidate_labels, candidate_bboxes = candidate_targets.split((1, 4), 2)
                candidate_mask_gt = candidate_bboxes.sum(2, keepdim=True).gt_(0.0)
                _, _, _, candidate_fg, candidate_gt_idx = self.assigner(
                    pred_scores.detach().sigmoid(),
                    (pred_bboxes.detach() * stride_tensor).type(candidate_bboxes.dtype),
                    anchor_points * stride_tensor,
                    candidate_labels,
                    candidate_bboxes,
                    candidate_mask_gt,
                )
                candidate_fg = candidate_fg.bool()
                candidate_assignments.append(torch.where(candidate_fg, candidate_gt_idx, -1))
            all_assignments = torch.stack(candidate_assignments)
            stability, ambiguous_negative, identity_switch, foreground_frequency = assignment_stability(all_assignments)
            baseline_stability = stability[fg_mask_reliable]
            histogram = torch.bincount((baseline_stability * 9).round().to(torch.long), minlength=10)
            self.last_stability_stats = {
                "pseudo_label_count": int(reliable_mask.sum()),
                "positive_count": int(fg_mask_reliable.sum()),
                "negative_count": int((~fg_mask_reliable & ~ambiguous_negative).sum()),
                "ambiguous_negative_count": int(ambiguous_negative.sum()),
                "identity_switch_count": int(identity_switch.sum()),
                "mean_stability": float(baseline_stability.mean()) if baseline_stability.numel() else 0.0,
                "median_stability": float(baseline_stability.median()) if baseline_stability.numel() else 0.0,
                "effective_regression_weight_sum": float(
                    (target_scores_reliable.sum(-1)[fg_mask_reliable] * baseline_stability).sum()
                ),
                "mean_foreground_frequency": float(foreground_frequency.mean()),
                "stability_histogram_0_to_9": histogram.tolist(),
            }
        else:
            self.last_stability_stats = {
                "pseudo_label_count": int(reliable_mask.sum()),
                "positive_count": int(fg_mask_reliable.sum()),
                "negative_count": int((~fg_mask_reliable).sum()),
                "ambiguous_negative_count": 0,
                "identity_switch_count": 0,
                "mean_stability": 1.0 if fg_mask_reliable.any() else 0.0,
                "median_stability": 1.0 if fg_mask_reliable.any() else 0.0,
                "effective_regression_weight_sum": float(target_scores_reliable.sum(-1)[fg_mask_reliable].sum()),
                "mean_foreground_frequency": float(fg_mask_reliable.to(torch.float32).mean()),
                "stability_histogram_0_to_9": [0] * 9 + [int(fg_mask_reliable.sum())],
            }
        _, _, _, fg_mask_unreliable, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(unreliable_gt_bboxes.dtype),
            anchor_points * stride_tensor,
            unreliable_gt_labels,
            unreliable_gt_bboxes,
            unreliable_mask_gt,
        )
        self.last_stability_stats["assigner_seconds"] = time.perf_counter() - assigner_start

        ignore_mask = fg_mask_unreliable.bool() & ~fg_mask_reliable.bool()
        if self.assignment_stability_method == "r2":
            ignore_mask |= ambiguous_negative
        target_scores_sum_reliable = max((target_scores_reliable.sum(-1)*((~ignore_mask))).sum(), 1)

        # ---- Training-time-only diagnostics (see ultralytics.utils.ssod_diagnostics) ----
        # Cheap per-call bookkeeping: how many pseudo-boxes actually produced a positive anchor,
        # not just how many pseudo-boxes were adopted (num_pseudo_boxes can stay flat while
        # positive-anchor count collapses -- that dissociation is exactly what this is meant to
        # surface later).
        self.last_assignment_stats = self._compute_assignment_stats(
            target_gt_idx_reliable, fg_mask_reliable, reliable_mask_gt, target_scores_reliable, reliable_mask
        )
        # Training-time diagnostics for the per-edge DFL ablation. The random mode keeps the
        # selected counts from the DFL mask and samples positions independently of confidence.
        # Initialize these fields for empty-foreground batches as well, keeping CSV columns stable.
        self.last_edge_mask_stats = {
            "edge_mask_mode": "bypassed_by_edge_dfl_reweight" if self.edge_dfl_reweight else (
                self.edge_conf_mask_mode if self.use_edge_conf else "disabled"
            ),
            "edge_mask_eligible_edges": 0,
            "edge_mask_reliable_boxes": int(reliable_mask.sum()),
            "edge_mask_target_total": 0,
            "edge_mask_target_L": 0,
            "edge_mask_target_T": 0,
            "edge_mask_target_R": 0,
            "edge_mask_target_B": 0,
            "edge_mask_applied_total": 0,
            "edge_mask_applied_L": 0,
            "edge_mask_applied_T": 0,
            "edge_mask_applied_R": 0,
            "edge_mask_applied_B": 0,
        }
        self.last_assignment_stats.update(self.last_edge_mask_stats)
        self.last_edge_dfl_stats = self._empty_edge_dfl_reweight_stats(reliable_mask)
        self.last_assignment_stats.update(self.last_edge_dfl_stats)
        self.last_diag = None
        if compute_extra_diag:
            self.last_diag = self._compute_negative_supervision_bins(
                pred_scores,
                anchor_points,
                stride_tensor,
                imgsz,
                fg_mask_reliable,
                ignore_mask,
                target_scores_reliable,
                unlabeled_bboxes,
                unlabeled_cls,
                unlabeled_batch_idx,
                unlabeled_conf,
                reliable_mask,
            )

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores_reliable.to(dtype)).sum() / target_scores_sum_reliable  # BCE
        zero_pseudo_batch = fg_mask_reliable.sum() == 0
        self.last_assignment_stats["cls_loss_skipped"] = bool(self.skip_zero_pseudo_cls_loss and zero_pseudo_batch)
        skip_this_batch = self.skip_zero_pseudo_cls_loss and zero_pseudo_batch
        if skip_this_batch and not compute_extra_diag:
            # This batch adopted zero pseudo-labels as positives, so target_scores_reliable is
            # all-zero and target_scores_sum_reliable sits at its max(...,1) floor -- computing
            # the BCE below would train every anchor in the grid as a hard negative, summed over
            # the whole grid and divided by 1 instead of a real normalizer. Skip it entirely.
            loss[1] = torch.zeros((), device=self.device, dtype=dtype)
        else:
            cls_loss_per_anchor = self.bce(pred_scores, target_scores_reliable.to(dtype)).sum(-1)
            if self.positive_only_cls_loss:
                # Do not turn omitted pseudo objects/regions into hard negatives.  The
                # box/DFL terms already use fg_mask_reliable, so this makes all three
                # unlabeled terms selected-positive-only.
                cls_loss_numerator = cls_loss_per_anchor[fg_mask_reliable].sum()
            else:
                cls_loss_numerator = (cls_loss_per_anchor * (~ignore_mask)).sum()
            if compute_extra_diag:
                # Retained (differentiable) so callers can build normalization-ablation
                # counterfactual losses from the exact same numerator/model state -- see
                # scripts/crowdhuman/normalization_ablation.py.
                self.last_cls_loss_numerator = cls_loss_numerator
                self.last_target_scores_sum_reliable = float(target_scores_sum_reliable)
            if skip_this_batch:
                loss[1] = torch.zeros((), device=self.device, dtype=dtype)
            elif self.cls_loss_denom == "fixed":
                # Content-independent reference denominator: batch_size * anchors_per_image.
                # Box/DFL loss below still use target_scores_sum_reliable unchanged.
                cls_denom = pred_scores.shape[0] * pred_scores.shape[1]
                loss[1] = cls_loss_numerator / cls_denom
            elif self.cls_loss_denom == "ema":
                # Detectron2-style smoothed denominator: an EMA of this step's target-score mass,
                # not the raw current-step value, so one low-pseudo-label batch doesn't spike the
                # divisor down. State persists across calls (one EfficientTeacherLoss instance
                # lives for the whole training run).
                current_s_t = float(target_scores_sum_reliable)
                self.ema_target_score_sum = (
                    current_s_t
                    if self.ema_target_score_sum is None
                    else self.ema_denom_beta * self.ema_target_score_sum + (1 - self.ema_denom_beta) * current_s_t
                )
                loss[1] = cls_loss_numerator / max(self.ema_target_score_sum, 1.0)
            else:
                loss[1] = cls_loss_numerator / target_scores_sum_reliable
        # Bbox loss
        if fg_mask_reliable.sum():
            if self.edge_dfl_reweight:
                if self.assignment_stability_method != "off":
                    raise ValueError("edge_dfl_reweight cannot be combined with assignment_stability_method")
                selected_per_pseudo, oracle_matched = self._selected_edges_for_reliable_pseudos(
                    unlabeled_batch_idx,
                    unlabeled_edge_conf,
                    reliable_mask,
                    reliable_gt_bboxes,
                    reliable_gt_labels,
                    reliable_mask_gt,
                    imgsz,
                    oracle_gt_bboxes,
                    oracle_gt_cls,
                    oracle_gt_batch_idx,
                )
                selected_edge_mask = self._map_pseudo_edge_values_to_foreground(
                    selected_per_pseudo, target_gt_idx_reliable, fg_mask_reliable
                )
                edge_weight_result = make_edge_dfl_weights(
                    selected_edge_mask, self.edge_dfl_weight, self.edge_dfl_normalize
                )
                if edge_weight_result.normalization_zero_sum:
                    self.edge_dfl_zero_sum_batches += 1
                self.last_edge_dfl_stats = self._edge_dfl_reweight_stats(
                    selected_edge_mask,
                    edge_weight_result,
                    pred_distri,
                    anchor_points,
                    target_bboxes_reliable,
                    target_scores_reliable,
                    target_scores_sum_reliable,
                    fg_mask_reliable,
                    oracle_matched,
                    reliable_mask,
                )
                self.last_assignment_stats.update(self.last_edge_dfl_stats)
                # Exact baseline path is intentionally delegated to BboxLoss. This makes w=1
                # bit-identical to the ordinary SSOD loss even when a selector is active.
                if self.edge_dfl_weight == 1.0:
                    loss[0], loss[2] = self.bbox_loss(
                        pred_distri,
                        pred_bboxes,
                        anchor_points,
                        target_bboxes_reliable / stride_tensor,
                        target_scores_reliable,
                        target_scores_sum_reliable,
                        fg_mask_reliable,
                    )
                else:
                    loss[0], loss[2] = self._bbox_loss_with_edge_weights(
                        pred_distri,
                        pred_bboxes,
                        anchor_points,
                        target_bboxes_reliable / stride_tensor,
                        target_scores_reliable,
                        target_scores_sum_reliable,
                        fg_mask_reliable,
                        edge_weight_result.normalized,
                    )
            elif self.assignment_stability_method in {"r1", "r2"}:
                loss[0], loss[2] = self._bbox_loss_with_stability(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes_reliable / stride_tensor,
                    target_scores_reliable,
                    fg_mask_reliable,
                    stability,
                )
            elif self.use_edge_conf and unlabeled_edge_conf is not None:
                padded_edge_conf = self._pad_per_image(
                    unlabeled_batch_idx[reliable_mask], unlabeled_edge_conf[reliable_mask], batch_size
                )  # (b, n_max_boxes, 4), aligned slot-for-slot with reliable_gt_bboxes
                # TaskAlignedAssigner.forward returns target_gt_idx as a LOCAL per-image index
                # (0..n_max_boxes-1); the batch offset it applies internally in get_targets() is
                # not reflected in what it returns, so it must be re-added here before indexing
                # into the flattened (b*n_max_boxes, 4) padded tensor.
                n_max_boxes = padded_edge_conf.shape[1]
                batch_ind = torch.arange(batch_size, device=target_gt_idx_reliable.device).unsqueeze(-1)
                flat_gt_idx = target_gt_idx_reliable + batch_ind * n_max_boxes  # (b, h*w)
                target_edge_conf = padded_edge_conf.view(-1, 4)[flat_gt_idx]  # (b, h*w, 4)
                selected_edge_mask = target_edge_conf[fg_mask_reliable] >= self.edge_conf_threshold  # (N_fg, 4)
                edge_mask = selected_edge_mask
                if self.edge_conf_mask_mode == "random":
                    # DFL confidence determines counts only; randperm determines positions.
                    edge_mask = torch.zeros_like(selected_edge_mask)
                    n_foreground = selected_edge_mask.shape[0]
                    for edge_idx in range(4):
                        n_selected = int(selected_edge_mask[:, edge_idx].sum().item())
                        if n_selected:
                            random_idx = torch.randperm(n_foreground, device=selected_edge_mask.device)[:n_selected]
                            edge_mask[random_idx, edge_idx] = True
                target_counts = selected_edge_mask.sum(0).to(torch.long)
                applied_counts = edge_mask.sum(0).to(torch.long)
                self.last_edge_mask_stats = {
                    "edge_mask_mode": self.edge_conf_mask_mode,
                    "edge_mask_eligible_edges": int(selected_edge_mask.numel()),
                    "edge_mask_reliable_boxes": int(reliable_mask.sum()),
                    "edge_mask_target_total": int(selected_edge_mask.sum()),
                    "edge_mask_target_L": int(target_counts[0]),
                    "edge_mask_target_T": int(target_counts[1]),
                    "edge_mask_target_R": int(target_counts[2]),
                    "edge_mask_target_B": int(target_counts[3]),
                    "edge_mask_applied_total": int(edge_mask.sum()),
                    "edge_mask_applied_L": int(applied_counts[0]),
                    "edge_mask_applied_T": int(applied_counts[1]),
                    "edge_mask_applied_R": int(applied_counts[2]),
                    "edge_mask_applied_B": int(applied_counts[3]),
                }
                self.last_assignment_stats.update(self.last_edge_mask_stats)
                loss[0], loss[2] = self._bbox_loss_with_edge_mask(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes_reliable / stride_tensor,
                    target_scores_reliable,
                    target_scores_sum_reliable,
                    fg_mask_reliable,
                    edge_mask,
                )
            else:
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes_reliable / stride_tensor,
                    target_scores_reliable,
                    target_scores_sum_reliable,
                    fg_mask_reliable,
                )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach(), reliable_mask, unreliable_mask  # loss(box, cls, dfl)

    def _bbox_loss_with_stability(
        self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, fg_mask, stability
    ):
        """Weight localization responsibility by same-object stability and renormalize by its weight sum."""
        target_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        stability_weight = stability[fg_mask].to(target_weight.dtype).unsqueeze(-1)
        weight = target_weight * stability_weight
        weight_sum = weight.sum().clamp_min(1.0)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / weight_sum

        if self.bbox_loss.dfl_loss:
            reg_max = self.bbox_loss.dfl_loss.reg_max
            target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)
            loss_dfl = self.bbox_loss.dfl_loss(
                pred_dist[fg_mask].view(-1, reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / weight_sum
        else:
            loss_dfl = torch.zeros((), device=pred_dist.device)
        return loss_iou, loss_dfl
    
    def _get_reliable_and_unreliable_mask(self, unlabeled_conf, unlabeled_loc_conf=None):
        """
        Args:
            unlabeled_conf: [num_unlabeled_boxes, 1] classification confidence.
            unlabeled_loc_conf: [num_unlabeled_boxes, 1], optional DFL box-level localization
                confidence. When `self.use_loc_conf` is True, a box additionally needs
                loc_conf >= self.loc_conf_threshold to be considered reliable (dfl_ssod_algorithm
                Option B: "reliable pseudo-label のみ localization で制限").

        Returns:
            torch.BoolTensor: [num_unlabeled_boxes] しきい値以上のボックスだけ True のマスク
        """
        conf_flat = unlabeled_conf.squeeze(-1) # [num_unlabeled_boxes]
        reliable_mask = (conf_flat >= self.conf_threshold_high)
        if self.use_loc_conf and unlabeled_loc_conf is not None:
            reliable_mask = reliable_mask & (unlabeled_loc_conf.squeeze(-1) >= self.loc_conf_threshold)
        unreliable_mask = (conf_flat >= self.conf_threshold_low) & ~reliable_mask
        return reliable_mask, unreliable_mask

    @staticmethod
    def _pad_per_image(batch_idx, values, batch_size):
        """
        Pad a flat list of per-box values into (batch_size, n_max_boxes, feature_dim), grouped by
        image exactly like `v8DetectionLoss.preprocess` groups bboxes -- so that when `batch_idx`
        and the row order match what was used to build `reliable_gt_bboxes`, the two padded
        tensors line up slot-for-slot and `target_gt_idx` from the assigner indexes both correctly.

        Args:
            batch_idx: [n, 1] image index per row (same rows/order as the corresponding bboxes).
            values: [n, feature_dim] per-box values to pad (e.g. per-edge DFL confidence).
            batch_size (int): number of images in the batch.
        """
        n = values.shape[0]
        feat_dim = values.shape[1]
        if n == 0:
            return torch.zeros(batch_size, 0, feat_dim, device=values.device)
        i = batch_idx.squeeze(-1)
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), feat_dim, device=values.device)
        for j in range(batch_size):
            matches = i == j
            if n_j := matches.sum():
                out[j, :n_j] = values[matches]
        return out

    def _empty_edge_dfl_reweight_stats(self, reliable_mask):
        """Stable diagnostic schema, including empty-foreground batches."""
        return {
            "edge_dfl_reweight_enabled": bool(self.edge_dfl_reweight),
            "edge_dfl_selector": self.edge_dfl_selector if self.edge_dfl_reweight else "disabled",
            "edge_dfl_selected_weight": self.edge_dfl_weight if self.edge_dfl_reweight else 1.0,
            "edge_dfl_normalize": bool(self.edge_dfl_normalize) if self.edge_dfl_reweight else False,
            "edge_dfl_legacy_mask_bypassed": bool(self.edge_dfl_reweight and self.use_edge_conf),
            "edge_dfl_reliable_boxes": int(reliable_mask.sum()),
            "edge_dfl_oracle_matched_pseudo_boxes": 0,
            "edge_dfl_eligible_edges": 0,
            "edge_dfl_selected_edges": 0,
            "edge_dfl_selected_edge_rate": 0.0,
            "edge_dfl_raw_weight_mean": 1.0,
            "edge_dfl_normalized_weight_mean": 1.0,
            "edge_dfl_selected_normalized_weight_mean": 0.0,
            "edge_dfl_nonselected_normalized_weight_mean": 1.0,
            "edge_dfl_raw_loss": 0.0,
            "edge_dfl_weighted_raw_loss": 0.0,
            "edge_dfl_weighted_normalized_loss": 0.0,
            "edge_dfl_normalization_zero_sum_batch": False,
            "edge_dfl_normalization_zero_sum_batches": self.edge_dfl_zero_sum_batches,
            "edge_dfl_loss_balance_factor": 1.0,
            "edge_dfl_final_contribution": 0.0,
        }

    def _selected_edges_for_reliable_pseudos(
        self,
        unlabeled_batch_idx,
        unlabeled_edge_conf,
        reliable_mask,
        reliable_gt_bboxes,
        reliable_gt_labels,
        reliable_mask_gt,
        imgsz,
        oracle_gt_bboxes,
        oracle_gt_cls,
        oracle_gt_batch_idx,
    ):
        """Build a four-edge selector mask per padded reliable pseudo object."""
        batch_size, n_max_boxes = reliable_gt_bboxes.shape[:2]
        empty = torch.zeros((batch_size, n_max_boxes, 4), device=reliable_gt_bboxes.device, dtype=torch.bool)
        oracle_matched = torch.zeros((batch_size, n_max_boxes, 1), device=reliable_gt_bboxes.device, dtype=torch.bool)
        if n_max_boxes == 0:
            return empty, oracle_matched
        if self.edge_dfl_selector == "dfl":
            if unlabeled_edge_conf is None:
                raise ValueError("edge_dfl_selector='dfl' requires unlabeled_edge_conf")
            padded_conf = self._pad_per_image(
                unlabeled_batch_idx[reliable_mask], unlabeled_edge_conf[reliable_mask], batch_size
            )
            return select_dfl_low_confidence_edges(padded_conf, self.edge_conf_threshold), oracle_matched
        if oracle_gt_bboxes is None or oracle_gt_cls is None or oracle_gt_batch_idx is None:
            raise ValueError("edge_dfl_selector='oracle' requires transformed oracle GT tensors")
        oracle_targets = torch.cat((oracle_gt_batch_idx, oracle_gt_cls, oracle_gt_bboxes), 1)
        oracle_targets = self.preprocess(oracle_targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        oracle_labels, oracle_boxes = oracle_targets.split((1, 4), 2)
        oracle_valid = oracle_boxes.sum(2, keepdim=True).gt_(0.0)
        return oracle_selected_edges(
            reliable_gt_bboxes,
            reliable_gt_labels,
            reliable_mask_gt,
            oracle_boxes,
            oracle_labels,
            oracle_valid,
            image_h=imgsz[0],
            image_w=imgsz[1],
            error_threshold=self.oracle_edge_error_threshold,
            match_iou=0.5,
        )

    @staticmethod
    def _map_pseudo_edge_values_to_foreground(padded_values, target_gt_idx, fg_mask):
        """Use TaskAlignedAssigner's object identity to broadcast values to positives."""
        if not fg_mask.any():
            return torch.zeros((0, 4), device=fg_mask.device, dtype=padded_values.dtype)
        n_max_boxes = padded_values.shape[1]
        if n_max_boxes == 0:
            return torch.zeros((int(fg_mask.sum()), 4), device=fg_mask.device, dtype=padded_values.dtype)
        batch_ind = torch.arange(fg_mask.shape[0], device=fg_mask.device).unsqueeze(-1)
        flat_idx = target_gt_idx.long() + batch_ind * n_max_boxes
        return padded_values.reshape(-1, 4)[flat_idx][fg_mask]

    @torch.no_grad()
    def _edge_dfl_reweight_stats(
        self,
        selected_edges,
        edge_weight_result,
        pred_dist,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        oracle_matched,
        reliable_mask,
    ):
        stats = self._empty_edge_dfl_reweight_stats(reliable_mask)
        stats["edge_dfl_oracle_matched_pseudo_boxes"] = int(oracle_matched.sum())
        stats["edge_dfl_eligible_edges"] = int(selected_edges.numel())
        stats["edge_dfl_selected_edges"] = int(selected_edges.sum())
        stats["edge_dfl_selected_edge_rate"] = float(selected_edges.float().mean()) if selected_edges.numel() else 0.0
        stats["edge_dfl_raw_weight_mean"] = float(edge_weight_result.raw.mean()) if edge_weight_result.raw.numel() else 1.0
        stats["edge_dfl_normalized_weight_mean"] = float(edge_weight_result.normalized.mean()) if edge_weight_result.normalized.numel() else 1.0
        if selected_edges.any():
            stats["edge_dfl_selected_normalized_weight_mean"] = float(edge_weight_result.normalized[selected_edges].mean())
        if (~selected_edges).any():
            stats["edge_dfl_nonselected_normalized_weight_mean"] = float(edge_weight_result.normalized[~selected_edges].mean())
        stats["edge_dfl_normalization_zero_sum_batch"] = edge_weight_result.normalization_zero_sum
        stats["edge_dfl_normalization_zero_sum_batches"] = self.edge_dfl_zero_sum_batches
        if not selected_edges.numel() or not self.bbox_loss.dfl_loss:
            return stats
        reg_max = self.bbox_loss.dfl_loss.reg_max
        target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)[fg_mask]
        per_edge = dfl_per_edge_loss(pred_dist[fg_mask].view(-1, 4, reg_max), target_ltrb, reg_max)
        target_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        ones = torch.ones_like(edge_weight_result.raw)
        stats["edge_dfl_raw_loss"] = float(reduce_weighted_dfl(per_edge, target_weight, target_scores_sum, ones).detach())
        stats["edge_dfl_weighted_raw_loss"] = float(
            reduce_weighted_dfl(per_edge, target_weight, target_scores_sum, edge_weight_result.raw).detach()
        )
        stats["edge_dfl_weighted_normalized_loss"] = float(
            reduce_weighted_dfl(per_edge, target_weight, target_scores_sum, edge_weight_result.normalized).detach()
        )
        return stats

    def _bbox_loss_with_edge_weights(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        edge_weights,
    ):
        """Baseline CIoU plus DFL with common weights applied before the edge mean."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if not self.bbox_loss.dfl_loss:
            return loss_iou, torch.zeros((), device=pred_dist.device)
        reg_max = self.bbox_loss.dfl_loss.reg_max
        target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)[fg_mask]
        per_edge = dfl_per_edge_loss(pred_dist[fg_mask].view(-1, 4, reg_max), target_ltrb, reg_max)
        return loss_iou, reduce_weighted_dfl(per_edge, weight, target_scores_sum, edge_weights)

    def _bbox_loss_with_edge_mask(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        edge_mask,
    ):
        """
        Same as `BboxLoss.forward` (box gain via CIoU, unaffected -- CIoU couples all 4
        coordinates so it can't be split per edge) except the DFL term is averaged only over
        edges where `edge_mask` is True for that box, instead of always averaging all 4.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        reg_max = self.bbox_loss.dfl_loss.reg_max
        target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)[fg_mask]  # (N_fg, 4)
        pred_dist_fg = pred_dist[fg_mask].view(-1, 4, reg_max)  # (N_fg, 4, reg_max)

        target_ltrb = target_ltrb.clamp(0, reg_max - 1 - 0.01)
        tl = target_ltrb.long()
        tr = tl + 1
        wl = tr - target_ltrb
        wr = 1 - wl
        pd = pred_dist_fg.reshape(-1, reg_max)
        ce_l = F.cross_entropy(pd, tl.reshape(-1), reduction="none").view(tl.shape)
        ce_r = F.cross_entropy(pd, tr.reshape(-1), reduction="none").view(tl.shape)
        per_edge_loss = ce_l * wl + ce_r * wr  # (N_fg, 4)

        mask = edge_mask.to(per_edge_loss.dtype)
        denom = mask.sum(-1).clamp(min=1)
        loss_dfl_per_box = (per_edge_loss * mask).sum(-1) / denom  # (N_fg,), skip low-confidence edges
        loss_dfl = (loss_dfl_per_box.unsqueeze(-1) * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    @torch.no_grad()
    def _compute_assignment_stats(target_gt_idx, fg_mask, mask_gt, target_scores, reliable_mask):
        """Per-pseudo-box positive-anchor counts, for section 4 of the SSOD diagnostics spec."""
        # TaskAlignedAssigner's n_max_boxes==0 fast path (no reliable pseudo-labels this batch)
        # returns target_gt_idx as torch.zeros_like(pd_scores[..., 0]) -- i.e. in pred_scores'
        # dtype (fp16 under AMP), not long -- so it must be cast before use as a scatter index.
        target_gt_idx = target_gt_idx.long()
        n_max_boxes = mask_gt.shape[1]
        fg_target_scores = target_scores.sum(-1)[fg_mask]
        if n_max_boxes == 0:
            # No reliable pseudo-labels anywhere in this batch (common early in training / at
            # very low label ratios) -- preprocess() returns a zero-width padded dim in that
            # case, so there is nothing to scatter into.
            per_box_counts = torch.zeros(0, device=fg_mask.device)
        else:
            counts = torch.zeros(mask_gt.shape[0], n_max_boxes, device=fg_mask.device, dtype=torch.float32)
            idx = torch.where(fg_mask, target_gt_idx, torch.zeros_like(target_gt_idx))
            counts.scatter_add_(1, idx, fg_mask.to(counts.dtype))
            per_box_counts = counts[mask_gt.squeeze(-1).bool()]
        return {
            "num_pseudo_boxes": int(reliable_mask.sum()),
            "num_unsup_assigned": int(reliable_mask.sum()),
            "num_unsup_positive": int(fg_mask.sum()),
            "sum_target_scores_unsup": float(target_scores.sum()),
            "assigned_anchors_per_pseudo_box_mean": float(per_box_counts.mean()) if per_box_counts.numel() else 0.0,
            "assigned_anchors_per_pseudo_box_std": float(per_box_counts.std()) if per_box_counts.numel() > 1 else 0.0,
            "target_score_mean_unsup": float(fg_target_scores.mean()) if fg_target_scores.numel() else 0.0,
            "target_score_std_unsup": float(fg_target_scores.std()) if fg_target_scores.numel() > 1 else 0.0,
        }

    @staticmethod
    @torch.no_grad()
    def _compute_negative_supervision_bins(
        pred_scores,
        anchor_points,
        stride_tensor,
        imgsz,
        fg_mask_reliable,
        ignore_mask,
        target_scores_reliable,
        unlabeled_bboxes,
        unlabeled_cls,
        unlabeled_batch_idx,
        unlabeled_conf,
        reliable_mask,
    ):
        """
        For every teacher NMS-survivor candidate (reliable or not), find the single nearest
        anchor by pixel distance and record what classification supervision that anchor
        actually receives from the *student* this step -- in particular whether a
        threshold-rejected candidate's anchor is being trained as a hard negative (target
        score 0, not excluded via ignore_mask) even though the teacher itself scored it above
        zero. No GT is used. Aggregated into 10 teacher-confidence bins (score 0.0-0.1, ...,
        0.9-1.0) to keep the log small; see section 5 of the SSOD diagnostics spec.
        """
        if unlabeled_bboxes.numel() == 0:
            return [dict(_EMPTY_NEGSUP_BIN) for _ in range(10)]

        anchor_px = anchor_points * stride_tensor  # (A, 2), shared across the batch
        centers_px = unlabeled_bboxes[:, :2] * imgsz[0]  # (N, 2); square-image assumption matches preprocess()
        img_idx = unlabeled_batch_idx.squeeze(-1).long()
        cls_idx = unlabeled_cls.squeeze(-1).long()
        conf_flat = unlabeled_conf.squeeze(-1)

        nearest_anchor = torch.cdist(centers_px.unsqueeze(0), anchor_px.unsqueeze(0)).squeeze(0).argmin(dim=1)

        student_score = pred_scores.detach().sigmoid()[img_idx, nearest_anchor, cls_idx]
        student_target_score = target_scores_reliable[img_idx, nearest_anchor, cls_idx]
        is_ignored = ignore_mask[img_idx, nearest_anchor]
        is_positive = fg_mask_reliable[img_idx, nearest_anchor]
        cls_logit = pred_scores[img_idx, nearest_anchor, cls_idx]
        cls_loss = F.binary_cross_entropy_with_logits(cls_logit, student_target_score, reduction="none")
        is_negative = (~is_ignored) & (~is_positive)

        bin_idx = (conf_flat.clamp(0, 0.999999) * 10).long()
        bins = []
        for b in range(10):
            m = bin_idx == b
            n = int(m.sum())
            if n == 0:
                bins.append(dict(_EMPTY_NEGSUP_BIN))
                continue
            bins.append(
                {
                    "count": n,
                    "teacher_score_mean": float(conf_flat[m].mean()),
                    "selected_as_pseudo_label_rate": float(reliable_mask[m].float().mean()),
                    "student_score_mean": float(student_score[m].mean()),
                    "student_target_score_mean": float(student_target_score[m].mean()),
                    "student_is_positive_rate": float(is_positive[m].float().mean()),
                    "student_is_negative_rate": float(is_negative[m].float().mean()),
                    "classification_loss_mean": float(cls_loss[m].mean()),
                }
            )
        return bins



import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None





class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdversarialNet(nn.Module):
    """
    x: [B, in_dim] （flatten 済み特徴）を想定したドメイン分類器
    """
    def __init__(self, in_dim: int, num_classes: int = 2, lambda_: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(lambda_=lambda_)
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, in_dim]
        x = self.grl(x)
        logits = self.domain_classifier(x)
        return logits
