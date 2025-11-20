from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy

import torch

class EfficientTeacherLoss(v8DetectionLoss):
    def __init__(self, model, conf_threshold_high=0.6, conf_threshold_low=0.1):
        super().__init__(model)
        self.conf_threshold_high = conf_threshold_high
        self.conf_threshold_low = conf_threshold_low

    def __call__(self, preds, unlabeled_bboxes, unlabeled_cls, unlabeled_conf, unlabeled_batch_idx):
        """
        Args:
            preds:
            unlabeled_bboxes: [num_unlabeled_boxes, 4]
            unlabeled_cls: [num_unlabeled_boxes, 1]
            unlabeled_conf: [num_unlabeled_boxes, 1]
            unlabeled_batch_idx: [num_unlabeled_boxes, 1]
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
        reliable_mask, unreliable_mask = self._get_reliable_and_unreliable_mask(unlabeled_conf)

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

        _, target_bboxes_reliable, target_scores_reliable, fg_mask_reliable, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(reliable_gt_bboxes.dtype),
            anchor_points * stride_tensor,
            reliable_gt_labels,
            reliable_gt_bboxes,
            reliable_mask_gt,
        )
        _, _, _, fg_mask_unreliable, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(unreliable_gt_bboxes.dtype),
            anchor_points * stride_tensor,
            unreliable_gt_labels,
            unreliable_gt_bboxes,
            unreliable_mask_gt,
        )

        ignore_mask = fg_mask_unreliable & ~fg_mask_reliable
        target_scores_sum_reliable = max((target_scores_reliable.sum(-1)*((~ignore_mask).to(dtype))).sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores_reliable.to(dtype)).sum() / target_scores_sum_reliable  # BCE
        loss[1] = (self.bce(pred_scores, target_scores_reliable.to(dtype)).sum(-1) * ((~ignore_mask).to(dtype))).sum() / target_scores_sum_reliable  # BCE
        # Bbox loss
        if fg_mask_reliable.sum():
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

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def _get_reliable_and_unreliable_mask(self, unlabeled_conf):
        """
        Args:
            unlabeled_conf: [num_unlabeled_boxes, 1]
        
        Returns:
            torch.BoolTensor: [num_unlabeled_boxes] しきい値以上のボックスだけ True のマスク
        """
        # しきい値は self.conf_thresh があればそれを、無ければ 0.5 を既定値として使用
        conf_flat = unlabeled_conf.squeeze(-1) # [num_unlabeled_boxes]
        reliable_mask = (conf_flat >= self.conf_threshold_high)
        unreliable_mask = (conf_flat >= self.conf_threshold_low) & ~reliable_mask
        return reliable_mask, unreliable_mask