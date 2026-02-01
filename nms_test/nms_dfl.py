import argparse
import torch
from torchvision.ops import nms as tv_nms   # ★ 追加
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import ops
from pathlib import Path



# ---------------------------------------------------------
# 1. NMS + アンカー index 追跡
# ---------------------------------------------------------
def non_max_suppression_with_idx(prediction, conf_thres, iou_thres, max_det, nc):
    """
    prediction: [B, N, 4+nc]
    dets_list[b]: [K, 6] = [x1,y1,x2,y2,conf,cls]
    idx_list[b]:  [K]    = 元アンカー index
    """
    device = prediction.device
    bs, N, no = prediction.shape
    assert no == nc + 4

    dets_list, idx_list = [], []

    for b in range(bs):
        x = prediction[b]  # [N, 4+nc]
        if not x.numel():
            dets_list.append(torch.zeros((0, 6), device=device))
            idx_list.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        cls_scores = x[:, 4:]  # [N, nc]
        conf, cls_idx = cls_scores.max(1, keepdim=True)  # [N, 1]
        x = torch.cat((x[:, :4], conf, cls_idx.float()), dim=1)  # [N, 6]

        mask = conf.view(-1) > conf_thres
        if not mask.any():
            dets_list.append(torch.zeros((0, 6), device=device))
            idx_list.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        x = x[mask]  # [M, 6]
        anchor_idx = torch.arange(N, device=device)[mask]  # [M]

        boxes = ops.xywh2xyxy(x[:, :4])
        scores = x[:, 4]

        keep = tv_nms(boxes, scores, iou_thres)[:max_det]


        dets = torch.cat((boxes[keep], x[keep, 4:6]), dim=1)  # [K, 6]
        dets_list.append(dets)
        idx_list.append(anchor_idx[keep])

    return dets_list, idx_list


# ---------------------------------------------------------
# 2. DFL logits → 辺ごとの分布
# ---------------------------------------------------------
def decode_dfl_logits_to_probs(logits: torch.Tensor):
    """
    logits: [B, 4*reg_max, N]
    return: probs_4: [B, 4, reg_max, N]
    """
    if logits is None:
        return None
    B, C, N = logits.shape
    reg_max = C // 4
    logits_4 = logits.view(B, 4, reg_max, N)
    probs_4 = torch.softmax(logits_4, dim=2)
    return probs_4


# ---------------------------------------------------------
# 3. DFL 対応 Predictor
# ---------------------------------------------------------
class DFLDetectionPredictor(DetectionPredictor):
    """
    - Detect.dfl に hook を仕込んで DFL logits を保存
    - postprocess で自前 NMS (with anchor idx) を実行
    - construct_results は元の実装のまま使用
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

        self._dfl_logits = None       # [B, 4*reg_max, N]
        self.dfl_probs = None         # [B, 4, reg_max, N]
        self.anchor_indices = None    # list[Tensor]
        self._dfl_handle = None       # hook ハンドル

    def setup_model(self, model=None):
        """
        モデル構築後に Detect head の DFL に hook を登録する。
        """
        # まず親クラス側で self.model を構築 (AutoBackend を作る)
        super().setup_model(model)

        # ---- Detect モジュールを総当たりで探す ----
        detect = None
        for m in self.model.modules():
            # SSOD フォーク等も考えて "Detect" を含むクラス名を探す
            if "Detect" in m.__class__.__name__:
                detect = m

        if detect is None:
            raise RuntimeError(
                f"Detect-like module not found in model (type={type(self.model)})"
            )

        if not hasattr(detect, "dfl"):
            raise RuntimeError(
                f"Found Detect-like module ({detect.__class__.__name__}) "
                f"but it has no attribute 'dfl'"
            )

        # ---- DFL への hook 登録 ----
        def hook_dfl_input(module, inp, out):
            # inp[0]: [B, 4*reg_max, N]
            self._dfl_logits = inp[0].detach()

        self._dfl_handle = detect.dfl.register_forward_hook(hook_dfl_input)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        - preds: モデルの生出力 (NMS 前)
        - ここで自前 NMS をかけて anchor idx を保持
        - construct_results を呼んで Results を生成
        """
        # preds が list で来るケースにも対応
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.dim() != 3:
            raise RuntimeError(f"Unexpected preds shape: {preds.shape}")

        B, d1, d2 = preds.shape
        # 通常: [B, no, N] なので d1 < d2 なら permute
    if d1 < d2:
            prediction = preds.permute(0, 2, 1).contiguous()  # [B, N, 4+nc]
    else:
            prediction = preds

        nc = len(self.model.names)

        # 1) 自前 NMS
        dets_list, idx_list = non_max_suppression_with_idx(
            prediction,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            max_det=self.args.max_det,
            nc=nc,
        )

        # 2) DFL logits → probs
        self.dfl_probs = decode_dfl_logits_to_probs(self._dfl_logits)
        self.anchor_indices = idx_list

        # 3) orig_imgs を list[np.ndarray] に揃える
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 4) 既存の construct_results を利用
        results = self.construct_results(dets_list, img, orig_imgs)

        # 5) 対応関係を Results に付ける
        for res, idxs in zip(results, idx_list):
            res.anchor_idx = idxs  # [K]

        return results


# ---------------------------------------------------------
# 4. メイン: argparse で --source / --model を受け取る
# ---------------------------------------------------------
import cv2  # ファイルの先頭あたりで追加

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--outdir", type=str, default=None, help="出力先ディレクトリ（未指定ならCWD/vis_nms_dfl）")
    args = parser.parse_args()

    overrides = dict(
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    predictor = DFLDetectionPredictor(overrides=overrides)

    # 出力ディレクトリ設定
    out_dir = Path(args.outdir) if args.outdir else (Path.cwd() / "vis_nms_dfl")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 推論本体（NMS + DFL hook つき）: バッチごとに保存
    batch_index = 0
    for batch_results in predictor.stream_inference(source=args.source):
        results = predictor.results  # list[Results] for current batch
        dfl_probs = predictor.dfl_probs  # [B, 4, reg_max, N] for current batch
        anchor_indices = predictor.anchor_indices  # list[Tensor] len=B
        if not results:
            continue

        # バッチ内の各画像について保存
        for i, res in enumerate(results):
            im_bgr = res.plot()
            # 保存ファイル名は元画像名ベース
            img_name = Path(res.path).name if getattr(res, "path", None) else f"batch{batch_index}_img{i}.jpg"
            out_path = out_dir / img_name
            cv2.imwrite(str(out_path), im_bgr)
            print(f"saved: {out_path}")

        # 任意: 先頭画像について DFL 情報をログ出力（バッチ単位での確認用）
        try:
            res0 = results[0]
            idxs0 = res0.anchor_idx  # [K]
            print(f"#detections for first image in batch {batch_index}: {len(res0.boxes)}")
            if dfl_probs is not None and len(idxs0):
                for k, box in enumerate(res0.boxes.xyxy[:5]):  # 最初の数件のみログ
                    n = int(idxs0[k])
                    dist_edges = dfl_probs[0, :, :, n]  # [4, reg_max]
                    edge_max = dist_edges.max(dim=1).values  # [4]
                    print(
                        f"det {k}: box={box.tolist()}, anchor_idx={n}, "
                        f"edge_max_probs={edge_max.tolist()}"
                    )
        except Exception:
            pass
        batch_index += 1


if __name__ == "__main__":
    main()

