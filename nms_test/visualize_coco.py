#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import cv2
from pycocotools.coco import COCO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-dir", type=str,
                   default="/work/kawano/LA/datasets/coco")
    p.add_argument("--split", type=str, default="val2017")  # train2017 も可
    p.add_argument("--ann-file", type=str,
                   default="annotations/instances_val2017.json")
    p.add_argument("--output-dir", type=str, default="vis_coco_json")
    p.add_argument("--num-images", type=int, default=0)
    args = p.parse_args()

    coco_dir = Path(args.coco_dir)
    ann_path = coco_dir / args.ann_file
    img_dir = coco_dir / "images" / args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()

    if 0 < args.num_images < len(img_ids):
        random.seed(0)
        img_ids = random.sample(img_ids, args.num_images)

    # カテゴリ名テーブル
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    rng = random.Random(0)
    cat_id_to_color = {
        cid: (rng.randint(0,255), rng.randint(0,255), rng.randint(0,255))
        for cid in cat_id_to_name.keys()
    }

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = img_dir / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 画像読めない: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCOは [x, y, w, h] (絶対座標, 左上+幅高さ)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cid = ann["category_id"]
            name = cat_id_to_name.get(cid, str(cid))
            color = cat_id_to_color[cid]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = name
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yt = max(th + bl + 2, y1)
            cv2.rectangle(img, (x1, yt - th - bl - 2), (x1 + tw, yt), color, -1)
            cv2.putText(img, label, (x1, yt - bl - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1, cv2.LINE_AA)

        out_path = out_dir / file_name
        cv2.imwrite(str(out_path), img)
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
