"""Convert CrowdHuman .odgt annotations to YOLO-format label .txt files.

Single class (0=person), using the full-body box ("fbox": [x, y, w, h] in absolute
pixels, top-left origin) for each gtbox tagged "person". Boxes tagged "mask"
(ignore regions) are skipped, matching common CrowdHuman-to-YOLO conversions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def convert(odgt_path: Path, images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_images = 0
    n_boxes = 0
    n_missing = 0
    with open(odgt_path) as f:
        for line in f:
            rec = json.loads(line)
            img_path = images_dir / f"{rec['ID']}.jpg"
            if not img_path.exists():
                n_missing += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size

            lines = []
            for box in rec["gtboxes"]:
                if box["tag"] != "person":
                    continue
                x, y, bw, bh = box["fbox"]
                # clip to image bounds before normalizing
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                if bw <= 0 or bh <= 0:
                    continue
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_boxes += 1

            out_path = labels_dir / f"{rec['ID']}.txt"
            out_path.write_text("\n".join(lines))
            n_images += 1

    print(f"{odgt_path.name}: wrote {n_images} label files ({n_boxes} boxes), {n_missing} images missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/kawano/LA/datasets/crowdhuman")
    args = parser.parse_args()
    root = Path(args.root)

    convert(root / "annotation_train.odgt", root / "images/train", root / "labels/train")
    convert(root / "annotation_val.odgt", root / "images/val", root / "labels/val")


if __name__ == "__main__":
    main()
