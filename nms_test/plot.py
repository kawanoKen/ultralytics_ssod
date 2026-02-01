#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.plotting_ssod import (
    plot_results_compare,
    plot_results_map_compare,
    plot_results_da_compare,
)


def parse_args():
    p = argparse.ArgumentParser(description="Compare two Ultralytics results.csv files and save overlayed plots.")
    p.add_argument("--csv-a", type=str, required=True, help="Path to first results.csv")
    p.add_argument("--csv-b", type=str, required=True, help="Path to second results.csv")
    p.add_argument("--label-a", type=str, default=None, help="Legend label for first CSV")
    p.add_argument("--label-b", type=str, default=None, help="Legend label for second CSV")
    p.add_argument("--out", type=str, default=None, help="Output image path (default: results_compare.png beside CSV-A)")
    p.add_argument("--map-only", action="store_true", help="Plot only mAP columns")
    p.add_argument("--da-only", action="store_true", help="Plot only DA-loss columns")
    return p.parse_args()


def main():
    args = parse_args()
    csv_a = Path(args.csv_a).expanduser().resolve()
    csv_b = Path(args.csv_b).expanduser().resolve()
    assert csv_a.exists(), f"CSV not found: {csv_a}"
    assert csv_b.exists(), f"CSV not found: {csv_b}"

    labels = None
    if args.label_a or args.label_b:
        labels = [args.label_a or csv_a.stem, args.label_b or csv_b.stem]

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    if args.map_only:
        plot_results_map_compare(files=[str(csv_a), str(csv_b)], labels=labels, out=str(out_path) if out_path else None)
        LOGGER.info(f"Saved mAP comparison plot to: {out_path or (csv_a.parent / 'results_map_compare.png')}")
    elif args.da_only:
        plot_results_da_compare(files=[str(csv_a), str(csv_b)], labels=labels, out=str(out_path) if out_path else None)
        LOGGER.info(f"Saved DA-loss comparison plot to: {out_path or (csv_a.parent / 'results_da_compare.png')}")
    else:
        plot_results_compare(files=[str(csv_a), str(csv_b)], labels=labels, out=str(out_path) if out_path else None)
        LOGGER.info(f"Saved comparison plot to: {out_path or (csv_a.parent / 'results_compare.png')}")


if __name__ == "__main__":
    main()


