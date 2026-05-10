#!/usr/bin/env python3
import argparse
import json

import numpy as np

from aide_visualization_utils import draw_tsne_comparison, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Redraw AIDE t-SNE comparison from saved coordinate json files.")
    parser.add_argument("--baseline_coords_json", required=True)
    parser.add_argument("--method_coords_json", required=True)
    parser.add_argument("--baseline_title", default="Zero-shot Baseline")
    parser.add_argument("--method_title", default="Image-Text Method")
    parser.add_argument("--marker_scale", type=float, default=0.88)
    parser.add_argument("--plot_style", choices=["default", "block"], default="default")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_coord_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    coords = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)
    labels = [str(row["label"]) for row in rows]
    return coords, labels


def main():
    args = parse_args()
    baseline_path = resolve_path(args.baseline_coords_json)
    method_path = resolve_path(args.method_coords_json)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_coords, baseline_labels = load_coord_rows(baseline_path)
    method_coords, method_labels = load_coord_rows(method_path)
    if baseline_labels != method_labels:
        raise ValueError("Baseline and method coordinate label orders differ; cannot redraw paired comparison safely.")

    draw_tsne_comparison(
        baseline_coords=baseline_coords,
        method_coords=method_coords,
        labels=baseline_labels,
        baseline_title=args.baseline_title,
        method_title=args.method_title,
        output_png=output_dir / "aide_tsne_comparison.png",
        output_pdf=output_dir / "aide_tsne_comparison.pdf",
        plot_style=args.plot_style,
        marker_scale=args.marker_scale,
    )


if __name__ == "__main__":
    main()