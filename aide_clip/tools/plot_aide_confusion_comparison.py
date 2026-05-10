#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from paper_figure_style import apply_paper_style, save_figure


EMOTION_LABELS = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paper-style confusion matrix comparison for AIDE results.")
    parser.add_argument("--method_json", default="results/adapter_sweep/h2048_d02.json")
    parser.add_argument("--baseline_json", default="results/zeroshot/clip_zeroshot.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output_png", default="results/paper_figures/aide_confusion_comparison.png")
    parser.add_argument("--output_pdf", default="results/paper_figures/aide_confusion_comparison.pdf")
    parser.add_argument("--export_panels", action="store_true", default=True)
    return parser.parse_args()


def load_result(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def confusion_to_matrix(confusion_dict):
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=float)
    for row_idx, true_label in enumerate(EMOTION_LABELS):
        row = confusion_dict.get(true_label, {})
        for col_idx, pred_label in enumerate(EMOTION_LABELS):
            matrix[row_idx, col_idx] = float(row.get(pred_label, 0))
    return matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def annotate_heatmap(ax, matrix, fmt=".2f", threshold=0.5):
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if abs(value) > threshold else "black"
            ax.text(col_idx, row_idx, format(value, fmt), ha="center", va="center", color=color, fontsize=11)


def draw_heatmap(ax, matrix, title, cmap, vmin, vmax):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(EMOTION_LABELS)))
    ax.set_yticks(np.arange(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return image


def panel_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def export_single_panel(matrix, title, cmap, vmin, vmax, fmt, threshold, colorbar_label, output_png: Path, output_pdf: Path = None):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = draw_heatmap(ax, matrix, title, cmap, vmin, vmax)
    annotate_heatmap(ax, matrix, fmt=fmt, threshold=threshold)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_png, output_pdf)
    plt.close(fig)


def main():
    args = parse_args()
    apply_paper_style()

    method = load_result(Path(args.method_json))
    baseline = load_result(Path(args.baseline_json))
    method_matrix = normalize_rows(confusion_to_matrix(method[args.split]["confusion_matrix"]))
    baseline_matrix = normalize_rows(confusion_to_matrix(baseline[args.split]["confusion_matrix"]))
    delta_matrix = method_matrix - baseline_matrix

    warm_cmap = LinearSegmentedColormap.from_list("paper_warm", ["#fff7f2", "#f6cfcb", "#e9a6a1", "#c9534d"])
    diff_cmap = LinearSegmentedColormap.from_list("paper_diff", ["#4e79a7", "#f7f7f7", "#c9534d"])

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8))
    im0 = draw_heatmap(axes[0], baseline_matrix, "Baseline", warm_cmap, 0.0, 1.0)
    im1 = draw_heatmap(axes[1], method_matrix, "Image-Text Method", warm_cmap, 0.0, 1.0)
    im2 = draw_heatmap(axes[2], delta_matrix, "Method - Baseline", diff_cmap, -0.4, 0.4)

    annotate_heatmap(axes[0], baseline_matrix, fmt=".2f", threshold=0.55)
    annotate_heatmap(axes[1], method_matrix, fmt=".2f", threshold=0.55)
    annotate_heatmap(axes[2], delta_matrix, fmt="+.2f", threshold=0.18)

    cbar0 = fig.colorbar(im1, ax=axes[:2], fraction=0.028, pad=0.02)
    cbar0.set_label("Row-normalized fraction")
    cbar1 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar1.set_label("Delta")

    fig.suptitle(f"AIDE Confusion Matrix Comparison ({args.split.title()})", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)
    plt.close(fig)

    if args.export_panels:
        output_png = Path(args.output_png)
        output_pdf = Path(args.output_pdf) if args.output_pdf else None
        export_single_panel(
            baseline_matrix,
            "Baseline",
            warm_cmap,
            0.0,
            1.0,
            ".2f",
            0.55,
            "Row-normalized fraction",
            panel_output_path(output_png, "baseline"),
            panel_output_path(output_pdf, "baseline") if output_pdf else None,
        )
        export_single_panel(
            method_matrix,
            "Image-Text Method",
            warm_cmap,
            0.0,
            1.0,
            ".2f",
            0.55,
            "Row-normalized fraction",
            panel_output_path(output_png, "method"),
            panel_output_path(output_pdf, "method") if output_pdf else None,
        )
        export_single_panel(
            delta_matrix,
            "Method - Baseline",
            diff_cmap,
            -0.4,
            0.4,
            "+.2f",
            0.18,
            "Delta",
            panel_output_path(output_png, "delta"),
            panel_output_path(output_pdf, "delta") if output_pdf else None,
        )


if __name__ == "__main__":
    main()
