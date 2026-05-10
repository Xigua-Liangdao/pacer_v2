#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_similarity_matrix(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    similarity = payload.get("text_prototype_similarity")
    if not similarity:
        raise ValueError(f"text_prototype_similarity not found in {json_path}")

    labels = list(similarity.keys())
    matrix = np.array([[float(similarity[row][col]) for col in labels] for row in labels], dtype=float)
    return labels, matrix


def prettify_label(label: str) -> str:
    return label.replace("_", " ").title()


def draw_single_heatmap(ax, labels, matrix, title, cmap, vmin, vmax):
    pretty_labels = [prettify_label(label) for label in labels]
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pretty_labels)))
    ax.set_yticks(np.arange(len(pretty_labels)))
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    ax.set_yticklabels(pretty_labels)
    ax.set_xlabel("Text Prototype")
    ax.set_ylabel("Text Prototype")
    ax.set_title(title)

    threshold = (vmin + vmax) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < threshold else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(pretty_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pretty_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main():
    parser = argparse.ArgumentParser(description="Plot a paper-style comparison figure for two text prototype similarity matrices.")
    parser.add_argument("--left_json", required=True)
    parser.add_argument("--right_json", required=True)
    parser.add_argument("--left_title", default="Original Facial-Cue Prompts")
    parser.add_argument("--right_title", default="Auto-Selected Prompts")
    parser.add_argument("--suptitle", default="Comparison of Text Prototype Similarity Matrices")
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_pdf", default=None)
    parser.add_argument("--vmin", type=float, default=0.90)
    parser.add_argument("--vmax", type=float, default=1.00)
    parser.add_argument("--cmap", default="Blues")
    args = parser.parse_args()

    left_labels, left_matrix = load_similarity_matrix(Path(args.left_json))
    right_labels, right_matrix = load_similarity_matrix(Path(args.right_json))
    if left_labels != right_labels:
        raise ValueError("Left and right matrices must use the same label order")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)
    im = draw_single_heatmap(axes[0], left_labels, left_matrix, args.left_title, args.cmap, args.vmin, args.vmax)
    draw_single_heatmap(axes[1], right_labels, right_matrix, args.right_title, args.cmap, args.vmin, args.vmax)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Cosine Similarity")
    fig.suptitle(args.suptitle, fontsize=17)

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    if args.output_pdf:
        output_pdf = Path(args.output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
