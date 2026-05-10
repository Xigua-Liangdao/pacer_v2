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


def draw_heatmap(labels, matrix, title: str, output_png: Path, output_pdf: Path | None, vmin: float | None, vmax: float | None):
    pretty_labels = [prettify_label(label) for label in labels]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pretty_labels)))
    ax.set_yticks(np.arange(len(pretty_labels)))
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    ax.set_yticklabels(pretty_labels)
    ax.set_xlabel("Text Prototype")
    ax.set_ylabel("Text Prototype")
    ax.set_title(title)

    threshold = (matrix.max() + matrix.min()) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < threshold else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity")

    ax.set_xticks(np.arange(-0.5, len(pretty_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pretty_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a publication-style heatmap for text prototype similarity.")
    parser.add_argument("--input_json", required=True, help="Path to a result JSON file containing text_prototype_similarity.")
    parser.add_argument("--output_png", required=True, help="Output PNG path.")
    parser.add_argument("--output_pdf", default=None, help="Optional output PDF path.")
    parser.add_argument("--title", default="Prototype Similarity Matrix of Text Embeddings", help="Figure title.")
    parser.add_argument("--vmin", type=float, default=0.90, help="Heatmap lower bound.")
    parser.add_argument("--vmax", type=float, default=1.00, help="Heatmap upper bound.")
    args = parser.parse_args()

    labels, matrix = load_similarity_matrix(Path(args.input_json))
    draw_heatmap(
        labels=labels,
        matrix=matrix,
        title=args.title,
        output_png=Path(args.output_png),
        output_pdf=Path(args.output_pdf) if args.output_pdf else None,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
