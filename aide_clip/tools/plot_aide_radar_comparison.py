#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import BASELINE_CORAL, METHOD_BLUE, PROMPT_GREEN, apply_paper_style, save_figure


EMOTION_LABELS = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot radar comparison across multiple AIDE methods.")
    parser.add_argument("--method_json", default="results/adapter_sweep/h2048_d02.json")
    parser.add_argument("--prompt_json", default="results/ablations/single_prompt.json")
    parser.add_argument("--baseline_json", default="results/zeroshot/clip_zeroshot.json")
    parser.add_argument("--output_png", default="results/paper_figures/aide_radar_comparison.png")
    parser.add_argument("--output_pdf", default="results/paper_figures/aide_radar_comparison.pdf")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_recall(confusion_matrix, label):
    row = confusion_matrix.get(label, {})
    total = float(sum(row.values()))
    if total <= 0:
        return 0.0
    return float(row.get(label, 0)) / total


def result_to_vector(result):
    vector = [float(result["test"]["accuracy"]), float(result["test"]["weighted_f1"])]
    vector.extend(class_recall(result["test"]["confusion_matrix"], label) for label in EMOTION_LABELS)
    return vector


def main():
    args = parse_args()
    apply_paper_style()

    method = load_json(Path(args.method_json))
    prompt = load_json(Path(args.prompt_json))
    baseline = load_json(Path(args.baseline_json))

    labels = ["Accuracy", "Weighted F1", "Recall A", "Recall P", "Recall W", "Recall H", "Recall G"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    series = [
        ("Best image-text", result_to_vector(method), METHOD_BLUE),
        ("Single prompt", result_to_vector(prompt), PROMPT_GREEN),
        ("Zero-shot baseline", result_to_vector(baseline), BASELINE_CORAL),
    ]

    fig = plt.figure(figsize=(8.4, 7.0))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(alpha=0.25)

    for name, values, color in series:
        values = values + values[:1]
        ax.plot(angles, values, color=color, linewidth=2.8, label=name)
        ax.fill(angles, values, color=color, alpha=0.14)

    ax.set_title("Multi-metric Comparison on AIDE", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12), frameon=False)
    save_figure(fig, Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)
    plt.close(fig)


if __name__ == "__main__":
    main()