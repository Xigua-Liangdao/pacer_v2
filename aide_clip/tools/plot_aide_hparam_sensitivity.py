#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import BASELINE_CORAL, METHOD_BLUE, METHOD_BLUE_LIGHT, METHOD_BLUE_PALE, PROMPT_GREEN, PROMPT_GREEN_LIGHT, apply_paper_style, save_figure, style_axes


def parse_args():
    parser = argparse.ArgumentParser(description="Plot hyperparameter sensitivity figures for AIDE.")
    parser.add_argument("--adapter_summary", default="results/adapter_sweep/adapter_sweep_summary.json")
    parser.add_argument("--ablation_summary", default="results/ablations/ablation_summary.json")
    parser.add_argument("--meanpool_json", default="results/repro/clip_emotion_strict_repro_c.json")
    parser.add_argument("--transformer_json", default="results/repro/clip_emotion_strict_repro_c_transformer_try_v2.json")
    parser.add_argument("--output_png", default="results/paper_figures/aide_hparam_sensitivity.png")
    parser.add_argument("--output_pdf", default="results/paper_figures/aide_hparam_sensitivity.pdf")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    apply_paper_style()

    adapter_summary = load_json(Path(args.adapter_summary))
    ablation_summary = load_json(Path(args.ablation_summary))
    meanpool_result = load_json(Path(args.meanpool_json))
    transformer_result = load_json(Path(args.transformer_json))

    records = adapter_summary["results"]
    by_dropout = {}
    for row in records:
        by_dropout.setdefault(float(row["adapter_dropout"]), []).append(row)
    for rows in by_dropout.values():
        rows.sort(key=lambda item: float(item["adapter_hidden_dim"]))

    ablation_rows = {row["name"]: row for row in ablation_summary["results"]}
    full_row = ablation_summary["reference"]

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.8))
    axes = axes.ravel()

    color_by_dropout = {0.1: METHOD_BLUE_LIGHT, 0.2: METHOD_BLUE, 0.3: BASELINE_CORAL}
    for dropout, rows in sorted(by_dropout.items()):
        xs = [int(row["adapter_hidden_dim"]) for row in rows]
        ys = [float(row["wf1"]) for row in rows]
        axes[0].plot(xs, ys, marker="o", linewidth=2.5, color=color_by_dropout.get(dropout, METHOD_BLUE_PALE), label=f"dropout={dropout}")
    axes[0].set_title("Adapter Hidden Dimension")
    axes[0].set_xlabel("Hidden dimension")
    axes[0].set_ylabel("Test weighted F1")
    style_axes(axes[0])
    axes[0].legend(frameon=False, loc="lower right")

    prompt_x = [1, 7]
    prompt_y = [float(ablation_rows["single_prompt"]["wf1"]), float(full_row["wf1"])]
    axes[1].plot(prompt_x, prompt_y, marker="o", linewidth=2.8, color=PROMPT_GREEN)
    axes[1].fill_between(prompt_x, prompt_y, alpha=0.16, color=PROMPT_GREEN_LIGHT)
    axes[1].set_title("Prompt Count")
    axes[1].set_xlabel("Prompts per class")
    axes[1].set_ylabel("Test weighted F1")
    axes[1].set_xticks(prompt_x)
    style_axes(axes[1])

    temp_x = [0, 1]
    temp_y = [float(ablation_rows["no_class_temperature"]["wf1"]), float(full_row["wf1"])]
    axes[2].plot(temp_x, temp_y, marker="o", linewidth=2.8, color=BASELINE_CORAL)
    axes[2].fill_between(temp_x, temp_y, alpha=0.16, color="#f6cfc3")
    axes[2].set_title("Class Temperature")
    axes[2].set_xlabel("Enabled")
    axes[2].set_ylabel("Test weighted F1")
    axes[2].set_xticks(temp_x)
    axes[2].set_xticklabels(["Off", "On"])
    style_axes(axes[2])

    frame_x = [1, 5]
    frame_y = [float(ablation_rows["one_frame_only"]["wf1"]), float(full_row["wf1"])]
    axes[3].plot(frame_x, frame_y, marker="o", linewidth=2.8, color=METHOD_BLUE_LIGHT)
    axes[3].fill_between(frame_x, frame_y, alpha=0.16, color=METHOD_BLUE_PALE)
    axes[3].set_title("Frame Count")
    axes[3].set_xlabel("Number of frames")
    axes[3].set_ylabel("Test weighted F1")
    axes[3].set_xticks(frame_x)
    style_axes(axes[3])

    smooth_x = [0.0, 0.03]
    smooth_y = [float(ablation_rows["no_label_smoothing"]["wf1"]), float(full_row["wf1"])]
    axes[4].plot(smooth_x, smooth_y, marker="o", linewidth=2.8, color="#8f78b9")
    axes[4].fill_between(smooth_x, smooth_y, alpha=0.16, color="#d6c8e8")
    axes[4].set_title("Label Smoothing")
    axes[4].set_xlabel("Label smoothing")
    axes[4].set_ylabel("Test weighted F1")
    axes[4].set_xticks(smooth_x)
    style_axes(axes[4])

    temporal_labels = ["Mean pooled\n(no temporal head)", "Transformer\n2 layers, CLS"]
    temporal_y = [float(meanpool_result["test"]["weighted_f1"]), float(transformer_result["test"]["weighted_f1"])]
    axes[5].bar(np.arange(2), temporal_y, color=[METHOD_BLUE, BASELINE_CORAL], width=0.58)
    axes[5].set_title("Temporal Module Variant")
    axes[5].set_ylabel("Test weighted F1")
    axes[5].set_xticks(np.arange(2))
    axes[5].set_xticklabels(temporal_labels)
    style_axes(axes[5])
    axes[5].text(0.02, 0.02, "Available runs compare no temporal head vs transformer.\nNo dense temporal-layer sweep found in the repo.", transform=axes[5].transAxes, fontsize=10.5, ha="left", va="bottom")

    fig.suptitle("AIDE Hyperparameter and Design Sensitivity", fontsize=22)
    fig.tight_layout()
    save_figure(fig, Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
