#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from aide_visualization_utils import build_method_features_and_predictions, build_test_split_from_result, limit_samples_per_class, load_method, resolve_path, set_seed
from paper_figure_style import ACCENT_GOLD, BASELINE_CORAL, METHOD_BLUE, PROMPT_GREEN, PROMPT_GREEN_LIGHT, apply_paper_style, save_figure, style_axes


def parse_args():
    parser = argparse.ArgumentParser(description="Plot prompt comparison figure with quantitative metrics and t-SNE.")
    parser.add_argument("--simple_ckpt", default="results/ablations/single_prompt.ckpt.pt")
    parser.add_argument("--simple_result_json", default="results/ablations/single_prompt.json")
    parser.add_argument("--rich_ckpt", default="results/ablations/full_best.ckpt.pt")
    parser.add_argument("--rich_result_json", default="results/ablations/full_best.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples_per_class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", type=float, default=200.0)
    parser.add_argument("--tsne_iterations", type=int, default=2500)
    parser.add_argument("--output_png", default="results/paper_figures/aide_prompt_comparison.png")
    parser.add_argument("--output_pdf", default="results/paper_figures/aide_prompt_comparison.pdf")
    parser.add_argument("--export_panels", action="store_true", default=True)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_tsne_panel(ax, coords, labels, title):
    color_map = {
        "Anxiety": "#D55E00",
        "Peace": "#009E73",
        "Weariness": "#0072B2",
        "Happiness": "#E69F00",
        "Anger": "#CC79A7",
    }
    for label, color in color_map.items():
        idx = [i for i, item in enumerate(labels) if item == label]
        if not idx:
            continue
        subset = coords[idx]
        ax.scatter(subset[:, 0], subset[:, 1], s=70, alpha=0.94, c=color, label=label, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.14, linewidth=0.8)


def panel_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def export_metric_panel(metric_names, simple_metrics, rich_metrics, output_png: Path, output_pdf: Path = None):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    x = np.arange(len(metric_names))
    width = 0.34
    ax.bar(x - width / 2, simple_metrics, width=width, color=PROMPT_GREEN_LIGHT, label="Simple prompt")
    ax.bar(x + width / 2, rich_metrics, width=width, color=METHOD_BLUE, label="Semantic-rich prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("Prompt Design Improves Performance")
    style_axes(ax)
    for xpos, value in zip(x - width / 2, simple_metrics):
        ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    for xpos, value in zip(x + width / 2, rich_metrics):
        ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_png, output_pdf)
    plt.close(fig)


def export_tsne_only(coords, labels, title, output_png: Path, output_pdf: Path = None):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    draw_tsne_panel(ax, coords, labels, title)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, output_png, output_pdf)
    plt.close(fig)


def main():
    args = parse_args()
    apply_paper_style()
    set_seed(args.seed)

    simple_result = load_json(resolve_path(args.simple_result_json))
    rich_result = load_json(resolve_path(args.rich_result_json))

    train_samples, _, test_samples, _ = build_test_split_from_result(resolve_path(args.rich_result_json), seed_override=args.seed)
    test_samples = limit_samples_per_class(test_samples, args.max_samples_per_class, args.seed)

    simple_method = load_method(resolve_path(args.simple_ckpt), name="single_prompt", device=args.device, baseline_mode="adapter", seed=args.seed)
    rich_method = load_method(resolve_path(args.rich_ckpt), name="rich_prompt", device=args.device, baseline_mode="adapter", seed=args.seed)
    simple_outputs = build_method_features_and_predictions(simple_method, test_samples, args.batch_size)
    rich_outputs = build_method_features_and_predictions(rich_method, test_samples, args.batch_size)

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.tsne_perplexity,
        "learning_rate": args.tsne_learning_rate,
        "n_iter": args.tsne_iterations,
        "random_state": args.seed,
        "init": "pca",
    }
    simple_coords = TSNE(**tsne_kwargs).fit_transform(simple_outputs["features"])
    rich_coords = TSNE(**tsne_kwargs).fit_transform(rich_outputs["features"])

    metric_names = ["Val Acc", "Val WF1", "Test Acc", "Test WF1"]
    simple_metrics = [
        simple_result["val"]["accuracy"],
        simple_result["val"]["weighted_f1"],
        simple_result["test"]["accuracy"],
        simple_result["test"]["weighted_f1"],
    ]
    rich_metrics = [
        rich_result["val"]["accuracy"],
        rich_result["val"]["weighted_f1"],
        rich_result["test"]["accuracy"],
        rich_result["test"]["weighted_f1"],
    ]

    fig = plt.figure(figsize=(17.2, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x = np.arange(len(metric_names))
    width = 0.34
    ax0.bar(x - width / 2, simple_metrics, width=width, color=PROMPT_GREEN_LIGHT, label="Simple prompt")
    ax0.bar(x + width / 2, rich_metrics, width=width, color=METHOD_BLUE, label="Semantic-rich prompt")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_names)
    ax0.set_ylim(0.45, 0.85)
    ax0.set_ylabel("Score")
    ax0.set_title("Prompt Design Improves Performance")
    style_axes(ax0)
    for xpos, value in zip(x - width / 2, simple_metrics):
        ax0.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    for xpos, value in zip(x + width / 2, rich_metrics):
        ax0.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    ax0.legend(frameon=False, loc="upper left")

    draw_tsne_panel(ax1, simple_coords, simple_outputs["labels"], "Simple Prompt t-SNE")
    draw_tsne_panel(ax2, rich_coords, rich_outputs["labels"], "Semantic-Rich Prompt t-SNE")
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.67, 0.98))
    fig.suptitle("Prompt Engineering Analysis on AIDE", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_figure(fig, Path(args.output_png), Path(args.output_pdf) if args.output_pdf else None)
    plt.close(fig)

    if args.export_panels:
        output_png = Path(args.output_png)
        output_pdf = Path(args.output_pdf) if args.output_pdf else None
        export_metric_panel(
            metric_names,
            simple_metrics,
            rich_metrics,
            panel_output_path(output_png, "metrics"),
            panel_output_path(output_pdf, "metrics") if output_pdf else None,
        )
        export_tsne_only(
            simple_coords,
            simple_outputs["labels"],
            "Simple Prompt t-SNE",
            panel_output_path(output_png, "simple_tsne"),
            panel_output_path(output_pdf, "simple_tsne") if output_pdf else None,
        )
        export_tsne_only(
            rich_coords,
            rich_outputs["labels"],
            "Semantic-Rich Prompt t-SNE",
            panel_output_path(output_png, "rich_tsne"),
            panel_output_path(output_pdf, "rich_tsne") if output_pdf else None,
        )


if __name__ == "__main__":
    main()
