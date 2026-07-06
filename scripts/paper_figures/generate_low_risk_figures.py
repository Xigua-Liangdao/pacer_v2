#!/usr/bin/env python3
"""Generate low-risk paper figures from existing frozen feature alignment results.

This script does not train models and does not invent values. It reads
existing tables, JSON summaries, and checkpoints, then exports both the
figure PDFs and the source CSV files used for plotting.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/paper_figures_final"
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#3775BA",
    "blue_light": "#9AB9DF",
    "green": "#5E9F6E",
    "green_light": "#BFE3C7",
    "gold": "#D8A03D",
    "coral": "#C95D5D",
    "gray": "#6F7378",
    "pale_gray": "#F3F4F6",
    "dark": "#222222",
    "purple": "#7B61A8",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_dual(fig, name: str) -> None:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    shutil.copy2(pdf, REPO / f"{name}.pdf")
    plt.close(fig)


def parse_seed_table() -> list[dict]:
    path = REPO / "paper_tables/tableS2_a0_extended_seed_analysis.tex"
    rows: list[dict] = []
    pattern = re.compile(r"^(canonical|historical)\s*&\s*(\d+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip("\\")
        match = pattern.match(line)
        if not match:
            continue
        rows.append(
            {
                "source": match.group(1),
                "training_seed": int(match.group(2)),
                "test_acc": float(match.group(3)),
                "test_wf1": float(match.group(4)),
                "source_file": str(path),
            }
        )
    if len(rows) != 9:
        raise RuntimeError(f"Expected 9 AIDE seed rows, found {len(rows)}")
    return rows


def fig_seed_stability() -> None:
    rows = parse_seed_table()
    write_csv(
        OUT / "fig_seed_stability_source.csv",
        rows,
        ["source", "training_seed", "test_acc", "test_wf1", "source_file"],
    )
    x = np.arange(len(rows))
    labels = [str(r["training_seed"]) for r in rows]
    acc = [r["test_acc"] for r in rows]
    wf1 = [r["test_wf1"] for r in rows]
    colors = [COLORS["blue"] if r["source"] == "canonical" else COLORS["green"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.plot(x, acc, color=COLORS["blue"], marker="o", linewidth=1.8, label="Accuracy")
    ax.plot(x, wf1, color=COLORS["coral"], marker="s", linewidth=1.8, label="Weighted F1")
    ax.scatter(x, acc, s=35, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(x, wf1, s=35, c=colors, marker="s", edgecolor="white", linewidth=0.7, zorder=3)
    ax.axhline(np.mean(acc), color=COLORS["blue"], linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(np.mean(wf1), color=COLORS["coral"], linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Score")
    ax.set_ylim(0.74, 0.84)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("AIDE nine-seed stability")
    ax.legend(loc="lower right", frameon=False)
    note = "canonical seeds: 42, 123, 2024; historical seeds: 20, 11, 14, 18, 24, 28"
    ax.text(0.01, -0.33, note, transform=ax.transAxes, fontsize=7.2, color=COLORS["gray"])
    fig.tight_layout()
    save_dual(fig, "fig_seed_stability")


def aide_ablation_rows() -> list[dict]:
    return [
        {"dataset": "AIDE", "method": "Full framework", "wf1_mean": 0.814, "wf1_std": 0.000, "delta": 0.000, "source": "docs/RESULTS.md"},
        {"dataset": "AIDE", "method": "Uniform prompt averaging", "wf1_mean": 0.800, "wf1_std": 0.000, "delta": -0.014, "source": "docs/RESULTS.md"},
        {"dataset": "AIDE", "method": "w/o affine scale", "wf1_mean": 0.762, "wf1_std": 0.000, "delta": -0.052, "source": "docs/RESULTS.md"},
        {"dataset": "AIDE", "method": "w/o affine bias", "wf1_mean": 0.800, "wf1_std": 0.000, "delta": -0.014, "source": "docs/RESULTS.md"},
        {"dataset": "AIDE", "method": "w/o residual adapter", "wf1_mean": 0.438, "wf1_std": 0.000, "delta": -0.376, "source": "docs/RESULTS.md"},
    ]


def yawdd_ablation_rows() -> list[dict]:
    data = json.loads((REPO / "results/final_runs/yawdd/B_ablation/B_pch_ablation_summary.json").read_text())
    rows = []
    for item in data["summary"]:
        rows.append(
            {
                "dataset": "YawDD",
                "method": item["name"],
                "wf1_mean": item["wf1_mean"],
                "wf1_std": item["wf1_std"],
                "delta": 0.0 if item["delta_vs_YA0_mean"] is None else item["delta_vs_YA0_mean"],
                "source": "results/final_runs/yawdd/B_ablation/B_pch_ablation_summary.json",
            }
        )
    return rows


def fig_ablation_bars() -> None:
    rows = aide_ablation_rows() + yawdd_ablation_rows()
    write_csv(OUT / "fig_ablation_bars_source.csv", rows, ["dataset", "method", "wf1_mean", "wf1_std", "delta", "source"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, dataset, color in zip(axes, ["AIDE", "YawDD"], [COLORS["blue"], COLORS["green"]]):
        subset = [r for r in rows if r["dataset"] == dataset]
        x = np.arange(len(subset))
        vals = [r["wf1_mean"] for r in subset]
        err = [r["wf1_std"] for r in subset]
        ax.bar(x, vals, yerr=err, capsize=3, color=color, alpha=0.86, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["method"] for r in subset], rotation=32, ha="right")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.30, 0.84)
    axes[0].set_ylabel("Weighted F1 mean")
    fig.suptitle("Component ablations")
    fig.tight_layout()
    save_dual(fig, "fig_ablation_bars")


def fig_temporal_aggregation() -> None:
    rows = [
        {"method": "Mean pooling\n(final)", "acc_best": 0.801, "acc_mean": 0.789, "acc_std": 0.017, "wf1_best": 0.788, "wf1_mean": 0.777, "wf1_std": 0.011, "delta": 0.0, "source": "paper_tables/tableS1_aide_architecture_choice.tex"},
        {"method": "Mean pooling\n(variant)", "acc_best": 0.815, "acc_mean": 0.807, "acc_std": 0.011, "wf1_best": 0.800, "wf1_mean": 0.794, "wf1_std": 0.011, "delta": 0.017, "source": "paper_tables/tableS1_aide_architecture_choice.tex"},
        {"method": "CGP-FG", "acc_best": 0.744, "acc_mean": 0.731, "acc_std": 0.021, "wf1_best": 0.720, "wf1_mean": 0.701, "wf1_std": 0.026, "delta": -0.076, "source": "paper_tables/tableS1_aide_architecture_choice.tex"},
        {"method": "TAGA", "acc_best": 0.688, "acc_mean": 0.626, "acc_std": 0.054, "wf1_best": 0.610, "wf1_mean": 0.499, "wf1_std": 0.096, "delta": -0.278, "source": "paper_tables/tableS1_aide_architecture_choice.tex"},
    ]
    write_csv(OUT / "fig_temporal_aggregation_source.csv", rows, ["method", "acc_best", "acc_mean", "acc_std", "wf1_best", "wf1_mean", "wf1_std", "delta", "source"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(x - 0.17, [r["acc_mean"] for r in rows], yerr=[r["acc_std"] for r in rows], width=0.34, capsize=3, color=COLORS["blue"], label="Accuracy")
    ax.bar(x + 0.17, [r["wf1_mean"] for r in rows], yerr=[r["wf1_std"] for r in rows], width=0.34, capsize=3, color=COLORS["coral"], label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"] for r in rows])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0.40, 0.84)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("AIDE temporal aggregation")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_dual(fig, "fig_temporal_aggregation")


def fig_perclass_f1() -> None:
    rows = [
        {"class": "Anxiety", "f1_mean": 0.643, "f1_std": 0.016, "source": "paper_tables/table1_main.tex"},
        {"class": "Peace", "f1_mean": 0.858, "f1_std": 0.011, "source": "paper_tables/table1_main.tex"},
        {"class": "Weariness", "f1_mean": 0.791, "f1_std": 0.025, "source": "paper_tables/table1_main.tex"},
        {"class": "Happiness", "f1_mean": 0.737, "f1_std": 0.044, "source": "paper_tables/table1_main.tex"},
        {"class": "Anger", "f1_mean": 0.502, "f1_std": 0.051, "source": "paper_tables/table1_main.tex"},
    ]
    write_csv(OUT / "fig_perclass_f1_source.csv", rows, ["class", "f1_mean", "f1_std", "source"])
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(rows))
    ax.bar(x, [r["f1_mean"] for r in rows], yerr=[r["f1_std"] for r in rows], capsize=3, color=[COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["purple"], COLORS["coral"]])
    ax.set_xticks(x)
    ax.set_xticklabels([r["class"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("F1 mean")
    ax.set_ylim(0.40, 0.90)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("AIDE per-class F1 across nine seeds")
    fig.tight_layout()
    save_dual(fig, "fig_perclass_f1")


def fig_yawdd_sweep() -> None:
    rows = json.loads((REPO / "results/final_runs/yawdd/B_sweep/sweep_summary.json").read_text())
    rows = sorted(rows, key=lambda r: r["test_wf1"], reverse=True)
    for r in rows:
        r["source"] = "results/final_runs/yawdd/B_sweep/sweep_summary.json"
    write_csv(
        OUT / "fig_yawdd_sweep_source.csv",
        rows,
        [
            "config",
            "hidden",
            "dropout",
            "lr",
            "wd",
            "loss",
            "ls",
            "frames",
            "sampling",
            "cw",
            "val_best_wf1",
            "test_wf1",
            "test_acc",
            "train_time_sec",
            "killed",
            "exitcode",
            "gpu",
            "path",
            "source",
        ],
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    focal = [r for r in rows if str(r["loss"]).startswith("focal")]
    ce = [r for r in rows if str(r["loss"]).lower().startswith("ce")]
    for group, marker, color, label in [(focal, "o", COLORS["blue"], "focal"), (ce, "s", COLORS["coral"], "CE")]:
        if not group:
            continue
        sizes = [42 + math.sqrt(float(r["hidden"])) * 3.2 for r in group]
        ax.scatter([r["val_best_wf1"] for r in group], [r["test_wf1"] for r in group], s=sizes, marker=marker, color=color, alpha=0.80, edgecolor="white", linewidth=0.7, label=label)
        for r in group:
            ax.text(r["val_best_wf1"] + 0.002, r["test_wf1"] + 0.002, r["config"], fontsize=7)
    ax.plot([0.75, 0.94], [0.75, 0.94], linestyle="--", color=COLORS["gray"], linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Validation weighted F1")
    ax.set_ylabel("Test weighted F1")
    ax.set_xlim(0.75, 0.94)
    ax.set_ylim(0.70, 0.84)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("YawDD hyperparameter sweep")
    fig.tight_layout()
    save_dual(fig, "fig_yawdd_sweep")


def fig_clip_baselines() -> None:
    base = json.loads((REPO / "results/final_runs/yawdd/B_table3_baselines/yawdd_table3_baselines_aggregate.json").read_text())
    vanilla = json.loads((REPO / "results/final_runs/yawdd/B_table3_baselines/vanilla_clip_adapter/vanilla_clip_adapter_yawdd_aggregate.json").read_text())
    wanted = {
        "clip_zero_shot": "Zero-shot CLIP",
        "tip_adapter": "Tip-Adapter",
        "clip_linear_probe": "Linear Probe",
        "coop": "CoOp",
        "maple": "MaPLe",
    }
    rows = []
    for item in base["summary"]:
        if item["baseline_id"] in wanted:
            rows.append(
                {
                    "method": wanted[item["baseline_id"]],
                    "seeds": item["seed_count"],
                    "wf1_mean": item["weighted_f1"]["mean"],
                    "wf1_std": item["weighted_f1"]["std"],
                    "wf1_best": item["weighted_f1"]["best"],
                    "source": "results/final_runs/yawdd/B_table3_baselines/yawdd_table3_baselines_aggregate.json",
                }
            )
    rows.insert(
        3,
        {
            "method": "Vanilla CLIP-Adapter",
            "seeds": 5,
            "wf1_mean": vanilla["wf1"]["mean"],
            "wf1_std": vanilla["wf1"]["std"],
            "wf1_best": vanilla["wf1"]["best"],
            "source": "results/final_runs/yawdd/B_table3_baselines/vanilla_clip_adapter/vanilla_clip_adapter_yawdd_aggregate.json",
        },
    )
    rows.append(
        {
            "method": "Ours",
            "seeds": 5,
            "wf1_mean": 0.8007,
            "wf1_std": 0.03861,
            "wf1_best": 0.840359,
            "source": "YAWDD_FINAL_TABLES_AND_TEXT.md",
        }
    )
    write_csv(OUT / "fig_clip_baselines_source.csv", rows, ["method", "seeds", "wf1_mean", "wf1_std", "wf1_best", "source"])
    rows_sorted = rows
    x = np.arange(len(rows_sorted))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = [COLORS["blue_light"]] * len(rows_sorted)
    colors[-1] = COLORS["blue"]
    ax.bar(x, [r["wf1_mean"] for r in rows_sorted], yerr=[r["wf1_std"] for r in rows_sorted], capsize=3, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"] for r in rows_sorted], rotation=28, ha="right")
    ax.set_ylabel("Weighted F1 mean")
    ax.set_ylim(0.30, 0.86)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("YawDD CLIP-adaptation baselines")
    fig.tight_layout()
    save_dual(fig, "fig_clip_baselines")


def fig_promptweight() -> None:
    ckpt_path = REPO / "results/final_runs/aide/A0_ours_full_seed42_canonical/best.ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    logits = ckpt["adapter_state_dict"]["prompt_weight_logits"].float()
    alpha = torch.softmax(logits, dim=1).numpy()
    idx2label = ckpt["idx2label"]
    labels = [idx2label[i] for i in range(len(idx2label))]
    prompts = [f"P{i}" for i in range(1, alpha.shape[1] + 1)]
    rows = []
    for ci, label in enumerate(labels):
        for pi, prompt in enumerate(prompts):
            rows.append(
                {
                    "class": label,
                    "prompt": prompt,
                    "alpha": float(alpha[ci, pi]),
                    "prompt_text": ckpt["prompt_groups"][ci][pi],
                    "checkpoint": str(ckpt_path),
                }
            )
    write_csv(OUT / "fig_promptweight_alpha.csv", rows, ["class", "prompt", "alpha", "prompt_text", "checkpoint"])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    im = ax.imshow(alpha, cmap="Blues", vmin=0.0, vmax=max(0.20, float(alpha.max())))
    ax.set_xticks(np.arange(len(prompts)))
    ax.set_xticklabels(prompts)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            val = alpha[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > alpha.max() * 0.65 else COLORS["dark"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Prompt weight")
    ax.set_xlabel("Prompt template")
    ax.set_title("AIDE PCH prompt weights")
    fig.tight_layout()
    save_dual(fig, "fig_promptweight")


def rounded_box(ax, xy, w, h, text, fc, ec, fontsize=8.5, lw=1.2):
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["dark"])


def arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color=color))


def fig_arch() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.03, 0.58), 0.14, 0.22, "Input\nframes", "#F2F2F2", "#A0A0A0")
    rounded_box(ax, (0.22, 0.58), 0.17, 0.22, "Frozen CLIP\nimage encoder", "#DDEBFA", COLORS["blue"])
    rounded_box(ax, (0.44, 0.58), 0.14, 0.22, "Temporal\nmean pooling", "#F6E8C9", COLORS["gold"])
    rounded_box(ax, (0.63, 0.58), 0.15, 0.22, "Residual\nimage adapter", "#E2F2E4", COLORS["green"])
    rounded_box(ax, (0.83, 0.58), 0.13, 0.22, "PCH\nclassifier", "#F6DCDC", COLORS["coral"])
    rounded_box(ax, (0.22, 0.18), 0.17, 0.18, "Frozen CLIP\ntext encoder", "#DDEBFA", COLORS["blue"])
    rounded_box(ax, (0.46, 0.18), 0.20, 0.18, "Class prompt\nfeature bank", "#EAE4F2", COLORS["purple"])
    rounded_box(ax, (0.83, 0.18), 0.13, 0.18, "Prediction", "#F2F2F2", "#A0A0A0")
    for x1, x2 in [(0.17, 0.22), (0.39, 0.44), (0.58, 0.63), (0.78, 0.83)]:
        arrow(ax, x1, 0.69, x2, 0.69)
    arrow(ax, 0.39, 0.27, 0.46, 0.27)
    arrow(ax, 0.66, 0.27, 0.83, 0.62)
    arrow(ax, 0.90, 0.58, 0.90, 0.36)
    ax.text(0.305, 0.84, "frozen", ha="center", color=COLORS["blue"], fontsize=8, weight="bold")
    ax.text(0.705, 0.84, "trainable", ha="center", color=COLORS["green"], fontsize=8, weight="bold")
    ax.text(0.90, 0.84, "trainable", ha="center", color=COLORS["coral"], fontsize=8, weight="bold")
    fig.tight_layout()
    save_dual(fig, "fig_arch")


def fig_pch_detail() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.04, 0.58), 0.17, 0.20, "Adapted visual\nfeature h", "#E2F2E4", COLORS["green"])
    rounded_box(ax, (0.04, 0.18), 0.17, 0.20, "Prompt bank\nC x P x D", "#EAE4F2", COLORS["purple"])
    rounded_box(ax, (0.29, 0.38), 0.16, 0.20, "Cosine\nsimilarities", "#F2F2F2", "#A0A0A0")
    rounded_box(ax, (0.53, 0.60), 0.15, 0.18, "softmax theta\nalpha(c,p)", "#F6DCDC", COLORS["coral"])
    rounded_box(ax, (0.53, 0.30), 0.15, 0.18, "weighted prompt\nscore", "#F6E8C9", COLORS["gold"])
    rounded_box(ax, (0.74, 0.42), 0.15, 0.20, "gamma_c score\n+ beta_c", "#F6DCDC", COLORS["coral"])
    rounded_box(ax, (0.91, 0.42), 0.07, 0.20, "tau\nlogits", "#F6DCDC", COLORS["coral"], fontsize=8)
    arrow(ax, 0.21, 0.68, 0.29, 0.50)
    arrow(ax, 0.21, 0.28, 0.29, 0.46)
    arrow(ax, 0.45, 0.50, 0.53, 0.39)
    arrow(ax, 0.61, 0.60, 0.61, 0.48)
    arrow(ax, 0.68, 0.39, 0.74, 0.50)
    arrow(ax, 0.89, 0.52, 0.91, 0.52)
    ax.text(0.605, 0.82, "trainable prompt weights", ha="center", color=COLORS["coral"], fontsize=8)
    ax.text(0.815, 0.68, "trainable class calibration", ha="center", color=COLORS["coral"], fontsize=8)
    ax.text(0.13, 0.08, "CLIP text prototypes are frozen; PCH learns only lightweight calibration parameters.", ha="left", fontsize=7.8, color=COLORS["gray"])
    fig.tight_layout()
    save_dual(fig, "fig_pch_detail")


def main() -> None:
    apply_style()
    fig_seed_stability()
    fig_ablation_bars()
    fig_temporal_aggregation()
    fig_perclass_f1()
    fig_yawdd_sweep()
    fig_clip_baselines()
    fig_promptweight()
    fig_arch()
    fig_pch_detail()
    manifest = {
        "generated": [
            "fig_seed_stability",
            "fig_ablation_bars",
            "fig_temporal_aggregation",
            "fig_perclass_f1",
            "fig_yawdd_sweep",
            "fig_clip_baselines",
            "fig_promptweight",
            "fig_arch",
            "fig_pch_detail",
        ],
        "output_dir": str(OUT),
        "root_copies": [str(REPO / f"fig_{name}.pdf") for name in []],
    }
    (OUT / "figure_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
