#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_BLUE = "#3775BA"
METHOD_BLUE_LIGHT = "#7097CA"
METHOD_BLUE_PALE = "#B8C9E5"
PROMPT_GREEN = "#8BCF8B"
PROMPT_GREEN_LIGHT = "#AADCA9"
BASELINE_CORAL = "#E9A6A1"
BASELINE_CORAL_LIGHT = "#F6CFCB"
ACCENT_GOLD = "#FFF6CC"
GRAY = "#BFBFBF"
DARK = "#222222"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.0,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "legend.fontsize": 13,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_png: Path, output_pdf: Path = None) -> None:
    ensure_parent(output_png)
    fig.savefig(output_png)
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf)
