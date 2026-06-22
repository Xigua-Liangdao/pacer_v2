#!/usr/bin/env python3
"""Select AIDE samples where adapted patch relevance moves toward the face.

This script compares raw frozen CLIP patch relevance against the saved
PACER/Ours adapter patch relevance. It is a qualitative diagnostic and
does not use or imply a supervised non-CLIP baseline.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
AIDE_TOOLS = REPO_ROOT / "aide_clip" / "tools"
AIDE_SRC = REPO_ROOT / "aide_clip" / "src"
for path in [str(AIDE_TOOLS), str(AIDE_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from aide_visualization_utils import (  # noqa: E402
    build_test_split_from_result,
    default_driver_focus_bbox,
    detect_primary_face_bbox,
    generate_patch_heatmap,
    heatmap_mass_in_bbox,
    load_method,
    set_seed,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], pad: float = 0.10) -> np.ndarray:
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    px = int(round(w * pad))
    py = int(round(h * pad))
    left = max(0, x - px)
    top = max(0, y - py)
    right = min(iw, x + w + px)
    bottom = min(ih, y + h + py)
    return image[top:bottom, left:right]


def add_bbox(ax, bbox: Tuple[int, int, int, int]) -> None:
    x, y, w, h = bbox
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=1.8))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=0.7, linestyle="--"))


def driver_face_prior_bbox(image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """AIDE in-cabin view prior for the driver's face region.

    Haar frontal-face detection often fires on the shoulder/torso for
    profile driver faces. This prior deliberately constrains the crop to
    the upper-left driver head region in 1920x1080 AIDE in-cabin frames.
    """
    image_h, image_w = image_shape[:2]
    x = int(round(image_w * 0.105))
    y = int(round(image_h * 0.295))
    w = int(round(image_w * 0.230))
    h = int(round(image_h * 0.345))
    return x, y, w, h


def is_plausible_aide_face_bbox(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> bool:
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    cx = (x + 0.5 * w) / image_w
    cy = (y + 0.5 * h) / image_h
    bw = w / image_w
    bh = h / image_h
    # Reject the common false positive on the driver's shoulder/torso.
    if cy > 0.58:
        return False
    if cx > 0.48:
        return False
    if bw < 0.05 or bw > 0.28 or bh < 0.08 or bh > 0.42:
        return False
    return True


def choose_bbox(image_rgb: np.ndarray, mode: str) -> Tuple[Tuple[int, int, int, int], str]:
    if mode == "prior":
        return driver_face_prior_bbox(image_rgb.shape), "driver_face_prior"
    if mode in {"haar", "haar_or_prior"}:
        bbox = detect_primary_face_bbox(image_rgb)
        if bbox is not None and is_plausible_aide_face_bbox(bbox, image_rgb.shape):
            return bbox, "haar_filtered"
        if mode == "haar":
            return None, "haar_missing_or_rejected"
        return driver_face_prior_bbox(image_rgb.shape), "driver_face_prior"
    raise ValueError(f"Unsupported bbox mode: {mode}")


def save_comparison_grid(selected: Sequence[Dict], output_pdf: Path, output_png: Path, title: str) -> None:
    n = len(selected)
    if n == 0:
        return
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": 180, "savefig.dpi": 300})
    # Match displayed heights across 16:9 whole-frame panels and the
    # face-crop panels by giving the whole-frame columns larger widths.
    # This avoids tall crop columns forcing large row gaps.
    fig, axes = plt.subplots(
        n,
        5,
        figsize=(10.8, max(1.9, 1.12 * n + 0.42)),
        gridspec_kw={"width_ratios": [1.78, 1.78, 1.78, 1.18, 1.18]},
    )
    if n == 1:
        axes = np.asarray(axes).reshape(1, 5)
    col_titles = ["Source", "Raw CLIP", "Ours", "Raw face crop", "Ours face crop"]
    for col, col_title in enumerate(col_titles):
        axes[0, col].set_title(col_title, fontsize=8.5)
    for row_idx, item in enumerate(selected):
        sample = item["sample"]
        bbox = item["bbox"]
        raw = item["raw_heatmap"]
        ours = item["ours_heatmap"]
        images = [
            raw["image_rgb"],
            raw["overlay_rgb"],
            ours["overlay_rgb"],
            crop_bbox(raw["overlay_rgb"], bbox),
            crop_bbox(ours["overlay_rgb"], bbox),
        ]
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_anchor("C")
            axes[row_idx, col_idx].axis("off")
            if col_idx in {0, 1, 2}:
                add_bbox(axes[row_idx, col_idx], bbox)
        axes[row_idx, 0].set_ylabel(
            f"{sample['sequence_id']}\nGT={sample['label']}\n"
            f"raw={item['raw_face_mass']:.2f}, ours={item['ours_face_mass']:.2f}",
            fontsize=7.0,
        )
    fig.suptitle(title, y=0.995, fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.006, hspace=0.015, wspace=0.018)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aide_ckpt", default=str(REPO_ROOT / "results/final_runs/aide/A0_ours_full_seed42_canonical/best.ckpt.pt"))
    parser.add_argument("--aide_result_json", default=str(REPO_ROOT / "results/final_runs/aide/A0_ours_full_seed42_canonical/result.json"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results/paper_figures_final/face_attention_selection"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--min_ours_mass", type=float, default=0.18)
    parser.add_argument("--max_raw_mass", type=float, default=0.18)
    parser.add_argument("--min_delta", type=float, default=0.05)
    parser.add_argument("--require_haar_face", action="store_true")
    parser.add_argument("--sample_ids", default="", help="Comma-separated AIDE sequence ids to scan.")
    parser.add_argument("--bbox_mode", choices=["prior", "haar", "haar_or_prior"], default="prior")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    _, _, samples, _ = build_test_split_from_result(
        result_json_path=Path(args.aide_result_json),
        seed_override=args.seed,
    )
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    if sample_ids:
        wanted = set(sample_ids)
        samples = [sample for sample in samples if sample["sequence_id"] in wanted]

    raw_method = load_method(Path(args.aide_ckpt), "raw_clip", device, baseline_mode="pure_clip", seed=args.seed)
    ours_method = load_method(Path(args.aide_ckpt), "ours", device, baseline_mode="adapter", seed=args.seed)

    rows: List[Dict] = []
    candidates: List[Dict] = []
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[scan] {idx}/{total} sample={sample['sequence_id']}", flush=True)
        target_label = sample["label"]
        raw_heatmap = generate_patch_heatmap(raw_method, sample, target_label=target_label)
        bbox, bbox_source = choose_bbox(raw_heatmap["image_rgb"], args.bbox_mode)
        if bbox is None:
            if args.require_haar_face or args.bbox_mode == "haar":
                continue
            bbox = driver_face_prior_bbox(raw_heatmap["image_rgb"].shape)
            bbox_source = "driver_face_prior"
        ours_heatmap = generate_patch_heatmap(ours_method, sample, target_label=target_label)
        raw_mass = float(heatmap_mass_in_bbox(raw_heatmap["heatmap"], bbox))
        ours_mass = float(heatmap_mass_in_bbox(ours_heatmap["heatmap"], bbox))
        delta = ours_mass - raw_mass
        row = {
            "sequence_id": sample["sequence_id"],
            "label": sample["label"],
            "frame_path": sample["frame_path"],
            "bbox_source": bbox_source,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "raw_pred": raw_heatmap["pred_label"],
            "raw_confidence": round(float(raw_heatmap["pred_confidence"]), 6),
            "ours_pred": ours_heatmap["pred_label"],
            "ours_confidence": round(float(ours_heatmap["pred_confidence"]), 6),
            "raw_face_mass": round(raw_mass, 6),
            "ours_face_mass": round(ours_mass, 6),
            "delta_face_mass": round(delta, 6),
        }
        rows.append(row)
        candidates.append(
            {
                "row": row,
                "sample": sample,
                "bbox": bbox,
                "raw_heatmap": raw_heatmap,
                "ours_heatmap": ours_heatmap,
                "raw_face_mass": raw_mass,
                "ours_face_mass": ours_mass,
                "delta_face_mass": delta,
            }
        )

    write_csv(output_dir / "face_attention_scan_all.csv", rows)

    strict = [
        item for item in candidates
        if item["ours_face_mass"] >= args.min_ours_mass
        and item["raw_face_mass"] <= args.max_raw_mass
        and item["delta_face_mass"] >= args.min_delta
    ]
    strict.sort(key=lambda item: (item["delta_face_mass"], item["ours_face_mass"]), reverse=True)
    fallback = sorted(candidates, key=lambda item: (item["delta_face_mass"], item["ours_face_mass"]), reverse=True)
    selected = strict[: args.top_k] if strict else fallback[: args.top_k]
    selected_rows = []
    for rank, item in enumerate(selected, start=1):
        row = dict(item["row"])
        row["rank"] = rank
        row["selection_mode"] = "strict_threshold" if strict else "fallback_top_delta_face_crop"
        selected_rows.append(row)
        np.save(output_dir / f"raw_heatmap_{row['sequence_id']}.npy", item["raw_heatmap"]["heatmap"])
        np.save(output_dir / f"ours_heatmap_{row['sequence_id']}.npy", item["ours_heatmap"]["heatmap"])
    write_csv(output_dir / "face_attention_selected.csv", selected_rows)

    save_comparison_grid(
        selected,
        output_dir / "fig_gradcam_face_selected.pdf",
        output_dir / "fig_gradcam_face_selected.png",
        "Raw CLIP vs Ours patch relevance near the driver face",
    )
    # Copy to paper root for LaTeX-friendly inclusion if selected later.
    save_comparison_grid(
        selected,
        REPO_ROOT / "fig_gradcam_face_selected.pdf",
        REPO_ROOT / "fig_gradcam_face_selected.png",
        "Raw CLIP vs Ours patch relevance near the driver face",
    )

    summary = {
        "device": device,
        "checkpoint": args.aide_ckpt,
        "result_json": args.aide_result_json,
        "num_scanned": len(rows),
        "num_candidates": len(candidates),
        "num_strict_matches": len(strict),
        "top_k": args.top_k,
        "selection_mode": "strict_threshold" if strict else "fallback_top_delta_face_crop",
        "thresholds": {
            "min_ours_mass": args.min_ours_mass,
            "max_raw_mass": args.max_raw_mass,
            "min_delta": args.min_delta,
            "require_haar_face": args.require_haar_face,
            "bbox_mode": args.bbox_mode,
        },
        "selected_csv": str(output_dir / "face_attention_selected.csv"),
        "all_csv": str(output_dir / "face_attention_scan_all.csv"),
        "figure_pdf": str(output_dir / "fig_gradcam_face_selected.pdf"),
    }
    (output_dir / "face_attention_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
