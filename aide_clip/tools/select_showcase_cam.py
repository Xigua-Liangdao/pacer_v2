#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from aide_visualization_utils import (
    build_method_features_and_predictions,
    build_test_split_from_result,
    evaluate_cam_showcase_candidate,
    format_frame_index,
    generate_patch_heatmap,
    load_method,
    resolve_path,
    save_cam_comparison,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Select the most presentation-friendly AIDE CAM examples without changing predictions or heatmaps.")
    parser.add_argument("--method_ckpt", default="results/adapter_sweep/h2048_d02.ckpt.pt")
    parser.add_argument("--method_result_json", default="results/adapter_sweep/h2048_d02.json")
    parser.add_argument("--baseline_ckpt", default="results/ablations/one_frame_only.ckpt.pt")
    parser.add_argument("--baseline_result_json", default="results/ablations/one_frame_only.json")
    parser.add_argument("--baseline_mode", choices=["adapter", "linear_probe", "zeroshot", "pure_clip"], default="adapter")
    parser.add_argument("--aide_root", default=None)
    parser.add_argument("--annotation_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_mode", choices=["predicted", "ground_truth"], default="predicted")
    parser.add_argument("--selection_mode", choices=["showcase", "face_shift_only", "face_contrast"], default="face_shift_only")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--label_filter", default="", help="Comma-separated ground-truth labels to keep, e.g. Anxiety,Anger")
    parser.add_argument("--require_baseline_wrong", action="store_true", help="Keep only samples where baseline is wrong and method is correct.")
    parser.add_argument("--min_focus_gap", type=float, default=0.0,
                        help="Minimum (method_focus_mass - baseline_focus_mass) to keep. "
                             "Only used with face_contrast mode. Try 0.05~0.15.")
    parser.add_argument("--output_dir", default="results/visualizations/aide_cam_showcase")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    method_result_json = resolve_path(args.method_result_json)
    method_ckpt = resolve_path(args.method_ckpt)
    baseline_ckpt = resolve_path(args.baseline_ckpt)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, _, test_samples, _ = build_test_split_from_result(
        result_json_path=method_result_json,
        aide_root=args.aide_root,
        annotation_root=args.annotation_root,
        seed_override=args.seed,
    )

    method = load_method(
        checkpoint_path=method_ckpt,
        name="image_text_method",
        device=args.device,
        baseline_mode="adapter",
        seed=args.seed,
    )
    baseline = load_method(
        checkpoint_path=baseline_ckpt,
        name="image_only_baseline",
        device=args.device,
        baseline_mode=args.baseline_mode,
        linear_probe_train_samples=train_samples,
        seed=args.seed,
    )

    method_outputs = build_method_features_and_predictions(method, test_samples, args.batch_size)
    baseline_outputs = build_method_features_and_predictions(baseline, test_samples, args.batch_size)

    label_filter = {item.strip() for item in args.label_filter.split(",") if item.strip()}

    candidates = []
    for idx, sample in enumerate(test_samples):
        if label_filter and sample["label"] not in label_filter:
            continue
        target_label = sample["label"] if args.target_mode == "ground_truth" else None
        baseline_heatmap = generate_patch_heatmap(baseline, sample, target_label=target_label)
        method_heatmap = generate_patch_heatmap(method, sample, target_label=target_label)
        candidate = evaluate_cam_showcase_candidate(sample, baseline_heatmap, method_heatmap)
        candidate["method_prediction_from_split_pass"] = method_outputs["predictions"][idx]
        candidate["baseline_prediction_from_split_pass"] = baseline_outputs["predictions"][idx]
        candidate["method_confidence_from_split_pass"] = round(float(method_outputs["confidences"][idx]), 6)
        candidate["baseline_confidence_from_split_pass"] = round(float(baseline_outputs["confidences"][idx]), 6)
        candidate["frame_index"] = format_frame_index(sample["frame_path"])
        candidate["baseline_heatmap"] = baseline_heatmap
        candidate["method_heatmap"] = method_heatmap
        if args.require_baseline_wrong and not (candidate["method_correct"] and (not candidate["baseline_correct"])):
            continue
        candidates.append(candidate)

    # ---- Diagnostic dump: inspect raw metrics for all candidates ----
    import csv
    diag_path = output_dir / "candidate_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "method_correct", "baseline_correct",
            "method_focus_mass", "baseline_focus_mass", "focus_gap",
            "method_bg_mass", "baseline_bg_mass",
            "face_shift_advantage", "method_more_face_centered",
            "exclude_reason", "showcase_score",
        ])
        w.writeheader()
        for c in sorted(candidates,
                        key=lambda r: r["raw_metrics"]["method_focus_mass"] - r["raw_metrics"]["baseline_focus_mass"],
                        reverse=True):
            w.writerow({
                "sample_id": c["sample"]["sequence_id"],
                "method_correct": c["method_correct"],
                "baseline_correct": c["baseline_correct"],
                "method_focus_mass": round(c["raw_metrics"]["method_focus_mass"], 4),
                "baseline_focus_mass": round(c["raw_metrics"]["baseline_focus_mass"], 4),
                "focus_gap": round(c["raw_metrics"]["method_focus_mass"] - c["raw_metrics"]["baseline_focus_mass"], 4),
                "method_bg_mass": round(c["raw_metrics"]["method_bg_mass"], 4),
                "baseline_bg_mass": round(c["raw_metrics"]["baseline_bg_mass"], 4),
                "face_shift_advantage": c["face_shift_advantage"],
                "method_more_face_centered": c["method_more_face_centered"],
                "exclude_reason": c["exclude_reason"],
                "showcase_score": c["showcase_score"],
            })

    if args.selection_mode == "face_shift_only":
        ranked = [
            row
            for row in candidates
            if row["face_shift_advantage"] and row["method_correct"] and (not row["baseline_correct"] or row["method_more_face_centered"])
        ]
        ranked.sort(
            key=lambda row: (
                row["scores"]["correctness_advantage"],
                row["raw_metrics"]["method_focus_mass"] - row["raw_metrics"]["baseline_focus_mass"],
                row["raw_metrics"]["baseline_bg_mass"] - row["raw_metrics"]["method_bg_mass"],
                row["showcase_score"],
            ),
            reverse=True,
        )
    elif args.selection_mode == "face_contrast":
        # ---- Bypass face_shift_advantage flag entirely ----
        # Only require: method is correct, and method puts more mass on
        # the face region than the baseline (raw metric, no thresholding).
        ranked = [
            row
            for row in candidates
            if row["method_correct"]
               and (row["raw_metrics"]["method_focus_mass"] - row["raw_metrics"]["baseline_focus_mass"]) > args.min_focus_gap
        ]
        # Rank by largest face-focus gap first (most visually dramatic)
        ranked.sort(
            key=lambda row: (
                row["raw_metrics"]["method_focus_mass"] - row["raw_metrics"]["baseline_focus_mass"],
                row["raw_metrics"]["baseline_bg_mass"] - row["raw_metrics"]["method_bg_mass"],
                row["scores"]["correctness_advantage"],
            ),
            reverse=True,
        )
    else:
        ranked = [row for row in candidates if row["exclude_reason"] is None]
        ranked.sort(key=lambda row: row["showcase_score"], reverse=True)
    top_ranked = ranked[: args.top_k] if args.top_k > 0 else ranked

    report_rows = []
    for rank, candidate in enumerate(top_ranked, start=1):
        sample = candidate["sample"]
        png_path = output_dir / f"showcase_cam_rank{rank:02d}_{sample['sequence_id']}.png"
        pdf_path = output_dir / f"showcase_cam_rank{rank:02d}_{sample['sequence_id']}.pdf"
        save_cam_comparison(
            sample=sample,
            baseline_heatmap=candidate["baseline_heatmap"],
            method_heatmap=candidate["method_heatmap"],
            output_png=png_path,
            output_pdf=pdf_path,
        )
        report_rows.append(
            {
                "rank": rank,
                "sample_id": sample["sequence_id"],
                "frame_path": sample["frame_path"],
                "frame_index": candidate["frame_index"],
                "ground_truth_label": sample["label"],
                "baseline_prediction": candidate["baseline_pred"],
                "baseline_confidence": round(float(candidate["baseline_confidence"]), 6),
                "image_text_prediction": candidate["method_pred"],
                "image_text_confidence": round(float(candidate["method_confidence"]), 6),
                "showcase_score": candidate["showcase_score"],
                "scores": candidate["scores"],
                "short_explanation": candidate["short_explanation"],
                "method_more_face_centered": bool(candidate["method_more_face_centered"]),
                "face_shift_advantage": bool(candidate["face_shift_advantage"]),
                "method_focus_mass": candidate["raw_metrics"]["method_focus_mass"],
                "baseline_focus_mass": candidate["raw_metrics"]["baseline_focus_mass"],
                "method_bg_mass": candidate["raw_metrics"]["method_bg_mass"],
                "baseline_bg_mass": candidate["raw_metrics"]["baseline_bg_mass"],
                "paper_caption": candidate["paper_caption"],
                "is_visually_persuasive_top10": rank <= 10,
                "png": str(png_path),
                "pdf": str(pdf_path),
            }
        )

    excluded_rows = []
    for candidate in sorted(candidates, key=lambda row: row["showcase_score"], reverse=True):
        if candidate["exclude_reason"] is None:
            continue
        sample = candidate["sample"]
        excluded_rows.append(
            {
                "sample_id": sample["sequence_id"],
                "ground_truth_label": sample["label"],
                "baseline_prediction": candidate["baseline_pred"],
                "image_text_prediction": candidate["method_pred"],
                "showcase_score": candidate["showcase_score"],
                "exclude_reason": candidate["exclude_reason"],
            }
        )

    report = {
        "method_checkpoint": str(method_ckpt),
        "baseline_checkpoint": str(baseline_ckpt),
        "baseline_mode": args.baseline_mode,
        "target_mode": args.target_mode,
        "selection_mode": args.selection_mode,
        "label_filter": sorted(label_filter),
        "require_baseline_wrong": bool(args.require_baseline_wrong),
        "top_k": args.top_k,
        "selection_policy": "presentation_oriented_showcase",
        "priorities": [
            "method correct while baseline wrong",
            "method more face-centered and baseline more context-biased",
            "visually obvious attribution differences",
            "clean and publication-friendly frames",
        ],
        "selected": report_rows,
        "excluded_summary": excluded_rows[:50],
    }
    with (output_dir / "showcase_cam_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()