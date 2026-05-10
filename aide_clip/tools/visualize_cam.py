#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from aide_visualization_utils import (
    build_method_features_and_predictions,
    build_test_split_from_result,
    generate_patch_heatmap,
    load_method,
    rank_samples_for_visualization,
    resolve_path,
    save_cam_comparison,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AIDE CAM-like patch heatmap comparisons for CLIP-based models.")
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
    parser.add_argument("--selection_mode", choices=["correct_only", "top_confidence", "predefined_ids"], default="correct_only")
    parser.add_argument("--sample_ids", default="", help="Comma-separated sequence ids used when --selection_mode predefined_ids.")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--target_mode", choices=["predicted", "ground_truth"], default="predicted")
    parser.add_argument("--output_dir", default="results/visualizations/aide_cam")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    method_result_json = resolve_path(args.method_result_json)
    baseline_result_json = resolve_path(args.baseline_result_json)
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

    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    selected = rank_samples_for_visualization(
        samples=test_samples,
        baseline_outputs=baseline_outputs,
        method_outputs=method_outputs,
        selection_mode=args.selection_mode,
        top_k=args.top_k,
        sample_ids=sample_ids,
    )

    manifest = []
    for sample in selected:
        target_label = sample["label"] if args.target_mode == "ground_truth" else None
        baseline_heatmap = generate_patch_heatmap(baseline, sample, target_label=target_label)
        method_heatmap = generate_patch_heatmap(method, sample, target_label=target_label)
        png_path = output_dir / f"cam_compare_{sample['sequence_id']}.png"
        pdf_path = output_dir / f"cam_compare_{sample['sequence_id']}.pdf"
        save_cam_comparison(
            sample=sample,
            baseline_heatmap=baseline_heatmap,
            method_heatmap=method_heatmap,
            output_png=png_path,
            output_pdf=pdf_path,
        )
        manifest.append(
            {
                "sequence_id": sample["sequence_id"],
                "label": sample["label"],
                "frame_path": sample["frame_path"],
                "baseline_pred": baseline_heatmap["pred_label"],
                "baseline_confidence": round(float(baseline_heatmap["pred_confidence"]), 6),
                "method_pred": method_heatmap["pred_label"],
                "method_confidence": round(float(method_heatmap["pred_confidence"]), 6),
                "target_label": method_heatmap["target_label"],
                "png": str(png_path),
                "pdf": str(pdf_path),
            }
        )

    with (output_dir / "cam_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method_checkpoint": str(method_ckpt),
                "baseline_checkpoint": str(baseline_ckpt),
                "selection_mode": args.selection_mode,
                "target_mode": args.target_mode,
                "num_examples": len(manifest),
                "examples": manifest,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()