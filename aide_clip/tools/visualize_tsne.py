#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from aide_visualization_utils import (
    build_method_features_and_predictions,
    build_test_split_from_result,
    draw_tsne_comparison,
    draw_tsne_plot,
    limit_samples_per_class,
    load_method,
    resolve_path,
    save_tsne_coordinates,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AIDE t-SNE comparisons for image-text and image-only methods.")
    parser.add_argument("--method_ckpt", default="results/adapter_sweep/h2048_d02.ckpt.pt")
    parser.add_argument("--method_result_json", default="results/adapter_sweep/h2048_d02.json")
    parser.add_argument("--baseline_ckpt", default="results/ablations/one_frame_only.ckpt.pt")
    parser.add_argument("--baseline_result_json", default="results/ablations/one_frame_only.json")
    parser.add_argument("--baseline_mode", choices=["adapter", "linear_probe", "zeroshot", "pure_clip"], default="adapter")
    parser.add_argument("--aide_root", default=None)
    parser.add_argument("--annotation_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples_per_class", type=int, default=120)
    parser.add_argument("--confidence_topk_per_class", type=int, default=0)
    parser.add_argument("--confidence_source", choices=["method", "baseline"], default="method")
    parser.add_argument("--confidence_correct_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", type=float, default=200.0)
    parser.add_argument("--tsne_iterations", type=int, default=2000)
    parser.add_argument("--plot_style", choices=["default", "block"], default="default")
    parser.add_argument("--baseline_title", default=None)
    parser.add_argument("--method_title", default="Image-Text Method")
    parser.add_argument("--compute_silhouette", action="store_true")
    parser.add_argument("--output_dir", default="results/visualizations/aide_tsne")
    return parser.parse_args()


def to_coordinate_rows(samples, coords, labels):
    rows = []
    for sample, xy, label in zip(samples, coords, labels):
        rows.append(
            {
                "sequence_id": sample["sequence_id"],
                "label": label,
                "x": round(float(xy[0]), 8),
                "y": round(float(xy[1]), 8),
            }
        )
    return rows


def select_high_confidence_subset(
    samples: List[Dict],
    labels: List[str],
    method_outputs: Dict[str, object],
    baseline_outputs: Dict[str, object],
    topk_per_class: int,
    source: str,
    correct_only: bool,
) -> List[int]:
    if topk_per_class <= 0:
        return list(range(len(samples)))

    source_outputs = method_outputs if source == "method" else baseline_outputs
    grouped: Dict[str, List[tuple]] = {}
    for idx, label in enumerate(labels):
        confidence = float(source_outputs["confidences"][idx])
        predicted = str(source_outputs["predictions"][idx])
        if correct_only and predicted != label:
            continue
        grouped.setdefault(label, []).append((confidence, idx))

    selected_indices = []
    for label in sorted(grouped.keys()):
        ranked = sorted(grouped[label], key=lambda item: item[0], reverse=True)
        selected_indices.extend(idx for _, idx in ranked[:topk_per_class])
    return sorted(selected_indices)


def subset_output_rows(outputs: Dict[str, object], indices: List[int]) -> Dict[str, object]:
    return {
        "features": outputs["features"][indices],
        "labels": [outputs["labels"][idx] for idx in indices],
        "predictions": [outputs["predictions"][idx] for idx in indices],
        "confidences": [outputs["confidences"][idx] for idx in indices],
    }


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
    test_samples = limit_samples_per_class(test_samples, args.max_samples_per_class, args.seed)

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

    selected_indices = select_high_confidence_subset(
        samples=test_samples,
        labels=method_outputs["labels"],
        method_outputs=method_outputs,
        baseline_outputs=baseline_outputs,
        topk_per_class=args.confidence_topk_per_class,
        source=args.confidence_source,
        correct_only=args.confidence_correct_only,
    )
    if len(selected_indices) != len(test_samples):
        test_samples = [test_samples[idx] for idx in selected_indices]
        method_outputs = subset_output_rows(method_outputs, selected_indices)
        baseline_outputs = subset_output_rows(baseline_outputs, selected_indices)

    labels = method_outputs["labels"]
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.tsne_perplexity,
        "learning_rate": args.tsne_learning_rate,
        "n_iter": args.tsne_iterations,
        "random_state": args.seed,
        "init": "pca",
    }
    baseline_coords = TSNE(**tsne_kwargs).fit_transform(baseline_outputs["features"])
    method_coords = TSNE(**tsne_kwargs).fit_transform(method_outputs["features"])
    baseline_title = args.baseline_title or ("Zero-shot CLIP" if args.baseline_mode in {"zeroshot", "pure_clip"} else "Baseline")
    method_title = args.method_title

    draw_tsne_plot(
        coords=baseline_coords,
        labels=labels,
        title=baseline_title,
        output_png=output_dir / "aide_tsne_baseline.png",
        output_pdf=output_dir / "aide_tsne_baseline.pdf",
        plot_style=args.plot_style,
    )
    draw_tsne_plot(
        coords=method_coords,
        labels=labels,
        title=method_title,
        output_png=output_dir / "aide_tsne_method.png",
        output_pdf=output_dir / "aide_tsne_method.pdf",
        plot_style=args.plot_style,
    )
    draw_tsne_comparison(
        baseline_coords=baseline_coords,
        method_coords=method_coords,
        labels=labels,
        baseline_title=baseline_title,
        method_title=method_title,
        output_png=output_dir / "aide_tsne_comparison.png",
        output_pdf=output_dir / "aide_tsne_comparison.pdf",
        plot_style=args.plot_style,
    )

    save_tsne_coordinates(output_dir / "aide_tsne_baseline_coords.csv", to_coordinate_rows(test_samples, baseline_coords, labels))
    save_tsne_coordinates(output_dir / "aide_tsne_baseline_coords.json", to_coordinate_rows(test_samples, baseline_coords, labels))
    save_tsne_coordinates(output_dir / "aide_tsne_method_coords.csv", to_coordinate_rows(test_samples, method_coords, labels))
    save_tsne_coordinates(output_dir / "aide_tsne_method_coords.json", to_coordinate_rows(test_samples, method_coords, labels))

    summary = {
        "method_checkpoint": str(method_ckpt),
        "baseline_checkpoint": str(baseline_ckpt),
        "method_result_json": str(method_result_json),
        "baseline_result_json": str(baseline_result_json),
        "num_test_samples_used": len(test_samples),
        "max_samples_per_class": args.max_samples_per_class,
        "confidence_topk_per_class": args.confidence_topk_per_class,
        "confidence_source": args.confidence_source,
        "confidence_correct_only": args.confidence_correct_only,
        "plot_style": args.plot_style,
        "seed": args.seed,
        "tsne": tsne_kwargs,
    }
    if args.compute_silhouette:
        from sklearn.preprocessing import LabelEncoder

        encoded = LabelEncoder().fit_transform(labels)
        summary["silhouette"] = {
            "baseline": round(float(silhouette_score(baseline_outputs["features"], encoded)), 6),
            "method": round(float(silhouette_score(method_outputs["features"], encoded)), 6),
        }

    with (output_dir / "aide_tsne_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
