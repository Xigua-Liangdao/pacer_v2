#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import json

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from ravdess_visualization_utils import (
    build_method_features_and_predictions,
    draw_tsne_comparison,
    draw_tsne_plot,
    filter_features_by_sequence_ids,
    load_cached_split_from_result,
    load_method,
    resolve_path,
    save_tsne_coordinates,
    set_seed,
    subset_samples_per_class,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAVDESS t-SNE comparisons for image-text and zero-shot baselines.")
    parser.add_argument("--method_ckpt", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.ckpt.pt")
    parser.add_argument("--method_result_json", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.json")
    parser.add_argument("--baseline_ckpt", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.ckpt.pt")
    parser.add_argument("--baseline_result_json", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.json")
    parser.add_argument("--baseline_mode", choices=["adapter", "zeroshot", "pure_clip"], default="zeroshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples_per_class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", type=float, default=200.0)
    parser.add_argument("--tsne_iterations", type=int, default=2500)
    parser.add_argument("--plot_style", choices=["default", "block"], default="default")
    parser.add_argument("--compute_silhouette", action="store_true")
    parser.add_argument("--output_dir", default="results/visualizations/ravdess_tsne")
    return parser.parse_args()


def to_coordinate_rows(samples, coords):
    rows = []
    for sample, xy in zip(samples, coords):
        rows.append(
            {
                "sequence_id": sample["sequence_id"],
                "label": sample["label"],
                "actor_id": sample.get("actor_id"),
                "video_path": sample.get("video_path"),
                "x": round(float(xy[0]), 8),
                "y": round(float(xy[1]), 8),
            }
        )
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_samples, method_cache_features, _ = load_cached_split_from_result(resolve_path(args.method_result_json), split_name="test")
    baseline_samples, baseline_cache_features, _ = load_cached_split_from_result(resolve_path(args.baseline_result_json), split_name="test")
    aligned_samples, method_cache_features, baseline_cache_features = filter_and_align(
        method_samples,
        method_cache_features,
        baseline_samples,
        baseline_cache_features,
        args.max_samples_per_class,
        args.seed,
    )

    labels = [sample["label"] for sample in aligned_samples]
    method = load_method(resolve_path(args.method_ckpt), name="ravdess_image_text_method", device=args.device, mode="adapter")
    baseline = load_method(resolve_path(args.baseline_ckpt), name="ravdess_baseline", device=args.device, mode=args.baseline_mode)

    method_outputs = build_method_features_and_predictions(method, method_cache_features, labels, args.batch_size)
    baseline_outputs = build_method_features_and_predictions(baseline, baseline_cache_features, labels, args.batch_size)

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
    baseline_title = "Zero-shot CLIP" if args.baseline_mode in {"zeroshot", "pure_clip"} else "Baseline"

    draw_tsne_plot(
        coords=baseline_coords,
        labels=labels,
        title=f"RAVDESS t-SNE: {baseline_title}",
        output_png=output_dir / "ravdess_tsne_baseline.png",
        output_pdf=output_dir / "ravdess_tsne_baseline.pdf",
        plot_style=args.plot_style,
    )
    draw_tsne_plot(
        coords=method_coords,
        labels=labels,
        title="RAVDESS t-SNE: Image-Text Method",
        output_png=output_dir / "ravdess_tsne_method.png",
        output_pdf=output_dir / "ravdess_tsne_method.pdf",
        plot_style=args.plot_style,
    )
    draw_tsne_comparison(
        baseline_coords=baseline_coords,
        method_coords=method_coords,
        labels=labels,
        baseline_title=baseline_title,
        method_title="Image-Text Method",
        output_png=output_dir / "ravdess_tsne_comparison.png",
        output_pdf=output_dir / "ravdess_tsne_comparison.pdf",
        plot_style=args.plot_style,
    )

    save_tsne_coordinates(output_dir / "ravdess_tsne_baseline_coords.csv", to_coordinate_rows(aligned_samples, baseline_coords))
    save_tsne_coordinates(output_dir / "ravdess_tsne_baseline_coords.json", to_coordinate_rows(aligned_samples, baseline_coords))
    save_tsne_coordinates(output_dir / "ravdess_tsne_method_coords.csv", to_coordinate_rows(aligned_samples, method_coords))
    save_tsne_coordinates(output_dir / "ravdess_tsne_method_coords.json", to_coordinate_rows(aligned_samples, method_coords))

    summary = {
        "method_checkpoint": str(resolve_path(args.method_ckpt)),
        "baseline_checkpoint": str(resolve_path(args.baseline_ckpt)),
        "method_result_json": str(resolve_path(args.method_result_json)),
        "baseline_result_json": str(resolve_path(args.baseline_result_json)),
        "baseline_mode": args.baseline_mode,
        "num_test_samples_used": len(aligned_samples),
        "max_samples_per_class": args.max_samples_per_class,
        "plot_style": args.plot_style,
        "seed": args.seed,
        "tsne": tsne_kwargs,
    }
    if args.compute_silhouette:
        encoded = LabelEncoder().fit_transform(labels)
        summary["silhouette"] = {
            "baseline": round(float(silhouette_score(baseline_outputs["features"], encoded)), 6),
            "method": round(float(silhouette_score(method_outputs["features"], encoded)), 6),
        }

    with (output_dir / "ravdess_tsne_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def filter_and_align(method_samples, method_cache_features, baseline_samples, baseline_cache_features, max_samples_per_class, seed):
    from ravdess_visualization_utils import align_samples_and_features

    aligned_samples, aligned_method, aligned_baseline = align_samples_and_features(
        method_samples,
        method_cache_features,
        baseline_samples,
        baseline_cache_features,
    )
    subset = subset_samples_per_class(aligned_samples, max_samples_per_class, seed)
    keep_ids = [sample["sequence_id"] for sample in subset]
    subset_samples, subset_method = filter_features_by_sequence_ids(aligned_samples, aligned_method, keep_ids)
    _, subset_baseline = filter_features_by_sequence_ids(aligned_samples, aligned_baseline, keep_ids)
    return subset_samples, subset_method, subset_baseline


if __name__ == "__main__":
    main()
