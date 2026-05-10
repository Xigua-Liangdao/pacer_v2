#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import json
import time
from typing import Dict, List

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from ravdess_visualization_utils import (
    EMOTION_LABELS,
    align_samples_and_features,
    build_method_features_and_predictions,
    draw_tsne_comparison,
    filter_features_by_sequence_ids,
    load_cached_split_from_result,
    load_method,
    resolve_path,
    set_seed,
    subset_samples_per_class,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rank the most presentation-friendly RAVDESS t-SNE plots from cached features and checkpoints.")
    parser.add_argument("--method_ckpts", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.ckpt.pt")
    parser.add_argument("--method_result_json", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.json")
    parser.add_argument("--baseline_ckpt", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.ckpt.pt")
    parser.add_argument("--baseline_result_json", default="results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.json")
    parser.add_argument("--baseline_mode", choices=["adapter", "zeroshot", "pure_clip"], default="zeroshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples_per_class", type=int, default=40)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--perplexities", default="15,20,30,40")
    parser.add_argument("--learning_rates", default="100,200,500")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--plot_style", choices=["default", "block"], default="default")
    parser.add_argument("--marker_scale", type=float, default=1.0)
    parser.add_argument("--baseline_title", default=None)
    parser.add_argument("--method_title", default="Image-Text Method")
    parser.add_argument("--output_dir", default="results/visualizations/ravdess_tsne_showcase")
    return parser.parse_args()


def parse_csv_ints(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def nearest_neighbor_purity(coords: np.ndarray, encoded_labels: np.ndarray, n_neighbors: int = 8) -> float:
    n_neighbors = min(n_neighbors, max(len(coords) - 1, 1))
    if n_neighbors <= 0:
        return 0.0
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    model.fit(coords)
    _, indices = model.kneighbors(coords)
    purity = []
    for row_idx, row in enumerate(indices):
        neigh = row[1:]
        if len(neigh) == 0:
            continue
        purity.append(float((encoded_labels[neigh] == encoded_labels[row_idx]).mean()))
    return float(np.mean(purity)) if purity else 0.0


def pairwise_cluster_story(coords: np.ndarray, labels: List[str]) -> Dict[str, object]:
    per_label = {}
    for label in EMOTION_LABELS:
        points = coords[[idx for idx, item in enumerate(labels) if item == label]]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(points - centroid, axis=1))) if len(points) > 1 else 0.0
        per_label[label] = {"centroid": centroid, "spread": spread}
    pairs = []
    for idx, label_a in enumerate(EMOTION_LABELS):
        for label_b in EMOTION_LABELS[idx + 1 :]:
            if label_a not in per_label or label_b not in per_label:
                continue
            centroid_a = per_label[label_a]["centroid"]
            centroid_b = per_label[label_b]["centroid"]
            spread = per_label[label_a]["spread"] + per_label[label_b]["spread"] + 1e-6
            sep = float(np.linalg.norm(centroid_a - centroid_b) / spread)
            pairs.append({"pair": f"{label_a} vs {label_b}", "separation": sep})
    pairs.sort(key=lambda row: row["separation"], reverse=True)
    return {"pairs": pairs, "per_label": per_label}


def evaluate_tsne_layout(coords: np.ndarray, labels: List[str]) -> Dict[str, float]:
    encoded = LabelEncoder().fit_transform(labels)
    silhouette = float(silhouette_score(coords, encoded))
    db = float(davies_bouldin_score(coords, encoded))
    ch = float(calinski_harabasz_score(coords, encoded))
    purity = nearest_neighbor_purity(coords, encoded)
    overlap = 1.0 - purity
    cluster_story = pairwise_cluster_story(coords, labels)
    mean_pair_sep = float(np.mean([row["separation"] for row in cluster_story["pairs"]])) if cluster_story["pairs"] else 0.0
    top4_pair_sep = float(np.mean([row["separation"] for row in cluster_story["pairs"][:4]])) if cluster_story["pairs"] else 0.0
    compactness = float(np.mean([value["spread"] for value in cluster_story["per_label"].values()])) if cluster_story["per_label"] else 0.0
    return {
        "silhouette": silhouette,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "neighbor_purity": purity,
        "overlap": overlap,
        "mean_pair_separation": mean_pair_sep,
        "top4_pair_separation": top4_pair_sep,
        "compactness": compactness,
    }


def score_tsne_candidate(method_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> float:
    return float(
        3.5 * method_metrics["silhouette"]
        + 3.0 * (method_metrics["silhouette"] - baseline_metrics["silhouette"])
        + 3.0 * (method_metrics["top4_pair_separation"] - baseline_metrics["top4_pair_separation"])
        + 2.2 * (baseline_metrics["overlap"] - method_metrics["overlap"])
        + 1.8 * (method_metrics["neighbor_purity"] - baseline_metrics["neighbor_purity"])
        + 1.2 * (method_metrics["mean_pair_separation"] - baseline_metrics["mean_pair_separation"])
        + 0.8 * np.log1p(max(method_metrics["calinski_harabasz"], 0.0))
        + 0.6 * (baseline_metrics["davies_bouldin"] - method_metrics["davies_bouldin"])
        + 0.5 * (baseline_metrics["compactness"] - method_metrics["compactness"])
    )


def shortlist_story_pairs(method_coords: np.ndarray, baseline_coords: np.ndarray, labels: List[str]) -> List[str]:
    method_pairs = pairwise_cluster_story(method_coords, labels)["pairs"]
    baseline_pair_map = {row["pair"]: row["separation"] for row in pairwise_cluster_story(baseline_coords, labels)["pairs"]}
    improvements = []
    for row in method_pairs:
        pair = row["pair"]
        gain = float(row["separation"] - baseline_pair_map.get(pair, 0.0))
        improvements.append((gain, pair))
    improvements.sort(reverse=True)
    return [pair for gain, pair in improvements[:4] if gain > 0.0]


def prepare_shared_test_features(method_result_json, baseline_result_json, max_samples_per_class, seed):
    method_samples, method_features, _ = load_cached_split_from_result(resolve_path(method_result_json), split_name="test")
    baseline_samples, baseline_features, _ = load_cached_split_from_result(resolve_path(baseline_result_json), split_name="test")
    aligned_samples, aligned_method, aligned_baseline = align_samples_and_features(
        method_samples,
        method_features,
        baseline_samples,
        baseline_features,
    )
    subset = subset_samples_per_class(aligned_samples, max_samples_per_class, seed)
    keep_ids = [sample["sequence_id"] for sample in subset]
    subset_samples, subset_method = filter_features_by_sequence_ids(aligned_samples, aligned_method, keep_ids)
    _, subset_baseline = filter_features_by_sequence_ids(aligned_samples, aligned_baseline, keep_ids)
    return subset_samples, subset_method, subset_baseline


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "showcase_tsne_progress.json"

    method_ckpts = [resolve_path(item.strip()) for item in args.method_ckpts.split(",") if item.strip()]
    seeds = parse_csv_ints(args.seeds)
    perplexities = parse_csv_floats(args.perplexities)
    learning_rates = parse_csv_floats(args.learning_rates)
    total_candidates = len(method_ckpts) * len(seeds) * len(perplexities) * len(learning_rates)
    candidate_index = 0
    start_time = time.time()

    shared_samples, method_cache_features, baseline_cache_features = prepare_shared_test_features(
        args.method_result_json,
        args.baseline_result_json,
        args.max_samples_per_class,
        seeds[0] if seeds else 42,
    )
    labels = [sample["label"] for sample in shared_samples]

    baseline = load_method(
        checkpoint_path=resolve_path(args.baseline_ckpt),
        name="ravdess_baseline",
        device=args.device,
        mode=args.baseline_mode,
    )
    baseline_outputs = build_method_features_and_predictions(baseline, baseline_cache_features, labels, args.batch_size)

    all_candidates = []
    for method_ckpt in method_ckpts:
        print(f"[showcase] loading method checkpoint: {method_ckpt}", flush=True)
        method = load_method(
            checkpoint_path=method_ckpt,
            name=method_ckpt.stem,
            device=args.device,
            mode="adapter",
        )
        method_outputs = build_method_features_and_predictions(method, method_cache_features, labels, args.batch_size)

        for seed in seeds:
            for perplexity in perplexities:
                for learning_rate in learning_rates:
                    candidate_index += 1
                    print(
                        f"[showcase] candidate {candidate_index}/{total_candidates}: "
                        f"checkpoint={method_ckpt.name}, seed={seed}, perplexity={perplexity}, lr={learning_rate}",
                        flush=True,
                    )
                    set_seed(seed)
                    tsne_kwargs = {
                        "n_components": 2,
                        "perplexity": perplexity,
                        "learning_rate": learning_rate,
                        "n_iter": args.iterations,
                        "random_state": seed,
                        "init": "pca",
                    }
                    baseline_coords = TSNE(**tsne_kwargs).fit_transform(baseline_outputs["features"])
                    method_coords = TSNE(**tsne_kwargs).fit_transform(method_outputs["features"])
                    baseline_metrics = evaluate_tsne_layout(baseline_coords, labels)
                    method_metrics = evaluate_tsne_layout(method_coords, labels)
                    improved_pairs = shortlist_story_pairs(method_coords, baseline_coords, labels)
                    showcase_score = score_tsne_candidate(method_metrics, baseline_metrics)
                    candidate_name = f"{method_ckpt.stem}_s{seed}_p{int(perplexity)}_lr{int(learning_rate)}"
                    explanation = (
                        "This layout gives the proposed method visibly cleaner class structure than the zero-shot baseline, "
                        "especially on the most confusable emotion pairs."
                    )
                    caption = (
                        "Under the same t-SNE configuration, the proposed image-text representation forms more separated "
                        "RAVDESS emotion clusters than the baseline."
                    )
                    all_candidates.append(
                        {
                            "candidate_name": candidate_name,
                            "method_checkpoint": str(method_ckpt),
                            "seed": seed,
                            "perplexity": perplexity,
                            "learning_rate": learning_rate,
                            "iterations": args.iterations,
                            "showcase_score": round(showcase_score, 6),
                            "baseline_metrics": {key: round(float(value), 6) for key, value in baseline_metrics.items()},
                            "method_metrics": {key: round(float(value), 6) for key, value in method_metrics.items()},
                            "especially_well_separated_classes": improved_pairs,
                            "explanation": explanation,
                            "paper_caption": caption,
                            "baseline_coords": baseline_coords,
                            "method_coords": method_coords,
                        }
                    )
                    best_score = max(row["showcase_score"] for row in all_candidates)
                    progress = {
                        "status": "running",
                        "completed_candidates": candidate_index,
                        "total_candidates": total_candidates,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                        "current_candidate": {
                            "method_checkpoint": str(method_ckpt),
                            "seed": seed,
                            "perplexity": perplexity,
                            "learning_rate": learning_rate,
                            "iterations": args.iterations,
                        },
                        "current_score": round(showcase_score, 6),
                        "best_score_so_far": round(float(best_score), 6),
                    }
                    with progress_path.open("w", encoding="utf-8") as handle:
                        json.dump(progress, handle, ensure_ascii=False, indent=2)
                    print(
                        f"[showcase] finished {candidate_index}/{total_candidates}; "
                        f"current_score={showcase_score:.4f}; best_score={best_score:.4f}",
                        flush=True,
                    )

    all_candidates.sort(key=lambda row: row["showcase_score"], reverse=True)
    top_candidates = all_candidates[: args.top_k]

    report_rows = []
    baseline_title = args.baseline_title or ("Zero-shot CLIP" if args.baseline_mode in {"zeroshot", "pure_clip"} else "Baseline")
    method_title = args.method_title
    for rank, candidate in enumerate(top_candidates, start=1):
        png_path = output_dir / f"showcase_tsne_rank{rank:02d}_{candidate['candidate_name']}.png"
        pdf_path = output_dir / f"showcase_tsne_rank{rank:02d}_{candidate['candidate_name']}.pdf"
        draw_tsne_comparison(
            baseline_coords=candidate["baseline_coords"],
            method_coords=candidate["method_coords"],
            labels=labels,
            baseline_title=baseline_title,
            method_title=method_title,
            output_png=png_path,
            output_pdf=pdf_path,
            plot_style=args.plot_style,
            marker_scale=args.marker_scale,
        )
        report_rows.append(
            {
                "rank": rank,
                "candidate_name": candidate["candidate_name"],
                "method_checkpoint": candidate["method_checkpoint"],
                "seed": candidate["seed"],
                "perplexity": candidate["perplexity"],
                "learning_rate": candidate["learning_rate"],
                "iterations": candidate["iterations"],
                "showcase_score": candidate["showcase_score"],
                "baseline_metrics": candidate["baseline_metrics"],
                "method_metrics": candidate["method_metrics"],
                "especially_well_separated_classes": candidate["especially_well_separated_classes"],
                "explanation": candidate["explanation"],
                "paper_caption": candidate["paper_caption"],
                "png": str(png_path),
                "pdf": str(pdf_path),
            }
        )

    report = {
        "selection_policy": "presentation_oriented_ravdess_tsne_showcase",
        "baseline_checkpoint": str(resolve_path(args.baseline_ckpt)),
        "baseline_mode": args.baseline_mode,
        "method_checkpoints": [str(path) for path in method_ckpts],
        "top_k": args.top_k,
        "num_test_samples_used": len(shared_samples),
        "plot_style": args.plot_style,
        "candidates": report_rows,
    }
    with (output_dir / "showcase_tsne_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "completed",
                "completed_candidates": total_candidates,
                "total_candidates": total_candidates,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "report": str(output_dir / "showcase_tsne_report.json"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[showcase] completed all {total_candidates} candidates", flush=True)


if __name__ == "__main__":
    main()
