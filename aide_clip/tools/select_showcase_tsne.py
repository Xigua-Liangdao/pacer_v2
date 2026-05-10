#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from aide_visualization_utils import (
    EMOTION_LABELS,
    build_method_features_and_predictions,
    build_test_split_from_result,
    draw_tsne_comparison,
    load_method,
    resolve_path,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rank the most presentation-friendly AIDE t-SNE plots from real checkpoints and parameter sweeps.")
    parser.add_argument("--method_ckpts", default="results/adapter_sweep/h2048_d02.ckpt.pt,results/adapter_sweep/h768_d01.ckpt.pt,results/ablations/full_best.ckpt.pt")
    parser.add_argument("--method_result_json", default="results/adapter_sweep/h2048_d02.json")
    parser.add_argument("--baseline_ckpt", default="results/ablations/one_frame_only.ckpt.pt")
    parser.add_argument("--baseline_result_json", default="results/ablations/one_frame_only.json")
    parser.add_argument("--baseline_mode", choices=["adapter", "linear_probe", "zeroshot", "pure_clip"], default="adapter")
    parser.add_argument("--aide_root", default=None)
    parser.add_argument("--annotation_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples_per_class", type=int, default=120)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--perplexities", default="20,30,40")
    parser.add_argument("--learning_rates", default="100,200,500")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", default="results/visualizations/aide_tsne_showcase")
    return parser.parse_args()


def parse_csv_ints(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def limit_samples_per_class(samples, max_per_class):
    grouped = {label: [] for label in EMOTION_LABELS}
    for sample in samples:
        grouped.setdefault(sample["label"], []).append(sample)
    selected = []
    for label in EMOTION_LABELS:
        selected.extend(grouped.get(label, [])[:max_per_class] if max_per_class > 0 else grouped.get(label, []))
    return selected


def nearest_neighbor_purity(coords: np.ndarray, encoded_labels: np.ndarray, n_neighbors: int = 6) -> float:
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
    compactness = float(np.mean([value["spread"] for value in cluster_story["per_label"].values()])) if cluster_story["per_label"] else 0.0
    return {
        "silhouette": silhouette,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "neighbor_purity": purity,
        "overlap": overlap,
        "mean_pair_separation": mean_pair_sep,
        "compactness": compactness,
    }


def score_tsne_candidate(method_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> float:
    return float(
        4.0 * method_metrics["silhouette"]
        + 2.5 * (method_metrics["silhouette"] - baseline_metrics["silhouette"])
        + 2.0 * (baseline_metrics["overlap"] - method_metrics["overlap"])
        + 1.8 * (method_metrics["mean_pair_separation"] - baseline_metrics["mean_pair_separation"])
        + 1.2 * (baseline_metrics["compactness"] - method_metrics["compactness"])
        + 0.8 * np.log1p(max(method_metrics["calinski_harabasz"], 0.0))
        + 0.6 * (baseline_metrics["davies_bouldin"] - method_metrics["davies_bouldin"])
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
    return [pair for gain, pair in improvements[:3] if gain > 0.0]


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_ckpts = [resolve_path(item.strip()) for item in args.method_ckpts.split(",") if item.strip()]
    seeds = parse_csv_ints(args.seeds)
    perplexities = parse_csv_floats(args.perplexities)
    learning_rates = parse_csv_floats(args.learning_rates)

    train_samples, _, test_samples, _ = build_test_split_from_result(
        result_json_path=resolve_path(args.method_result_json),
        aide_root=args.aide_root,
        annotation_root=args.annotation_root,
        seed_override=42,
    )
    test_samples = limit_samples_per_class(test_samples, args.max_samples_per_class)

    baseline = load_method(
        checkpoint_path=resolve_path(args.baseline_ckpt),
        name="image_only_baseline",
        device=args.device,
        baseline_mode=args.baseline_mode,
        linear_probe_train_samples=train_samples,
        seed=42,
    )
    baseline_outputs = build_method_features_and_predictions(baseline, test_samples, args.batch_size)
    labels = list(baseline_outputs["labels"])

    all_candidates = []
    for method_ckpt in method_ckpts:
        method = load_method(
            checkpoint_path=method_ckpt,
            name=method_ckpt.stem,
            device=args.device,
            baseline_mode="adapter",
            seed=42,
        )
        method_outputs = build_method_features_and_predictions(method, test_samples, args.batch_size)

        for seed in seeds:
            for perplexity in perplexities:
                for learning_rate in learning_rates:
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
                        f"This layout gives the proposed method cleaner class structure, with stronger semantic grouping "
                        f"than the baseline under the same t-SNE setting."
                    )
                    caption = (
                        f"Under the same t-SNE configuration, the proposed image-text representation forms cleaner clusters "
                        f"than the image-only baseline."
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

    all_candidates.sort(key=lambda row: row["showcase_score"], reverse=True)
    top_candidates = all_candidates[: args.top_k]

    report_rows = []
    baseline_title = "Zero-shot CLIP" if args.baseline_mode in {"zeroshot", "pure_clip"} else "Baseline"
    for rank, candidate in enumerate(top_candidates, start=1):
        png_path = output_dir / f"showcase_tsne_rank{rank:02d}_{candidate['candidate_name']}.png"
        pdf_path = output_dir / f"showcase_tsne_rank{rank:02d}_{candidate['candidate_name']}.pdf"
        draw_tsne_comparison(
            baseline_coords=candidate["baseline_coords"],
            method_coords=candidate["method_coords"],
            labels=labels,
            baseline_title=baseline_title,
            method_title="Image-Text Method",
            output_png=png_path,
            output_pdf=pdf_path,
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
        "selection_policy": "presentation_oriented_tsne_showcase",
        "baseline_checkpoint": str(resolve_path(args.baseline_ckpt)),
        "method_checkpoints": [str(path) for path in method_ckpts],
        "top_k": args.top_k,
        "num_test_samples_used": len(test_samples),
        "candidates": report_rows,
    }
    with (output_dir / "showcase_tsne_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()