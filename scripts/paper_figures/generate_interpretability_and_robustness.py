#!/usr/bin/env python3
"""Generate real analysis figures for the driver-state paper revisions.

The script intentionally performs inference/extraction from existing
checkpoints and caches only. It does not train models and does not invent
baseline outputs.
"""

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
AIDE_TOOLS = REPO_ROOT / "aide_clip" / "tools"
AIDE_SRC = REPO_ROOT / "aide_clip" / "src"
for path in [str(AIDE_TOOLS), str(AIDE_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from aide_visualization_utils import (  # noqa: E402
    EMOTION_LABELS as AIDE_LABELS,
    build_method_features_and_predictions,
    build_test_split_from_result,
    generate_patch_heatmap,
    load_method,
    select_frame_paths,
    set_seed,
)
import clip_aide_emotion_train as aide_base  # noqa: E402


AIDE_COLORS = {
    "Anxiety": "#d62728",
    "Peace": "#15957f",
    "Weariness": "#2f72bd",
    "Happiness": "#2ca02c",
    "Anger": "#b94f90",
}
YAWDD_LABELS = ["notdrowsy", "drowsy"]
YAWDD_DISPLAY = {"notdrowsy": "non-yawning", "drowsy": "yawning cue"}
YAWDD_COLORS = {"notdrowsy": "#2f72bd", "drowsy": "#c76f00"}


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


def tsne_coords(features: np.ndarray, seed: int, perplexity: float, n_iter: int) -> np.ndarray:
    n = int(features.shape[0])
    effective_perplexity = min(float(perplexity), max(2.0, (n - 1) / 3.0))
    model = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=200.0,
        n_iter=int(n_iter),
        random_state=int(seed),
        init="pca",
    )
    return model.fit_transform(features.astype(np.float32))


def plot_tsne_panel(ax, coords: np.ndarray, labels: Sequence[str], ordered_labels: Sequence[str],
                    color_map: Dict[str, str], title: str, display_map: Dict[str, str] = None) -> None:
    for label in ordered_labels:
        idx = [i for i, item in enumerate(labels) if item == label]
        if not idx:
            continue
        display = display_map.get(label, label) if display_map else label
        subset = coords[idx]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=24,
            alpha=0.88,
            c=color_map[label],
            label=display,
            edgecolors="white",
            linewidths=0.18,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.45)


def save_tsne_rows(path: Path, samples: Sequence[Dict], labels: Sequence[str], coords: np.ndarray,
                   dataset: str, representation: str) -> None:
    rows = []
    for sample, label, xy in zip(samples, labels, coords):
        rows.append(
            {
                "dataset": dataset,
                "representation": representation,
                "sequence_id": sample.get("sequence_id", sample.get("video_path", "")),
                "label": label,
                "x": round(float(xy[0]), 8),
                "y": round(float(xy[1]), 8),
            }
        )
    write_csv(path, rows)


def generate_tsne(args, output_dir: Path, device: str) -> Dict:
    print("[tsne] loading AIDE cached features and split metadata", flush=True)
    _, _, aide_test_samples, _ = build_test_split_from_result(
        result_json_path=Path(args.aide_result_json),
        aide_root=args.aide_root,
        annotation_root=args.aide_annotation_root,
        seed_override=args.seed,
    )
    aide_cache = torch.load(args.aide_feature_cache, map_location="cpu")
    aide_ckpt = torch.load(args.aide_ckpt, map_location="cpu")
    aide_raw_features = aide_cache["test_x"].float()
    aide_labels = [AIDE_LABELS[int(idx)] for idx in aide_cache["test_y"].tolist()]
    if len(aide_test_samples) != int(aide_raw_features.shape[0]):
        raise RuntimeError(
            f"AIDE split/cache length mismatch: split={len(aide_test_samples)}, cache={aide_raw_features.shape[0]}"
        )
    if tuple(aide_cache["text_features"].shape) != tuple(aide_ckpt["text_features"].shape):
        raise RuntimeError(
            f"AIDE prompt/text cache mismatch: cache={tuple(aide_cache['text_features'].shape)}, "
            f"ckpt={tuple(aide_ckpt['text_features'].shape)}"
        )
    aide_cfg = dict(aide_ckpt.get("config", {}))
    aide_state = aide_ckpt["adapter_state_dict"]
    aide_adapter = aide_base.ClipImageAdapter(
        dim=int(aide_raw_features.shape[-1]),
        device=device,
        hidden_dim=int(aide_cfg.get("adapter_hidden_dim", 1024)),
        dropout=float(aide_cfg.get("adapter_dropout", 0.2)),
        num_classes=int(aide_ckpt["text_features"].shape[0]),
        num_prompts=int(aide_ckpt["text_features"].shape[1]),
        use_prompt_weight=bool(aide_state.get("use_prompt_weight", True)),
        use_class_temperature=bool(aide_state.get("use_class_temperature", True)),
        use_class_bias=bool(aide_state.get("use_class_bias", True)),
        adapter_mode=str(aide_state.get("adapter_mode", "full")),
    )
    aide_adapter.load_state_dict(aide_state)
    aide_adapter.eval()
    with torch.no_grad():
        aide_adapted_features = aide_adapter._adapt_image(aide_raw_features.to(device)).detach().cpu().numpy()
    aide_raw_features_np = aide_raw_features.numpy()

    print("[tsne] loading YawDD cached features and full-model checkpoint", flush=True)
    yawdd_cache = torch.load(args.yawdd_cache, map_location="cpu")
    yawdd_ckpt = torch.load(args.yawdd_ckpt, map_location="cpu")
    yawdd_features = yawdd_cache["features"].float()
    yawdd_samples = yawdd_cache["samples"]
    yawdd_labels = [sample["label"] for sample in yawdd_samples]
    text_features = yawdd_ckpt["text_features"].float()
    config = dict(yawdd_ckpt.get("config", {}))
    state = yawdd_ckpt["adapter_state_dict"]
    adapter = aide_base.ClipImageAdapter(
        dim=int(text_features.shape[-1]),
        device=device,
        hidden_dim=int(config.get("adapter_hidden_dim", 512)),
        dropout=float(config.get("adapter_dropout", 0.3)),
        num_classes=int(text_features.shape[0]),
        num_prompts=int(text_features.shape[1]),
        use_prompt_weight=bool(state.get("use_prompt_weight", True)),
        use_class_temperature=bool(state.get("use_class_temperature", True)),
        use_class_bias=bool(state.get("use_class_bias", True)),
        adapter_mode=str(state.get("adapter_mode", "full")),
    )
    adapter.load_state_dict(state)
    adapter.eval()
    with torch.no_grad():
        yawdd_adapted_features = adapter._adapt_image(yawdd_features.to(device)).detach().cpu().numpy()
    yawdd_raw_features = yawdd_features.numpy()

    print("[tsne] fitting TSNE", flush=True)
    aide_raw_coords = tsne_coords(aide_raw_features_np, args.seed, args.tsne_perplexity, args.tsne_iter)
    aide_adapted_coords = tsne_coords(aide_adapted_features, args.seed, args.tsne_perplexity, args.tsne_iter)
    yawdd_raw_coords = tsne_coords(yawdd_raw_features, args.seed, args.tsne_perplexity, args.tsne_iter)
    yawdd_adapted_coords = tsne_coords(yawdd_adapted_features, args.seed, args.tsne_perplexity, args.tsne_iter)

    save_tsne_rows(output_dir / "fig_tsne_aide_raw_coords.csv", aide_test_samples, aide_labels, aide_raw_coords, "AIDE", "raw_clip")
    save_tsne_rows(output_dir / "fig_tsne_aide_adapted_coords.csv", aide_test_samples, aide_labels, aide_adapted_coords, "AIDE", "adapted")
    save_tsne_rows(output_dir / "fig_tsne_yawdd_raw_coords.csv", yawdd_samples, yawdd_labels, yawdd_raw_coords, "YawDD", "raw_clip")
    save_tsne_rows(output_dir / "fig_tsne_yawdd_adapted_coords.csv", yawdd_samples, yawdd_labels, yawdd_adapted_coords, "YawDD", "adapted")
    np.save(output_dir / "fig_tsne_aide_raw_features.npy", aide_raw_features_np)
    np.save(output_dir / "fig_tsne_aide_adapted_features.npy", aide_adapted_features)
    np.save(output_dir / "fig_tsne_yawdd_raw_features.npy", yawdd_raw_features)
    np.save(output_dir / "fig_tsne_yawdd_adapted_features.npy", yawdd_adapted_features)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "figure.dpi": 180, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.0))
    plot_tsne_panel(axes[0, 0], aide_raw_coords, aide_labels, AIDE_LABELS, AIDE_COLORS, "AIDE raw CLIP")
    plot_tsne_panel(axes[0, 1], aide_adapted_coords, aide_labels, AIDE_LABELS, AIDE_COLORS, "AIDE adapted")
    plot_tsne_panel(axes[1, 0], yawdd_raw_coords, yawdd_labels, YAWDD_LABELS, YAWDD_COLORS, "YawDD raw CLIP", YAWDD_DISPLAY)
    plot_tsne_panel(axes[1, 1], yawdd_adapted_coords, yawdd_labels, YAWDD_LABELS, YAWDD_COLORS, "YawDD adapted", YAWDD_DISPLAY)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=True)
    handles2, labels2 = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles2, labels2, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.11, hspace=0.36, wspace=0.22)
    for suffix in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig_tsne.{suffix}", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(REPO_ROOT / f"fig_tsne.{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return {
        "aide_checkpoint": args.aide_ckpt,
        "aide_result_json": args.aide_result_json,
        "aide_feature_cache": args.aide_feature_cache,
        "aide_test_samples": len(aide_test_samples),
        "yawdd_checkpoint": args.yawdd_ckpt,
        "yawdd_cache": args.yawdd_cache,
        "yawdd_test_samples": len(yawdd_samples),
        "output_pdf": str(output_dir / "fig_tsne.pdf"),
    }


def choose_attention_samples(samples: Sequence[Dict], outputs: Dict, max_examples: int) -> List[Tuple[int, Dict]]:
    grouped: Dict[str, List[Tuple[float, int, Dict]]] = {label: [] for label in AIDE_LABELS}
    for idx, sample in enumerate(samples):
        label = sample["label"]
        pred = outputs["predictions"][idx]
        conf = float(outputs["confidences"][idx])
        if label == pred:
            grouped.setdefault(label, []).append((conf, idx, sample))
    selected = []
    for label in AIDE_LABELS:
        rows = sorted(grouped.get(label, []), key=lambda item: item[0], reverse=True)
        if rows:
            conf, idx, sample = rows[0]
            selected.append((idx, sample))
        if len(selected) >= max_examples:
            break
    return selected


def generate_attention(args, output_dir: Path, device: str) -> Dict:
    print("[attention] loading AIDE cache for sample selection", flush=True)
    _, _, samples, _ = build_test_split_from_result(
        result_json_path=Path(args.aide_result_json),
        aide_root=args.aide_root,
        annotation_root=args.aide_annotation_root,
        seed_override=args.seed,
    )
    aide_cache = torch.load(args.aide_feature_cache, map_location="cpu")
    aide_ckpt = torch.load(args.aide_ckpt, map_location="cpu")
    features = aide_cache["test_x"].float()
    labels = [AIDE_LABELS[int(idx)] for idx in aide_cache["test_y"].tolist()]
    cfg = dict(aide_ckpt.get("config", {}))
    state = aide_ckpt["adapter_state_dict"]
    adapter = aide_base.ClipImageAdapter(
        dim=int(features.shape[-1]),
        device=device,
        hidden_dim=int(cfg.get("adapter_hidden_dim", 1024)),
        dropout=float(cfg.get("adapter_dropout", 0.2)),
        num_classes=int(aide_ckpt["text_features"].shape[0]),
        num_prompts=int(aide_ckpt["text_features"].shape[1]),
        use_prompt_weight=bool(state.get("use_prompt_weight", True)),
        use_class_temperature=bool(state.get("use_class_temperature", True)),
        use_class_bias=bool(state.get("use_class_bias", True)),
        adapter_mode=str(state.get("adapter_mode", "full")),
    )
    adapter.load_state_dict(state)
    adapter.eval()
    with torch.no_grad():
        logits = adapter.logits(features.to(device), aide_ckpt["text_features"].to(device))
        probs = torch.softmax(logits, dim=-1).detach().cpu()
    pred_idx = probs.argmax(dim=-1).tolist()
    outputs = {
        "predictions": [AIDE_LABELS[int(idx)] for idx in pred_idx],
        "confidences": probs.max(dim=-1).values.tolist(),
    }
    print("[attention] loading CLIP only for selected patch heatmaps", flush=True)
    method = load_method(
        checkpoint_path=Path(args.aide_ckpt),
        name="aide_ours",
        device=device,
        baseline_mode="adapter",
        seed=args.seed,
    )
    selected = choose_attention_samples(samples, outputs, args.attention_examples)
    if not selected:
        raise RuntimeError("No correctly predicted AIDE samples were available for attention maps.")

    heatmaps = []
    manifest = []
    for idx, sample in selected:
        heat = generate_patch_heatmap(method, sample, target_label=sample["label"])
        heatmaps.append((sample, heat))
        np.save(output_dir / f"fig_gradcam_heatmap_{sample['sequence_id']}.npy", heat["heatmap"])
        manifest.append(
            {
                "sequence_id": sample["sequence_id"],
                "label": sample["label"],
                "prediction": heat["pred_label"],
                "confidence": round(float(heat["pred_confidence"]), 6),
                "target_label": heat["target_label"],
                "frame_path": sample["frame_path"],
                "method": "adapted_patch_text_similarity",
            }
        )
    write_csv(output_dir / "fig_gradcam_manifest.csv", manifest)

    n = len(heatmaps)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 180, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, n, figsize=(max(7.1, 1.55 * n), 3.2))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, (sample, heat) in enumerate(heatmaps):
        axes[0, col].imshow(heat["image_rgb"])
        axes[0, col].set_title(f"{sample['label']}\\nsource")
        axes[1, col].imshow(heat["overlay_rgb"])
        axes[1, col].set_title(f"pred {heat['pred_label']}\\n{heat['pred_confidence']:.2f}")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, hspace=0.08, wspace=0.04)
    for suffix in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig_gradcam.{suffix}", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(REPO_ROOT / f"fig_gradcam.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "aide_checkpoint": args.aide_ckpt,
        "aide_result_json": args.aide_result_json,
        "num_examples": len(manifest),
        "manifest": str(output_dir / "fig_gradcam_manifest.csv"),
        "output_pdf": str(output_dir / "fig_gradcam.pdf"),
        "method_note": "Patch relevance from adapted CLIP patch-text similarity, not supervised Grad-CAM.",
    }


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.uint8)


def corrupt_image(image: Image.Image, corruption: str, severity: int) -> Image.Image:
    if corruption == "clean":
        return image.convert("RGB")
    arr = pil_to_array(image)
    sev = int(severity)
    if corruption == "motion_blur":
        k = [5, 9, 15][sev - 1]
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return Image.fromarray(cv2.filter2D(arr, -1, kernel))
    if corruption == "low_light":
        factors = [0.70, 0.50, 0.32]
        out = np.clip(arr.astype(np.float32) * factors[sev - 1], 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if corruption == "occlusion":
        out = arr.copy()
        h, w = out.shape[:2]
        scales = [0.18, 0.26, 0.34]
        box_w = int(w * scales[sev - 1])
        box_h = int(h * scales[sev - 1])
        x0 = int(w * 0.5 - box_w / 2)
        y0 = int(h * 0.30 - box_h / 2)
        out[max(0, y0):min(h, y0 + box_h), max(0, x0):min(w, x0 + box_w)] = 0
        return Image.fromarray(out)
    if corruption == "jpeg":
        qualities = [55, 35, 20]
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=qualities[sev - 1])
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if corruption == "gaussian_noise":
        stds = [8.0, 16.0, 28.0]
        rng = np.random.default_rng(10_000 + sev)
        noise = rng.normal(0.0, stds[sev - 1], size=arr.shape)
        out = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    raise ValueError(f"Unknown corruption: {corruption}")


def extract_corrupted_batch(method, batch: Sequence[Dict], corruption: str, severity: int) -> torch.Tensor:
    image_groups: List[List[Image.Image]] = []
    for sample in batch:
        paths = select_frame_paths(sample, method.num_frames)
        images = [corrupt_image(Image.open(path).convert("RGB"), corruption, severity) for path in paths]
        image_groups.append(images)
    flat_images = [img for group in image_groups for img in group]
    inputs = method.processor(images=flat_images, return_tensors="pt", padding=True).to(method.device)
    with torch.no_grad():
        feats = method.model.get_image_features(pixel_values=inputs["pixel_values"]).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    rows = []
    offset = 0
    for group in image_groups:
        count = len(group)
        item = feats[offset:offset + count]
        offset += count
        if method.feature_layout == "sequence":
            if item.shape[0] < method.num_frames:
                pad = item[-1:].expand(method.num_frames - item.shape[0], -1)
                item = torch.cat([item, pad], dim=0)
            rows.append(item[: method.num_frames])
        else:
            pooled = item.mean(dim=0)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            rows.append(pooled)
    return torch.stack(rows, dim=0)


def predict_corruption(method, samples: Sequence[Dict], corruption: str, severity: int,
                       batch_size: int) -> Tuple[List[str], List[float]]:
    preds: List[str] = []
    confs: List[float] = []
    for start in range(0, len(samples), batch_size):
        if start == 0 or start % max(batch_size * 10, 1) == 0:
            print(
                f"[robustness]   batch {start // batch_size + 1}/{math.ceil(len(samples) / batch_size)} "
                f"({corruption}, severity={severity})",
                flush=True,
            )
        batch = samples[start:start + batch_size]
        x = extract_corrupted_batch(method, batch, corruption, severity)
        with torch.no_grad():
            logits = method.adapter.logits(x.to(method.device), method.text_features)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).detach().cpu().tolist()
            conf = probs.max(dim=-1).values.detach().cpu().tolist()
        preds.extend([AIDE_LABELS[int(i)] for i in pred_idx])
        confs.extend([float(v) for v in conf])
    return preds, confs


def generate_robustness(args, output_dir: Path, device: str) -> Dict:
    print("[robustness] loading AIDE method and test split", flush=True)
    _, _, samples, _ = build_test_split_from_result(
        result_json_path=Path(args.aide_result_json),
        aide_root=args.aide_root,
        annotation_root=args.aide_annotation_root,
        seed_override=args.seed,
    )
    method = load_method(
        checkpoint_path=Path(args.aide_ckpt),
        name="aide_ours",
        device=device,
        baseline_mode="adapter",
        seed=args.seed,
    )
    labels = [sample["label"] for sample in samples]
    corruptions = ["motion_blur", "low_light", "occlusion", "jpeg", "gaussian_noise"]
    summary_rows = []
    pred_rows = []

    runs = [("clean", 0)] + [(corr, sev) for corr in corruptions for sev in [1, 2, 3]]
    for corr, sev in runs:
        print(f"[robustness] evaluating {corr} severity={sev}", flush=True)
        preds, confs = predict_corruption(method, samples, corr, sev, args.robust_batch_size)
        acc = float(accuracy_score(labels, preds))
        wf1 = float(f1_score(labels, preds, labels=AIDE_LABELS, average="weighted", zero_division=0))
        summary_rows.append(
            {
                "method": "Ours",
                "corruption": corr,
                "severity": sev,
                "accuracy": round(acc, 6),
                "weighted_f1": round(wf1, 6),
                "num_samples": len(samples),
            }
        )
        for sample, label, pred, conf in zip(samples, labels, preds, confs):
            pred_rows.append(
                {
                    "sequence_id": sample["sequence_id"],
                    "label": label,
                    "prediction": pred,
                    "confidence": round(float(conf), 6),
                    "corruption": corr,
                    "severity": sev,
                    "frame_path": sample["frame_path"],
                }
            )
    write_csv(output_dir / "fig_robustness_source.csv", summary_rows)
    write_csv(output_dir / "fig_robustness_predictions.csv", pred_rows)

    clean_wf1 = next(row["weighted_f1"] for row in summary_rows if row["corruption"] == "clean")
    clean_acc = next(row["accuracy"] for row in summary_rows if row["corruption"] == "clean")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "figure.dpi": 180, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    palette = {
        "motion_blur": "#2f72bd",
        "low_light": "#15957f",
        "occlusion": "#b94f90",
        "jpeg": "#d8a000",
        "gaussian_noise": "#c76f00",
    }
    label_map = {
        "motion_blur": "Motion blur",
        "low_light": "Low light",
        "occlusion": "Occlusion",
        "jpeg": "JPEG",
        "gaussian_noise": "Gaussian noise",
    }
    for corr in corruptions:
        rows = [row for row in summary_rows if row["corruption"] == corr]
        rows.sort(key=lambda row: row["severity"])
        ax.plot(
            [row["severity"] for row in rows],
            [row["weighted_f1"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=palette[corr],
            label=label_map[corr],
        )
    ax.axhline(clean_wf1, color="#444444", linewidth=1.1, linestyle="--", label=f"Clean WF1={clean_wf1:.3f}")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.0, min(1.0, max(clean_wf1 + 0.08, 0.9)))
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower left", ncol=2, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.15)
    for suffix in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig_robustness.{suffix}", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(REPO_ROOT / f"fig_robustness.{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return {
        "aide_checkpoint": args.aide_ckpt,
        "aide_result_json": args.aide_result_json,
        "num_test_samples": len(samples),
        "clean_accuracy": clean_acc,
        "clean_weighted_f1": clean_wf1,
        "source_csv": str(output_dir / "fig_robustness_source.csv"),
        "prediction_csv": str(output_dir / "fig_robustness_predictions.csv"),
        "output_pdf": str(output_dir / "fig_robustness.pdf"),
        "method_note": "Ours-only corruption stress test; no CoCoOp or CLIP-Adapter robustness checkpoint was found in final AIDE runs.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "paper_figures_final"))
    parser.add_argument("--aide_ckpt", default=str(REPO_ROOT / "results" / "final_runs" / "aide" / "A0_ours_full_seed42_canonical" / "best.ckpt.pt"))
    parser.add_argument("--aide_result_json", default=str(REPO_ROOT / "results" / "final_runs" / "aide" / "A0_ours_full_seed42_canonical" / "result.json"))
    parser.add_argument("--aide_feature_cache", default=str(REPO_ROOT / "aide_clip" / "cache" / "features" / "strict_features_e3837d32059da7b0.pt"))
    parser.add_argument("--aide_root", default=None)
    parser.add_argument("--aide_annotation_root", default=None)
    parser.add_argument("--yawdd_ckpt", default=str(REPO_ROOT / "results" / "final_runs" / "yawdd" / "B_A0_seed42_canonical" / "best.ckpt.pt"))
    parser.add_argument("--yawdd_cache", default=str(REPO_ROOT / "results" / "final_runs" / "yawdd" / "B_sanity_seed42" / "cache" / "yawdd-binary_test_openai_clip-vit-base-patch32_f10_diff_guided_n62_eddc4f62578edf6f.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--robust_batch_size", type=int, default=12)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--attention_examples", type=int, default=5)
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_attention", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    summary = {"device": device, "seed": args.seed, "outputs": {}}

    if not args.skip_tsne:
        summary["outputs"]["tsne"] = generate_tsne(args, output_dir, device)
    if not args.skip_attention:
        summary["outputs"]["attention"] = generate_attention(args, output_dir, device)
    if not args.skip_robustness:
        summary["outputs"]["robustness"] = generate_robustness(args, output_dir, device)

    with (output_dir / "interpretability_robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
