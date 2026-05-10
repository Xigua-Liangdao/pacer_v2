#!/usr/bin/env python3
# pyright: reportMissingImports=false
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_ravdess_emotion_train import (  # noqa: E402
    ClipImageAdapter,
    EMOTION_LABELS,
    TemporalTransformerClipImageAdapter,
    zeroshot_logits,
)


CLASS_COLOR_MAP = {
    "neutral": "#7a7a7a",
    "calm": "#1f9d8a",
    "happy": "#e3ab17",
    "sad": "#4f8fcf",
    "angry": "#e76f51",
    "fearful": "#9b59b6",
    "disgust": "#36c46b",
    "surprised": "#f09a3e",
}


@dataclass
class LoadedMethod:
    name: str
    mode: str
    checkpoint: Dict
    config: Dict
    checkpoint_path: Path
    prompt_groups: List[List[str]]
    text_features: torch.Tensor
    adapter: Optional[object]
    device: str
    feature_layout: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def normalize_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def pool_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        return normalize_feature_tensor(features.mean(dim=1))
    return normalize_feature_tensor(features)


def instantiate_adapter_from_checkpoint(checkpoint: Dict, config: Dict, device: str):
    text_features = checkpoint.get("text_features")
    if text_features is None:
        raise ValueError("Checkpoint missing text_features; cannot reconstruct adapter.")
    adapter_state = checkpoint.get("adapter_state_dict")
    if adapter_state is None:
        raise ValueError("Checkpoint missing adapter_state_dict; expected RAVDESS adapter checkpoint.")

    temporal_head = str(config.get("temporal_head", "none"))
    use_prompt_weight = bool(config.get("use_prompt_weight", True))
    use_class_temperature = bool(config.get("use_class_temperature", True))
    use_class_bias = bool(config.get("use_class_bias", True))

    if temporal_head == "transformer":
        adapter = TemporalTransformerClipImageAdapter(
            dim=int(text_features.shape[-1]),
            device=device,
            hidden_dim=int(config.get("adapter_hidden_dim", 256)),
            dropout=float(config.get("adapter_dropout", 0.2)),
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            num_frames=int(config.get("num_frames", 5)),
            temporal_num_heads=int(config.get("temporal_num_heads", 4)),
            temporal_num_layers=int(config.get("temporal_num_layers", 1)),
            temporal_pooling=str(config.get("temporal_pooling", "cls")),
            use_intensity_aux=bool(config.get("use_intensity_aux", False)),
            use_global_logit_scale=bool(config.get("use_global_logit_scale", False)),
            use_prompt_weight=use_prompt_weight,
            use_class_temperature=use_class_temperature,
            use_class_bias=use_class_bias,
        )
    else:
        adapter = ClipImageAdapter(
            dim=int(text_features.shape[-1]),
            device=device,
            hidden_dim=int(config.get("adapter_hidden_dim", 256)),
            dropout=float(config.get("adapter_dropout", 0.2)),
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            use_intensity_aux=bool(config.get("use_intensity_aux", False)),
            use_global_logit_scale=bool(config.get("use_global_logit_scale", False)),
            use_prompt_weight=use_prompt_weight,
            use_class_temperature=use_class_temperature,
            use_class_bias=use_class_bias,
        )

    adapter.load_state_dict(adapter_state)
    adapter.eval()
    return adapter


def load_method(
    checkpoint_path: Path,
    name: str,
    device: str,
    mode: str,
) -> LoadedMethod:
    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint.get("config", {}))
    prompt_groups = checkpoint.get("prompt_groups") or []
    text_features = checkpoint.get("text_features")
    if text_features is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not include text_features.")
    text_features = text_features.to(device)

    adapter = None
    normalized_mode = mode if mode in {"adapter", "zeroshot", "pure_clip"} else "adapter"
    if normalized_mode == "adapter":
        adapter = instantiate_adapter_from_checkpoint(checkpoint, config, device)

    return LoadedMethod(
        name=name,
        mode=normalized_mode,
        checkpoint=checkpoint,
        config=config,
        checkpoint_path=checkpoint_path,
        prompt_groups=prompt_groups,
        text_features=text_features,
        adapter=adapter,
        device=device,
        feature_layout=str(config.get("feature_layout", "sequence")),
    )


def load_cached_split_from_result(result_json_path: Path, split_name: str = "test") -> Tuple[List[Dict], torch.Tensor, Dict]:
    result = load_json(result_json_path)
    cache_paths = result.get("config", {}).get("resolved_feature_cache_paths", {})
    cache_path = cache_paths.get(split_name)
    if not cache_path:
        raise ValueError(f"Result JSON {result_json_path} does not include resolved_feature_cache_paths['{split_name}'].")
    payload = torch.load(resolve_path(cache_path), map_location="cpu")
    samples = list(payload.get("samples", []))
    features = payload.get("features")
    if features is None:
        raise ValueError(f"Feature cache {cache_path} does not include 'features'.")
    return samples, features.float(), result


def align_samples_and_features(
    reference_samples: List[Dict],
    reference_features: torch.Tensor,
    other_samples: List[Dict],
    other_features: torch.Tensor,
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    ref_map = {sample["sequence_id"]: (sample, reference_features[idx]) for idx, sample in enumerate(reference_samples)}
    other_map = {sample["sequence_id"]: other_features[idx] for idx, sample in enumerate(other_samples)}
    shared_ids = [sample["sequence_id"] for sample in reference_samples if sample["sequence_id"] in other_map]
    aligned_samples = [ref_map[seq_id][0] for seq_id in shared_ids]
    aligned_ref = torch.stack([ref_map[seq_id][1] for seq_id in shared_ids], dim=0)
    aligned_other = torch.stack([other_map[seq_id] for seq_id in shared_ids], dim=0)
    return aligned_samples, aligned_ref.float(), aligned_other.float()


def subset_samples_per_class(samples: List[Dict], max_per_class: int, seed: int) -> List[Dict]:
    if max_per_class <= 0:
        return list(samples)
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = {label: [] for label in EMOTION_LABELS}
    for sample in samples:
        grouped.setdefault(sample["label"], []).append(sample)
    selected = []
    for label in EMOTION_LABELS:
        items = list(grouped.get(label, []))
        rng.shuffle(items)
        selected.extend(items[:max_per_class])
    selected.sort(key=lambda sample: sample["sequence_id"])
    return selected


def filter_features_by_sequence_ids(samples: List[Dict], features: torch.Tensor, keep_ids: Sequence[str]) -> Tuple[List[Dict], torch.Tensor]:
    keep_set = set(keep_ids)
    chosen_samples = []
    chosen_features = []
    for idx, sample in enumerate(samples):
        if sample["sequence_id"] in keep_set:
            chosen_samples.append(sample)
            chosen_features.append(features[idx])
    if not chosen_features:
        raise ValueError("No overlapping samples remained after filtering by sequence ids.")
    return chosen_samples, torch.stack(chosen_features, dim=0).float()


def aggregate_text_feature_for_class(method: LoadedMethod, class_idx: int) -> torch.Tensor:
    text_x = method.text_features[class_idx]
    if method.adapter is not None and getattr(method.adapter, "use_prompt_weight", False):
        weights = torch.softmax(method.adapter.prompt_weight_logits[class_idx], dim=-1)
        agg = (text_x * weights.unsqueeze(-1)).sum(dim=0)
    else:
        agg = text_x.mean(dim=0)
    return normalize_feature_tensor(agg)


def build_method_features_and_predictions(
    method: LoadedMethod,
    features: torch.Tensor,
    labels: Sequence[str],
    batch_size: int,
) -> Dict[str, object]:
    feature_rows = []
    predictions: List[str] = []
    confidences: List[float] = []

    if method.mode == "adapter":
        for start in range(0, int(features.shape[0]), batch_size):
            batch_x = features[start:start + batch_size].to(method.device)
            with torch.no_grad():
                logits = method.adapter.logits(batch_x, method.text_features)
                probs = torch.softmax(logits, dim=-1)
                adapted = method.adapter._adapt_image(batch_x)
            pred_idx = probs.argmax(dim=-1).detach().cpu().tolist()
            pred_conf = probs.max(dim=-1).values.detach().cpu().tolist()
            feature_rows.extend(adapted.detach().cpu().numpy())
            predictions.extend([EMOTION_LABELS[int(idx)] for idx in pred_idx])
            confidences.extend([float(item) for item in pred_conf])
    else:
        class_vectors = torch.stack(
            [aggregate_text_feature_for_class(method, idx).float() for idx in range(len(EMOTION_LABELS))],
            dim=0,
        ).to(method.device)
        pooled_features = pool_feature_tensor(features)
        for start in range(0, int(pooled_features.shape[0]), batch_size):
            batch_x = pooled_features[start:start + batch_size].to(method.device)
            with torch.no_grad():
                logits = zeroshot_logits(batch_x, method.text_features) if method.mode in {"zeroshot", "pure_clip"} else batch_x @ class_vectors.t()
                probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).detach().cpu().tolist()
            pred_conf = probs.max(dim=-1).values.detach().cpu().tolist()
            feature_rows.extend(batch_x.detach().cpu().numpy())
            predictions.extend([EMOTION_LABELS[int(idx)] for idx in pred_idx])
            confidences.extend([float(item) for item in pred_conf])

    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "labels": list(labels),
        "predictions": predictions,
        "confidences": confidences,
    }


def save_tsne_coordinates(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        return
    if suffix == ".csv":
        import csv

        fieldnames = list(rows[0].keys()) if rows else ["sequence_id", "label", "x", "y"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError(f"Unsupported coordinate output format: {path}")


def draw_tsne_plot(
    coords: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_png: Path,
    output_pdf: Optional[Path] = None,
    plot_style: str = "default",
    marker_scale: float = 1.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    if plot_style == "block":
        figure_size = (7.2, 5.45)
        marker_size = 82
        alpha = 0.95
        edgecolors = "none"
        linewidths = 0.0
    else:
        figure_size = (7.45, 5.6)
        marker_size = 64
        alpha = 0.92
        edgecolors = "white"
        linewidths = 0.28
    marker_size = max(12.0, marker_size * float(marker_scale))
    fig, ax = plt.subplots(figsize=figure_size)
    for label in EMOTION_LABELS:
        idx = [i for i, item in enumerate(labels) if item == label]
        if not idx:
            continue
        subset = coords[idx]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=marker_size,
            alpha=alpha,
            c=CLASS_COLOR_MAP[label],
            label=label,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    ax.set_title(title, pad=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.55,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.86)
    ensure_parent(output_png)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.03)
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_tsne_comparison(
    baseline_coords: np.ndarray,
    method_coords: np.ndarray,
    labels: Sequence[str],
    baseline_title: str,
    method_title: str,
    output_png: Path,
    output_pdf: Optional[Path] = None,
    plot_style: str = "default",
    marker_scale: float = 1.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    if plot_style == "block":
        figure_size = (12.8, 5.2)
        marker_size = 64
        alpha = 0.94
        edgecolors = "none"
        linewidths = 0.0
    else:
        figure_size = (13.1, 5.45)
        marker_size = 50
        alpha = 0.9
        edgecolors = "white"
        linewidths = 0.24
    marker_size = max(10.0, marker_size * float(marker_scale))
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=False, sharey=False)
    for ax, coords, title in zip(axes, [baseline_coords, method_coords], [baseline_title, method_title]):
        for label in EMOTION_LABELS:
            idx = [i for i, item in enumerate(labels) if item == label]
            if not idx:
                continue
            subset = coords[idx]
            ax.scatter(
                subset[:, 0],
                subset[:, 1],
                s=marker_size,
                alpha=alpha,
                c=CLASS_COLOR_MAP[label],
                label=label,
                edgecolors=edgecolors,
                linewidths=linewidths,
            )
        ax.set_title(title, pad=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.18, linewidth=0.5)
    handles, label_text = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        label_text,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.992),
        columnspacing=1.2,
        handletextpad=0.55,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.87, wspace=0.12)
    ensure_parent(output_png)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.03)
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
