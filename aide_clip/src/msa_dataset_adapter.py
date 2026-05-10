import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SUPPORTED_DATASETS = {"mosi", "mosei"}
SUPPORTED_LABEL_MODES = {"regression", "binary", "ternary", "seven"}


def load_mmsa_pickle(input_path: str) -> Dict:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload, got: {type(data)}")
    return data


def _to_scalar_label(value) -> float:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("Empty label value")
    return float(array.reshape(-1)[0])


def sentiment_to_class(value: float, label_mode: str) -> Tuple[Optional[str], Optional[int]]:
    if label_mode == "regression":
        return None, None
    if label_mode == "binary":
        if value == 0:
            return None, None
        return ("positive", 1) if value > 0 else ("negative", 0)
    if label_mode == "ternary":
        if value > 0:
            return "positive", 2
        if value < 0:
            return "negative", 0
        return "neutral", 1
    if label_mode == "seven":
        discrete = int(np.clip(np.round(value), -3, 3))
        return f"sent_{discrete:+d}", discrete + 3
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _safe_shape(value) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None


def normalize_processed_split(
    split_name: str,
    split_payload: Dict,
    dataset_name: str,
    label_mode: str,
) -> List[Dict]:
    raw_texts = list(split_payload.get("raw_text", []))
    sample_ids = list(split_payload.get("id", []))
    regression_labels = split_payload.get("regression_labels", [])
    annotations = list(split_payload.get("annotations", [])) if "annotations" in split_payload else []

    if sample_ids and len(sample_ids) != len(raw_texts):
        raise ValueError(f"Split {split_name}: len(id)={len(sample_ids)} != len(raw_text)={len(raw_texts)}")

    if len(regression_labels) != len(raw_texts):
        raise ValueError(
            f"Split {split_name}: len(regression_labels)={len(regression_labels)} != len(raw_text)={len(raw_texts)}"
        )

    text_features = split_payload.get("text")
    vision_features = split_payload.get("vision")
    audio_features = split_payload.get("audio")

    samples = []
    for idx in range(len(raw_texts)):
        reg_label = _to_scalar_label(regression_labels[idx])
        label_name, label_id = sentiment_to_class(reg_label, label_mode)
        samples.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "sample_index": idx,
                "sample_id": sample_ids[idx] if sample_ids else f"{split_name}_{idx:06d}",
                "raw_text": str(raw_texts[idx]),
                "annotation": str(annotations[idx]) if idx < len(annotations) else None,
                "regression_label": reg_label,
                "label_name": label_name,
                "label_id": label_id,
                "text_feature_shape": _safe_shape(text_features[idx]) if text_features is not None else None,
                "vision_feature_shape": _safe_shape(vision_features[idx]) if vision_features is not None else None,
                "audio_feature_shape": _safe_shape(audio_features[idx]) if audio_features is not None else None,
            }
        )
    return samples


def convert_mmsa_pickle_to_manifest(input_path: str, dataset_name: str, label_mode: str) -> Dict:
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if label_mode not in SUPPORTED_LABEL_MODES:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    data = load_mmsa_pickle(input_path)
    manifest = {
        "dataset": dataset_name,
        "source_format": "mmsa_processed_pickle",
        "input_path": str(Path(input_path).resolve()),
        "label_mode": label_mode,
        "splits": {},
    }

    for split_name in ["train", "valid", "test"]:
        if split_name not in data:
            continue
        manifest["splits"][split_name] = normalize_processed_split(
            split_name=split_name,
            split_payload=data[split_name],
            dataset_name=dataset_name,
            label_mode=label_mode,
        )
    return manifest