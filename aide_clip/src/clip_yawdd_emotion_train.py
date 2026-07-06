import argparse
import atexit
import json
import numpy as np
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_SRC_DIR not in sys.path:
    sys.path.insert(0, CURRENT_SRC_DIR)

import clip_aide_emotion_train as aide_base
import clip_cremad_emotion_train as cremad_base
import clip_ravdess_emotion_train as ravdess_base
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

YAWDD_BEHAVIOR_LABELS = ["Normal", "Talking", "Yawning", "TalkingYawning"]
YAWDD_BINARY_LABELS = ["notdrowsy", "drowsy"]
YAWDD_MULTI4_CLASS_NAMES = list(YAWDD_BEHAVIOR_LABELS)
YAWDD_BEHAVIOR_DISPLAY_MAP = {
    "Normal": "driving normally",
    "Talking": "talking or singing",
    "Yawning": "yawning",
    "TalkingYawning": "yawning while talking",
}
YAWDD_BINARY_DISPLAY_MAP = {
    "notdrowsy": "non-yawning",
    "drowsy": "yawning cue",
}


def resolve_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            text=True,
        ).strip()
    except Exception:
        return "unknown"
YAWDD_BEHAVIOR_PROMPT_GROUPS = {
    "Normal": [
        "a close-up photo of a driver with a closed mouth looking forward",
        "a driver's face with lips together and a relaxed jaw",
        "a driver with mouth closed and a neutral facial expression",
        "a photo of a driver's face, the mouth is not open",
        "a driver looking ahead with a resting closed mouth",
    ],
    "Talking": [
        "a close-up photo of a driver with mouth partially open shaped for speech",
        "a driver's face mid-speech with lips forming a word",
        "a driver with small rapid mouth movements of talking",
        "a photo of a driver pronouncing a word, mouth narrowly open",
        "a driver with teeth and lips visible in a speaking shape",
    ],
    "Yawning": [
        "a close-up photo of a driver with the mouth wide open in a yawn",
        "a driver with the jaw stretched fully open in a wide yawn",
        "a driver yawning with the mouth at maximum opening",
        "a photo of a driver with a large gaping open mouth and the tongue visible",
        "a driver's face during a deep yawn with stretched lips",
    ],
    "TalkingYawning": [
        "a close-up of a driver with a wide open yawning mouth while also speaking",
        "a driver whose wide open yawn overlaps with speech mouth movements",
        "a driver's face showing a yawn in progress combined with talking",
        "a photo of a driver mid-yawn with a speaking lip shape mixed in",
        "a driver with an open yawning mouth and active lip motion",
    ],
}
YAWDD_BINARY_PROMPT_GROUPS = {
    "notdrowsy": [
        "The driver's mouth remains closed without a visible yawn.",
        "The visible facial behavior shows a non-yawning driver.",
        "This clip shows ordinary mouth posture rather than a yawning cue.",
        "The person appears non-yawning with no wide mouth-opening cue.",
        "The facial cues indicate the non-yawning class.",
    ],
    "drowsy": [
        "The driver's mouth is wide open in a visible yawning cue.",
        "The visible facial behavior shows a yawning cue.",
        "This clip shows a yawning cue rather than ordinary mouth posture.",
        "The person shows a wide mouth-opening cue associated with yawning.",
        "The facial cues indicate the yawning-cue class.",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAWDD_ROOT = os.environ.get("YAWDD_ROOT", str(PROJECT_ROOT / "data" / "yawdd"))
DEFAULT_FEATURE_CACHE_DIR = str(PROJECT_ROOT / "cache" / "yawdd_features")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "yawdd" / "clip_yawdd_results.json")
LOG_FILE_HANDLE = None
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
GENERIC_FACE_SPLIT_DIRS = {"train_face_image", "test_face_image"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
YAWDD_BEHAVIOR_MATCH_ORDER = [
    ("talkingyawning", "TalkingYawning"),
    ("yawningtalking", "TalkingYawning"),
    ("talkingandyawning", "TalkingYawning"),
    ("yawningwhiletalking", "TalkingYawning"),
    ("talking", "Talking"),
    ("yawning", "Yawning"),
    ("normal", "Normal"),
]


def compute_output_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": round(float(values[0]), 6), "std": 0.0}
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def summarize_fold_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    average_mode = "binary" if len(labels) == 2 else "weighted"
    kwargs = {"labels": labels, "zero_division": 0}
    if average_mode == "binary":
        kwargs["average"] = "binary"
        kwargs["pos_label"] = labels[-1]
    else:
        kwargs["average"] = "weighted"
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "average_mode": average_mode,
    }


def add_suffix_to_path(path_value: str, suffix: str) -> str:
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def ensure_suffix_in_path(path_value: Optional[str], suffix: str) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.stem.endswith(suffix):
        return path_value
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def resolve_eval_output_paths(args, eval_mode: str, label_mode: str) -> Tuple[str, Optional[str], Optional[str]]:
    if eval_mode == "single":
        eval_suffix = ""
    elif eval_mode == "fixed":
        eval_suffix = "_fixed"
    elif eval_mode == "loso":
        eval_suffix = "_loso"
    elif eval_mode == "group_kfold":
        eval_suffix = f"_gkf{args.n_folds}"
    elif eval_mode == "sequence_kfold":
        eval_suffix = f"_skf{args.n_folds}"
    else:
        eval_suffix = "_random"
    baseline_suffix = "" if getattr(args, "baseline_mode", "none") == "none" else f"_{args.baseline_mode}"
    if label_mode != "multi4":
        if eval_mode in {"fixed", "single"}:
            output_path = ensure_suffix_in_path(args.output, baseline_suffix) if baseline_suffix else args.output
            log_path = ensure_suffix_in_path(args.log_file, baseline_suffix) if baseline_suffix else args.log_file
            checkpoint_path = ensure_suffix_in_path(args.checkpoint_output, baseline_suffix) if baseline_suffix else args.checkpoint_output
            return output_path, log_path, checkpoint_path
        output_path = ensure_suffix_in_path(args.output, f"{eval_suffix}{baseline_suffix}")
        log_path = ensure_suffix_in_path(args.log_file, f"{eval_suffix}{baseline_suffix}")
        checkpoint_path = ensure_suffix_in_path(args.checkpoint_output, f"{eval_suffix}{baseline_suffix}")
        return output_path, log_path, checkpoint_path

    full_suffix = f"_multi4{eval_suffix}{baseline_suffix}"
    output_path = ensure_suffix_in_path(args.output, full_suffix)
    log_path = ensure_suffix_in_path(args.log_file, full_suffix)
    checkpoint_path = ensure_suffix_in_path(args.checkpoint_output, full_suffix)
    return output_path, log_path, checkpoint_path


def log(message: str) -> None:
    global LOG_FILE_HANDLE

    line = f"[{cremad_base.time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(line + "\n")
        LOG_FILE_HANDLE.flush()


def init_log_file(log_file: Optional[str]) -> Optional[str]:
    global LOG_FILE_HANDLE

    if not log_file:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_HANDLE = path.open("a", encoding="utf-8")
    return str(path)


def close_log_file() -> None:
    global LOG_FILE_HANDLE

    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label) / len(y_true)


def weighted_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    weighted = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        weighted += (support.get(label, 0) / total) * f1
    return weighted


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def summarize_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, labels), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "prediction_distribution": {label: int(Counter(y_pred).get(label, 0)) for label in labels},
    }


def summarize_multi4_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    label2idx = {label: idx for idx, label in enumerate(labels)}
    class_names = list(labels)
    y_true_idx = [label2idx[label] for label in y_true]
    y_pred_idx = [label2idx[label] for label in y_pred]
    matrix = sklearn_confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))
    report_text = classification_report(
        y_true_idx,
        y_pred_idx,
        labels=list(range(len(labels))),
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true_idx,
        y_pred_idx,
        labels=list(range(len(labels))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = {
        class_name: {
            "precision": round(float(report_dict[class_name]["precision"]), 6),
            "recall": round(float(report_dict[class_name]["recall"]), 6),
            "f1": round(float(report_dict[class_name]["f1-score"]), 6),
            "support": int(report_dict[class_name]["support"]),
        }
        for class_name in class_names
    }
    weighted_metrics = {
        "precision": round(float(report_dict["weighted avg"]["precision"]), 6),
        "recall": round(float(report_dict["weighted avg"]["recall"]), 6),
        "f1": round(float(report_dict["weighted avg"]["f1-score"]), 6),
        "support": int(report_dict["weighted avg"]["support"]),
    }
    prediction_distribution = {
        class_names[index]: int(sum(1 for pred_idx in y_pred_idx if pred_idx == index))
        for index in range(len(class_names))
    }
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "precision": weighted_metrics["precision"],
        "recall": weighted_metrics["recall"],
        "f1": weighted_metrics["f1"],
        "average_mode": "weighted",
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": class_names,
        "per_class_metrics": per_class_metrics,
        "weighted_average_metrics": weighted_metrics,
        "prediction_distribution": prediction_distribution,
        "classification_report": report_text,
    }


def summarize_predictions_by_mode(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    label_mode: str,
) -> Dict[str, object]:
    if label_mode == "multi4":
        return summarize_multi4_predictions(y_true, y_pred, labels)
    return summarize_predictions(y_true, y_pred, labels)


def resolve_class_names(label_mode: str, emotion_labels: List[str]) -> List[str]:
    return list(emotion_labels)


def compute_split_class_distribution(
    emotion_labels: List[str],
    class_names: List[str],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
) -> Dict[str, Dict[str, int]]:
    label_to_name = {label: class_name for label, class_name in zip(emotion_labels, class_names)}

    def summarize(samples: List[Dict]) -> Dict[str, int]:
        counter = Counter(sample["label"] for sample in samples)
        return {label_to_name[label]: int(counter.get(label, 0)) for label in emotion_labels}

    return {
        "train": summarize(train_samples),
        "val": summarize(val_samples),
        "test": summarize(test_samples),
    }


def compute_multi4_class_weights(train_samples: List[Dict], emotion_labels: List[str]) -> List[float]:
    counts = Counter(sample["label"] for sample in train_samples)
    total_samples = sum(int(counts.get(label, 0)) for label in emotion_labels)
    class_count = len(emotion_labels)
    weights: List[float] = []
    for label in emotion_labels:
        count = int(counts.get(label, 0))
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(float(total_samples) / float(class_count * count))
    return weights


def split_samples_stratified_ratio(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    label_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        label_groups.setdefault(sample["label"], []).append(sample)

    rng = random.Random(seed)
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for label in sorted(label_groups):
        label_samples = sorted(
            label_groups[label],
            key=lambda sample: (
                str(sample.get("actor_id", "")),
                str(sample.get("sequence_id", "")),
                str(sample.get("video_path", "")),
                str(sample.get("file_name", "")),
            ),
        )
        rng.shuffle(label_samples)

        total = len(label_samples)
        if total <= 1:
            train_count = total
            val_count = 0
            test_count = 0
        else:
            val_count = int(round(total * val_ratio))
            test_count = int(round(total * test_ratio))

            if val_ratio > 0 and val_count == 0 and total >= 2:
                val_count = 1
            if test_ratio > 0 and test_count == 0 and total - val_count >= 2:
                test_count = 1

            while val_count + test_count >= total:
                if test_count > 0 and (test_count >= val_count or val_count == 0):
                    test_count -= 1
                elif val_count > 0:
                    val_count -= 1
                else:
                    break

            train_count = total - val_count - test_count
            if train_ratio > 0 and train_count <= 0:
                if val_count >= test_count and val_count > 0:
                    val_count -= 1
                elif test_count > 0:
                    test_count -= 1
                train_count = total - val_count - test_count

        train_samples.extend(label_samples[:train_count])
        val_samples.extend(label_samples[train_count:train_count + val_count])
        test_samples.extend(label_samples[train_count + val_count:train_count + val_count + test_count])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    split_info = {
        "mode": "split_label_stratified",
        "targets": {
            "train": round(len(samples) * train_ratio, 3),
            "val": round(len(samples) * val_ratio, 3),
            "test": round(len(samples) * test_ratio, 3),
        },
        "train": cremad_base.compute_split_summary(train_samples),
        "val": cremad_base.compute_split_summary(val_samples),
        "test": cremad_base.compute_split_summary(test_samples),
    }
    return {"train": train_samples, "val": val_samples, "test": test_samples}, split_info


def augment_cached_training_features(
    train_x,
    train_y,
    train_samples: List[Dict],
    labels_to_augment: List[str],
    min_count: int,
    noise_std: float,
    seed: int,
    augmentation_mode: str,
    mixup_alpha: float,
):
    import torch

    if min_count <= 0 or not labels_to_augment:
        return train_x, train_y, list(train_samples), None

    counts = Counter(sample["label"] for sample in train_samples)
    labels_to_apply = [label for label in labels_to_augment if 0 < int(counts.get(label, 0)) < min_count]
    if not labels_to_apply:
        return train_x, train_y, list(train_samples), None

    generator = torch.Generator()
    generator.manual_seed(seed)
    rng = random.Random(seed)
    base_scale = float(train_x.float().std(unbiased=False).item()) if train_x.numel() > 0 else 1.0
    if base_scale <= 0:
        base_scale = 1.0

    augmented_x_parts = []
    augmented_y_parts = []
    augmented_samples: List[Dict] = []
    diagnostics = {
        "mode": "feature_noise_oversample" if augmentation_mode == "noise" else "feature_mixup_oversample",
        "labels": {},
        "min_count": int(min_count),
        "base_feature_std": round(base_scale, 6),
        "original_train_size": len(train_samples),
    }
    if augmentation_mode == "noise":
        diagnostics["noise_std"] = float(noise_std)
    else:
        diagnostics["mixup_alpha"] = float(mixup_alpha)

    for label in labels_to_apply:
        source_indices = [index for index, sample in enumerate(train_samples) if sample["label"] == label]
        needed = int(min_count) - len(source_indices)
        if needed <= 0:
            continue

        for repeat_index in range(needed):
            source_index = source_indices[repeat_index % len(source_indices)]
            feature = train_x[source_index:source_index + 1].clone()
            augmented_sample = dict(train_samples[source_index])
            augmented_sample["augmentation_source_index"] = int(source_index)
            if augmentation_mode == "mixup":
                partner_index = source_indices[rng.randrange(len(source_indices))]
                mix_ratio = rng.betavariate(float(mixup_alpha), float(mixup_alpha))
                partner_feature = train_x[partner_index:partner_index + 1]
                feature = feature * float(mix_ratio) + partner_feature * float(1.0 - mix_ratio)
                augmented_sample["augmentation"] = "feature_mixup_oversample"
                augmented_sample["augmentation_partner_index"] = int(partner_index)
                augmented_sample["augmentation_mix_ratio"] = round(float(mix_ratio), 6)
            else:
                if noise_std > 0:
                    noise = torch.randn(feature.shape, generator=generator, dtype=feature.dtype, device=feature.device)
                    feature = feature + noise * float(noise_std) * base_scale
                augmented_sample["augmentation"] = "feature_noise_oversample"
            augmented_x_parts.append(feature)
            augmented_y_parts.append(train_y[source_index:source_index + 1].clone())
            augmented_samples.append(augmented_sample)

        diagnostics["labels"][label] = {
            "original_count": len(source_indices),
            "target_count": int(min_count),
            "added_count": needed,
        }

    if not augmented_x_parts:
        return train_x, train_y, list(train_samples), None

    combined_x = torch.cat([train_x] + augmented_x_parts, dim=0)
    combined_y = torch.cat([train_y] + augmented_y_parts, dim=0)
    combined_samples = list(train_samples) + augmented_samples
    diagnostics["augmented_train_size"] = len(combined_samples)
    diagnostics["augmented_class_distribution"] = dict(Counter(sample["label"] for sample in combined_samples))
    return combined_x, combined_y, combined_samples, diagnostics


def normalize_label_source(raw_source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw_source).strip().lower())


def infer_multi4_behavior_label(raw_source: str) -> str:
    normalized = normalize_label_source(raw_source)
    for token, label in YAWDD_BEHAVIOR_MATCH_ORDER:
        if token in normalized:
            return label
    if "talking" in normalized and "yawning" in normalized:
        return "TalkingYawning"
    if "talking" in normalized:
        return "Talking"
    if "yawning" in normalized:
        return "Yawning"
    return "Normal"


def extract_multi4_label_source(image_path: Path, root: Path, sequence_stem: str) -> str:
    relative_parent = image_path.parent.relative_to(root)
    parent_tokens = [part for part in relative_parent.parts if part not in GENERIC_FACE_SPLIT_DIRS]
    if parent_tokens:
        return "/".join(parent_tokens)
    return sequence_stem


def extract_actor_id_from_name(*candidates: str) -> str:
    for candidate in candidates:
        match = re.match(r"^(\d+)", str(candidate).strip())
        if match:
            return match.group(1)
    return "unknown"


def collect_frame_sequence_dirs(root: Path) -> List[Path]:
    frame_dirs: List[Path] = []
    for current_root, _, file_names in os.walk(root):
        current_path = Path(current_root)
        if any(Path(file_name).suffix.lower() in IMAGE_EXTENSIONS for file_name in file_names):
            frame_dirs.append(current_path)
    return sorted(frame_dirs)


def sorted_frame_paths_from_dir(frame_dir: Path) -> List[Path]:
    frame_paths = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frame_paths)


def build_face_sequence_sample_from_dir(
    frame_dir: Path,
    data_root: Path,
    label_mode: str,
    source_split: str,
) -> Optional[Dict]:
    frame_paths = sorted_frame_paths_from_dir(frame_dir)
    if not frame_paths:
        return None
    relative_dir = frame_dir.relative_to(data_root)
    label_source = "/".join(part for part in relative_dir.parts if part not in GENERIC_FACE_SPLIT_DIRS)
    behavior_label = infer_multi4_behavior_label(label_source or frame_dir.name)
    binary_label = map_behavior_to_binary(behavior_label)
    label = binary_label if label_mode == "binary" else behavior_label
    actor_id = extract_actor_id_from_name(frame_dir.name, relative_dir.parts[0] if relative_dir.parts else "")
    sequence_name = relative_dir.as_posix()
    return {
        "sequence_id": f"{source_split}::{sequence_name}",
        "video_path": str(frame_paths[0]),
        "frame_dir": str(frame_dir),
        "frame_paths": [str(path) for path in frame_paths],
        "frame_path": str(frame_paths[len(frame_paths) // 2]),
        "file_name": sequence_name,
        "actor_id": actor_id,
        "subject_id": actor_id,
        "appearance": sequence_name,
        "label": label,
        "behavior_label": behavior_label,
        "binary_label": binary_label,
        "view": "face_crop",
        "group": source_split,
        "ext": ".jpg",
        "source_split": source_split,
    }


def build_face_sequence_samples_from_dirs(
    frame_dirs: List[Path],
    data_root: Path,
    label_mode: str,
    source_split: str,
) -> List[Dict]:
    samples: List[Dict] = []
    for frame_dir in frame_dirs:
        sample = build_face_sequence_sample_from_dir(frame_dir, data_root=data_root, label_mode=label_mode, source_split=source_split)
        if sample is not None:
            samples.append(sample)
    return samples


def build_face_sequence_samples_from_paths(
    image_paths: List[Path],
    data_root: Path,
    label_mode: str,
    source_split: str,
) -> List[Dict]:
    grouped: Dict[str, List[Path]] = {}
    for image_path in sorted(image_paths):
        stem = image_path.stem.rsplit("-", 1)[0]
        grouped.setdefault(stem, []).append(image_path)

    samples: List[Dict] = []
    for stem, frame_paths in sorted(grouped.items()):
        frame_paths = sorted(frame_paths)
        first_frame = frame_paths[0]
        relative_parent = first_frame.parent.relative_to(data_root)
        label_source = extract_multi4_label_source(first_frame, data_root, stem)
        behavior_label = infer_multi4_behavior_label(label_source)
        binary_label = map_behavior_to_binary(behavior_label)
        label = binary_label if label_mode == "binary" else behavior_label
        actor_id = extract_actor_id_from_name(stem, first_frame.name)
        sequence_name = stem if str(relative_parent) == "." else f"{relative_parent.as_posix()}/{stem}"
        samples.append(
            {
                "sequence_id": f"{source_split}::{sequence_name}",
                "video_path": str(first_frame),
                "frame_dir": str(first_frame.parent),
                "frame_paths": [str(path) for path in frame_paths],
                "frame_path": str(frame_paths[len(frame_paths) // 2]),
                "file_name": sequence_name,
                "actor_id": actor_id,
                "subject_id": actor_id,
                "appearance": sequence_name,
                "label": label,
                "behavior_label": behavior_label,
                "binary_label": binary_label,
                "view": "face_crop",
                "group": source_split,
                "ext": first_frame.suffix.lower(),
                "source_split": source_split,
            }
        )
    return samples


def normalize_behavior_label(raw_label: str) -> Optional[str]:
    normalized = normalize_label_source(raw_label)
    if not normalized:
        return None
    return infer_multi4_behavior_label(normalized)


def map_behavior_to_binary(behavior_label: str) -> str:
    normalized = normalize_behavior_label(behavior_label)
    return "drowsy" if normalized in {"Yawning", "TalkingYawning"} else "notdrowsy"


def parse_yawdd_filename(file_name: str) -> Optional[Dict[str, str]]:
    path = Path(file_name)
    ext = path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return None
    stem = path.stem
    parts = [part.strip() for part in stem.split("-") if part.strip()]
    if len(parts) < 3:
        return None
    subject_id = parts[0]
    raw_label = parts[-1]
    behavior_label = normalize_behavior_label(raw_label)
    if behavior_label is None:
        return None
    appearance = "-".join(parts[1:-1])
    return {
        "sequence_id": stem,
        "subject_id": subject_id,
        "actor_id": subject_id,
        "appearance": appearance,
        "behavior_label": behavior_label,
        "binary_label": map_behavior_to_binary(behavior_label),
        "ext": ext,
    }


def collect_yawdd_samples(
    yawdd_root: str,
    label_mode: str,
    include_dash: bool = False,
    max_sequences: int = 0,
) -> Tuple[List[Dict], Dict[str, object]]:
    root = Path(yawdd_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"YawDD root not found: {root}")

    samples: List[Dict] = []
    invalid_name_examples: List[str] = []
    source_dirs: List[str] = []
    search_roots = [root / "Mirror"]
    if include_dash:
        search_roots.append(root / "Dash")

    for search_root in search_roots:
        if not search_root.exists():
            continue
        source_dirs.append(str(search_root))
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                continue
            meta = parse_yawdd_filename(path.name)
            if meta is None:
                if len(invalid_name_examples) < 20:
                    invalid_name_examples.append(path.name)
                continue
            relative_parent = path.parent.relative_to(root)
            label = meta["binary_label"] if label_mode == "binary" else meta["behavior_label"]
            samples.append(
                {
                    "sequence_id": f"{relative_parent.as_posix()}::{meta['sequence_id']}",
                    "video_path": str(path),
                    "file_name": path.name,
                    "actor_id": meta["actor_id"],
                    "subject_id": meta["subject_id"],
                    "appearance": meta["appearance"],
                    "label": label,
                    "behavior_label": meta["behavior_label"],
                    "binary_label": meta["binary_label"],
                    "view": relative_parent.parts[0] if relative_parent.parts else "unknown",
                    "group": relative_parent.parts[1] if len(relative_parent.parts) > 1 else "unknown",
                    "ext": meta["ext"],
                }
            )

    if max_sequences > 0:
        samples = samples[:max_sequences]

    diagnostics = {
        "total_valid_samples": len(samples),
        "subject_count": len({sample["subject_id"] for sample in samples}),
        "class_distribution": dict(Counter(sample["label"] for sample in samples)),
        "behavior_distribution": dict(Counter(sample["behavior_label"] for sample in samples)),
        "view_distribution": dict(Counter(sample["view"] for sample in samples)),
        "group_distribution": dict(Counter(sample["group"] for sample in samples)),
        "extension_distribution": dict(Counter(sample["ext"] for sample in samples)),
        "source_dirs": source_dirs,
        "invalid_name_examples": invalid_name_examples,
    }
    return samples, diagnostics


def collect_preprocessed_face_sequence_samples(
    preprocessed_root: str,
    label_mode: str,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    if label_mode != "binary":
        raise ValueError("External face-sequence preprocessing currently supports only binary labels")

    root = Path(preprocessed_root).resolve()
    train_root = root / "train_face_image"
    test_root = root / "test_face_image"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(f"Expected train_face_image and test_face_image under: {root}")

    def build_samples_from_paths(image_paths: List[Path], frame_dir: Path, source_split: str) -> List[Dict]:
        grouped: Dict[str, List[Path]] = {}
        label_by_stem: Dict[str, str] = {}
        for image_path in sorted(image_paths):
            stem = image_path.stem.rsplit("-", 1)[0]
            label_token = image_path.stem.rsplit("_", 1)[-1]
            grouped.setdefault(stem, []).append(image_path)
            existing_label = label_by_stem.get(stem)
            if existing_label is not None and existing_label != label_token:
                raise RuntimeError(f"Inconsistent labels within sequence {stem}: {existing_label} vs {label_token}")
            label_by_stem[stem] = label_token

        samples: List[Dict] = []
        for stem, frame_paths in sorted(grouped.items()):
            subject_id = stem.split("-", 1)[0]
            binary_label = "drowsy" if label_by_stem[stem] == "1" else "notdrowsy"
            behavior_label = "yawning cue" if binary_label == "drowsy" else "non-yawning"
            samples.append(
                {
                    "sequence_id": f"{source_split}::{stem}",
                    "video_path": str(frame_paths[0]),
                    "frame_dir": str(frame_dir),
                    "frame_paths": [str(path) for path in sorted(frame_paths)],
                    "file_name": stem,
                    "actor_id": subject_id,
                    "subject_id": subject_id,
                    "appearance": stem,
                    "label": binary_label,
                    "behavior_label": behavior_label,
                    "binary_label": binary_label,
                    "view": "face_crop",
                    "group": source_split,
                    "ext": ".jpg",
                    "source_split": source_split,
                }
            )
        return samples

    train_pool_samples = build_samples_from_paths(list(train_root.glob("*.jpg")), train_root, source_split="train_pool")
    fixed_test_samples = build_samples_from_paths(list(test_root.glob("*.jpg")), test_root, source_split="fixed_test")
    diagnostics = {
        "mode": "external_face_sequences",
        "preprocessed_root": str(root),
        "train_pool_sequences": len(train_pool_samples),
        "fixed_test_sequences": len(fixed_test_samples),
        "train_pool_subjects": len({sample["subject_id"] for sample in train_pool_samples}),
        "fixed_test_subjects": len({sample["subject_id"] for sample in fixed_test_samples}),
        "train_pool_distribution": dict(Counter(sample["label"] for sample in train_pool_samples)),
        "fixed_test_distribution": dict(Counter(sample["label"] for sample in fixed_test_samples)),
    }
    return train_pool_samples, fixed_test_samples, diagnostics


def collect_fixed_all_face_sequence_samples(
    all_face_image: str,
    label_mode: str,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    root = Path(all_face_image).resolve()
    train_root = root / "train_face_image"
    test_root = root / "test_face_image"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(f"Expected train_face_image and test_face_image under: {root}")

    train_frame_dirs = [path for path in collect_frame_sequence_dirs(train_root) if path != train_root]
    test_frame_dirs = [path for path in collect_frame_sequence_dirs(test_root) if path != test_root]
    train_flat_frames = sorted(path for path in train_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    test_flat_frames = sorted(path for path in test_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    if train_frame_dirs:
        train_pool_samples = build_face_sequence_samples_from_dirs(
            train_frame_dirs,
            data_root=train_root,
            label_mode=label_mode,
            source_split="train_pool",
        )
    else:
        train_pool_samples = build_face_sequence_samples_from_paths(
            train_flat_frames,
            data_root=train_root,
            label_mode=label_mode,
            source_split="train_pool",
        )

    if test_frame_dirs:
        fixed_test_samples = build_face_sequence_samples_from_dirs(
            test_frame_dirs,
            data_root=test_root,
            label_mode=label_mode,
            source_split="fixed_test",
        )
    else:
        fixed_test_samples = build_face_sequence_samples_from_paths(
            test_flat_frames,
            data_root=test_root,
            label_mode=label_mode,
            source_split="fixed_test",
        )
    diagnostics = {
        "mode": "all_face_fixed_sequences",
        "all_face_image": str(root),
        "layout": "directory_sequences" if train_frame_dirs or test_frame_dirs else "flat_grouped_frames",
        "train_pool_sequences": len(train_pool_samples),
        "fixed_test_sequences": len(fixed_test_samples),
        "train_pool_subjects": len({sample["subject_id"] for sample in train_pool_samples}),
        "fixed_test_subjects": len({sample["subject_id"] for sample in fixed_test_samples}),
        "train_pool_distribution": dict(Counter(sample["label"] for sample in train_pool_samples)),
        "fixed_test_distribution": dict(Counter(sample["label"] for sample in fixed_test_samples)),
    }
    return train_pool_samples, fixed_test_samples, diagnostics


def collect_all_face_sequence_samples(
    all_face_image: str,
    label_mode: str,
) -> Tuple[List[Dict], Dict[str, object]]:
    root = Path(all_face_image).resolve()
    if not root.exists():
        raise FileNotFoundError(f"All face image path not found: {root}")

    train_root = root / "train_face_image"
    test_root = root / "test_face_image"
    if train_root.exists() and test_root.exists():
        train_frame_dirs = [path for path in collect_frame_sequence_dirs(train_root) if path != train_root]
        test_frame_dirs = [path for path in collect_frame_sequence_dirs(test_root) if path != test_root]
        train_flat_frames = sorted(path for path in train_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        test_flat_frames = sorted(path for path in test_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

        samples: List[Dict] = []
        if train_frame_dirs:
            samples.extend(build_face_sequence_samples_from_dirs(train_frame_dirs, data_root=train_root, label_mode=label_mode, source_split="train_pool"))
        elif train_flat_frames:
            samples.extend(build_face_sequence_samples_from_paths(train_flat_frames, data_root=train_root, label_mode=label_mode, source_split="train_pool"))

        if test_frame_dirs:
            samples.extend(build_face_sequence_samples_from_dirs(test_frame_dirs, data_root=test_root, label_mode=label_mode, source_split="fixed_test"))
        elif test_flat_frames:
            samples.extend(build_face_sequence_samples_from_paths(test_flat_frames, data_root=test_root, label_mode=label_mode, source_split="fixed_test"))
    else:
        frame_dirs = [path for path in collect_frame_sequence_dirs(root) if path != root]
        flat_frames = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if frame_dirs:
            samples = build_face_sequence_samples_from_dirs(frame_dirs, data_root=root, label_mode=label_mode, source_split="all_face_image")
        elif flat_frames:
            samples = build_face_sequence_samples_from_paths(flat_frames, data_root=root, label_mode=label_mode, source_split="all_face_image")
        else:
            raise FileNotFoundError(f"No frame-sequence folders or grouped frame files found under: {root}")

    if not samples:
        raise FileNotFoundError(f"No usable all-face samples found under: {root}")

    diagnostics = {
        "mode": "all_face_sequences",
        "all_face_image": str(root),
        "sequence_count": len(samples),
        "subject_count": len({sample["subject_id"] for sample in samples}),
        "class_distribution": dict(Counter(sample["label"] for sample in samples)),
    }
    return samples, diagnostics


def resolve_label_space(label_mode: str) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    if label_mode == "binary":
        return YAWDD_BINARY_LABELS, YAWDD_BINARY_DISPLAY_MAP, YAWDD_BINARY_PROMPT_GROUPS
    return YAWDD_BEHAVIOR_LABELS, YAWDD_BEHAVIOR_DISPLAY_MAP, YAWDD_BEHAVIOR_PROMPT_GROUPS


def build_prompt_templates(prompt_template: str, prompt_set: str) -> List[str]:
    if prompt_set == "yawdd_behavior_5":
        return [
            "The driver is <LABEL>.",
            "Driver behavior: <LABEL>.",
            "This driver is currently <LABEL>.",
            "The person in the car is <LABEL>.",
            "Driving state: <LABEL>.",
        ]
    return aide_base.build_prompt_templates(prompt_template, prompt_set)


def build_class_prompts(label_mode: str, prompt_template: str, prompt_set: str) -> List[List[str]]:
    labels, display_map, prompt_groups = resolve_label_space(label_mode)
    if prompt_set == "yawdd_facial_cues":
        return [list(prompt_groups[label]) for label in labels]
    if prompt_set == "yawdd_facial_cues_single":
        return [[prompt_groups[label][0]] for label in labels]
    if prompt_set == "single":
        return [[prompt_template.replace("<LABEL>", display_map[label])] for label in labels]
    templates = build_prompt_templates(prompt_template, prompt_set)
    return [[template.replace("<LABEL>", display_map[label]) for template in templates] for label in labels]


def split_yawdd_samples_random_stratified(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    label_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        label_groups.setdefault(sample["label"], []).append(sample)

    rng = random.Random(seed)
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []
    for group in label_groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:n_train + n_val])
        test_samples.extend(shuffled[n_train + n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    split_info = {
        "mode": "split_random_stratified",
        "targets": {
            "train": round(len(samples) * train_ratio, 3),
            "val": round(len(samples) * val_ratio, 3),
            "test": round(len(samples) * test_ratio, 3),
        },
        "train": cremad_base.compute_split_summary(train_samples),
        "val": cremad_base.compute_split_summary(val_samples),
        "test": cremad_base.compute_split_summary(test_samples),
    }
    return {"train": train_samples, "val": val_samples, "test": test_samples}, split_info


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone YawDD CLIP drowsiness training with strict frozen CLIP adapter")
    parser.add_argument("--yawdd_root", default=DEFAULT_YAWDD_ROOT)
    parser.add_argument("--label_mode", choices=["binary", "behavior_4", "multi4"], default="binary")
    parser.add_argument("--include_dash", action="store_true")
    parser.add_argument("--external_face_root", default=None)
    parser.add_argument("--all_face_image", default=None)
    parser.add_argument(
        "--eval_mode",
        choices=["single", "fixed", "loso", "group_kfold", "random", "sequence_kfold"],
        default="fixed",
        help="Evaluation mode; sequence_kfold: stratified k-fold on sequence index, ignoring actor grouping",
    )
    parser.add_argument(
        "--baseline_mode",
        choices=["none", "dlib_mar_svm", "resnet18_finetune", "clip_linear_probe"],
        default="none",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--video_extensions", default=".avi,.mp4,.mov,.mkv")
    parser.add_argument("--split_name", choices=["train", "val", "test"], default=None)
    parser.add_argument("--cv_mode", choices=["5fold", "split"], default="5fold")
    parser.add_argument("--split_mode", choices=["speaker_independent", "random_stratified"], default="speaker_independent")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="The driver looks <LABEL>.")
    parser.add_argument("--prompt_set", default="yawdd_facial_cues")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--frame_sampling_mode", choices=["uniform", "middle_late", "diff_guided"], default="middle_late")
    parser.add_argument("--feature_layout", choices=["pooled", "sequence"], default="sequence")
    parser.add_argument("--sampling_window_start", type=float, default=0.4)
    parser.add_argument("--sampling_window_end", type=float, default=0.9)
    parser.add_argument("--diff_alpha", type=float, default=0.6)
    parser.add_argument("--diff_beta", type=float, default=0.4)
    parser.add_argument("--min_gap_ratio", type=float, default=0.08)
    parser.add_argument("--score_smooth_window", type=int, default=3)
    parser.add_argument("--frame_diff_metric", choices=["gray_l1", "gray_l2"], default="gray_l1")
    parser.add_argument("--ref_frame_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--disable_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.3)
    parser.add_argument("--pool_adapter_variant", choices=["legacy", "stronger"], default="legacy")
    parser.add_argument("--adapter_mode", choices=["full", "identity"], default="full",
                        help="full=trainable adapter MLP; identity=skip adapter, only L2-normalize CLIP features (for ablation)")
    parser.add_argument("--temporal_head", choices=["none", "attention", "transformer"], default="transformer")
    parser.add_argument("--temporal_module", choices=["none", "cgp_fg", "taga", "mean_pool"], default="none")
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--temporal_num_layers", type=int, default=1)
    parser.add_argument("--temporal_pool_mode", choices=["cls", "mean", "hybrid"], default="mean")
    parser.add_argument("--no_frame_gate", action="store_true")
    parser.add_argument("--no_gem", action="store_true")
    parser.add_argument("--no_residual_blend", action="store_true")
    parser.add_argument("--gem_init_p", type=float, default=1.0)
    parser.add_argument("--use_class_weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--disable_class_weight", dest="use_class_weight", action="store_false")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", dest="use_test_ensemble", action="store_true")
    parser.add_argument("--disable_test_ensemble", dest="use_test_ensemble", action="store_false")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_mode", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--feature_cache_dir", default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--multi4_oversample_labels", default="")
    parser.add_argument("--multi4_oversample_min_count", type=int, default=0)
    parser.add_argument("--multi4_oversample_mode", choices=["noise", "mixup"], default="noise")
    parser.add_argument("--multi4_oversample_noise_std", type=float, default=0.0)
    parser.add_argument("--multi4_mixup_alpha", type=float, default=0.4)
    parser.add_argument("--adapter_use_prompt_weight", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--adapter_use_class_temperature", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--adapter_use_class_bias", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--use_causal_contrastive", action="store_true")
    parser.add_argument("--ccl_weight", type=float, default=0.5)
    parser.add_argument("--ccl_temperature", type=float, default=0.5)
    parser.add_argument("--use_causal_alignment", action="store_true")
    parser.add_argument("--cfa_weight", type=float, default=0.1)
    parser.add_argument("--use_counterfactual_aug", action="store_true")
    parser.add_argument("--cda_prob", type=float, default=0.3)
    parser.add_argument("--cda_n_replace_max", type=int, default=3)
    parser.add_argument("--use_cda_v2_mixstyle", action="store_true")
    parser.add_argument("--cda_v2_prob", type=float, default=0.5)
    parser.add_argument("--cda_v2_kl_weight", type=float, default=0.5)
    parser.add_argument("--use_ccl_v2_counterfactual", action="store_true")
    parser.add_argument("--ccl_v2_weight", type=float, default=0.1)
    parser.add_argument("--ccl_v2_temperature", type=float, default=0.1)
    parser.add_argument("--use_cfa_v2_textanchor", action="store_true")
    parser.add_argument("--cfa_v2_weight", type=float, default=0.05)
    parser.add_argument("--cfa_v2_anchor_weight", type=float, default=1.0)
    parser.add_argument("--cfa_v2_ema_momentum", type=float, default=0.99)
    parser.add_argument("--force_reextract", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--extract_only", action="store_true")
    mode_group.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--delete_shards_after_merge", action="store_true")
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--run_zero_shot_eval", action="store_true")
    parser.add_argument("--dlib_shape_predictor", default=None)
    parser.set_defaults(
        use_class_weight=False,
        use_test_ensemble=True,
        pin_memory=True,
        use_amp=True,
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        log(f"[ARGS] ignored deprecated/unknown arguments: {' '.join(unknown_args)}")
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"
    if (args.extract_only or args.merge_shards) and not args.split_name:
        parser.error("--split_name must be provided when using --extract_only or --merge_shards")
    if args.total_shards < 1:
        parser.error("--total_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.total_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < total_shards")
    if args.n_folds < 2:
        parser.error("--n_folds must be >= 2")
    if args.multi4_oversample_min_count < 0:
        parser.error("--multi4_oversample_min_count must be >= 0")
    if args.multi4_oversample_noise_std < 0:
        parser.error("--multi4_oversample_noise_std must be >= 0")
    if args.multi4_mixup_alpha <= 0:
        parser.error("--multi4_mixup_alpha must be > 0")
    if args.scheduler_min_lr < 0:
        parser.error("--scheduler_min_lr must be >= 0")
    if args.gem_init_p <= 0:
        parser.error("--gem_init_p must be > 0")
    if args.temporal_head != "none" or args.temporal_module in {"cgp_fg", "taga", "mean_pool"}:
        args.feature_layout = "sequence"
    if args.ccl_weight < 0:
        parser.error("--ccl_weight must be >= 0")
    if args.ccl_temperature <= 0:
        parser.error("--ccl_temperature must be > 0")
    if args.cfa_weight < 0:
        parser.error("--cfa_weight must be >= 0")
    if args.cda_prob < 0 or args.cda_prob > 1:
        parser.error("--cda_prob must satisfy 0 <= cda_prob <= 1")
    if args.cda_n_replace_max < 1:
        parser.error("--cda_n_replace_max must be >= 1")
    if args.cda_v2_prob < 0 or args.cda_v2_prob > 1:
        parser.error("--cda_v2_prob must satisfy 0 <= cda_v2_prob <= 1")
    if args.cda_v2_kl_weight < 0:
        parser.error("--cda_v2_kl_weight must be >= 0")
    if args.ccl_v2_weight < 0:
        parser.error("--ccl_v2_weight must be >= 0")
    if args.ccl_v2_temperature <= 0:
        parser.error("--ccl_v2_temperature must be > 0")
    if args.cfa_v2_weight < 0:
        parser.error("--cfa_v2_weight must be >= 0")
    if args.cfa_v2_anchor_weight < 0:
        parser.error("--cfa_v2_anchor_weight must be >= 0")
    if args.cfa_v2_ema_momentum < 0 or args.cfa_v2_ema_momentum >= 1:
        parser.error("--cfa_v2_ema_momentum must satisfy 0 <= cfa_v2_ema_momentum < 1")

    def _resolve(value, default_on=True):
        if value == "on":
            return True
        if value == "off":
            return False
        return default_on

    args.resolved_use_prompt_weight = _resolve(args.adapter_use_prompt_weight)
    args.resolved_use_class_temperature = _resolve(args.adapter_use_class_temperature)
    args.resolved_use_class_bias = _resolve(args.adapter_use_class_bias)
    if args.training_seed is None:
        args.training_seed = args.seed
    return args


def build_external_fixed_dataset_summary(
    args,
    emotion_labels: List[str],
    dataset_diagnostics: Dict[str, object],
    split_info: Dict[str, object],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    train_cache_payload: Dict[str, object],
    val_cache_payload: Dict[str, object],
    test_cache_payload: Dict[str, object],
    train_cache_samples: List[Dict],
    val_cache_samples: List[Dict],
    test_cache_samples: List[Dict],
) -> Dict[str, object]:
    return {
        "total": len(train_samples) + len(val_samples) + len(test_samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "subject_count": dataset_diagnostics["train_pool_subjects"] + dataset_diagnostics["fixed_test_subjects"],
        "class_distribution_total": {
            label: dataset_diagnostics["train_pool_distribution"].get(label, 0)
            + dataset_diagnostics["fixed_test_distribution"].get(label, 0)
            for label in emotion_labels
        },
        "behavior_distribution_total": {},
        "view_distribution_total": {"face_crop": len(train_samples) + len(val_samples) + len(test_samples)},
        "group_distribution_total": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "extension_distribution_total": {".jpg": len(train_samples) + len(val_samples) + len(test_samples)},
        "source_dirs": [
            str(Path(args.external_face_root) / "train_face_image"),
            str(Path(args.external_face_root) / "test_face_image"),
        ],
        "invalid_name_examples": [],
        "split_summary": split_info,
        "failed_samples_train": train_cache_payload.get("failed_count", 0),
        "failed_samples_val": val_cache_payload.get("failed_count", 0),
        "failed_samples_test": test_cache_payload.get("failed_count", 0),
        "label_distribution_train": dict(Counter(sample["label"] for sample in train_cache_samples)),
        "label_distribution_val": dict(Counter(sample["label"] for sample in val_cache_samples)),
        "label_distribution_test": dict(Counter(sample["label"] for sample in test_cache_samples)),
    }


def build_all_face_dataset_summary(
    args,
    dataset_diagnostics: Dict[str, object],
    split_info: Dict[str, object],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    train_cache_payload: Dict[str, object],
    val_cache_payload: Dict[str, object],
    test_cache_payload: Dict[str, object],
    train_cache_samples: List[Dict],
    val_cache_samples: List[Dict],
    test_cache_samples: List[Dict],
) -> Dict[str, object]:
    total_count = len(train_samples) + len(val_samples) + len(test_samples)
    return {
        "total": total_count,
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "subject_count": len({sample["subject_id"] for sample in train_samples + val_samples + test_samples}),
        "class_distribution_total": dict(Counter(sample["label"] for sample in train_samples + val_samples + test_samples)),
        "behavior_distribution_total": {},
        "view_distribution_total": {"face_crop": total_count},
        "group_distribution_total": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "extension_distribution_total": {".jpg": total_count},
        "source_dirs": [str(Path(args.all_face_image).resolve())],
        "invalid_name_examples": [],
        "split_summary": split_info,
        "all_face_diagnostics": dataset_diagnostics,
        "failed_samples_train": train_cache_payload.get("failed_count", 0),
        "failed_samples_val": val_cache_payload.get("failed_count", 0),
        "failed_samples_test": test_cache_payload.get("failed_count", 0),
        "label_distribution_train": dict(Counter(sample["label"] for sample in train_cache_samples)),
        "label_distribution_val": dict(Counter(sample["label"] for sample in val_cache_samples)),
        "label_distribution_test": dict(Counter(sample["label"] for sample in test_cache_samples)),
    }


def build_fixed_all_face_dataset_summary(
    args,
    dataset_diagnostics: Dict[str, object],
    split_info: Dict[str, object],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    train_cache_payload: Dict[str, object],
    val_cache_payload: Dict[str, object],
    test_cache_payload: Dict[str, object],
    train_cache_samples: List[Dict],
    val_cache_samples: List[Dict],
    test_cache_samples: List[Dict],
) -> Dict[str, object]:
    total_count = len(train_samples) + len(val_samples) + len(test_samples)
    return {
        "total": total_count,
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "subject_count": dataset_diagnostics["train_pool_subjects"] + dataset_diagnostics["fixed_test_subjects"],
        "class_distribution_total": {
            **dataset_diagnostics["train_pool_distribution"],
            **{
                key: int(dataset_diagnostics["train_pool_distribution"].get(key, 0))
                + int(dataset_diagnostics["fixed_test_distribution"].get(key, 0))
                for key in set(dataset_diagnostics["train_pool_distribution"]) | set(dataset_diagnostics["fixed_test_distribution"])
            },
        },
        "behavior_distribution_total": {},
        "view_distribution_total": {"face_crop": total_count},
        "group_distribution_total": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "extension_distribution_total": {".jpg": total_count},
        "source_dirs": [
            str(Path(args.all_face_image) / "train_face_image"),
            str(Path(args.all_face_image) / "test_face_image"),
        ],
        "invalid_name_examples": [],
        "split_summary": split_info,
        "all_face_diagnostics": dataset_diagnostics,
        "failed_samples_train": train_cache_payload.get("failed_count", 0),
        "failed_samples_val": val_cache_payload.get("failed_count", 0),
        "failed_samples_test": test_cache_payload.get("failed_count", 0),
        "label_distribution_train": dict(Counter(sample["label"] for sample in train_cache_samples)),
        "label_distribution_val": dict(Counter(sample["label"] for sample in val_cache_samples)),
        "label_distribution_test": dict(Counter(sample["label"] for sample in test_cache_samples)),
    }


def build_standard_yawdd_dataset_summary(
    args,
    dataset_diagnostics: Dict[str, object],
    split_info: Dict[str, object],
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    train_cache_payload: Dict[str, object],
    val_cache_payload: Dict[str, object],
    test_cache_payload: Dict[str, object],
    train_cache_samples: List[Dict],
    val_cache_samples: List[Dict],
    test_cache_samples: List[Dict],
) -> Dict[str, object]:
    return {
        "total": dataset_diagnostics["total_valid_samples"],
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "subject_count": dataset_diagnostics["subject_count"],
        "class_distribution_total": dataset_diagnostics["class_distribution"],
        "behavior_distribution_total": dataset_diagnostics["behavior_distribution"],
        "view_distribution_total": dataset_diagnostics["view_distribution"],
        "group_distribution_total": dataset_diagnostics["group_distribution"],
        "extension_distribution_total": dataset_diagnostics["extension_distribution"],
        "source_dirs": dataset_diagnostics["source_dirs"],
        "invalid_name_examples": dataset_diagnostics["invalid_name_examples"],
        "split_summary": split_info,
        "failed_samples_train": train_cache_payload.get("failed_count", 0),
        "failed_samples_val": val_cache_payload.get("failed_count", 0),
        "failed_samples_test": test_cache_payload.get("failed_count", 0),
        "label_distribution_train": dict(Counter(sample["label"] for sample in train_cache_samples)),
        "label_distribution_val": dict(Counter(sample["label"] for sample in val_cache_samples)),
        "label_distribution_test": dict(Counter(sample["label"] for sample in test_cache_samples)),
    }


def build_single_run_aggregate(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        key: {"mean": round(float(value), 6), "std": 0.0}
        for key, value in metrics.items()
        if key in {"accuracy", "precision", "recall", "f1"}
    }


def extract_primary_metrics(
    test_summary: Dict[str, object],
    label_mode: str,
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, float]:
    if label_mode == "multi4":
        weighted = test_summary["weighted_average_metrics"]
        return {
            "accuracy": round(float(test_summary["accuracy"]), 6),
            "precision": round(float(weighted["precision"]), 6),
            "recall": round(float(weighted["recall"]), 6),
            "f1": round(float(weighted["f1"]), 6),
        }
    fold_metrics = summarize_fold_classification_metrics(y_true, y_pred, labels)
    return {
        "accuracy": fold_metrics["accuracy"],
        "precision": fold_metrics["precision"],
        "recall": fold_metrics["recall"],
        "f1": fold_metrics["f1"],
    }


def select_metric_value(
    metric_name: str,
    summary: Dict[str, object],
    label_mode: str,
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> float:
    if metric_name == "accuracy":
        return float(summary["accuracy"])
    if label_mode == "multi4":
        return float(summary["weighted_average_metrics"]["f1"])
    if "weighted_f1" in summary:
        return float(summary["weighted_f1"])
    return float(summarize_fold_classification_metrics(y_true, y_pred, labels)["f1"])


def resolve_effective_prompt_set(args, label_mode: str) -> str:
    return args.prompt_set


def resolve_effective_temporal_pool_mode(args, label_mode: str) -> str:
    if label_mode == "multi4" and args.temporal_pool_mode == "mean":
        return "hybrid"
    return args.temporal_pool_mode


def resolve_effective_clip_mode(args, label_mode: str) -> str:
    if label_mode == "multi4" and args.clip_mode == "auto":
        return "offline_only"
    return args.clip_mode


def load_clip_components(model_id: str, device: str, clip_mode: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False)
        except Exception:
            processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


def sample_sequence_frame_paths(sample: Dict, num_frames: int) -> List[str]:
    frame_paths = list(sample.get("frame_paths") or [])
    if not frame_paths:
        frame_path = sample.get("frame_path") or sample.get("video_path")
        frame_paths = [frame_path] if frame_path else []
    return [str(path) for path in aide_base.sample_frame_paths(frame_paths, num_frames)]


def build_result_payload(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    benchmark_mode: str,
    dataset_summary: Dict[str, object],
    class_distribution: Dict[str, Dict[str, int]],
    val_summary: Dict[str, object],
    test_summary: Dict[str, object],
    val_true: List[str],
    val_pred: List[str],
    test_true: List[str],
    test_pred: List[str],
    resolved_feature_cache_paths: Optional[Dict[str, str]] = None,
    output_path_value: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    zero_shot_result: Optional[Dict[str, object]] = None,
    extra_config: Optional[Dict[str, object]] = None,
    extra_payload: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    class_names = resolve_class_names(label_mode, emotion_labels)
    effective_prompt_set = resolve_effective_prompt_set(args, label_mode)
    effective_clip_mode = resolve_effective_clip_mode(args, label_mode)
    effective_temporal_pool_mode = resolve_effective_temporal_pool_mode(args, label_mode)
    primary_metrics = extract_primary_metrics(test_summary, label_mode, test_true, test_pred, emotion_labels)
    result = {
        "git_commit": resolve_git_commit(),
        "config": {
            "method": "clip_supervised_text_video_drowsiness" if args.baseline_mode == "none" else args.baseline_mode,
            "execution_mode": "strict_frozen_clip_adapter" if args.baseline_mode == "none" else "baseline",
            "dataset": "YawDD",
            "task": "drowsiness" if len(emotion_labels) == 2 else "driver_behavior",
            "benchmark_mode": benchmark_mode,
            "baseline_mode": args.baseline_mode,
            "cv_mode": args.cv_mode,
            "fold_idx": args.fold_idx if args.cv_mode == "5fold" else None,
            "train_ratio": args.train_ratio if args.cv_mode == "split" else None,
            "val_ratio": args.val_ratio if args.cv_mode == "split" else None,
            "yawdd_root": str(Path(args.yawdd_root).resolve()),
            "label_mode": label_mode,
            "class_names": class_names,
            "include_dash": args.include_dash,
            "class_count": len(emotion_labels),
            "video_extensions": sorted(cremad_base.parse_extension_set(args.video_extensions)),
            "model_id": args.model_id,
            "clip_mode": effective_clip_mode,
            "prompt_template": args.prompt_template,
            "prompt_set": effective_prompt_set,
            "epochs": args.epochs,
            "extract_batch_size": args.extract_batch_size,
            "train_batch_size": args.train_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "num_frames": args.num_frames,
            "feature_layout": args.feature_layout,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "pool_adapter_variant": args.pool_adapter_variant,
            "adapter_mode": args.adapter_mode,
            "temporal_head": args.temporal_head,
            "temporal_module": args.temporal_module,
            "use_frame_gate": not args.no_frame_gate,
            "use_gem": not args.no_gem,
            "use_residual_blend": not args.no_residual_blend,
            "gem_init_p": args.gem_init_p,
            "temporal_num_heads": args.temporal_num_heads,
            "temporal_num_layers": args.temporal_num_layers,
            "temporal_pool_mode": effective_temporal_pool_mode,
            "use_class_weight": True if label_mode == "multi4" else args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "loss_type": "ce" if label_mode == "multi4" else args.loss_type,
            "requested_loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "use_amp": args.use_amp,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "frame_sampling_mode": args.frame_sampling_mode,
            "sampling_window_start": args.sampling_window_start,
            "sampling_window_end": args.sampling_window_end,
            "diff_alpha": args.diff_alpha,
            "diff_beta": args.diff_beta,
            "min_gap_ratio": args.min_gap_ratio,
            "score_smooth_window": args.score_smooth_window,
            "frame_diff_metric": args.frame_diff_metric,
            "ref_frame_ratio": args.ref_frame_ratio,
            "feature_cache_dir": args.feature_cache_dir,
            "resolved_feature_cache_paths": resolved_feature_cache_paths or {},
            "checkpoint_output": str(checkpoint_path) if checkpoint_path else None,
            "log_file": args.log_file,
            "seed": args.seed,
            "max_sequences": args.max_sequences,
            "run_zero_shot_eval": args.run_zero_shot_eval,
            "eval_mode": args.eval_mode,
            "all_face_image": str(Path(args.all_face_image).resolve()) if args.all_face_image else None,
            "dlib_shape_predictor": args.dlib_shape_predictor,
            "use_causal_contrastive": args.use_causal_contrastive,
            "ccl_weight": args.ccl_weight,
            "ccl_temperature": args.ccl_temperature,
            "use_causal_alignment": args.use_causal_alignment,
            "cfa_weight": args.cfa_weight,
            "use_counterfactual_aug": args.use_counterfactual_aug,
            "cda_prob": args.cda_prob,
            "cda_n_replace_max": args.cda_n_replace_max,
            "use_cda_v2_mixstyle": args.use_cda_v2_mixstyle,
            "cda_v2_prob": args.cda_v2_prob,
            "cda_v2_kl_weight": args.cda_v2_kl_weight,
            "use_ccl_v2_counterfactual": args.use_ccl_v2_counterfactual,
            "ccl_v2_weight": args.ccl_v2_weight,
            "ccl_v2_temperature": args.ccl_v2_temperature,
            "use_cfa_v2_textanchor": args.use_cfa_v2_textanchor,
            "cfa_v2_weight": args.cfa_v2_weight,
            "cfa_v2_anchor_weight": args.cfa_v2_anchor_weight,
            "cfa_v2_ema_momentum": args.cfa_v2_ema_momentum,
        },
        "dataset": dataset_summary,
        "label_mode": label_mode,
        "class_names": class_names,
        "class_distribution": class_distribution,
        "label_map": {label: label for label in emotion_labels},
        "prompt_groups": prompt_groups,
        "val": val_summary,
        "test": test_summary,
        "val_metrics": extract_primary_metrics(val_summary, label_mode, val_true, val_pred, emotion_labels),
        "test_metrics": primary_metrics,
        "test_class_distribution": class_distribution.get("test", {}),
        "folds": [],
        "aggregate": build_single_run_aggregate(primary_metrics),
        "zero_shot": zero_shot_result,
    }
    if label_mode == "multi4":
        result["per_class_metrics"] = test_summary["per_class_metrics"]
        result["confusion_matrix"] = test_summary["confusion_matrix"]
        result["classification_report"] = test_summary["classification_report"]
    if extra_config:
        result["config"].update(extra_config)
    if extra_payload:
        result.update(extra_payload)
    return result


def write_result_payload(result: Dict[str, object], output_path_value: str) -> None:
    output_path = Path(output_path_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def save_checkpoint_payload(checkpoint_path: Optional[Path], payload: Optional[Dict[str, object]]) -> None:
    if checkpoint_path is None or payload is None:
        return
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


class ResnetSequenceDataset:
    def __init__(self, samples: List[Dict], label2idx: Dict[str, int], num_frames: int, transform):
        self.samples = list(samples)
        self.label2idx = dict(label2idx)
        self.num_frames = int(num_frames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        sample = self.samples[index]
        frame_paths = sample_sequence_frame_paths(sample, self.num_frames)
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(image))
        if not frames:
            raise RuntimeError(f"No frames available for sample: {sample.get('sequence_id')}")
        if len(frames) < self.num_frames:
            frames.extend([frames[-1].clone() for _ in range(self.num_frames - len(frames))])
        return torch.stack(frames[: self.num_frames], dim=0), self.label2idx[sample["label"]], sample


def evaluate_sequence_model(model, data_loader, idx2label: Dict[int, str], device: str) -> Tuple[List[str], List[str]]:
    import torch

    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    with torch.no_grad():
        for frames, targets, _ in data_loader:
            frames = frames.to(device)
            targets = targets.to(device)
            batch_size, frame_count = frames.shape[:2]
            logits = model(frames.view(batch_size * frame_count, *frames.shape[2:]))
            logits = logits.view(batch_size, frame_count, -1).mean(dim=1)
            pred_indices = logits.argmax(dim=-1).detach().cpu().tolist()
            true_indices = targets.detach().cpu().tolist()
            y_true.extend([idx2label[index] for index in true_indices])
            y_pred.extend([idx2label[index] for index in pred_indices])
    return y_true, y_pred


def extract_clip_linear_probe_features(samples: List[Dict], processor, model, args, split_name: str):
    features = aide_base.extract_image_features(
        samples,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        tag=f"{split_name}_linear_probe",
        feature_layout="pooled",
    )
    return features.cpu().numpy()


def compute_dlib_mar_from_frame(gray_image, detector, predictor) -> Optional[float]:
    import numpy as np

    detections = detector(gray_image, 1)
    if len(detections) == 0:
        import dlib

        detections = [dlib.rectangle(left=0, top=0, right=gray_image.shape[1] - 1, bottom=gray_image.shape[0] - 1)]
    shape = predictor(gray_image, detections[0])
    points = np.array([[shape.part(index).x, shape.part(index).y] for index in range(68)], dtype=np.float32)
    inner = points[60:68]
    horizontal = np.linalg.norm(inner[0] - inner[4])
    if horizontal <= 1e-6:
        return None
    vertical = np.linalg.norm(inner[1] - inner[7]) + np.linalg.norm(inner[2] - inner[6]) + np.linalg.norm(inner[3] - inner[5])
    return float(vertical / (3.0 * horizontal))


def extract_dlib_mar_features(samples: List[Dict], args) -> List[List[float]]:
    import cv2
    import numpy as np

    try:
        import dlib
    except ImportError as exc:
        raise RuntimeError("dlib is required for baseline_mode=dlib_mar_svm") from exc
    if not args.dlib_shape_predictor:
        raise ValueError("--dlib_shape_predictor is required for baseline_mode=dlib_mar_svm")
    predictor_path = Path(args.dlib_shape_predictor)
    if not predictor_path.exists():
        raise FileNotFoundError(f"dlib shape predictor not found: {predictor_path}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    feature_rows: List[List[float]] = []
    for sample in samples:
        mar_values: List[float] = []
        for frame_path in sample_sequence_frame_paths(sample, args.num_frames):
            gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            try:
                mar = compute_dlib_mar_from_frame(gray, detector, predictor)
            except Exception:
                mar = None
            if mar is not None:
                mar_values.append(float(mar))
        if not mar_values:
            mar_values = [0.0]
        mar_array = np.asarray(mar_values, dtype=np.float32)
        feature_rows.append(
            [
                float(mar_array.mean()),
                float(mar_array.std()),
                float(mar_array.min()),
                float(mar_array.max()),
                float(mar_array.max() - mar_array.min()),
                float(np.mean(np.diff(mar_array)) if mar_array.size > 1 else 0.0),
            ]
        )
    return feature_rows


def run_clip_linear_probe_baseline(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits: Dict[str, List[Dict]],
    split_info: Dict[str, object],
    benchmark_mode: str,
    dataset_summary_builder,
    dataset_summary_builder_kwargs: Dict[str, object],
    output_path_override: Optional[str] = None,
    checkpoint_output_override: Optional[str] = None,
) -> Dict[str, object]:
    import pickle
    from sklearn.linear_model import LogisticRegression

    processor, model = load_clip_components(args.model_id, args.device, resolve_effective_clip_mode(args, label_mode))
    train_samples = list(splits["train"])
    val_samples = list(splits["val"])
    test_samples = list(splits["test"])
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    train_x = extract_clip_linear_probe_features(train_samples, processor, model, args, "train")
    val_x = extract_clip_linear_probe_features(val_samples, processor, model, args, "val")
    test_x = extract_clip_linear_probe_features(test_samples, processor, model, args, "test")
    train_y = [label2idx[sample["label"]] for sample in train_samples]
    val_y = [label2idx[sample["label"]] for sample in val_samples]
    test_y = [label2idx[sample["label"]] for sample in test_samples]

    class_weight = "balanced" if (label_mode == "multi4" or args.use_class_weight) else None
    classifier = LogisticRegression(
        max_iter=2000,
        random_state=args.seed,
        multi_class="auto",
        class_weight=class_weight,
    )
    classifier.fit(train_x, train_y)
    val_pred = [idx2label[int(index)] for index in classifier.predict(val_x)]
    test_pred = [idx2label[int(index)] for index in classifier.predict(test_x)]
    val_true = [idx2label[int(index)] for index in val_y]
    test_true = [idx2label[int(index)] for index in test_y]
    val_summary = summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
    test_summary = summarize_predictions_by_mode(test_true, test_pred, emotion_labels, label_mode)
    class_names = resolve_class_names(label_mode, emotion_labels)
    class_distribution = compute_split_class_distribution(emotion_labels, class_names, train_samples, val_samples, test_samples)
    dataset_summary = dataset_summary_builder(
        args=args,
        split_info=split_info,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_cache_payload={"failed_count": 0},
        val_cache_payload={"failed_count": 0},
        test_cache_payload={"failed_count": 0},
        train_cache_samples=train_samples,
        val_cache_samples=val_samples,
        test_cache_samples=test_samples,
        **dataset_summary_builder_kwargs,
    )
    output_path_value = output_path_override or args.output
    checkpoint_path = Path(checkpoint_output_override) if checkpoint_output_override else (
        Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(output_path_value))
    )
    result = build_result_payload(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode=benchmark_mode,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        extra_config={"baseline_classifier": "logistic_regression", "feature_source": "clip_image_features"},
    )
    write_result_payload(result, output_path_value)
    save_checkpoint_payload(
        checkpoint_path,
        {
            "checkpoint_type": "clip_linear_probe",
            "config": result["config"],
            "classifier_bytes": pickle.dumps(classifier),
            "label2idx": label2idx,
            "idx2label": idx2label,
            "output_path": output_path_value,
        },
    )
    log(f"[DONE] saved YawDD report to: {output_path_value}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    return result


def run_dlib_mar_svm_baseline(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits: Dict[str, List[Dict]],
    split_info: Dict[str, object],
    benchmark_mode: str,
    dataset_summary_builder,
    dataset_summary_builder_kwargs: Dict[str, object],
    output_path_override: Optional[str] = None,
    checkpoint_output_override: Optional[str] = None,
) -> Dict[str, object]:
    import pickle
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    train_samples = list(splits["train"])
    val_samples = list(splits["val"])
    test_samples = list(splits["test"])
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    train_x = extract_dlib_mar_features(train_samples, args)
    val_x = extract_dlib_mar_features(val_samples, args)
    test_x = extract_dlib_mar_features(test_samples, args)
    train_y = [label2idx[sample["label"]] for sample in train_samples]
    val_y = [label2idx[sample["label"]] for sample in val_samples]
    test_y = [label2idx[sample["label"]] for sample in test_samples]

    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced" if (label_mode == "multi4" or args.use_class_weight) else None),
    )
    classifier.fit(train_x, train_y)
    val_pred = [idx2label[int(index)] for index in classifier.predict(val_x)]
    test_pred = [idx2label[int(index)] for index in classifier.predict(test_x)]
    val_true = [idx2label[int(index)] for index in val_y]
    test_true = [idx2label[int(index)] for index in test_y]
    val_summary = summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
    test_summary = summarize_predictions_by_mode(test_true, test_pred, emotion_labels, label_mode)
    class_names = resolve_class_names(label_mode, emotion_labels)
    class_distribution = compute_split_class_distribution(emotion_labels, class_names, train_samples, val_samples, test_samples)
    dataset_summary = dataset_summary_builder(
        args=args,
        split_info=split_info,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_cache_payload={"failed_count": 0},
        val_cache_payload={"failed_count": 0},
        test_cache_payload={"failed_count": 0},
        train_cache_samples=train_samples,
        val_cache_samples=val_samples,
        test_cache_samples=test_samples,
        **dataset_summary_builder_kwargs,
    )
    output_path_value = output_path_override or args.output
    checkpoint_path = Path(checkpoint_output_override) if checkpoint_output_override else (
        Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(output_path_value))
    )
    result = build_result_payload(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode=benchmark_mode,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        extra_config={"baseline_classifier": "svm_rbf", "feature_source": "dlib_mar_statistics"},
    )
    write_result_payload(result, output_path_value)
    save_checkpoint_payload(
        checkpoint_path,
        {
            "checkpoint_type": "dlib_mar_svm",
            "config": result["config"],
            "classifier_bytes": pickle.dumps(classifier),
            "label2idx": label2idx,
            "idx2label": idx2label,
            "output_path": output_path_value,
        },
    )
    log(f"[DONE] saved YawDD report to: {output_path_value}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    return result


def run_resnet18_finetune_baseline(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits: Dict[str, List[Dict]],
    split_info: Dict[str, object],
    benchmark_mode: str,
    dataset_summary_builder,
    dataset_summary_builder_kwargs: Dict[str, object],
    output_path_override: Optional[str] = None,
    checkpoint_output_override: Optional[str] = None,
) -> Dict[str, object]:
    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights, resnet18

    train_samples = list(splits["train"])
    val_samples = list(splits["val"])
    test_samples = list(splits["test"])
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    train_dataset = ResnetSequenceDataset(train_samples, label2idx, args.num_frames, transform)
    val_dataset = ResnetSequenceDataset(val_samples, label2idx, args.num_frames, transform)
    test_dataset = ResnetSequenceDataset(test_samples, label2idx, args.num_frames, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(emotion_labels))
    model = model.to(args.device)
    class_weight_tensor = None
    if label_mode == "multi4" or args.use_class_weight:
        class_weight_tensor = torch.tensor(compute_multi4_class_weights(train_samples, emotion_labels), dtype=torch.float32, device=args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and str(args.device).startswith("cuda")))

    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for frames, targets, _ in train_loader:
            frames = frames.to(args.device)
            targets = targets.to(args.device)
            batch_size, frame_count = frames.shape[:2]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp and str(args.device).startswith("cuda"))):
                logits = model(frames.view(batch_size * frame_count, *frames.shape[2:]))
                logits = logits.view(batch_size, frame_count, -1).mean(dim=1)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        val_true, val_pred = evaluate_sequence_model(model, val_loader, idx2label, args.device)
        val_summary = summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
        current_metric = select_metric_value(args.select_metric, val_summary, label_mode, val_true, val_pred, emotion_labels)
        if current_metric > best_metric + args.early_stopping_min_delta:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    val_true, val_pred = evaluate_sequence_model(model, val_loader, idx2label, args.device)
    test_true, test_pred = evaluate_sequence_model(model, test_loader, idx2label, args.device)
    val_summary = summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
    test_summary = summarize_predictions_by_mode(test_true, test_pred, emotion_labels, label_mode)
    class_names = resolve_class_names(label_mode, emotion_labels)
    class_distribution = compute_split_class_distribution(emotion_labels, class_names, train_samples, val_samples, test_samples)
    dataset_summary = dataset_summary_builder(
        args=args,
        split_info=split_info,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_cache_payload={"failed_count": 0},
        val_cache_payload={"failed_count": 0},
        test_cache_payload={"failed_count": 0},
        train_cache_samples=train_samples,
        val_cache_samples=val_samples,
        test_cache_samples=test_samples,
        **dataset_summary_builder_kwargs,
    )
    output_path_value = output_path_override or args.output
    checkpoint_path = Path(checkpoint_output_override) if checkpoint_output_override else (
        Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(output_path_value))
    )
    result = build_result_payload(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode=benchmark_mode,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        extra_config={"baseline_backbone": "resnet18", "feature_source": "sequence_frame_average_logits"},
    )
    write_result_payload(result, output_path_value)
    save_checkpoint_payload(
        checkpoint_path,
        {
            "checkpoint_type": "resnet18_finetune",
            "config": result["config"],
            "model_state_dict": model.state_dict(),
            "label2idx": label2idx,
            "idx2label": idx2label,
            "output_path": output_path_value,
        },
    )
    log(f"[DONE] saved YawDD report to: {output_path_value}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    return result


def run_baseline_training_evaluation(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits: Dict[str, List[Dict]],
    split_info: Dict[str, object],
    benchmark_mode: str,
    dataset_summary_builder,
    dataset_summary_builder_kwargs: Dict[str, object],
    output_path_override: Optional[str] = None,
    checkpoint_output_override: Optional[str] = None,
) -> Dict[str, object]:
    if args.baseline_mode == "clip_linear_probe":
        return run_clip_linear_probe_baseline(
            args=args,
            label_mode=label_mode,
            emotion_labels=emotion_labels,
            prompt_groups=prompt_groups,
            splits=splits,
            split_info=split_info,
            benchmark_mode=benchmark_mode,
            dataset_summary_builder=dataset_summary_builder,
            dataset_summary_builder_kwargs=dataset_summary_builder_kwargs,
            output_path_override=output_path_override,
            checkpoint_output_override=checkpoint_output_override,
        )
    if args.baseline_mode == "dlib_mar_svm":
        return run_dlib_mar_svm_baseline(
            args=args,
            label_mode=label_mode,
            emotion_labels=emotion_labels,
            prompt_groups=prompt_groups,
            splits=splits,
            split_info=split_info,
            benchmark_mode=benchmark_mode,
            dataset_summary_builder=dataset_summary_builder,
            dataset_summary_builder_kwargs=dataset_summary_builder_kwargs,
            output_path_override=output_path_override,
            checkpoint_output_override=checkpoint_output_override,
        )
    if args.baseline_mode == "resnet18_finetune":
        return run_resnet18_finetune_baseline(
            args=args,
            label_mode=label_mode,
            emotion_labels=emotion_labels,
            prompt_groups=prompt_groups,
            splits=splits,
            split_info=split_info,
            benchmark_mode=benchmark_mode,
            dataset_summary_builder=dataset_summary_builder,
            dataset_summary_builder_kwargs=dataset_summary_builder_kwargs,
            output_path_override=output_path_override,
            checkpoint_output_override=checkpoint_output_override,
        )
    raise ValueError(f"Unsupported baseline_mode: {args.baseline_mode}")


def run_single_training_evaluation(
    args,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits: Dict[str, List[Dict]],
    split_info: Dict[str, object],
    benchmark_mode: str,
    dataset_summary_builder,
    dataset_summary_builder_kwargs: Dict[str, object],
    output_path_override: Optional[str] = None,
    checkpoint_output_override: Optional[str] = None,
) -> Dict[str, object]:
    import torch

    train_samples = cremad_base.build_split_samples_with_index(splits["train"], split_name="train")
    val_samples = cremad_base.build_split_samples_with_index(splits["val"], split_name="val")
    test_samples = cremad_base.build_split_samples_with_index(splits["test"], split_name="test")
    split_samples_map = {"train": train_samples, "val": val_samples, "test": test_samples}
    log(f"[DATA] split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    split_cache_plans = {
        split_name: cremad_base.build_split_cache_plan(
            args.feature_cache_dir,
            f"YawDD-{label_mode}",
            split_name,
            split_samples_map[split_name],
            args,
        )
        for split_name in ["train", "val", "test"]
    }
    for split_name, cache_plan in split_cache_plans.items():
        log(
            f"[CACHE] plan for {split_name}: final={cache_plan['final_path']} "
            f"existing_shards={cremad_base.count_existing_shards(cache_plan)}/{args.total_shards}"
        )

    if args.merge_shards:
        merged_path = cremad_base.merge_feature_shards(
            split_cache_plans[args.split_name],
            delete_shards_after_merge=args.delete_shards_after_merge,
        )
        log(f"[DONE] merge-only mode finished: {merged_path}")
        return {"mode": "merge_only", "merged_path": str(merged_path)}

    effective_clip_mode = resolve_effective_clip_mode(args, label_mode)
    effective_temporal_pool_mode = resolve_effective_temporal_pool_mode(args, label_mode)
    processor, model = load_clip_components(args.model_id, args.device, effective_clip_mode)
    log(f"[INFO] model loaded: {args.model_id} on {args.device}")

    if args.extract_only:
        target_samples = split_samples_map[args.split_name]
        if args.total_shards == 1:
            cremad_base.extract_split_feature_cache(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        else:
            cremad_base.extract_feature_shard(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        log("[DONE] extraction-only mode finished")
        return {"mode": "extract_only", "split_name": args.split_name}

    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    idx2label = {idx: label for idx, label in enumerate(emotion_labels)}
    resolved_split_cache_paths: Dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        resolved_path = cremad_base.ensure_training_split_cache(
            split_name=split_name,
            split_samples=split_samples_map[split_name],
            cache_plan=split_cache_plans[split_name],
            processor=processor,
            model=model,
            args=args,
        )
        resolved_split_cache_paths[split_name] = str(resolved_path)

    train_x, train_y, train_cache_samples, train_cache_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["train"]["final_path"],
        label2idx,
    )
    val_x, val_y, val_cache_samples, val_cache_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["val"]["final_path"],
        label2idx,
    )
    test_x, test_y, test_cache_samples, test_cache_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["test"]["final_path"],
        label2idx,
    )
    class_names = resolve_class_names(label_mode, emotion_labels)

    def build_actor_tensor(samples: List[Dict]):
        import torch

        actor_names = [str(sample.get("actor_id") or sample.get("subject_id") or "") for sample in samples]
        actor_vocab = {name: idx for idx, name in enumerate(sorted(set(actor_names)))}
        return torch.tensor([actor_vocab[name] for name in actor_names], dtype=torch.long)

    train_actor_ids = build_actor_tensor(train_cache_samples)

    class_distribution = compute_split_class_distribution(
        emotion_labels,
        class_names,
        train_cache_samples,
        val_cache_samples,
        test_cache_samples,
    )
    train_model_x = train_x
    train_model_y = train_y
    train_model_samples = list(train_cache_samples)
    training_augmentation_summary = None
    class_weights_override = None
    effective_use_class_weight = args.use_class_weight
    effective_loss_type = args.loss_type
    if label_mode == "multi4":
        log(f"[DATA] multi4 class distribution train={class_distribution['train']}")
        log(f"[DATA] multi4 class distribution val={class_distribution['val']}")
        log(f"[DATA] multi4 class distribution test={class_distribution['test']}")
        oversample_labels = [item.strip() for item in str(args.multi4_oversample_labels or "").split(",") if item.strip()]
        if oversample_labels and args.multi4_oversample_min_count > 0:
            train_model_x, train_model_y, train_model_samples, training_augmentation_summary = augment_cached_training_features(
                train_model_x,
                train_model_y,
                train_model_samples,
                labels_to_augment=oversample_labels,
                min_count=args.multi4_oversample_min_count,
                noise_std=args.multi4_oversample_noise_std,
                seed=args.seed,
                augmentation_mode=args.multi4_oversample_mode,
                mixup_alpha=args.multi4_mixup_alpha,
            )
            if training_augmentation_summary is not None:
                log("[TRAIN] multi4 oversample=" + json.dumps(training_augmentation_summary, ensure_ascii=False))
        multi4_class_weights = compute_multi4_class_weights(train_model_samples, emotion_labels)
        log(
            "[TRAIN] multi4 class weights="
            + json.dumps(
                {class_name: round(weight, 6) for class_name, weight in zip(class_names, multi4_class_weights)},
                ensure_ascii=False,
            )
        )
        class_weights_override = torch.tensor(multi4_class_weights, dtype=torch.float32, device=args.device)
        effective_use_class_weight = True
        effective_loss_type = "ce"
    else:
        oversample_labels = []
    text_features = aide_base.extract_text_features(prompt_groups, processor, model, args.device)
    configure_random_seeds(args.training_seed)
    log(f"[SEED] split_seed={args.seed} training_seed={args.training_seed}")

    feature_dim = int(train_model_x.shape[-1])
    temporal_param_count = 0
    adapter_param_count = 0
    pch_param_count = 0
    if args.temporal_module in {"cgp_fg", "taga", "mean_pool"}:
        adapter = aide_base.TemporalTransformerClipImageAdapter(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(emotion_labels),
            num_prompts=int(text_features.shape[1]),
            num_frames=args.num_frames,
            temporal_num_heads=args.temporal_num_heads,
            temporal_num_layers=args.temporal_num_layers,
            temporal_pool_mode=effective_temporal_pool_mode,
            use_prompt_weight=args.resolved_use_prompt_weight,
            use_class_temperature=args.resolved_use_class_temperature,
            use_class_bias=args.resolved_use_class_bias,
            temporal_module=args.temporal_module,
            use_frame_gate=not args.no_frame_gate,
            use_gem=not args.no_gem,
            use_residual_blend=not args.no_residual_blend,
            gem_init_p=args.gem_init_p,
            adapter_mode=args.adapter_mode,
        )
    elif args.temporal_head == "attention":
        adapter = cremad_base.TemporalClipImageAdapter(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(emotion_labels),
            num_prompts=int(text_features.shape[1]),
            num_frames=args.num_frames,
            use_global_logit_scale=(label_mode == "multi4"),
            use_prompt_weight=args.resolved_use_prompt_weight,
            use_class_temperature=args.resolved_use_class_temperature,
            use_class_bias=args.resolved_use_class_bias,
        )
    elif args.temporal_head == "transformer":
        adapter = aide_base.TemporalTransformerClipImageAdapter(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(emotion_labels),
            num_prompts=int(text_features.shape[1]),
            num_frames=args.num_frames,
            temporal_num_heads=args.temporal_num_heads,
            temporal_num_layers=args.temporal_num_layers,
            temporal_pool_mode=effective_temporal_pool_mode,
            use_prompt_weight=args.resolved_use_prompt_weight,
            use_class_temperature=args.resolved_use_class_temperature,
            use_class_bias=args.resolved_use_class_bias,
            temporal_module="taga",
            adapter_mode=args.adapter_mode,
        )
    else:
        pool_adapter_cls = aide_base.StrongerClipImageAdapter if args.pool_adapter_variant == "stronger" else aide_base.ClipImageAdapter
        adapter = pool_adapter_cls(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(emotion_labels),
            num_prompts=int(text_features.shape[1]),
            use_prompt_weight=args.resolved_use_prompt_weight,
            use_class_temperature=args.resolved_use_class_temperature,
            use_class_bias=args.resolved_use_class_bias,
            adapter_mode=args.adapter_mode,
        )
    if hasattr(adapter, "taga_parameters"):
        temporal_param_count = aide_base.count_parameters(adapter.taga_parameters())
    if hasattr(adapter, "adapter_parameters"):
        adapter_param_count = aide_base.count_parameters(adapter.adapter_parameters())
    if hasattr(adapter, "qcpa_parameters"):
        pch_param_count = aide_base.count_parameters(adapter.qcpa_parameters())
    trainable_param_count = aide_base.count_parameters(adapter.parameters())
    log(f"[PCH] params | head={pch_param_count} | temporal={temporal_param_count} | adapter={adapter_param_count} | trainable={trainable_param_count} | adapter_mode={args.adapter_mode} | temporal_module={args.temporal_module}")
    adapter = cremad_base.train_strict_frozen_clip(
        train_x=train_model_x,
        train_y=train_model_y,
        val_x=val_x,
        val_y=val_y,
        text_features=text_features,
        adapter=adapter,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=effective_use_class_weight,
        label_smoothing=args.label_smoothing,
        loss_type=effective_loss_type,
        focal_gamma=args.focal_gamma,
        select_metric=args.select_metric,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
        use_amp=args.use_amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        class_weights_override=class_weights_override,
        force_loss_type=effective_loss_type if label_mode == "multi4" else None,
        lr_scheduler_mode=args.lr_scheduler_mode,
        scheduler_min_lr=args.scheduler_min_lr,
        train_actor_ids=train_actor_ids,
        use_causal_contrastive=args.use_causal_contrastive,
        ccl_weight=args.ccl_weight,
        ccl_temperature=args.ccl_temperature,
        use_causal_alignment=args.use_causal_alignment,
        cfa_weight=args.cfa_weight,
        use_counterfactual_aug=args.use_counterfactual_aug,
        cda_prob=args.cda_prob,
        cda_n_replace_max=args.cda_n_replace_max,
        use_cda_v2_mixstyle=args.use_cda_v2_mixstyle,
        cda_v2_prob=args.cda_v2_prob,
        cda_v2_kl_weight=args.cda_v2_kl_weight,
        use_ccl_v2_counterfactual=args.use_ccl_v2_counterfactual,
        ccl_v2_weight=args.ccl_v2_weight,
        ccl_v2_temperature=args.ccl_v2_temperature,
        use_cfa_v2_textanchor=args.use_cfa_v2_textanchor,
        cfa_v2_weight=args.cfa_v2_weight,
        cfa_v2_anchor_weight=args.cfa_v2_anchor_weight,
        cfa_v2_ema_momentum=args.cfa_v2_ema_momentum,
    )

    val_pred = cremad_base.predict_emotion_from_features(
        val_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )
    test_pred = cremad_base.predict_emotion_from_features(
        test_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )
    val_true = [emotion_labels[int(item.item())] for item in val_y]
    test_true = [emotion_labels[int(item.item())] for item in test_y]
    val_summary = summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
    test_summary = summarize_predictions_by_mode(test_true, test_pred, emotion_labels, label_mode)

    zero_shot_result = None
    if args.run_zero_shot_eval:
        zero_shot_val_pred = ravdess_base.predict_zeroshot_from_features(
            val_x,
            text_features,
            idx2label,
            args.train_batch_size,
            use_test_ensemble=args.use_test_ensemble,
            ensemble_group_size=args.ensemble_group_size,
        )
        zero_shot_test_pred = ravdess_base.predict_zeroshot_from_features(
            test_x,
            text_features,
            idx2label,
            args.train_batch_size,
            use_test_ensemble=args.use_test_ensemble,
            ensemble_group_size=args.ensemble_group_size,
        )
        zero_shot_result = {
            "val": summarize_predictions_by_mode(val_true, zero_shot_val_pred, emotion_labels, label_mode),
            "test": summarize_predictions_by_mode(test_true, zero_shot_test_pred, emotion_labels, label_mode),
        }

    output_path_value = output_path_override or args.output
    checkpoint_path = Path(checkpoint_output_override) if checkpoint_output_override else (
        Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(output_path_value))
    )
    dataset_summary = dataset_summary_builder(
        args=args,
        split_info=split_info,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_cache_payload=train_cache_payload,
        val_cache_payload=val_cache_payload,
        test_cache_payload=test_cache_payload,
        train_cache_samples=train_cache_samples,
        val_cache_samples=val_cache_samples,
        test_cache_samples=test_cache_samples,
        **dataset_summary_builder_kwargs,
    )

    result = build_result_payload(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode=benchmark_mode,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        resolved_feature_cache_paths=resolved_split_cache_paths,
        output_path_value=output_path_value,
        checkpoint_path=checkpoint_path,
        zero_shot_result=zero_shot_result,
        extra_config={
            "loss_type": effective_loss_type,
            "clip_mode": effective_clip_mode,
            "training_seed": args.training_seed,
            "multi4_oversample_labels": oversample_labels,
            "multi4_oversample_min_count": args.multi4_oversample_min_count,
            "multi4_oversample_mode": args.multi4_oversample_mode,
            "multi4_oversample_noise_std": args.multi4_oversample_noise_std,
            "multi4_mixup_alpha": args.multi4_mixup_alpha,
            "lr_scheduler_mode": args.lr_scheduler_mode,
            "scheduler_min_lr": args.scheduler_min_lr,
            "group_kfold_inner_split_mode": "stratified_sample" if args.eval_mode == "group_kfold" else "default",
            "train_actor_count": int(train_actor_ids.unique().numel()),
        },
        extra_payload={"train_augmentation": training_augmentation_summary} if training_augmentation_summary else None,
    )
    write_result_payload(result, output_path_value)
    save_checkpoint_payload(
        checkpoint_path,
        {
            "checkpoint_type": "strict_frozen_clip_adapter",
            "config": result["config"],
            "dataset": result["dataset"],
            "label_map": result["label_map"],
            "metrics": {"val": result["val"], "test": result["test"]},
            "prompt_groups": prompt_groups,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "adapter_state_dict": adapter.state_dict(),
            "text_features": text_features.cpu(),
            "output_path": output_path_value,
        },
    )

    log(f"[DONE] saved YawDD report to: {output_path_value}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    log(f"[DONE] final test metrics: {json.dumps(result['test'], ensure_ascii=False)}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def build_group_based_eval_folds(
    all_samples: List[Dict],
    args,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    groups = [sample["actor_id"] for sample in all_samples]
    actor_count = len(set(groups))
    if actor_count < 2:
        raise RuntimeError("Need at least 2 distinct actors for grouped evaluation")
    if args.eval_mode == "loso":
        splitter = LeaveOneGroupOut()
        benchmark_mode = "loso_subject_independent"
    else:
        if actor_count < args.n_folds:
            raise RuntimeError(f"Need at least {args.n_folds} distinct actors for group_kfold evaluation")
        splitter = GroupKFold(n_splits=args.n_folds)
        benchmark_mode = f"group_kfold{args.n_folds}_subject_independent"

    if args.eval_mode == "group_kfold":
        inner_train_ratio = 0.8
        inner_val_ratio = 0.2
    else:
        total_ratio = args.train_ratio + args.val_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be > 0 for grouped evaluation")
        inner_train_ratio = args.train_ratio / total_ratio
        inner_val_ratio = args.val_ratio / total_ratio

    fold_payloads: List[Dict[str, object]] = []
    for fold_index, (train_val_indices, test_indices) in enumerate(splitter.split(all_samples, groups=groups), start=1):
        train_val_samples = [all_samples[index] for index in train_val_indices]
        held_out_test_samples = [all_samples[index] for index in test_indices]
        if args.eval_mode == "group_kfold":
            inner_splits, inner_split_info = split_samples_stratified_ratio(
                train_val_samples,
                train_ratio=inner_train_ratio,
                val_ratio=inner_val_ratio,
                seed=args.seed + fold_index,
            )
        else:
            inner_splits, inner_split_info = cremad_base.split_cremad_samples_actor_ratio(
                train_val_samples,
                train_ratio=inner_train_ratio,
                val_ratio=inner_val_ratio,
                seed=args.seed,
            )
        fold_splits = {
            "train": inner_splits["train"],
            "val": inner_splits["val"],
            "test": held_out_test_samples,
        }
        fold_split_info = {
            "mode": args.eval_mode,
            "fold_index": fold_index,
            "train_val_split": inner_split_info,
            "held_out_test": {
                "sample_count": len(held_out_test_samples),
                "actor_count": len({sample["actor_id"] for sample in held_out_test_samples}),
                "actors": sorted({sample["actor_id"] for sample in held_out_test_samples}),
                "class_distribution": dict(Counter(sample["label"] for sample in held_out_test_samples)),
            },
        }
        fold_payloads.append(
            {
                "fold_index": fold_index,
                "splits": fold_splits,
                "split_info": fold_split_info,
                "held_out_actors": fold_split_info["held_out_test"]["actors"],
            }
        )
    return fold_payloads, {"benchmark_mode": benchmark_mode, "fold_count": len(fold_payloads)}


def build_sequence_kfold_eval_folds(
    samples: List[Dict],
    args,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    indices = np.arange(len(samples))
    labels = np.array([sample["label"] for sample in samples])
    n_folds = int(getattr(args, "n_folds", 5))
    seed = int(getattr(args, "seed", 42))
    total_ratio = float(getattr(args, "train_ratio", 0.65)) + float(getattr(args, "val_ratio", 0.15))
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be > 0 for sequence_kfold evaluation")
    train_ratio = float(getattr(args, "train_ratio", 0.65)) / total_ratio

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_payloads: List[Dict[str, object]] = []
    for fold_index, (train_val_indices, test_indices) in enumerate(splitter.split(indices, labels), start=1):
        train_val_labels = labels[train_val_indices]
        rng = np.random.RandomState(seed + fold_index)
        train_indices: List[int] = []
        val_indices: List[int] = []
        for class_label in np.unique(train_val_labels):
            class_indices = train_val_indices[train_val_labels == class_label].copy()
            rng.shuffle(class_indices)
            class_train_count = int(round(len(class_indices) * train_ratio))
            class_train_count = max(1, min(class_train_count, len(class_indices) - 1)) if len(class_indices) > 1 else len(class_indices)
            train_indices.extend(class_indices[:class_train_count].tolist())
            val_indices.extend(class_indices[class_train_count:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        train_samples = [samples[index] for index in train_indices]
        val_samples = [samples[index] for index in val_indices]
        test_samples = [samples[index] for index in test_indices.tolist()]
        fold_split_info = {
            "mode": "sequence_kfold",
            "fold_index": fold_index,
            "train": cremad_base.compute_split_summary(train_samples),
            "val": cremad_base.compute_split_summary(val_samples),
            "held_out_test": {
                "sample_count": len(test_samples),
                "actor_count": len({sample["actor_id"] for sample in test_samples}),
                "actors": [],
                "class_distribution": dict(Counter(sample["label"] for sample in test_samples)),
            },
        }
        fold_payloads.append(
            {
                "fold_index": fold_index,
                "splits": {
                    "train": train_samples,
                    "val": val_samples,
                    "test": test_samples,
                },
                "split_info": fold_split_info,
                "held_out_actors": [],
            }
        )

    return fold_payloads, {
        "benchmark_mode": f"sequence_kfold{n_folds}",
        "fold_count": len(fold_payloads),
        "eval_mode": "sequence_kfold",
        "n_folds": n_folds,
        "seed": seed,
        "train_ratio": train_ratio,
        "total_samples": len(samples),
    }


def configure_random_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    configure_random_seeds(args.seed)
    label_mode = args.label_mode if args.label_mode in {"behavior_4", "multi4"} else "binary"
    args.prompt_set = resolve_effective_prompt_set(args, label_mode)
    args.temporal_pool_mode = resolve_effective_temporal_pool_mode(args, label_mode)
    args.output, args.log_file, args.checkpoint_output = resolve_eval_output_paths(args, args.eval_mode, label_mode)
    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    resolved_log_file = init_log_file(args.log_file)
    atexit.register(close_log_file)
    if resolved_log_file:
        log(f"[LOG] writing log file to: {resolved_log_file}")

    emotion_labels, _, _ = resolve_label_space(label_mode)
    cremad_base.EMOTION_LABELS = list(emotion_labels)
    ravdess_base.EMOTION_LABELS = list(emotion_labels)
    prompt_groups = build_class_prompts(label_mode, args.prompt_template, args.prompt_set)
    runner = run_single_training_evaluation if args.baseline_mode == "none" else run_baseline_training_evaluation

    if args.baseline_mode != "none" and (args.extract_only or args.merge_shards):
        raise ValueError("Baseline modes do not support --extract_only or --merge_shards")

    if args.eval_mode in {"loso", "group_kfold", "sequence_kfold"}:
        if args.extract_only or args.merge_shards:
            raise ValueError("Grouped evaluation modes do not support --extract_only or --merge_shards")
        if not args.all_face_image:
            raise ValueError(f"--all_face_image is required when --eval_mode={args.eval_mode}")
        all_samples, all_dataset_diagnostics = collect_all_face_sequence_samples(args.all_face_image, label_mode)
        if len(all_samples) < 10:
            raise RuntimeError(f"Too few all-face-image samples: {len(all_samples)}")
        log(
            f"[DATA] grouped eval source: samples={len(all_samples)} subjects={all_dataset_diagnostics['subject_count']} "
            f"class_distribution={all_dataset_diagnostics['class_distribution']}"
        )
        if args.eval_mode == "sequence_kfold":
            fold_payloads, fold_meta = build_sequence_kfold_eval_folds(all_samples, args)
        else:
            fold_payloads, fold_meta = build_group_based_eval_folds(all_samples, args)
        fold_results: List[Dict[str, object]] = []
        accuracy_values: List[float] = []
        precision_values: List[float] = []
        recall_values: List[float] = []
        f1_values: List[float] = []
        for fold_payload in fold_payloads:
            fold_index = fold_payload["fold_index"]
            held_out_actors = fold_payload["held_out_actors"]
            log(f"[EVAL] fold {fold_index}/{fold_meta['fold_count']} held_out_actors={held_out_actors}")
            fold_output = add_suffix_to_path(args.output, f".fold{fold_index:02d}")
            fold_checkpoint = add_suffix_to_path(args.checkpoint_output, f".fold{fold_index:02d}") if args.checkpoint_output else None
            fold_result = runner(
                args=args,
                label_mode=label_mode,
                emotion_labels=emotion_labels,
                prompt_groups=prompt_groups,
                splits=fold_payload["splits"],
                split_info=fold_payload["split_info"],
                benchmark_mode=fold_meta["benchmark_mode"],
                dataset_summary_builder=build_all_face_dataset_summary,
                dataset_summary_builder_kwargs={"dataset_diagnostics": all_dataset_diagnostics},
                output_path_override=fold_output,
                checkpoint_output_override=fold_checkpoint,
            )
            fold_metrics = dict(fold_result["test_metrics"])
            fold_metrics["average_mode"] = "weighted" if label_mode == "multi4" else "binary"
            accuracy_values.append(fold_metrics["accuracy"])
            precision_values.append(fold_metrics["precision"])
            recall_values.append(fold_metrics["recall"])
            f1_values.append(fold_metrics["f1"])
            fold_record = {
                "fold_index": fold_index,
                "held_out_actors": held_out_actors,
                "metrics": fold_metrics,
                "result_path": fold_output,
                "checkpoint_path": fold_checkpoint,
                "held_out_test_distribution": fold_payload["split_info"]["held_out_test"]["class_distribution"],
            }
            fold_record["test_metrics"] = fold_result["test"]
            fold_record["class_distribution"] = fold_result["class_distribution"]
            if label_mode == "multi4":
                fold_record["per_class_metrics"] = fold_result["per_class_metrics"]
                fold_record["confusion_matrix"] = fold_result["confusion_matrix"]
            fold_results.append(fold_record)
            log(
                f"[EVAL] fold {fold_index} metrics: "
                f"accuracy={fold_metrics['accuracy']:.6f} precision={fold_metrics['precision']:.6f} "
                f"recall={fold_metrics['recall']:.6f} f1={fold_metrics['f1']:.6f}"
            )

        aggregate = {
            "accuracy": compute_output_stats(accuracy_values),
            "precision": compute_output_stats(precision_values),
            "recall": compute_output_stats(recall_values),
            "f1": compute_output_stats(f1_values),
        }
        summary_payload = {
            "label_mode": label_mode,
            "class_names": resolve_class_names(label_mode, emotion_labels),
            "config": {
                "eval_mode": args.eval_mode,
                "label_mode": label_mode,
                "class_names": resolve_class_names(label_mode, emotion_labels),
                "all_face_image": str(Path(args.all_face_image).resolve()),
                "model_id": args.model_id,
                "n_folds": args.n_folds,
                "seed": args.seed,
                "epochs": args.epochs,
                "extract_batch_size": args.extract_batch_size,
                "train_batch_size": args.train_batch_size,
                "lr": args.lr,
                "label_smoothing": args.label_smoothing,
                "ensemble_group_size": args.ensemble_group_size,
                "num_frames": args.num_frames,
                "frame_sampling_mode": args.frame_sampling_mode,
                "temporal_head": args.temporal_head,
                "temporal_num_heads": args.temporal_num_heads,
                "temporal_num_layers": args.temporal_num_layers,
                "temporal_pool_mode": args.temporal_pool_mode,
                "prompt_set": args.prompt_set,
                "baseline_mode": args.baseline_mode,
            },
            "dataset": all_dataset_diagnostics,
            "folds": fold_results,
            "aggregate": aggregate,
        }
        summary_output_path = Path(args.output)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        log(f"[DONE] saved cross-eval summary to: {summary_output_path}")
        log(
            f"[DONE] {args.eval_mode} aggregate: "
            f"accuracy={aggregate['accuracy']['mean']:.6f}+-{aggregate['accuracy']['std']:.6f} "
            f"precision={aggregate['precision']['mean']:.6f}+-{aggregate['precision']['std']:.6f} "
            f"recall={aggregate['recall']['mean']:.6f}+-{aggregate['recall']['std']:.6f} "
            f"f1={aggregate['f1']['mean']:.6f}+-{aggregate['f1']['std']:.6f}"
        )
        print(json.dumps({"aggregate": aggregate}, ensure_ascii=False, indent=2), flush=True)
        return

    if args.eval_mode == "random":
        if args.all_face_image:
            samples, dataset_diagnostics = collect_all_face_sequence_samples(args.all_face_image, label_mode)
            if len(samples) < 10:
                raise RuntimeError(f"Too few all-face samples for random evaluation: {len(samples)}")
            log(
                f"[DATA] all-face random eval: samples={len(samples)} subjects={dataset_diagnostics['subject_count']} "
                f"class_distribution={dataset_diagnostics['class_distribution']}"
            )
            splits, split_info = split_yawdd_samples_random_stratified(
                samples,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            benchmark_mode = "all_face_random_stratified"
            dataset_summary_builder = build_all_face_dataset_summary
            dataset_summary_builder_kwargs = {"dataset_diagnostics": dataset_diagnostics}
        else:
            samples, dataset_diagnostics = collect_yawdd_samples(
                yawdd_root=args.yawdd_root,
                label_mode=label_mode,
                include_dash=args.include_dash,
                max_sequences=args.max_sequences,
            )
            if len(samples) < 10:
                raise RuntimeError(f"Too few valid YawDD samples: {len(samples)}")
            splits, split_info = split_yawdd_samples_random_stratified(
                samples,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            benchmark_mode = "random_stratified"
            dataset_summary_builder = build_standard_yawdd_dataset_summary
            dataset_summary_builder_kwargs = {"dataset_diagnostics": dataset_diagnostics}
        return runner(
            args=args,
            label_mode=label_mode,
            emotion_labels=emotion_labels,
            prompt_groups=prompt_groups,
            splits=splits,
            split_info=split_info,
            benchmark_mode=benchmark_mode,
            dataset_summary_builder=dataset_summary_builder,
            dataset_summary_builder_kwargs=dataset_summary_builder_kwargs,
        )

    if args.eval_mode == "fixed" and args.all_face_image and not args.external_face_root:
        if args.cv_mode != "split":
            raise ValueError("--all_face_image fixed evaluation currently requires --cv_mode split")
        train_pool_samples, fixed_test_samples, dataset_diagnostics = collect_fixed_all_face_sequence_samples(
            all_face_image=args.all_face_image,
            label_mode=label_mode,
        )
        if len(train_pool_samples) < 10 or len(fixed_test_samples) < 2:
            raise RuntimeError(
                f"Too few all-face fixed-test samples: train_pool={len(train_pool_samples)} test={len(fixed_test_samples)}"
            )
        log(
            f"[DATA] all-face fixed sequences: train_pool={len(train_pool_samples)} fixed_test={len(fixed_test_samples)} "
            f"train_subjects={dataset_diagnostics['train_pool_subjects']} test_subjects={dataset_diagnostics['fixed_test_subjects']} "
            f"train_distribution={dataset_diagnostics['train_pool_distribution']} test_distribution={dataset_diagnostics['fixed_test_distribution']}"
        )
        total_ratio = args.train_ratio + args.val_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be > 0 when using all-face fixed sequences")
        inner_train_ratio = args.train_ratio / total_ratio
        inner_val_ratio = args.val_ratio / total_ratio
        inner_splits, inner_split_info = cremad_base.split_cremad_samples_actor_ratio(
            train_pool_samples,
            train_ratio=inner_train_ratio,
            val_ratio=inner_val_ratio,
            seed=args.seed,
        )
        splits = {
            "train": inner_splits["train"],
            "val": inner_splits["val"],
            "test": fixed_test_samples,
        }
        split_info = {
            "mode": "all_face_sequences_fixed_test",
            "inner_train_val_split": inner_split_info,
            "fixed_test": {
                "sample_count": len(fixed_test_samples),
                "subjects": len({sample["subject_id"] for sample in fixed_test_samples}),
                "class_distribution": dict(Counter(sample["label"] for sample in fixed_test_samples)),
            },
        }
        benchmark_mode = "all_face_sequences_fixed_test"
        dataset_summary_builder = build_fixed_all_face_dataset_summary
        dataset_summary_builder_kwargs = {"dataset_diagnostics": dataset_diagnostics}
    elif args.external_face_root:
        if args.cv_mode != "split":
            raise ValueError("--external_face_root currently requires --cv_mode split")
        train_pool_samples, fixed_test_samples, dataset_diagnostics = collect_preprocessed_face_sequence_samples(
            preprocessed_root=args.external_face_root,
            label_mode=label_mode,
        )
        if len(train_pool_samples) < 10 or len(fixed_test_samples) < 2:
            raise RuntimeError(
                f"Too few external face-sequence samples: train_pool={len(train_pool_samples)} test={len(fixed_test_samples)}"
            )
        log(
            f"[DATA] external face sequences: train_pool={len(train_pool_samples)} fixed_test={len(fixed_test_samples)} "
            f"train_subjects={dataset_diagnostics['train_pool_subjects']} test_subjects={dataset_diagnostics['fixed_test_subjects']} "
            f"train_distribution={dataset_diagnostics['train_pool_distribution']} test_distribution={dataset_diagnostics['fixed_test_distribution']}"
        )
        total_ratio = args.train_ratio + args.val_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be > 0 when using external face sequences")
        inner_train_ratio = args.train_ratio / total_ratio
        inner_val_ratio = args.val_ratio / total_ratio
        inner_splits, inner_split_info = cremad_base.split_cremad_samples_actor_ratio(
            train_pool_samples,
            train_ratio=inner_train_ratio,
            val_ratio=inner_val_ratio,
            seed=args.seed,
        )
        splits = {
            "train": inner_splits["train"],
            "val": inner_splits["val"],
            "test": fixed_test_samples,
        }
        split_info = {
            "mode": "external_face_sequences_fixed_test",
            "inner_train_val_split": inner_split_info,
            "fixed_test": {
                "sample_count": len(fixed_test_samples),
                "subjects": len({sample["subject_id"] for sample in fixed_test_samples}),
                "class_distribution": dict(Counter(sample["label"] for sample in fixed_test_samples)),
            },
        }
        benchmark_mode = "external_face_sequences_fixed_test"
        dataset_summary_builder = build_external_fixed_dataset_summary
        dataset_summary_builder_kwargs = {
            "emotion_labels": emotion_labels,
            "dataset_diagnostics": dataset_diagnostics,
        }
    else:
        samples, dataset_diagnostics = collect_yawdd_samples(
            yawdd_root=args.yawdd_root,
            label_mode=label_mode,
            include_dash=args.include_dash,
            max_sequences=args.max_sequences,
        )
        if len(samples) < 10:
            raise RuntimeError(f"Too few valid YawDD samples: {len(samples)}")
        log(
            f"[DATA] samples={len(samples)} subjects={dataset_diagnostics['subject_count']} "
            f"class_distribution={dataset_diagnostics['class_distribution']} views={dataset_diagnostics['view_distribution']}"
        )
        if not args.include_dash:
            log("[DATA] Dash subset excluded by default because its filenames do not expose stable supervision labels.")

        if args.cv_mode == "5fold":
            splits, split_info = cremad_base.split_cremad_samples_5fold(samples, fold_idx=args.fold_idx, seed=args.seed)
            benchmark_mode = "5fold_subject_independent"
        else:
            if args.split_mode == "random_stratified":
                splits, split_info = split_yawdd_samples_random_stratified(
                    samples,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
                benchmark_mode = "split_random_stratified"
            else:
                splits, split_info = cremad_base.split_cremad_samples_actor_ratio(
                    samples,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
                benchmark_mode = "split_subject_independent"
        dataset_summary_builder = build_standard_yawdd_dataset_summary
        dataset_summary_builder_kwargs = {"dataset_diagnostics": dataset_diagnostics}
    result = runner(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        splits=splits,
        split_info=split_info,
        benchmark_mode=benchmark_mode,
        dataset_summary_builder=dataset_summary_builder,
        dataset_summary_builder_kwargs=dataset_summary_builder_kwargs,
    )
    return result


if __name__ == "__main__":
    main()
