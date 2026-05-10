import argparse
import atexit
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import clip_aide_emotion_train as aide_base
import clip_ravdess_emotion_train as ravdess_base

EMOTION_LABELS = ["notdrowsy", "drowsy"]
EMOTION_DISPLAY_MAP = {
    "notdrowsy": "alert and not drowsy",
    "drowsy": "drowsy and fatigued",
}
DDD_PROMPT_GROUPS = {
    "notdrowsy": [
        "The driver looks alert, awake, and not drowsy.",
        "The visible face appears attentive and not fatigued.",
        "This frame shows a driver who is not drowsy.",
        "The person looks awake, focused, and not sleepy.",
        "The facial cues suggest an alert driver rather than a drowsy one.",
    ],
    "drowsy": [
        "The driver looks drowsy, sleepy, and fatigued.",
        "The visible face appears tired and consistent with drowsiness.",
        "This frame shows a driver who is drowsy.",
        "The person looks sleepy, fatigued, and less alert.",
        "The facial cues suggest a drowsy driver rather than an alert one.",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DDD_ROOT = os.environ.get("DDD_ROOT", str(PROJECT_ROOT / "data" / "DDD" / "train_data"))
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "ddd" / "clip_ddd_results.json")
LOG_FILE_HANDLE = None
DDD_FILENAME_RE = re.compile(
    r"^(?P<subject_id>\d+)_(?P<eyewear>[^_]+)_(?P<scenario>.+)_(?P<frame_index>\d+)_(?P<label>drowsy|notdrowsy)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def log(message: str) -> None:
    global LOG_FILE_HANDLE

    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
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


def summarize_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, EMOTION_LABELS), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, EMOTION_LABELS),
        "prediction_distribution": {label: int(Counter(y_pred).get(label, 0)) for label in EMOTION_LABELS},
    }


def parse_ddd_file_name(file_name: str) -> Optional[Dict[str, str]]:
    match = DDD_FILENAME_RE.match(file_name)
    if match is None:
        return None
    meta = match.groupdict()
    meta["label"] = meta["label"].lower()
    meta["clip_id"] = f"{meta['subject_id']}_{meta['eyewear']}_{meta['scenario']}"
    meta["sequence_id"] = Path(file_name).stem
    return meta


def is_readable_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def collect_ddd_samples(ddd_root: str, max_sequences: int = 0) -> Tuple[List[Dict], Dict[str, object]]:
    root = Path(ddd_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"DDD root not found: {root}")

    samples: List[Dict] = []
    invalid_name_examples: List[str] = []
    mismatched_label_examples: List[str] = []
    unreadable_image_examples: List[str] = []
    unreadable_image_count = 0

    for label in EMOTION_LABELS:
        class_dir = root / label
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            meta = parse_ddd_file_name(path.name)
            if meta is None:
                if len(invalid_name_examples) < 10:
                    invalid_name_examples.append(path.name)
                continue
            if meta["label"] != label:
                if len(mismatched_label_examples) < 10:
                    mismatched_label_examples.append(path.name)
                continue
            if not is_readable_image(path):
                unreadable_image_count += 1
                if len(unreadable_image_examples) < 10:
                    unreadable_image_examples.append(path.name)
                continue
            samples.append(
                {
                    "sequence_id": meta["sequence_id"],
                    "frame_path": str(path),
                    "frame_paths": [str(path)],
                    "label": meta["label"],
                    "subject_id": meta["subject_id"],
                    "actor_id": meta["subject_id"],
                    "eyewear": meta["eyewear"],
                    "scenario": meta["scenario"],
                    "clip_id": meta["clip_id"],
                    "ext": path.suffix.lower(),
                }
            )

    if max_sequences > 0:
        samples = samples[:max_sequences]

    diagnostics = {
        "total_valid_samples": len(samples),
        "subject_count": len({sample["subject_id"] for sample in samples}),
        "clip_count": len({sample["clip_id"] for sample in samples}),
        "class_distribution": dict(Counter(sample["label"] for sample in samples)),
        "scenario_distribution": dict(Counter(sample["scenario"] for sample in samples)),
        "invalid_name_examples": invalid_name_examples,
        "mismatched_label_examples": mismatched_label_examples,
        "unreadable_image_count": unreadable_image_count,
        "unreadable_image_examples": unreadable_image_examples,
    }
    return samples, diagnostics


def compute_group_split_counts(total_groups: int, ratios: List[float]) -> List[int]:
    raw_counts = [total_groups * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total_groups - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: (raw_counts[idx] - counts[idx]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def split_samples_group_disjoint(
    samples: List[Dict],
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    grouped_by_label: Dict[str, Dict[str, List[Dict]]] = {label: {} for label in EMOTION_LABELS}
    for sample in samples:
        grouped_by_label[sample["label"]].setdefault(str(sample[group_key]), []).append(sample)

    rng = random.Random(seed)
    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []
    assigned_groups = {"train": [], "val": [], "test": []}
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    for label in EMOTION_LABELS:
        group_ids = sorted(grouped_by_label[label])
        rng.shuffle(group_ids)
        n_train, n_val, n_test = compute_group_split_counts(len(group_ids), [train_ratio, val_ratio, test_ratio])
        train_group_ids = group_ids[:n_train]
        val_group_ids = group_ids[n_train:n_train + n_val]
        test_group_ids = group_ids[n_train + n_val:n_train + n_val + n_test]
        train.extend(sample for group_id in train_group_ids for sample in grouped_by_label[label][group_id])
        val.extend(sample for group_id in val_group_ids for sample in grouped_by_label[label][group_id])
        test.extend(sample for group_id in test_group_ids for sample in grouped_by_label[label][group_id])
        assigned_groups["train"].extend(train_group_ids)
        assigned_groups["val"].extend(val_group_ids)
        assigned_groups["test"].extend(test_group_ids)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    split_info = {
        "group_key": group_key,
        "train_group_count": len(set(assigned_groups["train"])),
        "val_group_count": len(set(assigned_groups["val"])),
        "test_group_count": len(set(assigned_groups["test"])),
        "train_groups": sorted(set(assigned_groups["train"]))[:50],
        "val_groups": sorted(set(assigned_groups["val"]))[:50],
        "test_groups": sorted(set(assigned_groups["test"]))[:50],
        "train_class_distribution": dict(Counter(sample["label"] for sample in train)),
        "val_class_distribution": dict(Counter(sample["label"] for sample in val)),
        "test_class_distribution": dict(Counter(sample["label"] for sample in test)),
    }
    return {"train": train, "val": val, "test": test}, split_info


def build_class_prompts(prompt_template: str, prompt_set: str) -> List[List[str]]:
    if prompt_set == "ddd_binary_facial_cues":
        return [list(DDD_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]
    if prompt_set == "single":
        return [[prompt_template.replace("<LABEL>", EMOTION_DISPLAY_MAP[label])] for label in EMOTION_LABELS]
    templates = [item.strip() for item in prompt_set.split("||") if item.strip()]
    if not templates:
        templates = [prompt_template]
    return [[template.replace("<LABEL>", EMOTION_DISPLAY_MAP[label]) for template in templates] for label in EMOTION_LABELS]


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone DDD CLIP drowsiness training with strict frozen CLIP adapter")
    parser.add_argument("--ddd_root", default=DEFAULT_DDD_ROOT)
    parser.add_argument("--split_mode", choices=["subject_id", "clip_id"], default="clip_id")
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="The driver looks <LABEL>.")
    parser.add_argument("--prompt_set", default="ddd_binary_facial_cues")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--feature_layout", choices=["pooled", "sequence"], default="pooled")
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", action="store_true")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--run_zero_shot_eval", action="store_true")
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"

    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    resolved_log_file = init_log_file(args.log_file)
    atexit.register(close_log_file)
    if resolved_log_file:
        log(f"[LOG] writing log file to: {resolved_log_file}")

    samples, dataset_diagnostics = collect_ddd_samples(args.ddd_root, max_sequences=args.max_sequences)
    if len(samples) < 10:
        raise RuntimeError(f"Too few valid DDD samples: {len(samples)}")
    log(
        f"[DATA] samples={len(samples)} subjects={dataset_diagnostics['subject_count']} clips={dataset_diagnostics['clip_count']} "
        f"class_distribution={dataset_diagnostics['class_distribution']}"
    )
    if dataset_diagnostics["unreadable_image_count"] > 0:
        log(
            f"[WARN] skipped unreadable images: count={dataset_diagnostics['unreadable_image_count']} "
            f"examples={dataset_diagnostics['unreadable_image_examples']}"
        )
    if dataset_diagnostics["subject_count"] < 5:
        log("[WARN] DDD has very few subject ids; subject-disjoint splitting is possible but statistically fragile.")

    splits, split_info = split_samples_group_disjoint(
        samples,
        group_key=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    log(f"[DATA] split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    random.seed(args.seed)
    aide_base.EMOTION_LABELS = list(EMOTION_LABELS)
    ravdess_base.EMOTION_LABELS = list(EMOTION_LABELS)

    if args.clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(args.model_id)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False)
        except Exception:
            processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    log(f"[INFO] model loaded: {args.model_id} on {args.device}")

    prompt_groups = build_class_prompts(args.prompt_template, args.prompt_set)
    text_features = aide_base.extract_text_features(prompt_groups, processor, model, args.device)

    train_x = aide_base.extract_image_features(
        train_samples,
        processor,
        model,
        args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        tag="train",
        feature_layout=args.feature_layout,
    )
    val_x = aide_base.extract_image_features(
        val_samples,
        processor,
        model,
        args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        tag="val",
        feature_layout=args.feature_layout,
    )
    test_x = aide_base.extract_image_features(
        test_samples,
        processor,
        model,
        args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        tag="test",
        feature_layout=args.feature_layout,
    )

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
    train_y = torch.tensor([label2idx[sample["label"]] for sample in train_samples], dtype=torch.long)
    val_y = torch.tensor([label2idx[sample["label"]] for sample in val_samples], dtype=torch.long)
    test_y = torch.tensor([label2idx[sample["label"]] for sample in test_samples], dtype=torch.long)

    feature_dim = int(train_x.shape[-1])
    adapter = aide_base.ClipImageAdapter(
        dim=feature_dim,
        device=args.device,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        num_classes=len(EMOTION_LABELS),
        num_prompts=int(text_features.shape[1]),
        use_prompt_weight=False,
        use_class_temperature=False,
        use_class_bias=False,
    )
    adapter = aide_base.train_strict_frozen_clip(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        text_features=text_features,
        adapter=adapter,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=args.use_class_weight,
        label_smoothing=args.label_smoothing,
        select_metric=args.select_metric,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )

    val_pred = aide_base.predict_emotion_from_features(
        val_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )
    test_pred = aide_base.predict_emotion_from_features(
        test_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )
    val_true = [EMOTION_LABELS[int(item.item())] for item in val_y]
    test_true = [EMOTION_LABELS[int(item.item())] for item in test_y]
    val_summary = summarize_predictions(val_true, val_pred)
    test_summary = summarize_predictions(test_true, test_pred)

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
            "val": summarize_predictions(val_true, zero_shot_val_pred),
            "test": summarize_predictions(test_true, zero_shot_test_pred),
        }

    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else aide_base.default_checkpoint_path(Path(args.output))
    result = {
        "config": {
            "method": "clip_supervised_text_image_drowsiness",
            "execution_mode": "strict_frozen_clip_adapter",
            "dataset": "DDD",
            "task": "drowsiness",
            "ddd_root": str(Path(args.ddd_root).resolve()),
            "split_mode": args.split_mode,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "model_id": args.model_id,
            "prompt_template": args.prompt_template,
            "prompt_set": args.prompt_set,
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
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "run_zero_shot_eval": args.run_zero_shot_eval,
            "checkpoint_output": str(checkpoint_path),
            "log_file": resolved_log_file,
            "max_sequences": args.max_sequences,
        },
        "dataset": {
            "total": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "subject_count": dataset_diagnostics["subject_count"],
            "clip_count": dataset_diagnostics["clip_count"],
            "class_distribution_total": dataset_diagnostics["class_distribution"],
            "scenario_distribution_total": dataset_diagnostics["scenario_distribution"],
            "split_summary": split_info,
            "invalid_name_examples": dataset_diagnostics["invalid_name_examples"],
            "mismatched_label_examples": dataset_diagnostics["mismatched_label_examples"],
            "unreadable_image_count": dataset_diagnostics["unreadable_image_count"],
            "unreadable_image_examples": dataset_diagnostics["unreadable_image_examples"],
        },
        "label_map": {label: label for label in EMOTION_LABELS},
        "prompt_groups": prompt_groups,
        "val": val_summary,
        "test": test_summary,
        "zero_shot": zero_shot_result,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
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
            "output_path": str(output_path),
        },
        checkpoint_path,
    )

    log(f"[DONE] saved DDD report to: {output_path}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    log(f"[DONE] final test metrics: {json.dumps(result['test'], ensure_ascii=False)}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()