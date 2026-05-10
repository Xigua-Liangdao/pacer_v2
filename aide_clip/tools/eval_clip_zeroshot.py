#!/usr/bin/env python3
"""
Pure CLIP zero-shot evaluation on AIDE emotion dataset.
Uses the same data split as training to ensure comparability.

Usage:
  python tools/eval_clip_zeroshot.py \
    --reference_result_json results/adapter_sweep/h2048_d02.json \
    --device cuda:0 \
    --output results/zeroshot/clip_zeroshot.json
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image

# ---- Reuse data utilities from training script ----
EMOTION_LABELS = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
EMOTION_NORMALIZE_MAP = {
    "anxiety": "Anxiety",
    "peace": "Peace",
    "weariness": "Weariness",
    "happiness": "Happiness",
    "anger": "Anger",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AIDE_ROOT = os.environ.get("AIDE_ROOT", "/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset")
DEFAULT_ANNOTATION_ROOT = os.environ.get("AIDE_ANNOTATION_ROOT", os.path.join(DEFAULT_AIDE_ROOT, "annotation"))


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def normalize_emotion_label(label: str) -> str:
    key = str(label).strip().lower()
    return EMOTION_NORMALIZE_MAP.get(key, str(label).strip())


def sorted_frame_paths(folder: str) -> List[str]:
    frames = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    def frame_key(name: str):
        stem = os.path.splitext(name)[0]
        return (int(stem) if stem.isdigit() else 10**9, name)
    frames.sort(key=frame_key)
    return [os.path.join(folder, f) for f in frames]


def middle_frame(frame_paths: List[str]) -> str:
    if not frame_paths:
        raise ValueError("No frame paths provided")
    return frame_paths[len(frame_paths) // 2]


def collect_samples(aide_root: str, annotation_root: str, max_sequences: int = 0) -> List[Dict]:
    candidate_set = set(EMOTION_LABELS)
    seq_ids = [d for d in os.listdir(aide_root) if os.path.isdir(os.path.join(aide_root, d)) and d.isdigit()]
    seq_ids.sort()
    samples = []
    for seq_id in seq_ids:
        anno_path = os.path.join(annotation_root, f"{seq_id}.json")
        incar_dir = os.path.join(aide_root, seq_id, "incarframes")
        if not os.path.exists(anno_path) or not os.path.isdir(incar_dir):
            continue
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
        label = normalize_emotion_label(anno.get("emotion_label", "Unknown"))
        if label not in candidate_set:
            continue
        frames = sorted_frame_paths(incar_dir)
        if not frames:
            continue
        samples.append({
            "sequence_id": seq_id,
            "label": label,
            "frame_path": middle_frame(frames),
            "frame_paths": frames,
        })
    if max_sequences > 0:
        samples = samples[:max_sequences]
    return samples


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int):
    label_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        label_groups.setdefault(sample["label"], []).append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, group in label_groups.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def weighted_f1(y_true, y_pred, labels):
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    wf1 = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        wf1 += (support.get(label, 0) / total) * f1
    return wf1


def confusion_matrix(y_true, y_pred, labels):
    mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def evaluate_split(y_true, y_pred):
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, EMOTION_LABELS), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, EMOTION_LABELS),
    }


def build_class_prompts(prompt_template: str, prompt_set: str) -> List[List[str]]:
    if prompt_set == "single":
        templates = [prompt_template]
    elif prompt_set == "driving_7":
        templates = [
            "Driver is <LABEL>.",
            "The driver's emotional state is <LABEL>.",
            "In this cabin, the driver's emotion is <LABEL>.",
            "This driving clip shows a <LABEL> driver.",
            "The person behind the wheel appears <LABEL>.",
            "Emotion label for the driver: <LABEL>.",
            "Current driver affect: <LABEL>.",
        ]
    elif prompt_set == "default_5":
        templates = [
            "Driver is <LABEL>.",
            "The driver's emotion is <LABEL>.",
            "Emotion state: <LABEL>.",
            "The person appears <LABEL>.",
            "This driver feels <LABEL>.",
        ]
    else:
        custom = [x.strip() for x in prompt_set.split("||") if x.strip()]
        templates = custom if custom else [prompt_template]
    return [[tpl.replace("<LABEL>", label) for tpl in templates] for label in EMOTION_LABELS]


# ---- Zero-shot evaluation ----

def zeroshot_predict(samples, processor, model, text_features, device, batch_size):
    """
    Pure zero-shot: image_features @ mean_text_features -> argmax.
    No adapter, no learned temperature, no bias.
    """
    import torch

    preds = []
    confs = []
    # text_features: [num_classes, num_prompts, dim] -> average over prompts
    mean_text = text_features.mean(dim=1)  # [num_classes, dim]
    mean_text = mean_text / mean_text.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    mean_text = mean_text.to(device)

    total = len(samples)
    num_batches = math.ceil(total / batch_size)
    log(f"[ZEROSHOT] predicting {total} samples in {num_batches} batches")

    for batch_idx, start in enumerate(range(0, total, batch_size), 1):
        batch = samples[start:start + batch_size]
        images = [Image.open(s["frame_path"]).convert("RGB") for s in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            image_features = image_features.float()

            # cosine similarity -> logits (scale by CLIP's logit_scale)
            logit_scale = model.logit_scale.exp().clamp(max=100.0)
            sim = logit_scale * (image_features @ mean_text.T)  # [B, num_classes]

            probs = torch.softmax(sim, dim=-1)
            pred_idx = sim.argmax(dim=-1).cpu().tolist()
            pred_conf = probs.max(dim=-1).values.cpu().tolist()

        preds.extend([EMOTION_LABELS[i] for i in pred_idx])
        confs.extend(pred_conf)

        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches:
            log(f"[ZEROSHOT] batch {batch_idx}/{num_batches}")

    return preds, confs


def extract_text_features(prompt_groups, processor, model, device):
    import torch

    class_feats = []
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_feats.append(text_features)
    return torch.stack(class_feats, dim=0).float().detach().cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Pure CLIP zero-shot evaluation on AIDE emotion")
    parser.add_argument("--reference_result_json", default=None,
                        help="Reference result JSON to copy split config from (ensures same train/val/test split)")
    parser.add_argument("--aide_root", default=DEFAULT_AIDE_ROOT)
    parser.add_argument("--annotation_root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="offline_only")
    parser.add_argument("--prompt_template", default="Driver is <LABEL>.")
    parser.add_argument("--prompt_set", default="driving_7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "zeroshot" / "clip_zeroshot.json"))
    parser.add_argument("--checkpoint_output", default=None,
                        help="Path for checkpoint. Default: <output>.ckpt.pt")
    return parser.parse_args()


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    args = parse_args()
    random.seed(args.seed)

    # If reference JSON provided, copy split config to ensure identical split
    if args.reference_result_json:
        ref = json.load(open(args.reference_result_json))
        ref_cfg = ref.get("config", {})
        args.seed = ref_cfg.get("seed", args.seed)
        args.train_ratio = ref_cfg.get("split", {}).get("train", args.train_ratio)
        args.val_ratio = ref_cfg.get("split", {}).get("val", args.val_ratio)
        args.max_sequences = ref_cfg.get("max_sequences", args.max_sequences)
        args.model_id = ref_cfg.get("model_id", args.model_id)
        args.prompt_template = ref_cfg.get("prompt_template", args.prompt_template)
        args.prompt_set = ref_cfg.get("prompt_set", args.prompt_set)
        random.seed(args.seed)
        log(f"[INFO] loaded split config from {args.reference_result_json}")
        log(f"[INFO] seed={args.seed}, train={args.train_ratio}, val={args.val_ratio}")

    # Collect and split data
    samples = collect_samples(args.aide_root, args.annotation_root, args.max_sequences)
    log(f"[INFO] total samples: {len(samples)}")
    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    val_samples = splits["val"]
    test_samples = splits["test"]
    log(f"[INFO] split -> train: {len(splits['train'])}, val: {len(val_samples)}, test: {len(test_samples)}")

    # Load CLIP model (no training, pure zero-shot)
    if args.clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(args.model_id)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False)
        except Exception:
            log(f"[INFO] falling back to local CLIP cache for: {args.model_id}")
            processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)
    else:
        log(f"[INFO] loading CLIP from local cache only: {args.model_id}")
        processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    model.eval()
    log(f"[INFO] CLIP model loaded: {args.model_id} on {args.device}")

    # Build text features
    prompt_groups = build_class_prompts(args.prompt_template, args.prompt_set)
    text_features = extract_text_features(prompt_groups, processor, model, args.device)
    log(f"[INFO] text features: {text_features.shape}")  # [num_classes, num_prompts, dim]

    # Zero-shot predict
    val_pred, val_conf = zeroshot_predict(val_samples, processor, model, text_features, args.device, args.batch_size)
    test_pred, test_conf = zeroshot_predict(test_samples, processor, model, text_features, args.device, args.batch_size)

    val_true = [s["label"] for s in val_samples]
    test_true = [s["label"] for s in test_samples]

    val_metrics = evaluate_split(val_true, val_pred)
    test_metrics = evaluate_split(test_true, test_pred)

    log(f"[RESULT] val  accuracy={val_metrics['accuracy']:.4f}  wf1={val_metrics['weighted_f1']:.4f}")
    log(f"[RESULT] test accuracy={test_metrics['accuracy']:.4f}  wf1={test_metrics['weighted_f1']:.4f}")

    # Save result JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else output_path.with_suffix(".ckpt.pt")

    result = {
        "config": {
            "method": "clip_zeroshot",
            "task": "emotion",
            "split": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": round(1 - args.train_ratio - args.val_ratio, 6),
            },
            "model_id": args.model_id,
            "clip_mode": args.clip_mode,
            "prompt_template": args.prompt_template,
            "prompt_set": args.prompt_set,
            "epochs": 0,
            "batch_size": args.batch_size,
            "lr": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "num_frames": 1,
            "adapter_hidden_dim": 0,
            "adapter_dropout": 0.0,
            "use_class_weight": False,
            "label_smoothing": 0.0,
            "select_metric": "accuracy",
            "use_test_ensemble": False,
            "ensemble_group_size": 0,
            "strict_frozen_clip": True,
            "use_prompt_weight": False,
            "use_class_temperature": False,
            "use_class_bias": False,
            "feature_cache_dir": None,
            "checkpoint_output": str(checkpoint_path),
            "seed": args.seed,
            "max_sequences": args.max_sequences,
        },
        "dataset": {
            "total": len(samples),
            "train": len(splits["train"]),
            "val": len(val_samples),
            "test": len(test_samples),
            "label_distribution_train": dict(Counter([s["label"] for s in splits["train"]])),
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[DONE] saved result to {output_path}")

    # Save checkpoint (compatible with load_method)
    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

    checkpoint_payload = {
        "checkpoint_type": "clip_zeroshot",
        "config": result["config"],
        "dataset": result["dataset"],
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "prompt_groups": prompt_groups,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "text_features": text_features.cpu(),
        "output_path": str(output_path),
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)
    log(f"[DONE] saved checkpoint to {checkpoint_path}")
    print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
