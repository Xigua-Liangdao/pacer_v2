#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODULE_PATH = SRC_DIR / "clip_ravdess_emotion_train.py"
MODULE_SPEC = importlib.util.spec_from_file_location("clip_ravdess_emotion_train", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Failed to load module from {MODULE_PATH}")
ravdess_train = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = ravdess_train
MODULE_SPEC.loader.exec_module(ravdess_train)


def load_model_bundle(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != "strict_frozen_clip_adapter":
        raise ValueError(f"Unsupported checkpoint_type in {checkpoint_path}: {checkpoint.get('checkpoint_type')}")

    config = checkpoint["config"]
    text_features = checkpoint["text_features"].float().to(device)
    adapter = ravdess_train.ClipImageAdapter(
        dim=int(text_features.shape[-1]),
        device=device,
        hidden_dim=int(config["adapter_hidden_dim"]),
        dropout=float(config["adapter_dropout"]),
        num_classes=len(ravdess_train.EMOTION_LABELS),
        num_prompts=int(text_features.shape[1]),
        use_global_logit_scale=bool(config.get("use_global_logit_scale", False)),
        use_prompt_weight=bool(config.get("use_prompt_weight", False)),
        use_class_temperature=bool(config.get("use_class_temperature", False)),
        use_class_bias=bool(config.get("use_class_bias", False)),
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "prompt_set": config.get("prompt_set"),
        "adapter": adapter,
        "text_features": text_features,
    }


def load_checkpoints(paths, device: str):
    return [load_model_bundle(Path(path), device) for path in paths]


def batch_probs(bundle, features, batch_size: int, device: str):
    outputs = []
    adapter = bundle["adapter"]
    text_features = bundle["text_features"]
    for start in range(0, features.shape[0], batch_size):
        batch_x = features[start:start + batch_size].to(device)
        with torch.no_grad():
            logits = adapter.logits(batch_x, text_features)
            probs = torch.softmax(logits, dim=-1)
        outputs.append(probs.detach().cpu())
    return torch.cat(outputs, dim=0)


def average_group_probs(bundles, features, batch_size: int, device: str):
    probs = [batch_probs(bundle, features, batch_size, device) for bundle in bundles]
    stacked = torch.stack(probs, dim=0)
    return stacked.mean(dim=0)


def predict(scores, idx2label):
    idxs = scores.argmax(dim=-1).tolist()
    return [idx2label[int(i)] for i in idxs]


def summarize(y_true, y_pred):
    return ravdess_train.summarize_predictions(y_true, y_pred)


def parse_paths(raw: str):
    return [item for item in raw.split(",") if item]


def main():
    parser = argparse.ArgumentParser(description="Late fusion for two groups of RAVDESS checkpoints.")
    parser.add_argument("--group_a", required=True, help="Comma-separated checkpoint paths for group A")
    parser.add_argument("--group_b", required=True, help="Comma-separated checkpoint paths for group B")
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--select_metric", choices=["weighted_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = torch.load(args.feature_cache, map_location="cpu")
    val_x = cache["val_x"].float()
    test_x = cache["test_x"].float()
    val_true = [ravdess_train.EMOTION_LABELS[int(x.item())] for x in cache["val_y"]]
    test_true = [ravdess_train.EMOTION_LABELS[int(x.item())] for x in cache["test_y"]]
    idx2label = {idx: label for idx, label in enumerate(ravdess_train.EMOTION_LABELS)}

    group_a_paths = parse_paths(args.group_a)
    group_b_paths = parse_paths(args.group_b)
    group_a = load_checkpoints(group_a_paths, args.device)
    group_b = load_checkpoints(group_b_paths, args.device)

    avg_a_val = average_group_probs(group_a, val_x, args.batch_size, args.device)
    avg_b_val = average_group_probs(group_b, val_x, args.batch_size, args.device)
    avg_a_test = average_group_probs(group_a, test_x, args.batch_size, args.device)
    avg_b_test = average_group_probs(group_b, test_x, args.batch_size, args.device)

    trials = []
    best = None
    for step in range(11):
        alpha = step / 10.0
        val_scores = alpha * avg_a_val + (1.0 - alpha) * avg_b_val
        test_scores = alpha * avg_a_test + (1.0 - alpha) * avg_b_test
        val_pred = predict(val_scores, idx2label)
        test_pred = predict(test_scores, idx2label)
        val_metrics = summarize(val_true, val_pred)
        test_metrics = summarize(test_true, test_pred)
        trial = {
            "alpha_for_group_a": round(alpha, 4),
            "alpha_for_group_b": round(1.0 - alpha, 4),
            "val": val_metrics,
            "test": test_metrics,
        }
        trials.append(trial)
        metric = val_metrics[args.select_metric]
        tie = val_metrics["accuracy"] if args.select_metric == "weighted_f1" else val_metrics["weighted_f1"]
        ranking = (-metric, -tie, -test_metrics["accuracy"])
        if best is None or ranking < best["ranking"]:
            best = {"ranking": ranking, "trial": trial}

    single_a_val = summarize(val_true, predict(avg_a_val, idx2label))
    single_a_test = summarize(test_true, predict(avg_a_test, idx2label))
    single_b_val = summarize(val_true, predict(avg_b_val, idx2label))
    single_b_test = summarize(test_true, predict(avg_b_test, idx2label))

    result = {
        "config": {
            "group_a": group_a_paths,
            "group_b": group_b_paths,
            "feature_cache": args.feature_cache,
            "device": args.device,
            "batch_size": args.batch_size,
            "select_metric": args.select_metric,
        },
        "single_group_a": {"val": single_a_val, "test": single_a_test},
        "single_group_b": {"val": single_b_val, "test": single_b_test},
        "best_fusion": best["trial"],
        "all_trials": trials,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "best_fusion": best["trial"],
        "single_group_a_test": single_a_test,
        "single_group_b_test": single_b_test,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
