#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch


COLUMNS = [
    "文件名",
    "strict_frozen_clip",
    "temporal_head",
    "temporal_pool_mode",
    "num_frames",
    "use_prompt_weight",
    "use_class_temperature",
    "use_class_bias",
    "use_test_ensemble",
    "ensemble_group_size",
    "prompt_set",
    "adapter_hidden_dim",
    "label_smoothing",
    "epochs",
    "lr",
    "val accuracy",
    "val weighted_f1",
    "test accuracy",
    "test weighted_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively scan checkpoint/result files and summarize configs and metrics."
    )
    parser.add_argument(
        "scan_dir",
        nargs="?",
        default="results",
        help="Directory to scan recursively. Defaults to results/",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many rows to print after sorting by test weighted_f1.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def iter_candidate_files(scan_dir: Path) -> Iterable[Path]:
    for path in sorted(scan_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".ckpt.pt") or path.suffix.lower() == ".json":
            yield path


def normalize_metrics(metrics: Optional[Dict]) -> Dict:
    return metrics if isinstance(metrics, dict) else {}


def choose_metric_block(container: Dict, preferred_keys: List[str]) -> Dict:
    for key in preferred_keys:
        value = container.get(key)
        if isinstance(value, dict) and any(metric_key in value for metric_key in ("accuracy", "weighted_f1")):
            return value
    return {}


def extract_val_test_metrics(payload: Dict) -> Tuple[Dict, Dict]:
    val_metrics = choose_metric_block(
        payload,
        ["val", "val_metrics", "best_val", "zero_shot_val_metrics"],
    )
    test_metrics = choose_metric_block(
        payload,
        ["test", "test_metrics", "best_test", "zero_shot_test_metrics"],
    )

    metrics_block = normalize_metrics(payload.get("metrics"))
    if not val_metrics:
        val_metrics = choose_metric_block(
            metrics_block,
            ["val", "val_metrics", "best_val", "zero_shot_val_metrics"],
        )
    if not test_metrics:
        test_metrics = choose_metric_block(
            metrics_block,
            ["test", "test_metrics", "best_test", "zero_shot_test_metrics"],
        )

    return normalize_metrics(val_metrics), normalize_metrics(test_metrics)


def extract_config(payload: Dict) -> Dict:
    config = payload.get("config")
    if isinstance(config, dict):
        return config
    metrics_block = payload.get("metrics")
    if isinstance(metrics_block, dict):
        nested_config = metrics_block.get("config")
        if isinstance(nested_config, dict):
            return nested_config
    return {}


def config_value(config: Dict, *keys: str):
    for key in keys:
        if key in config:
            return config.get(key)
    return None


def build_row(relative_name: str, config: Dict, val_metrics: Dict, test_metrics: Dict) -> Dict:
    return {
        "文件名": relative_name,
        "strict_frozen_clip": config_value(config, "strict_frozen_clip"),
        "temporal_head": config_value(config, "temporal_head"),
        "temporal_pool_mode": config_value(config, "temporal_pool_mode", "temporal_pooling"),
        "num_frames": config_value(config, "num_frames"),
        "use_prompt_weight": config_value(config, "use_prompt_weight"),
        "use_class_temperature": config_value(config, "use_class_temperature"),
        "use_class_bias": config_value(config, "use_class_bias"),
        "use_test_ensemble": config_value(config, "use_test_ensemble"),
        "ensemble_group_size": config_value(config, "ensemble_group_size"),
        "prompt_set": config_value(config, "prompt_set"),
        "adapter_hidden_dim": config_value(config, "adapter_hidden_dim"),
        "label_smoothing": config_value(config, "label_smoothing"),
        "epochs": config_value(config, "epochs"),
        "lr": config_value(config, "lr"),
        "val accuracy": val_metrics.get("accuracy"),
        "val weighted_f1": val_metrics.get("weighted_f1"),
        "test accuracy": test_metrics.get("accuracy"),
        "test weighted_f1": test_metrics.get("weighted_f1"),
    }


def load_payload(path: Path) -> Dict:
    if path.name.endswith(".ckpt.pt"):
        payload = torch.load(path, map_location="cpu")
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("payload is not a dict")
    return payload


def should_keep_result(config: Dict, val_metrics: Dict, test_metrics: Dict) -> bool:
    return bool(config or val_metrics or test_metrics)


def main() -> int:
    args = parse_args()
    scan_dir = Path(args.scan_dir).expanduser().resolve()
    if not scan_dir.exists():
        print(f"Scan directory does not exist: {scan_dir}", file=sys.stderr)
        return 1

    rows: List[Dict] = []
    best_config: Optional[Dict] = None

    for path in iter_candidate_files(scan_dir):
        try:
            payload = load_payload(path)
            config = extract_config(payload)
            val_metrics, test_metrics = extract_val_test_metrics(payload)
            if not should_keep_result(config, val_metrics, test_metrics):
                warn(f"Skipping non-result file with no config/metrics: {path}")
                continue
            relative_name = str(path.relative_to(scan_dir.parent))
            rows.append(build_row(relative_name, config, val_metrics, test_metrics))
        except Exception as exc:
            warn(f"Failed to read {path}: {exc}")

    if not rows:
        print("No readable checkpoint/result files were found.")
        return 0

    df = pd.DataFrame(rows, columns=COLUMNS)
    sort_key = pd.to_numeric(df["test weighted_f1"], errors="coerce")
    accuracy_key = pd.to_numeric(df["test accuracy"], errors="coerce")
    df = df.assign(_sort_key=sort_key.fillna(float("-inf")), _accuracy_key=accuracy_key.fillna(float("-inf")))
    df = df.sort_values(by=["_sort_key", "_accuracy_key"], ascending=[False, False], kind="mergesort")
    df = df.drop(columns=["_sort_key", "_accuracy_key"]).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 120)

    top_df = df.head(args.top_k)
    print(top_df.to_string(index=False))

    best_row = df.iloc[0]
    best_path = scan_dir.parent / Path(str(best_row["文件名"]))
    try:
        best_payload = load_payload(best_path)
        best_config = extract_config(best_payload)
    except Exception as exc:
        warn(f"Failed to reload best config from {best_path}: {exc}")
        best_config = {}

    print("\nBest Config Summary")
    print(json.dumps(best_config or {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())