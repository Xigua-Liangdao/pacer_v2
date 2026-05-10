#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


KNOWN_METRICS = {
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "uar",
    "war",
    "precision",
    "recall",
    "f1",
    "drowsy_f1",
}

ALIASES = {
    "acc": "accuracy",
    "wf1": "weighted_f1",
}


class MetricsError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def round_metric(value: float) -> float:
    return round(float(value), 6)


def infer_dataset(payload: Dict[str, Any], source_path: Path) -> str | None:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    dataset = config.get("dataset")
    if isinstance(dataset, str) and dataset:
        return dataset
    lower_path = str(source_path).lower()
    if "yawdd" in lower_path:
        return "YawDD"
    if "aide" in lower_path:
        return "AIDE"
    if "aide_root" in config:
        return "AIDE"
    if "yawdd_root" in config:
        return "YawDD"
    return None


def infer_task(payload: Dict[str, Any]) -> str | None:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    task = config.get("task")
    if isinstance(task, str) and task:
        return task
    return None


def maybe_float(value: Any) -> float | None:
    if is_number(value):
        return round_metric(value)
    return None


def parse_confusion_matrix(matrix: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    labels = list(matrix.keys())
    if not labels:
        return {}

    total = 0.0
    correct = 0.0
    weighted_f1_numerator = 0.0
    recalls = []
    f1s = []

    for label in labels:
        row = matrix.get(label, {})
        support = float(sum(float(row.get(pred, 0.0)) for pred in row))
        tp = float(row.get(label, 0.0))
        fp = 0.0
        for other_label in labels:
            if other_label == label:
                continue
            fp += float(matrix.get(other_label, {}).get(label, 0.0))
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        total += support
        correct += tp
        weighted_f1_numerator += support * f1
        recalls.append(recall)
        f1s.append(f1)

    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    uar = sum(recalls) / len(recalls) if recalls else 0.0
    weighted_f1 = weighted_f1_numerator / total if total else 0.0
    return {
        "accuracy": round_metric(accuracy),
        "war": round_metric(accuracy),
        "uar": round_metric(uar),
        "macro_f1": round_metric(macro_f1),
        "weighted_f1": round_metric(weighted_f1),
    }


def normalize_metric_dict(section: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in section.items():
        canonical_key = ALIASES.get(key, key)
        if canonical_key in KNOWN_METRICS and is_number(value):
            metrics[canonical_key] = round_metric(value)

    confusion_matrix = section.get("confusion_matrix")
    if isinstance(confusion_matrix, dict):
        derived = parse_confusion_matrix(confusion_matrix)
        for key, value in derived.items():
            metrics.setdefault(key, value)

    return metrics


def normalize_aggregate(section: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in section.items():
        canonical_key = ALIASES.get(key, key)
        if isinstance(value, dict):
            mean_value = maybe_float(value.get("mean"))
            std_value = maybe_float(value.get("std"))
            if mean_value is not None:
                metrics[f"{canonical_key}_mean"] = mean_value
                if canonical_key not in metrics:
                    metrics[canonical_key] = mean_value
            if std_value is not None:
                metrics[f"{canonical_key}_std"] = std_value
        elif canonical_key in KNOWN_METRICS and is_number(value):
            metrics[canonical_key] = round_metric(value)
    return metrics


def normalize_zero_shot(section: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if not isinstance(section, dict):
        return {}, "zero_shot"
    if "test" in section and isinstance(section["test"], dict):
        metrics = normalize_metric_dict(section["test"])
        return {"zero_shot.test": metrics}, "zero_shot.test"
    metrics = normalize_metric_dict(section)
    return {"zero_shot": metrics}, "zero_shot"


def normalize_summary_results(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise MetricsError("summary payload does not contain a non-empty results list")
    best = results[0]
    if not isinstance(best, dict):
        raise MetricsError("summary payload best entry is not a JSON object")
    metrics: Dict[str, float] = {}
    for key in ("accuracy", "acc", "weighted_f1", "wf1", "macro_f1", "uar", "war", "f1"):
        canonical_key = ALIASES.get(key, key)
        value = best.get(key)
        if is_number(value):
            metrics[canonical_key] = round_metric(value)
    if not metrics:
        raise MetricsError("summary payload best entry is missing recognized metric fields")
    return {"summary_best": metrics}, "summary_best", "summary_list"


def choose_primary_section(sections: Dict[str, Dict[str, float]]) -> str:
    for preferred in ("test", "aggregate", "summary_best", "zero_shot.test", "zero_shot", "val"):
        if preferred in sections and sections[preferred]:
            return preferred
    for key, metrics in sections.items():
        if metrics:
            return key
    raise MetricsError("no usable metric sections were found in the result JSON")


def choose_primary_metric(metrics: Dict[str, float]) -> str:
    for preferred in ("weighted_f1", "macro_f1", "accuracy", "uar", "war", "f1"):
        if preferred in metrics:
            return preferred
    raise MetricsError("no recognized scalar metrics were found in the selected section")


def normalize_payload(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    if isinstance(payload.get("metrics"), dict) and isinstance(payload.get("selected_section"), str):
        return payload

    dataset = infer_dataset(payload, source_path)
    task = infer_task(payload)
    source_type = "experiment_result"
    sections: Dict[str, Dict[str, float]] = {}

    if isinstance(payload.get("val"), dict):
        sections["val"] = normalize_metric_dict(payload["val"])
    if isinstance(payload.get("test"), dict):
        sections["test"] = normalize_metric_dict(payload["test"])
    if isinstance(payload.get("val_metrics"), dict):
        sections.setdefault("val", {}).update(normalize_metric_dict(payload["val_metrics"]))
    if isinstance(payload.get("test_metrics"), dict):
        sections.setdefault("test", {}).update(normalize_metric_dict(payload["test_metrics"]))
    if isinstance(payload.get("aggregate"), dict):
        sections["aggregate"] = normalize_aggregate(payload["aggregate"])
    if payload.get("zero_shot") is not None:
        zero_shot_sections, _ = normalize_zero_shot(payload["zero_shot"])
        sections.update(zero_shot_sections)

    if not any(sections.values()) and isinstance(payload.get("results"), list):
        sections, selected_section, source_type = normalize_summary_results(payload)
    else:
        selected_section = choose_primary_section(sections)

    metrics = sections.get(selected_section, {})
    if not metrics:
        raise MetricsError(f"selected section {selected_section!r} is present but has no usable metrics")
    primary_metric = choose_primary_metric(metrics)

    return {
        "source_file": str(source_path),
        "source_type": source_type,
        "dataset": dataset,
        "task": task,
        "selected_section": selected_section,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "sections": sections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize repository result JSON files into a stable metrics.json format.")
    parser.add_argument("--input", required=True, help="Path to a raw experiment result JSON or an existing metrics.json.")
    parser.add_argument("--output", required=True, help="Path to the normalized metrics.json output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"[ERROR] metrics input not found: {input_path}")

    try:
        payload = load_json(input_path)
        normalized = normalize_payload(payload, input_path)
    except MetricsError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    dump_json(output_path, normalized)
    print(f"[DONE] wrote normalized metrics to {output_path}")
    print(f"[INFO] selected_section={normalized['selected_section']}")
    print(f"[INFO] primary_metric={normalized['primary_metric']}")
    for name, value in normalized["metrics"].items():
        print(f"[METRIC] {name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())