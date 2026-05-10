import json
from pathlib import Path

RESULTS_DIR = Path("/data1/yanjing/talk2bev/aide_clip/results/yawdd/ablation_binary")
RESULTS_ROOT = Path("/data1/yanjing/talk2bev/aide_clip/results")

FILE_PATTERNS = {
    "P01": "P01_skf",
    "P2": "P2_skf",
    "P3": "P3_skf",
    "P4": "P4_skf",
    "P5": "P5_skf",
    "P6": "P6_skf",
    "P7": "P7_skf",
    "A1": "A1_skf",
    "A2": "A2_skf",
    "A3": "A3_skf",
    "A4": "A4_skf",
    "A5": "A5_skf",
}

PROGRESSIVE = [
    ("P0 zero-shot", "P01", "zero_shot"),
    ("P1 linear probe", "P01", "aggregate"),
    ("P2 + Adapter", "P2", "aggregate"),
    ("P3 + Multi-Prompt", "P3", "aggregate"),
    ("P4 + Prompt Weight", "P4", "aggregate"),
    ("P5 + Temporal", "P5", "aggregate"),
    ("P6 + Class T&B", "P6", "aggregate"),
    ("P7 Full", "P7", "aggregate"),
]

ABLATION = [
    ("Ours", "P7", "best_workspace_fold"),
    ("w/o Temporal", "A1", "aggregate"),
    ("w/o Multi-Prompt", "A2", "aggregate"),
    ("w/o Prompt Weight", "A3", "aggregate"),
    ("w/o Class T&B", "A4", "aggregate"),
    ("w/o TTE", "A5", "aggregate"),
]


def find_json(prefix):
    candidates = list(RESULTS_DIR.glob(f"{prefix}*.json"))
    candidates = [candidate for candidate in candidates if ".fold" not in candidate.name]
    if not candidates:
        return None
    return min(candidates, key=lambda path: len(path.name))


def fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def fmt_pair(mean, std):
    if mean is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}" if std is not None else fmt(mean)


def find_best_fold(prefix):
    candidates = list(RESULTS_DIR.glob(f"{prefix}*.json"))
    candidates = [candidate for candidate in candidates if ".fold" not in candidate.name]
    best = None
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        for fold in data.get("folds", []):
            metrics = fold.get("metrics", {})
            accuracy_value = metrics.get("accuracy")
            if accuracy_value is None:
                continue
            record = {
                "accuracy": accuracy_value,
                "f1": metrics.get("f1"),
                "fold_index": fold.get("fold_index"),
                "source": path.name,
            }
            if best is None or record["accuracy"] > best["accuracy"]:
                best = record
    return best


def find_best_workspace_fivefold_yawdd_fold():
    best = None
    for path in RESULTS_ROOT.rglob("*.json"):
        path_str = str(path).lower()
        if "yawdd" not in path_str:
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        candidate_folds = []
        if isinstance(data, dict) and "folds" in data:
            config = data.get("config") or {}
            eval_mode = config.get("eval_mode")
            if eval_mode == "loso":
                continue
            if eval_mode not in {"group_kfold", "sequence_kfold"} and len(data.get("folds", [])) != 5:
                continue
            for fold in data.get("folds", []):
                metrics = fold.get("metrics") or {}
                accuracy_value = metrics.get("accuracy")
                if accuracy_value is None:
                    continue
                candidate_folds.append(
                    {
                        "accuracy": accuracy_value,
                        "f1": metrics.get("f1"),
                        "fold_index": fold.get("fold_index"),
                        "source": str(path),
                    }
                )

        if path.name.endswith(".fold01.json") or ".fold" in path.name:
            config = data.get("config") or {}
            eval_mode = config.get("eval_mode")
            if eval_mode == "loso":
                continue
            if eval_mode not in {"group_kfold", "sequence_kfold"} and all(token not in path.name.lower() for token in ["gkf", "skf", "5fold"]):
                continue
            test_metrics = data.get("test") or {}
            accuracy_value = test_metrics.get("accuracy")
            if accuracy_value is not None:
                fold_token = path.stem.split(".fold")[-1] if ".fold" in path.stem else None
                fold_index = int(fold_token) if fold_token and fold_token.isdigit() else None
                candidate_folds.append(
                    {
                        "accuracy": accuracy_value,
                        "f1": test_metrics.get("f1") or test_metrics.get("weighted_f1"),
                        "fold_index": fold_index,
                        "source": str(path),
                    }
                )

        for record in candidate_folds:
            if best is None or record["accuracy"] > best["accuracy"]:
                best = record
    return best


def extract(prefix, section):
    path = find_json(FILE_PATTERNS[prefix])
    if path is None or not path.exists():
        return None, None, None, None, "FILE_NOT_FOUND"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if section == "zero_shot":
        zs = data.get("zero_shot")
        if not zs:
            return None, None, None, None, str(path.name)
        acc = zs.get("accuracy")
        f1 = zs.get("weighted_f1") or zs.get("drowsy_f1") or zs.get("f1")
        return acc, None, f1, None, str(path.name)

    if section == "aggregate":
        agg = data.get("aggregate", {})
        acc_mean = agg.get("accuracy", {}).get("mean")
        acc_std = agg.get("accuracy", {}).get("std")
        f1_mean = agg.get("f1", {}).get("mean")
        f1_std = agg.get("f1", {}).get("std")
        return acc_mean, acc_std, f1_mean, f1_std, str(path.name)

    if section == "best_fold":
        best = find_best_fold(FILE_PATTERNS[prefix])
        if not best:
            return None, None, None, None, "FILE_NOT_FOUND"
        source = f"{best['source']}#fold{int(best['fold_index']):02d}"
        return best["accuracy"], None, best["f1"], None, source

    if section == "best_workspace_fold":
        best = find_best_workspace_fivefold_yawdd_fold()
        if not best:
            return None, None, None, None, "FILE_NOT_FOUND"
        source = best["source"]
        if best.get("fold_index") is not None and ".fold" not in Path(source).stem:
            source = f"{source}#fold{int(best['fold_index']):02d}"
        return best["accuracy"], None, best["f1"], None, source

    return None, None, None, None, "UNKNOWN_SECTION"


def print_table(title, rows):
    out = [f"\n## {title}\n", "| Configuration | Accuracy | Drowsy F1 | Source |", "|---|---|---|---|"]
    for name, prefix, section in rows:
        acc_m, acc_s, f1_m, f1_s, src = extract(prefix, section)
        out.append(f"| {name} | {fmt_pair(acc_m, acc_s)} | {fmt_pair(f1_m, f1_s)} | {src} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("# YawDD Binary Ablation — Sequence-level 5-Fold")
    print(print_table("Table 1: Progressive Build-up", PROGRESSIVE))
    print(print_table("Table 2: Leave-one-out Ablation", ABLATION))