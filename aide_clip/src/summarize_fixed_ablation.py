import json
from pathlib import Path

RESULTS_DIR = Path("/data1/yanjing/talk2bev/aide_clip/results/yawdd/ablation_fixed")

EXPS = {
    "P01": "P01",
    "P2": "P2",
    "P3": "P3",
    "P4": "P4",
    "P5": "P5",
    "P6": "P6",
    "P7": "P7",
    "A1": "A1",
    "A2": "A2",
    "A3": "A3",
    "A4": "A4",
    "A5": "A5",
}

PROGRESSIVE = [
    ("P0 zero-shot", "P01", "zero_shot"),
    ("P1 linear probe", "P01", "test"),
    ("P2 + Adapter", "P2", "test"),
    ("P3 + Multi-Prompt", "P3", "test"),
    ("P4 + Prompt Weight", "P4", "test"),
    ("P5 + Temporal", "P5", "test"),
    ("P6 + Class T&B", "P6", "test"),
    ("P7 Full", "P7", "test"),
]

ABLATION = [
    ("Full (P7)", "P7", "test"),
    ("w/o Temporal", "A1", "test"),
    ("w/o Multi-Prompt", "A2", "test"),
    ("w/o Prompt Weight", "A3", "test"),
    ("w/o Class T&B", "A4", "test"),
    ("w/o TTE", "A5", "test"),
]


def find_json(prefix):
    candidates = [candidate for candidate in RESULTS_DIR.glob(f"{prefix}*.json") if ".fold" not in candidate.name]
    if not candidates:
        return None
    return min(candidates, key=lambda path: len(path.name))


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def extract(prefix, section):
    path = find_json(EXPS[prefix])
    if path is None or not path.exists():
        return None, None, None, None, "FILE_NOT_FOUND"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if section == "zero_shot":
        zero_shot = data.get("zero_shot")
        if not zero_shot:
            return None, None, None, None, str(path.name)
        acc = zero_shot.get("accuracy")
        f1 = zero_shot.get("weighted_f1") or zero_shot.get("f1")
        precision = zero_shot.get("precision")
        recall = zero_shot.get("recall")
        return acc, f1, precision, recall, str(path.name)

    test = data.get("test") or data.get("aggregate")
    if not test:
        return None, None, None, None, str(path.name)

    if isinstance(test.get("accuracy"), dict):
        acc = test["accuracy"].get("mean")
        f1 = test.get("f1", {}).get("mean") or test.get("weighted_f1", {}).get("mean")
        precision = test.get("precision", {}).get("mean")
        recall = test.get("recall", {}).get("mean")
    else:
        acc = test.get("accuracy")
        f1 = test.get("weighted_f1") or test.get("f1")
        precision = test.get("precision")
        recall = test.get("recall")

    return acc, f1, precision, recall, str(path.name)


def print_table(title, rows):
    out = [f"\n## {title}\n", "| Configuration | Accuracy | F1 | Precision | Recall | Source |", "|---|---|---|---|---|---|"]
    for name, prefix, section in rows:
        acc, f1, precision, recall, src = extract(prefix, section)
        out.append(f"| {name} | {fmt(acc)} | {fmt(f1)} | {fmt(precision)} | {fmt(recall)} | {src} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("# YawDD Binary Ablation — Fixed Split")
    print(print_table("Table 1: Progressive Build-up", PROGRESSIVE))
    print(print_table("Table 2: Leave-one-out Ablation", ABLATION))