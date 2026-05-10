import json
import statistics
from pathlib import Path

RESULTS = Path("/data1/yanjing/talk2bev/aide_clip/results/yawdd/ablation_v2")
SEEDS = [7]

EXPS = [
    ("P0 zero-shot", "P01", "zero_shot"),
    ("P1 linear probe", "P01", "test"),
    ("P2 + Adapter", "P2", "test"),
    ("P3 + Multi-Prompt", "P3", "test"),
    ("P4 + Prompt Weight", "P4", "test"),
    ("P5 + Temporal", "P5", "test"),
    ("P6 Full (ours)", "P6", "test"),
]

ABL = [
    ("Full (P6)", "P6", "test"),
    ("w/o Temporal", "A1", "test"),
    ("w/o Multi-Prompt", "A2", "test"),
    ("w/o Prompt Weight", "A3", "test"),
    ("w/o Class T&B", "A4", "test"),
]


def find_json(prefix, seed):
    candidates = [candidate for candidate in RESULTS.glob(f"{prefix}_seed{seed}*.json") if ".fold" not in candidate.name]
    if not candidates:
        return None
    return min(candidates, key=lambda path: len(path.name))


def get_metrics(prefix, seed, section):
    path = find_json(prefix, seed)
    if not path or not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if section == "zero_shot":
        zero_shot = data.get("zero_shot")
        if not zero_shot:
            return None
        return {
            "acc": zero_shot.get("accuracy"),
            "f1": zero_shot.get("weighted_f1") or zero_shot.get("f1"),
        }
    test = data.get("test") or {}
    acc = test.get("accuracy")
    if isinstance(acc, dict):
        acc = acc.get("mean")
    f1 = test.get("weighted_f1") or test.get("f1")
    if isinstance(f1, dict):
        f1 = f1.get("mean")
    return {"acc": acc, "f1": f1}


def aggregate(prefix, section):
    accs = []
    f1s = []
    for seed in SEEDS:
        metrics = get_metrics(prefix, seed, section)
        if metrics and metrics["acc"] is not None:
            accs.append(metrics["acc"])
        if metrics and metrics["f1"] is not None:
            f1s.append(metrics["f1"])
    if not accs:
        return "N/A", "N/A"
    if len(accs) == 1:
        return f"{accs[0]:.4f}", f"{f1s[0]:.4f}" if f1s else "N/A"
    return (
        f"{statistics.mean(accs):.4f} +- {statistics.stdev(accs):.4f}",
        f"{statistics.mean(f1s):.4f} +- {statistics.stdev(f1s):.4f}" if len(f1s) > 1 else "N/A",
    )


def print_table(title, rows):
    print(f"\n## {title}\n")
    print("| Configuration | Accuracy | F1 |")
    print("|---|---|---|")
    for name, prefix, section in rows:
        acc, f1 = aggregate(prefix, section)
        print(f"| {name} | {acc} | {f1} |")


def print_baseline():
    path = RESULTS / "frame_zero_shot_baseline.json"
    if not path.exists():
        print("\nFrame-level zero-shot baseline: NOT FOUND")
        return
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print("\n## Baseline: Frame-level Zero-shot Majority Vote\n")
    print(f"- Accuracy: {data['accuracy']:.4f}")
    print(f"- Per-class: {data['per_class_accuracy']}")


if __name__ == "__main__":
    if len(SEEDS) == 1:
        seed_desc = f"Seed {SEEDS[0]}"
    else:
        seed_desc = f"{len(SEEDS)} Seeds"
    print(f"# YawDD Binary Final Ablation V2 — Fixed Split, {seed_desc}")
    print_table("Table 1: Progressive Build-up", EXPS)
    print_table("Table 2: Leave-one-out Ablation", ABL)
    print_baseline()