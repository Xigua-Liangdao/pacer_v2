#!/usr/bin/env python3

import argparse
import json
import os
import random
from pathlib import Path


EMOTION_LABELS = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
EMOTION_NORMALIZE_MAP = {
    "anxiety": "Anxiety",
    "peace": "Peace",
    "weariness": "Weariness",
    "happiness": "Happiness",
    "anger": "Anger",
}

DEFAULT_AIDE_ROOT = "/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset"
DEFAULT_ANNOTATION_ROOT = str(Path(DEFAULT_AIDE_ROOT) / "annotation")


def normalize_emotion_label(label: str) -> str:
    key = str(label).strip().lower()
    return EMOTION_NORMALIZE_MAP.get(key, str(label).strip())


def collect_samples(aide_root: str, annotation_root: str, max_sequences: int = 0):
    candidate_labels = set(EMOTION_LABELS)
    seq_ids = [
        name
        for name in os.listdir(aide_root)
        if name.isdigit() and os.path.isdir(os.path.join(aide_root, name))
    ]
    seq_ids.sort()

    samples = []
    for seq_id in seq_ids:
        anno_path = os.path.join(annotation_root, f"{seq_id}.json")
        incar_dir = os.path.join(aide_root, seq_id, "incarframes")
        if not os.path.isfile(anno_path) or not os.path.isdir(incar_dir):
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        label = normalize_emotion_label(anno.get("emotion_label", "Unknown"))
        if label not in candidate_labels:
            continue

        frame_files = [
            name
            for name in os.listdir(incar_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not frame_files:
            continue

        samples.append({"sequence_id": seq_id, "label": label})

    if max_sequences > 0:
        samples = samples[:max_sequences]
    return samples


def split_samples(samples, train_ratio: float, val_ratio: float, seed: int):
    label_groups = {}
    for sample in samples:
        label_groups.setdefault(sample["label"], []).append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for group in label_groups.values():
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


def build_count_table(splits):
    rows = []
    for label in EMOTION_LABELS:
        train_count = sum(1 for sample in splits["train"] if sample["label"] == label)
        val_count = sum(1 for sample in splits["val"] if sample["label"] == label)
        test_count = sum(1 for sample in splits["test"] if sample["label"] == label)
        rows.append(
            {
                "class": label,
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": train_count + val_count + test_count,
            }
        )
    return rows


def print_table(rows, split_sizes):
    headers = ["Class", "Train", "Val", "Test", "Total"]
    body = [
        [row["class"], str(row["train"]), str(row["val"]), str(row["test"]), str(row["total"])]
        for row in rows
    ]
    body.append(
        [
            "Total",
            str(split_sizes["train"]),
            str(split_sizes["val"]),
            str(split_sizes["test"]),
            str(sum(split_sizes.values())),
        ]
    )

    widths = [max(len(item) for item in column) for column in zip(headers, *body)]
    fmt = "  ".join(f"{{:{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in body:
        print(fmt.format(*row))


def build_latex(rows, split_sizes):
    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & Train & Val & Test & Total \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(f"{row['class']} & {row['train']} & {row['val']} & {row['test']} & {row['total']} \\\\")
    latex_lines.extend(
        [
            r"\midrule",
            f"\\textbf{{Total}} & {split_sizes['train']} & {split_sizes['val']} & {split_sizes['test']} & {sum(split_sizes.values())} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(latex_lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count AIDE-Emotion samples per class for train/val/test using the same stratified split as clip_aide_emotion_train.py"
    )
    parser.add_argument("--aide-root", default=DEFAULT_AIDE_ROOT)
    parser.add_argument("--annotation-root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--expect-train", type=int, default=0)
    parser.add_argument("--expect-val", type=int, default=0)
    parser.add_argument("--expect-test", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = collect_samples(args.aide_root, args.annotation_root, args.max_sequences)
    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    split_sizes = {name: len(items) for name, items in splits.items()}
    if args.expect_train and split_sizes["train"] != args.expect_train:
        raise ValueError(f"Expected train={args.expect_train}, got {split_sizes['train']}")
    if args.expect_val and split_sizes["val"] != args.expect_val:
        raise ValueError(f"Expected val={args.expect_val}, got {split_sizes['val']}")
    if args.expect_test and split_sizes["test"] != args.expect_test:
        raise ValueError(f"Expected test={args.expect_test}, got {split_sizes['test']}")

    rows = build_count_table(splits)

    print(f"AIDE root       : {os.path.abspath(args.aide_root)}")
    print(f"Annotation root : {os.path.abspath(args.annotation_root)}")
    print(f"Valid samples   : {len(samples)}")
    print(f"Split ratios    : train={args.train_ratio}, val={args.val_ratio}, test={1 - args.train_ratio - args.val_ratio}")
    print(f"Seed            : {args.seed}")
    print()
    print_table(rows, split_sizes)
    print()
    print("LaTeX:")
    print(build_latex(rows, split_sizes))
    print()
    print("JSON summary:")
    print(
        json.dumps(
            {
                "dataset": {
                    "total": len(samples),
                    **split_sizes,
                },
                "counts": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()