import argparse
import pickle
from pathlib import Path

import numpy as np


def safe_shape(value):
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None


def summarize_split(name, payload):
    summary = {"split": name, "keys": sorted(payload.keys())}
    if "raw_text" in payload:
        summary["num_samples"] = len(payload["raw_text"])
        summary["raw_text_preview"] = payload["raw_text"][:2]
    elif "id" in payload:
        summary["num_samples"] = len(payload["id"])
        summary["id_preview"] = payload["id"][:3]

    for key in ["text", "text_bert", "audio", "vision", "regression_labels", "classification_labels"]:
        if key in payload:
            summary[f"{key}_shape"] = safe_shape(payload[key])

    if "annotations" in payload:
        summary["annotations_preview"] = payload["annotations"][:3]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect MMSA-style processed feature pickle")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload, got: {type(data)}")

    print(f"[INFO] file: {input_path}")
    print(f"[INFO] top-level keys: {sorted(data.keys())}")

    for split_name in ["train", "valid", "test"]:
        if split_name in data and isinstance(data[split_name], dict):
            print(f"\n[SUMMARY] {split_name}")
            summary = summarize_split(split_name, data[split_name])
            for key, value in summary.items():
                print(f"- {key}: {value}")


if __name__ == "__main__":
    main()