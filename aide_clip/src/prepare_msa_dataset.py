import argparse
import json
from collections import Counter
from pathlib import Path

from msa_dataset_adapter import convert_mmsa_pickle_to_manifest


def summarize_labels(samples):
    names = [sample["label_name"] for sample in samples if sample["label_name"] is not None]
    return dict(Counter(names))


def main():
    parser = argparse.ArgumentParser(description="Convert MMSA processed pickle to aide_clip manifest")
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label_mode", default="ternary", choices=["regression", "binary", "ternary", "seven"])
    args = parser.parse_args()

    manifest = convert_mmsa_pickle_to_manifest(
        input_path=args.input,
        dataset_name=args.dataset,
        label_mode=args.label_mode,
    )

    split_summary = {}
    for split_name, samples in manifest["splits"].items():
        split_summary[split_name] = {
            "num_samples": len(samples),
            "label_distribution": summarize_labels(samples),
        }
    manifest["summary"] = split_summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote manifest: {output_path}")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()