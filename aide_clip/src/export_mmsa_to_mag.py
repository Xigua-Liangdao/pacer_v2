import argparse
import pickle
from pathlib import Path

import numpy as np


def detect_valid_length(array: np.ndarray) -> int:
    mask = np.abs(array).sum(axis=-1) > 0
    return int(mask.sum())


def clean_words(raw_text: str):
    return [word for word in str(raw_text).strip().split() if word]


def convert_split(payload, split_name: str):
    examples = []
    for idx in range(len(payload["raw_text"])):
        raw_text = str(payload["raw_text"][idx])
        words = clean_words(raw_text)
        visual = np.asarray(payload["vision"][idx], dtype=np.float32)
        acoustic = np.asarray(payload["audio"][idx], dtype=np.float32)
        label = float(np.asarray(payload["regression_labels"][idx]).reshape(-1)[0])
        segment = str(payload["id"][idx])

        valid_len = min(detect_valid_length(visual), detect_valid_length(acoustic))
        if valid_len <= 0:
            continue

        if len(words) == 0:
            continue

        usable_len = min(len(words), valid_len)
        words = words[:usable_len]
        visual = visual[:usable_len]
        acoustic = acoustic[:usable_len]

        examples.append(((words, visual, acoustic), label, segment))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert MMSA aligned pickle to MAG-style dataset pickle")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("rb") as f:
        data = pickle.load(f)

    converted = {
        "train": convert_split(data["train"], "train"),
        "dev": convert_split(data["valid"], "dev"),
        "test": convert_split(data["test"], "test"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(converted, f)

    print(f"[DONE] wrote MAG-style pickle: {output_path}")
    for split_name in ["train", "dev", "test"]:
        split = converted[split_name]
        print(split_name, len(split))
        if split:
            (words, visual, acoustic), label, segment = split[0]
            print(
                f"  sample words={len(words)} visual_shape={visual.shape} acoustic_shape={acoustic.shape} "
                f"label={label} segment={segment}"
            )


if __name__ == "__main__":
    main()