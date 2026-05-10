#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODULE_PATH = SRC_DIR / "clip_ravdess_emotion_train.py"
MODULE_SPEC = importlib.util.spec_from_file_location("clip_ravdess_emotion_train", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Failed to load module from {MODULE_PATH}")
ravdess_train = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = ravdess_train
MODULE_SPEC.loader.exec_module(ravdess_train)


def main():
    parser = argparse.ArgumentParser(description="Select RAVDESS auto prompts with CLIP text embeddings.")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts_per_class", type=int, default=4)
    parser.add_argument("--refine_passes", type=int, default=2)
    parser.add_argument("--top_pairs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)
    model = model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    prompt_groups, text_features, diagnostics = ravdess_train.select_ravdess_auto_prompt_groups(
        processor=processor,
        model=model,
        device=args.device,
        prompts_per_class=args.prompts_per_class,
        refine_passes=args.refine_passes,
        top_pairs=args.top_pairs,
    )

    result = {
        "config": {
            "model_id": args.model_id,
            "device": args.device,
            "prompt_set": "ravdess_8_auto_selected",
            "prompts_per_class": args.prompts_per_class,
            "refine_passes": args.refine_passes,
            "top_pairs": args.top_pairs,
        },
        "prompt_groups": prompt_groups,
        "text_prototype_similarity": diagnostics["prototype_similarity"],
        "text_similarity_stats": {
            "mean_off_diagonal_similarity": diagnostics["mean_off_diagonal_similarity"],
            "max_off_diagonal_similarity": diagnostics["max_off_diagonal_similarity"],
            "most_confusing_pairs": diagnostics["most_confusing_pairs"],
            "mean_within_class_similarity": diagnostics["mean_within_class_similarity"],
        },
        "prompt_selection": diagnostics,
        "text_feature_shape": list(text_features.shape),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output": str(output_path),
        "mean_off_diagonal_similarity": diagnostics["mean_off_diagonal_similarity"],
        "max_off_diagonal_similarity": diagnostics["max_off_diagonal_similarity"],
        "most_confusing_pairs": diagnostics["most_confusing_pairs"][:3],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
