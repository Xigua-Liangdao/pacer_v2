"""Frame-level CLIP zero-shot majority vote baseline for YawDD binary fixed-test split."""

from pathlib import Path
import json
import sys

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from clip_yawdd_emotion_train import collect_all_face_sequence_samples, resolve_label_space  # noqa: E402


MODEL_ID = "openai/clip-vit-base-patch32"
DATA_ROOT = "/data1/yanjing/talk2bev/fatigue-drive-yawning-detection/extracted_face_multi4"
OUTPUT = "/data1/yanjing/talk2bev/aide_clip/results/yawdd/ablation_v2/frame_zero_shot_baseline.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    all_samples, _ = collect_all_face_sequence_samples(DATA_ROOT, "binary")
    test_samples = [sample for sample in all_samples if sample.get("source_split") == "fixed_test"]
    print(f"test sequences: {len(test_samples)}")

    labels, _, prompt_groups = resolve_label_space("binary")
    prompts_per_class = [prompt_groups[label] for label in labels]
    all_texts = [prompt for group in prompts_per_class for prompt in group]
    text_inputs = processor(text=all_texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    class_indices = []
    offset = 0
    for group in prompts_per_class:
        class_indices.append(list(range(offset, offset + len(group))))
        offset += len(group)

    correct = 0
    total = 0
    predictions = []
    class_stats = {label: {"correct": 0, "total": 0} for label in labels}

    for sample in test_samples:
        frames = [frame for frame in sample.get("frame_paths", []) if Path(frame).exists()]
        if not frames:
            continue
        images = [Image.open(frame_path).convert("RGB") for frame_path in frames]
        img_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_feats = model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            sim = img_feats @ text_feats.T
        class_scores = torch.stack([sim[:, idxs].mean(dim=-1) for idxs in class_indices], dim=1)
        frame_preds = class_scores.argmax(dim=-1)
        vote = torch.bincount(frame_preds, minlength=len(labels))
        seq_pred_idx = int(vote.argmax().item())
        seq_pred = labels[seq_pred_idx]
        ground_truth = sample["label"]
        is_correct = seq_pred == ground_truth
        correct += int(is_correct)
        total += 1
        class_stats[ground_truth]["total"] += 1
        if is_correct:
            class_stats[ground_truth]["correct"] += 1
        predictions.append(
            {
                "sequence_id": sample.get("sequence_id"),
                "actor_id": sample.get("actor_id"),
                "gt": ground_truth,
                "pred": seq_pred,
                "n_frames": len(frames),
                "vote": vote.tolist(),
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "method": "frame_zero_shot_majority_vote",
        "model": MODEL_ID,
        "test_sequences": total,
        "accuracy": accuracy,
        "per_class_accuracy": {
            label: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for label, stats in class_stats.items()
        },
        "per_class_count": class_stats,
        "predictions": predictions,
    }
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"accuracy={accuracy:.4f} | per_class={result['per_class_accuracy']}")


if __name__ == "__main__":
    main()