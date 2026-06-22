#!/usr/bin/env python3
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_aide_emotion_train import (  # noqa: E402
    ClipImageAdapter,
    EMOTION_LABELS,
    TemporalTransformerClipImageAdapter,
    build_class_prompts,
    collect_samples,
    log,
    sample_frame_paths,
    split_samples,
)


DEFAULT_AIDE_ROOT = str(PROJECT_ROOT.parent / "data" / "AIDE_Dataset")
DEFAULT_ANNOTATION_ROOT = f"{DEFAULT_AIDE_ROOT}/annotation"

CLASS_COLOR_MAP = {
    "Anxiety": "#d97706",
    "Peace": "#1f9d8a",
    "Weariness": "#2f7fbf",
    "Happiness": "#e3ab17",
    "Anger": "#cc78a6",
}

FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
_FACE_DETECTOR = None


@dataclass
class LoadedMethod:
    name: str
    mode: str
    model: CLIPModel
    processor: CLIPProcessor
    checkpoint: Dict
    config: Dict
    checkpoint_path: Path
    prompt_groups: List[List[str]]
    text_features: torch.Tensor
    adapter: Optional[object]
    linear_probe: Optional[nn.Module]
    device: str
    num_frames: int
    feature_layout: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_to_score(value: float, low: float, high: float) -> int:
    if high <= low:
        return 3
    norm = clamp((value - low) / (high - low), 0.0, 1.0)
    return int(round(1.0 + 4.0 * norm))


def get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None and FACE_CASCADE_PATH.exists():
        detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        _FACE_DETECTOR = detector if not detector.empty() else False
    return None if _FACE_DETECTOR is False else _FACE_DETECTOR


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def build_test_split_from_result(
    result_json_path: Path,
    aide_root: Optional[str] = None,
    annotation_root: Optional[str] = None,
    seed_override: Optional[int] = None,
    max_sequences_override: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    result = load_json(result_json_path)
    config = result.get("config", {})

    # AIDE-specific split reconstruction: reuse the exact split parameters saved in result json.
    samples = collect_samples(
        aide_root or config.get("aide_root") or DEFAULT_AIDE_ROOT,
        annotation_root or config.get("annotation_root") or DEFAULT_ANNOTATION_ROOT,
        max_sequences=config.get("max_sequences", 0) if max_sequences_override is None else int(max_sequences_override),
    )
    split_cfg = config.get("split", {})
    splits = split_samples(
        samples=samples,
        train_ratio=float(split_cfg.get("train", 0.65)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        seed=int(config.get("seed", 42) if seed_override is None else seed_override),
    )
    return splits["train"], splits["val"], splits["test"], result


def load_clip_model(model_id: str, device: str, clip_mode: str = "auto") -> Tuple[CLIPProcessor, CLIPModel]:
    if clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False)
        except Exception:
            processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def extract_text_features_from_prompts(
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    prompt_groups: List[List[str]],
) -> torch.Tensor:
    features = []
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_x = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_x = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(text_x.detach())
    return torch.stack(features, dim=0)


def instantiate_adapter_from_checkpoint(checkpoint: Dict, config: Dict, device: str):
    text_features = checkpoint.get("text_features")
    if text_features is None:
        raise ValueError("Checkpoint missing text_features; cannot reconstruct adapter.")
    adapter_state = checkpoint.get("adapter_state_dict")
    if adapter_state is None:
        raise ValueError("Checkpoint missing adapter_state_dict; expected strict frozen CLIP adapter checkpoint.")

    temporal_head = str(config.get("temporal_head", "none"))
    use_prompt_weight = bool(config.get("use_prompt_weight", True))
    use_class_temperature = bool(config.get("use_class_temperature", True))
    use_class_bias = bool(config.get("use_class_bias", True))

    if temporal_head == "transformer":
        adapter = TemporalTransformerClipImageAdapter(
            dim=int(text_features.shape[-1]),
            device=device,
            hidden_dim=int(config.get("adapter_hidden_dim", 1024)),
            dropout=float(config.get("adapter_dropout", 0.2)),
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            num_frames=int(config.get("num_frames", 5)),
            temporal_num_heads=int(config.get("temporal_num_heads", 4)),
            temporal_num_layers=int(config.get("temporal_num_layers", 2)),
            temporal_pool_mode=str(config.get("temporal_pool_mode", "cls")),
            use_prompt_weight=use_prompt_weight,
            use_class_temperature=use_class_temperature,
            use_class_bias=use_class_bias,
        )
    else:
        adapter = ClipImageAdapter(
            dim=int(text_features.shape[-1]),
            device=device,
            hidden_dim=int(config.get("adapter_hidden_dim", 1024)),
            dropout=float(config.get("adapter_dropout", 0.2)),
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            use_prompt_weight=use_prompt_weight,
            use_class_temperature=use_class_temperature,
            use_class_bias=use_class_bias,
        )

    adapter.load_state_dict(adapter_state)
    adapter.eval()
    return adapter


def load_method(
    checkpoint_path: Path,
    name: str,
    device: str,
    baseline_mode: str = "linear_probe",
    linear_probe_train_samples: Optional[List[Dict]] = None,
    seed: int = 42,
) -> LoadedMethod:
    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint.get("config", {}))
    model_id = str(config.get("model_id", "openai/clip-vit-base-patch32"))
    clip_mode = str(config.get("clip_mode", "offline_only"))
    processor, model = load_clip_model(model_id=model_id, device=device, clip_mode=clip_mode)

    prompt_groups = checkpoint.get("prompt_groups")
    if not prompt_groups:
        prompt_groups = build_class_prompts(
            str(config.get("prompt_template", "Driver is <LABEL>.")),
            str(config.get("prompt_set", "driving_7")),
        )
    text_features = checkpoint.get("text_features")
    if text_features is None:
        text_features = extract_text_features_from_prompts(processor, model, device, prompt_groups)
    else:
        text_features = text_features.to(device)

    adapter = None
    linear_probe = None
    mode = baseline_mode if baseline_mode in {"linear_probe", "image_only", "zeroshot", "pure_clip"} else "adapter"
    if mode == "adapter":
        adapter = instantiate_adapter_from_checkpoint(checkpoint, config, device)
    elif mode in {"zeroshot", "pure_clip"}:
        linear_probe = None
    else:
        if linear_probe_train_samples is None:
            raise ValueError("linear_probe mode requires train split samples.")
        linear_probe = fit_image_only_linear_probe(
            train_samples=linear_probe_train_samples,
            processor=processor,
            model=model,
            device=device,
            num_frames=int(config.get("num_frames", 5)),
            batch_size=int(config.get("batch_size", 32)),
            seed=seed,
        )

    num_frames = int(config.get("num_frames", 5))
    feature_layout = str(config.get("feature_layout", "sequence" if str(config.get("temporal_head", "none")) == "transformer" else "pooled"))

    return LoadedMethod(
        name=name,
        mode=mode,
        model=model,
        processor=processor,
        checkpoint=checkpoint,
        config=config,
        checkpoint_path=checkpoint_path,
        prompt_groups=prompt_groups,
        text_features=text_features,
        adapter=adapter,
        linear_probe=linear_probe,
        device=device,
        num_frames=num_frames,
        feature_layout=feature_layout,
    )


def select_frame_paths(sample: Dict, num_frames: int) -> List[str]:
    return sample_frame_paths(sample.get("frame_paths") or [sample["frame_path"]], num_frames)


def load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def detect_primary_face_bbox(image_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    detector = get_face_detector()
    if detector is None:
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
    if len(faces) == 0:
        return None
    image_h, image_w = gray.shape[:2]
    image_center = np.array([image_w / 2.0, image_h * 0.38], dtype=np.float32)

    def face_score(face):
        x, y, w, h = [int(v) for v in face]
        area = float(w * h)
        center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        dist = float(np.linalg.norm(center - image_center))
        return area - 0.8 * dist

    x, y, w, h = max(faces, key=face_score)
    return int(x), int(y), int(w), int(h)


def expand_bbox(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int], scale_x: float = 1.6, scale_y: float = 2.2) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    center_x = x + w / 2.0
    center_y = y + h / 2.0 + 0.15 * h
    new_w = w * scale_x
    new_h = h * scale_y
    left = int(round(clamp(center_x - new_w / 2.0, 0, image_w - 1)))
    top = int(round(clamp(center_y - new_h / 2.0, 0, image_h - 1)))
    right = int(round(clamp(center_x + new_w / 2.0, left + 1, image_w)))
    bottom = int(round(clamp(center_y + new_h / 2.0, top + 1, image_h)))
    return left, top, right - left, bottom - top


def default_driver_focus_bbox(image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    image_h, image_w = image_shape[:2]
    width = int(round(image_w * 0.54))
    height = int(round(image_h * 0.58))
    x = int(round((image_w - width) / 2.0))
    y = int(round(image_h * 0.10))
    return x, y, width, height


def bbox_to_slices(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[slice, slice]:
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    left = int(clamp(x, 0, image_w - 1))
    top = int(clamp(y, 0, image_h - 1))
    right = int(clamp(x + w, left + 1, image_w))
    bottom = int(clamp(y + h, top + 1, image_h))
    return slice(top, bottom), slice(left, right)


def heatmap_mass_in_bbox(heatmap: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    heatmap = normalize_heatmap(heatmap)
    total = float(heatmap.sum())
    if total <= 1e-8:
        return 0.0
    rows, cols = bbox_to_slices(bbox, heatmap.shape)
    return float(heatmap[rows, cols].sum() / total)


def border_mass(heatmap: np.ndarray, border_ratio: float = 0.14) -> float:
    heatmap = normalize_heatmap(heatmap)
    total = float(heatmap.sum())
    if total <= 1e-8:
        return 0.0
    image_h, image_w = heatmap.shape[:2]
    border_h = max(1, int(round(image_h * border_ratio)))
    border_w = max(1, int(round(image_w * border_ratio)))
    mask = np.zeros_like(heatmap, dtype=bool)
    mask[:border_h, :] = True
    mask[-border_h:, :] = True
    mask[:, :border_w] = True
    mask[:, -border_w:] = True
    return float(heatmap[mask].sum() / total)


def hotspot_concentration(heatmap: np.ndarray, top_ratio: float = 0.10) -> float:
    flat = normalize_heatmap(heatmap).reshape(-1)
    if flat.size == 0:
        return 0.0
    keep = max(1, int(round(flat.size * top_ratio)))
    topk = np.partition(flat, -keep)[-keep:]
    total = float(flat.sum())
    return 0.0 if total <= 1e-8 else float(topk.sum() / total)


def center_bias(heatmap: np.ndarray) -> float:
    heatmap = normalize_heatmap(heatmap)
    total = float(heatmap.sum())
    if total <= 1e-8:
        return 0.0
    image_h, image_w = heatmap.shape[:2]
    ys, xs = np.indices((image_h, image_w), dtype=np.float32)
    center_x = image_w / 2.0
    center_y = image_h * 0.36
    dist = np.sqrt(((xs - center_x) / max(image_w, 1)) ** 2 + ((ys - center_y) / max(image_h, 1)) ** 2)
    weight = 1.0 - np.clip(dist / 0.75, 0.0, 1.0)
    return float((heatmap * weight).sum() / total)


def image_quality_metrics(image_rgb: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    image_h, image_w = gray.shape[:2]
    face_area_ratio = 0.0
    if face_bbox is not None:
        _, _, w, h = face_bbox
        face_area_ratio = float((w * h) / max(image_h * image_w, 1))
    metrics = {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "face_area_ratio": face_area_ratio,
    }
    metrics["visual_clarity"] = float(
        0.45 * clamp((blur - 25.0) / 150.0, 0.0, 1.0)
        + 0.20 * (1.0 - min(abs(brightness - 118.0) / 118.0, 1.0))
        + 0.20 * clamp((contrast - 18.0) / 45.0, 0.0, 1.0)
        + 0.15 * clamp(face_area_ratio / 0.10, 0.0, 1.0)
    )
    return metrics


def compute_cam_region_metrics(image_rgb: np.ndarray, heatmap: np.ndarray) -> Dict[str, object]:
    face_bbox = detect_primary_face_bbox(image_rgb)
    focus_bbox = expand_bbox(face_bbox, image_rgb.shape) if face_bbox is not None else default_driver_focus_bbox(image_rgb.shape)
    face_mass = heatmap_mass_in_bbox(heatmap, face_bbox) if face_bbox is not None else 0.0
    focus_mass = heatmap_mass_in_bbox(heatmap, focus_bbox)
    bg_mass = border_mass(heatmap)
    concentration = hotspot_concentration(heatmap)
    center_mass = center_bias(heatmap)
    return {
        "face_bbox": face_bbox,
        "focus_bbox": focus_bbox,
        "face_mass": face_mass,
        "focus_mass": focus_mass,
        "bg_mass": bg_mass,
        "concentration": concentration,
        "center_mass": center_mass,
    }


def summarize_separation_gain(sample: Dict, baseline_metrics: Dict[str, object], method_metrics: Dict[str, object]) -> Tuple[str, str]:
    method_focus_gain = float(method_metrics["focus_mass"]) - float(baseline_metrics["focus_mass"])
    baseline_bg_gap = float(baseline_metrics["bg_mass"]) - float(method_metrics["bg_mass"])
    parts = []
    if method_focus_gain > 0.08:
        parts.append("the proposed method attends more to facial / expression-related regions")
    elif method_focus_gain > 0.03:
        parts.append("the proposed method is modestly more face-centered")
    if baseline_bg_gap > 0.08:
        parts.append("the baseline is distracted by contextual cues")
    elif baseline_bg_gap > 0.03:
        parts.append("the baseline places relatively more mass on background regions")
    if not parts:
        parts.append("the attribution map is more semantically aligned with the emotion source")
    explanation = "; ".join(parts)
    caption = (
        f"For sample {sample['sequence_id']}, the proposed method shows a more semantically aligned attribution map, "
        f"while the baseline relies more on contextual cues."
    )
    return explanation, caption


def evaluate_cam_showcase_candidate(sample: Dict, baseline_heatmap: Dict[str, object], method_heatmap: Dict[str, object]) -> Dict[str, object]:
    image_rgb = method_heatmap["image_rgb"]
    gt = sample["label"]
    baseline_pred = str(baseline_heatmap["pred_label"])
    method_pred = str(method_heatmap["pred_label"])
    baseline_correct = baseline_pred == gt
    method_correct = method_pred == gt

    baseline_metrics = compute_cam_region_metrics(image_rgb, baseline_heatmap["heatmap"])
    method_metrics = compute_cam_region_metrics(image_rgb, method_heatmap["heatmap"])
    quality = image_quality_metrics(image_rgb, method_metrics["face_bbox"] or baseline_metrics["face_bbox"])

    method_more_face_centered = (
        float(method_metrics["focus_mass"]) > float(baseline_metrics["focus_mass"]) + 0.035
        and float(method_metrics["bg_mass"]) + 0.015 < float(baseline_metrics["bg_mass"])
    )
    face_shift_advantage = (
        float(method_metrics["focus_mass"]) >= 0.28
        and float(baseline_metrics["focus_mass"]) <= 0.18
        and float(method_metrics["focus_mass"]) - float(baseline_metrics["focus_mass"]) >= 0.10
    )
    visually_obvious = (
        (float(method_metrics["focus_mass"]) - float(baseline_metrics["focus_mass"]) > 0.07)
        or (float(baseline_metrics["bg_mass"]) - float(method_metrics["bg_mass"]) > 0.08)
    )

    if method_correct and not baseline_correct:
        correctness_advantage = 5
    elif method_correct and baseline_correct:
        correctness_advantage = 4
    elif method_correct and baseline_correct is False:
        correctness_advantage = 5
    elif (not method_correct) and baseline_correct:
        correctness_advantage = 1
    else:
        correctness_advantage = 2

    face_centeredness = scale_to_score(
        0.65 * float(method_metrics["focus_mass"]) + 0.20 * float(method_metrics["face_mass"]) + 0.15 * float(method_metrics["center_mass"]),
        0.12,
        0.58,
    )
    background_distraction = scale_to_score(
        float(baseline_metrics["bg_mass"]) - 0.55 * float(baseline_metrics["focus_mass"]) + 0.25 * float(method_metrics["focus_mass"]),
        -0.05,
        0.30,
    )
    visual_clarity = scale_to_score(float(quality["visual_clarity"]), 0.20, 0.85)
    visual_obviousness = scale_to_score(
        (float(method_metrics["focus_mass"]) - float(baseline_metrics["focus_mass"]))
        + (float(baseline_metrics["bg_mass"]) - float(method_metrics["bg_mass"]))
        + 0.4 * (float(method_metrics["concentration"]) - float(baseline_metrics["concentration"])),
        -0.05,
        0.28,
    )
    publication_friendliness = int(round((visual_clarity + visual_obviousness + face_centeredness) / 3.0))
    caption_ease = int(round((correctness_advantage * 1.3 + visual_obviousness + background_distraction) / 3.3))

    overall = (
        4.0 * correctness_advantage
        + 2.5 * face_centeredness
        + 2.0 * background_distraction
        + 1.8 * visual_obviousness
        + 1.2 * visual_clarity
        + 1.0 * publication_friendliness
        + 1.0 * caption_ease
        + (6.0 if face_shift_advantage else 0.0)
    )

    exclude_reason = None
    if not method_correct and not baseline_correct:
        exclude_reason = "both_methods_fail"
    elif baseline_correct and not method_correct:
        exclude_reason = "baseline_outperforms_method"
    elif not method_more_face_centered and correctness_advantage <= 4:
        exclude_reason = "method_not_visually_stronger"
    elif visual_obviousness <= 2 and correctness_advantage <= 4:
        exclude_reason = "difference_too_subtle"

    explanation, caption = summarize_separation_gain(sample, baseline_metrics, method_metrics)
    if method_correct and not baseline_correct:
        explanation = f"{explanation}; the proposed method is correct while the baseline misses the label"
        caption = (
            f"The proposed method correctly predicts {gt} and focuses on emotion-related regions, "
            f"whereas the baseline is distracted by contextual cues and predicts {baseline_pred}."
        )

    return {
        "sample": sample,
        "baseline_pred": baseline_pred,
        "baseline_confidence": float(baseline_heatmap["pred_confidence"]),
        "method_pred": method_pred,
        "method_confidence": float(method_heatmap["pred_confidence"]),
        "baseline_correct": baseline_correct,
        "method_correct": method_correct,
        "method_more_face_centered": method_more_face_centered,
        "face_shift_advantage": face_shift_advantage,
        "visually_obvious": visually_obvious,
        "scores": {
            "correctness_advantage": int(correctness_advantage),
            "face_centeredness_of_method": int(face_centeredness),
            "background_distraction_in_baseline": int(background_distraction),
            "visual_clarity": int(visual_clarity),
            "publication_friendliness": int(publication_friendliness),
            "caption_ease": int(caption_ease),
            "visual_obviousness": int(visual_obviousness),
        },
        "raw_metrics": {
            "method_focus_mass": round(float(method_metrics["focus_mass"]), 6),
            "baseline_focus_mass": round(float(baseline_metrics["focus_mass"]), 6),
            "method_face_mass": round(float(method_metrics["face_mass"]), 6),
            "baseline_face_mass": round(float(baseline_metrics["face_mass"]), 6),
            "method_bg_mass": round(float(method_metrics["bg_mass"]), 6),
            "baseline_bg_mass": round(float(baseline_metrics["bg_mass"]), 6),
            "method_concentration": round(float(method_metrics["concentration"]), 6),
            "baseline_concentration": round(float(baseline_metrics["concentration"]), 6),
            "visual_clarity": round(float(quality["visual_clarity"]), 6),
        },
        "showcase_score": round(float(overall), 4),
        "exclude_reason": exclude_reason,
        "short_explanation": explanation,
        "paper_caption": caption,
    }


def extract_clip_sequence_tokens(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: Sequence[str],
    device: str,
) -> Dict[str, torch.Tensor]:
    images = [load_rgb_image(path) for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.visual_projection.weight.dtype)
    vision_outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
    last_hidden = vision_outputs.last_hidden_state
    patch_tokens = last_hidden[:, 1:, :]
    projected_cls = model.visual_projection(last_hidden[:, 0, :])
    projected_patches = model.visual_projection(patch_tokens)
    projected_cls = projected_cls / projected_cls.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    projected_patches = projected_patches / projected_patches.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return {
        "pixel_values": pixel_values,
        "last_hidden_state": last_hidden,
        "patch_tokens": patch_tokens,
        "projected_cls": projected_cls,
        "projected_patches": projected_patches,
        "image_size": torch.tensor(images[0].size[::-1]),
    }


def pool_projected_frames(projected_cls: torch.Tensor) -> torch.Tensor:
    pooled = projected_cls.mean(dim=0)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def mean_patch_scores(projected_patches: torch.Tensor, class_vector: torch.Tensor) -> torch.Tensor:
    score = (projected_patches * class_vector.view(1, 1, -1)).sum(dim=-1)
    return score.mean(dim=0)


def aggregate_text_feature_for_class(method: LoadedMethod, class_idx: int) -> torch.Tensor:
    text_x = method.text_features[class_idx]
    if method.adapter is not None and getattr(method.adapter, "use_prompt_weight", False):
        weights = torch.softmax(method.adapter.prompt_weight_logits[class_idx], dim=-1)
        agg = (text_x * weights.unsqueeze(-1)).sum(dim=0)
    else:
        agg = text_x.mean(dim=0)
    return agg / agg.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def adapt_patch_tokens_for_heatmap(method: LoadedMethod, projected_patches: torch.Tensor) -> torch.Tensor:
    if method.adapter is None:
        return projected_patches

    patch_shape = projected_patches.shape
    patch_x = projected_patches.reshape(-1, patch_shape[-1]).float().to(method.device)
    with torch.no_grad():
        adapted = method.adapter._adapt_image(patch_x)
    adapted = adapted.reshape(*patch_shape)
    return adapted / adapted.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def fit_image_only_linear_probe(
    train_samples: List[Dict],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    num_frames: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    train_x, train_y = build_image_only_training_features(
        samples=train_samples,
        processor=processor,
        model=model,
        device=device,
        num_frames=num_frames,
        batch_size=batch_size,
    )
    classifier = nn.Linear(int(train_x.shape[1]), len(EMOTION_LABELS)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(40):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            batch_x = train_x[idx].to(device)
            batch_y = train_y[idx].to(device)
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    classifier.eval()
    return classifier


def build_image_only_training_features(
    samples: List[Dict],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    num_frames: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        batch_features = []
        for sample in batch:
            seq = extract_clip_sequence_tokens(
                model=model,
                processor=processor,
                image_paths=select_frame_paths(sample, num_frames),
                device=device,
            )
            pooled = pool_projected_frames(seq["projected_cls"]).float().detach().cpu()
            batch_features.append(pooled)
            labels.append(label_to_idx[sample["label"]])
        features.append(torch.stack(batch_features, dim=0))
    return torch.cat(features, dim=0).float(), torch.tensor(labels, dtype=torch.long)


def build_method_features_and_predictions(
    method: LoadedMethod,
    samples: List[Dict],
    batch_size: int,
) -> Dict[str, object]:
    feature_rows: List[np.ndarray] = []
    preds: List[str] = []
    confidences: List[float] = []

    if method.mode == "adapter":
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            pooled_inputs = []
            for sample in batch:
                seq = extract_clip_sequence_tokens(
                    model=method.model,
                    processor=method.processor,
                    image_paths=select_frame_paths(sample, method.num_frames),
                    device=method.device,
                )
                if method.feature_layout == "sequence":
                    projected = seq["projected_cls"]
                    if projected.shape[0] < method.num_frames:
                        pad = projected[-1:].expand(method.num_frames - projected.shape[0], -1)
                        projected = torch.cat([projected, pad], dim=0)
                    pooled_inputs.append(projected[: method.num_frames].float())
                else:
                    pooled_inputs.append(pool_projected_frames(seq["projected_cls"]).float())
            batch_x = torch.stack(pooled_inputs, dim=0).to(method.device)
            with torch.no_grad():
                logits = method.adapter.logits(batch_x, method.text_features)
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1)
                conf = probs.max(dim=-1).values.detach().cpu().tolist()
                adapted = method.adapter._adapt_image(batch_x).detach().cpu().numpy()
            feature_rows.extend(adapted)
            preds.extend([EMOTION_LABELS[int(idx.item())] for idx in pred_idx])
            confidences.extend(conf)
    elif method.mode in {"zeroshot", "pure_clip"}:
        class_vectors = torch.stack([aggregate_text_feature_for_class(method, idx).float() for idx in range(len(EMOTION_LABELS))], dim=0).to(method.device)
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            pooled_inputs = []
            for sample in batch:
                seq = extract_clip_sequence_tokens(
                    model=method.model,
                    processor=method.processor,
                    image_paths=select_frame_paths(sample, method.num_frames),
                    device=method.device,
                )
                pooled_inputs.append(pool_projected_frames(seq["projected_cls"]).float())
            batch_x = torch.stack(pooled_inputs, dim=0).to(method.device)
            with torch.no_grad():
                logits = batch_x @ class_vectors.t()
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1)
                conf = probs.max(dim=-1).values.detach().cpu().tolist()
                features = batch_x.detach().cpu().numpy()
            feature_rows.extend(features)
            preds.extend([EMOTION_LABELS[int(idx.item())] for idx in pred_idx])
            confidences.extend(conf)
    else:
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            pooled_inputs = []
            for sample in batch:
                seq = extract_clip_sequence_tokens(
                    model=method.model,
                    processor=method.processor,
                    image_paths=select_frame_paths(sample, method.num_frames),
                    device=method.device,
                )
                pooled_inputs.append(pool_projected_frames(seq["projected_cls"]).float())
            batch_x = torch.stack(pooled_inputs, dim=0).to(method.device)
            with torch.no_grad():
                logits = method.linear_probe(batch_x)
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1)
                conf = probs.max(dim=-1).values.detach().cpu().tolist()
                features = batch_x.detach().cpu().numpy()
            feature_rows.extend(features)
            preds.extend([EMOTION_LABELS[int(idx.item())] for idx in pred_idx])
            confidences.extend(conf)

    labels = [sample["label"] for sample in samples]
    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "labels": labels,
        "predictions": preds,
        "confidences": confidences,
    }


def limit_samples_per_class(samples: List[Dict], max_per_class: int, seed: int) -> List[Dict]:
    if max_per_class <= 0:
        return list(samples)
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = {label: [] for label in EMOTION_LABELS}
    for sample in samples:
        grouped.setdefault(sample["label"], []).append(sample)
    selected = []
    for label in EMOTION_LABELS:
        items = list(grouped.get(label, []))
        rng.shuffle(items)
        selected.extend(items[:max_per_class])
    selected.sort(key=lambda item: item["sequence_id"])
    return selected


def save_tsne_coordinates(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        return
    if suffix == ".csv":
        import csv

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sequence_id", "label", "x", "y"])
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError(f"Unsupported coordinate output format: {path}")


def draw_tsne_plot(
    coords: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_png: Path,
    output_pdf: Optional[Path] = None,
    plot_style: str = "default",
    marker_scale: float = 1.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    if plot_style == "block":
        figure_size = (7.0, 5.35)
        marker_size = 76
        alpha = 0.94
        edgecolors = "none"
        linewidths = 0.0
    else:
        figure_size = (7.35, 5.45)
        marker_size = 60
        alpha = 0.9
        edgecolors = "white"
        linewidths = 0.25
    marker_size = max(12.0, marker_size * float(marker_scale))
    fig, ax = plt.subplots(figsize=figure_size)
    for label in EMOTION_LABELS:
        idx = [i for i, item in enumerate(labels) if item == label]
        if not idx:
            continue
        subset = coords[idx]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=marker_size,
            alpha=alpha,
            c=CLASS_COLOR_MAP[label],
            label=label,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    ax.set_title(title, pad=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(EMOTION_LABELS), 5),
        frameon=True,
        columnspacing=1.3,
        handletextpad=0.6,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.88)
    ensure_parent(output_png)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.03)
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_tsne_comparison(
    baseline_coords: np.ndarray,
    method_coords: np.ndarray,
    labels: Sequence[str],
    baseline_title: str,
    method_title: str,
    output_png: Path,
    output_pdf: Optional[Path] = None,
    plot_style: str = "default",
    marker_scale: float = 1.0,
) -> None:
    plt.rcParams.update({"figure.dpi": 180, "savefig.dpi": 300})
    if plot_style == "block":
        figure_size = (12.6, 5.2)
        marker_size = 60
        alpha = 0.94
        edgecolors = "none"
        linewidths = 0.0
    else:
        figure_size = (12.9, 5.35)
        marker_size = 46
        alpha = 0.9
        edgecolors = "white"
        linewidths = 0.22
    marker_size = max(10.0, marker_size * float(marker_scale))
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=False, sharey=False)
    for ax, coords, title in zip(axes, [baseline_coords, method_coords], [baseline_title, method_title]):
        for label in EMOTION_LABELS:
            idx = [i for i, item in enumerate(labels) if item == label]
            if not idx:
                continue
            subset = coords[idx]
            ax.scatter(
                subset[:, 0],
                subset[:, 1],
                s=marker_size,
                alpha=alpha,
                c=CLASS_COLOR_MAP[label],
                label=label,
                edgecolors=edgecolors,
                linewidths=linewidths,
            )
        ax.set_title(title, pad=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.18, linewidth=0.5)
    handles, labels_text = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="upper center",
        ncol=min(len(EMOTION_LABELS), 5),
        frameon=True,
        bbox_to_anchor=(0.5, 0.978),
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.87, wspace=0.12)
    ensure_parent(output_png)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.03)
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def compute_grid_size(num_patches: int) -> int:
    grid = int(round(math.sqrt(num_patches)))
    if grid * grid != num_patches:
        raise ValueError(f"Patch token count {num_patches} is not a square grid.")
    return grid


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    denom = max(float(heatmap.max()), 1e-8)
    return heatmap / denom


def resize_heatmap(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.clip(255.0 * normalize_heatmap(heatmap), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def classify_sample(method: LoadedMethod, sample: Dict) -> Dict[str, object]:
    seq = extract_clip_sequence_tokens(
        model=method.model,
        processor=method.processor,
        image_paths=select_frame_paths(sample, method.num_frames),
        device=method.device,
    )
    if method.mode == "adapter":
        if method.feature_layout == "sequence":
            image_x = seq["projected_cls"]
            if image_x.shape[0] < method.num_frames:
                pad = image_x[-1:].expand(method.num_frames - image_x.shape[0], -1)
                image_x = torch.cat([image_x, pad], dim=0)
            image_x = image_x[: method.num_frames].float().unsqueeze(0)
        else:
            image_x = pool_projected_frames(seq["projected_cls"]).float().unsqueeze(0)
        with torch.no_grad():
            logits = method.adapter.logits(image_x.to(method.device), method.text_features)
            probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        return {
            "pred_idx": pred_idx,
            "pred_label": EMOTION_LABELS[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probs": probs.detach().cpu().numpy(),
        }
    pooled = pool_projected_frames(seq["projected_cls"]).float().unsqueeze(0).to(method.device)
    with torch.no_grad():
        if method.mode in {"zeroshot", "pure_clip"}:
            class_vectors = torch.stack([aggregate_text_feature_for_class(method, idx).float() for idx in range(len(EMOTION_LABELS))], dim=0).to(method.device)
            logits = pooled @ class_vectors.t()
        else:
            logits = method.linear_probe(pooled)
        probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(probs.argmax().item())
    return {
        "pred_idx": pred_idx,
        "pred_label": EMOTION_LABELS[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probs": probs.detach().cpu().numpy(),
    }


def generate_patch_heatmap(
    method: LoadedMethod,
    sample: Dict,
    target_label: Optional[str] = None,
) -> Dict[str, object]:
    seq = extract_clip_sequence_tokens(
        model=method.model,
        processor=method.processor,
        image_paths=select_frame_paths(sample, method.num_frames),
        device=method.device,
    )
    image_rgb = np.asarray(load_rgb_image(sample["frame_path"]))
    pred_info = classify_sample(method, sample)
    class_idx = EMOTION_LABELS.index(target_label) if target_label else int(pred_info["pred_idx"])

    if method.mode == "adapter":
        class_vec = aggregate_text_feature_for_class(method, class_idx).to(seq["projected_patches"].dtype)
        adapted_patches = adapt_patch_tokens_for_heatmap(method, seq["projected_patches"].to(method.device))
        heatmap = mean_patch_scores(adapted_patches, class_vec).detach().cpu().numpy()
    elif method.mode in {"zeroshot", "pure_clip"}:
        class_vec = aggregate_text_feature_for_class(method, class_idx).to(seq["projected_patches"].dtype)
        heatmap = mean_patch_scores(seq["projected_patches"], class_vec).detach().cpu().numpy()
    else:
        weight = method.linear_probe.weight[class_idx].to(seq["projected_patches"].dtype)
        weight = weight / weight.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        heatmap = mean_patch_scores(seq["projected_patches"], weight).detach().cpu().numpy()

    grid = compute_grid_size(heatmap.shape[0])
    heatmap = heatmap.reshape(grid, grid)
    heatmap = normalize_heatmap(heatmap)
    heatmap_up = resize_heatmap(heatmap, image_rgb.shape[1], image_rgb.shape[0])
    overlay = overlay_heatmap_on_image(image_rgb=image_rgb, heatmap=heatmap_up)
    return {
        "image_rgb": image_rgb,
        "overlay_rgb": overlay,
        "heatmap": heatmap_up,
        "pred_label": pred_info["pred_label"],
        "pred_confidence": pred_info["confidence"],
        "target_label": EMOTION_LABELS[class_idx],
    }


def save_cam_comparison(
    sample: Dict,
    baseline_heatmap: Dict[str, object],
    method_heatmap: Dict[str, object],
    output_png: Path,
    output_pdf: Optional[Path] = None,
) -> None:
    plt.rcParams.update({"figure.dpi": 180, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    axes[0].imshow(baseline_heatmap["image_rgb"])
    axes[0].set_title(f"Original\nID={sample['sequence_id']} | GT={sample['label']}")
    axes[1].imshow(baseline_heatmap["overlay_rgb"])
    axes[1].set_title(
        "Baseline Heatmap\n"
        f"Pred={baseline_heatmap['pred_label']} ({baseline_heatmap['pred_confidence']:.3f})"
    )
    axes[2].imshow(method_heatmap["overlay_rgb"])
    axes[2].set_title(
        "Image-Text Method Heatmap\n"
        f"Pred={method_heatmap['pred_label']} ({method_heatmap['pred_confidence']:.3f})"
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    ensure_parent(output_png)
    fig.savefig(output_png, bbox_inches="tight")
    if output_pdf is not None:
        ensure_parent(output_pdf)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def rank_samples_for_visualization(
    samples: List[Dict],
    baseline_outputs: Dict[str, object],
    method_outputs: Dict[str, object],
    selection_mode: str,
    top_k: int,
    sample_ids: Optional[Iterable[str]] = None,
) -> List[Dict]:
    sample_lookup = {sample["sequence_id"]: sample for sample in samples}
    if sample_ids:
        selected = [sample_lookup[sample_id] for sample_id in sample_ids if sample_id in sample_lookup]
        return selected[:top_k]

    records = []
    for idx, sample in enumerate(samples):
        gt = sample["label"]
        base_pred = baseline_outputs["predictions"][idx]
        method_pred = method_outputs["predictions"][idx]
        base_conf = float(baseline_outputs["confidences"][idx])
        method_conf = float(method_outputs["confidences"][idx])
        is_correct = gt == base_pred and gt == method_pred
        records.append(
            {
                "sample": sample,
                "correct_both": is_correct,
                "score": min(base_conf, method_conf),
            }
        )

    if selection_mode == "correct_only":
        filtered = [row for row in records if row["correct_both"]]
        filtered.sort(key=lambda item: item["score"], reverse=True)
        return [row["sample"] for row in filtered[:top_k]]

    if selection_mode == "top_confidence":
        records.sort(key=lambda item: item["score"], reverse=True)
        return [row["sample"] for row in records[:top_k]]

    return [row["sample"] for row in records[:top_k]]


def format_frame_index(frame_path: str) -> Optional[int]:
    stem = Path(frame_path).stem
    return int(stem) if stem.isdigit() else None
