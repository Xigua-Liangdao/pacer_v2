import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
from PIL import Image


EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_DISPLAY = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "joy": "joyful",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_DISPLAY = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".ckpt.pt")


def clone_module_state_dict(module) -> Dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def weighted_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    weighted = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        weighted += (support.get(label, 0) / total) * f1
    return weighted


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in matrix and pred_label in matrix[true_label]:
            matrix[true_label][pred_label] += 1
    return matrix


def evaluate_split(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, labels), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
    }


def normalize_emotion_label(label: str) -> Optional[str]:
    value = str(label).strip().lower()
    return value if value in EMOTION_LABELS else None


def normalize_sentiment_label(label: str) -> Optional[str]:
    value = str(label).strip().lower()
    if value == "positive":
        return "positive"
    if value == "negative":
        return "negative"
    if value == "neutral":
        return "neutral"
    return None


def get_task_labels(task_mode: str) -> List[str]:
    if task_mode == "emotion":
        return EMOTION_LABELS
    if task_mode == "sentiment":
        return SENTIMENT_LABELS
    raise ValueError(f"Unsupported task_mode: {task_mode}")


def get_task_display_map(task_mode: str) -> Dict[str, str]:
    if task_mode == "emotion":
        return EMOTION_DISPLAY
    if task_mode == "sentiment":
        return SENTIMENT_DISPLAY
    raise ValueError(f"Unsupported task_mode: {task_mode}")


def build_prompt_templates(prompt_set: str, task_mode: str) -> List[str]:
    if prompt_set == "single":
        return ["The speaker looks <LABEL>."]
    if prompt_set == "meld_sentiment_7":
        return [
            "The speaker looks <LABEL>.",
            "The visible sentiment is <LABEL>.",
            "This dialogue clip has a <LABEL> tone.",
            "The speaker's expression appears <LABEL>.",
            "The video shows a <LABEL> reaction.",
            "The person on screen gives a <LABEL> impression.",
            "The emotional polarity is <LABEL>.",
        ]
    if prompt_set == "meld_7":
        return [
            "The speaker looks <LABEL>.",
            "The facial expression is <LABEL>.",
            "This dialogue clip shows a <LABEL> person.",
            "The visible emotion is <LABEL>.",
            "The person on screen appears <LABEL>.",
            "Emotion in this video clip: <LABEL>.",
            "The speaker's expression is <LABEL>.",
        ]
    if prompt_set == "meld_scene_9":
        return [
            "The speaker looks <LABEL>.",
            "The face in this dialogue scene is <LABEL>.",
            "This conversation clip shows a <LABEL> reaction.",
            "The visible emotion is <LABEL>.",
            "The person on screen appears <LABEL>.",
            "The speaker's expression is <LABEL>.",
            "The emotional state in this video is <LABEL>.",
            "The dialogue partner looks <LABEL>.",
            "This utterance is delivered with a <LABEL> expression.",
        ]
    if task_mode == "sentiment":
        return [
            "The speaker looks <LABEL>.",
            "The visible sentiment is <LABEL>.",
            "This clip feels <LABEL>.",
        ]
    custom = [item.strip() for item in prompt_set.split("||") if item.strip()]
    return custom if custom else ["The speaker looks <LABEL>."]


def build_class_prompts(prompt_set: str, task_mode: str) -> List[List[str]]:
    templates = build_prompt_templates(prompt_set, task_mode)
    labels = get_task_labels(task_mode)
    display_map = get_task_display_map(task_mode)
    return [[template.replace("<LABEL>", display_map[label]) for template in templates] for label in labels]


def build_prompt_group_indices(num_prompts: int, group_size: int) -> List[List[int]]:
    if group_size <= 0 or group_size >= num_prompts:
        return [list(range(num_prompts))]
    groups = []
    for start in range(0, num_prompts, group_size):
        groups.append(list(range(start, min(start + group_size, num_prompts))))
    return groups


def load_clip_processor_and_model(model_id: str, device: str, clip_mode: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

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

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


def index_video_files(video_root: str) -> Dict[str, str]:
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root not found: {root}")

    mapping: Dict[str, str] = {}
    for path in root.rglob("*.mp4"):
        mapping.setdefault(path.name, str(path))
    return mapping


def build_video_name(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"


def collect_split_samples(csv_path: str, video_index: Dict[str, str], split_name: str, task_mode: str) -> Dict[str, object]:
    frame = pd.read_csv(csv_path)
    samples: List[Dict[str, object]] = []
    missing: List[str] = []

    for row in frame.itertuples(index=False):
        if task_mode == "emotion":
            label = normalize_emotion_label(getattr(row, "Emotion", None))
        else:
            label = normalize_sentiment_label(getattr(row, "Sentiment", None))
        if label is None:
            continue

        video_name = build_video_name(getattr(row, "Dialogue_ID"), getattr(row, "Utterance_ID"))
        video_path = video_index.get(video_name)
        if video_path is None:
            missing.append(video_name)
            continue

        samples.append(
            {
                "split": split_name,
                "dialogue_id": int(getattr(row, "Dialogue_ID")),
                "utterance_id": int(getattr(row, "Utterance_ID")),
                "label": label,
                "video_name": video_name,
                "video_path": video_path,
                "utterance": str(getattr(row, "Utterance", "")),
                "speaker": str(getattr(row, "Speaker", "")),
            }
        )

    return {"samples": samples, "missing": missing}


def stratified_subsample(samples: List[Dict[str, object]], max_samples: int, seed: int) -> List[Dict[str, object]]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples

    label_groups: Dict[str, List[Dict[str, object]]] = {}
    for sample in samples:
        label_groups.setdefault(str(sample["label"]), []).append(sample)

    rng = random.Random(seed)
    selected: List[Dict[str, object]] = []
    total = len(samples)
    labels = sorted(label_groups.keys())

    target_counts: Dict[str, int] = {}
    for label in labels:
        group = label_groups[label]
        rng.shuffle(group)
        target = max(1, round(len(group) / total * max_samples))
        target_counts[label] = min(len(group), target)

    current_total = sum(target_counts.values())
    while current_total > max_samples:
        for label in sorted(labels, key=lambda item: target_counts[item], reverse=True):
            if current_total <= max_samples:
                break
            if target_counts[label] > 1:
                target_counts[label] -= 1
                current_total -= 1
    while current_total < max_samples:
        for label in labels:
            if current_total >= max_samples:
                break
            if target_counts[label] < len(label_groups[label]):
                target_counts[label] += 1
                current_total += 1

    for label in labels:
        selected.extend(label_groups[label][:target_counts[label]])
    rng.shuffle(selected)
    return selected


def sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return [0]
    if num_frames <= 1:
        return [max(0, total_frames // 2)]
    if total_frames <= num_frames:
        return list(range(total_frames))
    return [round(index * (total_frames - 1) / (num_frames - 1)) for index in range(num_frames)]


def read_sampled_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = sample_frame_indices(total_frames, num_frames)
    images: List[Image.Image] = []

    for frame_index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))

    if not images:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

    capture.release()

    if not images:
        raise RuntimeError(f"Failed to decode frames from video: {video_path}")
    return images


def extract_image_features(samples: List[Dict], processor, model, device: str, batch_size: int, num_frames: int, tag: str):
    import torch

    features = []
    total_batches = max(1, math.ceil(len(samples) / batch_size))
    start_time = time.time()
    log(f"[FEATURES] start {tag}: samples={len(samples)}, batches={total_batches}, num_frames={num_frames}")

    for batch_index, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch = samples[start:start + batch_size]
        batch_features = []
        for sample in batch:
            frames = read_sampled_frames(sample["video_path"], num_frames)
            inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            pooled = image_features.mean(dim=0)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            batch_features.append(pooled)
        features.append(torch.stack(batch_features, dim=0).float().cpu())

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / batch_index * (total_batches - batch_index)
            log(f"[FEATURES] {tag}: batch {batch_index}/{total_batches}, elapsed={format_duration(elapsed)}, eta={format_duration(eta)}")

    return torch.cat(features, dim=0)


def extract_text_features(prompt_groups: List[List[str]], processor, model, device: str):
    import torch

    class_prompt_features = []
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_prompt_features.append(text_features.float().detach())
    return torch.stack(class_prompt_features, dim=0)


class ClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        init_logit_scale: float,
        init_class_temperature: float,
        class_temperature_min: float,
        class_temperature_max: float,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.class_temperature_min = class_temperature_min
        self.class_temperature_max = class_temperature_max
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(max(init_logit_scale, 1e-6)), device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(
            torch.full((num_classes,), math.log(max(init_class_temperature, 1e-6)), device=device)
        )
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))

    def parameters(self):
        return (
            list(self.input_proj.parameters())
            + list(self.net.parameters())
            + list(self.out_proj.parameters())
            + [self.logit_scale, self.prompt_weight_logits, self.class_logit_scale, self.class_bias]
        )

    def state_dict(self):
        return {
            "input_proj": clone_module_state_dict(self.input_proj),
            "net": clone_module_state_dict(self.net),
            "out_proj": clone_module_state_dict(self.out_proj),
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
        }

    def load_state_dict(self, state):
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))

    def train(self):
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()

    def eval(self):
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()

    def initialize_calibration(self, train_y, labels: List[str], bias_mode: str, temperature_mode: str):
        import torch

        counts = torch.bincount(train_y.detach().cpu(), minlength=len(labels)).float()
        priors = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-6)

        if bias_mode == "neg_log_prior":
            bias = -torch.log(priors)
            bias = bias - bias.mean()
            self.class_bias.data.copy_(bias.to(self.device))
        elif bias_mode == "zero":
            self.class_bias.data.zero_()
        else:
            raise ValueError(f"Unsupported bias_mode: {bias_mode}")

        if temperature_mode == "inverse_prior":
            scale = (priors.mean() / priors).clamp(min=self.class_temperature_min, max=self.class_temperature_max)
            self.class_logit_scale.data.copy_(scale.log().to(self.device))
        elif temperature_mode == "sqrt_inverse_prior":
            scale = (priors.mean() / priors).sqrt().clamp(min=self.class_temperature_min, max=self.class_temperature_max)
            self.class_logit_scale.data.copy_(scale.log().to(self.device))
        elif temperature_mode == "constant":
            pass
        else:
            raise ValueError(f"Unsupported temperature_mode: {temperature_mode}")

    def _adapt_image(self, image_x):
        base = self.input_proj(image_x)
        delta = self.net(base)
        fused = base + delta
        image = self.out_proj(fused)
        return image / image.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def logits(self, image_x, text_x):
        import torch
        import torch.nn.functional as F

        image = self._adapt_image(image_x)
        text = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarity = torch.einsum("bd,cpd->bcp", image, text)
        prompt_weight = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
        class_similarity = (similarity * prompt_weight).sum(dim=-1)
        global_scale = self.logit_scale.exp().clamp(max=100.0)
        class_scale = self.class_logit_scale.exp().clamp(min=self.class_temperature_min, max=self.class_temperature_max).unsqueeze(0)
        return global_scale * class_similarity * class_scale + self.class_bias.unsqueeze(0)

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        image = self._adapt_image(image_x)
        text = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarity = torch.einsum("bd,cpd->bcp", image, text)
        group_scores = []
        for group in group_indices:
            group_scores.append(similarity[:, :, group].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)
        global_scale = self.logit_scale.exp().clamp(max=100.0)
        class_scale = self.class_logit_scale.exp().clamp(min=self.class_temperature_min, max=self.class_temperature_max).view(1, -1, 1)
        return global_scale * scores * class_scale + self.class_bias.view(1, -1, 1)


def predict_from_features(image_features, text_features, adapter, batch_size: int, use_test_ensemble: bool, ensemble_group_size: int, labels: List[str]) -> List[str]:
    import torch

    predictions = []
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)
    for start in range(0, image_features.shape[0], batch_size):
        batch_x = image_features[start:start + batch_size].to(adapter.device)
        with torch.no_grad():
            if use_test_ensemble and len(group_indices) > 1:
                grouped_logits = adapter.grouped_logits(batch_x, text_features.to(adapter.device), group_indices)
                grouped_pred = grouped_logits.argmax(dim=1)
                total_scores = grouped_logits.sum(dim=-1)
                indices = []
                for sample_index in range(grouped_pred.shape[0]):
                    votes = torch.bincount(grouped_pred[sample_index], minlength=len(labels))
                    top_votes = votes.max()
                    candidates = (votes == top_votes).nonzero(as_tuple=False).view(-1)
                    if candidates.numel() == 1:
                        indices.append(int(candidates[0].item()))
                    else:
                        candidate_scores = total_scores[sample_index, candidates]
                        indices.append(int(candidates[candidate_scores.argmax()].item()))
            else:
                logits = adapter.logits(batch_x, text_features.to(adapter.device))
                indices = logits.argmax(dim=-1).detach().cpu().tolist()
        predictions.extend([labels[index] for index in indices])
    return predictions


def train_model(train_x, train_y, val_x, val_y, text_features, adapter, labels: List[str], epochs: int, batch_size: int, lr: float, weight_decay: float, max_grad_norm: float, use_class_weight: bool, label_smoothing: float, select_metric: str, use_test_ensemble: bool, ensemble_group_size: int):
    import torch
    import torch.nn as nn

    class_weights = None
    if use_class_weight:
        class_counts = torch.bincount(train_y, minlength=len(labels)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_metric = -1.0
    overall_start = time.time()

    for epoch_index in range(epochs):
        adapter.train()
        permutation = torch.randperm(train_x.shape[0])
        train_x = train_x[permutation]
        train_y = train_y[permutation]
        losses = []

        for start in range(0, train_x.shape[0], batch_size):
            batch_x = train_x[start:start + batch_size].to(adapter.device)
            batch_y = train_y[start:start + batch_size].to(adapter.device)
            logits = adapter.logits(batch_x, text_features.to(adapter.device))
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        adapter.eval()
        val_pred = predict_from_features(val_x, text_features, adapter, batch_size, use_test_ensemble, ensemble_group_size, labels)
        val_true = [labels[int(index)] for index in val_y.cpu().tolist()]
        val_acc = accuracy(val_true, val_pred)
        val_wf1 = weighted_f1(val_true, val_pred, labels)
        metric = val_wf1 if select_metric == "weighted_f1" else val_acc

        if metric > best_metric:
            best_metric = metric
            best_state = adapter.state_dict()

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_index + 1) * (epochs - epoch_index - 1)
        log(
            f"[TRAIN] epoch={epoch_index + 1} loss={sum(losses) / max(len(losses), 1):.4f} val_acc={val_acc:.4f} val_wf1={val_wf1:.4f} elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
        )

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter


def default_cache_path(feature_cache_dir: str, model_id: str, num_frames: int, split_name: str, csv_path: str, sample_count: int) -> Path:
    cache_root = Path(feature_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"meld_{split_name}_{Path(csv_path).stem}_{model_id}_{num_frames}_{sample_count}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_model_id = model_id.replace("/", "_")
    return cache_root / f"{split_name}_{safe_model_id}_f{num_frames}_{digest}.pt"


def load_or_extract_features(samples: List[Dict], processor, model, device: str, batch_size: int, num_frames: int, tag: str, cache_path: Optional[Path]):
    import torch

    if cache_path is not None and cache_path.exists():
        log(f"[CACHE] loading {tag} features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    features = extract_image_features(samples, processor, model, device, batch_size, num_frames, tag)
    if cache_path is not None:
        torch.save(features, cache_path)
        log(f"[CACHE] saved {tag} features to {cache_path}")
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-based MELD video classification")
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--task_mode", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--prompt_set", default="meld_scene_9")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--init_logit_scale", type=float, default=1.0)
    parser.add_argument("--init_class_temperature", type=float, default=1.0)
    parser.add_argument("--init_bias_mode", choices=["zero", "neg_log_prior"], default="zero")
    parser.add_argument("--init_temperature_mode", choices=["constant", "inverse_prior", "sqrt_inverse_prior"], default="constant")
    parser.add_argument("--class_temperature_min", type=float, default=0.5)
    parser.add_argument("--class_temperature_max", type=float, default=2.5)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", action="store_true")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_dev_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)
    task_labels = get_task_labels(args.task_mode)

    video_index = index_video_files(args.video_root)
    log(f"[DATA] indexed videos: {len(video_index)}")

    train_data = collect_split_samples(args.train_csv, video_index, "train", args.task_mode)
    dev_data = collect_split_samples(args.dev_csv, video_index, "dev", args.task_mode)
    test_data = collect_split_samples(args.test_csv, video_index, "test", args.task_mode)

    train_samples = train_data["samples"]
    dev_samples = dev_data["samples"]
    test_samples = test_data["samples"]

    train_samples = stratified_subsample(train_samples, args.max_train_samples, args.seed + 11)
    dev_samples = stratified_subsample(dev_samples, args.max_dev_samples, args.seed + 23)
    test_samples = stratified_subsample(test_samples, args.max_test_samples, args.seed + 37)

    if not train_samples or not dev_samples or not test_samples:
        raise RuntimeError("At least one MELD split has zero matched samples. Check csv paths and extracted video root.")

    processor, model = load_clip_processor_and_model(args.model_id, args.device, args.clip_mode)
    prompt_groups = build_class_prompts(args.prompt_set, args.task_mode)
    text_features = extract_text_features(prompt_groups, processor, model, args.device)
    clip_dim = int(text_features.shape[-1])

    cache_dir = args.feature_cache_dir
    train_cache = default_cache_path(cache_dir, args.model_id, args.num_frames, "train", args.train_csv, len(train_samples)) if cache_dir else None
    dev_cache = default_cache_path(cache_dir, args.model_id, args.num_frames, "dev", args.dev_csv, len(dev_samples)) if cache_dir else None
    test_cache = default_cache_path(cache_dir, args.model_id, args.num_frames, "test", args.test_csv, len(test_samples)) if cache_dir else None

    train_x = load_or_extract_features(train_samples, processor, model, args.device, args.batch_size, args.num_frames, "train", train_cache)
    dev_x = load_or_extract_features(dev_samples, processor, model, args.device, args.batch_size, args.num_frames, "dev", dev_cache)
    test_x = load_or_extract_features(test_samples, processor, model, args.device, args.batch_size, args.num_frames, "test", test_cache)
    train_y = torch.tensor([task_labels.index(sample["label"]) for sample in train_samples], dtype=torch.long)
    dev_y = torch.tensor([task_labels.index(sample["label"]) for sample in dev_samples], dtype=torch.long)
    test_y = torch.tensor([task_labels.index(sample["label"]) for sample in test_samples], dtype=torch.long)

    adapter = ClipImageAdapter(
        dim=clip_dim,
        device=args.device,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        num_classes=len(task_labels),
        num_prompts=int(text_features.shape[1]),
        init_logit_scale=args.init_logit_scale,
        init_class_temperature=args.init_class_temperature,
        class_temperature_min=args.class_temperature_min,
        class_temperature_max=args.class_temperature_max,
    )
    adapter.initialize_calibration(train_y, task_labels, args.init_bias_mode, args.init_temperature_mode)

    adapter = train_model(
        train_x=train_x.float(),
        train_y=train_y,
        val_x=dev_x.float(),
        val_y=dev_y,
        text_features=text_features,
        adapter=adapter,
        labels=task_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=args.use_class_weight,
        label_smoothing=args.label_smoothing,
        select_metric=args.select_metric,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )

    dev_pred = predict_from_features(dev_x.float(), text_features, adapter, args.batch_size, args.use_test_ensemble, args.ensemble_group_size, task_labels)
    test_pred = predict_from_features(test_x.float(), text_features, adapter, args.batch_size, args.use_test_ensemble, args.ensemble_group_size, task_labels)
    dev_true = [sample["label"] for sample in dev_samples]
    test_true = [sample["label"] for sample in test_samples]

    result = {
        "config": {
            "video_root": str(Path(args.video_root).resolve()),
            "train_csv": str(Path(args.train_csv).resolve()),
            "dev_csv": str(Path(args.dev_csv).resolve()),
            "test_csv": str(Path(args.test_csv).resolve()),
            "task_mode": args.task_mode,
            "model_id": args.model_id,
            "prompt_set": args.prompt_set,
            "num_frames": args.num_frames,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "init_logit_scale": args.init_logit_scale,
            "init_class_temperature": args.init_class_temperature,
            "init_bias_mode": args.init_bias_mode,
            "init_temperature_mode": args.init_temperature_mode,
            "class_temperature_min": args.class_temperature_min,
            "class_temperature_max": args.class_temperature_max,
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "seed": args.seed,
            "active_labels": task_labels,
            "feature_dims": {"clip": clip_dim},
            "feature_cache_dir": str(Path(args.feature_cache_dir).resolve()) if args.feature_cache_dir else None,
        },
        "dataset": {
            "train": len(train_samples),
            "dev": len(dev_samples),
            "test": len(test_samples),
            "train_label_distribution": dict(Counter(sample["label"] for sample in train_samples)),
            "missing_videos": {
                "train": len(train_data["missing"]),
                "dev": len(dev_data["missing"]),
                "test": len(test_data["missing"]),
            },
        },
        "dev": evaluate_split(dev_true, dev_pred, task_labels),
        "test": evaluate_split(test_true, test_pred, task_labels),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = default_checkpoint_path(output_path)

    import torch

    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "config": result["config"],
            "dataset": result["dataset"],
            "labels": task_labels,
        },
        checkpoint_path,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    log(f"[DONE] wrote {output_path}")
    log(f"[DONE] wrote {checkpoint_path}")


if __name__ == "__main__":
    main()