import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional  # QCPA:

import numpy as np
from PIL import Image

import clip_cremad_emotion_train as cremad_base
from prompts import NUM_PROMPTS, PROMPT_GROUP_IDS, build_prompts_for_classes  # QCPA:
from qcpa import QCPAHead  # QCPA:

EMOTION_LABELS = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
EMOTION_NORMALIZE_MAP = {
    "anxiety": "Anxiety",
    "peace": "Peace",
    "weariness": "Weariness",
    "happiness": "Happiness",
    "anger": "Anger",
}
EMOTION_PROMPT_CLASS_NAMES = [label.lower() for label in EMOTION_LABELS]  # QCPA:

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AIDE_ROOT = os.environ.get("AIDE_ROOT", "/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset")
DEFAULT_ANNOTATION_ROOT = os.environ.get("AIDE_ANNOTATION_ROOT", os.path.join(DEFAULT_AIDE_ROOT, "annotation"))
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "clip_emotion_supervised_results.json")


def parse_on_off(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise ValueError(f"Expected 'on' or 'off', got: {value}")


def normalize_emotion_label(label: str) -> str:
    key = str(label).strip().lower()
    return EMOTION_NORMALIZE_MAP.get(key, str(label).strip())


def configure_random_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sorted_frame_paths(folder: str) -> List[str]:
    frames = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

    def frame_key(name: str):
        stem = os.path.splitext(name)[0]
        return (int(stem) if stem.isdigit() else 10**9, name)

    frames.sort(key=frame_key)
    return [os.path.join(folder, f) for f in frames]


def middle_frame(frame_paths: List[str]) -> str:
    if not frame_paths:
        raise ValueError("No frame paths provided")
    return frame_paths[len(frame_paths) // 2]


def sample_frame_paths(frame_paths: List[str], num_frames: int) -> List[str]:
    if not frame_paths:
        return []
    if num_frames <= 1:
        return [middle_frame(frame_paths)]
    if len(frame_paths) <= num_frames:
        return frame_paths
    idxs = [round(i * (len(frame_paths) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frame_paths[i] for i in idxs]


def collect_samples(aide_root: str, annotation_root: str, max_sequences: int = 0) -> List[Dict]:
    candidate_set = set(EMOTION_LABELS)
    seq_ids = [d for d in os.listdir(aide_root) if os.path.isdir(os.path.join(aide_root, d)) and d.isdigit()]
    seq_ids.sort()

    samples = []
    for seq_id in seq_ids:
        anno_path = os.path.join(annotation_root, f"{seq_id}.json")
        incar_dir = os.path.join(aide_root, seq_id, "incarframes")
        if not os.path.exists(anno_path) or not os.path.isdir(incar_dir):
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        label = normalize_emotion_label(anno.get("emotion_label", "Unknown"))
        if label not in candidate_set:
            continue

        frames = sorted_frame_paths(incar_dir)
        if not frames:
            continue

        samples.append(
            {
                "sequence_id": seq_id,
                "label": label,
                "frame_path": middle_frame(frames),
                "frame_paths": frames,
                "driver_behavior_label": anno.get("driver_behavior_label"),
                "scene_centric_context_label": anno.get("scene_centric_context_label"),
                "vehicle_based_context_label": anno.get("vehicle_based_context_label"),
            }
        )

    if max_sequences > 0:
        samples = samples[:max_sequences]
    return samples


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    label_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        label_groups.setdefault(sample["label"], []).append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, group in label_groups.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


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
    mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".ckpt.pt")


def evaluate_split(y_true: List[str], y_pred: List[str]) -> Dict:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, EMOTION_LABELS), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, EMOTION_LABELS),
    }


def build_causal_group_ids(samples: List[Dict], group_source: str):
    import torch

    keys = []
    for sample in samples:
        scene = sample.get("scene_centric_context_label") or "unknown_scene"
        vehicle = sample.get("vehicle_based_context_label") or "unknown_vehicle"
        behavior = sample.get("driver_behavior_label") or "unknown_behavior"
        if group_source == "scene":
            keys.append(str(scene))
        elif group_source == "vehicle":
            keys.append(str(vehicle))
        elif group_source == "behavior":
            keys.append(str(behavior))
        else:
            keys.append(f"{scene}||{vehicle}")
    unique_keys = {key: idx for idx, key in enumerate(sorted(set(keys)))}
    return torch.tensor([unique_keys[key] for key in keys], dtype=torch.long), unique_keys


def build_class_prompts(class_names: Optional[List[str]] = None) -> List[List[str]]:  # QCPA:
    effective_class_names = class_names or EMOTION_PROMPT_CLASS_NAMES  # QCPA:
    return build_prompts_for_classes(effective_class_names)  # QCPA:


def build_prompt_group_indices(num_prompts: int, group_size: int) -> List[List[int]]:
    if num_prompts == NUM_PROMPTS and len(PROMPT_GROUP_IDS) == NUM_PROMPTS:  # QCPA:
        prompt_groups = []  # QCPA:
        for group_id in sorted(set(PROMPT_GROUP_IDS)):  # QCPA:
            prompt_groups.append([idx for idx, gid in enumerate(PROMPT_GROUP_IDS) if gid == group_id])  # QCPA:
        return prompt_groups  # QCPA:
    # QCPA TODO: keep the legacy contiguous grouping fallback for old ensemble code paths outside the structured 9-prompt bank.
    if group_size <= 0 or group_size >= num_prompts:
        return [list(range(num_prompts))]
    groups = []
    for start in range(0, num_prompts, group_size):
        groups.append(list(range(start, min(start + group_size, num_prompts))))
    return groups


def count_parameters(parameters) -> int:  # QCPA:
    return int(sum(param.numel() for param in parameters if getattr(param, "requires_grad", False)))  # QCPA:


def format_tensor_list(tensor, precision: int = 4) -> List[float]:  # QCPA:
    if tensor is None:  # QCPA:
        return []  # QCPA:
    values = tensor.detach().cpu().view(-1).tolist()  # QCPA:
    return [round(float(value), precision) for value in values]  # QCPA:


def extract_image_features(
    samples: List[Dict],
    processor,
    model,
    device: str,
    batch_size: int,
    num_frames: int,
    tag: str,
    feature_layout: str = "pooled",
):
    import torch

    feats = []
    total_batches = max(1, math.ceil(len(samples) / batch_size))
    start_time = time.time()
    log(
        f"[FEATURES] start {tag}: samples={len(samples)}, batches={total_batches}, "
        f"num_frames={num_frames}, feature_layout={feature_layout}"
    )
    for batch_idx, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch = samples[start:start + batch_size]
        batch_features = []
        for sample in batch:
            fpaths = sample.get("frame_paths") or [sample["frame_path"]]
            selected = sample_frame_paths(fpaths, num_frames)
            images = [Image.open(fp).convert("RGB") for fp in selected]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            if feature_layout == "sequence":
                if image_features.shape[0] < num_frames:
                    pad = image_features[-1:].expand(num_frames - image_features.shape[0], -1)
                    image_features = torch.cat([image_features, pad], dim=0)
                batch_features.append(image_features[:num_frames])
            else:
                pooled = image_features.mean(dim=0)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                batch_features.append(pooled)
        feats.append(torch.stack(batch_features, dim=0).float().cpu())
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            log(
                f"[FEATURES] {tag}: batch {batch_idx}/{total_batches}, "
                f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)}"
            )
    return torch.cat(feats, dim=0)


def extract_text_features(prompt_groups: List[List[str]], processor, model, device: str):
    import torch

    class_prompt_features = []
    log(f"[TEXT] start text feature extraction: classes={len(prompt_groups)}, prompts_per_class={len(prompt_groups[0]) if prompt_groups else 0}")
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_prompt_features.append(text_features)
    log("[TEXT] done text feature extraction")
    return torch.stack(class_prompt_features, dim=0).float().detach()


class ClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        attention_scope: str = "local",  # QCPA:
        dual_stage: bool = False,  # QCPA:
        use_residual_gate: bool = True,  # QCPA:
        use_mahalanobis_temp: bool = True,  # QCPA:
        use_lowrank_bias: bool = True,  # QCPA:
        bias_rank: int = 16,  # QCPA:
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        self.head = QCPAHead(  # QCPA:
            feat_dim=dim,  # QCPA:
            num_classes=num_classes,  # QCPA:
            num_prompts=num_prompts,  # QCPA:
            num_heads=4,  # QCPA:
            attention_scope=attention_scope,  # QCPA:
            dual_stage=dual_stage,  # QCPA:
            use_residual_gate=use_residual_gate,  # QCPA:
            use_mahalanobis_temp=use_mahalanobis_temp,  # QCPA:
            use_lowrank_bias=use_lowrank_bias,  # QCPA:
            bias_rank=bias_rank,  # QCPA:
        ).to(device)  # QCPA:

    def parameters(self):
        return self.adapter_parameters() + self.qcpa_parameters()  # QCPA:

    def adapter_parameters(self):  # QCPA:
        return list(self.input_proj.parameters()) + list(self.net.parameters()) + list(self.out_proj.parameters())  # QCPA:

    def taga_parameters(self):  # QCPA:
        return []  # QCPA:

    def qcpa_parameters(self):  # QCPA:
        return list(self.head.parameters())  # QCPA:

    def state_dict(self):
        return {
            "input_proj": self.input_proj.state_dict(),
            "net": self.net.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "head": self.head.state_dict(),  # QCPA:
        }

    def load_state_dict(self, state):
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.head.load_state_dict(state["head"])  # QCPA:

    def train(self):
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()
        self.head.train()  # QCPA:

    def eval(self):
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()
        self.head.eval()  # QCPA:

    def _prepare_input_features(self, image_x):
        return image_x

    def _encode_prelogits(self, prepared_x):
        base = self.input_proj(prepared_x)
        delta = self.net(base)
        fused = base + delta
        return self.out_proj(fused)

    def _adapt_image(self, image_x):
        prepared_x = self._prepare_input_features(image_x)
        img = self._encode_prelogits(prepared_x)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def compute_logits_from_adapted(self, adapted_features, text_x, return_aux: bool = False):  # QCPA:
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # QCPA:
        return self.head(adapted_features, txt, return_aux=return_aux)  # QCPA:

    def logits(self, image_x, text_x, return_aux: bool = False):  # QCPA:
        img = self._adapt_image(image_x)  # QCPA:
        return self.compute_logits_from_adapted(img, text_x, return_aux=return_aux)  # QCPA:

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)
        # QCPA TODO: keep legacy ensemble voting alive with grouped cosine fallback; this path does not reproduce full QCPA attention for grouped prompts.
        global_scale = self.head.log_scale.exp().clamp(max=100.0).view(1, 1, 1)  # QCPA:
        class_scale = self.head.log_tau.exp().view(1, -1, 1)  # QCPA:
        class_bias = self.head.bias_static.view(1, -1, 1) if hasattr(self.head, "bias_static") else 0.0  # QCPA:
        return global_scale * scores * class_scale + class_bias  # QCPA:


class TemporalTransformerClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        num_frames: int,
        temporal_num_heads: int,
        temporal_num_layers: int,
        temporal_pool_mode: str,
        attention_scope: str = "local",  # QCPA:
        dual_stage: bool = False,  # QCPA:
        use_residual_gate: bool = True,  # QCPA:
        use_mahalanobis_temp: bool = True,  # QCPA:
        use_lowrank_bias: bool = True,  # QCPA:
        bias_rank: int = 16,  # QCPA:
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.num_frames = int(num_frames)
        self.frame_proj = nn.Linear(dim, hidden_dim).to(device)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.num_frames + 1, hidden_dim, device=device))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, device=device))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ).to(device)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_num_layers).to(device)
        self.temporal_norm = nn.LayerNorm(hidden_dim).to(device)
        self.temporal_pool_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.temporal_out = nn.Linear(hidden_dim, dim).to(device)
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        self.temporal_num_heads = int(temporal_num_heads)
        self.temporal_num_layers = int(temporal_num_layers)
        self.temporal_pool_mode = str(temporal_pool_mode)
        self.head = QCPAHead(  # QCPA:
            feat_dim=dim,  # QCPA:
            num_classes=num_classes,  # QCPA:
            num_prompts=num_prompts,  # QCPA:
            num_heads=4,  # QCPA:
            attention_scope=attention_scope,  # QCPA:
            dual_stage=dual_stage,  # QCPA:
            use_residual_gate=use_residual_gate,  # QCPA:
            use_mahalanobis_temp=use_mahalanobis_temp,  # QCPA:
            use_lowrank_bias=use_lowrank_bias,  # QCPA:
            bias_rank=bias_rank,  # QCPA:
        ).to(device)  # QCPA:

    def parameters(self):
        return self.taga_parameters() + self.adapter_parameters() + self.qcpa_parameters()  # QCPA:

    def taga_parameters(self):  # QCPA:
        return list(self.frame_proj.parameters()) + [self.frame_pos, self.cls_token] + list(self.temporal_encoder.parameters()) + list(self.temporal_norm.parameters()) + list(self.temporal_pool_gate.parameters()) + list(self.temporal_out.parameters())  # QCPA:

    def adapter_parameters(self):  # QCPA:
        return list(self.input_proj.parameters()) + list(self.net.parameters()) + list(self.out_proj.parameters())  # QCPA:

    def qcpa_parameters(self):  # QCPA:
        return list(self.head.parameters())  # QCPA:

    def state_dict(self):
        return {
            "frame_proj": self.frame_proj.state_dict(),
            "frame_pos": self.frame_pos.detach().cpu().clone(),
            "cls_token": self.cls_token.detach().cpu().clone(),
            "temporal_encoder": self.temporal_encoder.state_dict(),
            "temporal_norm": self.temporal_norm.state_dict(),
            "temporal_pool_gate": self.temporal_pool_gate.state_dict(),
            "temporal_out": self.temporal_out.state_dict(),
            "input_proj": self.input_proj.state_dict(),
            "net": self.net.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "num_frames": self.num_frames,
            "temporal_num_heads": self.temporal_num_heads,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_pool_mode": self.temporal_pool_mode,
            "head": self.head.state_dict(),  # QCPA:
        }

    def load_state_dict(self, state):
        self.frame_proj.load_state_dict(state["frame_proj"])
        self.temporal_encoder.load_state_dict(state["temporal_encoder"])
        self.temporal_norm.load_state_dict(state["temporal_norm"])
        if "temporal_pool_gate" in state:
            self.temporal_pool_gate.load_state_dict(state["temporal_pool_gate"])
        self.temporal_out.load_state_dict(state["temporal_out"])
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.frame_pos.data.copy_(state["frame_pos"].to(self.device))
        self.cls_token.data.copy_(state["cls_token"].to(self.device))
        self.num_frames = int(state.get("num_frames", self.num_frames))
        self.temporal_num_heads = int(state.get("temporal_num_heads", self.temporal_num_heads))
        self.temporal_num_layers = int(state.get("temporal_num_layers", self.temporal_num_layers))
        self.temporal_pool_mode = str(state.get("temporal_pool_mode", self.temporal_pool_mode))
        self.head.load_state_dict(state["head"])  # QCPA:

    def train(self):
        self.frame_proj.train()
        self.temporal_encoder.train()
        self.temporal_norm.train()
        self.temporal_pool_gate.train()
        self.temporal_out.train()
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()
        self.head.train()  # QCPA:

    def eval(self):
        self.frame_proj.eval()
        self.temporal_encoder.eval()
        self.temporal_norm.eval()
        self.temporal_pool_gate.eval()
        self.temporal_out.eval()
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()
        self.head.eval()  # QCPA:

    def _pool_frames(self, image_x):
        import torch

        if image_x.ndim != 3:
            raise ValueError(
                f"TemporalTransformerClipImageAdapter expects [batch, frames, dim] inputs, got shape={tuple(image_x.shape)}"
            )
        frame_count = int(image_x.shape[1])
        if frame_count > self.num_frames:
            raise ValueError(
                f"TemporalTransformerClipImageAdapter received {frame_count} frames but was initialized for {self.num_frames}"
            )
        hidden = self.frame_proj(image_x)
        cls = self.cls_token.expand(hidden.shape[0], -1, -1)
        tokens = torch.cat([cls, hidden], dim=1)
        tokens = tokens + self.frame_pos[:, : frame_count + 1, :]
        encoded = self.temporal_encoder(tokens)
        cls_hidden = self.temporal_norm(encoded[:, 0, :])
        frame_hidden = self.temporal_norm(encoded[:, 1 : frame_count + 1, :])

        if self.temporal_pool_mode == "mean":
            pooled_hidden = frame_hidden.mean(dim=1)
        elif self.temporal_pool_mode == "hybrid":
            mean_hidden = frame_hidden.mean(dim=1)
            gate_input = torch.cat([cls_hidden, mean_hidden], dim=-1)
            gate = torch.sigmoid(self.temporal_pool_gate(gate_input))
            pooled_hidden = gate * cls_hidden + (1.0 - gate) * mean_hidden
        else:
            pooled_hidden = cls_hidden
        pooled = self.temporal_out(pooled_hidden)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _prepare_input_features(self, image_x):
        return self._pool_frames(image_x)

    def _encode_prelogits(self, prepared_x):
        base = self.input_proj(prepared_x)
        delta = self.net(base)
        fused = base + delta
        return self.out_proj(fused)

    def _adapt_image(self, image_x):
        prepared_x = self._prepare_input_features(image_x)
        img = self._encode_prelogits(prepared_x)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def compute_logits_from_adapted(self, adapted_features, text_x, return_aux: bool = False):  # QCPA:
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # QCPA:
        return self.head(adapted_features, txt, return_aux=return_aux)  # QCPA:

    def logits(self, image_x, text_x, return_aux: bool = False):  # QCPA:
        img = self._adapt_image(image_x)  # QCPA:
        return self.compute_logits_from_adapted(img, text_x, return_aux=return_aux)  # QCPA:

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)
        # QCPA TODO: keep legacy ensemble voting alive with grouped cosine fallback; this path does not reproduce full QCPA attention for grouped prompts.
        global_scale = self.head.log_scale.exp().clamp(max=100.0).view(1, 1, 1)  # QCPA:
        class_scale = self.head.log_tau.exp().view(1, -1, 1)  # QCPA:
        class_bias = self.head.bias_static.view(1, -1, 1) if hasattr(self.head, "bias_static") else 0.0  # QCPA:
        return global_scale * scores * class_scale + class_bias  # QCPA:


def predict_emotion_from_features(
    image_features,
    text_features,
    adapter,
    idx2label: Dict[int, str],
    batch_size: int,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    dump_attn_path: Optional[str] = None,  # QCPA:
) -> List[str]:
    import torch

    preds = []
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)
    attn_batches = []  # QCPA:

    for start in range(0, image_features.shape[0], batch_size):
        batch_x = image_features[start:start + batch_size]
        with torch.no_grad():
            if dump_attn_path:  # QCPA:
                _, aux = adapter.logits(batch_x.to(adapter.device), text_features.to(adapter.device), return_aux=True)  # QCPA:
                if aux is not None and aux.get("attn") is not None:  # QCPA:
                    attn_batches.append(aux["attn"].detach().cpu().numpy())  # QCPA:
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = adapter.grouped_logits(batch_x.to(adapter.device), text_features.to(adapter.device), group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(idx2label))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = adapter.logits(batch_x.to(adapter.device), text_features.to(adapter.device))
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx2label[i] for i in idxs])
    if dump_attn_path and attn_batches:  # QCPA:
        dump_path = Path(dump_attn_path)  # QCPA:
        dump_path.parent.mkdir(parents=True, exist_ok=True)  # QCPA:
        np.save(str(dump_path), np.concatenate(attn_batches, axis=0))  # QCPA:
        log(f"[QCPA] attention dump saved to: {dump_path}")  # QCPA:
    return preds


def predict_emotion(samples: List[Dict], processor, model, prompts: List[str], device: str, batch_size: int) -> List[str]:
    import torch

    preds = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [Image.open(s["frame_path"]).convert("RGB") for s in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([EMOTION_LABELS[i] for i in idxs])
    return preds


def train_clip_supervised(
    model,
    processor,
    train_samples: List[Dict],
    val_samples: List[Dict],
    prompts: List[str],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
):
    import torch
    import torch.nn as nn

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    overall_start = time.time()

    for epoch_idx in range(epochs):
        model.train()
        random.shuffle(train_samples)
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start:start + batch_size]
            images = [Image.open(s["frame_path"]).convert("RGB") for s in batch]
            targets = torch.tensor([label2idx[s["label"]] for s in batch], dtype=torch.long, device=device)
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)

            outputs = model(**inputs)
            logits = outputs.logits_per_image
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = predict_emotion(val_samples, processor, model, prompts, device, batch_size)
            val_true = [s["label"] for s in val_samples]
            val_acc = accuracy(val_true, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1)
        log(
            f"[TRAIN] epoch {epoch_idx + 1}/{epochs} | loss={running_loss / max(1, num_batches):.6f} | "
            f"val_acc={val_acc:.6f} | best_val_acc={best_val_acc:.6f} | "
            f"epoch_time={format_duration(time.time() - epoch_start)} | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_strict_frozen_clip(
    train_x,
    train_y,
    val_x,
    val_y,
    text_features,
    adapter: ClipImageAdapter,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    use_class_weight: bool,
    label_smoothing: float,
    select_metric: str,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    train_group_ids=None,
    use_causal_contrastive: bool = False,
    ccl_weight: float = 0.5,
    ccl_temperature: float = 0.5,
    use_causal_alignment: bool = False,
    cfa_weight: float = 0.1,
    use_counterfactual_aug: bool = False,
    cda_prob: float = 0.3,
    cda_n_replace_max: int = 3,
    use_cda_v2_mixstyle: bool = False,
    cda_v2_prob: float = 0.5,
    cda_v2_kl_weight: float = 0.5,
    use_ccl_v2_counterfactual: bool = False,
    ccl_v2_weight: float = 0.1,
    ccl_v2_temperature: float = 0.1,
    use_cfa_v2_textanchor: bool = False,
    cfa_v2_weight: float = 0.05,
    cfa_v2_anchor_weight: float = 1.0,
    cfa_v2_ema_momentum: float = 0.99,
):
    import torch
    import torch.nn as nn

    class_weights = None
    if use_class_weight:
        class_counts = torch.bincount(train_y, minlength=len(EMOTION_LABELS)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_metric = -1.0
    best_epoch_idx = -1
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
    overall_start = time.time()
    text_features_device = text_features.to(adapter.device)
    cfa_v2_state = {"global": {}, "group": {}}

    if (
        use_causal_contrastive
        or use_causal_alignment
        or use_counterfactual_aug
        or use_cda_v2_mixstyle
        or use_ccl_v2_counterfactual
        or use_cfa_v2_textanchor
    ) and train_group_ids is None:
        raise RuntimeError("train_group_ids are required when causal training options are enabled")

    for epoch_idx in range(epochs):
        adapter.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_cda_v2_loss = 0.0
        running_ccl_v2_loss = 0.0
        running_cfa_v2_loss = 0.0
        num_batches = 0
        perm = torch.randperm(train_x.shape[0])
        train_x = train_x[perm]
        train_y = train_y[perm]
        if train_group_ids is not None:
            train_group_ids = train_group_ids[perm]

        for start in range(0, train_x.shape[0], batch_size):
            bx = train_x[start:start + batch_size].to(adapter.device)
            by = train_y[start:start + batch_size].to(adapter.device)
            batch_group_ids = None
            if train_group_ids is not None:
                batch_group_ids = train_group_ids[start:start + batch_size].to(adapter.device)
            model_input_x = bx
            cda_v2_loss_value = 0.0
            ccl_v2_loss_value = 0.0
            cfa_v2_loss_value = 0.0
            if use_counterfactual_aug and batch_group_ids is not None:
                model_input_x = cremad_base.counterfactual_feature_aug(
                    bx,
                    by,
                    batch_group_ids,
                    p=cda_prob,
                    n_replace_max=cda_n_replace_max,
                )
            adapter_features = adapter._adapt_image(model_input_x)
            logits = cremad_base.compute_adapter_logits_from_features(adapter, adapter_features, text_features_device)
            loss = criterion(logits, by)
            if use_causal_contrastive and batch_group_ids is not None:
                loss = loss + (float(ccl_weight) * cremad_base.causal_contrastive_loss(
                    adapter_features,
                    by,
                    batch_group_ids,
                    temperature=ccl_temperature,
                ))
            if use_causal_alignment and batch_group_ids is not None:
                loss = loss + (float(cfa_weight) * cremad_base.causal_feature_alignment_loss(
                    adapter_features,
                    by,
                    batch_group_ids,
                ))
            if use_cda_v2_mixstyle and batch_group_ids is not None:
                cf_bx, cf_mask, _ = cremad_base.build_v2_counterfactual_batch(
                    bx,
                    by,
                    batch_group_ids,
                    p=cda_v2_prob,
                )
                if bool(cf_mask.any().item()):
                    cf_features = adapter._adapt_image(cf_bx)
                    cf_logits = cremad_base.compute_adapter_logits_from_features(adapter, cf_features, text_features_device)
                    cda_ce_loss = criterion(cf_logits[cf_mask], by[cf_mask])
                    cda_kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(cf_logits[cf_mask], dim=-1),
                        torch.nn.functional.softmax(logits[cf_mask].detach(), dim=-1),
                        reduction="batchmean",
                    )
                    cda_v2_loss = cda_ce_loss + (float(cda_v2_kl_weight) * cda_kl_loss)
                    loss = loss + cda_v2_loss
                    cda_v2_loss_value = float(cda_v2_loss.item())
                    if use_ccl_v2_counterfactual:
                        ccl_v2_loss = float(ccl_v2_weight) * cremad_base.counterfactual_anchored_contrastive_loss(
                            adapter_features[cf_mask],
                            cf_features[cf_mask],
                            by[cf_mask],
                            adapter_features,
                            by,
                            temperature=ccl_v2_temperature,
                        )
                        loss = loss + ccl_v2_loss
                        ccl_v2_loss_value = float(ccl_v2_loss.item())
            elif use_ccl_v2_counterfactual and batch_group_ids is not None:
                ccl_v2_loss = float(ccl_v2_weight) * cremad_base.causal_contrastive_loss(
                    adapter_features,
                    by,
                    batch_group_ids,
                    temperature=ccl_v2_temperature,
                )
                loss = loss + ccl_v2_loss
                ccl_v2_loss_value = float(ccl_v2_loss.item())
            if use_cfa_v2_textanchor and batch_group_ids is not None:
                cfa_v2_loss = float(cfa_v2_weight) * cremad_base.cfa_v2_text_anchor_loss(
                    adapter_features,
                    by,
                    batch_group_ids,
                    text_features_device,
                    cfa_v2_state,
                    momentum=cfa_v2_ema_momentum,
                    anchor_weight=cfa_v2_anchor_weight,
                )
                loss = loss + cfa_v2_loss
                cfa_v2_loss_value = float(cfa_v2_loss.item())

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item())
            running_cda_v2_loss += cda_v2_loss_value
            running_ccl_v2_loss += ccl_v2_loss_value
            running_cfa_v2_loss += cfa_v2_loss_value
            num_batches += 1

        adapter.eval()
        with torch.no_grad():
            val_pred = predict_emotion_from_features(
                val_x,
                text_features,
                adapter,
                idx2label,
                batch_size,
                use_test_ensemble=use_test_ensemble,
                ensemble_group_size=ensemble_group_size,
            )
            val_true = [EMOTION_LABELS[int(x.item())] for x in val_y]
            val_acc = accuracy(val_true, val_pred)
            val_wf1 = weighted_f1(val_true, val_pred, EMOTION_LABELS)
            metric = val_wf1 if select_metric == "weighted_f1" else val_acc

        if metric > best_val_metric:
            best_val_metric = metric
            best_epoch_idx = epoch_idx
            best_state = adapter.state_dict()

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1)
        component_parts = []
        if use_cda_v2_mixstyle:
            component_parts.append(f"cda_v2={running_cda_v2_loss / max(1, num_batches):.6f}")
        if use_ccl_v2_counterfactual:
            component_parts.append(f"ccl_v2={running_ccl_v2_loss / max(1, num_batches):.6f}")
        if use_cfa_v2_textanchor:
            component_parts.append(f"cfa_v2={running_cfa_v2_loss / max(1, num_batches):.6f}")
        component_summary = f" | {' | '.join(component_parts)}" if component_parts else ""
        log(
            f"[TRAIN] epoch {epoch_idx + 1}/{epochs} | loss={running_loss / max(1, num_batches):.6f}{component_summary} | "
            f"val_acc={val_acc:.6f} | val_wf1={val_wf1:.6f} | best_metric={best_val_metric:.6f} | "
            f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0} | "
            f"epoch_time={format_duration(time.time() - epoch_start)} | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
        )
        if hasattr(adapter, "head"):  # QCPA:
            gate_values = format_tensor_list(torch.sigmoid(adapter.head.gate_logit)) if getattr(adapter.head, "gate_logit", None) is not None else []  # QCPA:
            tau_values = format_tensor_list(adapter.head.log_tau.exp()) if getattr(adapter.head, "log_tau", None) is not None else []  # QCPA:
            scale_value = round(float(adapter.head.log_scale.exp().detach().cpu().item()), 6) if getattr(adapter.head, "log_scale", None) is not None else 0.0  # QCPA:
            log(f"[QCPA] epoch {epoch_idx + 1} | gate={gate_values} | tau={tau_values} | scale={scale_value}")  # QCPA:

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone AIDE CLIP emotion training")
    parser.add_argument("--aide_root", default=DEFAULT_AIDE_ROOT)
    parser.add_argument("--annotation_root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="Driver is <LABEL>.")
    parser.add_argument("--prompt_set", default="driving_7", help="single | default_5 | driving_7 | custom templates joined by ||")
    parser.add_argument("--debug", action="store_true")  # QCPA:
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--temporal_head", choices=["none", "transformer"], default="none")
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--temporal_pool_mode", choices=["cls", "mean", "hybrid"], default="cls")
    parser.add_argument("--use_class_weight", choices=["on", "off"], default="off")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", choices=["on", "off"], default="off")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--strict_frozen_clip", choices=["on", "off"], default="off", help="Freeze CLIP and train only custom adapter module")
    parser.add_argument("--use_prompt_weight", choices=["on", "off"], default="on")
    parser.add_argument("--use_class_temperature", choices=["on", "off"], default="on")
    parser.add_argument("--use_class_bias", choices=["on", "off"], default="on")
    parser.add_argument("--attention_scope", choices=["local", "global"], default="local")  # QCPA:
    parser.add_argument("--dual_stage", action="store_true")  # QCPA:
    parser.add_argument("--no_residual_gate", action="store_true")  # QCPA:
    parser.add_argument("--no_mahalanobis", action="store_true")  # QCPA:
    parser.add_argument("--no_lowrank_bias", action="store_true")  # QCPA:
    parser.add_argument("--bias_rank", type=int, default=16)  # QCPA:
    parser.add_argument("--dump_attn", action="store_true")  # QCPA:
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--use_causal_contrastive", choices=["on", "off"], default="off")
    parser.add_argument("--ccl_weight", type=float, default=0.5)
    parser.add_argument("--ccl_temperature", type=float, default=0.5)
    parser.add_argument("--use_causal_alignment", choices=["on", "off"], default="off")
    parser.add_argument("--cfa_weight", type=float, default=0.1)
    parser.add_argument("--use_counterfactual_aug", choices=["on", "off"], default="off")
    parser.add_argument("--cda_prob", type=float, default=0.3)
    parser.add_argument("--cda_n_replace_max", type=int, default=3)
    parser.add_argument("--use_cda_v2_mixstyle", choices=["on", "off"], default="off")
    parser.add_argument("--cda_v2_prob", type=float, default=0.5)
    parser.add_argument("--cda_v2_kl_weight", type=float, default=0.5)
    parser.add_argument("--use_ccl_v2_counterfactual", choices=["on", "off"], default="off")
    parser.add_argument("--ccl_v2_weight", type=float, default=0.1)
    parser.add_argument("--ccl_v2_temperature", type=float, default=0.1)
    parser.add_argument("--use_cfa_v2_textanchor", choices=["on", "off"], default="off")
    parser.add_argument("--cfa_v2_weight", type=float, default=0.05)
    parser.add_argument("--cfa_v2_anchor_weight", type=float, default=1.0)
    parser.add_argument("--cfa_v2_ema_momentum", type=float, default=0.99)
    parser.add_argument("--causal_group_source", choices=["scene", "vehicle", "behavior", "scene_vehicle"], default="scene_vehicle")
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.use_class_weight = parse_on_off(args.use_class_weight)
    args.use_test_ensemble = parse_on_off(args.use_test_ensemble)
    args.strict_frozen_clip = parse_on_off(args.strict_frozen_clip)
    args.use_prompt_weight = parse_on_off(args.use_prompt_weight)
    args.use_class_temperature = parse_on_off(args.use_class_temperature)
    args.use_class_bias = parse_on_off(args.use_class_bias)
    args.use_causal_contrastive = parse_on_off(args.use_causal_contrastive)
    args.use_causal_alignment = parse_on_off(args.use_causal_alignment)
    args.use_counterfactual_aug = parse_on_off(args.use_counterfactual_aug)
    args.use_cda_v2_mixstyle = parse_on_off(args.use_cda_v2_mixstyle)
    args.use_ccl_v2_counterfactual = parse_on_off(args.use_ccl_v2_counterfactual)
    args.use_cfa_v2_textanchor = parse_on_off(args.use_cfa_v2_textanchor)
    if args.training_seed is None:
        args.training_seed = args.seed

    if args.temporal_num_heads < 1:
        parser.error("--temporal_num_heads must be >= 1")
    if args.temporal_num_layers < 1:
        parser.error("--temporal_num_layers must be >= 1")
    if args.bias_rank < 1:  # QCPA:
        parser.error("--bias_rank must be >= 1")  # QCPA:
    if args.temporal_head != "none" and not args.strict_frozen_clip:
        parser.error("--temporal_head requires --strict_frozen_clip")
    if args.temporal_head != "none" and args.adapter_hidden_dim % args.temporal_num_heads != 0:
        parser.error("--adapter_hidden_dim must be divisible by --temporal_num_heads")
    if args.ccl_weight < 0:
        parser.error("--ccl_weight must be >= 0")
    if args.ccl_temperature <= 0:
        parser.error("--ccl_temperature must be > 0")
    if args.cfa_weight < 0:
        parser.error("--cfa_weight must be >= 0")
    if args.cda_prob < 0 or args.cda_prob > 1:
        parser.error("--cda_prob must satisfy 0 <= cda_prob <= 1")
    if args.cda_n_replace_max < 1:
        parser.error("--cda_n_replace_max must be >= 1")
    if args.cda_v2_prob < 0 or args.cda_v2_prob > 1:
        parser.error("--cda_v2_prob must satisfy 0 <= cda_v2_prob <= 1")
    if args.cda_v2_kl_weight < 0:
        parser.error("--cda_v2_kl_weight must be >= 0")
    if args.ccl_v2_weight < 0:
        parser.error("--ccl_v2_weight must be >= 0")
    if args.ccl_v2_temperature <= 0:
        parser.error("--ccl_v2_temperature must be > 0")
    if args.cfa_v2_weight < 0:
        parser.error("--cfa_v2_weight must be >= 0")
    if args.cfa_v2_anchor_weight < 0:
        parser.error("--cfa_v2_anchor_weight must be >= 0")
    if args.cfa_v2_ema_momentum < 0 or args.cfa_v2_ema_momentum >= 1:
        parser.error("--cfa_v2_ema_momentum must satisfy 0 <= cfa_v2_ema_momentum < 1")
    if args.debug and args.max_sequences <= 0:  # QCPA:
        args.max_sequences = 64  # QCPA:
    if args.debug:  # QCPA:
        args.batch_size = min(args.batch_size, 8)  # QCPA:
    return args


def main():
    args = parse_args()
    configure_random_seeds(args.seed)
    feature_layout = "sequence" if args.temporal_head == "transformer" else "pooled"
    class_names = list(EMOTION_PROMPT_CLASS_NAMES)  # QCPA:

    samples = collect_samples(args.aide_root, args.annotation_root, args.max_sequences)
    if len(samples) < 10:
        raise RuntimeError(f"Too few valid samples: {len(samples)}")
    log(f"[INFO] valid samples: {len(samples)}")

    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    log(f"[INFO] split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    prompt_groups = build_class_prompts(class_names)  # QCPA:
    single_prompts = [x[0] for x in prompt_groups]  # QCPA:

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if args.clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(args.model_id)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False)
        except Exception:
            processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    log(f"[INFO] model loaded: {args.model_id} on {args.device}")

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

    if args.strict_frozen_clip:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        cache_path = None
        if args.feature_cache_dir:
            os.makedirs(args.feature_cache_dir, exist_ok=True)
            cache_key_src = {
                "aide_root": os.path.abspath(args.aide_root),
                "annotation_root": os.path.abspath(args.annotation_root),
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "max_sequences": args.max_sequences,
                "model_id": args.model_id,
                "prompt_template": "structured_qcpa_9",  # QCPA:
                "prompt_set": "structured_9",  # QCPA:
                "prompt_group_ids": list(PROMPT_GROUP_IDS),  # QCPA:
                "num_frames": args.num_frames,
            }
            if feature_layout != "pooled":
                cache_key_src["feature_layout"] = feature_layout
            cache_key = hashlib.sha1(json.dumps(cache_key_src, sort_keys=True).encode("utf-8")).hexdigest()[:16]
            cache_path = os.path.join(args.feature_cache_dir, f"strict_features_{cache_key}.pt")

        if cache_path and os.path.exists(cache_path):
            log(f"[CACHE] loading strict features from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            train_x = cached["train_x"]
            val_x = cached["val_x"]
            test_x = cached["test_x"]
            text_features = cached["text_features"]
            train_y = cached["train_y"]
            val_y = cached["val_y"]
            test_y = cached["test_y"]
        else:
            log("[CACHE] miss" if cache_path else "[CACHE] disabled")
            train_x = extract_image_features(
                train_samples,
                processor,
                model,
                args.device,
                args.batch_size,
                args.num_frames,
                tag="train",
                feature_layout=feature_layout,
            )
            val_x = extract_image_features(
                val_samples,
                processor,
                model,
                args.device,
                args.batch_size,
                args.num_frames,
                tag="val",
                feature_layout=feature_layout,
            )
            test_x = extract_image_features(
                test_samples,
                processor,
                model,
                args.device,
                args.batch_size,
                args.num_frames,
                tag="test",
                feature_layout=feature_layout,
            )
            text_features = extract_text_features(prompt_groups, processor, model, args.device)
            if int(text_features.shape[1]) != NUM_PROMPTS:  # QCPA:
                raise RuntimeError(f"QCPA expects {NUM_PROMPTS} prompts per class, got {int(text_features.shape[1])}")  # QCPA:

            train_y = torch.tensor([label2idx[s["label"]] for s in train_samples], dtype=torch.long)
            val_y = torch.tensor([label2idx[s["label"]] for s in val_samples], dtype=torch.long)
            test_y = torch.tensor([label2idx[s["label"]] for s in test_samples], dtype=torch.long)

            if cache_path:
                torch.save(
                    {
                        "train_x": train_x,
                        "val_x": val_x,
                        "test_x": test_x,
                        "text_features": text_features,
                        "train_y": train_y,
                        "val_y": val_y,
                        "test_y": test_y,
                    },
                    cache_path,
                )
                log(f"[CACHE] saved strict features to: {cache_path}")

        train_group_ids = None
        group_mapping = None
        if (
            args.use_causal_contrastive
            or args.use_causal_alignment
            or args.use_counterfactual_aug
            or args.use_cda_v2_mixstyle
            or args.use_ccl_v2_counterfactual
            or args.use_cfa_v2_textanchor
        ):
            train_group_ids, group_mapping = build_causal_group_ids(train_samples, args.causal_group_source)

        configure_random_seeds(args.training_seed)
        log(f"[SEED] split_seed={args.seed} training_seed={args.training_seed}")

        if args.temporal_head == "transformer":
            adapter = TemporalTransformerClipImageAdapter(
                dim=int(train_x.shape[-1]),
                device=args.device,
                hidden_dim=args.adapter_hidden_dim,
                dropout=args.adapter_dropout,
                num_classes=len(EMOTION_LABELS),
                num_prompts=NUM_PROMPTS,  # QCPA:
                num_frames=args.num_frames,
                temporal_num_heads=args.temporal_num_heads,
                temporal_num_layers=args.temporal_num_layers,
                temporal_pool_mode=args.temporal_pool_mode,
                attention_scope=args.attention_scope,  # QCPA:
                dual_stage=args.dual_stage,  # QCPA:
                use_residual_gate=not args.no_residual_gate,  # QCPA:
                use_mahalanobis_temp=not args.no_mahalanobis,  # QCPA:
                use_lowrank_bias=not args.no_lowrank_bias,  # QCPA:
                bias_rank=args.bias_rank,  # QCPA:
            )
        else:
            adapter = ClipImageAdapter(
                dim=int(train_x.shape[1]),
                device=args.device,
                hidden_dim=args.adapter_hidden_dim,
                dropout=args.adapter_dropout,
                num_classes=len(EMOTION_LABELS),
                num_prompts=NUM_PROMPTS,  # QCPA:
                attention_scope=args.attention_scope,  # QCPA:
                dual_stage=args.dual_stage,  # QCPA:
                use_residual_gate=not args.no_residual_gate,  # QCPA:
                use_mahalanobis_temp=not args.no_mahalanobis,  # QCPA:
                use_lowrank_bias=not args.no_lowrank_bias,  # QCPA:
                bias_rank=args.bias_rank,  # QCPA:
            )
        backbone_params = list(model.parameters())  # QCPA:
        if any(param.requires_grad for param in backbone_params):  # QCPA:
            raise RuntimeError("QCPA requires the CLIP backbone to remain frozen")  # QCPA:
        qcpa_param_count = count_parameters(adapter.qcpa_parameters())  # QCPA:
        taga_param_count = count_parameters(adapter.taga_parameters())  # QCPA:
        adapter_param_count = count_parameters(adapter.adapter_parameters())  # QCPA:
        trainable_param_count = count_parameters(adapter.parameters())  # QCPA:
        frozen_backbone_count = int(sum(param.numel() for param in backbone_params))  # QCPA:
        ratio = (trainable_param_count / frozen_backbone_count) if frozen_backbone_count > 0 else 0.0  # QCPA:
        log(f"[QCPA] params | head={qcpa_param_count} | taga={taga_param_count} | adapter={adapter_param_count} | trainable={trainable_param_count} | frozen_backbone={frozen_backbone_count} | ratio={ratio:.6f}")  # QCPA:
        adapter = train_strict_frozen_clip(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            text_features=text_features,
            adapter=adapter,
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
            train_group_ids=train_group_ids,
            use_causal_contrastive=args.use_causal_contrastive,
            ccl_weight=args.ccl_weight,
            ccl_temperature=args.ccl_temperature,
            use_causal_alignment=args.use_causal_alignment,
            cfa_weight=args.cfa_weight,
            use_counterfactual_aug=args.use_counterfactual_aug,
            cda_prob=args.cda_prob,
            cda_n_replace_max=args.cda_n_replace_max,
            use_cda_v2_mixstyle=args.use_cda_v2_mixstyle,
            cda_v2_prob=args.cda_v2_prob,
            cda_v2_kl_weight=args.cda_v2_kl_weight,
            use_ccl_v2_counterfactual=args.use_ccl_v2_counterfactual,
            ccl_v2_weight=args.ccl_v2_weight,
            ccl_v2_temperature=args.ccl_v2_temperature,
            use_cfa_v2_textanchor=args.use_cfa_v2_textanchor,
            cfa_v2_weight=args.cfa_v2_weight,
            cfa_v2_anchor_weight=args.cfa_v2_anchor_weight,
            cfa_v2_ema_momentum=args.cfa_v2_ema_momentum,
        )

        val_pred = predict_emotion_from_features(
            val_x,
            text_features,
            adapter,
            idx2label,
            args.batch_size,
            use_test_ensemble=args.use_test_ensemble,
            ensemble_group_size=args.ensemble_group_size,
            dump_attn_path=str(Path(args.output).with_name(f"{Path(args.output).stem}_val_attn.npy")) if args.dump_attn else None,  # QCPA:
        )
        test_pred = predict_emotion_from_features(
            test_x,
            text_features,
            adapter,
            idx2label,
            args.batch_size,
            use_test_ensemble=args.use_test_ensemble,
            ensemble_group_size=args.ensemble_group_size,
            dump_attn_path=str(Path(args.output).with_name(f"{Path(args.output).stem}_test_attn.npy")) if args.dump_attn else None,  # QCPA:
        )
        val_true = [EMOTION_LABELS[int(i.item())] for i in val_y]
        test_true = [EMOTION_LABELS[int(i.item())] for i in test_y]
    else:
        model = train_clip_supervised(
            model=model,
            processor=processor,
            train_samples=train_samples,
            val_samples=val_samples,
            prompts=single_prompts,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
        val_pred = predict_emotion(val_samples, processor, model, single_prompts, args.device, args.batch_size)
        test_pred = predict_emotion(test_samples, processor, model, single_prompts, args.device, args.batch_size)
        val_true = [s["label"] for s in val_samples]
        test_true = [s["label"] for s in test_samples]

    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else default_checkpoint_path(Path(args.output))

    result = {
        "git_commit": resolve_git_commit(),
        "config": {
            "method": "clip_supervised_text_image_emotion",
            "task": "emotion",
            "split": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": round(1 - args.train_ratio - args.val_ratio, 6),
            },
            "model_id": args.model_id,
            "prompt_template": "structured_qcpa_9",  # QCPA:
            "prompt_set": "structured_9",  # QCPA:
            "prompt_group_ids": list(PROMPT_GROUP_IDS),  # QCPA:
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "num_frames": args.num_frames,
            "feature_layout": feature_layout,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "temporal_head": args.temporal_head,
            "temporal_num_heads": args.temporal_num_heads,
            "temporal_num_layers": args.temporal_num_layers,
            "temporal_pool_mode": args.temporal_pool_mode,
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "strict_frozen_clip": args.strict_frozen_clip,
            "use_prompt_weight": args.use_prompt_weight,
            "use_class_temperature": args.use_class_temperature,
            "use_class_bias": args.use_class_bias,
            "attention_scope": args.attention_scope,  # QCPA:
            "dual_stage": args.dual_stage,  # QCPA:
            "use_residual_gate": not args.no_residual_gate,  # QCPA:
            "use_mahalanobis_temp": not args.no_mahalanobis,  # QCPA:
            "use_lowrank_bias": not args.no_lowrank_bias,  # QCPA:
            "bias_rank": args.bias_rank,  # QCPA:
            "dump_attn": args.dump_attn,  # QCPA:
            "feature_cache_dir": args.feature_cache_dir,
            "checkpoint_output": str(checkpoint_path),
            "seed": args.seed,
            "training_seed": args.training_seed,
            "max_sequences": args.max_sequences,
            "use_causal_contrastive": args.use_causal_contrastive,
            "ccl_weight": args.ccl_weight,
            "ccl_temperature": args.ccl_temperature,
            "use_causal_alignment": args.use_causal_alignment,
            "cfa_weight": args.cfa_weight,
            "use_counterfactual_aug": args.use_counterfactual_aug,
            "cda_prob": args.cda_prob,
            "cda_n_replace_max": args.cda_n_replace_max,
            "causal_group_source": args.causal_group_source,
            "use_cda_v2_mixstyle": args.use_cda_v2_mixstyle,
            "cda_v2_prob": args.cda_v2_prob,
            "cda_v2_kl_weight": args.cda_v2_kl_weight,
            "use_ccl_v2_counterfactual": args.use_ccl_v2_counterfactual,
            "ccl_v2_weight": args.ccl_v2_weight,
            "ccl_v2_temperature": args.ccl_v2_temperature,
            "use_cfa_v2_textanchor": args.use_cfa_v2_textanchor,
            "cfa_v2_weight": args.cfa_v2_weight,
            "cfa_v2_anchor_weight": args.cfa_v2_anchor_weight,
            "cfa_v2_ema_momentum": args.cfa_v2_ema_momentum,
        },
        "dataset": {
            "total": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "label_distribution_train": dict(Counter([s["label"] for s in train_samples])),
        },
        "val": evaluate_split(val_true, val_pred),
        "test": evaluate_split(test_true, test_pred),
    }
    if args.strict_frozen_clip and (args.use_causal_contrastive or args.use_causal_alignment or args.use_counterfactual_aug):
        result["dataset"]["causal_group_count"] = len(group_mapping) if group_mapping is not None else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "config": result["config"],
        "dataset": result["dataset"],
        "metrics": {
            "val": result["val"],
            "test": result["test"],
        },
        "prompt_groups": prompt_groups,  # QCPA:
        "prompt_group_ids": list(PROMPT_GROUP_IDS),  # QCPA:
        "label2idx": label2idx,
        "idx2label": idx2label,
        "output_path": str(output_path),
    }
    if args.strict_frozen_clip:
        checkpoint_payload.update(
            {
                "checkpoint_type": "strict_frozen_clip_adapter",
                "adapter_state_dict": adapter.state_dict(),
                "text_features": text_features.cpu(),
            }
        )
    else:
        checkpoint_payload.update(
            {
                "checkpoint_type": "clip_finetune_model",
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        )
    torch.save(checkpoint_payload, checkpoint_path)

    log(f"[DONE] saved supervised CLIP emotion report to: {output_path}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
