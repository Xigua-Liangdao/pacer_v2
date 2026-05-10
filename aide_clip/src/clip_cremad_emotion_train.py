import argparse
import atexit
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from clip_ravdess_emotion_train import (
    ClipImageAdapter,
    FocalCrossEntropyLoss,
    default_checkpoint_path,
    extract_text_features,
    predict_emotion_from_features,
    read_sampled_media,
)

# CREMA-D-specific dataset logic added here while preserving the existing strict_frozen_clip_adapter method family.
CREMAD_LABEL_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
EMOTION_LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
CREMAD_FACE_VOTE_MAP = {
    "A": "angry",
    "D": "disgust",
    "F": "fearful",
    "H": "happy",
    "N": "neutral",
    "S": "sad",
}
DEFAULT_VIDEO_EXTENSIONS = {".flv", ".mp4"}
VIDEO_EXTENSION_PRIORITY = {".mp4": 2, ".flv": 1}

CREMAD_PROMPT_GROUPS = {
    "angry": [
        "The face shows anger with furrowed brows, a tense jaw, and tight lips.",
        "The visible expression looks angry, stern, and tense around the eyes and mouth.",
        "The person has an angry face with knitted brows and a hard stare.",
        "The facial cues suggest anger through brow tension, compressed lips, and facial strain.",
        "The expression appears angry with a rigid mouth and confrontational eye area.",
    ],
    "disgust": [
        "The face shows disgust with a wrinkled nose and a raised upper lip.",
        "The visible expression looks disgusted with a rejecting grimace and nose tension.",
        "The person has a disgusted face with an aversive mouth shape and curled upper lip.",
        "The facial cues suggest disgust through nose wrinkling and a repulsed expression.",
        "The expression appears disgusted with lip curl, nasal tension, and visible aversion.",
    ],
    "fearful": [
        "The face shows fear with widened eyes, raised brows, and a tense mouth.",
        "The visible expression looks fearful, anxious, and strained around the eyes.",
        "The person has a fearful face with alarmed eyes and tight facial tension.",
        "The facial cues suggest fear through eye widening, brow lift, and guarded tension.",
        "The expression appears fearful with worried eyes and a strained mouth shape.",
    ],
    "happy": [
        "The face shows happiness with a smile, lifted cheeks, and bright eyes.",
        "The visible expression looks happy, cheerful, and clearly smiling.",
        "The person has a happy face with raised cheeks and smiling mouth corners.",
        "The facial cues suggest happiness through a warm smile and lively eyes.",
        "The expression appears happy with lifted cheeks, a curved mouth, and positive energy.",
    ],
    "neutral": [
        "The face looks neutral with little visible emotion and relaxed features.",
        "The visible expression appears neutral, composed, and free of strong facial tension.",
        "The person has a neutral face with level brows and a relaxed mouth.",
        "The facial cues suggest neutrality through balanced features and minimal expression.",
        "The expression appears neutral with a steady gaze and little emotional activation.",
    ],
    "sad": [
        "The face shows sadness with downturned lips, lowered gaze, and drooping eyelids.",
        "The visible expression looks sad, subdued, and pulled downward in the mouth.",
        "The person has a sad face with low-energy features and a downcast expression.",
        "The facial cues suggest sadness through drooped eyes and downward mouth corners.",
        "The expression appears sad with a lowered gaze and softly sagging facial features.",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CREMAD_ROOT = os.environ.get("CREMAD_ROOT", str(PROJECT_ROOT / "data" / "crema_d"))
DEFAULT_FEATURE_CACHE_DIR = str(PROJECT_ROOT / "cache" / "cremad_features")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "cremad" / "clip_cremad_emotion_supervised_results.json")
LOG_FILE_HANDLE = None
FEATURE_CACHE_VERSION = "v2"
SUPPORTED_FRAME_SAMPLING_MODES = {"uniform", "middle_late", "diff_guided"}


class StrongerClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        ).to(device)

        self.logit_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes, device=device))
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))
        self.use_global_logit_scale = use_global_logit_scale
        self.use_prompt_weight = use_prompt_weight
        self.use_class_temperature = use_class_temperature
        self.use_class_bias = use_class_bias

    def parameters(self):
        params = list(self.input_proj.parameters()) + list(self.block1.parameters()) + list(self.block2.parameters()) + list(self.out_proj.parameters())
        if self.use_global_logit_scale:
            params.append(self.logit_scale)
        if self.use_prompt_weight:
            params.append(self.prompt_weight_logits)
        if self.use_class_temperature:
            params.append(self.class_logit_scale)
        if self.use_class_bias:
            params.append(self.class_bias)
        return params

    def state_dict(self):
        return {
            "input_proj": self.input_proj.state_dict(),
            "block1": self.block1.state_dict(),
            "block2": self.block2.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
            "use_global_logit_scale": self.use_global_logit_scale,
            "use_prompt_weight": self.use_prompt_weight,
            "use_class_temperature": self.use_class_temperature,
            "use_class_bias": self.use_class_bias,
        }

    def load_state_dict(self, state):
        self.input_proj.load_state_dict(state["input_proj"])
        self.block1.load_state_dict(state["block1"])
        self.block2.load_state_dict(state["block2"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))
        self.use_global_logit_scale = state.get("use_global_logit_scale", False)
        self.use_prompt_weight = state.get("use_prompt_weight", True)
        self.use_class_temperature = state.get("use_class_temperature", True)
        self.use_class_bias = state.get("use_class_bias", True)

    def train(self):
        self.input_proj.train()
        self.block1.train()
        self.block2.train()
        self.out_proj.train()

    def eval(self):
        self.input_proj.eval()
        self.block1.eval()
        self.block2.eval()
        self.out_proj.eval()

    def _prepare_input_features(self, image_x):
        return image_x

    def _encode_prelogits(self, prepared_x):
        base = self.input_proj(prepared_x)
        hidden = base + self.block1(base)
        hidden = hidden + self.block2(hidden)
        return self.out_proj(hidden)

    def _adapt_image(self, image_x):
        prepared_x = self._prepare_input_features(image_x)
        img = self._encode_prelogits(prepared_x)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def logits(self, image_x, text_x):
        import torch
        import torch.nn.functional as F

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        if self.use_prompt_weight:
            prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_sim = (sim * prompt_w).sum(dim=-1)
        else:
            class_sim = sim.mean(dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_sim * class_scale + class_bias

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).view(1, -1, 1)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.view(1, -1, 1)
        else:
            class_bias = 0.0
        return global_scale * scores * class_scale + class_bias


class TemporalClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        num_frames: int,
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.num_frames = int(num_frames)
        self.frame_proj = nn.Linear(dim, hidden_dim).to(device)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.num_frames, hidden_dim, device=device))
        self.frame_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
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

        self.logit_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes, device=device))
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))
        self.use_global_logit_scale = use_global_logit_scale
        self.use_prompt_weight = use_prompt_weight
        self.use_class_temperature = use_class_temperature
        self.use_class_bias = use_class_bias

    def parameters(self):
        params = (
            list(self.frame_proj.parameters())
            + list(self.frame_score.parameters())
            + [self.frame_pos]
            + list(self.temporal_out.parameters())
            + list(self.input_proj.parameters())
            + list(self.net.parameters())
            + list(self.out_proj.parameters())
        )
        if self.use_global_logit_scale:
            params.append(self.logit_scale)
        if self.use_prompt_weight:
            params.append(self.prompt_weight_logits)
        if self.use_class_temperature:
            params.append(self.class_logit_scale)
        if self.use_class_bias:
            params.append(self.class_bias)
        return params

    def state_dict(self):
        return {
            "frame_proj": self.frame_proj.state_dict(),
            "frame_pos": self.frame_pos.detach().cpu().clone(),
            "frame_score": self.frame_score.state_dict(),
            "temporal_out": self.temporal_out.state_dict(),
            "input_proj": self.input_proj.state_dict(),
            "net": self.net.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
            "use_global_logit_scale": self.use_global_logit_scale,
            "use_prompt_weight": self.use_prompt_weight,
            "use_class_temperature": self.use_class_temperature,
            "use_class_bias": self.use_class_bias,
            "num_frames": self.num_frames,
        }

    def load_state_dict(self, state):
        self.frame_proj.load_state_dict(state["frame_proj"])
        self.frame_score.load_state_dict(state["frame_score"])
        self.temporal_out.load_state_dict(state["temporal_out"])
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.frame_pos.data.copy_(state["frame_pos"].to(self.device))
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))
        self.use_global_logit_scale = state.get("use_global_logit_scale", False)
        self.use_prompt_weight = state.get("use_prompt_weight", True)
        self.use_class_temperature = state.get("use_class_temperature", True)
        self.use_class_bias = state.get("use_class_bias", True)
        self.num_frames = int(state.get("num_frames", self.num_frames))

    def train(self):
        self.frame_proj.train()
        self.frame_score.train()
        self.temporal_out.train()
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()

    def eval(self):
        self.frame_proj.eval()
        self.frame_score.eval()
        self.temporal_out.eval()
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()

    def _pool_frames(self, image_x):
        import torch

        if image_x.ndim != 3:
            raise ValueError(
                f"TemporalClipImageAdapter expects [batch, frames, dim] inputs, got shape={tuple(image_x.shape)}"
            )
        frame_count = int(image_x.shape[1])
        if frame_count > self.frame_pos.shape[1]:
            raise ValueError(
                f"TemporalClipImageAdapter received {frame_count} frames but was initialized for {self.frame_pos.shape[1]}"
            )
        hidden = self.frame_proj(image_x)
        hidden = hidden + self.frame_pos[:, :frame_count, :]
        attn_logits = self.frame_score(hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled_hidden = (hidden * attn_weights).sum(dim=1)
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

    def logits(self, image_x, text_x):
        import torch
        import torch.nn.functional as F

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        if self.use_prompt_weight:
            prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_sim = (sim * prompt_w).sum(dim=-1)
        else:
            class_sim = sim.mean(dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_sim * class_scale + class_bias

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).view(1, -1, 1)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.view(1, -1, 1)
        else:
            class_bias = 0.0
        return global_scale * scores * class_scale + class_bias


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
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
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

        self.logit_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes, device=device))
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))
        self.use_global_logit_scale = use_global_logit_scale
        self.use_prompt_weight = use_prompt_weight
        self.use_class_temperature = use_class_temperature
        self.use_class_bias = use_class_bias
        self.temporal_num_heads = int(temporal_num_heads)
        self.temporal_num_layers = int(temporal_num_layers)
        self.temporal_pool_mode = str(temporal_pool_mode)

    def parameters(self):
        params = (
            list(self.frame_proj.parameters())
            + [self.frame_pos, self.cls_token]
            + list(self.temporal_encoder.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.temporal_pool_gate.parameters())
            + list(self.temporal_out.parameters())
            + list(self.input_proj.parameters())
            + list(self.net.parameters())
            + list(self.out_proj.parameters())
        )
        if self.use_global_logit_scale:
            params.append(self.logit_scale)
        if self.use_prompt_weight:
            params.append(self.prompt_weight_logits)
        if self.use_class_temperature:
            params.append(self.class_logit_scale)
        if self.use_class_bias:
            params.append(self.class_bias)
        return params

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
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
            "use_global_logit_scale": self.use_global_logit_scale,
            "use_prompt_weight": self.use_prompt_weight,
            "use_class_temperature": self.use_class_temperature,
            "use_class_bias": self.use_class_bias,
            "num_frames": self.num_frames,
            "temporal_num_heads": self.temporal_num_heads,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_pool_mode": self.temporal_pool_mode,
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
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))
        self.use_global_logit_scale = state.get("use_global_logit_scale", False)
        self.use_prompt_weight = state.get("use_prompt_weight", True)
        self.use_class_temperature = state.get("use_class_temperature", True)
        self.use_class_bias = state.get("use_class_bias", True)
        self.num_frames = int(state.get("num_frames", self.num_frames))
        self.temporal_num_heads = int(state.get("temporal_num_heads", self.temporal_num_heads))
        self.temporal_num_layers = int(state.get("temporal_num_layers", self.temporal_num_layers))
        self.temporal_pool_mode = str(state.get("temporal_pool_mode", self.temporal_pool_mode))

    def train(self):
        self.frame_proj.train()
        self.temporal_encoder.train()
        self.temporal_norm.train()
        self.temporal_pool_gate.train()
        self.temporal_out.train()
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()

    def eval(self):
        self.frame_proj.eval()
        self.temporal_encoder.eval()
        self.temporal_norm.eval()
        self.temporal_pool_gate.eval()
        self.temporal_out.eval()
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()

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

    def logits(self, image_x, text_x):
        import torch
        import torch.nn.functional as F

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        if self.use_prompt_weight:
            prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_sim = (sim * prompt_w).sum(dim=-1)
        else:
            class_sim = sim.mean(dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_sim * class_scale + class_bias

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        import torch

        img = self._adapt_image(image_x)
        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).view(1, -1, 1)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.view(1, -1, 1)
        else:
            class_bias = 0.0
        return global_scale * scores * class_scale + class_bias


def log(message: str) -> None:
    global LOG_FILE_HANDLE

    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(line + "\n")
        LOG_FILE_HANDLE.flush()


def init_log_file(log_file: Optional[str]) -> Optional[str]:
    global LOG_FILE_HANDLE

    if not log_file:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_HANDLE = path.open("a", encoding="utf-8")
    return str(path)


def close_log_file() -> None:
    global LOG_FILE_HANDLE

    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None


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


def prediction_distribution(y_pred: List[str], labels: List[str]) -> Dict[str, int]:
    counts = Counter(y_pred)
    return {label: int(counts.get(label, 0)) for label in labels}


def evaluate_split(y_true: List[str], y_pred: List[str]) -> Dict:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, EMOTION_LABELS), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, EMOTION_LABELS),
    }


def summarize_predictions(y_true: List[str], y_pred: List[str]) -> Dict:
    metrics = evaluate_split(y_true, y_pred)
    metrics["prediction_distribution"] = prediction_distribution(y_pred, EMOTION_LABELS)
    return metrics


def parse_facelevel_value(face_level_raw: Optional[str]) -> Optional[float]:
    if face_level_raw is None:
        return None
    tokens = [token.strip() for token in str(face_level_raw).replace(";", ":").split(":") if token.strip()]
    numeric_values: List[float] = []
    for token in tokens:
        try:
            numeric_values.append(float(token))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return max(numeric_values)


def build_facelevel_sample_weights(samples: List[Dict], args):
    import torch

    weights: List[float] = []
    num_missing_facelevel = 0
    num_low = 0
    num_mid = 0
    num_high = 0

    for sample in samples:
        parsed_value = parse_facelevel_value(sample.get("face_level"))
        if parsed_value is None:
            weight_value = 1.0
            num_missing_facelevel += 1
        elif parsed_value < args.facelevel_low_thresh:
            weight_value = float(args.facelevel_low_weight)
            num_low += 1
        elif parsed_value < args.facelevel_high_thresh:
            weight_value = float(args.facelevel_mid_weight)
            num_mid += 1
        else:
            weight_value = float(args.facelevel_high_weight)
            num_high += 1
        weights.append(weight_value)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    stats = {
        "num_missing_facelevel": num_missing_facelevel,
        "num_low": num_low,
        "num_mid": num_mid,
        "num_high": num_high,
        "weight_mean": round(float(weight_tensor.mean().item()), 6) if weight_tensor.numel() > 0 else 0.0,
        "weight_min": round(float(weight_tensor.min().item()), 6) if weight_tensor.numel() > 0 else 0.0,
        "weight_max": round(float(weight_tensor.max().item()), 6) if weight_tensor.numel() > 0 else 0.0,
    }
    return weight_tensor, stats


def parse_extension_set(raw_value: str) -> Set[str]:
    value = str(raw_value or "all").strip().lower()
    if not value or value == "all":
        return set(DEFAULT_VIDEO_EXTENSIONS)
    parsed = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.add(item if item.startswith(".") else f".{item}")
    return parsed or set(DEFAULT_VIDEO_EXTENSIONS)


def normalize_face_vote(face_vote: Optional[str]) -> Optional[str]:
    if face_vote is None:
        return None
    tokens = [token.strip() for token in str(face_vote).split(":") if token.strip()]
    if not tokens:
        return None
    mapped = [CREMAD_FACE_VOTE_MAP.get(token, token) for token in tokens]
    return ":".join(mapped)


def parse_cremad_filename(file_name: str) -> Optional[Dict[str, str]]:
    path = Path(file_name)
    ext = path.suffix.lower()
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    actor_id, sentence_code, emotion_code, level_code = parts
    if len(actor_id) != 4 or not actor_id.isdigit():
        return None
    if emotion_code not in CREMAD_LABEL_MAP:
        return None
    return {
        "sequence_id": stem,
        "actor_id": actor_id,
        "sentence_code": sentence_code,
        "emotion_code": emotion_code,
        "label": CREMAD_LABEL_MAP[emotion_code],
        "level_code": level_code,
        "ext": ext,
    }


def load_sentence_filename_index(cremad_root: str) -> Dict[str, int]:
    csv_path = Path(cremad_root) / "SentenceFilenames.csv"
    if not csv_path.exists():
        return {}
    index: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = str(row.get("Filename", "")).strip()
            if not stem:
                continue
            try:
                index[stem] = int(row.get("Stimulus_Number", 0))
            except ValueError:
                index[stem] = 0
    return index


def load_video_demographics(cremad_root: str) -> Dict[str, Dict[str, str]]:
    csv_path = Path(cremad_root) / "VideoDemographics.csv"
    if not csv_path.exists():
        return {}
    demographics: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actor_id = str(row.get("ActorID", "")).strip()
            if not actor_id:
                continue
            demographics[actor_id] = {
                "age": str(row.get("Age", "")).strip(),
                "sex": str(row.get("Sex", "")).strip(),
                "race": str(row.get("Race", "")).strip(),
                "ethnicity": str(row.get("Ethnicity", "")).strip(),
            }
    return demographics


def resolve_summary_table_path(cremad_root: str, explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    root = Path(cremad_root)
    candidates.append(root / "processedResults" / "summaryTable.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(root.rglob("summaryTable.csv"))
    return found[0] if found else None


def load_cremad_summary_table(cremad_root: str, summary_table_path: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Optional[str]]], Optional[str]]:
    path = resolve_summary_table_path(cremad_root, summary_table_path)
    if path is None or not path.exists():
        return {}, None

    summary_map: Dict[str, Dict[str, Optional[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = str(row.get("fileName") or row.get("FileName") or "").strip()
            if not file_name:
                continue
            sequence_id = Path(file_name).stem
            summary_map[sequence_id] = {
                "face_vote": normalize_face_vote(row.get("FaceVote")),
                "face_vote_raw": str(row.get("FaceVote", "")).strip() or None,
                "face_level": str(row.get("FaceLevel", "")).strip() or None,
                "voice_vote": str(row.get("VoiceVote", "")).strip() or None,
                "multimodal_vote": str(row.get("MultiModalVote", "")).strip() or None,
            }
    return summary_map, str(path)


def collect_cremad_samples(
    cremad_root: str,
    max_sequences: int = 0,
    allowed_extensions: Optional[Set[str]] = None,
    summary_table_path: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, object]]:
    root = Path(cremad_root).resolve()
    video_root = root / "VideoFlash"
    if not video_root.exists():
        raise FileNotFoundError(f"CREMA-D VideoFlash directory not found: {video_root}")

    allowed_extensions = {ext.lower() for ext in (allowed_extensions or DEFAULT_VIDEO_EXTENSIONS)}
    resolved_root = str(root)
    sentence_index = load_sentence_filename_index(resolved_root)
    demographics = load_video_demographics(resolved_root)
    summary_map, resolved_summary_path = load_cremad_summary_table(resolved_root, summary_table_path)

    selected_by_sequence: Dict[str, Dict] = {}
    duplicate_replacements = 0
    invalid_name_examples: List[str] = []

    for path in sorted(video_root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            continue
        if path.stat().st_size <= 0:
            continue
        meta = parse_cremad_filename(path.name)
        if meta is None:
            if len(invalid_name_examples) < 10:
                invalid_name_examples.append(path.name)
            continue

        stimulus_number = sentence_index.get(meta["sequence_id"])
        actor_demographics = demographics.get(meta["actor_id"], {})
        summary_row = summary_map.get(meta["sequence_id"], {})
        sample = {
            "sequence_id": meta["sequence_id"],
            "video_path": str(path),
            "file_name": path.name,
            "actor_id": meta["actor_id"],
            "sentence_code": meta["sentence_code"],
            "emotion_code": meta["emotion_code"],
            "label": meta["label"],
            "level_code": meta["level_code"],
            "ext": meta["ext"],
            "stimulus_number": stimulus_number,
            "face_vote": summary_row.get("face_vote"),
            "face_vote_raw": summary_row.get("face_vote_raw"),
            "face_level": summary_row.get("face_level"),
            "voice_vote": summary_row.get("voice_vote"),
            "multimodal_vote": summary_row.get("multimodal_vote"),
            "age": actor_demographics.get("age"),
            "sex": actor_demographics.get("sex"),
            "race": actor_demographics.get("race"),
            "ethnicity": actor_demographics.get("ethnicity"),
        }

        existing = selected_by_sequence.get(meta["sequence_id"])
        if existing is None:
            selected_by_sequence[meta["sequence_id"]] = sample
            continue

        old_priority = VIDEO_EXTENSION_PRIORITY.get(str(existing.get("ext", "")).lower(), 0)
        new_priority = VIDEO_EXTENSION_PRIORITY.get(meta["ext"], 0)
        if new_priority > old_priority or (new_priority == old_priority and path.stat().st_size > Path(existing["video_path"]).stat().st_size):
            selected_by_sequence[meta["sequence_id"]] = sample
            duplicate_replacements += 1

    samples = [selected_by_sequence[key] for key in sorted(selected_by_sequence)]
    if max_sequences > 0:
        samples = samples[:max_sequences]

    missing_summary_examples = [sample["sequence_id"] for sample in samples if sample.get("face_vote") is None][:10]
    expected_sequences = set(sentence_index)
    observed_sequences = {sample["sequence_id"] for sample in samples}
    missing_from_videos = sorted(expected_sequences - observed_sequences)[:10]
    extra_sequences = sorted(observed_sequences - expected_sequences)[:10] if expected_sequences else []

    diagnostics = {
        "resolved_summary_table_path": resolved_summary_path,
        "auxiliary_facevote_loaded": bool(summary_map),
        "total_valid_samples": len(samples),
        "actor_count": len({sample["actor_id"] for sample in samples}),
        "class_distribution": dict(Counter(sample["label"] for sample in samples)),
        "extension_distribution": dict(Counter(sample["ext"] for sample in samples)),
        "duplicate_sequence_replacements": duplicate_replacements,
        "invalid_name_examples": invalid_name_examples,
        "missing_summary_examples": missing_summary_examples,
        "sentence_filename_rows": len(sentence_index),
        "missing_from_video_examples": missing_from_videos,
        "extra_video_examples": extra_sequences,
    }
    return samples, diagnostics


def compute_split_summary(samples: List[Dict]) -> Dict[str, object]:
    return {
        "sample_count": len(samples),
        "actor_count": len({sample["actor_id"] for sample in samples}),
        "actors": sorted({sample["actor_id"] for sample in samples}),
        "class_distribution": dict(Counter(sample["label"] for sample in samples)),
        "extension_distribution": dict(Counter(sample["ext"] for sample in samples)),
    }


def deterministic_actor_order(actor_groups: Dict[str, List[Dict]], seed: int) -> List[str]:
    def tie_key(actor_id: str) -> str:
        return hashlib.sha1(f"{seed}:{actor_id}".encode("utf-8")).hexdigest()

    return sorted(actor_groups, key=lambda actor_id: (-len(actor_groups[actor_id]), tie_key(actor_id), actor_id))


def build_balanced_actor_folds(samples: List[Dict], num_folds: int, seed: int) -> Dict[str, int]:
    actor_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        actor_groups.setdefault(sample["actor_id"], []).append(sample)

    fold_sample_counts = [0 for _ in range(num_folds)]
    fold_actor_counts = [0 for _ in range(num_folds)]
    actor_to_fold: Dict[str, int] = {}

    for actor_id in deterministic_actor_order(actor_groups, seed):
        actor_sample_count = len(actor_groups[actor_id])
        candidate_folds = sorted(
            range(num_folds),
            key=lambda fold_idx: (
                fold_sample_counts[fold_idx],
                fold_actor_counts[fold_idx],
                hashlib.sha1(f"{seed}:{actor_id}:{fold_idx}".encode("utf-8")).hexdigest(),
                fold_idx,
            ),
        )
        chosen_fold = candidate_folds[0]
        actor_to_fold[actor_id] = chosen_fold
        fold_sample_counts[chosen_fold] += actor_sample_count
        fold_actor_counts[chosen_fold] += 1

    return actor_to_fold


def build_fold_diagnostics(samples: List[Dict], actor_to_fold: Dict[str, int], num_folds: int) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    for fold_idx in range(num_folds):
        fold_samples = [sample for sample in samples if actor_to_fold[sample["actor_id"]] == fold_idx]
        diagnostics[str(fold_idx)] = compute_split_summary(fold_samples)
    return diagnostics


def split_cremad_samples_5fold(samples: List[Dict], fold_idx: int, seed: int) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    num_folds = 5
    if fold_idx < 0 or fold_idx >= num_folds:
        raise ValueError(f"fold_idx must be in [0, {num_folds - 1}], got {fold_idx}")

    actor_to_fold = build_balanced_actor_folds(samples, num_folds=num_folds, seed=seed)
    val_fold = (fold_idx + 1) % num_folds
    train_folds = [idx for idx in range(num_folds) if idx not in {fold_idx, val_fold}]

    train_samples = [sample for sample in samples if actor_to_fold[sample["actor_id"]] in train_folds]
    val_samples = [sample for sample in samples if actor_to_fold[sample["actor_id"]] == val_fold]
    test_samples = [sample for sample in samples if actor_to_fold[sample["actor_id"]] == fold_idx]

    split_info = {
        "mode": "5fold_speaker_independent",
        "fold_idx": fold_idx,
        "val_fold": val_fold,
        "train_folds": train_folds,
        "actor_to_fold": actor_to_fold,
        "folds": build_fold_diagnostics(samples, actor_to_fold, num_folds=num_folds),
        "train": compute_split_summary(train_samples),
        "val": compute_split_summary(val_samples),
        "test": compute_split_summary(test_samples),
    }
    return {"train": train_samples, "val": val_samples, "test": test_samples}, split_info


def split_cremad_samples_actor_ratio(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    actor_groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        actor_groups.setdefault(sample["actor_id"], []).append(sample)

    actor_order = deterministic_actor_order(actor_groups, seed)
    total_samples = len(samples)
    targets = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * test_ratio,
    }
    assigned_actors = {"train": [], "val": [], "test": []}
    current_counts = {"train": 0, "val": 0, "test": 0}

    split_names = ["train", "val", "test"]
    for actor_id in actor_order:
        actor_count = len(actor_groups[actor_id])
        best_split = None
        best_score = None
        for split_name in split_names:
            projected = current_counts[split_name] + actor_count
            target = targets[split_name]
            score = (
                projected > target and target > 0,
                abs(projected - target),
                current_counts[split_name],
                split_name,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assigned_actors[best_split].append(actor_id)
        current_counts[best_split] += actor_count

    train_actor_ids = set(assigned_actors["train"])
    val_actor_ids = set(assigned_actors["val"])
    test_actor_ids = set(assigned_actors["test"])

    train_samples = [sample for sample in samples if sample["actor_id"] in train_actor_ids]
    val_samples = [sample for sample in samples if sample["actor_id"] in val_actor_ids]
    test_samples = [sample for sample in samples if sample["actor_id"] in test_actor_ids]

    split_info = {
        "mode": "split_speaker_independent",
        "targets": {key: round(value, 3) for key, value in targets.items()},
        "train": compute_split_summary(train_samples),
        "val": compute_split_summary(val_samples),
        "test": compute_split_summary(test_samples),
    }
    return {"train": train_samples, "val": val_samples, "test": test_samples}, split_info


def build_class_prompts(prompt_template: str, prompt_set: str) -> List[List[str]]:
    if prompt_set == "cremad_6_facial_cues":
        return [list(CREMAD_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]
    if prompt_set == "single":
        return [[prompt_template.replace("<LABEL>", label)] for label in EMOTION_LABELS]
    custom_templates = [item.strip() for item in prompt_set.split("||") if item.strip()]
    if not custom_templates:
        custom_templates = [prompt_template]
    return [[template.replace("<LABEL>", label) for template in custom_templates] for label in EMOTION_LABELS]


def resolve_splits(samples: List[Dict], args) -> Tuple[Dict[str, List[Dict]], Dict[str, object], str]:
    if args.cv_mode == "5fold":
        splits, split_info = split_cremad_samples_5fold(samples, fold_idx=args.fold_idx, seed=args.seed)
        benchmark_mode = "5fold_speaker_independent"
    else:
        splits, split_info = split_cremad_samples_actor_ratio(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        benchmark_mode = "split_speaker_independent"
    return splits, split_info, benchmark_mode


def sanitize_model_id(model_id: str) -> str:
    safe = str(model_id).replace("/", "_").replace(":", "_").replace(" ", "_")
    return safe[:80]


def build_split_samples_with_index(samples: List[Dict], split_name: str) -> List[Dict]:
    indexed_samples: List[Dict] = []
    for sample_index, sample in enumerate(samples):
        enriched = dict(sample)
        enriched["split_name"] = split_name
        enriched["sample_index"] = sample_index
        indexed_samples.append(enriched)
    return indexed_samples


def compute_samples_fingerprint(samples: List[Dict]) -> str:
    payload = [
        {
            "sample_index": int(sample["sample_index"]),
            "sequence_id": sample["sequence_id"],
            "video_path": sample["video_path"],
        }
        for sample in samples
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_extraction_cache_config(
    dataset_name: str,
    split_name: str,
    samples: List[Dict],
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    video_extensions: List[str],
    feature_layout: str,
) -> Dict[str, object]:
    return {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset": dataset_name,
        "split_name": split_name,
        "model_id": model_id,
        "num_frames": int(num_frames),
        "frame_sampling_mode": frame_sampling_mode,
        "feature_layout": feature_layout,
        "video_extensions": list(video_extensions),
        "sample_count": len(samples),
        "samples_fingerprint": compute_samples_fingerprint(samples),
    }


def build_feature_cache_name(
    dataset_name: str,
    split_name: str,
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    feature_layout: str,
    sample_count: int,
    samples_fingerprint: str,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> str:
    base = (
        f"{dataset_name.lower()}_{split_name}_{sanitize_model_id(model_id)}_"
        f"f{num_frames}_{frame_sampling_mode}_n{sample_count}_{samples_fingerprint}"
    )
    if feature_layout != "pooled":
        base = f"{base}_{feature_layout}"
    if shard_index is not None and total_shards is not None:
        return f"{base}_shard{shard_index}of{total_shards}.pt"
    return f"{base}.pt"


def build_feature_cache_path(
    feature_cache_dir: str,
    dataset_name: str,
    split_name: str,
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    feature_layout: str,
    sample_count: int,
    samples_fingerprint: str,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> Path:
    cache_root = Path(feature_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / build_feature_cache_name(
        dataset_name=dataset_name,
        split_name=split_name,
        model_id=model_id,
        num_frames=num_frames,
        frame_sampling_mode=frame_sampling_mode,
        feature_layout=feature_layout,
        sample_count=sample_count,
        samples_fingerprint=samples_fingerprint,
        shard_index=shard_index,
        total_shards=total_shards,
    )


def build_failure_log_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".failures.json")


def count_existing_shards(cache_plan: Dict[str, object]) -> int:
    return sum(1 for path in cache_plan["shard_paths"] if path.exists())


def list_missing_shards(cache_plan: Dict[str, object]) -> List[Path]:
    return [path for path in cache_plan["shard_paths"] if not path.exists()]


def validate_feature_cache_payload(cache_payload: Dict, expected_config: Dict[str, object], cache_path: Path) -> None:
    cache_config = dict(cache_payload.get("config", {}))
    if "feature_layout" not in cache_config:
        cache_config["feature_layout"] = "pooled"
    expected_subset = {
        "cache_version": expected_config["cache_version"],
        "dataset": expected_config["dataset"],
        "split_name": expected_config["split_name"],
        "model_id": expected_config["model_id"],
        "num_frames": expected_config["num_frames"],
        "frame_sampling_mode": expected_config["frame_sampling_mode"],
        "feature_layout": expected_config.get("feature_layout", "pooled"),
        "video_extensions": list(expected_config["video_extensions"]),
        "sample_count": expected_config["sample_count"],
        "samples_fingerprint": expected_config["samples_fingerprint"],
    }
    mismatches = {}
    for key, expected_value in expected_subset.items():
        observed_value = cache_config.get(key)
        if observed_value != expected_value:
            mismatches[key] = {"expected": expected_value, "observed": observed_value}
    if mismatches:
        raise RuntimeError(
            f"Feature cache config mismatch for {cache_path}: {json.dumps(mismatches, ensure_ascii=False)}"
        )


def resolve_existing_split_cache(cache_plan: Dict[str, object], split_name: str) -> Optional[Path]:
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] feature cache hit for {split_name}: {final_path}")
        return final_path
    return None


def decode_sample_frames(sample: Dict, num_frames: int, frame_sampling_mode: str, split_name: str) -> Dict[str, object]:
    try:
        frames, selected_indices, total_frames = read_sampled_media(
            sample,
            num_frames,
            frame_sampling_mode=frame_sampling_mode,
        )
        if not frames:
            raise RuntimeError("No decoded frames")
        return {
            "sample": sample,
            "frames": frames,
            "selected_indices": selected_indices,
            "total_frames": total_frames,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample": sample,
            "frames": None,
            "selected_indices": [],
            "total_frames": 0,
            "error": {
                "sequence_id": sample.get("sequence_id"),
                "video_path": sample.get("video_path"),
                "label": sample.get("label"),
                "actor_id": sample.get("actor_id"),
                "split_name": split_name,
                "sample_index": sample.get("sample_index"),
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def prepare_image_inputs(processor, images, device: str, pin_memory: bool):
    import torch

    normalized_images = list(images)
    while normalized_images and all(isinstance(item, (list, tuple)) for item in normalized_images):
        flattened = []
        for item in normalized_images:
            flattened.extend(item)
        normalized_images = flattened
    if not normalized_images:
        raise ValueError("No images available for CLIP preprocessing")
    if any(isinstance(item, (list, tuple)) for item in normalized_images):
        raise ValueError(f"Unexpected nested image batch structure: sample_type={type(normalized_images[0])}")

    inputs = processor(images=normalized_images, return_tensors="pt", padding=True)
    tensor_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if pin_memory and str(device).startswith("cuda") and hasattr(value, "pin_memory"):
                value = value.pin_memory()
            tensor_inputs[key] = value.to(device, non_blocking=pin_memory)
        else:
            tensor_inputs[key] = value
    return tensor_inputs


def build_split_cache_plan(feature_cache_dir: str, dataset_name: str, split_name: str, samples: List[Dict], args) -> Dict[str, object]:
    config = build_extraction_cache_config(
        dataset_name=dataset_name,
        split_name=split_name,
        samples=samples,
        model_id=args.model_id,
        num_frames=args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
        video_extensions=sorted(parse_extension_set(args.video_extensions)),
        feature_layout=args.feature_layout,
    )
    final_path = build_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        model_id=args.model_id,
        num_frames=args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        sample_count=len(samples),
        samples_fingerprint=str(config["samples_fingerprint"]),
    )
    shard_paths = [
        build_feature_cache_path(
            feature_cache_dir=feature_cache_dir,
            dataset_name=dataset_name,
            split_name=split_name,
            model_id=args.model_id,
            num_frames=args.num_frames,
            frame_sampling_mode=args.frame_sampling_mode,
            feature_layout=args.feature_layout,
            sample_count=len(samples),
            samples_fingerprint=str(config["samples_fingerprint"]),
            shard_index=shard_index,
            total_shards=args.total_shards,
        )
        for shard_index in range(args.total_shards)
    ]
    return {
        "config": config,
        "final_path": final_path,
        "shard_paths": shard_paths,
    }


def shard_samples(samples: List[Dict], shard_index: int, total_shards: int) -> List[Dict]:
    return samples[shard_index::total_shards]


def extract_image_features_with_metadata(
    samples: List[Dict],
    processor,
    model,
    device: str,
    batch_size: int,
    num_frames: int,
    split_name: str,
    frame_sampling_mode: str,
    feature_layout: str,
    num_workers: int,
    pin_memory: bool,
):
    import torch

    feature_dim = int(getattr(model.config, "projection_dim", 512))
    if not samples:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        return {
            "features": torch.empty(empty_shape, dtype=torch.float32),
            "samples": [],
            "failed_samples": [],
        }

    total_batches = max(1, math.ceil(len(samples) / batch_size))
    feats = []
    kept_samples: List[Dict] = []
    failed_samples: List[Dict] = []
    start_time = time.time()
    log(
        f"[FEATURES] start {split_name}: samples={len(samples)}, batches={total_batches}, "
        f"num_frames={num_frames}, frame_sampling_mode={frame_sampling_mode}, feature_layout={feature_layout}"
    )
    for batch_idx, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch = samples[start:start + batch_size]
        valid_samples: List[Dict] = []
        frame_groups: List[List] = []
        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                decoded_items = list(
                    executor.map(
                        lambda sample: decode_sample_frames(sample, num_frames, frame_sampling_mode, split_name),
                        batch,
                    )
                )
        else:
            decoded_items = [decode_sample_frames(sample, num_frames, frame_sampling_mode, split_name) for sample in batch]

        for decoded in decoded_items:
            if decoded["error"] is not None:
                failed_samples.append(decoded["error"])
                continue
            valid_samples.append(decoded["sample"])
            frame_groups.append(decoded["frames"])

        if valid_samples:
            flat_images = [image for images in frame_groups for image in images]
            frame_count = len(frame_groups[0])
            inputs = prepare_image_inputs(processor, flat_images, device=device, pin_memory=pin_memory)
            with torch.no_grad():
                autocast_enabled = str(device).startswith("cuda")
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            image_features = image_features.view(len(valid_samples), frame_count, -1)
            if feature_layout == "sequence":
                feats.append(image_features.float().cpu())
            else:
                pooled = image_features.mean(dim=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                feats.append(pooled.float().cpu())
            kept_samples.extend(valid_samples)

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / batch_idx * (total_batches - batch_idx) if batch_idx < total_batches else 0.0
            log(
                f"[FEATURES] {split_name}: batch {batch_idx}/{total_batches}, kept={len(kept_samples)}, "
                f"failed={len(failed_samples)}, elapsed={time.strftime('%M:%S', time.gmtime(max(0, int(elapsed))))}, "
                f"eta={time.strftime('%M:%S', time.gmtime(max(0, int(eta))))}"
            )

    if feats:
        feature_tensor = torch.cat(feats, dim=0)
    else:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        feature_tensor = torch.empty(empty_shape, dtype=torch.float32)
    return {
        "features": feature_tensor,
        "samples": kept_samples,
        "failed_samples": failed_samples,
    }


def save_failure_log(failure_log_path: Path, failed_samples: List[Dict]) -> None:
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_log_path.open("w", encoding="utf-8") as f:
        json.dump(failed_samples, f, ensure_ascii=False, indent=2)


def save_split_feature_cache(cache_path: Path, payload: Dict) -> None:
    import torch

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_split_feature_cache(cache_path: Path) -> Dict:
    import torch

    return torch.load(cache_path, map_location="cpu")


def extract_feature_shard(samples: List[Dict], processor, model, args, split_name: str, cache_plan: Dict[str, object]) -> Path:
    shard_path = cache_plan["shard_paths"][args.shard_index]
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] final merged cache already exists for {split_name}, skip shard extraction: {final_path}")
        return final_path
    if shard_path.exists():
        payload = load_split_feature_cache(shard_path)
        validate_feature_cache_payload(payload, cache_plan["config"], shard_path)
        log(f"[CACHE] shard cache hit for {split_name} shard {args.shard_index}/{args.total_shards}: {shard_path}")
        return shard_path

    existing_shards = count_existing_shards(cache_plan)
    log(
        f"[CACHE] shard extraction state for {split_name}: existing_shards={existing_shards}/{args.total_shards}, "
        f"target_shard={args.shard_index}"
    )
    shard_sample_list = shard_samples(samples, shard_index=args.shard_index, total_shards=args.total_shards)
    log(
        f"[SHARD] extracting split={split_name} shard={args.shard_index}/{args.total_shards} "
        f"samples={len(shard_sample_list)} device={args.device} output={shard_path}"
    )
    extracted = extract_image_features_with_metadata(
        shard_sample_list,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        split_name=f"{split_name}_shard{args.shard_index}",
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    failure_log_path = build_failure_log_path(shard_path)
    save_failure_log(failure_log_path, extracted["failed_samples"])
    payload = {
        "cache_type": "feature_shard_cache",
        "dataset": "CREMA-D",
        "split_name": split_name,
        "config": cache_plan["config"],
        "shard_index": args.shard_index,
        "total_shards": args.total_shards,
        "sample_count": len(extracted["samples"]),
        "failed_count": len(extracted["failed_samples"]),
        "features": extracted["features"],
        "samples": extracted["samples"],
        "failed_samples": extracted["failed_samples"],
        "failure_log_path": str(failure_log_path),
    }
    save_split_feature_cache(shard_path, payload)
    log(f"[CACHE] saved shard feature cache: {shard_path}")
    return shard_path


def merge_feature_shards(cache_plan: Dict[str, object], delete_shards_after_merge: bool = False) -> Path:
    final_path = cache_plan["final_path"]
    expected_shards = cache_plan["shard_paths"]
    missing = [str(path) for path in list_missing_shards(cache_plan)]
    if missing:
        raise FileNotFoundError(f"Missing shard caches for merge: {missing}")
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] merged split cache already exists: {final_path}")
        return final_path

    all_entries: List[Tuple[int, Dict, object]] = []
    failed_samples: List[Dict] = []
    split_name = None
    for shard_path in expected_shards:
        shard_payload = load_split_feature_cache(shard_path)
        validate_feature_cache_payload(shard_payload, cache_plan["config"], shard_path)
        split_name = split_name or shard_payload.get("split_name")
        features = shard_payload["features"]
        shard_samples = shard_payload["samples"]
        if int(features.shape[0]) != len(shard_samples):
            raise RuntimeError(f"Feature/sample count mismatch in shard: {shard_path}")
        failed_samples.extend(list(shard_payload.get("failed_samples", [])))
        for row_idx, sample in enumerate(shard_samples):
            all_entries.append((int(sample["sample_index"]), sample, features[row_idx]))

    all_entries.sort(key=lambda item: item[0])
    duplicate_indices = [idx for idx, count in Counter(item[0] for item in all_entries).items() if count > 1]
    if duplicate_indices:
        raise RuntimeError(f"Duplicate sample_index values found while merging shards: {duplicate_indices[:10]}")

    import torch

    merged_samples = [item[1] for item in all_entries]
    if all_entries:
        merged_features = torch.stack([item[2] for item in all_entries], dim=0).cpu()
    else:
        merged_features = torch.empty((0, 0), dtype=torch.float32)
    payload = {
        "cache_type": "feature_split_cache",
        "dataset": "CREMA-D",
        "split_name": split_name,
        "config": cache_plan["config"],
        "sample_count": len(merged_samples),
        "failed_count": len(failed_samples),
        "features": merged_features,
        "samples": merged_samples,
        "failed_samples": failed_samples,
        "source_shards": [str(path) for path in expected_shards],
    }
    save_split_feature_cache(final_path, payload)
    save_failure_log(build_failure_log_path(final_path), failed_samples)
    log(f"[CACHE] merged {len(expected_shards)} shards into final split cache: {final_path}")

    if delete_shards_after_merge:
        for shard_path in expected_shards:
            shard_path.unlink(missing_ok=True)
            build_failure_log_path(shard_path).unlink(missing_ok=True)
        log(f"[CACHE] deleted shard caches after merge for {split_name}")
    return final_path


def extract_split_feature_cache(samples: List[Dict], processor, model, args, split_name: str, cache_plan: Dict[str, object]) -> Path:
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] feature cache hit for {split_name}: {final_path}")
        return final_path

    log(f"[CACHE] feature cache miss for {split_name}: {final_path}")
    extracted = extract_image_features_with_metadata(
        samples,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        split_name=split_name,
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    payload = {
        "cache_type": "feature_split_cache",
        "dataset": "CREMA-D",
        "split_name": split_name,
        "config": cache_plan["config"],
        "sample_count": len(extracted["samples"]),
        "failed_count": len(extracted["failed_samples"]),
        "features": extracted["features"],
        "samples": extracted["samples"],
        "failed_samples": extracted["failed_samples"],
        "source_shards": [],
    }
    save_split_feature_cache(final_path, payload)
    save_failure_log(build_failure_log_path(final_path), extracted["failed_samples"])
    log(f"[CACHE] saved final split cache for {split_name}: {final_path}")
    return final_path


def load_features_and_labels_from_split_cache(cache_path: Path, label2idx: Dict[str, int]):
    import torch

    payload = load_split_feature_cache(cache_path)
    samples = payload.get("samples", [])
    labels = torch.tensor([label2idx[sample["label"]] for sample in samples], dtype=torch.long)
    return payload["features"], labels, samples, payload


def ensure_training_split_cache(
    split_name: str,
    split_samples: List[Dict],
    cache_plan: Dict[str, object],
    processor,
    model,
    args,
) -> Path:
    # Newly added cache/sharding logic: training reuses merged split caches, auto-merges full shard sets,
    # and avoids silently reusing mismatched caches when frame extraction settings change.
    existing_final = resolve_existing_split_cache(cache_plan, split_name)
    if existing_final is not None:
        return existing_final

    existing_shards = count_existing_shards(cache_plan)
    if args.total_shards > 1:
        if existing_shards == args.total_shards:
            return merge_feature_shards(cache_plan, delete_shards_after_merge=args.delete_shards_after_merge)
        if existing_shards > 0:
            missing = [str(path.name) for path in list_missing_shards(cache_plan)]
            raise RuntimeError(
                f"Partial shard cache state detected for {split_name}: "
                f"{existing_shards}/{args.total_shards} shard files present, missing={missing}. "
                f"Finish missing shards before training or run with total_shards=1."
            )

    return extract_split_feature_cache(
        split_samples,
        processor=processor,
        model=model,
        args=args,
        split_name=split_name,
        cache_plan=cache_plan,
    )


def compute_per_sample_focal_cross_entropy(
    logits,
    targets,
    gamma: float,
    class_weights=None,
    label_smoothing: float = 0.0,
):
    import torch.nn.functional as F

    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pt = probs.gather(dim=-1, index=targets.view(-1, 1)).squeeze(-1).clamp(min=1e-12, max=1.0)
    return ((1 - pt).clamp(min=0.0) ** gamma) * ce


def compute_adapter_logits_from_features(adapter, adapted_features, text_features):
    if hasattr(adapter, "compute_logits_from_adapted"):
        return adapter.compute_logits_from_adapted(adapted_features, text_features)

    import torch
    import torch.nn.functional as F

    txt = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sim = torch.einsum("bd,cpd->bcp", adapted_features, txt)

    if getattr(adapter, "use_prompt_weight", True):
        prompt_w = F.softmax(adapter.prompt_weight_logits, dim=-1).unsqueeze(0)
        class_sim = (sim * prompt_w).sum(dim=-1)
    else:
        class_sim = sim.mean(dim=-1)

    if getattr(adapter, "use_global_logit_scale", False):
        global_scale = adapter.logit_scale.exp().clamp(max=100.0)
    else:
        global_scale = 1.0
    if getattr(adapter, "use_class_temperature", True):
        class_scale = adapter.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
    else:
        class_scale = 1.0
    if getattr(adapter, "use_class_bias", True):
        class_bias = adapter.class_bias.unsqueeze(0)
    else:
        class_bias = 0.0
    return global_scale * class_sim * class_scale + class_bias


def causal_contrastive_loss(features, labels, actor_ids, temperature: float = 0.5):
    import torch
    import torch.nn.functional as F

    f = F.normalize(features, dim=-1)
    sim = f @ f.t()
    sim = sim / max(float(temperature), 1e-6)

    batch_size = f.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=f.device)
    labels = labels.view(-1)
    actor_ids = actor_ids.view(-1)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_actor = actor_ids.unsqueeze(0) == actor_ids.unsqueeze(1)
    pos_mask = same_label & (~same_actor) & (~eye)
    neg_mask = (~same_label) & (~eye)

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=f.device, requires_grad=True)

    sim_exp = sim.exp()
    pos_sum = (sim_exp * pos_mask).sum(dim=1)
    all_sum = (sim_exp * (pos_mask | neg_mask)).sum(dim=1)
    valid = has_pos & (pos_sum > 0) & (all_sum > 0)
    if not valid.any():
        return torch.tensor(0.0, device=f.device, requires_grad=True)
    return -torch.log(pos_sum[valid] / all_sum[valid]).mean()


def causal_feature_alignment_loss(features, labels, actor_ids):
    import torch

    total_var = torch.tensor(0.0, device=features.device)
    n_groups = 0
    for label in labels.unique():
        label_mask = labels == label
        if int(label_mask.sum().item()) < 2:
            continue
        label_feats = features[label_mask]
        label_actor_ids = actor_ids[label_mask]
        actor_means = []
        for actor_id in label_actor_ids.unique():
            actor_mask = label_actor_ids == actor_id
            if int(actor_mask.sum().item()) == 0:
                continue
            actor_means.append(label_feats[actor_mask].mean(dim=0))
        if len(actor_means) < 2:
            continue
        actor_means = torch.stack(actor_means, dim=0)
        total_var = total_var + actor_means.var(dim=0, unbiased=False).mean()
        n_groups += 1

    if n_groups == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)
    return total_var / n_groups


def counterfactual_feature_aug(features, labels, actor_ids, p: float = 0.3, n_replace_max: int = 3):
    import torch

    if features.ndim != 3:
        return features

    out = features.clone()
    _, num_frames, _ = out.shape
    max_replace = max(1, min(int(n_replace_max), num_frames))
    for index in range(out.shape[0]):
        if torch.rand(1, device=out.device).item() > p:
            continue
        candidates = ((labels == labels[index]) & (actor_ids != actor_ids[index])).nonzero(as_tuple=True)[0]
        if int(candidates.numel()) == 0:
            continue
        donor = candidates[torch.randint(int(candidates.numel()), (1,), device=out.device).item()]
        n_replace = int(torch.randint(1, max_replace + 1, (1,), device=out.device).item())
        replace_idx = torch.randperm(num_frames, device=out.device)[:n_replace]
        donor_idx = torch.randperm(num_frames, device=out.device)[:n_replace]
        out[index, replace_idx] = features[donor, donor_idx]
    return out


def _zero_loss_like(features):
    return features.sum() * 0.0


def sample_same_label_diff_group_partners(labels, group_ids):
    import torch

    partners = torch.full((labels.shape[0],), -1, dtype=torch.long, device=labels.device)
    for index in range(labels.shape[0]):
        candidates = ((labels == labels[index]) & (group_ids != group_ids[index])).nonzero(as_tuple=True)[0]
        if int(candidates.numel()) == 0:
            continue
        partner_index = torch.randint(int(candidates.numel()), (1,), device=labels.device).item()
        partners[index] = candidates[partner_index]
    return partners


def build_v2_counterfactual_batch(features, labels, group_ids, p: float = 0.5, pooled_mix_alpha: float = 0.3, eps: float = 1e-6):
    import torch

    out = features.clone()
    partners = sample_same_label_diff_group_partners(labels, group_ids)
    mix_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    trigger_mask = torch.rand(labels.shape[0], device=labels.device) < float(p)

    for index in range(features.shape[0]):
        donor = int(partners[index].item())
        if donor < 0 or not bool(trigger_mask[index].item()):
            continue
        mix_mask[index] = True
        if features.ndim == 3:
            source = features[index]
            donor_feat = features[donor]
            mu_source = source.mean(dim=0, keepdim=True)
            sig_source = source.std(dim=0, keepdim=True, unbiased=False).clamp(min=eps)
            mu_donor = donor_feat.mean(dim=0, keepdim=True)
            sig_donor = donor_feat.std(dim=0, keepdim=True, unbiased=False).clamp(min=eps)
            out[index] = sig_donor * (source - mu_source) / sig_source + mu_donor
        elif features.ndim == 2:
            mixed = torch.lerp(features[index], features[donor], float(pooled_mix_alpha))
            out[index] = mixed / mixed.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    return out, mix_mask, partners


def counterfactual_anchored_contrastive_loss(
    anchor_features,
    positive_features,
    anchor_labels,
    negative_features,
    negative_labels,
    temperature: float = 0.1,
):
    import torch
    import torch.nn.functional as F

    if anchor_features.numel() == 0:
        return _zero_loss_like(negative_features)

    anchors = F.normalize(anchor_features, dim=-1)
    positives = F.normalize(positive_features, dim=-1)
    negatives = F.normalize(negative_features, dim=-1)
    tau = max(float(temperature), 1e-6)

    loss_terms = []
    for index in range(anchors.shape[0]):
        neg_mask = negative_labels != anchor_labels[index]
        if not bool(neg_mask.any().item()):
            continue
        pos_score = torch.exp(torch.sum(anchors[index] * positives[index]) / tau)
        neg_scores = torch.exp((anchors[index].unsqueeze(0) @ negatives[neg_mask].t()).squeeze(0) / tau)
        denom = pos_score + neg_scores.sum()
        if torch.isfinite(denom) and float(denom.item()) > 0:
            loss_terms.append(-torch.log(pos_score / denom))

    if not loss_terms:
        return _zero_loss_like(negative_features)
    return torch.stack(loss_terms).mean()


def cfa_v2_text_anchor_loss(features, labels, group_ids, text_features, ema_state, momentum: float = 0.99, anchor_weight: float = 1.0):
    import torch
    import torch.nn.functional as F

    if features.ndim != 2:
        return _zero_loss_like(features)

    batch_global_means = {}
    batch_group_means = {}
    for label in labels.unique():
        label_idx = int(label.item())
        label_mask = labels == label
        if int(label_mask.sum().item()) == 0:
            continue
        batch_global_means[label_idx] = features[label_mask].mean(dim=0)
        label_groups = group_ids[label_mask]
        label_features = features[label_mask]
        for group_id in label_groups.unique():
            group_idx = int(group_id.item())
            group_mask = label_groups == group_id
            if int(group_mask.sum().item()) == 0:
                continue
            batch_group_means[(label_idx, group_idx)] = label_features[group_mask].mean(dim=0)

    if not batch_global_means:
        return _zero_loss_like(features)

    text_proto = text_features.mean(dim=1)
    text_proto = F.normalize(text_proto, dim=-1)

    inv_terms = []
    anchor_terms = []
    for label_idx, batch_mean in batch_global_means.items():
        anchor_terms.append(1.0 - F.cosine_similarity(
            F.normalize(batch_mean.unsqueeze(0), dim=-1),
            text_proto[label_idx].unsqueeze(0),
            dim=-1,
        ).mean())

    for key, batch_mean in batch_group_means.items():
        label_idx = key[0]
        target = ema_state["global"].get(label_idx)
        if target is None:
            continue
        inv_terms.append(F.mse_loss(batch_mean, target.detach(), reduction="mean"))

    loss = _zero_loss_like(features)
    if inv_terms:
        loss = loss + torch.stack(inv_terms).mean()
    if anchor_terms:
        loss = loss + (float(anchor_weight) * torch.stack(anchor_terms).mean())

    momentum = float(momentum)
    for label_idx, batch_mean in batch_global_means.items():
        batch_detached = batch_mean.detach().float()
        if label_idx in ema_state["global"]:
            ema_state["global"][label_idx] = momentum * ema_state["global"][label_idx] + (1.0 - momentum) * batch_detached
        else:
            ema_state["global"][label_idx] = batch_detached
    for key, batch_mean in batch_group_means.items():
        batch_detached = batch_mean.detach().float()
        if key in ema_state["group"]:
            ema_state["group"][key] = momentum * ema_state["group"][key] + (1.0 - momentum) * batch_detached
        else:
            ema_state["group"][key] = batch_detached

    return loss


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
    loss_type: str,
    focal_gamma: float,
    select_metric: str,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    use_amp: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    train_sample_weights=None,
    class_weights_override=None,
    force_loss_type=None,
    lr_scheduler_mode: str = "plateau",
    scheduler_min_lr: float = 1e-6,
    train_actor_ids=None,
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

    class_weights = class_weights_override.to(adapter.device) if class_weights_override is not None else None
    if class_weights is None and use_class_weight:
        class_counts = torch.bincount(train_y, minlength=len(EMOTION_LABELS)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    effective_loss_type = force_loss_type or loss_type

    if effective_loss_type == "focal":
        def compute_per_sample_loss(logits, targets):
            return compute_per_sample_focal_cross_entropy(
                logits,
                targets,
                gamma=focal_gamma,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing, reduction="none")

        def compute_per_sample_loss(logits, targets):
            return criterion(logits, targets)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler_mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=scheduler_min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, early_stopping_patience // 2),
            threshold=early_stopping_min_delta,
            min_lr=scheduler_min_lr,
        )
    amp_enabled = bool(use_amp and str(adapter.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    text_features_device = text_features.to(adapter.device)

    best_state = None
    best_val_metric = -1.0
    best_epoch_idx = -1
    no_improve_epochs = 0
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
    overall_start = time.time()
    cfa_v2_state = {"global": {}, "group": {}}

    if (
        use_causal_contrastive
        or use_causal_alignment
        or use_counterfactual_aug
        or use_cda_v2_mixstyle
        or use_ccl_v2_counterfactual
        or use_cfa_v2_textanchor
    ) and train_actor_ids is None:
        raise RuntimeError("train_actor_ids are required when causal training options are enabled")

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
        if train_actor_ids is not None:
            train_actor_ids = train_actor_ids[perm]
        if train_sample_weights is not None:
            train_sample_weights = train_sample_weights[perm]

        for start in range(0, train_x.shape[0], batch_size):
            batch_x = train_x[start:start + batch_size].to(adapter.device)
            batch_y = train_y[start:start + batch_size].to(adapter.device)
            batch_actor_ids = None
            if train_actor_ids is not None:
                batch_actor_ids = train_actor_ids[start:start + batch_size].to(adapter.device)
            batch_weight = None
            if train_sample_weights is not None:
                batch_weight = train_sample_weights[start:start + batch_size].to(adapter.device, dtype=torch.float32)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                cda_v2_loss_value = 0.0
                ccl_v2_loss_value = 0.0
                cfa_v2_loss_value = 0.0
                model_input_x = batch_x
                if use_counterfactual_aug and batch_actor_ids is not None:
                    model_input_x = counterfactual_feature_aug(
                        batch_x,
                        batch_y,
                        batch_actor_ids,
                        p=cda_prob,
                        n_replace_max=cda_n_replace_max,
                    )
                adapter_features = adapter._adapt_image(model_input_x)
                logits = compute_adapter_logits_from_features(adapter, adapter_features, text_features_device)
                per_sample_loss = compute_per_sample_loss(logits, batch_y)
                if batch_weight is not None:
                    loss = (per_sample_loss * batch_weight).mean()
                else:
                    loss = per_sample_loss.mean()

                if use_causal_contrastive and batch_actor_ids is not None:
                    loss = loss + (float(ccl_weight) * causal_contrastive_loss(
                        adapter_features,
                        batch_y,
                        batch_actor_ids,
                        temperature=ccl_temperature,
                    ))
                if use_causal_alignment and batch_actor_ids is not None:
                    loss = loss + (float(cfa_weight) * causal_feature_alignment_loss(
                        adapter_features,
                        batch_y,
                        batch_actor_ids,
                    ))
                if use_cda_v2_mixstyle and batch_actor_ids is not None:
                    cf_batch_x, cf_mask, _ = build_v2_counterfactual_batch(
                        batch_x,
                        batch_y,
                        batch_actor_ids,
                        p=cda_v2_prob,
                    )
                    if bool(cf_mask.any().item()):
                        cf_features = adapter._adapt_image(cf_batch_x)
                        cf_logits = compute_adapter_logits_from_features(adapter, cf_features, text_features_device)
                        cf_targets = batch_y[cf_mask]
                        cf_per_sample_loss = compute_per_sample_loss(cf_logits[cf_mask], cf_targets)
                        if batch_weight is not None:
                            cda_ce_loss = (cf_per_sample_loss * batch_weight[cf_mask]).mean()
                        else:
                            cda_ce_loss = cf_per_sample_loss.mean()
                        cda_kl_loss = torch.nn.functional.kl_div(
                            torch.nn.functional.log_softmax(cf_logits[cf_mask], dim=-1),
                            torch.nn.functional.softmax(logits[cf_mask].detach(), dim=-1),
                            reduction="batchmean",
                        )
                        cda_v2_loss = cda_ce_loss + (float(cda_v2_kl_weight) * cda_kl_loss)
                        loss = loss + cda_v2_loss
                        cda_v2_loss_value = float(cda_v2_loss.item())

                        if use_ccl_v2_counterfactual:
                            ccl_v2_loss = float(ccl_v2_weight) * counterfactual_anchored_contrastive_loss(
                                adapter_features[cf_mask],
                                cf_features[cf_mask],
                                batch_y[cf_mask],
                                adapter_features,
                                batch_y,
                                temperature=ccl_v2_temperature,
                            )
                            loss = loss + ccl_v2_loss
                            ccl_v2_loss_value = float(ccl_v2_loss.item())
                elif use_ccl_v2_counterfactual and batch_actor_ids is not None:
                    ccl_v2_loss = float(ccl_v2_weight) * causal_contrastive_loss(
                        adapter_features,
                        batch_y,
                        batch_actor_ids,
                        temperature=ccl_v2_temperature,
                    )
                    loss = loss + ccl_v2_loss
                    ccl_v2_loss_value = float(ccl_v2_loss.item())

                if use_cfa_v2_textanchor and batch_actor_ids is not None:
                    cfa_v2_loss = float(cfa_v2_weight) * cfa_v2_text_anchor_loss(
                        adapter_features,
                        batch_y,
                        batch_actor_ids,
                        text_features_device,
                        cfa_v2_state,
                        momentum=cfa_v2_ema_momentum,
                        anchor_weight=cfa_v2_anchor_weight,
                    )
                    loss = loss + cfa_v2_loss
                    cfa_v2_loss_value = float(cfa_v2_loss.item())

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
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
            val_true = [EMOTION_LABELS[int(item.item())] for item in val_y]
            val_acc = accuracy(val_true, val_pred)
            val_wf1 = weighted_f1(val_true, val_pred, EMOTION_LABELS)
            metric = val_wf1 if select_metric == "weighted_f1" else val_acc

        if lr_scheduler_mode == "cosine":
            scheduler.step()
        else:
            scheduler.step(metric)

        if metric > best_val_metric + early_stopping_min_delta:
            best_val_metric = metric
            best_epoch_idx = epoch_idx
            no_improve_epochs = 0
            best_state = copy.deepcopy(adapter.state_dict())
        else:
            no_improve_epochs += 1

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1) if epoch_idx + 1 < epochs else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
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
            f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0} | no_improve={no_improve_epochs} | lr={current_lr:.6e} | "
            f"epoch_time={time.strftime('%M:%S', time.gmtime(max(0, int(time.time() - epoch_start))))} | "
            f"elapsed={time.strftime('%M:%S', time.gmtime(max(0, int(elapsed))))} | "
            f"eta={time.strftime('%M:%S', time.gmtime(max(0, int(eta))))}"
        )

        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            log(
                f"[TRAIN] early stopping triggered at epoch {epoch_idx + 1}; "
                f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0}, best_metric={best_val_metric:.6f}"
            )
            break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone CREMA-D CLIP emotion training with strict frozen CLIP adapter")
    parser.add_argument("--cremad_root", default=DEFAULT_CREMAD_ROOT)
    parser.add_argument("--summary_table_path", default=None)
    parser.add_argument("--video_extensions", default=".flv,.mp4")
    parser.add_argument("--split_name", choices=["train", "val", "test"], default=None)
    parser.add_argument("--cv_mode", choices=["5fold", "split"], default="5fold")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_fold_offset", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--label_source", choices=["filename", "facevote"], default="filename")
    parser.add_argument("--prompt_template", default="The face looks <LABEL>.")
    parser.add_argument("--prompt_set", default="cremad_6_facial_cues")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None, help="Deprecated alias for --train_batch_size")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--frame_sampling_mode", choices=["uniform", "middle_late", "diff_guided"], default="uniform")
    parser.add_argument("--feature_layout", choices=["pooled", "sequence"], default="pooled")
    parser.add_argument("--sampling_window_start", type=float, default=0.4)
    parser.add_argument("--sampling_window_end", type=float, default=0.9)
    parser.add_argument("--diff_alpha", type=float, default=0.6)
    parser.add_argument("--diff_beta", type=float, default=0.4)
    parser.add_argument("--min_gap_ratio", type=float, default=0.08)
    parser.add_argument("--score_smooth_window", type=int, default=3)
    parser.add_argument("--frame_diff_metric", choices=["gray_l1", "gray_l2"], default="gray_l1")
    parser.add_argument("--ref_frame_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers for per-batch video decoding inside one shard process")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--disable_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--adapter_head_type", choices=["baseline", "stronger"], default="baseline")
    parser.add_argument("--temporal_head", choices=["none", "attention", "transformer"], default="none")
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--temporal_pool_mode", choices=["cls", "mean", "hybrid"], default="cls")
    parser.add_argument("--use_class_weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--disable_class_weight", dest="use_class_weight", action="store_false")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_facelevel_weight", action="store_true", default=False)
    parser.add_argument("--facelevel_low_thresh", type=float, default=60.0)
    parser.add_argument("--facelevel_high_thresh", type=float, default=75.0)
    parser.add_argument("--facelevel_low_weight", type=float, default=0.8)
    parser.add_argument("--facelevel_mid_weight", type=float, default=1.0)
    parser.add_argument("--facelevel_high_weight", type=float, default=1.2)
    parser.add_argument("--use_test_ensemble", dest="use_test_ensemble", action="store_true")
    parser.add_argument("--disable_test_ensemble", dest="use_test_ensemble", action="store_false")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--strict_frozen_clip", dest="strict_frozen_clip", action="store_true")
    parser.add_argument("--disable_strict_frozen_clip", dest="strict_frozen_clip", action="store_false")
    parser.add_argument("--use_global_logit_scale", dest="use_global_logit_scale", action="store_true")
    parser.add_argument("--disable_global_logit_scale", dest="use_global_logit_scale", action="store_false")
    parser.add_argument("--use_prompt_weight", dest="use_prompt_weight", action="store_true")
    parser.add_argument("--disable_prompt_weight", dest="use_prompt_weight", action="store_false")
    parser.add_argument("--use_class_temperature", dest="use_class_temperature", action="store_true")
    parser.add_argument("--disable_class_temperature", dest="use_class_temperature", action="store_false")
    parser.add_argument("--use_class_bias", dest="use_class_bias", action="store_true")
    parser.add_argument("--disable_class_bias", dest="use_class_bias", action="store_false")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--feature_cache_dir", default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--cache_tag", default="")
    parser.add_argument("--force_reextract", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--extract_only", action="store_true", help="Only extract image features for one split and exit")
    mode_group.add_argument("--merge_shards", action="store_true", help="Merge cached feature shards for one split and exit")
    parser.add_argument("--delete_shards_after_merge", action="store_true")
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log_file", default=None)
    parser.set_defaults(
        use_class_weight=False,
        use_test_ensemble=False,
        strict_frozen_clip=True,
        use_global_logit_scale=False,
        use_prompt_weight=False,
        use_class_temperature=False,
        use_class_bias=False,
        pin_memory=True,
        use_amp=True,
    )
    args = parser.parse_args()

    # `--extract_only` and `--merge_shards` are mutually exclusive and require `--split_name`.
    if (args.extract_only or args.merge_shards) and not args.split_name:
        parser.error("--split_name must be provided when using --extract_only or --merge_shards")

    # Keep both for compatibility; when provided, `gpu_id` overrides `device`.
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"

    # Validate shard arguments at parse time.
    if args.total_shards < 1:
        parser.error("--total_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.total_shards:
        parser.error(
            f"--shard_index must satisfy 0 <= shard_index < total_shards; "
            f"got shard_index={args.shard_index}, total_shards={args.total_shards}"
        )

    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience must be >= 0")
    if args.early_stopping_min_delta < 0:
        parser.error("--early_stopping_min_delta must be >= 0")
    if args.facelevel_low_thresh > args.facelevel_high_thresh:
        parser.error("--facelevel_low_thresh must be <= --facelevel_high_thresh")

    if args.batch_size is not None:
        args.train_batch_size = args.batch_size

    if args.temporal_head != "none":
        args.feature_layout = "sequence"
    if args.temporal_head != "none" and args.adapter_head_type != "baseline":
        parser.error("--temporal_head currently supports only --adapter_head_type baseline")
    if args.temporal_num_heads < 1:
        parser.error("--temporal_num_heads must be >= 1")
    if args.temporal_num_layers < 1:
        parser.error("--temporal_num_layers must be >= 1")
    if args.adapter_hidden_dim % args.temporal_num_heads != 0:
        parser.error("--adapter_hidden_dim must be divisible by --temporal_num_heads")

    args.batch_size = args.train_batch_size
    return args


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.frame_sampling_mode not in SUPPORTED_FRAME_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported frame_sampling_mode: {args.frame_sampling_mode}. "
            f"Supported modes: {sorted(SUPPORTED_FRAME_SAMPLING_MODES)}"
        )

    if args.gpu_id is not None and str(args.device).startswith("cuda"):
        args.device = f"cuda:{args.gpu_id}"
    if args.total_shards <= 0:
        raise ValueError("total_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.total_shards:
        raise ValueError(f"shard_index must be in [0, {args.total_shards - 1}]")
    if (args.extract_only or args.merge_shards) and not args.split_name:
        raise ValueError("split_name is required when using --extract_only or --merge_shards")

    if not args.strict_frozen_clip:
        raise ValueError("This CREMA-D script is configured to preserve the strict_frozen_clip adapter pipeline. Do not disable it.")
    if args.temporal_head != "none" and args.feature_layout != "sequence":
        raise ValueError("temporal_head requires feature_layout=sequence")

    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    resolved_log_file = init_log_file(args.log_file)
    atexit.register(close_log_file)
    if resolved_log_file:
        log(f"[LOG] writing log file to: {resolved_log_file}")

    allowed_extensions = parse_extension_set(args.video_extensions)
    samples, dataset_diagnostics = collect_cremad_samples(
        cremad_root=args.cremad_root,
        max_sequences=args.max_sequences,
        allowed_extensions=allowed_extensions,
        summary_table_path=args.summary_table_path,
    )
    if len(samples) < 10:
        raise RuntimeError(f"Too few valid CREMA-D samples: {len(samples)}")

    log(f"[DATA] valid samples: {len(samples)}")
    log(f"[DATA] actor_count={dataset_diagnostics['actor_count']} class_distribution={dataset_diagnostics['class_distribution']}")
    log(f"[DATA] extension_distribution={dataset_diagnostics['extension_distribution']}")
    if dataset_diagnostics.get("resolved_summary_table_path"):
        log(f"[DATA] summaryTable loaded: {dataset_diagnostics['resolved_summary_table_path']}")
    else:
        log("[WARN] summaryTable.csv not found locally; FaceVote/FaceLevel auxiliary metadata will be empty.")
    if dataset_diagnostics.get("missing_summary_examples"):
        log(f"[DATA] missing summary rows examples: {dataset_diagnostics['missing_summary_examples']}")
    if dataset_diagnostics.get("missing_from_video_examples"):
        log(f"[DATA] expected-but-missing video examples: {dataset_diagnostics['missing_from_video_examples']}")

    splits, split_info, benchmark_mode = resolve_splits(samples, args)

    train_samples = build_split_samples_with_index(splits["train"], split_name="train")
    val_samples = build_split_samples_with_index(splits["val"], split_name="val")
    test_samples = build_split_samples_with_index(splits["test"], split_name="test")
    split_samples_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    log(
        f"[DATA] split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}"
    )
    log(
        f"[DATA] split actors -> train: {split_info['train']['actor_count']}, "
        f"val: {split_info['val']['actor_count']}, test: {split_info['test']['actor_count']}"
    )
    log(f"[MODE] feature_layout={args.feature_layout} temporal_head={args.temporal_head} adapter_head_type={args.adapter_head_type}")

    split_cache_plans = {
        split_name: build_split_cache_plan(args.feature_cache_dir, "CREMA-D", split_name, split_samples_map[split_name], args)
        for split_name in ["train", "val", "test"]
    }
    for split_name, cache_plan in split_cache_plans.items():
        log(
            f"[CACHE] plan for {split_name}: final={cache_plan['final_path']} "
            f"existing_shards={count_existing_shards(cache_plan)}/{args.total_shards}"
        )

    if args.merge_shards:
        merged_path = merge_feature_shards(
            split_cache_plans[args.split_name],
            delete_shards_after_merge=args.delete_shards_after_merge,
        )
        log(f"[DONE] merge-only mode finished: {merged_path}")
        return

    prompt_groups = build_class_prompts(args.prompt_template, args.prompt_set)

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
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    log(f"[INFO] model loaded: {args.model_id} on {args.device}")

    if args.extract_only:
        target_samples = split_samples_map[args.split_name]
        if args.total_shards == 1:
            extract_split_feature_cache(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        else:
            extract_feature_shard(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        log("[DONE] extraction-only mode finished")
        return

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

    resolved_split_cache_paths: Dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        resolved_path = ensure_training_split_cache(
            split_name=split_name,
            split_samples=split_samples_map[split_name],
            cache_plan=split_cache_plans[split_name],
            processor=processor,
            model=model,
            args=args,
        )
        resolved_split_cache_paths[split_name] = str(resolved_path)

    train_x, train_y, train_samples, train_cache_payload = load_features_and_labels_from_split_cache(
        split_cache_plans["train"]["final_path"],
        label2idx,
    )
    val_x, val_y, val_samples, val_cache_payload = load_features_and_labels_from_split_cache(
        split_cache_plans["val"]["final_path"],
        label2idx,
    )
    test_x, test_y, test_samples, test_cache_payload = load_features_and_labels_from_split_cache(
        split_cache_plans["test"]["final_path"],
        label2idx,
    )

    train_sample_weights = None
    if args.use_facelevel_weight:
        train_weight_samples = train_samples
        if not any(sample.get("face_level") not in {None, ""} for sample in train_weight_samples):
            current_split_train_samples = split_samples_map["train"]
            same_order = len(train_weight_samples) == len(current_split_train_samples) and all(
                cache_sample.get("sequence_id") == current_sample.get("sequence_id")
                and int(cache_sample.get("sample_index", -1)) == int(current_sample.get("sample_index", -2))
                for cache_sample, current_sample in zip(train_weight_samples, current_split_train_samples)
            )
            if same_order:
                train_weight_samples = current_split_train_samples
                log("[FACELEVEL] cache samples missing face_level metadata; using current split metadata for sample weighting")

        train_sample_weights, facelevel_weight_stats = build_facelevel_sample_weights(train_weight_samples, args)
        log("[FACELEVEL] FaceLevel weighting enabled")
        log(
            f"[FACELEVEL] thresholds: low<{args.facelevel_low_thresh:.3f}, "
            f"mid=[{args.facelevel_low_thresh:.3f},{args.facelevel_high_thresh:.3f}), "
            f"high>={args.facelevel_high_thresh:.3f}"
        )
        log(
            f"[FACELEVEL] weights: low={args.facelevel_low_weight:.3f}, "
            f"mid={args.facelevel_mid_weight:.3f}, high={args.facelevel_high_weight:.3f}"
        )
        log(
            f"[FACELEVEL] train stats: missing={facelevel_weight_stats['num_missing_facelevel']}, "
            f"low={facelevel_weight_stats['num_low']}, mid={facelevel_weight_stats['num_mid']}, "
            f"high={facelevel_weight_stats['num_high']}, mean={facelevel_weight_stats['weight_mean']:.6f}, "
            f"min={facelevel_weight_stats['weight_min']:.6f}, max={facelevel_weight_stats['weight_max']:.6f}"
        )

    text_features = extract_text_features(prompt_groups, processor, model, args.device)

    feature_dim = int(train_x.shape[-1])
    if args.temporal_head == "attention":
        adapter = TemporalClipImageAdapter(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            num_frames=args.num_frames,
            use_global_logit_scale=args.use_global_logit_scale,
            use_prompt_weight=args.use_prompt_weight,
            use_class_temperature=args.use_class_temperature,
            use_class_bias=args.use_class_bias,
        )
    elif args.temporal_head == "transformer":
        adapter = TemporalTransformerClipImageAdapter(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            num_frames=args.num_frames,
            temporal_num_heads=args.temporal_num_heads,
            temporal_num_layers=args.temporal_num_layers,
            temporal_pool_mode=args.temporal_pool_mode,
            use_global_logit_scale=args.use_global_logit_scale,
            use_prompt_weight=args.use_prompt_weight,
            use_class_temperature=args.use_class_temperature,
            use_class_bias=args.use_class_bias,
        )
    else:
        adapter_cls = StrongerClipImageAdapter if args.adapter_head_type == "stronger" else ClipImageAdapter
        adapter = adapter_cls(
            dim=feature_dim,
            device=args.device,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.adapter_dropout,
            num_classes=len(EMOTION_LABELS),
            num_prompts=int(text_features.shape[1]),
            use_global_logit_scale=args.use_global_logit_scale,
            use_prompt_weight=args.use_prompt_weight,
            use_class_temperature=args.use_class_temperature,
            use_class_bias=args.use_class_bias,
        )
    adapter = train_strict_frozen_clip(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        text_features=text_features,
        adapter=adapter,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=args.use_class_weight,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        select_metric=args.select_metric,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
        use_amp=args.use_amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        train_sample_weights=train_sample_weights,
    )

    val_pred = predict_emotion_from_features(
        val_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )
    test_pred = predict_emotion_from_features(
        test_x,
        text_features,
        adapter,
        idx2label,
        args.train_batch_size,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
    )

    val_true = [EMOTION_LABELS[int(item.item())] for item in val_y]
    test_true = [EMOTION_LABELS[int(item.item())] for item in test_y]
    val_summary = summarize_predictions(val_true, val_pred)
    test_summary = summarize_predictions(test_true, test_pred)

    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else default_checkpoint_path(Path(args.output))

    result = {
        "config": {
            "method": "clip_supervised_text_image_emotion",
            "execution_mode": "strict_frozen_clip_adapter",
            "dataset": "CREMA-D",
            "task": "emotion",
            "benchmark_mode": benchmark_mode,
            "cv_mode": args.cv_mode,
            "fold_idx": args.fold_idx if args.cv_mode == "5fold" else None,
            "train_ratio": args.train_ratio if args.cv_mode == "split" else None,
            "val_ratio": args.val_ratio if args.cv_mode == "split" else None,
            "cremad_root": str(Path(args.cremad_root).resolve()),
            "summary_table_path": dataset_diagnostics.get("resolved_summary_table_path"),
            "primary_label_source": "filename_emotion",
            "auxiliary_facevote_loaded": dataset_diagnostics.get("auxiliary_facevote_loaded", False),
            "actor_count": dataset_diagnostics.get("actor_count", 0),
            "class_count": len(EMOTION_LABELS),
            "video_extensions": sorted(allowed_extensions),
            "model_id": args.model_id,
            "prompt_template": args.prompt_template,
            "prompt_set": args.prompt_set,
            "epochs": args.epochs,
            "extract_batch_size": args.extract_batch_size,
            "train_batch_size": args.train_batch_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "num_frames": args.num_frames,
            "feature_layout": args.feature_layout,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "adapter_head_type": args.adapter_head_type,
            "temporal_head": args.temporal_head,
            "temporal_num_heads": args.temporal_num_heads,
            "temporal_num_layers": args.temporal_num_layers,
            "temporal_pool_mode": args.temporal_pool_mode,
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "select_metric": args.select_metric,
            "use_facelevel_weight": args.use_facelevel_weight,
            "facelevel_low_thresh": args.facelevel_low_thresh,
            "facelevel_high_thresh": args.facelevel_high_thresh,
            "facelevel_low_weight": args.facelevel_low_weight,
            "facelevel_mid_weight": args.facelevel_mid_weight,
            "facelevel_high_weight": args.facelevel_high_weight,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "strict_frozen_clip": args.strict_frozen_clip,
            "use_global_logit_scale": args.use_global_logit_scale,
            "use_prompt_weight": args.use_prompt_weight,
            "use_class_temperature": args.use_class_temperature,
            "use_class_bias": args.use_class_bias,
            "use_amp": args.use_amp,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "cache_tag": args.cache_tag,
            "force_reextract": args.force_reextract,
            "label_source": args.label_source,
            "frame_sampling_mode": args.frame_sampling_mode,
            "sampling_window_start": args.sampling_window_start,
            "sampling_window_end": args.sampling_window_end,
            "diff_alpha": args.diff_alpha,
            "diff_beta": args.diff_beta,
            "min_gap_ratio": args.min_gap_ratio,
            "score_smooth_window": args.score_smooth_window,
            "frame_diff_metric": args.frame_diff_metric,
            "ref_frame_ratio": args.ref_frame_ratio,
            "num_folds": args.num_folds,
            "val_fold_offset": args.val_fold_offset,
            "total_shards": args.total_shards,
            "feature_cache_dir": args.feature_cache_dir,
            "resolved_feature_cache_paths": resolved_split_cache_paths,
            "checkpoint_output": str(checkpoint_path),
            "log_file": resolved_log_file,
            "seed": args.seed,
            "max_sequences": args.max_sequences,
        },
        "dataset": {
            "total": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "actor_count": dataset_diagnostics.get("actor_count", 0),
            "class_distribution_total": dataset_diagnostics.get("class_distribution", {}),
            "extension_distribution_total": dataset_diagnostics.get("extension_distribution", {}),
            "missing_summary_examples": dataset_diagnostics.get("missing_summary_examples", []),
            "missing_from_video_examples": dataset_diagnostics.get("missing_from_video_examples", []),
            "extra_video_examples": dataset_diagnostics.get("extra_video_examples", []),
            "duplicate_sequence_replacements": dataset_diagnostics.get("duplicate_sequence_replacements", 0),
            "invalid_name_examples": dataset_diagnostics.get("invalid_name_examples", []),
            "split_summary": split_info,
            "failed_samples_train": train_cache_payload.get("failed_count", 0),
            "failed_samples_val": val_cache_payload.get("failed_count", 0),
            "failed_samples_test": test_cache_payload.get("failed_count", 0),
            "label_distribution_train": dict(Counter(sample["label"] for sample in train_samples)),
            "label_distribution_val": dict(Counter(sample["label"] for sample in val_samples)),
            "label_distribution_test": dict(Counter(sample["label"] for sample in test_samples)),
            "extension_distribution_train": dict(Counter(sample["ext"] for sample in train_samples)),
            "extension_distribution_val": dict(Counter(sample["ext"] for sample in val_samples)),
            "extension_distribution_test": dict(Counter(sample["ext"] for sample in test_samples)),
        },
        "label_map": CREMAD_LABEL_MAP,
        "prompt_groups": prompt_groups,
        "val": val_summary,
        "test": test_summary,
        "learned_global_logit_scale": round(float(adapter.logit_scale.exp().detach().cpu().item()), 6),
        "val_prediction_distribution": val_summary["prediction_distribution"],
        "test_prediction_distribution": test_summary["prediction_distribution"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "checkpoint_type": "strict_frozen_clip_adapter",
        "config": result["config"],
        "dataset": result["dataset"],
        "label_map": result["label_map"],
        "metrics": {"val": result["val"], "test": result["test"]},
        "prompt_groups": prompt_groups,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "adapter_state_dict": adapter.state_dict(),
        "text_features": text_features.cpu(),
        "output_path": str(output_path),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    log(f"[DONE] saved CREMA-D report to: {output_path}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    log(f"[DONE] final test metrics: {json.dumps(result['test'], ensure_ascii=False)}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
