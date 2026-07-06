#!/usr/bin/env python3
"""YawDD CLIP visual-backbone fine-tuning comparison.

Runs under the locked YawDD B-prime protocol:
driver-disjoint video-level split, split seed 42, T=10 diff-guided frames,
frozen CLIP text side, Adapter + PCH head, and the same result payload helpers
as the canonical full-model runs.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CURRENT_SRC_DIR = Path(__file__).resolve().parent
import sys

if str(CURRENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_SRC_DIR))

import clip_cremad_emotion_train as cremad_base
import clip_ravdess_emotion_train as ravdess_base
import clip_yawdd_emotion_train as yawdd_base


VARIANTS = ["partial_visual_last2", "full_visual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YawDD CLIP visual backbone fine-tuning under locked B-prime protocol")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--yawdd_root", default=yawdd_base.DEFAULT_YAWDD_ROOT)
    parser.add_argument("--label_mode", choices=["binary"], default="binary")
    parser.add_argument("--include_dash", action="store_true")
    parser.add_argument("--eval_mode", choices=["single"], default="single")
    parser.add_argument("--cv_mode", choices=["split"], default="split")
    parser.add_argument("--split_mode", choices=["speaker_independent"], default="speaker_independent")
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video_extensions", default=".avi,.mp4,.mov,.mkv")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="offline_only")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="The driver looks <LABEL>.")
    parser.add_argument("--prompt_set", default="yawdd_facial_cues")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_sampling_mode", choices=["diff_guided"], default="diff_guided")
    parser.add_argument("--feature_layout", choices=["pooled"], default="pooled")
    parser.add_argument("--sampling_window_start", type=float, default=0.4)
    parser.add_argument("--sampling_window_end", type=float, default=0.9)
    parser.add_argument("--diff_alpha", type=float, default=0.6)
    parser.add_argument("--diff_beta", type=float, default=0.4)
    parser.add_argument("--min_gap_ratio", type=float, default=0.08)
    parser.add_argument("--score_smooth_window", type=int, default=3)
    parser.add_argument("--frame_diff_metric", choices=["gray_l1", "gray_l2"], default="gray_l1")
    parser.add_argument("--ref_frame_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--disable_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_dropout", type=float, default=0.3)
    parser.add_argument("--adapter_mode", choices=["full"], default="full")
    parser.add_argument("--pool_adapter_variant", choices=["legacy"], default="legacy")
    parser.add_argument("--loss_type", choices=["focal", "ce"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--use_class_weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--disable_class_weight", dest="use_class_weight", action="store_false")
    parser.add_argument("--select_metric", choices=["weighted_f1", "accuracy"], default="weighted_f1")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_mode", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--pixel_cache_dir", default=None)
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_file", default=None)
    parser.set_defaults(use_class_weight=True, use_amp=True, pin_memory=True)
    args = parser.parse_args()
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"

    # Fields expected by the shared YawDD result payload.
    args.baseline_mode = f"backbone_finetune_{args.variant}"
    args.lr = args.adapter_lr
    args.fold_idx = 0
    args.total_shards = 1
    args.shard_index = 0
    args.delete_shards_after_merge = False
    args.feature_cache_dir = ""
    args.temporal_head = "none"
    args.temporal_module = "none"
    args.temporal_num_heads = 4
    args.temporal_num_layers = 1
    args.temporal_pool_mode = "mean"
    args.no_frame_gate = False
    args.no_gem = False
    args.no_residual_blend = False
    args.gem_init_p = 1.0
    args.use_test_ensemble = False
    args.ensemble_group_size = 2
    args.adapter_use_prompt_weight = "on"
    args.adapter_use_class_temperature = "on"
    args.adapter_use_class_bias = "on"
    args.resolved_use_prompt_weight = True
    args.resolved_use_class_temperature = True
    args.resolved_use_class_bias = True
    args.run_zero_shot_eval = False
    args.force_reextract = False
    args.all_face_image = None
    args.external_face_root = None
    args.dlib_shape_predictor = None
    for name, value in {
        "multi4_oversample_labels": "",
        "multi4_oversample_min_count": 0,
        "multi4_oversample_mode": "noise",
        "multi4_oversample_noise_std": 0.0,
        "multi4_mixup_alpha": 0.4,
        "use_causal_contrastive": False,
        "ccl_weight": 0.5,
        "ccl_temperature": 0.5,
        "use_causal_alignment": False,
        "cfa_weight": 0.1,
        "use_counterfactual_aug": False,
        "cda_prob": 0.3,
        "cda_n_replace_max": 3,
        "use_cda_v2_mixstyle": False,
        "cda_v2_prob": 0.5,
        "cda_v2_kl_weight": 0.5,
        "use_ccl_v2_counterfactual": False,
        "ccl_v2_weight": 0.1,
        "ccl_v2_temperature": 0.1,
        "use_cfa_v2_textanchor": False,
        "cfa_v2_weight": 0.05,
        "cfa_v2_anchor_weight": 1.0,
        "cfa_v2_ema_momentum": 0.99,
    }.items():
        setattr(args, name, value)
    return args


def log(message: str) -> None:
    yawdd_base.log(message)


def setup_logging(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    yawdd_base.init_log_file(args.log_file)
    atexit.register(yawdd_base.close_log_file)
    log(f"[ARGS] saved args snapshot to: {output_dir / 'args.json'}")
    log(f"[LOG] writing log file to: {args.log_file}")


def prepare_protocol(args):
    emotion_labels, _, _ = yawdd_base.resolve_label_space("binary")
    cremad_base.EMOTION_LABELS = list(emotion_labels)
    prompt_groups = yawdd_base.build_class_prompts(args.label_mode, args.prompt_template, args.prompt_set)
    samples, diagnostics = yawdd_base.collect_yawdd_samples(
        yawdd_root=args.yawdd_root,
        label_mode=args.label_mode,
        include_dash=args.include_dash,
        max_sequences=args.max_sequences,
    )
    splits, split_info = cremad_base.split_cremad_samples_actor_ratio(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    log(
        f"[DATA] samples={len(samples)} train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} split=split_subject_independent"
    )
    return emotion_labels, prompt_groups, splits, split_info, diagnostics


def pixel_cache_path(cache_dir: Optional[Path], sample: Dict, args, split_name: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    key = "||".join([
        split_name,
        str(sample.get("sequence_id", "")),
        str(sample.get("video_path", "")),
        str(args.model_id),
        str(args.num_frames),
        str(args.frame_sampling_mode),
        str(args.sampling_window_start),
        str(args.sampling_window_end),
        str(args.diff_alpha),
        str(args.diff_beta),
        str(args.min_gap_ratio),
        str(args.score_smooth_window),
        str(args.frame_diff_metric),
        str(args.ref_frame_ratio),
        "clip_processor_pixel_values",
    ])
    return cache_dir / split_name / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pt"


def build_pixel_values(sample: Dict, args, processor):
    import torch

    images, _, _ = ravdess_base.read_sampled_media(
        sample,
        args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
    )
    if not images:
        raise RuntimeError(f"No frames available for sample: {sample.get('sequence_id')}")
    if len(images) < args.num_frames:
        images = list(images) + [images[-1]] * (args.num_frames - len(images))
    inputs = processor(images=[image.convert("RGB") for image in images[: args.num_frames]], return_tensors="pt", padding=True)
    return inputs["pixel_values"]


def ensure_pixel_cache(split_name: str, samples: List[Dict], args, processor, cache_dir: Optional[Path]) -> Dict[str, object]:
    if cache_dir is None:
        return {"enabled": False}
    (cache_dir / split_name).mkdir(parents=True, exist_ok=True)
    created = 0
    reused = 0
    for index, sample in enumerate(samples, start=1):
        path = pixel_cache_path(cache_dir, sample, args, split_name)
        if path.exists():
            reused += 1
        else:
            tensor = build_pixel_values(sample, args, processor)
            tmp = path.with_suffix(".tmp")
            import torch

            torch.save(tensor.half(), tmp)
            tmp.replace(path)
            created += 1
        if index % 25 == 0 or index == len(samples):
            log(f"[PIXEL_CACHE] {split_name}: {index}/{len(samples)} created={created} reused={reused}")
    return {"enabled": True, "cache_dir": str(cache_dir), "split": split_name, "created": created, "reused": reused, "total": len(samples)}


class ClipPixelVideoDataset:
    def __init__(self, samples: List[Dict], label2idx: Dict[str, int], args, processor, split_name: str, cache_dir: Optional[Path]):
        self.samples = list(samples)
        self.label2idx = dict(label2idx)
        self.args = args
        self.processor = processor
        self.split_name = split_name
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        path = pixel_cache_path(self.cache_dir, sample, self.args, self.split_name)
        if path is not None and path.exists():
            pixel_values = torch.load(path, map_location="cpu").float()
        else:
            pixel_values = build_pixel_values(sample, self.args, self.processor)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pixel_values.half(), path)
        return pixel_values, self.label2idx[sample["label"]], sample["sequence_id"]


def freeze_all_clip(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def configure_visual_unfreeze(model, variant: str) -> List:
    freeze_all_clip(model)
    trainable = []
    if variant == "partial_visual_last2":
        layers = list(model.vision_model.encoder.layers)
        for layer in layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
                trainable.append(param)
    elif variant == "full_visual":
        for param in model.vision_model.parameters():
            param.requires_grad = True
            trainable.append(param)
        for param in model.visual_projection.parameters():
            param.requires_grad = True
            trainable.append(param)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return trainable


def encode_video_features(model, pixel_values, args):
    import torch
    import torch.nn.functional as F

    bsz, frame_count = pixel_values.shape[:2]
    flat = pixel_values.view(bsz * frame_count, *pixel_values.shape[2:]).to(args.device)
    autocast_enabled = bool(args.use_amp and str(args.device).startswith("cuda"))
    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        feats = model.get_image_features(pixel_values=flat)
        feats = F.normalize(feats, dim=-1)
        feats = feats.view(bsz, frame_count, -1).mean(dim=1)
        feats = F.normalize(feats, dim=-1)
    return feats


def compute_per_sample_loss(logits, targets, args, class_weights):
    import torch.nn as nn

    if args.loss_type == "focal":
        return cremad_base.compute_per_sample_focal_cross_entropy(
            logits,
            targets,
            gamma=args.focal_gamma,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing, reduction="none")
    return criterion(logits, targets)


def evaluate(model, adapter, loader, text_features, idx2label, args):
    import torch

    model.eval()
    adapter.eval()
    y_true, y_pred = [], []
    text_features_device = text_features.to(args.device)
    with torch.no_grad():
        for pixel_values, targets, _ in loader:
            targets = targets.to(args.device)
            feats = encode_video_features(model, pixel_values, args)
            logits = adapter.logits(feats, text_features_device)
            y_true.extend([idx2label[int(item)] for item in targets.detach().cpu().tolist()])
            y_pred.extend([idx2label[int(item)] for item in logits.argmax(dim=-1).detach().cpu().tolist()])
    return y_true, y_pred


def train_finetune_model(args, model, adapter, loaders, text_features, emotion_labels):
    import torch

    train_loader, val_loader = loaders["train"], loaders["val"]
    idx2label = {idx: label for idx, label in enumerate(emotion_labels)}
    backbone_params = configure_visual_unfreeze(model, args.variant)
    adapter_params = list(adapter.parameters())
    class_weights = None
    if args.use_class_weight:
        train_labels = []
        for _, targets, _ in train_loader:
            train_labels.extend([int(item) for item in targets.tolist()])
        counts = torch.bincount(torch.tensor(train_labels), minlength=len(emotion_labels)).float()
        class_weights = (counts.sum() / counts.clamp(min=1.0)).to(args.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": args.adapter_lr, "name": "adapter_pch"},
            {"params": backbone_params, "lr": args.backbone_lr, "name": "clip_visual_backbone"},
        ],
        weight_decay=args.weight_decay,
    )
    if args.lr_scheduler_mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.scheduler_min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, args.early_stopping_patience // 2),
            threshold=args.early_stopping_min_delta,
            min_lr=args.scheduler_min_lr,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and str(args.device).startswith("cuda")))
    text_features_device = text_features.to(args.device)
    best = None
    best_metric = float("-inf")
    best_epoch = -1
    no_improve = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.eval()
        adapter.train()
        running = 0.0
        batches = 0
        epoch_start = time.time()
        for pixel_values, targets, _ in train_loader:
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp and str(args.device).startswith("cuda"))):
                feats = encode_video_features(model, pixel_values, args)
                logits = adapter.logits(feats, text_features_device)
                loss_items = compute_per_sample_loss(logits, targets, args, class_weights)
                loss = loss_items.mean()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params + backbone_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            batches += 1
        val_true, val_pred = evaluate(model, adapter, val_loader, text_features, idx2label, args)
        val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, args.label_mode)
        metric = yawdd_base.select_metric_value(args.select_metric, val_summary, args.label_mode, val_true, val_pred, emotion_labels)
        if args.lr_scheduler_mode == "cosine":
            scheduler.step()
        else:
            scheduler.step(metric)
        if metric > best_metric + args.early_stopping_min_delta:
            best_metric = metric
            best_epoch = epoch + 1
            best = {
                "model_state_dict": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "adapter_state_dict": copy.deepcopy(adapter.state_dict()),
            }
            no_improve = 0
        else:
            no_improve += 1
        log(
            f"[TRAIN] {args.variant} epoch {epoch + 1}/{args.epochs} loss={running / max(1, batches):.6f} "
            f"val_metric={metric:.6f} best={best_metric:.6f} best_epoch={best_epoch} no_improve={no_improve} "
            f"epoch_sec={time.time() - epoch_start:.1f} elapsed={time.time() - start_time:.1f}"
        )
        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            log(f"[TRAIN] early stopping at epoch {epoch + 1}; best_epoch={best_epoch} best_metric={best_metric:.6f}")
            break
    if best is None:
        raise RuntimeError("No best checkpoint captured")
    model.load_state_dict(best["model_state_dict"], strict=True)
    adapter.load_state_dict(best["adapter_state_dict"])
    return best, {"best_epoch": best_epoch, "best_val_metric": float(best_metric), "trainable_backbone_param_count": sum(p.numel() for p in backbone_params)}


def summarize_and_write(args, emotion_labels, prompt_groups, splits, split_info, diagnostics, loaders, model, adapter, text_features, cache_info, train_info):
    import torch

    idx2label = {idx: label for idx, label in enumerate(emotion_labels)}
    val_true, val_pred = evaluate(model, adapter, loaders["val"], text_features, idx2label, args)
    test_true, test_pred = evaluate(model, adapter, loaders["test"], text_features, idx2label, args)
    val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, args.label_mode)
    test_summary = yawdd_base.summarize_predictions_by_mode(test_true, test_pred, emotion_labels, args.label_mode)
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    fake_bundle = {
        "train_cache_samples": list(splits["train"]),
        "val_cache_samples": list(splits["val"]),
        "test_cache_samples": list(splits["test"]),
        "train_payload": {"failed_count": 0},
        "val_payload": {"failed_count": 0},
        "test_payload": {"failed_count": 0},
        "resolved_paths": {},
    }
    class_names = yawdd_base.resolve_class_names(args.label_mode, emotion_labels)
    class_distribution = yawdd_base.compute_split_class_distribution(
        emotion_labels,
        class_names,
        fake_bundle["train_cache_samples"],
        fake_bundle["val_cache_samples"],
        fake_bundle["test_cache_samples"],
    )
    dataset_summary = yawdd_base.build_standard_yawdd_dataset_summary(
        args=args,
        dataset_diagnostics=diagnostics,
        split_info=split_info,
        train_samples=splits["train"],
        val_samples=splits["val"],
        test_samples=splits["test"],
        train_cache_payload=fake_bundle["train_payload"],
        val_cache_payload=fake_bundle["val_payload"],
        test_cache_payload=fake_bundle["test_payload"],
        train_cache_samples=fake_bundle["train_cache_samples"],
        val_cache_samples=fake_bundle["val_cache_samples"],
        test_cache_samples=fake_bundle["test_cache_samples"],
    )
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(args.output))
    extra_config = {
        "comparison_variant": args.variant,
        "fine_tuning_scope": "CLIP visual backbone fine-tuning",
        "text_encoder_trainable": False,
        "adapter_pch_trainable": True,
        "adapter_lr": args.adapter_lr,
        "backbone_lr": args.backbone_lr,
        "pixel_cache": cache_info,
        **train_info,
    }
    if args.variant == "partial_visual_last2":
        extra_config["unfrozen_clip_visual_modules"] = "vision_model.encoder.layers[-2:] only"
    else:
        extra_config["unfrozen_clip_visual_modules"] = "vision_model + visual_projection"
    result = yawdd_base.build_result_payload(
        args=args,
        label_mode=args.label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode="split_subject_independent",
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        resolved_feature_cache_paths={},
        output_path_value=args.output,
        checkpoint_path=checkpoint_path,
        extra_config=extra_config,
    )
    yawdd_base.write_result_payload(result, args.output)
    yawdd_base.save_checkpoint_payload(
        checkpoint_path,
        {
            "checkpoint_type": args.baseline_mode,
            "variant": args.variant,
            "adapter_state_dict": adapter.state_dict(),
            "clip_model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "text_features": text_features.detach().cpu(),
            "config": result["config"],
        },
    )
    log(f"[DONE] saved report to: {args.output}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    setup_logging(args)
    yawdd_base.configure_random_seeds(args.training_seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    emotion_labels, prompt_groups, splits, split_info, diagnostics = prepare_protocol(args)

    processor, model = yawdd_base.load_clip_components(args.model_id, args.device, yawdd_base.resolve_effective_clip_mode(args, args.label_mode))
    # The frozen-feature path loads CLIP in fp16 on CUDA. For gradient updates
    # with GradScaler, trainable CLIP parameters must be fp32.
    model = model.float()
    text_features = yawdd_base.aide_base.extract_text_features(prompt_groups, processor, model, args.device).detach()
    cache_dir = Path(args.pixel_cache_dir) if args.pixel_cache_dir else Path(args.output).parents[1] / "pixel_cache"
    cache_info = {
        split: ensure_pixel_cache(split, splits[split], args, processor, cache_dir)
        for split in ["train", "val", "test"]
    }
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    datasets = {
        split: ClipPixelVideoDataset(splits[split], label2idx, args, processor, split, cache_dir)
        for split in ["train", "val", "test"]
    }
    import torch
    from torch.utils.data import DataLoader

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory),
        "val": DataLoader(datasets["val"], batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory),
        "test": DataLoader(datasets["test"], batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory),
    }
    adapter = yawdd_base.aide_base.ClipImageAdapter(
        dim=int(text_features.shape[-1]),
        device=args.device,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        num_classes=len(emotion_labels),
        num_prompts=int(text_features.shape[1]),
        use_prompt_weight=True,
        use_class_temperature=True,
        use_class_bias=True,
        adapter_mode=args.adapter_mode,
    )
    train_info = {}
    try:
        _, train_info = train_finetune_model(args, model, adapter, loaders, text_features, emotion_labels)
        summarize_and_write(args, emotion_labels, prompt_groups, splits, split_info, diagnostics, loaders, model, adapter, text_features, cache_info, train_info)
    finally:
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
