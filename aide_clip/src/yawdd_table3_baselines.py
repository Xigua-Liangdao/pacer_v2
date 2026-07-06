#!/usr/bin/env python3
"""YawDD Table III baseline backfill runner.

This file is intentionally separate from clip_yawdd_emotion_train.py so the
locked full-model path stays untouched. It reuses the same YawDD collection,
driver-disjoint split, frozen-CLIP feature cache, metrics, and result JSON
payload helpers as the canonical B-prime runs.
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
REPO_ROOT = CURRENT_SRC_DIR.parents[1]
DEFAULT_TABLE3_PRETRAINED_DIR = REPO_ROOT / "results" / "final_runs" / "yawdd" / "B_table3_baselines" / "pretrained"
import sys

if str(CURRENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_SRC_DIR))

import clip_cremad_emotion_train as cremad_base
import clip_ravdess_emotion_train as ravdess_base
import clip_yawdd_emotion_train as yawdd_base


BASELINE_CHOICES = [
    "clip_zero_shot",
    "tip_adapter",
    "clip_linear_probe",
    "clip_adapter",
    "vanilla_clip_adapter",
    "coop",
    "maple",
    "resnet50_finetune",
    "mar_threshold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YawDD Table III baseline runner on the locked B-prime protocol")
    parser.add_argument("--baseline_mode", choices=BASELINE_CHOICES, required=True)
    parser.add_argument("--yawdd_root", default=yawdd_base.DEFAULT_YAWDD_ROOT)
    parser.add_argument("--label_mode", choices=["binary", "behavior_4", "multi4"], default="binary")
    parser.add_argument("--include_dash", action="store_true")
    parser.add_argument("--eval_mode", choices=["single", "fixed", "random"], default="single")
    parser.add_argument("--cv_mode", choices=["5fold", "split"], default="split")
    parser.add_argument("--split_mode", choices=["speaker_independent", "random_stratified"], default="speaker_independent")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--video_extensions", default=".avi,.mp4,.mov,.mkv")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="offline_only")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="The driver looks <LABEL>.")
    parser.add_argument("--prompt_set", default="yawdd_facial_cues")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_sampling_mode", choices=["uniform", "middle_late", "diff_guided"], default="diff_guided")
    parser.add_argument("--feature_layout", choices=["pooled", "sequence"], default="pooled")
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
    parser.add_argument("--pool_adapter_variant", choices=["legacy", "stronger"], default="legacy")
    parser.add_argument("--adapter_mode", choices=["full", "identity"], default="full")
    parser.add_argument("--temporal_head", choices=["none", "attention", "transformer"], default="none")
    parser.add_argument("--temporal_module", choices=["none", "cgp_fg", "taga", "mean_pool"], default="none")
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--temporal_num_layers", type=int, default=1)
    parser.add_argument("--temporal_pool_mode", choices=["cls", "mean", "hybrid"], default="mean")
    parser.add_argument("--no_frame_gate", action="store_true")
    parser.add_argument("--no_gem", action="store_true")
    parser.add_argument("--no_residual_blend", action="store_true")
    parser.add_argument("--gem_init_p", type=float, default=1.0)
    parser.add_argument("--use_class_weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--disable_class_weight", dest="use_class_weight", action="store_false")
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", dest="use_test_ensemble", action="store_true")
    parser.add_argument("--disable_test_ensemble", dest="use_test_ensemble", action="store_false")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_mode", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--feature_cache_dir", default=yawdd_base.DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--delete_shards_after_merge", action="store_true")
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--tip_alpha_grid", default="0.1,0.5,1,2,5,10,20")
    parser.add_argument("--tip_beta_grid", default="1,2,5,10,20,50")
    parser.add_argument("--coop_context_dim", type=int, default=0)
    parser.add_argument("--dlib_shape_predictor", default=None)
    parser.add_argument("--resnet50_weights", default=str(DEFAULT_TABLE3_PRETRAINED_DIR / "resnet50-11ad3fa6.pth"))
    parser.add_argument("--resnet_frame_cache_dir", default=None)
    parser.add_argument("--mar_shape_predictor", default=str(DEFAULT_TABLE3_PRETRAINED_DIR / "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument("--mar_score_cache", default=None)
    parser.add_argument("--mar_video_pool", choices=["max", "mean", "median"], default="max")
    parser.add_argument("--mar_detector_upsample", type=int, default=1)
    parser.add_argument("--mar_min_valid_frames", type=int, default=1)
    parser.set_defaults(
        use_class_weight=True,
        use_test_ensemble=False,
        pin_memory=True,
        use_amp=True,
    )
    args = parser.parse_args()
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"
    if args.training_seed is None:
        args.training_seed = args.seed

    # Fields consumed by yawdd_base.build_result_payload but intentionally
    # inactive in this backfill runner.
    for name, value in {
        "all_face_image": None,
        "external_face_root": None,
        "run_zero_shot_eval": False,
        "force_reextract": False,
        "multi4_oversample_labels": "",
        "multi4_oversample_min_count": 0,
        "multi4_oversample_mode": "noise",
        "multi4_oversample_noise_std": 0.0,
        "multi4_mixup_alpha": 0.4,
        "adapter_use_prompt_weight": "on",
        "adapter_use_class_temperature": "on",
        "adapter_use_class_bias": "on",
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
    if args.baseline_mode == "vanilla_clip_adapter":
        args.adapter_use_prompt_weight = "off"
        args.adapter_use_class_temperature = "off"
        args.adapter_use_class_bias = "off"
        args.resolved_use_prompt_weight = False
        args.resolved_use_class_temperature = False
        args.resolved_use_class_bias = False
    else:
        args.resolved_use_prompt_weight = True
        args.resolved_use_class_temperature = True
        args.resolved_use_class_bias = True
    return args


def log(message: str) -> None:
    yawdd_base.log(message)


def setup_logging(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    args_snapshot = output_dir / "args.json"
    args_snapshot.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    yawdd_base.init_log_file(args.log_file)
    atexit.register(yawdd_base.close_log_file)
    log(f"[ARGS] saved args snapshot to: {args_snapshot}")
    log(f"[LOG] writing log file to: {args.log_file}")


def parse_float_grid(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def prepare_protocol(args: argparse.Namespace):
    label_mode = args.label_mode if args.label_mode in {"behavior_4", "multi4"} else "binary"
    emotion_labels, _, _ = yawdd_base.resolve_label_space(label_mode)
    cremad_base.EMOTION_LABELS = list(emotion_labels)
    ravdess_base.EMOTION_LABELS = list(emotion_labels)
    prompt_groups = yawdd_base.build_class_prompts(label_mode, args.prompt_template, args.prompt_set)
    samples, dataset_diagnostics = yawdd_base.collect_yawdd_samples(
        yawdd_root=args.yawdd_root,
        label_mode=label_mode,
        include_dash=args.include_dash,
        max_sequences=args.max_sequences,
    )
    if len(samples) < 10:
        raise RuntimeError(f"Too few valid YawDD samples: {len(samples)}")
    if args.cv_mode == "5fold":
        splits, split_info = cremad_base.split_cremad_samples_5fold(samples, fold_idx=args.fold_idx, seed=args.seed)
        benchmark_mode = "5fold_subject_independent"
    elif args.split_mode == "random_stratified":
        splits, split_info = yawdd_base.split_yawdd_samples_random_stratified(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        benchmark_mode = "split_random_stratified"
    else:
        splits, split_info = cremad_base.split_cremad_samples_actor_ratio(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        benchmark_mode = "split_subject_independent"
    log(
        f"[DATA] samples={len(samples)} train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} split={benchmark_mode}"
    )
    return label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics


def load_clip_feature_bundle(args: argparse.Namespace, label_mode: str, emotion_labels: List[str], prompt_groups: List[List[str]], splits):
    processor, model = yawdd_base.load_clip_components(args.model_id, args.device, yawdd_base.resolve_effective_clip_mode(args, label_mode))
    split_samples_map = {
        split_name: cremad_base.build_split_samples_with_index(splits[split_name], split_name=split_name)
        for split_name in ["train", "val", "test"]
    }
    split_cache_plans = {
        split_name: cremad_base.build_split_cache_plan(
            args.feature_cache_dir,
            f"YawDD-{label_mode}",
            split_name,
            split_samples_map[split_name],
            args,
        )
        for split_name in ["train", "val", "test"]
    }
    resolved_paths = {}
    for split_name in ["train", "val", "test"]:
        resolved_paths[split_name] = str(
            cremad_base.ensure_training_split_cache(
                split_name=split_name,
                split_samples=split_samples_map[split_name],
                cache_plan=split_cache_plans[split_name],
                processor=processor,
                model=model,
                args=args,
            )
        )
    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    train_x, train_y, train_cache_samples, train_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["train"]["final_path"], label2idx
    )
    val_x, val_y, val_cache_samples, val_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["val"]["final_path"], label2idx
    )
    test_x, test_y, test_cache_samples, test_payload = cremad_base.load_features_and_labels_from_split_cache(
        split_cache_plans["test"]["final_path"], label2idx
    )
    text_features = yawdd_base.aide_base.extract_text_features(prompt_groups, processor, model, args.device)
    return {
        "processor": processor,
        "model": model,
        "label2idx": label2idx,
        "idx2label": {idx: label for idx, label in enumerate(emotion_labels)},
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "train_cache_samples": train_cache_samples,
        "val_cache_samples": val_cache_samples,
        "test_cache_samples": test_cache_samples,
        "train_payload": train_payload,
        "val_payload": val_payload,
        "test_payload": test_payload,
        "text_features": text_features,
        "resolved_paths": resolved_paths,
    }


def summarize_and_write(
    args: argparse.Namespace,
    label_mode: str,
    emotion_labels: List[str],
    prompt_groups: List[List[str]],
    splits,
    split_info,
    benchmark_mode: str,
    dataset_diagnostics: Dict[str, object],
    bundle,
    val_pred: List[str],
    test_pred: List[str],
    extra_config: Dict[str, object],
    checkpoint_payload: Optional[Dict[str, object]] = None,
):
    val_true = [emotion_labels[int(item.item())] for item in bundle["val_y"]]
    test_true = [emotion_labels[int(item.item())] for item in bundle["test_y"]]
    val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
    test_summary = yawdd_base.summarize_predictions_by_mode(test_true, test_pred, emotion_labels, label_mode)
    class_names = yawdd_base.resolve_class_names(label_mode, emotion_labels)
    class_distribution = yawdd_base.compute_split_class_distribution(
        emotion_labels,
        class_names,
        bundle["train_cache_samples"],
        bundle["val_cache_samples"],
        bundle["test_cache_samples"],
    )
    dataset_summary = yawdd_base.build_standard_yawdd_dataset_summary(
        args=args,
        dataset_diagnostics=dataset_diagnostics,
        split_info=split_info,
        train_samples=splits["train"],
        val_samples=splits["val"],
        test_samples=splits["test"],
        train_cache_payload=bundle["train_payload"],
        val_cache_payload=bundle["val_payload"],
        test_cache_payload=bundle["test_payload"],
        train_cache_samples=bundle["train_cache_samples"],
        val_cache_samples=bundle["val_cache_samples"],
        test_cache_samples=bundle["test_cache_samples"],
    )
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else cremad_base.default_checkpoint_path(Path(args.output))
    result = yawdd_base.build_result_payload(
        args=args,
        label_mode=label_mode,
        emotion_labels=emotion_labels,
        prompt_groups=prompt_groups,
        benchmark_mode=benchmark_mode,
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        val_summary=val_summary,
        test_summary=test_summary,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        resolved_feature_cache_paths=bundle["resolved_paths"],
        output_path_value=args.output,
        checkpoint_path=checkpoint_path,
        extra_config=extra_config,
    )
    yawdd_base.write_result_payload(result, args.output)
    if checkpoint_payload is None:
        checkpoint_payload = {"checkpoint_type": args.baseline_mode}
    checkpoint_payload.update({"config": result["config"], "output_path": args.output})
    yawdd_base.save_checkpoint_payload(checkpoint_path, checkpoint_payload)
    log(f"[DONE] saved YawDD baseline report to: {args.output}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def run_zero_shot(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    val_pred = ravdess_base.predict_zeroshot_from_features(
        bundle["val_x"], bundle["text_features"], bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size
    )
    test_pred = ravdess_base.predict_zeroshot_from_features(
        bundle["test_x"], bundle["text_features"], bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size
    )
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        val_pred,
        test_pred,
        {"baseline_variant": "pure_frozen_clip_zero_shot", "feature_source": "frozen_clip_cache"},
        {"checkpoint_type": "clip_zero_shot", "text_features": bundle["text_features"].cpu()},
    )


def run_linear_probe(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    import pickle
    from sklearn.linear_model import LogisticRegression

    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    class_weight = "balanced" if args.use_class_weight else None
    classifier = LogisticRegression(max_iter=2000, random_state=args.training_seed, class_weight=class_weight)
    classifier.fit(bundle["train_x"].cpu().numpy(), bundle["train_y"].cpu().numpy())
    val_pred = [bundle["idx2label"][int(index)] for index in classifier.predict(bundle["val_x"].cpu().numpy())]
    test_pred = [bundle["idx2label"][int(index)] for index in classifier.predict(bundle["test_x"].cpu().numpy())]
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        val_pred,
        test_pred,
        {"baseline_classifier": "logistic_regression", "feature_source": "frozen_clip_cache"},
        {"checkpoint_type": "clip_linear_probe", "classifier_bytes": pickle.dumps(classifier)},
    )


def tip_logits(image_x, train_x, train_y, text_features, alpha: float, beta: float, class_count: int):
    import torch
    import torch.nn.functional as F

    clip_logits = ravdess_base.zeroshot_logits(image_x, text_features)
    affinity = image_x @ train_x.t()
    cache_logits = torch.exp(-float(beta) * (1.0 - affinity)).matmul(F.one_hot(train_y, num_classes=class_count).float())
    return clip_logits + float(alpha) * cache_logits


def predict_tip(image_x, train_x, train_y, text_features, idx2label, batch_size, alpha, beta, class_count):
    preds = []
    device = text_features.device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    for start in range(0, image_x.shape[0], batch_size):
        batch_x = image_x[start:start + batch_size].to(device)
        logits = tip_logits(batch_x, train_x, train_y, text_features, alpha, beta, class_count)
        preds.extend([idx2label[int(index)] for index in logits.argmax(dim=-1).detach().cpu().tolist()])
    return preds


def run_tip_adapter(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    text_features = bundle["text_features"].to(args.device)
    train_x = bundle["train_x"].to(args.device)
    train_y = bundle["train_y"].to(args.device)
    val_true = [emotion_labels[int(item.item())] for item in bundle["val_y"]]
    best = None
    for alpha in parse_float_grid(args.tip_alpha_grid):
        for beta in parse_float_grid(args.tip_beta_grid):
            val_pred = predict_tip(
                bundle["val_x"],
                train_x,
                train_y,
                text_features,
                bundle["idx2label"],
                args.train_batch_size,
                alpha,
                beta,
                len(emotion_labels),
            )
            val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
            score = yawdd_base.select_metric_value(args.select_metric, val_summary, label_mode, val_true, val_pred, emotion_labels)
            if best is None or score > best["score"]:
                best = {"score": score, "alpha": alpha, "beta": beta, "val_pred": val_pred}
    test_pred = predict_tip(
        bundle["test_x"],
        train_x,
        train_y,
        text_features,
        bundle["idx2label"],
        args.train_batch_size,
        best["alpha"],
        best["beta"],
        len(emotion_labels),
    )
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        best["val_pred"],
        test_pred,
        {
            "baseline_variant": "tip_adapter_val_grid",
            "tip_alpha": best["alpha"],
            "tip_beta": best["beta"],
            "tip_val_score": round(float(best["score"]), 6),
            "feature_source": "frozen_clip_cache",
        },
        {"checkpoint_type": "tip_adapter", "train_features": bundle["train_x"].cpu(), "train_y": bundle["train_y"].cpu()},
    )


def run_clip_adapter(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    yawdd_base.configure_random_seeds(args.training_seed)
    adapter = yawdd_base.aide_base.ClipImageAdapter(
        dim=int(bundle["train_x"].shape[-1]),
        device=args.device,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        num_classes=len(emotion_labels),
        num_prompts=int(bundle["text_features"].shape[1]),
        use_prompt_weight=True,
        use_class_temperature=True,
        use_class_bias=True,
        adapter_mode="full",
    )
    adapter = cremad_base.train_strict_frozen_clip(
        train_x=bundle["train_x"],
        train_y=bundle["train_y"],
        val_x=bundle["val_x"],
        val_y=bundle["val_y"],
        text_features=bundle["text_features"],
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
        use_test_ensemble=False,
        ensemble_group_size=args.ensemble_group_size,
        use_amp=args.use_amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler_mode=args.lr_scheduler_mode,
        scheduler_min_lr=args.scheduler_min_lr,
    )
    val_pred = cremad_base.predict_emotion_from_features(bundle["val_x"], bundle["text_features"], adapter, bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size)
    test_pred = cremad_base.predict_emotion_from_features(bundle["test_x"], bundle["text_features"], adapter, bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size)
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        val_pred,
        test_pred,
        {"baseline_variant": "clip_adapter_visual_residual_with_prompt_calibration", "feature_source": "frozen_clip_cache"},
        {"checkpoint_type": "clip_adapter", "adapter_state_dict": adapter.state_dict(), "text_features": bundle["text_features"].cpu()},
    )


def run_vanilla_clip_adapter(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    yawdd_base.configure_random_seeds(args.training_seed)
    adapter = yawdd_base.aide_base.ClipImageAdapter(
        dim=int(bundle["train_x"].shape[-1]),
        device=args.device,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        num_classes=len(emotion_labels),
        num_prompts=int(bundle["text_features"].shape[1]),
        use_prompt_weight=False,
        use_class_temperature=False,
        use_class_bias=False,
        adapter_mode="full",
    )
    adapter = cremad_base.train_strict_frozen_clip(
        train_x=bundle["train_x"],
        train_y=bundle["train_y"],
        val_x=bundle["val_x"],
        val_y=bundle["val_y"],
        text_features=bundle["text_features"],
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
        use_test_ensemble=False,
        ensemble_group_size=args.ensemble_group_size,
        use_amp=args.use_amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler_mode=args.lr_scheduler_mode,
        scheduler_min_lr=args.scheduler_min_lr,
    )
    val_pred = cremad_base.predict_emotion_from_features(bundle["val_x"], bundle["text_features"], adapter, bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size)
    test_pred = cremad_base.predict_emotion_from_features(bundle["test_x"], bundle["text_features"], adapter, bundle["idx2label"], args.train_batch_size, False, args.ensemble_group_size)
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        val_pred,
        test_pred,
        {
            "baseline_variant": "vanilla_clip_adapter_residual_visual_only",
            "feature_source": "frozen_clip_cache",
            "use_prompt_weight": False,
            "use_class_temperature": False,
            "use_class_bias": False,
            "prompt_aggregation": "uniform_mean",
        },
        {"checkpoint_type": "vanilla_clip_adapter", "adapter_state_dict": adapter.state_dict(), "text_features": bundle["text_features"].cpu()},
    )


class FeaturePromptLearner:
    def __init__(self, text_features, device: str, visual_adapter: bool = False, hidden_dim: int = 512, dropout: float = 0.3):
        import torch
        import torch.nn as nn

        self.device = device
        self.text_features = nn.Parameter(text_features.clone().to(device))
        self.visual_adapter = visual_adapter
        dim = int(text_features.shape[-1])
        if visual_adapter:
            self.input_proj = nn.Linear(dim, hidden_dim).to(device)
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ).to(device)
            self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        else:
            self.input_proj = None
            self.net = None
            self.out_proj = None

    def parameters(self):
        params = [self.text_features]
        if self.visual_adapter:
            params += list(self.input_proj.parameters()) + list(self.net.parameters()) + list(self.out_proj.parameters())
        return params

    def state_dict(self):
        state = {"text_features": self.text_features.detach().cpu().clone(), "visual_adapter": self.visual_adapter}
        if self.visual_adapter:
            state.update({
                "input_proj": self.input_proj.state_dict(),
                "net": self.net.state_dict(),
                "out_proj": self.out_proj.state_dict(),
            })
        return state

    def train(self):
        for module in [self.input_proj, self.net, self.out_proj]:
            if module is not None:
                module.train()

    def eval(self):
        for module in [self.input_proj, self.net, self.out_proj]:
            if module is not None:
                module.eval()

    def logits(self, image_x):
        import torch
        import torch.nn.functional as F

        x = image_x.to(self.device)
        if self.visual_adapter:
            h = self.input_proj(x)
            h = h + self.net(h)
            x = self.out_proj(h)
        x = F.normalize(x, dim=-1)
        txt = F.normalize(self.text_features, dim=-1)
        return torch.einsum("bd,cpd->bcp", x, txt).mean(dim=-1)


def train_prompt_learner(args, bundle, emotion_labels, visual_adapter: bool):
    import torch
    import torch.nn as nn

    learner = FeaturePromptLearner(
        bundle["text_features"],
        device=args.device,
        visual_adapter=visual_adapter,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
    )
    class_weights = None
    if args.use_class_weight:
        counts = torch.bincount(bundle["train_y"], minlength=len(emotion_labels)).float()
        class_weights = (counts.sum() / counts.clamp(min=1.0))
        class_weights = (class_weights / class_weights.mean().clamp(min=1e-12)).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(learner.state_dict())
    best_metric = float("-inf")
    patience = 0
    yawdd_base.configure_random_seeds(args.training_seed)
    for epoch in range(args.epochs):
        learner.train()
        perm = torch.randperm(bundle["train_x"].shape[0])
        train_x = bundle["train_x"][perm]
        train_y = bundle["train_y"][perm]
        for start in range(0, train_x.shape[0], args.train_batch_size):
            bx = train_x[start:start + args.train_batch_size].to(args.device)
            by = train_y[start:start + args.train_batch_size].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(learner.logits(bx), by)
            loss.backward()
            if args.max_grad_norm > 0:
                import torch.nn.utils

                torch.nn.utils.clip_grad_norm_(learner.parameters(), args.max_grad_norm)
            optimizer.step()
        val_pred = predict_prompt_learner(learner, bundle["val_x"], bundle["idx2label"], args.train_batch_size)
        val_true = [emotion_labels[int(item.item())] for item in bundle["val_y"]]
        val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, "binary")
        metric = yawdd_base.select_metric_value(args.select_metric, val_summary, "binary", val_true, val_pred, emotion_labels)
        if metric > best_metric + args.early_stopping_min_delta:
            best_metric = metric
            best_state = copy.deepcopy(learner.state_dict())
            patience = 0
        else:
            patience += 1
        log(f"[TRAIN] prompt epoch {epoch + 1}/{args.epochs} val_metric={metric:.6f} best={best_metric:.6f}")
        if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
            break
    learner.text_features.data.copy_(best_state["text_features"].to(args.device))
    if visual_adapter:
        learner.input_proj.load_state_dict(best_state["input_proj"])
        learner.net.load_state_dict(best_state["net"])
        learner.out_proj.load_state_dict(best_state["out_proj"])
    learner.eval()
    return learner


def predict_prompt_learner(learner, image_x, idx2label, batch_size):
    preds = []
    for start in range(0, image_x.shape[0], batch_size):
        logits = learner.logits(image_x[start:start + batch_size])
        preds.extend([idx2label[int(index)] for index in logits.argmax(dim=-1).detach().cpu().tolist()])
    return preds


def run_prompt_baseline(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics, visual_adapter: bool):
    bundle = load_clip_feature_bundle(args, label_mode, emotion_labels, prompt_groups, splits)
    learner = train_prompt_learner(args, bundle, emotion_labels, visual_adapter=visual_adapter)
    val_pred = predict_prompt_learner(learner, bundle["val_x"], bundle["idx2label"], args.train_batch_size)
    test_pred = predict_prompt_learner(learner, bundle["test_x"], bundle["idx2label"], args.train_batch_size)
    variant = "feature_space_maple_text_visual_prompt" if visual_adapter else "feature_space_coop_text_prompt"
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        bundle,
        val_pred,
        test_pred,
        {"baseline_variant": variant, "feature_source": "frozen_clip_cache"},
        {"checkpoint_type": args.baseline_mode, "prompt_learner_state_dict": learner.state_dict()},
    )


def resnet_frame_cache_path(cache_dir: Optional[Path], sample: Dict, args, split_name: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    key_items = [
        split_name,
        str(sample.get("sequence_id", "")),
        str(sample.get("video_path", "")),
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
        "resnet50_imagenet_v1_transforms",
    ]
    digest = hashlib.sha1("||".join(key_items).encode("utf-8")).hexdigest()
    return cache_dir / split_name / f"{digest}.pt"


def build_resnet_frame_tensor(sample: Dict, args, transform):
    import torch

    images, _, _ = ravdess_base.read_sampled_media(
        sample,
        args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
    )
    frames = [transform(image.convert("RGB")) for image in images]
    if not frames:
        raise RuntimeError(f"No frames available for sample: {sample.get('sequence_id')}")
    if len(frames) < args.num_frames:
        frames.extend([frames[-1].clone() for _ in range(args.num_frames - len(frames))])
    return torch.stack(frames[: args.num_frames], dim=0)


def ensure_resnet_frame_cache(split_name: str, samples: List[Dict], args, transform, cache_dir: Optional[Path]) -> Dict[str, object]:
    if cache_dir is None:
        return {"enabled": False}
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / split_name).mkdir(parents=True, exist_ok=True)
    created = 0
    reused = 0
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        path = resnet_frame_cache_path(cache_dir, sample, args, split_name)
        if path.exists():
            reused += 1
        else:
            tensor = build_resnet_frame_tensor(sample, args, transform)
            tmp_path = path.with_suffix(".tmp")
            import torch

            torch.save(tensor.half(), tmp_path)
            tmp_path.replace(path)
            created += 1
        if index % 25 == 0 or index == total:
            log(f"[RESNET50][CACHE] {split_name}: {index}/{total} videos cached (created={created}, reused={reused})")
    return {
        "enabled": True,
        "cache_dir": str(cache_dir),
        "split": split_name,
        "created": created,
        "reused": reused,
        "total": total,
    }


class ResnetVideoDataset:
    def __init__(
        self,
        samples: List[Dict],
        label2idx: Dict[str, int],
        num_frames: int,
        frame_sampling_mode: str,
        transform,
        args=None,
        split_name: str = "",
        cache_dir: Optional[Path] = None,
    ):
        self.samples = list(samples)
        self.label2idx = dict(label2idx)
        self.num_frames = int(num_frames)
        self.frame_sampling_mode = frame_sampling_mode
        self.transform = transform
        self.args = args
        self.split_name = split_name
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        cache_path = resnet_frame_cache_path(self.cache_dir, sample, self.args, self.split_name) if self.args is not None else None
        if cache_path is not None and cache_path.exists():
            frames = torch.load(cache_path, map_location="cpu").float()
        else:
            frames = build_resnet_frame_tensor(sample, self.args or self, self.transform)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(frames.half(), cache_path)
        return frames, self.label2idx[sample["label"]], sample["sequence_id"]


def eval_resnet(model, loader, idx2label, device):
    import torch

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for frames, targets, _ in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            bsz, frame_count = frames.shape[:2]
            logits = model(frames.view(bsz * frame_count, *frames.shape[2:]))
            logits = logits.view(bsz, frame_count, -1).mean(dim=1)
            y_true.extend([idx2label[int(item)] for item in targets.detach().cpu().tolist()])
            y_pred.extend([idx2label[int(item)] for item in logits.argmax(dim=-1).detach().cpu().tolist()])
    return y_true, y_pred


def run_resnet50(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet50_Weights, resnet50

    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    idx2label = {idx: label for idx, label in enumerate(emotion_labels)}
    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    cache_dir = Path(args.resnet_frame_cache_dir) if args.resnet_frame_cache_dir else Path(args.output).parent.parent / "resnet_frame_cache"
    cache_info = {
        "train": ensure_resnet_frame_cache("train", splits["train"], args, transform, cache_dir),
        "val": ensure_resnet_frame_cache("val", splits["val"], args, transform, cache_dir),
        "test": ensure_resnet_frame_cache("test", splits["test"], args, transform, cache_dir),
    }
    train_ds = ResnetVideoDataset(splits["train"], label2idx, args.num_frames, args.frame_sampling_mode, transform, args=args, split_name="train", cache_dir=cache_dir)
    val_ds = ResnetVideoDataset(splits["val"], label2idx, args.num_frames, args.frame_sampling_mode, transform, args=args, split_name="val", cache_dir=cache_dir)
    test_ds = ResnetVideoDataset(splits["test"], label2idx, args.num_frames, args.frame_sampling_mode, transform, args=args, split_name="test", cache_dir=cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    yawdd_base.configure_random_seeds(args.training_seed)
    model = resnet50(weights=None)
    if args.resnet50_weights:
        weights_path = Path(args.resnet50_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"ResNet50 weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        log(f"[RESNET50] loaded ImageNet weights from: {weights_path}")
    else:
        raise RuntimeError("ResNet-50 fine-tuned baseline requires explicit ImageNet pretrained weights.")
    model.fc = nn.Linear(model.fc.in_features, len(emotion_labels))
    model = model.to(args.device)
    class_weights = None
    if args.use_class_weight:
        counts = Counter(sample["label"] for sample in splits["train"])
        weights_list = [len(splits["train"]) / max(1, len(emotion_labels) * counts[label]) for label in emotion_labels]
        class_weights = torch.tensor(weights_list, dtype=torch.float32, device=args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and str(args.device).startswith("cuda")))
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        for frames, targets, _ in train_loader:
            frames = frames.to(args.device)
            targets = targets.to(args.device)
            bsz, frame_count = frames.shape[:2]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp and str(args.device).startswith("cuda"))):
                logits = model(frames.view(bsz * frame_count, *frames.shape[2:]))
                logits = logits.view(bsz, frame_count, -1).mean(dim=1)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        val_true, val_pred = eval_resnet(model, val_loader, idx2label, args.device)
        val_summary = yawdd_base.summarize_predictions_by_mode(val_true, val_pred, emotion_labels, label_mode)
        metric = yawdd_base.select_metric_value(args.select_metric, val_summary, label_mode, val_true, val_pred, emotion_labels)
        if metric > best_metric + args.early_stopping_min_delta:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        log(f"[TRAIN] resnet50 epoch {epoch + 1}/{args.epochs} val_metric={metric:.6f} best={best_metric:.6f} epoch_sec={time.time() - epoch_start:.1f}")
        if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    val_true, val_pred = eval_resnet(model, val_loader, idx2label, args.device)
    test_true, test_pred = eval_resnet(model, test_loader, idx2label, args.device)

    # Minimal bundle compatible with summarize_and_write.
    import torch

    fake_bundle = {
        "val_y": torch.tensor([label2idx[x] for x in val_true]),
        "test_y": torch.tensor([label2idx[x] for x in test_true]),
        "train_cache_samples": list(splits["train"]),
        "val_cache_samples": list(splits["val"]),
        "test_cache_samples": list(splits["test"]),
        "train_payload": {"failed_count": 0},
        "val_payload": {"failed_count": 0},
        "test_payload": {"failed_count": 0},
        "resolved_paths": {},
    }
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        fake_bundle,
        val_pred,
        test_pred,
        {
            "baseline_backbone": "resnet50",
            "feature_source": "raw_video_diff_guided_frames",
            "pretrained_weights": str(args.resnet50_weights),
            "frame_cache": cache_info,
        },
        {"checkpoint_type": "resnet50_finetune", "model_state_dict": model.state_dict()},
    )


def dlib_shape_to_np(shape) -> np.ndarray:
    return np.asarray([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    # dlib 68-point mouth landmarks use zero-based indices 48..67.
    left = points[48]
    right = points[54]
    vertical = (
        np.linalg.norm(points[51] - points[59])
        + np.linalg.norm(points[52] - points[58])
        + np.linalg.norm(points[53] - points[57])
    )
    horizontal = 2.0 * np.linalg.norm(left - right)
    if horizontal <= 1e-6:
        return float("nan")
    return float(vertical / horizontal)


def compute_sample_mar_score(sample: Dict, args, detector, predictor) -> Tuple[float, Dict[str, object]]:
    import cv2

    images, _, _ = ravdess_base.read_sampled_media(
        sample,
        args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
    )
    scores = []
    failed_frames = 0
    for image in images:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, int(args.mar_detector_upsample))
        if not faces:
            failed_frames += 1
            continue
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = predictor(gray, face)
        mar = mouth_aspect_ratio(dlib_shape_to_np(shape))
        if np.isfinite(mar):
            scores.append(mar)
        else:
            failed_frames += 1
    if len(scores) < int(args.mar_min_valid_frames):
        return 0.0, {"valid_frames": len(scores), "failed_frames": failed_frames, "pooled": args.mar_video_pool}
    arr = np.asarray(scores, dtype=np.float32)
    if args.mar_video_pool == "mean":
        score = float(arr.mean())
    elif args.mar_video_pool == "median":
        score = float(np.median(arr))
    else:
        score = float(arr.max())
    return score, {"valid_frames": len(scores), "failed_frames": failed_frames, "pooled": args.mar_video_pool}


def score_split_mar(split_samples: List[Dict], args, detector, predictor) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    scores = []
    diagnostics = []
    total = len(split_samples)
    for index, sample in enumerate(split_samples, start=1):
        score, diag = compute_sample_mar_score(sample, args, detector, predictor)
        scores.append(score)
        diagnostics.append({"sequence_id": sample.get("sequence_id"), "score": score, **diag})
        if index % 25 == 0 or index == total:
            log(f"[MAR] scored {index}/{total} videos")
    return np.asarray(scores, dtype=np.float32), diagnostics


def predict_from_mar_scores(scores: np.ndarray, threshold: float, emotion_labels: List[str]) -> List[str]:
    not_label, drowsy_label = emotion_labels[0], emotion_labels[1]
    return [drowsy_label if float(score) >= float(threshold) else not_label for score in scores]


def choose_mar_threshold(args, val_scores: np.ndarray, val_true: List[str], emotion_labels: List[str], label_mode: str):
    candidates = set(float(x) for x in val_scores.tolist())
    if len(candidates) == 0:
        candidates.add(0.0)
    lo = min(candidates)
    hi = max(candidates)
    candidates.update(np.linspace(max(0.0, lo - 0.05), hi + 0.05, num=101).tolist())
    candidates.add(float("-inf"))
    candidates.add(float("inf"))
    best = None
    for threshold in sorted(candidates):
        pred = predict_from_mar_scores(val_scores, threshold, emotion_labels)
        summary = yawdd_base.summarize_predictions_by_mode(val_true, pred, emotion_labels, label_mode)
        score = yawdd_base.select_metric_value(args.select_metric, summary, label_mode, val_true, pred, emotion_labels)
        if best is None or score > best["score"]:
            best = {"threshold": float(threshold), "score": float(score), "val_pred": pred}
    return best


def run_mar_threshold(args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics):
    import dlib
    import torch

    label2idx = {label: idx for idx, label in enumerate(emotion_labels)}
    val_true = [sample["label"] for sample in splits["val"]]
    test_true = [sample["label"] for sample in splits["test"]]
    cache_path = Path(args.mar_score_cache) if args.mar_score_cache else Path(args.output).parent.parent / "mar_score_cache.pt"
    predictor_path = Path(args.mar_shape_predictor or args.dlib_shape_predictor or "")
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        train_scores = np.asarray(cached["train_scores"], dtype=np.float32)
        val_scores = np.asarray(cached["val_scores"], dtype=np.float32)
        test_scores = np.asarray(cached["test_scores"], dtype=np.float32)
        train_diag = list(cached.get("train_diag", []))
        val_diag = list(cached.get("val_diag", []))
        test_diag = list(cached.get("test_diag", []))
        log(f"[MAR] loaded score cache: {cache_path}")
    else:
        if not predictor_path.exists():
            raise FileNotFoundError(f"MAR shape predictor not found: {predictor_path}")
        log(f"[MAR] loading dlib detector and shape predictor: {predictor_path}")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))
        train_scores, train_diag = score_split_mar(splits["train"], args, detector, predictor)
        val_scores, val_diag = score_split_mar(splits["val"], args, detector, predictor)
        test_scores, test_diag = score_split_mar(splits["test"], args, detector, predictor)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_scores": train_scores,
                "val_scores": val_scores,
                "test_scores": test_scores,
                "train_diag": train_diag,
                "val_diag": val_diag,
                "test_diag": test_diag,
                "shape_predictor": str(predictor_path),
                "num_frames": args.num_frames,
                "frame_sampling_mode": args.frame_sampling_mode,
                "mar_video_pool": args.mar_video_pool,
            },
            cache_path,
        )
        log(f"[MAR] saved score cache: {cache_path}")
    best = choose_mar_threshold(args, val_scores, val_true, emotion_labels, label_mode)
    test_pred = predict_from_mar_scores(test_scores, best["threshold"], emotion_labels)

    fake_bundle = {
        "val_y": torch.tensor([label2idx[x] for x in val_true]),
        "test_y": torch.tensor([label2idx[x] for x in test_true]),
        "train_cache_samples": list(splits["train"]),
        "val_cache_samples": list(splits["val"]),
        "test_cache_samples": list(splits["test"]),
        "train_payload": {"failed_count": 0},
        "val_payload": {"failed_count": 0},
        "test_payload": {"failed_count": 0},
        "resolved_paths": {},
    }
    return summarize_and_write(
        args,
        label_mode,
        emotion_labels,
        prompt_groups,
        splits,
        split_info,
        benchmark_mode,
        dataset_diagnostics,
        fake_bundle,
        best["val_pred"],
        test_pred,
        {
            "baseline_variant": "dlib_68pt_mar_threshold",
            "feature_source": "raw_video_diff_guided_frames",
            "shape_predictor": str(predictor_path),
            "mar_video_pool": args.mar_video_pool,
            "mar_threshold": round(float(best["threshold"]), 6),
            "mar_val_score": round(float(best["score"]), 6),
            "train_valid_frame_mean": round(float(np.mean([x["valid_frames"] for x in train_diag])), 4),
            "val_valid_frame_mean": round(float(np.mean([x["valid_frames"] for x in val_diag])), 4),
            "test_valid_frame_mean": round(float(np.mean([x["valid_frames"] for x in test_diag])), 4),
        },
        {
            "checkpoint_type": "mar_threshold",
            "threshold": float(best["threshold"]),
            "train_scores": train_scores,
            "val_scores": val_scores,
            "test_scores": test_scores,
            "train_diag": train_diag,
            "val_diag": val_diag,
            "test_diag": test_diag,
        },
    )


def main() -> int:
    args = parse_args()
    setup_logging(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics = prepare_protocol(args)
    dispatch = {
        "clip_zero_shot": run_zero_shot,
        "tip_adapter": run_tip_adapter,
        "clip_linear_probe": run_linear_probe,
        "clip_adapter": run_clip_adapter,
        "vanilla_clip_adapter": run_vanilla_clip_adapter,
        "coop": lambda *items: run_prompt_baseline(*items, visual_adapter=False),
        "maple": lambda *items: run_prompt_baseline(*items, visual_adapter=True),
        "resnet50_finetune": run_resnet50,
        "mar_threshold": run_mar_threshold,
    }
    dispatch[args.baseline_mode](args, label_mode, emotion_labels, prompt_groups, splits, split_info, benchmark_mode, dataset_diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
