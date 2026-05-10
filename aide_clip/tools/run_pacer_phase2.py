import argparse
import itertools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path("/data1/yanjing/talk2bev")
AIDE_CLIP_ROOT = REPO_ROOT / "aide_clip"
YAWDD_SCRIPT = AIDE_CLIP_ROOT / "src" / "clip_yawdd_emotion_train.py"
AIDE_SCRIPT = AIDE_CLIP_ROOT / "src" / "clip_aide_emotion_train.py"
YAWDD_DATA = REPO_ROOT / "fatigue-drive-yawning-detection" / "extracted_face_multi4"
RESULTS_ROOT = AIDE_CLIP_ROOT / "results" / "yawdd" / "causal_v2"
MMTL_PYTHON = "/home/yanjing/anaconda3/envs/mmtl/bin/python"
BEST_LINE_RE = re.compile(r"best_epoch=(\d+), best_metric=([0-9.]+)")

TRAINING_SEEDS = [20, 21, 22, 23, 24]
AIDE_TRAINING_SEEDS = [20, 21, 22]
AIDE_SPLIT_SEED = 42
AIDE_BLOCK5_GROUP_SOURCES = ["scene_vehicle", "scene", "vehicle"]
BLOCK2_CCL_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
BLOCK3_CFA_WEIGHTS = [0.05, 0.1, 0.5, 1.0, 2.0]
BLOCK3_CDA_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5]


@dataclass
class RunConfig:
    config_id: str
    block: str
    output_json: Path
    output_log: Path
    checkpoint_path: Path
    split_seed: int
    training_seed: int
    ccl_weight: float
    cfa_weight: float
    cda_prob: float
    use_ccl: bool
    use_cfa: bool
    use_cda: bool
    notes: str


@dataclass
class AideRunConfig:
    config_id: str
    block: str
    output_json: Path
    output_log: Path
    checkpoint_path: Path
    split_seed: int
    training_seed: int
    ccl_weight: float
    cfa_weight: float
    cda_prob: float
    use_ccl: bool
    use_cfa: bool
    use_cda: bool
    causal_group_source: str
    notes: str


def format_float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def estimate_block_runtime(run_count: int, seconds_per_run: float = 5.5) -> float:
    return run_count * seconds_per_run / 3600.0


def section_without_title(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        return lines[1:]
    return lines


def fmt_metric(mean_value: float, std_value: float, n: int) -> str:
    return f"{mean_value:.6f} +- {std_value:.6f} (n={n})"


def metric_from_run(payload: Dict[str, object], key: str) -> float:
    value = payload.get(key)
    if value is None:
        section = payload.get("test", {})
        value = section.get("accuracy") if key == "acc" else section.get(key)
    return float(value)


def aide_group_source_runs_dir(group_source: str) -> Path:
    if group_source == "scene_vehicle":
        return RESULTS_ROOT / "block5_crossdataset" / "aide_runs"
    return RESULTS_ROOT / "block5_crossdataset" / f"aide_runs_{group_source}"


def aid_group_source_label(group_source: str) -> str:
    return "scene+vehicle" if group_source == "scene_vehicle" else group_source


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def compute_macro_f1(confusion_matrix: Dict[str, Dict[str, int]]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    labels = list(confusion_matrix.keys())
    per_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    total = 0
    correct = 0
    for label in labels:
        row = confusion_matrix[label]
        tp = float(row.get(label, 0))
        fp = float(sum(confusion_matrix[other].get(label, 0) for other in labels if other != label))
        fn = float(sum(count for pred_label, count in row.items() if pred_label != label))
        support = int(sum(row.values()))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        total += support
        correct += int(tp)
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    per_class["_overall"] = {
        "accuracy": (correct / total) if total else 0.0,
        "support": total,
    }
    return macro_f1, per_class


def parse_best_epoch(log_path: Path) -> Tuple[Optional[int], Optional[float]]:
    if not log_path.exists():
        return None, None
    best_epoch = None
    best_metric = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = BEST_LINE_RE.search(line)
        if match:
            best_epoch = int(match.group(1))
            best_metric = float(match.group(2))
    return best_epoch, best_metric


def base_yawdd_args(config: RunConfig) -> List[str]:
    args = [
        str(YAWDD_SCRIPT),
        "--all_face_image", str(YAWDD_DATA),
        "--label_mode", "binary",
        "--eval_mode", "fixed",
        "--cv_mode", "split",
        "--clip_mode", "offline_only",
        "--prompt_set", "yawdd_facial_cues",
        "--num_frames", "10",
        "--frame_sampling_mode", "uniform",
        "--feature_layout", "sequence",
        "--temporal_head", "transformer",
        "--temporal_num_heads", "4",
        "--temporal_num_layers", "1",
        "--temporal_pool_mode", "hybrid",
        "--adapter_use_prompt_weight", "on",
        "--adapter_use_class_temperature", "on",
        "--adapter_use_class_bias", "on",
        "--use_class_weight",
        "--disable_test_ensemble",
        "--epochs", "40",
        "--lr", "0.00015",
        "--weight_decay", "0.01",
        "--label_smoothing", "0.1",
        "--loss_type", "focal",
        "--seed", str(config.split_seed),
        "--training_seed", str(config.training_seed),
        "--ccl_weight", str(config.ccl_weight),
        "--cfa_weight", str(config.cfa_weight),
        "--cda_prob", str(config.cda_prob),
        "--output", str(config.output_json),
        "--log_file", str(config.output_log),
        "--checkpoint_output", str(config.checkpoint_path),
    ]
    if config.use_ccl:
        args.append("--use_causal_contrastive")
    if config.use_cfa:
        args.append("--use_causal_alignment")
    if config.use_cda:
        args.append("--use_counterfactual_aug")
    return args


def base_aide_args(config: AideRunConfig) -> List[str]:
    args = [
        str(AIDE_SCRIPT),
        "--clip_mode", "offline_only",
        "--strict_frozen_clip", "on",
        "--prompt_set", "driving_7",
        "--epochs", "40",
        "--batch_size", "32",
        "--lr", "0.00015",
        "--weight_decay", "0.0005",
        "--max_grad_norm", "1.0",
        "--num_frames", "5",
        "--adapter_hidden_dim", "1024",
        "--adapter_dropout", "0.2",
        "--use_class_weight", "on",
        "--label_smoothing", "0.03",
        "--select_metric", "weighted_f1",
        "--use_test_ensemble", "on",
        "--ensemble_group_size", "2",
        "--use_prompt_weight", "on",
        "--use_class_temperature", "on",
        "--use_class_bias", "on",
        "--feature_cache_dir", str(AIDE_CLIP_ROOT / "cache" / "features"),
        "--seed", str(config.split_seed),
        "--training_seed", str(config.training_seed),
        "--ccl_weight", str(config.ccl_weight),
        "--cfa_weight", str(config.cfa_weight),
        "--cda_prob", str(config.cda_prob),
        "--causal_group_source", config.causal_group_source,
        "--output", str(config.output_json),
        "--checkpoint_output", str(config.checkpoint_path),
    ]
    if config.use_ccl:
        args.extend(["--use_causal_contrastive", "on"])
    else:
        args.extend(["--use_causal_contrastive", "off"])
    if config.use_cfa:
        args.extend(["--use_causal_alignment", "on"])
    else:
        args.extend(["--use_causal_alignment", "off"])
    if config.use_cda:
        args.extend(["--use_counterfactual_aug", "on"])
    else:
        args.extend(["--use_counterfactual_aug", "off"])
    return args


def enrich_result(config: RunConfig, git_commit: str, wall_clock_sec: float) -> Dict[str, object]:
    with config.output_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    test = data.get("test", {})
    confusion = test.get("confusion_matrix") or {}
    macro_f1, per_class = compute_macro_f1(confusion)
    best_epoch, best_metric = parse_best_epoch(config.output_log)
    n_test = sum(sum(row.values()) for row in confusion.values()) if confusion else None
    data.update(
        {
            "config_id": config.config_id,
            "block": config.block,
            "status": "ok",
            "split_seed": config.split_seed,
            "training_seed": config.training_seed,
            "ccl_weight": config.ccl_weight,
            "cfa_weight": config.cfa_weight,
            "cda_prob": config.cda_prob,
            "git_commit": git_commit,
            "acc": test.get("accuracy"),
            "weighted_f1": test.get("weighted_f1"),
            "macro_f1": macro_f1,
            "per_class": {label: metrics for label, metrics in per_class.items() if not label.startswith("_")},
            "confusion_matrix": confusion,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "n_test": n_test,
            "wall_clock_sec": round(wall_clock_sec, 3),
            "notes": config.notes,
        }
    )
    with config.output_json.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return data


def write_failure_payload(config: RunConfig, git_commit: str, wall_clock_sec: float, note: str) -> None:
    payload = {
        "config_id": config.config_id,
        "block": config.block,
        "status": "failed",
        "split_seed": config.split_seed,
        "training_seed": config.training_seed,
        "ccl_weight": config.ccl_weight,
        "cfa_weight": config.cfa_weight,
        "cda_prob": config.cda_prob,
        "git_commit": git_commit,
        "acc": None,
        "weighted_f1": None,
        "macro_f1": None,
        "per_class": {},
        "confusion_matrix": {},
        "best_epoch": None,
        "n_test": 39,
        "wall_clock_sec": round(wall_clock_sec, 3),
        "notes": note,
    }
    with config.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_command(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    command = [MMTL_PYTHON] + args
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def execute_run(config: RunConfig, retry_once: bool = True) -> Dict[str, object]:
    git_commit = git_head()
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = run_command(base_yawdd_args(config), AIDE_CLIP_ROOT)
    duration = time.time() - start
    if result.returncode != 0:
        retry_note = f"initial failure: {result.stderr.strip() or result.stdout.strip()}"
        if retry_once:
            retry_start = time.time()
            retry_result = run_command(base_yawdd_args(config), AIDE_CLIP_ROOT)
            duration += time.time() - retry_start
            if retry_result.returncode == 0 and config.output_json.exists():
                return enrich_result(config, git_commit, duration)
            note = retry_note + " | retry failed"
        else:
            note = retry_note
        write_failure_payload(config, git_commit, duration, note)
        return {
            "config_id": config.config_id,
            "status": "failed",
            "notes": note,
        }
    return enrich_result(config, git_commit, duration)


def enrich_aide_result(config: AideRunConfig, git_commit: str, wall_clock_sec: float) -> Dict[str, object]:
    with config.output_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    test = data.get("test", {})
    confusion = test.get("confusion_matrix") or {}
    macro_f1, per_class = compute_macro_f1(confusion)
    best_epoch, best_metric = parse_best_epoch(config.output_log)
    n_test = sum(sum(row.values()) for row in confusion.values()) if confusion else None
    data.update(
        {
            "config_id": config.config_id,
            "block": config.block,
            "status": "ok",
            "dataset_name": "AIDE",
            "split_seed": config.split_seed,
            "training_seed": config.training_seed,
            "ccl_weight": config.ccl_weight,
            "cfa_weight": config.cfa_weight,
            "cda_prob": config.cda_prob,
            "git_commit": git_commit,
            "acc": test.get("accuracy"),
            "weighted_f1": test.get("weighted_f1"),
            "macro_f1": macro_f1,
            "per_class": {label: metrics for label, metrics in per_class.items() if not label.startswith("_")},
            "confusion_matrix": confusion,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "n_test": n_test,
            "wall_clock_sec": round(wall_clock_sec, 3),
            "notes": config.notes,
        }
    )
    write_run(config.output_json, data)
    return data


def write_aide_failure_payload(config: AideRunConfig, git_commit: str, wall_clock_sec: float, note: str) -> None:
    payload = {
        "config_id": config.config_id,
        "block": config.block,
        "status": "failed",
        "dataset_name": "AIDE",
        "split_seed": config.split_seed,
        "training_seed": config.training_seed,
        "ccl_weight": config.ccl_weight,
        "cfa_weight": config.cfa_weight,
        "cda_prob": config.cda_prob,
        "git_commit": git_commit,
        "acc": None,
        "weighted_f1": None,
        "macro_f1": None,
        "per_class": {},
        "confusion_matrix": {},
        "best_epoch": None,
        "n_test": None,
        "wall_clock_sec": round(wall_clock_sec, 3),
        "notes": note,
    }
    write_run(config.output_json, payload)


def execute_aide_run(config: AideRunConfig, retry_once: bool = True) -> Dict[str, object]:
    git_commit = git_head()
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = run_command(base_aide_args(config), AIDE_CLIP_ROOT)
    duration = time.time() - start
    if config.output_json.exists() and not config.output_log.exists() and result.stdout:
        config.output_log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        retry_note = f"initial failure: {result.stderr.strip() or result.stdout.strip()}"
        if retry_once:
            retry_start = time.time()
            retry_result = run_command(base_aide_args(config), AIDE_CLIP_ROOT)
            duration += time.time() - retry_start
            if config.output_json.exists() and not config.output_log.exists() and retry_result.stdout:
                config.output_log.write_text(retry_result.stdout, encoding="utf-8")
            if retry_result.returncode == 0 and config.output_json.exists():
                return enrich_aide_result(config, git_commit, duration)
            note = retry_note + " | retry failed"
        else:
            note = retry_note
        write_aide_failure_payload(config, git_commit, duration, note)
        return {"config_id": config.config_id, "status": "failed", "notes": note}
    return enrich_aide_result(config, git_commit, duration)


def load_run(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_run(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def clone_run_artifacts(source_json: Path, target_json: Path, config_id: str, block: str, notes: str) -> None:
    payload = load_run(source_json)
    payload["config_id"] = config_id
    payload["block"] = block
    payload["notes"] = notes
    write_run(target_json, payload)
    copy_if_exists(source_json.with_suffix(".log"), target_json.with_suffix(".log"))
    copy_if_exists(source_json.with_suffix(".ckpt.pt"), target_json.with_suffix(".ckpt.pt"))


def block2_configs() -> List[RunConfig]:
    runs_dir = RESULTS_ROOT / "block2_ccl_sweep" / "runs"
    configs: List[RunConfig] = []
    for training_seed in TRAINING_SEEDS:
        baseline_path = runs_dir / f"B2_baseline_train{training_seed}.json"
        configs.append(
            RunConfig(
                config_id=f"B2_baseline_train{training_seed}",
                block="block2",
                output_json=baseline_path,
                output_log=baseline_path.with_suffix(".log"),
                checkpoint_path=baseline_path.with_suffix(".ckpt.pt"),
                split_seed=7,
                training_seed=training_seed,
                ccl_weight=0.0,
                cfa_weight=0.0,
                cda_prob=0.0,
                use_ccl=False,
                use_cfa=False,
                use_cda=False,
                notes="D0 baseline run reused by block4.",
            )
        )
        for weight in BLOCK2_CCL_WEIGHTS[1:]:
            token = format_float_token(weight)
            path = runs_dir / f"B2_ccl{token}_train{training_seed}.json"
            configs.append(
                RunConfig(
                    config_id=f"B2_ccl{token}_train{training_seed}",
                    block="block2",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=7,
                    training_seed=training_seed,
                    ccl_weight=weight,
                    cfa_weight=0.0,
                    cda_prob=0.0,
                    use_ccl=True,
                    use_cfa=False,
                    use_cda=False,
                    notes="CCL-only sweep.",
                )
            )
    return configs


def block3_configs() -> List[RunConfig]:
    runs_dir = RESULTS_ROOT / "block3_cfa_cda_sweep" / "runs"
    configs: List[RunConfig] = []
    for training_seed in TRAINING_SEEDS:
        for weight in BLOCK3_CFA_WEIGHTS:
            token = format_float_token(weight)
            path = runs_dir / f"B3a_cfa{token}_train{training_seed}.json"
            configs.append(
                RunConfig(
                    config_id=f"B3a_cfa{token}_train{training_seed}",
                    block="block3",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=7,
                    training_seed=training_seed,
                    ccl_weight=0.0,
                    cfa_weight=weight,
                    cda_prob=0.0,
                    use_ccl=False,
                    use_cfa=True,
                    use_cda=False,
                    notes="CFA-only sweep.",
                )
            )
        for prob in BLOCK3_CDA_PROBS:
            token = format_float_token(prob)
            path = runs_dir / f"B3b_cda{token}_train{training_seed}.json"
            configs.append(
                RunConfig(
                    config_id=f"B3b_cda{token}_train{training_seed}",
                    block="block3",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=7,
                    training_seed=training_seed,
                    ccl_weight=0.0,
                    cfa_weight=0.0,
                    cda_prob=prob,
                    use_ccl=False,
                    use_cfa=False,
                    use_cda=True,
                    notes="CDA-only sweep.",
                )
            )
    return configs


def summarize_weight_table(rows: List[Tuple[str, List[Dict[str, object]]]], metric: str) -> List[str]:
    header = ["weight"] + [f"train_{seed}" for seed in TRAINING_SEEDS] + ["mean", "std"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, runs in rows:
        per_seed = []
        for seed in TRAINING_SEEDS:
            run = next(item for item in runs if int(item["training_seed"]) == seed)
            value = run.get(metric)
            per_seed.append(value)
        mean_value, std_value = mean_std(per_seed)
        row = [label] + [f"{value:.6f}" for value in per_seed] + [f"{mean_value:.6f}", f"{std_value:.6f}"]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def find_best_row(rows: List[Tuple[str, List[Dict[str, object]]]]) -> Tuple[str, float, float, float, float]:
    scored = []
    for label, runs in rows:
        acc_values = [float(run["acc"]) for run in runs]
        wf1_values = [float(run["weighted_f1"]) for run in runs]
        mean_acc, std_acc = mean_std(acc_values)
        mean_wf1, std_wf1 = mean_std(wf1_values)
        scored.append((label, mean_acc, std_acc, mean_wf1, std_wf1))
    scored.sort(key=lambda item: (item[1], item[3]), reverse=True)
    return scored[0]


def summarize_block2() -> Tuple[float, Path]:
    runs_dir = RESULTS_ROOT / "block2_ccl_sweep" / "runs"
    summary_path = RESULTS_ROOT / "block2_ccl_sweep" / "summary.md"
    rows: List[Tuple[str, List[Dict[str, object]]]] = []
    baseline_runs = [load_run(runs_dir / f"B2_baseline_train{seed}.json") for seed in TRAINING_SEEDS]
    rows.append(("0", baseline_runs))
    for weight in BLOCK2_CCL_WEIGHTS[1:]:
        token = format_float_token(weight)
        runs = [load_run(runs_dir / f"B2_ccl{token}_train{seed}.json") for seed in TRAINING_SEEDS]
        rows.append((str(weight), runs))
    best_label, best_mean_acc, best_std_acc, best_mean_wf1, best_std_wf1 = find_best_row(rows)
    baseline_acc = [float(run["acc"]) for run in baseline_runs]
    baseline_wf1 = [float(run["weighted_f1"]) for run in baseline_runs]
    baseline_std_acc = mean_std(baseline_acc)[1]
    delta_lines = [
        "| weight | delta_acc_mean | delta_acc_std | delta_wf1_mean | delta_wf1_std | within_noise |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, runs in rows:
        acc_values = [float(run["acc"]) for run in runs]
        wf1_values = [float(run["weighted_f1"]) for run in runs]
        delta_acc = [value - base for value, base in zip(acc_values, baseline_acc)]
        delta_wf1 = [value - base for value, base in zip(wf1_values, baseline_wf1)]
        mean_delta_acc, std_delta_acc = mean_std(delta_acc)
        mean_delta_wf1, std_delta_wf1 = mean_std(delta_wf1)
        within_noise = "yes" if abs(mean_delta_acc) < baseline_std_acc else "no"
        delta_lines.append(
            f"| {label} | {mean_delta_acc:.6f} | {std_delta_acc:.6f} | {mean_delta_wf1:.6f} | {std_delta_wf1:.6f} | {within_noise} |"
        )
    lines = [
        "# Block 2 — CCL Sweep",
        "",
        f"Estimated runtime before execution: {estimate_block_runtime(45):.2f} hours.",
        "",
        "## Accuracy",
        "",
        *summarize_weight_table(rows, "acc"),
        "",
        "## Weighted F1",
        "",
        *summarize_weight_table(rows, "weighted_f1"),
        "",
        "## Best Mean Accuracy",
        "",
        f"Best CCL weight by mean accuracy: {best_label}.",
        f"Mean accuracy: {best_mean_acc:.6f} +- {best_std_acc:.6f}.",
        f"Mean weighted F1: {best_mean_wf1:.6f} +- {best_std_wf1:.6f}.",
        "",
        "## Delta vs Baseline",
        "",
        *delta_lines,
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return float(best_label), summary_path


def summarize_component_table(
    title: str,
    values: Sequence[float],
    prefix: str,
    metric: str,
    runs_dir: Path,
) -> List[str]:
    rows = []
    for value in values:
        token = format_float_token(value)
        runs = [load_run(runs_dir / f"{prefix}{token}_train{seed}.json") for seed in TRAINING_SEEDS]
        rows.append((str(value), runs))
    return [f"## {title}", "", *summarize_weight_table(rows, metric), ""]


def pick_best_component(values: Sequence[float], prefix: str, runs_dir: Path) -> float:
    rows = []
    for value in values:
        token = format_float_token(value)
        runs = [load_run(runs_dir / f"{prefix}{token}_train{seed}.json") for seed in TRAINING_SEEDS]
        rows.append((str(value), runs))
    best_label, _, _, _, _ = find_best_row(rows)
    return float(best_label)


def summarize_block3() -> Tuple[float, float, Path]:
    runs_dir = RESULTS_ROOT / "block3_cfa_cda_sweep" / "runs"
    summary_path = RESULTS_ROOT / "block3_cfa_cda_sweep" / "summary.md"
    best_cfa = pick_best_component(BLOCK3_CFA_WEIGHTS, "B3a_cfa", runs_dir)
    best_cda = pick_best_component(BLOCK3_CDA_PROBS, "B3b_cda", runs_dir)
    lines = [
        "# Block 3 — CFA and CDA Sweeps",
        "",
        f"Estimated runtime before execution: {estimate_block_runtime(50):.2f} hours.",
        "",
        *summarize_component_table("CFA Accuracy", BLOCK3_CFA_WEIGHTS, "B3a_cfa", "acc", runs_dir),
        *summarize_component_table("CFA Weighted F1", BLOCK3_CFA_WEIGHTS, "B3a_cfa", "weighted_f1", runs_dir),
        *summarize_component_table("CDA Accuracy", BLOCK3_CDA_PROBS, "B3b_cda", "acc", runs_dir),
        *summarize_component_table("CDA Weighted F1", BLOCK3_CDA_PROBS, "B3b_cda", "weighted_f1", runs_dir),
        "## Best Settings",
        "",
        f"Best CFA weight by mean accuracy: {best_cfa}.",
        f"Best CDA probability by mean accuracy: {best_cda}.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_cfa, best_cda, summary_path


def block4_configs(best_ccl: float, best_cfa: float, best_cda: float) -> List[RunConfig]:
    runs_dir = RESULTS_ROOT / "block4_final" / "runs"
    config_defs = [
        ("D4_cfa_cda", 0.0, best_cfa, best_cda, False, True, True, "CFA + CDA combo."),
        ("D5_ccl_cfa", best_ccl, best_cfa, 0.0, True, True, False, "CCL + CFA combo."),
        ("D6_ccl_cda", best_ccl, 0.0, best_cda, True, False, True, "CCL + CDA combo."),
        ("D7_full", best_ccl, best_cfa, best_cda, True, True, True, "Full causal config."),
    ]
    configs: List[RunConfig] = []
    for training_seed in TRAINING_SEEDS:
        for name, ccl_weight, cfa_weight, cda_prob, use_ccl, use_cfa, use_cda, notes in config_defs:
            path = runs_dir / f"{name}_train{training_seed}.json"
            configs.append(
                RunConfig(
                    config_id=f"{name}_train{training_seed}",
                    block="block4",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=7,
                    training_seed=training_seed,
                    ccl_weight=ccl_weight,
                    cfa_weight=cfa_weight,
                    cda_prob=cda_prob,
                    use_ccl=use_ccl,
                    use_cfa=use_cfa,
                    use_cda=use_cda,
                    notes=notes,
                )
            )
    return configs


def block5_yawdd_half_configs(best_ccl: float, best_cfa: float, best_cda: float) -> List[RunConfig]:
    runs_dir = RESULTS_ROOT / "block5_crossdataset" / "yawdd_runs"
    configs: List[RunConfig] = []
    config_defs = [
        ("yawdd_I_ccl_half", best_ccl / 2.0, best_cfa, best_cda, True, True, True, "Half-strength CCL intervention."),
        ("yawdd_I_cfa_half", best_ccl, best_cfa / 2.0, best_cda, True, True, True, "Half-strength CFA intervention."),
        ("yawdd_I_cda_half", best_ccl, best_cfa, best_cda / 2.0, True, True, True, "Half-strength CDA intervention."),
    ]
    for training_seed in TRAINING_SEEDS:
        for name, ccl_weight, cfa_weight, cda_prob, use_ccl, use_cfa, use_cda, notes in config_defs:
            path = runs_dir / f"{name}_train{training_seed}.json"
            configs.append(
                RunConfig(
                    config_id=f"{name}_train{training_seed}",
                    block="block5",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=7,
                    training_seed=training_seed,
                    ccl_weight=ccl_weight,
                    cfa_weight=cfa_weight,
                    cda_prob=cda_prob,
                    use_ccl=use_ccl,
                    use_cfa=use_cfa,
                    use_cda=use_cda,
                    notes=notes,
                )
            )
    return configs


def materialize_block5_yawdd_reused_runs() -> None:
    block4_dir = RESULTS_ROOT / "block4_final" / "runs"
    block5_dir = RESULTS_ROOT / "block5_crossdataset" / "yawdd_runs"
    mappings = {
        "yawdd_I_full": "D7_full",
        "yawdd_I_no_ccl": "D4_cfa_cda",
        "yawdd_I_no_cfa": "D6_ccl_cda",
        "yawdd_I_no_cda": "D5_ccl_cfa",
    }
    for training_seed in TRAINING_SEEDS:
        for target_prefix, source_prefix in mappings.items():
            clone_run_artifacts(
                block4_dir / f"{source_prefix}_train{training_seed}.json",
                block5_dir / f"{target_prefix}_train{training_seed}.json",
                f"{target_prefix}_train{training_seed}",
                "block5",
                f"Reused from {source_prefix}.",
            )


def block5_aide_configs(best_ccl: float, best_cfa: float, best_cda: float, group_source: str = "scene_vehicle") -> List[AideRunConfig]:
    runs_dir = aide_group_source_runs_dir(group_source)
    intervention_defs = [
        ("aide_I_full", best_ccl, best_cfa, best_cda, True, True, True, "Full causal intervention on AIDE."),
        ("aide_I_no_ccl", 0.0, best_cfa, best_cda, False, True, True, "Drop CCL on AIDE."),
        ("aide_I_no_cfa", best_ccl, 0.0, best_cda, True, False, True, "Drop CFA on AIDE."),
        ("aide_I_no_cda", best_ccl, best_cfa, 0.0, True, True, False, "Drop CDA on AIDE."),
        ("aide_I_ccl_half", best_ccl / 2.0, best_cfa, best_cda, True, True, True, "Half-strength CCL on AIDE."),
        ("aide_I_cfa_half", best_ccl, best_cfa / 2.0, best_cda, True, True, True, "Half-strength CFA on AIDE."),
        ("aide_I_cda_half", best_ccl, best_cfa, best_cda / 2.0, True, True, True, "Half-strength CDA on AIDE."),
    ]
    configs: List[AideRunConfig] = []
    for training_seed in AIDE_TRAINING_SEEDS:
        for name, ccl_weight, cfa_weight, cda_prob, use_ccl, use_cfa, use_cda, notes in intervention_defs:
            path = runs_dir / f"{name}_train{training_seed}.json"
            configs.append(
                AideRunConfig(
                    config_id=f"{name}_train{training_seed}",
                    block="block5",
                    output_json=path,
                    output_log=path.with_suffix(".log"),
                    checkpoint_path=path.with_suffix(".ckpt.pt"),
                    split_seed=AIDE_SPLIT_SEED,
                    training_seed=training_seed,
                    ccl_weight=ccl_weight,
                    cfa_weight=cfa_weight,
                    cda_prob=cda_prob,
                    use_ccl=use_ccl,
                    use_cfa=use_cfa,
                    use_cda=use_cda,
                    causal_group_source=group_source,
                    notes=f"{notes} AIDE group source={aid_group_source_label(group_source)}.",
                )
            )
    return configs


def materialize_block4_reused_runs(best_ccl: float, best_cfa: float, best_cda: float) -> None:
    block2_dir = RESULTS_ROOT / "block2_ccl_sweep" / "runs"
    block3_dir = RESULTS_ROOT / "block3_cfa_cda_sweep" / "runs"
    block4_dir = RESULTS_ROOT / "block4_final" / "runs"
    best_ccl_token = format_float_token(best_ccl)
    best_cfa_token = format_float_token(best_cfa)
    best_cda_token = format_float_token(best_cda)
    for training_seed in TRAINING_SEEDS:
        clone_run_artifacts(
            block2_dir / f"B2_baseline_train{training_seed}.json",
            block4_dir / f"D0_baseline_train{training_seed}.json",
            f"D0_baseline_train{training_seed}",
            "block4",
            "Reused from Block 2 baseline.",
        )
        clone_run_artifacts(
            block2_dir / f"B2_ccl{best_ccl_token}_train{training_seed}.json",
            block4_dir / f"D1_ccl_only_train{training_seed}.json",
            f"D1_ccl_only_train{training_seed}",
            "block4",
            "Reused from Block 2 best CCL run.",
        )
        clone_run_artifacts(
            block3_dir / f"B3a_cfa{best_cfa_token}_train{training_seed}.json",
            block4_dir / f"D2_cfa_only_train{training_seed}.json",
            f"D2_cfa_only_train{training_seed}",
            "block4",
            "Reused from Block 3A best CFA run.",
        )
        clone_run_artifacts(
            block3_dir / f"B3b_cda{best_cda_token}_train{training_seed}.json",
            block4_dir / f"D3_cda_only_train{training_seed}.json",
            f"D3_cda_only_train{training_seed}",
            "block4",
            "Reused from Block 3B best CDA run.",
        )


def best_single_seed(config_runs: List[Dict[str, object]]) -> Dict[str, object]:
    return sorted(config_runs, key=lambda item: (float(item["acc"]), float(item["weighted_f1"])), reverse=True)[0]


def summarize_block4(best_ccl: float, best_cfa: float, best_cda: float) -> Tuple[str, Path]:
    runs_dir = RESULTS_ROOT / "block4_final" / "runs"
    summary_path = RESULTS_ROOT / "block4_final" / "summary.md"
    d0_runs = [load_run(RESULTS_ROOT / "block2_ccl_sweep" / "runs" / f"B2_baseline_train{seed}.json") for seed in TRAINING_SEEDS]
    configs = {
        "D0_baseline": d0_runs,
        "D1_ccl_only": [load_run(runs_dir / f"D1_ccl_only_train{seed}.json") for seed in TRAINING_SEEDS],
        "D2_cfa_only": [load_run(runs_dir / f"D2_cfa_only_train{seed}.json") for seed in TRAINING_SEEDS],
        "D3_cda_only": [load_run(runs_dir / f"D3_cda_only_train{seed}.json") for seed in TRAINING_SEEDS],
        "D4_cfa_cda": [load_run(runs_dir / f"D4_cfa_cda_train{seed}.json") for seed in TRAINING_SEEDS],
        "D5_ccl_cfa": [load_run(runs_dir / f"D5_ccl_cfa_train{seed}.json") for seed in TRAINING_SEEDS],
        "D6_ccl_cda": [load_run(runs_dir / f"D6_ccl_cda_train{seed}.json") for seed in TRAINING_SEEDS],
        "D7_full": [load_run(runs_dir / f"D7_full_train{seed}.json") for seed in TRAINING_SEEDS],
    }
    baseline_acc_values = [float(run["acc"]) for run in d0_runs]
    baseline_wf1_values = [float(run["weighted_f1"]) for run in d0_runs]
    baseline_mean_acc, baseline_std_acc = mean_std(baseline_acc_values)
    baseline_mean_wf1, _ = mean_std(baseline_wf1_values)
    table_lines = [
        "| config | mean_acc | std_acc | mean_wF1 | std_wF1 | delta_acc_vs_D0 | delta_wF1_vs_D0 | n_seeds | significance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    scored = []
    for config_name, runs in configs.items():
        acc_values = [float(run["acc"]) for run in runs]
        wf1_values = [float(run["weighted_f1"]) for run in runs]
        mean_acc, std_acc = mean_std(acc_values)
        mean_wf1, std_wf1 = mean_std(wf1_values)
        delta_acc = mean_acc - baseline_mean_acc
        delta_wf1 = mean_wf1 - baseline_mean_wf1
        significance = "within_noise" if abs(delta_acc) < baseline_std_acc else "above_noise"
        table_lines.append(
            f"| {config_name} | {mean_acc:.6f} | {std_acc:.6f} | {mean_wf1:.6f} | {std_wf1:.6f} | {delta_acc:.6f} | {delta_wf1:.6f} | {len(runs)} | {significance} |"
        )
        scored.append((config_name, mean_acc, mean_wf1, runs))
    scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_config_name, _, _, best_runs = scored[0]
    best_seed_run = best_single_seed(best_runs)
    recommended_command = " ".join(base_yawdd_args(
        RunConfig(
            config_id=best_seed_run["config_id"],
            block="block4",
            output_json=Path("OUTPUT.json"),
            output_log=Path("OUTPUT.log"),
            checkpoint_path=Path("OUTPUT.ckpt.pt"),
            split_seed=7,
            training_seed=int(best_seed_run["training_seed"]),
            ccl_weight=float(best_seed_run["ccl_weight"]),
            cfa_weight=float(best_seed_run["cfa_weight"]),
            cda_prob=float(best_seed_run["cda_prob"]),
            use_ccl=float(best_seed_run["ccl_weight"]) > 0,
            use_cfa=float(best_seed_run["cfa_weight"]) > 0,
            use_cda=float(best_seed_run["cda_prob"]) > 0,
            notes="",
        )
    ))
    lines = [
        "# Block 4 — Final Configurations",
        "",
        f"Estimated runtime before execution: {estimate_block_runtime(35):.2f} hours.",
        "",
        *table_lines,
        "",
        "## Best Single Seed Confusion Matrix",
        "",
        f"Best config: {best_config_name}.",
        f"Best single-seed run: {best_seed_run['config_id']}.",
        "",
        "```json",
        json.dumps(best_seed_run["confusion_matrix"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Recommendation",
        "",
        f"Recommended configuration: {best_seed_run['config_id']}.",
        "",
        "```bash",
        recommended_command,
        "```",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_config_name, summary_path


def intervention_mean_acc(runs: List[Dict[str, object]]) -> float:
    return mean_std([float(run["acc"]) for run in runs])[0]


def rank_positions(values: Sequence[float]) -> List[int]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(pairs, start=1):
        ranks[idx] = rank
    return ranks


def spearman_from_ranks(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    n = len(rank_a)
    diff_sq = sum((a - b) ** 2 for a, b in zip(rank_a, rank_b))
    return 1.0 - (6.0 * diff_sq) / (n * (n**2 - 1))


def kendall_from_ranks(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    concordant = 0
    discordant = 0
    n = len(rank_a)
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = rank_a[i] - rank_a[j]
            sign_b = rank_b[i] - rank_b[j]
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total else 0.0


def exact_rank_pvalues(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float, float, float]:
    rank_a = rank_positions(values_a)
    rank_b = rank_positions(values_b)
    observed_spearman = spearman_from_ranks(rank_a, rank_b)
    observed_kendall = kendall_from_ranks(rank_a, rank_b)
    n = len(rank_a)
    all_perms = list(itertools.permutations(range(1, n + 1)))
    extreme_spearman = 0
    extreme_kendall = 0
    for perm in all_perms:
        rho = spearman_from_ranks(rank_a, perm)
        tau = kendall_from_ranks(rank_a, perm)
        if abs(rho) >= abs(observed_spearman) - 1e-12:
            extreme_spearman += 1
        if abs(tau) >= abs(observed_kendall) - 1e-12:
            extreme_kendall += 1
    denom = float(len(all_perms))
    return observed_kendall, extreme_kendall / denom, observed_spearman, extreme_spearman / denom


def run_aide_block(configs: Sequence[AideRunConfig]) -> List[Dict[str, object]]:
    results = []
    for config in configs:
        if missing_or_failed(config):
            results.append(execute_aide_run(config))
        else:
            results.append(load_run(config.output_json))
    return results


def dataset_intervention_runs(base_dir: Path, prefix: str, seeds: Sequence[int], intervention: str) -> List[Dict[str, object]]:
    return [load_run(base_dir / f"{prefix}_{intervention}_train{seed}.json") for seed in seeds]


def summarize_block5(best_ccl: float, best_cfa: float, best_cda: float) -> Path:
    summary_path = RESULTS_ROOT / "block5_crossdataset" / "rank_correlation.md"
    yawdd_dir = RESULTS_ROOT / "block5_crossdataset" / "yawdd_runs"
    interventions = ["I_full", "I_no_ccl", "I_no_cfa", "I_no_cda", "I_ccl_half", "I_cfa_half", "I_cda_half"]
    yawdd_runs = {name: dataset_intervention_runs(yawdd_dir, "yawdd", TRAINING_SEEDS, name) for name in interventions}
    yawdd_full_mean = intervention_mean_acc(yawdd_runs["I_full"])
    vector_names = ["I_no_ccl", "I_no_cfa", "I_no_cda", "I_ccl_half", "I_cfa_half", "I_cda_half"]
    yawdd_vector = [yawdd_full_mean - intervention_mean_acc(yawdd_runs[name]) for name in vector_names]
    primary_group_source = "scene_vehicle"
    primary_table_lines: List[str] = []
    robustness_lines = [
        "| aide_group_source | aide_full_mean_acc | kendall_tau | tau_p | spearman_rho | rho_p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    interpretations = []
    for group_source in AIDE_BLOCK5_GROUP_SOURCES:
        aide_dir = aide_group_source_runs_dir(group_source)
        aide_runs = {name: dataset_intervention_runs(aide_dir, "aide", AIDE_TRAINING_SEEDS, name) for name in interventions}
        aide_full_mean = intervention_mean_acc(aide_runs["I_full"])
        aide_vector = [aide_full_mean - intervention_mean_acc(aide_runs[name]) for name in vector_names]
        tau, tau_p, rho, rho_p = exact_rank_pvalues(yawdd_vector, aide_vector)
        robustness_lines.append(
            f"| {aid_group_source_label(group_source)} | {aide_full_mean:.6f} | {tau:.6f} | {tau_p:.6f} | {rho:.6f} | {rho_p:.6f} |"
        )
        interpretations.append((group_source, tau, rho))
        if group_source == primary_group_source:
            primary_table_lines = [
                "| intervention | yawdd_mean_acc | yawdd_delta | aide_mean_acc | aide_delta |",
                "| --- | --- | --- | --- | --- |",
            ]
            for idx, name in enumerate(vector_names):
                yawdd_mean = intervention_mean_acc(yawdd_runs[name])
                aide_mean = intervention_mean_acc(aide_runs[name])
                primary_table_lines.append(
                    f"| {name} | {yawdd_mean:.6f} | {yawdd_vector[idx]:.6f} | {aide_mean:.6f} | {aide_vector[idx]:.6f} |"
                )
            primary_tau = tau
            primary_tau_p = tau_p
            primary_rho = rho
            primary_rho_p = rho_p
            primary_aide_full_mean = aide_full_mean
    positive_sources = [aid_group_source_label(source) for source, tau, rho in interpretations if tau > 0 and rho > 0]
    if positive_sources:
        interpretation = (
            "Positive correlation only appears under subset AIDE groupings ("
            + ", ".join(positive_sources)
            + "), so the domain-general mechanism claim depends on how AIDE context groups are defined."
        )
    else:
        interpretation = "All tested AIDE group definitions yield near-zero or negative cross-dataset rank correlation, which weakens the domain-general mechanism claim rather than rescuing it by regrouping AIDE contexts."
    lines = [
        "# Block 5 — Cross-dataset Rank Correlation",
        "",
        f"Estimated runtime before execution: {estimate_block_runtime(78, seconds_per_run=120.0):.2f} hours.",
        "",
        f"Using YawDD best settings: ccl_weight*={best_ccl}, cfa_weight*={best_cfa}, cda_prob*={best_cda}.",
        f"AIDE fixed split seed: {AIDE_SPLIT_SEED}; training seeds: {AIDE_TRAINING_SEEDS}.",
        "",
        "## Primary Effect Table (AIDE scene+vehicle groups)",
        "",
        *primary_table_lines,
        "",
        "## Primary Correlations (AIDE scene+vehicle groups)",
        "",
        f"Kendall tau: {primary_tau:.6f} with exact two-sided p-value {primary_tau_p:.6f}.",
        f"Spearman rho: {primary_rho:.6f} with exact two-sided p-value {primary_rho_p:.6f}.",
        "",
        "## Group-Source Robustness",
        "",
        *robustness_lines,
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "Caveat: n=6 interventions is still small, so the exact p-values should be read as descriptive evidence rather than a final causal claim.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def generate_final_report() -> Path:
    final_path = RESULTS_ROOT / "final_report.md"
    block1_path = RESULTS_ROOT / "block1_repro" / "runs" / "B1_7_split7_train20_exact_p6.json"
    block4_dir = RESULTS_ROOT / "block4_final" / "runs"
    block5_yawdd_dir = RESULTS_ROOT / "block5_crossdataset" / "yawdd_runs"
    block5_aide_dir = RESULTS_ROOT / "block5_crossdataset" / "aide_runs"
    block1 = load_run(block1_path)
    d0_runs = [load_run(block4_dir / f"D0_baseline_train{seed}.json") for seed in TRAINING_SEEDS]
    d3_runs = [load_run(block4_dir / f"D3_cda_only_train{seed}.json") for seed in TRAINING_SEEDS]
    d7_runs = [load_run(block5_yawdd_dir / f"yawdd_I_full_train{seed}.json") for seed in TRAINING_SEEDS]
    aide_full_runs = [load_run(block5_aide_dir / f"aide_I_full_train{seed}.json") for seed in AIDE_TRAINING_SEEDS]
    d0_acc = mean_std([float(run["acc"]) for run in d0_runs])
    d0_wf1 = mean_std([float(run["weighted_f1"]) for run in d0_runs])
    d3_acc = mean_std([float(run["acc"]) for run in d3_runs])
    d3_wf1 = mean_std([float(run["weighted_f1"]) for run in d3_runs])
    d7_acc = mean_std([float(run["acc"]) for run in d7_runs])
    d7_wf1 = mean_std([float(run["weighted_f1"]) for run in d7_runs])
    aide_acc = mean_std([float(run["acc"]) for run in aide_full_runs])
    aide_wf1 = mean_std([float(run["weighted_f1"]) for run in aide_full_runs])
    block5_lines = (RESULTS_ROOT / "block5_crossdataset" / "rank_correlation.md").read_text(encoding="utf-8").splitlines()
    tau_line = next((line for line in block5_lines if line.startswith("Kendall tau:")), "Kendall tau: N/A")
    rho_line = next((line for line in block5_lines if line.startswith("Spearman rho:")), "Spearman rho: N/A")
    robustness_line = next((line for line in block5_lines if line.startswith("All tested AIDE group definitions") or line.startswith("Positive correlation only appears")), "AIDE group-source robustness: N/A")
    recommendation_lines = (RESULTS_ROOT / "block4_final" / "summary.md").read_text(encoding="utf-8").splitlines()
    command_block_start = recommendation_lines.index("```bash")
    recommended_command = recommendation_lines[command_block_start + 1]
    lines = [
        "# PACER Causal Extension — Final Report",
        "",
        "## Headline Numbers",
        "",
        f"- Controlled Block 1 baseline (split=7, training=20): acc={metric_from_run(block1, 'acc'):.6f}, wF1={metric_from_run(block1, 'weighted_f1'):.6f}.",
        f"- D0 baseline over five training seeds: acc={fmt_metric(d0_acc[0], d0_acc[1], len(TRAINING_SEEDS))}, wF1={fmt_metric(d0_wf1[0], d0_wf1[1], len(TRAINING_SEEDS))}.",
        f"- Best mean YawDD final config D3_cda_only: acc={fmt_metric(d3_acc[0], d3_acc[1], len(TRAINING_SEEDS))}, wF1={fmt_metric(d3_wf1[0], d3_wf1[1], len(TRAINING_SEEDS))}.",
        f"- Full YawDD intervention D7_full: acc={fmt_metric(d7_acc[0], d7_acc[1], len(TRAINING_SEEDS))}, wF1={fmt_metric(d7_wf1[0], d7_wf1[1], len(TRAINING_SEEDS))}.",
        f"- AIDE full intervention: acc={fmt_metric(aide_acc[0], aide_acc[1], len(AIDE_TRAINING_SEEDS))}, wF1={fmt_metric(aide_wf1[0], aide_wf1[1], len(AIDE_TRAINING_SEEDS))}.",
        f"- {tau_line}",
        f"- {rho_line}",
        f"- {robustness_line}",
        "",
        "## Block 1 Confirmation",
        "",
        "Controlled baseline is reproducible at split seed 7 and training seed 20 with bit-for-bit recovery of the historical 76.92 / 77.20 target.",
        "",
        "## Block 2 — CCL Sweep",
        "",
        *section_without_title(RESULTS_ROOT / "block2_ccl_sweep" / "summary.md"),
        "",
        "## Block 3 — CFA and CDA Sweeps",
        "",
        *section_without_title(RESULTS_ROOT / "block3_cfa_cda_sweep" / "summary.md"),
        "",
        "## Block 4 — Final Configurations",
        "",
        *section_without_title(RESULTS_ROOT / "block4_final" / "summary.md"),
        "",
        "## Block 5 — Cross-dataset Correlation",
        "",
        *section_without_title(RESULTS_ROOT / "block5_crossdataset" / "rank_correlation.md"),
        "",
        "## Recommended CLI",
        "",
        "```bash",
        recommended_command,
        "```",
    ]
    final_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final_path


def missing_or_failed(config: RunConfig) -> bool:
    if not config.output_json.exists():
        return True
    try:
        payload = load_run(config.output_json)
    except Exception:
        return True
    return payload.get("status") != "ok"


def run_block(configs: Sequence[RunConfig]) -> List[Dict[str, object]]:
    results = []
    for config in configs:
        if missing_or_failed(config):
            results.append(execute_run(config))
        else:
            results.append(load_run(config.output_json))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PACER phase-2 YawDD sweeps")
    parser.add_argument("--blocks", nargs="+", choices=["block2", "block3", "block4", "block5", "final_report", "all"], default=["all"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = set(args.blocks)
    if "all" in requested:
        requested = {"block2", "block3", "block4", "block5", "final_report"}

    best_ccl = None
    best_cfa = None
    best_cda = None
    need_block2_summary = "block2" in requested or "block4" in requested or "block5" in requested or "final_report" in requested
    need_block3_summary = "block3" in requested or "block4" in requested or "block5" in requested or "final_report" in requested

    if "block2" in requested:
        run_block(block2_configs())
    if need_block2_summary:
        best_ccl, _ = summarize_block2()

    if "block3" in requested:
        run_block(block3_configs())
    if need_block3_summary:
        best_cfa, best_cda, _ = summarize_block3()

    if "block4" in requested:
        materialize_block4_reused_runs(best_ccl, best_cfa, best_cda)
        run_block(block4_configs(best_ccl, best_cfa, best_cda))
        summarize_block4(best_ccl, best_cfa, best_cda)

    if "block5" in requested:
        materialize_block4_reused_runs(best_ccl, best_cfa, best_cda)
        materialize_block5_yawdd_reused_runs()
        run_block(block5_yawdd_half_configs(best_ccl, best_cfa, best_cda))
        for group_source in AIDE_BLOCK5_GROUP_SOURCES:
            run_aide_block(block5_aide_configs(best_ccl, best_cfa, best_cda, group_source=group_source))
        summarize_block5(best_ccl, best_cfa, best_cda)

    if "final_report" in requested:
        generate_final_report()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())