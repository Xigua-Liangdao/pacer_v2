import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
AIDE_CLIP_ROOT = REPO_ROOT / "aide_clip"
PHASE1_ROOT = AIDE_CLIP_ROOT / "results" / "cca_v2" / "phase1"
YAWDD_RESULTS_ROOT = PHASE1_ROOT / "yawdd"
AIDE_RESULTS_ROOT = PHASE1_ROOT / "aide"
YAWDD_SCRIPT = AIDE_CLIP_ROOT / "src" / "clip_yawdd_emotion_train.py"
AIDE_SCRIPT = AIDE_CLIP_ROOT / "src" / "clip_aide_emotion_train.py"
YAWDD_DATA = REPO_ROOT / "fatigue-drive-yawning-detection" / "extracted_face_multi4"
DEFAULT_PYTHON = "/home/yanjing/anaconda3/envs/mmtl/bin/python"
YAWDD_SPLIT_SEED = 7
AIDE_SPLIT_SEED = 42
DEFAULT_TRAINING_SEEDS = [20, 21, 22]


@dataclass
class RunSpec:
    dataset: str
    config_id: str
    split_seed: int
    training_seed: int
    notes: str
    command: List[str]
    output_json: Path
    output_log: Path
    checkpoint_path: Path


def parse_csv_ints(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def mean_std(values: Sequence[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.6f} +- 0.000000 (n=1)"
    return f"{statistics.mean(values):.6f} +- {statistics.stdev(values):.6f} (n={len(values)})"


def phase1_configs() -> List[Dict[str, object]]:
    return [
        {
            "config_id": "baseline",
            "notes": "Reference run without any v2 causal component enabled.",
            "yawdd_args": [],
            "aide_args": [],
        },
        {
            "config_id": "cda_v2_only",
            "notes": "Counterfactual data augmentation v2 only.",
            "yawdd_args": ["--use_cda_v2_mixstyle", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5"],
            "aide_args": ["--use_cda_v2_mixstyle", "on", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5"],
        },
        {
            "config_id": "ccl_v2_only",
            "notes": "Counterfactual-anchored contrastive loss only.",
            "yawdd_args": ["--use_ccl_v2_counterfactual", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.07"],
            "aide_args": ["--use_ccl_v2_counterfactual", "on", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.1"],
        },
        {
            "config_id": "cfa_v2_only",
            "notes": "Text-anchor EMA feature alignment only.",
            "yawdd_args": ["--use_cfa_v2_textanchor", "--cfa_v2_weight", "0.1", "--cfa_v2_anchor_weight", "0.5", "--cfa_v2_ema_momentum", "0.9"],
            "aide_args": ["--use_cfa_v2_textanchor", "on", "--cfa_v2_weight", "0.05", "--cfa_v2_anchor_weight", "1.0", "--cfa_v2_ema_momentum", "0.99"],
        },
        {
            "config_id": "cda_v2_ccl_v2",
            "notes": "Counterfactual augmentation plus counterfactual-anchored contrastive loss.",
            "yawdd_args": [
                "--use_cda_v2_mixstyle", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_ccl_v2_counterfactual", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.07",
            ],
            "aide_args": [
                "--use_cda_v2_mixstyle", "on", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_ccl_v2_counterfactual", "on", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.1",
            ],
        },
        {
            "config_id": "cda_v2_cfa_v2",
            "notes": "Counterfactual augmentation plus text-anchor EMA alignment.",
            "yawdd_args": [
                "--use_cda_v2_mixstyle", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_cfa_v2_textanchor", "--cfa_v2_weight", "0.1", "--cfa_v2_anchor_weight", "0.5", "--cfa_v2_ema_momentum", "0.9",
            ],
            "aide_args": [
                "--use_cda_v2_mixstyle", "on", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_cfa_v2_textanchor", "on", "--cfa_v2_weight", "0.05", "--cfa_v2_anchor_weight", "1.0", "--cfa_v2_ema_momentum", "0.99",
            ],
        },
        {
            "config_id": "all_three_v2",
            "notes": "Full CCA-v2 stack: CDA-v2 + CCL-v2 + CFA-v2.",
            "yawdd_args": [
                "--use_cda_v2_mixstyle", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_ccl_v2_counterfactual", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.07",
                "--use_cfa_v2_textanchor", "--cfa_v2_weight", "0.1", "--cfa_v2_anchor_weight", "0.5", "--cfa_v2_ema_momentum", "0.9",
            ],
            "aide_args": [
                "--use_cda_v2_mixstyle", "on", "--cda_v2_prob", "0.5", "--cda_v2_kl_weight", "0.5",
                "--use_ccl_v2_counterfactual", "on", "--ccl_v2_weight", "0.1", "--ccl_v2_temperature", "0.1",
                "--use_cfa_v2_textanchor", "on", "--cfa_v2_weight", "0.05", "--cfa_v2_anchor_weight", "1.0", "--cfa_v2_ema_momentum", "0.99",
            ],
        },
    ]


def yawdd_base_args(epochs: int, training_seed: int, output_json: Path, output_log: Path, checkpoint_path: Path) -> List[str]:
    return [
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
        "--epochs", str(epochs),
        "--lr", "0.00015",
        "--weight_decay", "0.01",
        "--label_smoothing", "0.1",
        "--loss_type", "focal",
        "--seed", str(YAWDD_SPLIT_SEED),
        "--training_seed", str(training_seed),
        "--output", str(output_json),
        "--log_file", str(output_log),
        "--checkpoint_output", str(checkpoint_path),
    ]


def aide_base_args(epochs: int, training_seed: int, output_json: Path, checkpoint_path: Path) -> List[str]:
    return [
        str(AIDE_SCRIPT),
        "--clip_mode", "offline_only",
        "--strict_frozen_clip", "on",
        "--prompt_set", "driving_7",
        "--epochs", str(epochs),
        "--batch_size", "32",
        "--lr", "0.00015",
        "--weight_decay", "0.0005",
        "--max_grad_norm", "1.0",
        "--num_frames", "5",
        "--adapter_hidden_dim", "2048",
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
        "--seed", str(AIDE_SPLIT_SEED),
        "--training_seed", str(training_seed),
        "--causal_group_source", "scene_vehicle",
        "--output", str(output_json),
        "--checkpoint_output", str(checkpoint_path),
    ]


def build_specs(datasets: Sequence[str], epochs: int, training_seeds: Sequence[int], config_filter: str = "") -> List[RunSpec]:
    specs: List[RunSpec] = []
    allowed = {item.strip() for item in config_filter.split(",") if item.strip()}
    for config in phase1_configs():
        config_id = str(config["config_id"])
        if allowed and config_id not in allowed:
            continue
        for training_seed in training_seeds:
            if "yawdd" in datasets:
                yawdd_dir = YAWDD_RESULTS_ROOT
                stem = f"phase1_yawdd_{config_id}_s{YAWDD_SPLIT_SEED}_t{training_seed}"
                specs.append(
                    RunSpec(
                        dataset="yawdd",
                        config_id=config_id,
                        split_seed=YAWDD_SPLIT_SEED,
                        training_seed=training_seed,
                        notes=str(config["notes"]),
                        command=yawdd_base_args(
                            epochs,
                            training_seed,
                            yawdd_dir / f"{stem}.json",
                            yawdd_dir / f"{stem}.log",
                            yawdd_dir / f"{stem}.ckpt.pt",
                        ) + list(config["yawdd_args"]),
                        output_json=yawdd_dir / f"{stem}.json",
                        output_log=yawdd_dir / f"{stem}.log",
                        checkpoint_path=yawdd_dir / f"{stem}.ckpt.pt",
                    )
                )
            if "aide" in datasets:
                aide_dir = AIDE_RESULTS_ROOT
                stem = f"phase1_aide_{config_id}_s{AIDE_SPLIT_SEED}_t{training_seed}"
                specs.append(
                    RunSpec(
                        dataset="aide",
                        config_id=config_id,
                        split_seed=AIDE_SPLIT_SEED,
                        training_seed=training_seed,
                        notes=str(config["notes"]),
                        command=aide_base_args(
                            epochs,
                            training_seed,
                            aide_dir / f"{stem}.json",
                            aide_dir / f"{stem}.ckpt.pt",
                        ) + list(config["aide_args"]),
                        output_json=aide_dir / f"{stem}.json",
                        output_log=aide_dir / f"{stem}.log",
                        checkpoint_path=aide_dir / f"{stem}.ckpt.pt",
                    )
                )
    return specs


def run_spec(spec: RunSpec, python_bin: str, force: bool, dry_run: bool) -> str:
    spec.output_json.parent.mkdir(parents=True, exist_ok=True)
    if spec.output_json.exists() and not force:
        print(f"[SKIP] {spec.dataset}:{spec.config_id}:seed{spec.training_seed} -> {spec.output_json}")
        return "skipped"
    command = [python_bin] + spec.command
    if dry_run:
        print(f"[DRYRUN] {spec.dataset}:{spec.config_id}:seed{spec.training_seed}")
        print(" ".join(command))
        return "dry-run"
    print(f"[RUN] {spec.dataset}:{spec.config_id}:seed{spec.training_seed}")
    start = time.time()
    if spec.dataset == "aide":
        with spec.output_log.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                command,
                cwd=AIDE_CLIP_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    else:
        subprocess.run(command, cwd=AIDE_CLIP_ROOT, check=True)
    print(f"[DONE] {spec.output_json.name} in {time.time() - start:.1f}s")
    return "completed"


def load_result(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(specs: Sequence[RunSpec]) -> None:
    manifest = []
    for spec in specs:
        manifest.append(
            {
                "dataset": spec.dataset,
                "config_id": spec.config_id,
                "split_seed": spec.split_seed,
                "training_seed": spec.training_seed,
                "notes": spec.notes,
                "output_json": str(spec.output_json),
                "output_log": str(spec.output_log),
                "checkpoint_path": str(spec.checkpoint_path),
            }
        )
    manifest_path = PHASE1_ROOT / "phase1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def summarize_dataset(dataset: str, specs: Sequence[RunSpec]) -> List[str]:
    dataset_specs = [spec for spec in specs if spec.dataset == dataset]
    grouped: Dict[str, List[RunSpec]] = {}
    for spec in dataset_specs:
        grouped.setdefault(spec.config_id, []).append(spec)

    lines = [f"## {dataset.upper()}", "", "| Config | Completed / Planned | Test accuracy | Test weighted F1 | Git commits | Notes |", "|---|---:|---|---|---|---|"]
    for config in phase1_configs():
        config_id = str(config["config_id"])
        runs = grouped.get(config_id, [])
        payloads = []
        commits = []
        for spec in runs:
            if spec.output_json.exists():
                payload = load_result(spec.output_json)
                payloads.append(payload)
                commits.append(str(payload.get("git_commit", "missing")))
        acc_values = [float(item.get("test", {}).get("accuracy", 0.0)) for item in payloads]
        wf1_values = [float(item.get("test", {}).get("weighted_f1", 0.0)) for item in payloads]
        commit_summary = ", ".join(sorted(set(commits))) if commits else "pending"
        lines.append(
            f"| {config_id} | {len(payloads)} / {len(runs)} | {mean_std(acc_values)} | {mean_std(wf1_values)} | {commit_summary} | {config['notes']} |"
        )
    lines.append("")
    return lines


def write_summary(specs: Sequence[RunSpec], args: argparse.Namespace) -> None:
    lines = [
        "# CCA-v2 Phase 1 Summary",
        "",
        f"- planned datasets: {', '.join(args.datasets)}",
        f"- planned training seeds: {', '.join(str(seed) for seed in args.training_seeds)}",
        f"- epochs per run: {args.epochs}",
        f"- dry_run: {args.dry_run}",
        "",
    ]
    for dataset in args.datasets:
        lines.extend(summarize_dataset(dataset, specs))
    summary_path = PHASE1_ROOT / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[DONE] wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize CCA-v2 Phase 1 sweeps.")
    parser.add_argument("--datasets", nargs="+", choices=["yawdd", "aide"], default=["yawdd", "aide"])
    parser.add_argument("--training-seeds", type=parse_csv_ints, default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--config-filter", default="", help="Comma-separated config ids to keep.")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for the number of specs to execute after filtering.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = build_specs(args.datasets, args.epochs, args.training_seeds, args.config_filter)
    if args.max_runs > 0:
        specs = specs[:args.max_runs]
    write_manifest(specs)
    if not args.summary_only:
        for spec in specs:
            run_spec(spec, args.python_bin, args.force, args.dry_run)
    write_summary(specs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())