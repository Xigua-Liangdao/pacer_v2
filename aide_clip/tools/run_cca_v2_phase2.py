import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import run_cca_v2_phase1 as phase1


PHASE2_ROOT = phase1.AIDE_CLIP_ROOT / "results" / "cca_v2" / "phase2"
YAWDD_RESULTS_ROOT = PHASE2_ROOT / "yawdd"
AIDE_RESULTS_ROOT = PHASE2_ROOT / "aide"
DEFAULT_PHASE2_SEEDS = [30, 31, 32]
PHASE1_RESULT_RE = re.compile(r"phase1_(yawdd|aide)_(.+)_s\d+_t\d+\.json$")


@dataclass
class Phase1Aggregate:
    dataset: str
    config_id: str
    count: int
    mean_accuracy: float
    mean_weighted_f1: float
    source_paths: List[str]


@dataclass
class RunSpec:
    dataset: str
    config_id: str
    training_seed: int
    notes: str
    output_json: Path
    output_log: Path
    checkpoint_path: Path
    command: List[str]


def load_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_phase1_aggregates(datasets: Sequence[str]) -> Dict[str, List[Phase1Aggregate]]:
    grouped: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = {}
    dataset_dirs = {
        "yawdd": phase1.YAWDD_RESULTS_ROOT,
        "aide": phase1.AIDE_RESULTS_ROOT,
    }
    for dataset in datasets:
        for path in sorted(dataset_dirs[dataset].glob("phase1_*.json")):
            match = PHASE1_RESULT_RE.match(path.name)
            if not match:
                continue
            _, config_id = match.groups()
            payload = load_json(path)
            test = payload.get("test", {})
            acc = float(test.get("accuracy", 0.0))
            wf1 = float(test.get("weighted_f1", 0.0))
            grouped.setdefault((dataset, config_id), []).append((acc, wf1, str(path)))

    result: Dict[str, List[Phase1Aggregate]] = {dataset: [] for dataset in datasets}
    for (dataset, config_id), rows in grouped.items():
        acc_values = [row[0] for row in rows]
        wf1_values = [row[1] for row in rows]
        result[dataset].append(
            Phase1Aggregate(
                dataset=dataset,
                config_id=config_id,
                count=len(rows),
                mean_accuracy=float(statistics.mean(acc_values)),
                mean_weighted_f1=float(statistics.mean(wf1_values)),
                source_paths=[row[2] for row in rows],
            )
        )
    for dataset in datasets:
        result[dataset].sort(key=lambda item: (item.mean_weighted_f1, item.mean_accuracy), reverse=True)
    return result


def select_phase2_configs(
    aggregates: Dict[str, List[Phase1Aggregate]],
    top_k: int,
    min_phase1_runs: int,
    include_baseline_anchor: bool,
    require_beat_baseline: bool,
    baseline_margin: float,
) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    for dataset, rows in aggregates.items():
        eligible = [row for row in rows if row.count >= min_phase1_runs]
        baseline_row = next((row for row in eligible if row.config_id == "baseline"), None)
        threshold = None
        if baseline_row is not None and require_beat_baseline:
            threshold = baseline_row.mean_weighted_f1 + baseline_margin
        chosen = []
        for row in eligible:
            if row.config_id == "baseline":
                continue
            if threshold is not None and row.mean_weighted_f1 < threshold:
                continue
            chosen.append(row.config_id)
            if len(chosen) >= top_k:
                break
        if include_baseline_anchor and any(row.config_id == "baseline" for row in eligible):
            chosen = ["baseline"] + chosen
        selected[dataset] = chosen
    return selected


def config_lookup() -> Dict[str, Dict[str, object]]:
    return {str(item["config_id"]): item for item in phase1.phase1_configs()}


def build_phase2_specs(
    datasets: Sequence[str],
    selected_configs: Dict[str, List[str]],
    training_seeds: Sequence[int],
    epochs: int,
) -> List[RunSpec]:
    lookup = config_lookup()
    specs: List[RunSpec] = []
    for dataset in datasets:
        for config_id in selected_configs.get(dataset, []):
            config = lookup[config_id]
            for training_seed in training_seeds:
                if dataset == "yawdd":
                    stem = f"phase2_yawdd_{config_id}_s{phase1.YAWDD_SPLIT_SEED}_t{training_seed}"
                    output_json = YAWDD_RESULTS_ROOT / f"{stem}.json"
                    output_log = YAWDD_RESULTS_ROOT / f"{stem}.log"
                    checkpoint_path = YAWDD_RESULTS_ROOT / f"{stem}.ckpt.pt"
                    command = phase1.yawdd_base_args(epochs, training_seed, output_json, output_log, checkpoint_path) + list(config["yawdd_args"])
                else:
                    stem = f"phase2_aide_{config_id}_s{phase1.AIDE_SPLIT_SEED}_t{training_seed}"
                    output_json = AIDE_RESULTS_ROOT / f"{stem}.json"
                    output_log = AIDE_RESULTS_ROOT / f"{stem}.log"
                    checkpoint_path = AIDE_RESULTS_ROOT / f"{stem}.ckpt.pt"
                    command = phase1.aide_base_args(epochs, training_seed, output_json, checkpoint_path) + list(config["aide_args"])
                specs.append(
                    RunSpec(
                        dataset=dataset,
                        config_id=config_id,
                        training_seed=training_seed,
                        notes=str(config["notes"]),
                        output_json=output_json,
                        output_log=output_log,
                        checkpoint_path=checkpoint_path,
                        command=command,
                    )
                )
    return specs


def write_selection(aggregates: Dict[str, List[Phase1Aggregate]], selected_configs: Dict[str, List[str]], args: argparse.Namespace) -> None:
    payload = {
        "datasets": list(args.datasets),
        "top_k": args.top_k,
        "min_phase1_runs": args.min_phase1_runs,
        "include_baseline_anchor": args.include_baseline_anchor,
        "require_beat_baseline": args.require_beat_baseline,
        "baseline_margin": args.baseline_margin,
        "phase2_training_seeds": list(args.training_seeds),
        "selected_configs": selected_configs,
        "phase1_aggregates": {
            dataset: [
                {
                    "config_id": row.config_id,
                    "count": row.count,
                    "mean_accuracy": row.mean_accuracy,
                    "mean_weighted_f1": row.mean_weighted_f1,
                    "source_paths": row.source_paths,
                }
                for row in rows
            ]
            for dataset, rows in aggregates.items()
        },
    }
    path = PHASE2_ROOT / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
                cwd=phase1.AIDE_CLIP_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    else:
        subprocess.run(command, cwd=phase1.AIDE_CLIP_ROOT, check=True)
    print(f"[DONE] {spec.output_json.name} in {time.time() - start:.1f}s")
    return "completed"


def mean_std(values: Sequence[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.6f} +- 0.000000 (n=1)"
    return f"{statistics.mean(values):.6f} +- {statistics.stdev(values):.6f} (n={len(values)})"


def write_summary(specs: Sequence[RunSpec], selected_configs: Dict[str, List[str]], aggregates: Dict[str, List[Phase1Aggregate]], args: argparse.Namespace) -> None:
    lines = [
        "# CCA-v2 Phase 2 Summary",
        "",
        f"- datasets: {', '.join(args.datasets)}",
        f"- phase2 training seeds: {', '.join(str(seed) for seed in args.training_seeds)}",
        f"- epochs per run: {args.epochs}",
        f"- dry_run: {args.dry_run}",
        "",
    ]
    for dataset in args.datasets:
        lines.append(f"## {dataset.upper()} Selection")
        lines.append("")
        lines.append("| Config | Phase1 completed runs | Phase1 mean acc | Phase1 mean wF1 | Selected for Phase2 |")
        lines.append("|---|---:|---:|---:|---|")
        selected_set = set(selected_configs.get(dataset, []))
        for row in aggregates.get(dataset, []):
            lines.append(
                f"| {row.config_id} | {row.count} | {row.mean_accuracy:.6f} | {row.mean_weighted_f1:.6f} | {'yes' if row.config_id in selected_set else 'no'} |"
            )
        lines.append("")

        dataset_specs = [spec for spec in specs if spec.dataset == dataset]
        grouped: Dict[str, List[RunSpec]] = {}
        for spec in dataset_specs:
            grouped.setdefault(spec.config_id, []).append(spec)

        lines.append(f"## {dataset.upper()} Phase2 Runs")
        lines.append("")
        lines.append("| Config | Completed / Planned | Test accuracy | Test weighted F1 |")
        lines.append("|---|---:|---|---|")
        for config_id in selected_configs.get(dataset, []):
            runs = grouped.get(config_id, [])
            payloads = [load_json(spec.output_json) for spec in runs if spec.output_json.exists()]
            acc_values = [float(item.get("test", {}).get("accuracy", 0.0)) for item in payloads]
            wf1_values = [float(item.get("test", {}).get("weighted_f1", 0.0)) for item in payloads]
            lines.append(f"| {config_id} | {len(payloads)} / {len(runs)} | {mean_std(acc_values)} | {mean_std(wf1_values)} |")
        lines.append("")

    path = PHASE2_ROOT / "summary.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[DONE] wrote summary to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and run CCA-v2 Phase 2 confirmation jobs from completed Phase 1 results.")
    parser.add_argument("--datasets", nargs="+", choices=["yawdd", "aide"], default=["yawdd", "aide"])
    parser.add_argument("--training-seeds", type=phase1.parse_csv_ints, default=DEFAULT_PHASE2_SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--python-bin", default=phase1.DEFAULT_PYTHON)
    parser.add_argument("--top-k", type=int, default=2, help="Number of non-baseline configs to confirm per dataset.")
    parser.add_argument("--min-phase1-runs", type=int, default=3, help="Minimum completed Phase 1 runs required before a config can be selected.")
    parser.add_argument("--include-baseline-anchor", action="store_true", default=True)
    parser.add_argument("--no-baseline-anchor", dest="include_baseline_anchor", action="store_false")
    parser.add_argument("--require-beat-baseline", action="store_true", default=True)
    parser.add_argument("--allow-weaker-than-baseline", dest="require_beat_baseline", action="store_false")
    parser.add_argument("--baseline-margin", type=float, default=0.0, help="Required weighted-F1 improvement over baseline for a non-baseline config to enter Phase 2.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregates = collect_phase1_aggregates(args.datasets)
    selected_configs = select_phase2_configs(
        aggregates,
        top_k=args.top_k,
        min_phase1_runs=args.min_phase1_runs,
        include_baseline_anchor=args.include_baseline_anchor,
        require_beat_baseline=args.require_beat_baseline,
        baseline_margin=args.baseline_margin,
    )
    write_selection(aggregates, selected_configs, args)
    specs = build_phase2_specs(args.datasets, selected_configs, args.training_seeds, args.epochs)
    if not args.summary_only:
        for spec in specs:
            run_spec(spec, args.python_bin, args.force, args.dry_run)
    write_summary(specs, selected_configs, aggregates, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())