#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_metrics import MetricsError, load_json, normalize_payload


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def load_metrics_file(path: Path) -> dict:
    payload = load_json(path)
    return normalize_payload(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate metrics and emit a markdown report.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline metrics or raw result JSON.")
    parser.add_argument("--candidate", required=True, help="Path to the candidate metrics or raw result JSON.")
    parser.add_argument("--metric", default="weighted_f1", help="Metric name to compare. Defaults to weighted_f1.")
    parser.add_argument(
        "--mode",
        default="greater",
        choices=["greater", "greater_equal", "less", "less_equal"],
        help="Comparison direction. Defaults to greater.",
    )
    parser.add_argument("--output", default="compare_report.md", help="Markdown report output path.")
    return parser.parse_args()


def decide_pass(baseline: float, candidate: float, mode: str) -> bool:
    if mode == "greater":
        return candidate > baseline
    if mode == "greater_equal":
        return candidate >= baseline
    if mode == "less":
        return candidate < baseline
    if mode == "less_equal":
        return candidate <= baseline
    raise ValueError(f"unsupported mode: {mode}")


def build_report(
    baseline_payload: dict,
    candidate_payload: dict,
    metric_name: str,
    baseline_value: float,
    candidate_value: float,
    mode: str,
    passed: bool,
) -> str:
    absolute_delta = candidate_value - baseline_value
    relative_delta = None
    if baseline_value != 0:
        relative_delta = absolute_delta / baseline_value

    lines = [
        "# Server Evaluation Comparison Report",
        "",
        f"- Metric: {metric_name}",
        f"- Mode: {mode}",
        f"- Decision: {'PASS' if passed else 'FAIL'}",
        f"- Baseline source: {baseline_payload['source_file']}",
        f"- Candidate source: {candidate_payload['source_file']}",
        f"- Baseline section: {baseline_payload['selected_section']}",
        f"- Candidate section: {candidate_payload['selected_section']}",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Baseline | {format_metric(baseline_value)} |",
        f"| Candidate | {format_metric(candidate_value)} |",
        f"| Absolute delta | {absolute_delta:+.6f} |",
        f"| Relative delta | {f'{relative_delta:+.2%}' if relative_delta is not None else 'N/A'} |",
        "",
        "## Baseline Metrics",
        "",
    ]
    for key in sorted(baseline_payload["metrics"].keys()):
        lines.append(f"- {key}: {format_metric(baseline_payload['metrics'][key])}")
    lines.extend(["", "## Candidate Metrics", ""])
    for key in sorted(candidate_payload["metrics"].keys()):
        lines.append(f"- {key}: {format_metric(candidate_payload['metrics'][key])}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    output_path = Path(args.output).resolve()

    if not baseline_path.exists():
        raise SystemExit(f"[ERROR] baseline file not found: {baseline_path}")
    if not candidate_path.exists():
        raise SystemExit(f"[ERROR] candidate file not found: {candidate_path}")

    try:
        baseline_payload = load_metrics_file(baseline_path)
        candidate_payload = load_metrics_file(candidate_path)
    except MetricsError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    metric_name = args.metric
    baseline_value = baseline_payload["metrics"].get(metric_name)
    candidate_value = candidate_payload["metrics"].get(metric_name)

    if baseline_value is None:
        raise SystemExit(f"[ERROR] baseline metrics do not contain metric: {metric_name}")
    if candidate_value is None:
        raise SystemExit(f"[ERROR] candidate metrics do not contain metric: {metric_name}")

    passed = decide_pass(float(baseline_value), float(candidate_value), args.mode)
    report = build_report(
        baseline_payload=baseline_payload,
        candidate_payload=candidate_payload,
        metric_name=metric_name,
        baseline_value=float(baseline_value),
        candidate_value=float(candidate_value),
        mode=args.mode,
        passed=passed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n")

    print(f"[INFO] baseline {metric_name}={format_metric(float(baseline_value))}")
    print(f"[INFO] candidate {metric_name}={format_metric(float(candidate_value))}")
    print(f"[INFO] decision={'PASS' if passed else 'FAIL'}")
    print(f"[DONE] wrote report to {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())