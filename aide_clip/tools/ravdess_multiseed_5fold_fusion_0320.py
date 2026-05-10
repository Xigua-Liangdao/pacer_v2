#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed late fusion for 5 RAVDESS folds and summarize best/average.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--fusion_script", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--select_metric", default="weighted_f1")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = parse_int_list(args.seeds)
    folds = parse_int_list(args.folds)
    per_fold = []
    accuracies = []
    wf1s = []

    for fold in folds:
        auto_paths = []
        expc_paths = []
        feature_cache = None
        for seed in seeds:
            auto_json = results_dir / f"auto_fold{fold}_seed{seed}.json"
            expc_json = results_dir / f"expC_fold{fold}_seed{seed}.json"
            auto_ckpt = results_dir / f"auto_fold{fold}_seed{seed}.ckpt.pt"
            expc_ckpt = results_dir / f"expC_fold{fold}_seed{seed}.ckpt.pt"
            auto_paths.append(str(auto_ckpt))
            expc_paths.append(str(expc_ckpt))
            if feature_cache is None:
                feature_cache = read_json(auto_json)["config"].get("resolved_feature_cache_path")

        fold_output = results_dir / f"fusion_fold{fold}.json"
        cmd = [
            "/home/yanjing/anaconda3/envs/mmtl/bin/python",
            args.fusion_script,
            "--group_a",
            ",".join(expc_paths),
            "--group_b",
            ",".join(auto_paths),
            "--feature_cache",
            str(feature_cache),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--select_metric",
            args.select_metric,
            "--output",
            str(fold_output),
        ]
        subprocess.run(cmd, check=True)
        fold_result = read_json(fold_output)
        best_test = fold_result["best_fusion"]["test"]
        per_fold.append({
            "fold": fold,
            "best_fusion": fold_result["best_fusion"],
            "fusion_output": str(fold_output),
        })
        accuracies.append(best_test["accuracy"])
        wf1s.append(best_test["weighted_f1"])

    best_fold = max(per_fold, key=lambda item: (item["best_fusion"]["test"]["accuracy"], item["best_fusion"]["test"]["weighted_f1"]))
    summary = {
        "per_fold": per_fold,
        "best_fold": best_fold,
        "average_test_accuracy": round(sum(accuracies) / len(accuracies), 6),
        "average_test_weighted_f1": round(sum(wf1s) / len(wf1s), 6),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
