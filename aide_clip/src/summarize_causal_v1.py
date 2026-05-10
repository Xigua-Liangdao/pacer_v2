import json
from pathlib import Path

R = Path("/data1/yanjing/talk2bev/aide_clip/results/yawdd/causal_v1")

AB_EXPS = [
    ("A0 PACER baseline (no causal)", "A0_pacer_baseline"),
    ("A1 + CCL only", "A1_ccl_only"),
    ("A2 + CFA only", "A2_cfa_only"),
    ("A3 + CDA only", "A3_cda_only"),
    ("B1 CCL + CFA", "B1_ccl_cfa"),
    ("B2 CCL + CDA", "B2_ccl_cda"),
    ("B3 CFA + CDA", "B3_cfa_cda"),
    ("B4 All three (Full Causal)", "B4_all_three"),
]

SENS_EXPS = [
    ("C1 ccl_weight = 0.1", "C1_ccl_weight_0p1"),
    ("C2 ccl_weight = 1.0", "C2_ccl_weight_1p0"),
    ("C3 cfa_weight = 0.5", "C3_cfa_weight_0p5"),
    ("C4 cda_prob = 0.5", "C4_cda_prob_0p5"),
]


def find(prefix):
    cands = [c for c in R.glob(f"{prefix}*.json") if ".fold" not in c.name and "cross_dataset" not in c.name]
    return min(cands, key=lambda p: len(p.name)) if cands else None


def get(prefix):
    path = find(prefix)
    if not path:
        return None
    with path.open() as handle:
        data = json.load(handle)
    test = data.get("test") or {}
    acc = test.get("accuracy")
    if isinstance(acc, dict):
        acc = acc.get("mean")
    f1 = test.get("weighted_f1") or test.get("f1")
    if isinstance(f1, dict):
        f1 = f1.get("mean")
    return {"acc": acc, "f1": f1, "cm": test.get("confusion_matrix"), "src": path.name}


def fmt_percent(value):
    return f"{value * 100:.2f}" if value is not None else "N/A"


def print_table(title, rows, baseline_acc=None):
    print(f"## {title}\n")
    print("| Configuration | Acc (%) | wF1 (%) | ΔAcc | Source |")
    print("|---|---|---|---|---|")
    for name, prefix in rows:
        result = get(prefix)
        if not result:
            print(f"| {name} | N/A | N/A | N/A | NOT_FOUND |")
            continue
        delta = None
        if baseline_acc is not None and result["acc"] is not None:
            delta = (result["acc"] - baseline_acc) * 100.0
        delta_str = f"{delta:+.2f}" if delta is not None else "—"
        print(f"| {name} | {fmt_percent(result['acc'])} | {fmt_percent(result['f1'])} | {delta_str} | {result['src']} |")
    print()


def best_result(rows):
    best_name = None
    best_payload = None
    for name, prefix in rows:
        payload = get(prefix)
        if not payload or payload["acc"] is None:
            continue
        if best_payload is None or payload["acc"] > best_payload["acc"]:
            best_name = name
            best_payload = payload
    return best_name, best_payload


if __name__ == "__main__":
    print("# PACER Causal Extension — Ablation\n")
    baseline = get("A0_pacer_baseline")
    base_acc = baseline["acc"] if baseline else None

    print_table("Table: Causal Component Effects", AB_EXPS, baseline_acc=base_acc)
    print_table("Table: Sensitivity Analysis", SENS_EXPS, baseline_acc=base_acc)

    best_name, best_payload = best_result(AB_EXPS + SENS_EXPS)
    if best_payload:
        print(f"**Best config**: {best_name}, acc = {fmt_percent(best_payload['acc'])}%, wF1 = {fmt_percent(best_payload['f1'])}%")
        print(f"Confusion matrix: {best_payload['cm']}")
    else:
        print("**Best config**: NOT_FOUND")

    inv_path = R / "cross_dataset_invariance.json"
    if inv_path.exists():
        with inv_path.open() as handle:
            inv = json.load(handle)
        print("\n## Cross-dataset Causal Invariance\n")
        if inv.get("kendall_tau") is None or inv.get("spearman_rho") is None:
            print(f"- status = {inv.get('status', 'missing')}")
            if inv.get("reason"):
                print(f"- reason = {inv['reason']}")
        else:
            print(f"- Kendall tau = {inv.get('kendall_tau'):.3f}, p = {inv.get('kendall_p'):.4f}")
            print(f"- Spearman rho = {inv.get('spearman_rho'):.3f}, p = {inv.get('spearman_p'):.4f}")