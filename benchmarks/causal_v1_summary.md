# PACER Causal Extension — Ablation

## Table: Causal Component Effects

| Configuration | Acc (%) | wF1 (%) | ΔAcc | Source |
|---|---|---|---|---|
| A0 PACER baseline (no causal) | 61.54 | 46.89 | +0.00 | A0_pacer_baseline.json |
| A1 + CCL only | 38.46 | 21.37 | -23.08 | A1_ccl_only.json |
| A2 + CFA only | 61.54 | 46.89 | +0.00 | A2_cfa_only.json |
| A3 + CDA only | 61.54 | 46.89 | +0.00 | A3_cda_only.json |
| B1 CCL + CFA | 58.97 | 58.70 | -2.56 | B1_ccl_cfa.json |
| B2 CCL + CDA | 61.54 | 46.89 | +0.00 | B2_ccl_cda.json |
| B3 CFA + CDA | 66.67 | 66.18 | +5.13 | B3_cfa_cda.json |
| B4 All three (Full Causal) | 38.46 | 21.37 | -23.08 | B4_all_three.json |

## Table: Sensitivity Analysis

| Configuration | Acc (%) | wF1 (%) | ΔAcc | Source |
|---|---|---|---|---|
| C1 ccl_weight = 0.1 | 74.36 | 74.60 | +12.82 | C1_ccl_weight_0p1.json |
| C2 ccl_weight = 1.0 | 58.97 | 57.40 | -2.56 | C2_ccl_weight_1p0.json |
| C3 cfa_weight = 0.5 | 61.54 | 46.89 | +0.00 | C3_cfa_weight_0p5.json |
| C4 cda_prob = 0.5 | 38.46 | 25.44 | -23.08 | C4_cda_prob_0p5.json |

**Best config**: C1 ccl_weight = 0.1, acc = 74.36%, wF1 = 74.60%
Confusion matrix: {'notdrowsy': {'notdrowsy': 18, 'drowsy': 6}, 'drowsy': {'notdrowsy': 4, 'drowsy': 11}}

## Cross-dataset Causal Invariance

- status = missing_aide_ablation_numbers
- reason = Exact AIDE w/o Class Temp/Bias ablation was not found; only no_class_temperature and no_class_bias exist separately, so rank correlation was skipped.
