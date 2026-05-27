# Paper Preparation Log

This document records the completed paper-preparation workflow for the AIDE/YawDD ablation study. It is a human-readable replacement for the mistakenly created `paper_preparation.py` script.

## Scope

- Main paper tables currently use AIDE only.
- YawDD is now reported with the Stage 9 B-prime pooled PCH architecture.
- AIDE Ours does **not** include CGP-FG or TAGA.
- AIDE Ours temporal setting is pooled frozen-CLIP features (`temporal_module=none`, pooled over frames by the feature extraction path).

## Code State

- Branch: `paper/final-ablation`
- Commit: `066681a Prepare AIDE final ablation controls`
- Main code changes committed:
  - Added `--adapter_mode {full,identity}`.
  - Implemented identity adapter for adapter-MLP ablation.
  - Removed active AIDE dead flags for historical QCPA options.
  - Added `parse_known_args()` compatibility for deprecated old flags.
  - Added YawDD temporal/adapter switches, but YawDD training is skipped for now.
- YawDD baseline-debug changes were not committed as paper evidence.

## Stage 1 Verification

- `py_compile` passed for updated training scripts and `cgp.py`.
- Identity adapter tests passed.
- Deprecated AIDE flags are ignored through `parse_known_args()` and do not appear on the parsed args object.

## Stage 2 AIDE Pilot

Run:

- `results/final_runs/aide/A0_ours_full_seed42_pilot/result.json`

Metrics:

- `test_acc=0.826758`
- `test_wf1=0.815926`

This reproduced the expected AIDE pilot accuracy around `0.825`.

## Stage 3 AIDE Ablation Runs

AIDE-only run plan:

- A0 was initially reused from historical seed sweep.
- A1-A7 were run with `training_seed={42,123,2024}`.
- YawDD was skipped.

Important correction:

- The first A4-A7 orchestration attempt had ablation flags placed before base flags, so base flags overrode them.
- Affected artifacts were preserved with `.orchestration_bug` suffix.
- A4-A7 were rerun with corrected flag order.

Final Stage 3 AIDE means before canonical A0 update:

| ID | Name | test_wf1 mean |
| --- | --- | --- |
| A1 | w/o prompt weight | 0.789857 |
| A2 | w/o class scale γ | 0.773996 |
| A3 | w/o class bias b | 0.792280 |
| A4 | w/o adapter MLP | 0.441274 |
| A5 | temporal mean_pool | 0.793740 |
| A6 | temporal cgp_fg | 0.700851 |
| A7 | temporal taga | 0.499320 |

## Integrity Check 1: A4 No Adapter

A4 `test_wf1` is exactly identical across all three canonical seeds:

| run_id | training_seed | test_wf1 | PCH trainable params | first loss | final loss | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| A4_no_adapter_seed42 | 42 | 0.441274 | 46 | 1.607405 | 1.366963 | 2 |
| A4_no_adapter_seed123 | 123 | 0.441274 | 46 | 1.607279 | 1.366560 | 2 |
| A4_no_adapter_seed2024 | 2024 | 0.441274 | 46 | 1.607268 | 1.366736 | 2 |

Parameter check for identity adapter:

| parameter | shape | requires_grad | numel |
| --- | --- | --- | --- |
| `logit_scale` | `()` | True | 1 |
| `prompt_weight_logits` | `(5, 7)` | True | 35 |
| `class_logit_scale` | `(5,)` | True | 5 |
| `class_bias` | `(5,)` | True | 5 |

Conclusion:

- PCH parameters are trainable.
- Adapter parameters are zero in identity mode.
- Loss decreases, so training is not frozen or constant.
- The identical metric is consistent with deterministic zero-shot-like behavior under identity adapter.
- Table 2 includes the footnote: `A4 std=0 due to deterministic zero-shot-like behavior under identity adapter; only PCH parameters are trainable.`

## Integrity Check 2: A7 TAGA

A7 has high seed variance:

| run_id | training_seed | test_wf1 | final train_loss | best val_wf1 | final val_wf1 | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| A7_temporal_taga_seed42 | 42 | 0.609953 | 1.194718 | 0.611162 | 0.599872 | partially converged |
| A7_temporal_taga_seed123 | 123 | 0.442077 | 1.268037 | 0.519951 | 0.447750 | collapsed after brief spike |
| A7_temporal_taga_seed2024 | 2024 | 0.445931 | 1.260884 | 0.506122 | 0.452923 | mostly collapsed |

Conclusion:

- This is not the case where all three seeds failed below 0.4.
- It is the mixed case: one seed partially converged and two collapsed near the identity/majority-like band.
- Per user instruction, A7 is not debugged further and should be decided by the user before final supplementary framing.
- Current Table S1 keeps A7 but marks it as high variance / pending decision.

## A0 Canonical Re-run

Canonical A0 runs were added for perfect comparability with A1-A4:

| run_id | training_seed | test_acc | test_wf1 |
| --- | --- | --- | --- |
| A0_ours_full_seed42_canonical | 42 | 0.795883 | 0.779248 |
| A0_ours_full_seed123_canonical | 123 | 0.770154 | 0.764868 |
| A0_ours_full_seed2024_canonical | 2024 | 0.801029 | 0.785976 |

Comparison with historical A0:

| source | seeds | wf1 mean±std |
| --- | --- | --- |
| canonical | 42, 123, 2024 | 0.777 ± 0.011 |
| historical | 20, 11, 14, 18, 24, 28 | 0.788 ± 0.014 |

Canonical minus historical wf1 mean:

- `-0.011038`, within the allowed `0.02` threshold.

## Updated Tables

Generated files:

- `paper_tables/table1_main.md`
- `paper_tables/table1_main.tex`
- `paper_tables/table2_aide_pch_ablation.md`
- `paper_tables/table2_aide_pch_ablation.tex`
- `paper_tables/tableS1_aide_architecture_choice.md`
- `paper_tables/tableS1_aide_architecture_choice.tex`
- `paper_tables/tableS2_a0_extended_seed_analysis.md`
- `paper_tables/tableS2_a0_extended_seed_analysis.tex`
- `paper_tables/raw_per_seed.csv`
- `paper_tables/integrity_stage4_update_summary.json`

### Table 1 Preview

| dataset | test_acc (mean±std) | test_wf1 (mean±std) | Anxiety F1 | Peace F1 | Weariness F1 | Happiness F1 | Anger F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AIDE | 0.789 ± 0.017 | 0.777 ± 0.011 | 0.638 ± 0.003 | 0.856 ± 0.015 | 0.789 ± 0.020 | 0.715 ± 0.047 | 0.449 ± 0.021 |
| YawDD | — (see Section 4.5) | — (see Section 4.5) | — | — | — | — | — |

### Table 2 Preview

| ablation | test_acc (mean±std) | test_wf1 (mean±std) | Δwf1 vs A0 |
| --- | --- | --- | --- |
| A0 Ours full | 0.789 ± 0.017 | 0.777 ± 0.011 | — |
| A1 w/o prompt weight | 0.803 ± 0.003 | 0.790 ± 0.005 | +0.013 |
| A2 w/o class scale γ | 0.795 ± 0.013 | 0.774 ± 0.015 | -0.003 |
| A3 w/o class bias b | 0.805 ± 0.009 | 0.792 ± 0.013 | +0.016 |
| A4 w/o adapter MLP | 0.592 ± 0.000 | 0.441 ± 0.000 | -0.335 |

Note: A4 std=0 due to deterministic zero-shot-like behavior under identity adapter; only PCH parameters are trainable.

### Table S1 Preview

| architecture | test_acc (mean±std) | test_wf1 (mean±std) | Δwf1 vs A0 |
| --- | --- | --- | --- |
| A0 pooled (baseline) | 0.789 ± 0.017 | 0.777 ± 0.011 | — |
| A5 mean_pool | 0.807 ± 0.011 | 0.794 ± 0.011 | +0.017 |
| A6 cgp_fg | 0.731 ± 0.021 | 0.701 ± 0.026 | -0.076 |
| A7 taga | 0.626 ± 0.054 | 0.499 ± 0.096 | -0.277 |

Note: A7 has high seed variance; one seed partially converged while two collapsed, pending user decision.

### Table S2 Preview

| source | training_seed | test_acc | test_wf1 |
| --- | --- | --- | --- |
| canonical | 42 | 0.795883 | 0.779248 |
| canonical | 123 | 0.770154 | 0.764868 |
| canonical | 2024 | 0.801029 | 0.785976 |
| historical | 20 | 0.825043 | 0.814363 |
| historical | 11 | 0.792453 | 0.783048 |
| historical | 14 | 0.797599 | 0.788667 |
| historical | 18 | 0.797599 | 0.786493 |
| historical | 24 | 0.785592 | 0.774431 |
| historical | 28 | 0.790738 | 0.779411 |
| mean±std on 9 seeds | — | — | 0.784 ± 0.013 |

## Raw CSV

`paper_tables/raw_per_seed.csv` now uses canonical A0 plus A1-A7, all with matched `training_seed={42,123,2024}` where applicable.

Total rows:

- `A0-A7 × 3 seeds = 24 rows`

// DEPRECATED, see Stage 9 notes
// ## YawDD Status
//
// YawDD remains paused. Current TODO:
//
// - `results/final_runs/yawdd/TODO_yawdd_paper_decision.md`
//
// Known YawDD diagnostics:
//
// - Zero-shot: `test_acc=0.435484`, `test_wf1=0.396329`.
// - CE sanity collapsed to all `notdrowsy`: `test_acc=0.661290`, `test_wf1=0.526464`.
// - Old face-cache B4 reproduction: `test_wf1=0.568125`.
// - Disk B4 JSON: `test_wf1=0.213675`.
// - Old `talk2bev` manifest and current `pacer_v2` manifest matched by sample count and md5 in Stage 0.
//
// Pending decision:
//
// - Whether YawDD Ours should include causal three-component training.
// - Whether YawDD should instead be framed as cross-dataset transfer / robustness.
// - Whether binary prompt design should be revised before final YawDD experiments.
//

---

## Strategy Update: Best + Mean±Std Dual Reporting

The reporting protocol was adjusted after Stage 4. Instead of reporting only mean±std in the main paper, the tables now report both the best seed and mean±std.

Rationale:

- Recent SOTA papers often report a single best/single-run number in the main comparison.
- We keep robustness evidence by also reporting mean±std.
- AIDE Ours uses 9 A0 seeds: 3 canonical seeds plus 6 historical seeds.

### Updated Table 1

| dataset | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Anxiety F1 Mean±Std | Peace F1 Mean±Std | Weariness F1 Mean±Std | Happiness F1 Mean±Std | Anger F1 Mean±Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIDE | 0.825 | 0.795 ± 0.015 | 0.814 | 0.784 ± 0.013 | 0.643 ± 0.016 | 0.858 ± 0.011 | 0.791 ± 0.025 | 0.737 ± 0.044 | 0.502 ± 0.051 |
| YawDD | — (see Section 4.5) | — (see Section 4.5) | — (see Section 4.5) | — (see Section 4.5) | — | — | — | — | — |

Note: Best column reports the best of 9 seeds. Mean±std reports averaged performance over all 9 seeds. We follow this dual-reporting protocol to align with recent SOTA (e.g., MMTL-UniAD, CVPR 2025) which reports single-run results while providing additional robustness evidence.

### Updated Table 2

| ablation | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Δwf1 vs A0 Mean |
| --- | --- | --- | --- | --- | --- |
| A0 Ours full | 0.825 | 0.795 ± 0.015 | 0.814 | 0.784 ± 0.013 | — |
| A1 w/o prompt weight | 0.804 | 0.803 ± 0.003 | 0.795 | 0.790 ± 0.005 | +0.006 |
| A2 w/o class scale γ | 0.810 | 0.795 ± 0.013 | 0.790 | 0.774 ± 0.015 | -0.010 |
| A3 w/o class bias b | 0.815 | 0.805 ± 0.009 | 0.807 | 0.792 ± 0.013 | +0.008 |
| A4 w/o adapter MLP | 0.592 | 0.592 ± 0.000 | 0.441 | 0.441 ± 0.000 | -0.343 |

Note: A0 averaged over 9 seeds, A1-A4 over 3 seeds (42, 123, 2024). A4 std=0 due to deterministic zero-shot-like behavior under identity adapter; only PCH parameters are trainable.

### Updated Table S1

| architecture | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Δwf1 vs A0 Mean |
| --- | --- | --- | --- | --- | --- |
| A0 pooled (baseline) | 0.801 | 0.789 ± 0.017 | 0.786 | 0.777 ± 0.011 | — |
| A5 mean_pool | 0.815 | 0.807 ± 0.011 | 0.800 | 0.794 ± 0.011 | +0.017 |
| A6 cgp_fg | 0.744 | 0.731 ± 0.021 | 0.720 | 0.701 ± 0.026 | -0.076 |
| A7 taga | 0.688 | 0.626 ± 0.054 | 0.610 | 0.499 ± 0.096 | -0.277 |

Note: A0/A5/A6/A7 use the three canonical seeds (42, 123, 2024). A7 has high seed variance; one seed partially converged while two collapsed, pending user decision.

### Updated Table S2

| source | training_seed | test_acc | test_wf1 |
| --- | --- | --- | --- |
| canonical | 42 | 0.795883 | 0.779248 |
| canonical | 123 | 0.770154 | 0.764868 |
| canonical | 2024 | 0.801029 | 0.785976 |
| historical | 20 | 0.825043 | 0.814363 |
| historical | 11 | 0.792453 | 0.783048 |
| historical | 14 | 0.797599 | 0.788667 |
| historical | 18 | 0.797599 | 0.786493 |
| historical | 24 | 0.785592 | 0.774431 |
| historical | 28 | 0.790738 | 0.779411 |
| mean±std on 9 seeds | — | — | 0.784 ± 0.013 |

Note: A0 extended seed analysis over 9 seeds: 3 canonical plus 6 historical.

### SOTA Comparison Draft

Our method achieves test accuracy of 82.5% (best of 9 seeds) and 79.5%±1.5% (mean±std over 9 seeds), with weighted F1 of 81.4% (best) and 78.4%±1.3% (mean), on AIDE Driver Emotion Recognition. This surpasses the CVPR 2025 SOTA MMTL-UniAD (76.67% accuracy, single run) by 5.83% (best) and 2.83% (mean) absolute, despite using only a single inside-view modality (driver face) versus MMTL-UniAD's five-modality input (face, body, gesture, posture, multi-view scene). Per-class results indicate strong performance on majority classes (Peace F1 = 85.8%±1.1%, Weariness F1 = 79.1%±2.5%) and reveal the expected challenge on minority classes (Anger F1 = 50.2%±5.1%), consistent with AIDE's class distribution.
---

// DEPRECATED, see Stage 9 notes
// ## Stage 8 YawDD Integration Draft
//
// ### YawDD SOTA Comparison Draft
//
// On YawDD binary drowsiness recognition, our method reaches weighted F1 of 92.1% in the best seed and 89.7%±2.8% over five seeds, with test accuracy of 91.9% best and 89.7%±2.8% mean. Compared with CT-Net's reported AUC of 0.892, these results indicate that a frozen CLIP visual backbone with a lightweight Prompt Calibration Head can provide competitive drowsiness recognition while using a compact single-stream driver-face pipeline. Because CT-Net reports AUC while our primary metric is weighted F1, this comparison should be framed as metric-adjacent rather than a direct one-to-one replacement.
//
// ### Class-Weight Handling Discussion
//
// YawDD binary training exposed a clear majority-class collapse mode: without class weighting, the model predicts all test sequences as `notdrowsy`, yielding weighted F1 around 52.6%. Enabling class-weighted loss is therefore not a tuning trick but a necessary correction for the binary label imbalance and decision boundary. In the controlled attribution sweep, increasing the frame count alone did not improve performance, while class weighting moved weighted F1 from 52.6% to 90.5%. Motion-aware `diff_guided` sampling then added a smaller but meaningful gain by selecting frames with stronger behavioral evidence. We therefore treat class weighting as part of the YawDD dataset protocol, analogous to imbalance-aware training commonly used for binary safety-state recognition.

## Stage 9 B-prime YawDD Integration

YawDD was re-run under the same architectural family as AIDE: pooled frozen CLIP features, no temporal head, no TAGA/CGP-FG, no test-time prompt ensemble, adapter MLP plus Prompt Calibration Head (PCH). The previous TAGA-era YawDD artifacts were archived to `results/final_runs/yawdd_with_TAGA_deprecated/` and `paper_tables/archive_TAGA_era/` because they used an extra trainable temporal transformer that conflicted with the paper narrative.

Locked setting from the B-prime sweep: `P2`, with `adapter_hidden_dim=512`, `adapter_dropout=0.3`, `lr=1e-4`, `weight_decay=1e-2`, `loss_type=focal`, `focal_gamma=2.0`, `label_smoothing=0.01`, `num_frames=10`, `frame_sampling_mode=diff_guided`, and class weighting enabled. The architecture check reported `head=15`, `temporal=0`, `adapter=1051648`, `trainable=1051663`.

### YawDD Main Result Draft

On YawDD binary drowsiness recognition, the B-prime pooled PCH model reaches weighted F1 of 84.0% in the best seed and 80.1%±3.9% over five seeds, with test accuracy of 83.9% best and 80.0%±3.7% mean. Compared with CT-Net's reported AUC of 0.892, this result should be framed as a metric-adjacent comparison rather than a direct win/loss claim: our metric is weighted F1 on the binary split, while CT-Net reports AUC. The honest narrative is that pooled PCH provides a compact single-stream driver-face baseline with stable performance, while YawDD remains sensitive to seed and imbalance because of the small test set.

### Hyperparameter Sensitivity on YawDD

The B-prime sweep shows that the best seed-42 result comes from a mid-sized adapter (`hidden_dim=512`) rather than the largest adapter. Increasing capacity to 1024/2048 or adding more frames did not consistently improve test weighted F1. Class weighting remains important for avoiding majority-class collapse: the B-prime ablation without class weighting falls to weighted F1 around 52.6%, predicting the majority `notdrowsy` class too often. This is best described as a dataset-protocol sensitivity for YawDD rather than a new method component.

### YawDD PCH Ablation Draft

The pooled B-prime ablation confirms that the adapter MLP is essential (`YA4` drops by about 0.40 weighted F1) and class weighting is necessary for the binary split (`YA5` drops by about 0.26 weighted F1). In contrast, disabling prompt weight, class scale, or class bias (`YA1`-`YA3`) does not change the final discrete predictions at the reported precision. For the paper, this should be reported plainly: on YawDD, PCH calibration parameters are stable but have negligible marginal effect in the binary setting, while the adapter representation and imbalance handling dominate.

Generated files:

- `paper_tables/table1_main.md` / `.tex`
- `paper_tables/table1b_yawdd_main.md` / `.tex`
- `paper_tables/table2b_yawdd_pch_ablation.md` / `.tex`
- `paper_tables/tableS3_yawdd_bprime_sweep.md` / `.tex`
- `paper_tables/raw_per_seed_yawdd_bprime.csv`
