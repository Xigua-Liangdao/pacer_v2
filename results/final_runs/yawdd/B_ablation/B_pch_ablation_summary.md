# Stage 9 Step 8 B-prime YawDD PCH Ablation Summary

Architecture constraints: pooled CLIP + adapter MLP + PCH; `temporal_head=none`, `temporal_module=none`, `use_test_ensemble=off`.

| ablation | seeds | acc mean±std | wf1 mean±std | best wf1 | Δwf1 vs YA0 mean |
| --- | --- | --- | --- | ---: | ---: |
| YA0 Ours full | 42,123,2024 | 0.790323 ± 0.042674 | 0.789491 ± 0.044591 | 0.821486 | — |
| YA1 w/o prompt weight | 42,123,2024 | 0.790323 ± 0.042674 | 0.789491 ± 0.044591 | 0.821486 | +0.000000 |
| YA2 w/o class scale gamma | 42,123,2024 | 0.790323 ± 0.042674 | 0.789491 ± 0.044591 | 0.821486 | +0.000000 |
| YA3 w/o class bias b | 42,123,2024 | 0.790323 ± 0.042674 | 0.789491 ± 0.044591 | 0.821486 | +0.000000 |
| YA4 w/o adapter MLP | 42,123,2024 | 0.419355 ± 0.027936 | 0.387176 ± 0.043894 | 0.437860 | -0.402315 |
| YA5 w/o class weight | 42,123,2024 | 0.661290 ± 0.000000 | 0.526464 ± 0.000000 | 0.526464 | -0.263027 |

## Per-seed rows

| ablation | seed | test_acc | test_wf1 | val_wf1 | drowsy_f1 | path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YA0 | 42 | 0.822581 | 0.821486 | 0.897417 | NA | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_A0_seed42_canonical |
| YA0 | 123 | 0.806452 | 0.808431 | 0.877551 | NA | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_A0_seed123_canonical |
| YA0 | 2024 | 0.741935 | 0.738556 | 0.874175 | NA | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_A0_seed2024_canonical |
| YA1 | 42 | 0.822581 | 0.821486 | 0.897417 | 0.731707 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA1_seed42_canonical |
| YA1 | 123 | 0.806452 | 0.808431 | 0.877551 | 0.727273 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA1_seed123_canonical |
| YA1 | 2024 | 0.741935 | 0.738556 | 0.874175 | 0.6 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA1_seed2024_canonical |
| YA2 | 42 | 0.822581 | 0.821486 | 0.897417 | 0.731707 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA2_seed42_canonical |
| YA2 | 123 | 0.806452 | 0.808431 | 0.877551 | 0.727273 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA2_seed123_canonical |
| YA2 | 2024 | 0.741935 | 0.738556 | 0.874175 | 0.6 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA2_seed2024_canonical |
| YA3 | 42 | 0.822581 | 0.821486 | 0.897417 | 0.731707 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA3_seed42_canonical |
| YA3 | 123 | 0.806452 | 0.808431 | 0.877551 | 0.727273 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA3_seed123_canonical |
| YA3 | 2024 | 0.741935 | 0.738556 | 0.874175 | 0.6 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA3_seed2024_canonical |
| YA4 | 42 | 0.403226 | 0.361834 | 0.348694 | 0.478873 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA4_seed42_canonical |
| YA4 | 123 | 0.403226 | 0.361834 | 0.314221 | 0.478873 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA4_seed123_canonical |
| YA4 | 2024 | 0.451613 | 0.437860 | 0.335790 | 0.484848 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA4_seed2024_canonical |
| YA5 | 42 | 0.661290 | 0.526464 | 0.464996 | 0.0 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA5_seed42_canonical |
| YA5 | 123 | 0.661290 | 0.526464 | 0.464996 | 0.0 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA5_seed123_canonical |
| YA5 | 2024 | 0.661290 | 0.526464 | 0.464996 | 0.0 | /data1/yanjing/pacer_v2/results/final_runs/yawdd/B_ablation/B_YA5_seed2024_canonical |

- YA1-YA4 identical check: `False`
- If `True`, do not use the ablation table without further investigation.
