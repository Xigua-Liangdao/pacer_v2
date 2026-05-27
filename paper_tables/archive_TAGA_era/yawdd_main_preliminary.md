# YawDD Main Preliminary

Ensemble impact check: S6 seed42 `use_test_ensemble=on` test_wf1=0.919795; `off` test_wf1=0.919795; diff=0.000000. Decision: keep ensemble on.

| setting | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | drowsy F1 Mean±Std | notdrowsy F1 Mean±Std |
| --- | --- | --- | --- | --- | --- | --- |
| A0 Ours full (YawDD) | 0.919 | 0.897 ± 0.028 | 0.921 | 0.897 ± 0.028 | 0.850 ± 0.043 | 0.921 ± 0.021 |

## Per-seed distribution

| seed | test_acc | test_wf1 | drowsy_f1 | notdrowsy_f1 | confusion_matrix | result_json |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.919355 | 0.919795 | 0.883721 | 0.938272 | {"notdrowsy": {"notdrowsy": 38, "drowsy": 3}, "drowsy": {"notdrowsy": 2, "drowsy": 19}} | /data1/yanjing/pacer_v2/results/final_runs/yawdd/A0_seed42_canonical/result.json |
| 123 | 0.870968 | 0.872287 | 0.818182 | 0.900000 | {"notdrowsy": {"notdrowsy": 36, "drowsy": 5}, "drowsy": {"notdrowsy": 3, "drowsy": 18}} | /data1/yanjing/pacer_v2/results/final_runs/yawdd/A0_seed123_canonical/result.json |
| 2024 | 0.919355 | 0.920512 | 0.888889 | 0.936709 | {"notdrowsy": {"notdrowsy": 37, "drowsy": 4}, "drowsy": {"notdrowsy": 1, "drowsy": 20}} | /data1/yanjing/pacer_v2/results/final_runs/yawdd/A0_seed2024_canonical/result.json |
| 7 | 0.919355 | 0.918857 | 0.878049 | 0.939759 | {"notdrowsy": {"notdrowsy": 39, "drowsy": 2}, "drowsy": {"notdrowsy": 3, "drowsy": 18}} | /data1/yanjing/pacer_v2/results/final_runs/yawdd/A0_seed7_canonical/result.json |
| 31 | 0.854839 | 0.853943 | 0.780488 | 0.891566 | {"notdrowsy": {"notdrowsy": 37, "drowsy": 4}, "drowsy": {"notdrowsy": 5, "drowsy": 16}} | /data1/yanjing/pacer_v2/results/final_runs/yawdd/A0_seed31_canonical/result.json |

wf1 list: seed42=0.919795, seed123=0.872287, seed2024=0.920512, seed7=0.918857, seed31=0.853943
