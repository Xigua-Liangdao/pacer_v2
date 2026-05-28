# Stage 9.5 Verify Report

- branch: `paper/final-ablation`
- commit: `05d5fe0`
- status: `PASS`

## Recomputed Main Result

| seed | test_acc | test_wf1 | drowsy_f1 | notdrowsy_f1 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.790323 | 0.794668 | 0.723404 | 0.831169 |
| 31 | 0.838710 | 0.840359 | 0.772727 | 0.875000 |
| 42 | 0.822581 | 0.821486 | 0.731707 | 0.867470 |
| 123 | 0.806452 | 0.808431 | 0.727273 | 0.850000 |
| 2024 | 0.741935 | 0.738556 | 0.600000 | 0.809524 |

- acc best: `0.839`
- acc mean±std: `0.800 ± 0.037`
- wf1 best: `0.840`
- wf1 mean±std: `0.801 ± 0.039`
- drowsy F1 mean±std: `0.711 ± 0.065`
- notdrowsy F1 mean±std: `0.847 ± 0.027`

## Ablation Integrity

| ablation | seed wf1 values | mean wf1 |
| --- | --- | ---: |
| YA0 | 0.821486, 0.808431, 0.738556 | 0.789491 |
| YA1 | 0.821486, 0.808431, 0.738556 | 0.789491 |
| YA2 | 0.821486, 0.808431, 0.738556 | 0.789491 |
| YA3 | 0.821486, 0.808431, 0.738556 | 0.789491 |
| YA4 | 0.361834, 0.361834, 0.437860 | 0.387176 |
| YA5 | 0.526464, 0.526464, 0.526464 | 0.526464 |

## Architecture Checks

- All B-prime main and ablation logs show `temporal=0`.
- Main/YA1/YA2/YA3/YA5 adapter params: `1051648`; trainable params: `1051663`.
- YA4 identity adapter params: `0`; trainable params: `15`.
- `use_test_ensemble=False` in all checked result configs.

## Sweep Check

- 12 configs found; top config: `P2` with test_wf1 `0.821486`.

## Warnings

- none

## Errors

- none
