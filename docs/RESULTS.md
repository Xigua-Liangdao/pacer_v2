# Reference results

These are the paper-facing table values from the latest ITS manuscript. They are included to keep row naming, protocols, and metric aggregation unambiguous.

## AIDE Main Result

| Method | Input | Accuracy | Weighted F1 | Macro-F1 | Trainable parameters |
|---|---|---:|---:|---:|---:|
| Frozen CLIP + Adapter + PCH | in-cabin RGB only | 82.50 | 81.40 | 73.57 | 10.5M |

## AIDE Representative Comparison

| Category | Method | Input | Acc | Macro-F1 |
|---|---|---|---:|---:|
| 2D | GLMDriveNet | 5 streams | 71.38 | — |
| 2D+T | ResNet/TransE ensemble | 5 streams | 72.65 | — |
| 3D | TimeSFormer + ST-GCN | 5 streams | 74.87 | — |
| 3D | MMTL-UniAD/MARNet | 5 streams | 76.67 | — |
| CLIP | Linear probe | RGB | 63.52 | 32.31 |
| CLIP | CLIP-Adapter | RGB | 72.38 | 60.38 |
| CLIP | MaPLe | RGB | 72.44 | 60.24 |
| CLIP | CoCoOp | RGB | 73.99 | 60.84 |
| Ours | Frozen CLIP + Adapter + PCH | RGB | 82.50 | 73.57 |

The upper block contains published AIDE reference results from the AIDE/MMTL-UniAD literature. The lower block contains reproduced single-RGB CLIP-adaptation baselines under the repository protocol.

## AIDE Class Support

| Dataset | Class | Support | F1 |
|---|---|---:|---:|
| AIDE | Anxiety | 81 | 61.41 ± 2.87 |
| AIDE | Peace | 346 | 86.53 ± 0.43 |
| AIDE | Weariness | 64 | 82.80 ± 2.69 |
| AIDE | Happiness | 49 | 74.55 ± 2.71 |
| AIDE | Anger | 43 | 48.11 ± 2.31 |

## YawDD Class Support

| Dataset | Class | Support | F1 |
|---|---|---:|---:|
| YawDD | non-yawning | 41 | 84.66 ± 2.39 |
| YawDD | yawning cue | 21 | 71.10 ± 5.83 |

## YawDD Main Comparison

| Method | Best Acc | Acc mean ± std | Best W-F1 | W-F1 mean ± std | Y-F1 mean ± std |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 0.403 | 0.403 ± 0.000 | 0.362 | 0.362 ± 0.000 | 0.479 ± 0.000 |
| Linear probe | 0.758 | 0.758 ± 0.000 | 0.759 | 0.759 ± 0.000 | 0.651 ± 0.000 |
| CLIP-Adapter | 0.661 | 0.661 ± 0.000 | 0.526 | 0.526 ± 0.000 | 0.000 ± 0.000 |
| CoOp | 0.806 | 0.616 ± 0.131 | 0.808 | 0.612 ± 0.143 | 0.604 ± 0.082 |
| MaPLe | 0.823 | 0.790 ± 0.032 | 0.816 | 0.785 ± 0.031 | 0.661 ± 0.043 |
| Ours, frozen CLIP | 0.839 | 0.800 ± 0.037 | 0.840 | 0.801 ± 0.039 | 0.711 ± 0.058 |
| Ours, visual backbone fine-tuned | 0.839 | 0.737 ± 0.092 | 0.831 | 0.691 ± 0.154 | 0.463 ± 0.402 |

We do not claim that visual-backbone fine-tuning is intrinsically inferior. Under the present low-data driver-disjoint protocol and matched validation-based selection, the frozen adaptation setting is more stable.

## AIDE Component And Calibration Diagnostics

Panel A:

| Configuration | Acc | Macro-F1 |
|---|---:|---:|
| Full framework | 82.50 | 73.57 |
| Adapter-only fixed prompts, no PCH | 72.38 | 60.38 |

Panel A is a reference comparison rather than a fully matched component ablation.

Panel B:

| Configuration | Acc | WF1 |
|---|---:|---:|
| Full framework | 82.50 | 81.40 |
| w/o residual adapter | 46.90 | 43.80 |
| Uniform prompt averaging | 81.50 | 80.00 |
| w/o affine scale | 77.10 | 76.20 |
| w/o affine bias | 82.50 | 80.00 |

## Temporal Aggregation

| Temporal aggregation | Acc | WF1 | Macro-F1 |
|---|---:|---:|---:|
| Mean pooling | 82.50 | 81.40 | 73.57 |
| Cross-frame gating | 73.58 | 70.28 | 57.08 |
| Temporal attention | 67.92 | 58.93 | 39.18 |

Mean pooling is the selected temporal aggregation strategy for the reported AIDE model.
