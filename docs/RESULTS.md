# Reference results

These are the table values used in the paper draft. They are included to make row naming and metric aggregation unambiguous; reruns may differ slightly across CUDA/PyTorch environments.

## YawDD Table III

Frozen VLM / CLIP-adaptation baselines under the YawDD speaker-independent video-level protocol.

| Method | Seeds | Best Acc | Acc mean ± std | Best W-F1 | W-F1 mean ± std | Drowsy F1 mean ± std | Not-drowsy F1 mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 5 | 0.403 | 0.403 ± 0.000 | 0.362 | 0.362 ± 0.000 | 0.479 ± 0.000 | 0.302 ± 0.000 |
| Tip-Adapter | 5 | 0.694 | 0.694 ± 0.000 | 0.687 | 0.687 ± 0.000 | 0.513 ± 0.000 | 0.776 ± 0.000 |
| Linear Probe | 5 | 0.758 | 0.758 ± 0.000 | 0.759 | 0.759 ± 0.000 | 0.651 ± 0.000 | 0.815 ± 0.000 |
| Vanilla CLIP-Adapter | 5 | 0.661 | 0.661 ± 0.000 | 0.526 | 0.526 ± 0.000 | 0.000 ± 0.000 | 0.796 ± 0.000 |
| CoOp | 5 | 0.806 | 0.616 ± 0.131 | 0.808 | 0.612 ± 0.143 | 0.604 ± 0.082 | 0.616 ± 0.175 |
| MaPLe | 5 | 0.823 | 0.790 ± 0.032 | 0.816 | 0.785 ± 0.031 | 0.661 ± 0.043 | 0.848 ± 0.025 |
| Frozen CLIP + Adapter + PCH (Ours) | 5 | 0.839 | 0.800 ± 0.037 | 0.840 | 0.801 ± 0.039 | 0.711 ± 0.065 | 0.847 ± 0.027 |

## YawDD visual-backbone trainability

These runs fine-tune the CLIP visual backbone only. The CLIP text encoder remains frozen; Adapter + PCH remain trainable.

| Method | Seeds | Best Acc | Acc mean ± std | Best W-F1 | W-F1 mean ± std | Drowsy F1 mean ± std | Not-drowsy F1 mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen CLIP + Adapter + PCH | 5 | 0.839 | 0.800 ± 0.037 | 0.840 | 0.801 ± 0.039 | 0.711 ± 0.065 | 0.847 ± 0.027 |
| Partial CLIP visual backbone fine-tuning, last 2 ViT blocks | 3 | 0.935 | 0.871 ± 0.056 | 0.935 | 0.874 ± 0.054 | 0.832 ± 0.063 | 0.895 ± 0.049 |
| Full CLIP visual backbone fine-tuning | 3 | 0.839 | 0.737 ± 0.092 | 0.831 | 0.691 ± 0.154 | 0.463 ± 0.402 | 0.808 ± 0.073 |

## Naming notes

- `Vanilla CLIP-Adapter` means residual visual adapter only: no PCH prompt weighting, no class scale, no class bias, no focal/class-weighted loss.
- `Frozen CLIP + Adapter + PCH (Ours)` is the main frozen-backbone model.
- The backbone trainability table should say `CLIP visual backbone fine-tuning`, not `all CLIP parameters`; the text encoder remains frozen.
- ResNet-50 and MAR sanity checks are not part of the main CLIP-adaptation table.
