| ablation | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Δwf1 vs YA0 Mean |
| --- | --- | --- | --- | --- | --- |
| YA0 Ours full | 0.919 | 0.903 ± 0.023 | 0.921 | 0.904 ± 0.023 | — |
| YA1 w/o prompt weight | 0.919 | 0.903 ± 0.023 | 0.921 | 0.904 ± 0.023 | +0.000 |
| YA2 w/o class scale γ | 0.919 | 0.903 ± 0.023 | 0.921 | 0.904 ± 0.023 | +0.000 |
| YA3 w/o class bias b | 0.919 | 0.903 ± 0.023 | 0.921 | 0.904 ± 0.023 | +0.000 |
| YA4 w/o adapter MLP | 0.935 | 0.903 ± 0.035 | 0.936 | 0.904 ± 0.034 | +0.000 |
| YA5 w/o class weight | 0.661 | 0.661 ± 0.000 | 0.526 | 0.526 ± 0.000 | -0.378 |

Note: YA0 uses the canonical YawDD seeds 42/123/2024 from the locked S6 setting. YA1-YA5 use the same three seeds and modify one component at a time.
