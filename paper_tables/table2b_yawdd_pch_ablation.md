# Table 2b: YawDD PCH Component Ablation (B-prime pooled PCH)

| ablation | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Δwf1 vs YA0 Mean |
| --- | --- | --- | --- | --- | --- |
| YA0 Ours full | 0.823 | 0.790 ± 0.043 | 0.821 | 0.789 ± 0.045 | — |
| YA1 w/o prompt weight | 0.823 | 0.790 ± 0.043 | 0.821 | 0.789 ± 0.045 | +0.000 |
| YA2 w/o class scale gamma | 0.823 | 0.790 ± 0.043 | 0.821 | 0.789 ± 0.045 | +0.000 |
| YA3 w/o class bias b | 0.823 | 0.790 ± 0.043 | 0.821 | 0.789 ± 0.045 | +0.000 |
| YA4 w/o adapter MLP | 0.452 | 0.419 ± 0.028 | 0.438 | 0.387 ± 0.044 | -0.402 |
| YA5 w/o class weight | 0.661 | 0.661 ± 0.000 | 0.526 | 0.526 ± 0.000 | -0.263 |

Note: All rows use seeds 42, 123, and 2024. YA1-YA3 match YA0 at the reported precision, indicating that PCH calibration knobs did not change final discrete predictions on YawDD B-prime; YA4 verifies the adapter MLP effect and YA5 verifies class-weight sensitivity.
