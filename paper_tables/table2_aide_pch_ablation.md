| ablation | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Δwf1 vs A0 Mean |
| --- | --- | --- | --- | --- | --- |
| A0 Ours full | 0.825 | 0.795 ± 0.015 | 0.814 | 0.784 ± 0.013 | — |
| A1 w/o prompt weight | 0.804 | 0.803 ± 0.003 | 0.795 | 0.790 ± 0.005 | +0.006 |
| A2 w/o class scale γ | 0.810 | 0.795 ± 0.013 | 0.790 | 0.774 ± 0.015 | -0.010 |
| A3 w/o class bias b | 0.815 | 0.805 ± 0.009 | 0.807 | 0.792 ± 0.013 | +0.008 |
| A4 w/o adapter MLP | 0.592 | 0.592 ± 0.000 | 0.441 | 0.441 ± 0.000 | -0.343 |

Note: A0 averaged over 9 seeds, A1-A4 over 3 seeds (42, 123, 2024). A4 std=0 due to deterministic zero-shot-like behavior under identity adapter; only PCH parameters are trainable.
