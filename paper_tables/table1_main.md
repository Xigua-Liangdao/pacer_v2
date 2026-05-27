# Table 1: Main Results

## Table 1a: AIDE Driver Emotion Recognition

| dataset | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | Anxiety F1 Mean±Std | Peace F1 Mean±Std | Weariness F1 Mean±Std | Happiness F1 Mean±Std | Anger F1 Mean±Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIDE | 0.825 | 0.795 ± 0.015 | 0.814 | 0.784 ± 0.013 | 0.643 ± 0.016 | 0.858 ± 0.011 | 0.791 ± 0.025 | 0.737 ± 0.044 | 0.502 ± 0.051 |

## Table 1b: YawDD Binary Drowsiness Recognition (B-prime pooled PCH)

| dataset | acc Best | acc Mean±Std | wf1 Best | wf1 Mean±Std | drowsy F1 Mean±Std | notdrowsy F1 Mean±Std |
| --- | --- | --- | --- | --- | --- | --- |
| YawDD | 0.839 | 0.800 ± 0.037 | 0.840 | 0.801 ± 0.039 | 0.711 ± 0.065 | 0.847 ± 0.027 |

Note: AIDE reports best plus mean±std over 9 seeds. YawDD uses the Stage 9 B-prime pooled architecture and reports best plus mean±std over 5 seeds (42, 123, 2024, 7, 31). The deprecated TAGA-era YawDD numbers are archived under `paper_tables/archive_TAGA_era/` and are not used here.
