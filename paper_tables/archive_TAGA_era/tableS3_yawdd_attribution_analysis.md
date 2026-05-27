| variant_name | num_frames | loss | lr | label_smoothing | class_weight | sampling | test_wf1 | Δwf1 vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W0 baseline | 5 | ce | 1.5e-04 | 0.1 | off | middle_late | 0.526464 | — |
| W1 + num_frames=10 | 10 | ce | 1.5e-04 | 0.1 | off | middle_late | 0.526464 | +0.000000 |
| W2 + class weight | 10 | ce | 1.5e-04 | 0.1 | on | middle_late | 0.904952 | +0.378488 |
| W3 + focal gamma=1 | 10 | focal(gamma=1.0) | 1.5e-04 | 0.1 | on | middle_late | 0.904952 | +0.000000 |
| W4 + gamma=2/lr/ls | 10 | focal(gamma=2.0) | 1.0e-04 | 0.01 | on | middle_late | 0.904952 | +0.000000 |
| W5 + diff_guided sampling | 10 | focal(gamma=2.0) | 1.0e-04 | 0.01 | on | diff_guided | 0.919795 | +0.014843 |

Note: The dominant gain comes from enabling class-weighted training, which prevents majority-class collapse on YawDD binary.
