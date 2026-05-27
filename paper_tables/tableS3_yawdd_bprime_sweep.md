# Table S3: YawDD B-prime Hyperparameter Sensitivity

| config | hidden | dropout | lr | wd | loss | label smoothing | frames | sampling | class weight | val_best_wf1 | test_wf1 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| P2 | 512 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.897 | 0.821 |
| P5 | 1024 | 0.3 | 5e-5 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.776 | 0.796 |
| P4 | 2048 | 0.5 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.897 | 0.786 |
| P11 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 16 | diff_guided | on | 0.917 | 0.786 |
| P12 | 2048 | 0.5 | 5e-5 | 5e-2 | focal(gamma=2.0) | 0.01 | 16 | diff_guided | on | 0.858 | 0.771 |
| P8 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.0 | 10 | diff_guided | on | 0.874 | 0.768 |
| P3 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917 | 0.749 |
| P6 | 1024 | 0.3 | 2e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917 | 0.749 |
| P9 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=3.0) | 0.01 | 10 | diff_guided | on | 0.917 | 0.749 |
| P10 | 1024 | 0.3 | 1e-4 | 5e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917 | 0.749 |
| P1 | 256 | 0.2 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.859 | 0.724 |
| P7 | 1024 | 0.3 | 1e-4 | 1e-2 | ce | 0.01 | 10 | diff_guided | on | 0.917 | 0.720 |

Note: All variants use the Stage 9 B-prime architecture: pooled CLIP features, no temporal head, no TAGA, no test-time ensemble, adapter MLP plus PCH.
