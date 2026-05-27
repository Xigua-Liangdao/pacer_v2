# Stage 9 Step 4 B-prime Sweep Summary

| config | hidden | dropout | lr | wd | loss | ls | frames | sampling | cw | val_best_wf1 | test_wf1 | train_time | killed |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| P2 | 512 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.897417 | 0.821486 | 10.6 | False |
| P5 | 1024 | 0.3 | 5e-5 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.776486 | 0.795515 | 10.2 | False |
| P4 | 2048 | 0.5 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.897417 | 0.785958 | 10.2 | False |
| P11 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 16 | diff_guided | on | 0.917416 | 0.785958 | 474.0 | False |
| P12 | 2048 | 0.5 | 5e-5 | 5e-2 | focal(gamma=2.0) | 0.01 | 16 | diff_guided | on | 0.857764 | 0.771237 | 474.0 | False |
| P8 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.0 | 10 | diff_guided | on | 0.874175 | 0.767560 | 10.3 | False |
| P3 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917416 | 0.748670 | 10.4 | False |
| P6 | 1024 | 0.3 | 2e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917416 | 0.748670 | 10.2 | False |
| P9 | 1024 | 0.3 | 1e-4 | 1e-2 | focal(gamma=3.0) | 0.01 | 10 | diff_guided | on | 0.917416 | 0.748670 | 10.3 | False |
| P10 | 1024 | 0.3 | 1e-4 | 5e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.917416 | 0.748670 | 10.3 | False |
| P1 | 256 | 0.2 | 1e-4 | 1e-2 | focal(gamma=2.0) | 0.01 | 10 | diff_guided | on | 0.858610 | 0.724114 | 10.6 | False |
| P7 | 1024 | 0.3 | 1e-4 | 1e-2 | ce | 0.01 | 10 | diff_guided | on | 0.917416 | 0.720099 | 10.3 | False |

## Top-3

1. P2: test_wf1=0.821486, val_best_wf1=0.897417, hidden=512, lr=1e-4, loss=focal(gamma=2.0), frames=10
2. P5: test_wf1=0.795515, val_best_wf1=0.776486, hidden=1024, lr=5e-5, loss=focal(gamma=2.0), frames=10
3. P4: test_wf1=0.785958, val_best_wf1=0.897417, hidden=2048, lr=1e-4, loss=focal(gamma=2.0), frames=10
