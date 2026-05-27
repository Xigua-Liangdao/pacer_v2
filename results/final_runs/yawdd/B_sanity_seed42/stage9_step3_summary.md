# Stage 9 Step 3 STOP: B-prime Sanity Check

No commit was made.

## Step 1 Archive

- Moved prior YawDD TAGA-era artifacts from `results/final_runs/yawdd/` to `results/final_runs/yawdd_with_TAGA_deprecated/`.
- Wrote `results/final_runs/yawdd_with_TAGA_deprecated/README.md` explaining that these runs used `PCH+TAGA` and are historical only.
- Archived TAGA-era YawDD table files to `paper_tables/archive_TAGA_era/`.
- Marked old YawDD sections in `paper_preparation.md` with `// DEPRECATED, see Stage 9 notes`.

## Step 2 B-prime Sanity Setting

Output directory: `results/final_runs/yawdd/B_sanity_seed42/`

Hard architecture constraints:

- `feature_layout=pooled`
- `temporal_head=none`
- `temporal_module=none`
- `use_test_ensemble=False`
- `adapter_mode=full`
- PCH switches on: `adapter_use_prompt_weight=on`, `adapter_use_class_temperature=on`, `adapter_use_class_bias=on`

Hyperparameters:

- `adapter_hidden_dim=1024`
- `adapter_dropout=0.3`
- `lr=1e-4`
- `weight_decay=1e-2`
- `loss_type=focal`
- `focal_gamma=2.0`
- `label_smoothing=0.01`
- `num_frames=10`
- `frame_sampling_mode=diff_guided`
- `use_class_weight=True`
- `epochs=40`, early stopped at epoch 13

## Parameter Breakdown

Log line:

```text
[PCH] params | head=15 | temporal=0 | adapter=3151360 | trainable=3151375 | adapter_mode=full | temporal_module=none
```

Conclusion: architecture check passed. There is no TAGA / transformer / temporal trainable module on this path.

## Metrics

| split | accuracy | weighted_f1 | confusion matrix |
| --- | ---: | ---: | --- |
| val | 0.918367 | 0.917416 | notdrowsy: 29/30 correct; drowsy: 16/19 correct |
| test | 0.758065 | 0.748670 | notdrowsy: 36/41 correct; drowsy: 11/21 correct |

Binary drowsy test metrics:

- precision: `0.687500`
- recall: `0.523810`
- f1: `0.594595`

## Train/Val Curve

| epoch | train_loss | val_acc | val_wf1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.158463 | 0.755102 | 0.757559 |
| 2 | 0.157430 | 0.836735 | 0.828850 |
| 3 | 0.152536 | 0.755102 | 0.718995 |
| 4 | 0.160920 | 0.836735 | 0.828850 |
| 5 | 0.156757 | 0.408163 | 0.259420 |
| 6 | 0.150350 | 0.857143 | 0.851815 |
| 7 | 0.155400 | 0.918367 | 0.917416 |
| 8 | 0.149019 | 0.612245 | 0.595521 |
| 9 | 0.148774 | 0.897959 | 0.894153 |
| 10 | 0.144784 | 0.857143 | 0.857764 |
| 11 | 0.141704 | 0.877551 | 0.878501 |
| 12 | 0.144016 | 0.857143 | 0.858610 |
| 13 | 0.137552 | 0.857143 | 0.858610 |

## Decision Gate

`test_wf1=0.748670`, so this is above the `>0.65` threshold. Proceed to Step 4 sweep after user confirmation (`go step 4`). The sweep can be somewhat narrowed around pooled PCH settings, but should still include sampling / hidden_dim / loss comparisons because test recall for `drowsy` is only `0.523810`.
