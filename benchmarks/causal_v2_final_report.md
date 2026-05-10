# PACER Causal Extension — Final Report

## Headline Numbers

- Controlled Block 1 baseline (split=7, training=20): acc=0.769231, wF1=0.772028.
- D0 baseline over five training seeds: acc=0.646154 +- 0.077773 (n=5), wF1=0.587727 +- 0.132354 (n=5).
- Best mean YawDD final config D3_cda_only: acc=0.676923 +- 0.112645 (n=5), wF1=0.643832 +- 0.153810 (n=5).
- Full YawDD intervention D7_full: acc=0.651282 +- 0.138793 (n=5), wF1=0.629373 +- 0.162501 (n=5).
- AIDE full intervention: acc=0.788451 +- 0.008106 (n=3), wF1=0.784918 +- 0.007473 (n=3).
- Kendall tau: -0.066667 with exact two-sided p-value 1.000000.
- Spearman rho: -0.085714 with exact two-sided p-value 0.919444.
- Positive correlation only appears under subset AIDE groupings (vehicle), so the domain-general mechanism claim depends on how AIDE context groups are defined.

## Block 1 Confirmation

Controlled baseline is reproducible at split seed 7 and training seed 20 with bit-for-bit recovery of the historical 76.92 / 77.20 target.

## Block 2 — CCL Sweep


Estimated runtime before execution: 0.07 hours.

## Accuracy

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.769231 | 0.564103 | 0.615385 | 0.666667 | 0.615385 | 0.646154 | 0.077773 |
| 0.01 | 0.717949 | 0.564103 | 0.615385 | 0.538462 | 0.615385 | 0.610257 | 0.068802 |
| 0.05 | 0.615385 | 0.615385 | 0.615385 | 0.641026 | 0.564103 | 0.610257 | 0.028088 |
| 0.1 | 0.692308 | 0.769231 | 0.564103 | 0.589744 | 0.615385 | 0.646154 | 0.083874 |
| 0.2 | 0.435897 | 0.615385 | 0.487179 | 0.487179 | 0.641026 | 0.533333 | 0.089561 |
| 0.3 | 0.435897 | 0.666667 | 0.615385 | 0.538462 | 0.589744 | 0.569231 | 0.087706 |
| 0.5 | 0.410256 | 0.384615 | 0.615385 | 0.615385 | 0.487179 | 0.502564 | 0.109689 |
| 1.0 | 0.564103 | 0.384615 | 0.641026 | 0.615385 | 0.615385 | 0.564103 | 0.104155 |
| 2.0 | 0.615385 | 0.384615 | 0.538462 | 0.615385 | 0.615385 | 0.553846 | 0.100296 |

## Weighted F1

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.772028 | 0.557781 | 0.468864 | 0.671096 | 0.468864 | 0.587727 | 0.132354 |
| 0.01 | 0.720552 | 0.557781 | 0.468864 | 0.520710 | 0.468864 | 0.547354 | 0.103848 |
| 0.05 | 0.618934 | 0.602096 | 0.468864 | 0.557265 | 0.566496 | 0.562731 | 0.058231 |
| 0.1 | 0.693927 | 0.772297 | 0.541074 | 0.579976 | 0.468864 | 0.611228 | 0.121396 |
| 0.2 | 0.392977 | 0.609807 | 0.483806 | 0.493284 | 0.644440 | 0.524863 | 0.101969 |
| 0.3 | 0.347253 | 0.666667 | 0.468864 | 0.516865 | 0.594628 | 0.518855 | 0.122004 |
| 0.5 | 0.355208 | 0.213675 | 0.468864 | 0.468864 | 0.487179 | 0.398758 | 0.115974 |
| 1.0 | 0.569386 | 0.213675 | 0.524504 | 0.468864 | 0.468864 | 0.449059 | 0.138167 |
| 2.0 | 0.468864 | 0.213675 | 0.527473 | 0.468864 | 0.468864 | 0.429548 | 0.123316 |

## Best Mean Accuracy

Best CCL weight by mean accuracy: 0.1.
Mean accuracy: 0.646154 +- 0.083874.
Mean weighted F1: 0.611228 +- 0.121396.

## Delta vs Baseline

| weight | delta_acc_mean | delta_acc_std | delta_wf1_mean | delta_wf1_std | within_noise |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | yes |
| 0.01 | -0.035897 | 0.056177 | -0.040372 | 0.065414 | yes |
| 0.05 | -0.035897 | 0.076063 | -0.024996 | 0.105791 | yes |
| 0.1 | 0.000000 | 0.118892 | 0.023501 | 0.125372 | yes |
| 0.2 | -0.112821 | 0.157646 | -0.062864 | 0.217540 | no |
| 0.3 | -0.076923 | 0.165181 | -0.068871 | 0.228132 | yes |
| 0.5 | -0.143590 | 0.138794 | -0.188969 | 0.196746 | no |
| 1.0 | -0.082051 | 0.104784 | -0.138668 | 0.163800 | no |
| 2.0 | -0.092308 | 0.073871 | -0.158179 | 0.179964 | no |

## Block 3 — CFA and CDA Sweeps


Estimated runtime before execution: 0.08 hours.

## CFA Accuracy

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.769231 | 0.564103 | 0.615385 | 0.538462 | 0.615385 | 0.620513 | 0.089560 |
| 0.1 | 0.692308 | 0.538462 | 0.512821 | 0.564103 | 0.615385 | 0.584616 | 0.071151 |
| 0.5 | 0.589744 | 0.743590 | 0.512821 | 0.461538 | 0.615385 | 0.584616 | 0.107875 |
| 1.0 | 0.564103 | 0.564103 | 0.615385 | 0.615385 | 0.615385 | 0.594872 | 0.028088 |
| 2.0 | 0.589744 | 0.410256 | 0.615385 | 0.564103 | 0.615385 | 0.558975 | 0.085812 |

## CFA Weighted F1

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.772028 | 0.557781 | 0.468864 | 0.520710 | 0.468864 | 0.557649 | 0.125587 |
| 0.1 | 0.695971 | 0.535425 | 0.495988 | 0.541074 | 0.468864 | 0.547464 | 0.088130 |
| 0.5 | 0.587045 | 0.746642 | 0.518726 | 0.466508 | 0.468864 | 0.557557 | 0.116498 |
| 1.0 | 0.564103 | 0.549042 | 0.468864 | 0.609807 | 0.468864 | 0.532136 | 0.061942 |
| 2.0 | 0.579976 | 0.266938 | 0.468864 | 0.564103 | 0.468864 | 0.469749 | 0.124686 |

## CDA Accuracy

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.666667 | 0.769231 | 0.435897 | 0.564103 | 0.615385 | 0.610257 | 0.123504 |
| 0.2 | 0.615385 | 0.564103 | 0.615385 | 0.641026 | 0.615385 | 0.610257 | 0.028088 |
| 0.3 | 0.717949 | 0.589744 | 0.615385 | 0.717949 | 0.615385 | 0.651282 | 0.061752 |
| 0.4 | 0.717949 | 0.743590 | 0.615385 | 0.794872 | 0.512821 | 0.676923 | 0.112645 |
| 0.5 | 0.717949 | 0.487179 | 0.615385 | 0.717949 | 0.615385 | 0.630769 | 0.095252 |

## CDA Weighted F1

| weight | train_20 | train_21 | train_22 | train_23 | train_24 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.666667 | 0.772297 | 0.392977 | 0.549042 | 0.468864 | 0.569969 | 0.151883 |
| 0.2 | 0.620496 | 0.557781 | 0.468864 | 0.557265 | 0.468864 | 0.534654 | 0.065329 |
| 0.3 | 0.720552 | 0.579976 | 0.468864 | 0.720552 | 0.468864 | 0.591762 | 0.126016 |
| 0.4 | 0.721697 | 0.746642 | 0.468864 | 0.794872 | 0.487083 | 0.643832 | 0.153810 |
| 0.5 | 0.717949 | 0.492057 | 0.468864 | 0.720552 | 0.468864 | 0.573657 | 0.133248 |

## Best Settings

Best CFA weight by mean accuracy: 0.05.
Best CDA probability by mean accuracy: 0.4.

## Block 4 — Final Configurations


Estimated runtime before execution: 0.05 hours.

| config | mean_acc | std_acc | mean_wF1 | std_wF1 | delta_acc_vs_D0 | delta_wF1_vs_D0 | n_seeds | significance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_baseline | 0.646154 | 0.077773 | 0.587727 | 0.132354 | 0.000000 | 0.000000 | 5 | within_noise |
| D1_ccl_only | 0.646154 | 0.083874 | 0.611228 | 0.121396 | 0.000000 | 0.023501 | 5 | within_noise |
| D2_cfa_only | 0.620513 | 0.089560 | 0.557649 | 0.125587 | -0.025641 | -0.030077 | 5 | within_noise |
| D3_cda_only | 0.676923 | 0.112645 | 0.643832 | 0.153810 | 0.030769 | 0.056105 | 5 | within_noise |
| D4_cfa_cda | 0.646154 | 0.110881 | 0.612232 | 0.144006 | 0.000000 | 0.024505 | 5 | within_noise |
| D5_ccl_cfa | 0.651282 | 0.088080 | 0.616553 | 0.126411 | 0.005128 | 0.028826 | 5 | within_noise |
| D6_ccl_cda | 0.610257 | 0.123503 | 0.559953 | 0.149821 | -0.035897 | -0.027774 | 5 | within_noise |
| D7_full | 0.651282 | 0.138793 | 0.629373 | 0.162501 | 0.005128 | 0.041646 | 5 | within_noise |

## Best Single Seed Confusion Matrix

Best config: D3_cda_only.
Best single-seed run: D3_cda_only_train23.

```json
{
  "notdrowsy": {
    "notdrowsy": 20,
    "drowsy": 4
  },
  "drowsy": {
    "notdrowsy": 4,
    "drowsy": 11
  }
}
```

## Recommendation

Recommended configuration: D3_cda_only_train23.

```bash
/data1/yanjing/talk2bev/aide_clip/src/clip_yawdd_emotion_train.py --all_face_image /data1/yanjing/talk2bev/fatigue-drive-yawning-detection/extracted_face_multi4 --label_mode binary --eval_mode fixed --cv_mode split --clip_mode offline_only --prompt_set yawdd_facial_cues --num_frames 10 --frame_sampling_mode uniform --feature_layout sequence --temporal_head transformer --temporal_num_heads 4 --temporal_num_layers 1 --temporal_pool_mode hybrid --adapter_use_prompt_weight on --adapter_use_class_temperature on --adapter_use_class_bias on --use_class_weight --disable_test_ensemble --epochs 40 --lr 0.00015 --weight_decay 0.01 --label_smoothing 0.1 --loss_type focal --seed 7 --training_seed 23 --ccl_weight 0.0 --cfa_weight 0.0 --cda_prob 0.4 --output OUTPUT.json --log_file OUTPUT.log --checkpoint_output OUTPUT.ckpt.pt --use_counterfactual_aug
```

## Block 5 — Cross-dataset Correlation


Estimated runtime before execution: 2.60 hours.

Using YawDD best settings: ccl_weight*=0.1, cfa_weight*=0.05, cda_prob*=0.4.
AIDE fixed split seed: 42; training seeds: [20, 21, 22].

## Primary Effect Table (AIDE scene+vehicle groups)

| intervention | yawdd_mean_acc | yawdd_delta | aide_mean_acc | aide_delta |
| --- | --- | --- | --- | --- |
| I_no_ccl | 0.646154 | 0.005128 | 0.782161 | 0.006289 |
| I_no_cfa | 0.610257 | 0.041026 | 0.784448 | 0.004002 |
| I_no_cda | 0.651282 | 0.000000 | 0.788451 | 0.000000 |
| I_ccl_half | 0.646154 | 0.005128 | 0.780446 | 0.008005 |
| I_cfa_half | 0.625641 | 0.025641 | 0.795883 | -0.007433 |
| I_cda_half | 0.605128 | 0.046154 | 0.788451 | 0.000000 |

## Primary Correlations (AIDE scene+vehicle groups)

Kendall tau: -0.066667 with exact two-sided p-value 1.000000.
Spearman rho: -0.085714 with exact two-sided p-value 0.919444.

## Group-Source Robustness

| aide_group_source | aide_full_mean_acc | kendall_tau | tau_p | spearman_rho | rho_p |
| --- | --- | --- | --- | --- | --- |
| scene+vehicle | 0.788451 | -0.066667 | 1.000000 | -0.085714 | 0.919444 |
| scene | 0.801601 | -0.200000 | 0.719444 | -0.085714 | 0.919444 |
| vehicle | 0.786735 | 0.066667 | 1.000000 | 0.142857 | 0.802778 |

## Interpretation

Positive correlation only appears under subset AIDE groupings (vehicle), so the domain-general mechanism claim depends on how AIDE context groups are defined.

Caveat: n=6 interventions is still small, so the exact p-values should be read as descriptive evidence rather than a final causal claim.

## Recommended CLI

```bash
/data1/yanjing/talk2bev/aide_clip/src/clip_yawdd_emotion_train.py --all_face_image /data1/yanjing/talk2bev/fatigue-drive-yawning-detection/extracted_face_multi4 --label_mode binary --eval_mode fixed --cv_mode split --clip_mode offline_only --prompt_set yawdd_facial_cues --num_frames 10 --frame_sampling_mode uniform --feature_layout sequence --temporal_head transformer --temporal_num_heads 4 --temporal_num_layers 1 --temporal_pool_mode hybrid --adapter_use_prompt_weight on --adapter_use_class_temperature on --adapter_use_class_bias on --use_class_weight --disable_test_ensemble --epochs 40 --lr 0.00015 --weight_decay 0.01 --label_smoothing 0.1 --loss_type focal --seed 7 --training_seed 23 --ccl_weight 0.0 --cfa_weight 0.0 --cda_prob 0.4 --output OUTPUT.json --log_file OUTPUT.log --checkpoint_output OUTPUT.ckpt.pt --use_counterfactual_aug
```
