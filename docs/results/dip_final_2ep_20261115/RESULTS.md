# DIP fine-tuning: four metrics before and after

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](../../RECORD_LANGUAGE.md).

Test: s09/s10,19 recordings, 24,078 frames, all 24 masks. Within masks, frame-weight spatial errors and T−3-weight jitter; then equally average masks.

| Metric | Before | After two epochs | Relative improvement | Improved masks |
|---|---: |---: |---: |---: |
| MPJRE (deg) | 30.26337 | 22.46870 | +25.76% | 24/24 |
| MPJPE (cm) | 11.65586 | 9.54019 | +18.15% | 24/24 |
| MPJVE (cm) | 14.59548 | 11.96889 | +18.00% | 24/24 |
| Jitter (100 m/s³) | 1.61320 | 1.04836 | +35.01% | 24/24 |

Relative improvement=(before-after)/before×100%; positive means lower. MPJVE is mesh-vertex position error. Jitter is predicted-joint third difference, in100m/s³.

Inference device: cpu. Rolling125, stride1, last valid output, prefix startup; retain filtering, no IK.

Independent audit passes: 456 paired caches/all GT, 1,155,744 predicted frames; independent reduction/eight fixed FK-chunk rechecks pass.

Descriptive locked-model test comparison; two subjects do not establish significance. Historical25-mask initialization, window boundaries, future filtering and no IK remain paper-protocol differences.

![Four metrics](four_metrics.png)
