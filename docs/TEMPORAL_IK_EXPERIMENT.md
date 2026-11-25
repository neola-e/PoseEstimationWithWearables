# I02: independent Temporal IK experiment

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Completed: 2026-11-16, America/Chicago. Separate from I01 framewise IK; retain original model, training protocol, I01 algorithm, caches/results. **Temporal IK reduces jitter4.42% versus framewise IK and retains98.58% of its MPJVE gain; still11.36% above no-IK jitter, so mitigation is partial.**

## Locked test results

Same final two-epoch DIP checkpoint, all 19 s09/s10 recordings, 24 masks, 24,078 frames/mask. Shared mean male, pelvis translation alignment, 24 joints/6,890 vertices, unchanged four metrics/reduction.

| Metric ↓ | No IK | Framewise IK(I01) | Temporal IK(I02) |
|---|---: |---: |---: |
| MPJRE ° | 22.46870 | 22.16620 | 22.16886 |
| MPJPE cm | 9.54019 | 9.42636 | 9.42868 |
| MPJVE cm | 11.96889 | 11.82258 | 11.82466 |
| Jitter, 100m/s³ | 1.04836 | 1.22145 | 1.16742 |

Retained gain=`(MPJVE_noIK-MPJVE_Temporal)/(MPJVE_noIK-MPJVE_framewise)`, not a claim of identical three accuracy errors. Temporal costs0.00207 cm(~0.021 mm) MPJVE versus I01, but improves0.14423 cm versus no IK. All three errors remain below no IK in24/24 mask aggregates. Temporal jitter is below I01 in24/24, below no IK in0/24.

Predeclared compromise: three errors no worse than raw, retain≥50% I01 vertex gain, jitter<I01. Met. Stronger restoration to raw jitter or lower: unmet. These test scores do not reselect weights.

![Three branches, four metrics](results/temporal_ik_v1_20261116/temporal_four_metrics.png)

[Overall](results/temporal_ik_v1_20261116/comparison.csv) · [24 masks](results/temporal_ik_v1_20261116/per_mask.csv) · [19 recordings](results/temporal_ik_v1_20261116/by_sequence_id.csv) · [Categories](results/temporal_ik_v1_20261116/by_category.csv) · [Subjects](results/temporal_ik_v1_20261116/by_subject.csv) · [Joint jitter](results/temporal_ik_v1_20261116/joint_jitter.csv)

## Method: current plus past four frames

For output t, use network predictions and available IMU orientations over `[max(0,t-4),t]`. Jointly optimize the same I01 local rotational degrees of freedom across that window, zero increments around original network poses; emit only t. Fresh Adam per independent window, no optimizer state across windows/recordings.

```text
L = sum_valid_frames orientation_matrix_MSE
    + lambda * sum_valid_four_frame_stencils mean((D3 joint_position / 0.01m)^2)
```

Positions come from fixed mean-body FK/candidate local rotations, not observed GT. D3 is a discrete third difference in frame units; 0.01m is a predeclared scale. Evaluation jitter still uses25³/100 scaling. Temporal consistency is constrained during optimization, not by post-output low-pass filtering.

| Parameter | Frozen value |
|---|---|
| Budget | 10 Adam steps, lr 1e-3, betas0.9/0.999, eps1e-8 |
| Window | Five frames, current+past4, stride1, emit last |
| Weight | 0.001, selected on development only |
| Regularizer | Squared third difference of all 24 joint positions, scale0.01m |
| Variables/targets | Same I01 localSO(3) increments/available-orientation projection; FP64 optimize, FP32 output |
| Modifiable joints | Same mask-specific I01 list; other local6D entries bitwise retained |
| Startup/boundaries | Valid short prefixes; only complete valid four-frame stencils, no cross-recording support |
| Added future access | Zero; batching independent past windows only accelerates computation |

Five-frame history requires no future wait; original centered acceleration/filling/interpolation dependencies remain. This establishes no additional Temporal-IK lookahead, not strict causality of the entire raw-stream pipeline.

[temporal_ik.py](../tools/temporal_ik.py), [experiment_temporal_ik.py](../tools/experiment_temporal_ik.py). This is a new temporal-regularization experiment, not a method attributed retrospectively to original IMUPoser.

## Training-side selection, not test tuning

Development epoch2 SHA `4515e13ae95ee5ec49ff90d941c81149ec4c583ba293aadf21abe17de699ba16`, trained s01–s06. Calibrate on all 11 s07/s08 recordings: central96 consecutive frames plus4 historical pose outputs; one fixed representative of each 12 left/right-mirror display classes, 132 pairs, 12,672 metric frames.

Each network output retains125-frame rolling context; interval predictions checked against existing rolling implementation, not context-free chunks. Weight0 exactly reproduces framewise I01 in all 132 actual calibration pairs, max difference0.

Freeze candidates `[0.001,0.01,0.1,1]`, budget, rule: first require three errors≤raw, retain≥50% I01 MPJVE gain, jitter<I01; choose lowest jitter among eligible. If none, rank by recorded normalized spatial violation then jitter and report failure. Only 0.001 qualifies.

| Weight | Calibration MPJVE cm | Calibration jitter | Eligible |
|---|---: |---: |---|
| 0.001 | 10.47713 | 1.34948 | True |
| 0.01 | 10.51317 | 1.43462 | False |
| 0.1 | 10.56604 | 1.44215 | False |
| 1.0 | 10.58930 | 1.42812 | False |

Calibration raw/framewise jitter1.26048/1.37946; MPJVE10.58667/10.47090 cm. Larger regularization does not monotonically lower final jitter with fixed ten steps; larger weight does not guarantee smoother emitted motion. Training-side calibration precedes frozen [selection](results/temporal_ik_v1_20261116/calibration_selection.json)/[test protocol](results/temporal_ik_v1_20261116/PROTOCOL.json), then full 24-mask testing.

The test set already informed historical error analysis; not a fresh blind test. Temporal test scores did not select this weight. Short panels/two development participants do not establish optimal parameters or new-population generalization.

## Affected joints

Full 24-mask/all-recording weighted joint jitter, 100m/s³. Feet, ankles, head were major receivers in J01; partial recovery remains above raw.

| Joint | No IK | Framewise | Temporal |
|---|---: |---: |---: |
| R Foot | 1.51390 | 1.92978 | 1.79981 |
| L Foot | 1.52938 | 1.92641 | 1.80407 |
| R Ankle | 1.32719 | 1.70921 | 1.59154 |
| L Ankle | 1.37972 | 1.75033 | 1.63614 |
| Head | 0.79382 | 1.00696 | 0.93905 |

[J01](IK_JITTER_DIAGNOSIS.md) contains source attribution, spectra and controlled pelvis counterfactuals. No per-test-mask IK disabling, joint restoration or model reselection improves these aggregates.

## Validation/provenance

- Six temporal numerical/boundary/missing-sensor/future-perturbation tests, two interval/selection tests pass; plus three J01 signal/accounting checks.
- Fixed-skeleton versus original mean-body FK max position difference2.02e-7m.
- Independently verify922 original artifacts, 456 pairs, data/model/source/selection hashes, 11,437,050 unoptimized joint-frames.
- No GT in IK inputs; raw/I01 metric rows bitwise preserved; independent calibration selection passes.
- Overall reduction max difference1.78e-15.
- First/last masks×first/last recordings, frames0/1/3/4/124/125/last: independent past-only window reruns max difference0; startup/end/recording boundaries add no future frames.
- Full-recording four-metric FKchunk 11 versus32 max difference1.237e-12, below1e-4.

[Independent audit](results/temporal_ik_v1_20261116/independent_validation.json) · [Calibration protocol](results/temporal_ik_v1_20261116/calibration_PROTOCOL.json) · [Calibration metrics](results/temporal_ik_v1_20261116/calibration_metrics.json) · [Runtime](results/temporal_ik_v1_20261116/RUNTIME.json) · [Manifest](results/temporal_ik_v1_20261116/MANIFEST.json)

Full outputs:`outputs/local_dip/temporal_ik_v1_20261116/calibration/` and `test/`. Two CPU test shards about 7min each. No retraining; new network inference only for development calibration, test reuses original final-network caches. Old I01 files, training protocol, checkpoints, processed data unchanged; original/additional manifests separate.

Within this fixed pipeline, retain almost all spatial gain while mitigating some extra jitter; not restored to raw. Changing budget, regularizer or weight requires a new training-side-selected experiment, not reverse-engineering settings from current test outcomes.
