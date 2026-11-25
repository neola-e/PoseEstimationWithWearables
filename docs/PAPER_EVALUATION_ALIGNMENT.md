# Paper evaluation alignment: 2026-11-16

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Definitions, data, implementation/results have been checked item by item. **The executable DIP chain includes fine-tuning, rolling inference, IMU-only IK, and four metrics over all 24 configurations; strict author-equivalent settings remain unresolved.** Distinguish paper/public-code evidence from local conventions; do not infer settings from closeness to paper scores. Preserve the2026-11-11 audit/F03 baseline history.

[21-item CSV](results/dip_imu_only_ik_v1_20261116/paper_alignment.csv) · [IK implementation/results](IMU_ONLY_IK.md) · [Frozen settings](results/dip_imu_only_ik_v1_20261116/PROTOCOL.json)

## Verified conventions

| Item | Confirmed |
|---|---|
| Data/budget | Final first 8 training, last 2 testing; development-selected K=2; fixed checkpoint/19 recordings; no test model selection |
| Inputs/devices | Original25 Hz, acc/30, global orientations, missing-zero; all 24 one-to-three-location masks, exclude five-location global |
| Segment mapping | Public ji_mask18/19/1/2/15, not stale leaf list; training-side diagnostic, no GT-fitted calibration |
| Metrics | Global MPJRE, pelvis-translation-aligned24-joint MPJPE/6,890-vertex MPJVE; retain root rotation; MPJVE not velocity |
| Jitter | Predicted jerk, third difference×25³, within-recording T−3 weighting, divide100; no cross-recording differences or GT subtraction |
| Implemented IK | Available IMU orientation only, MSE+Adam, 10 updates/frame, shoulder/elbow optimization, no GT input, raw control retained |
| 24→12 groups | Ignore left/right; count hand/pocket/head: 3 single-location classes, 5 double, 4 triple; members in per_group.csv |

Definitions/groups: [IMUPoser §3–6](https://arxiv.org/html/2304.12518v1). Discrete jerk/mesh use previously validated tools; four-metric sources unchanged.

## Unconfirmed author details

| Item | Fixed local choice | Why unresolved |
|---|---|---|
| IK learning rate | 1e-3 from cited AvatarPoser paper | Not specified by IMUPoser; AvatarPoser public defaults/objective differ from paper/this task |
| Head/hip DOFs/parameterization | Sensed segment+direct parent, local rotation-vector increments, SO(3)-projected targets | No complete author IK to compare; shoulder/elbow textual support, others explicit reconstruction |
| IK state/output | Fresh Adam/frame, post-update10 output, no iterate selection/fallback | Boundary rules unpublished |
| Rolling stride/startup | F03 stride1, last output, short prefix | Text says125 history and1-sample overlap; complete public evaluation loop unavailable |
| Smoothing/causality | Original centered-five acc mean, entrywise orientation interpolation, NaN filling | Two-frame/80 ms acc lookahead; raw filling/interpolation causality unproven; IK itself adds no future access |
| Shape | Shared mean male, beta0 | Processed DIP saved default all-one betas, not individually fitted; author evaluation betas unpublished |
| Mesh switch | Public FK default, pose blendshape off | Actual author switch unpublished; sensitivity separately measured |
| Final reduction | Frame/T−3 within mask, equal24 masks | Twelve display groups do not fully specify recording/subject/mask weights |

Sources: [IMUPoser §4.4/5.3](https://arxiv.org/html/2304.12518v1#S4.SS4), [public a0e9d33](https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2), [AvatarPoser paper](https://static.siplab.org/papers/eccv2022-avatarposer.pdf)/[IK function](https://github.com/eth-siplab/AvatarPoser/blob/0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0/models/network.py). Public inverse_kinematics_R is rotation conversion, not this pose optimizer.

## Measured reduction sensitivity

Same final IK predictions: no retraining, inference, shape change or reselection. Primary rule frozen before results, not replaced by a closer auxiliary paper score.

| Reduction | MPJRE ° | MPJPE cm | MPJVE cm | Jitter, 100m/s³ |
|---|---: |---: |---: |---: |
| Equal24 masks, frame/T−3 within(primary) | 22.16620 | 9.42636 | 11.82258 | 1.22145 |
| Equal24 masks, equal recordings within(aux) | 22.04929 | 9.06523 | 11.41680 | 1.59500 |
| Equal12 display classes(aux) | 22.74294 | 9.71844 | 12.16981 | 1.19235 |
| Equal24 masks, equal two subjects(aux) | 22.20689 | 9.45620 | 11.86733 | 1.21784 |

Equal24 masks→equal12 classes alone changes MPJVE11.82258→12.16981 cm(~0.34722 cm). Recording-equal jitter also changes appreciably. [Group members/details](results/dip_imu_only_ik_v1_20261116/per_group.csv), [all reductions, both branches](results/dip_imu_only_ik_v1_20261116/aggregation_sensitivity.csv).

## Fixed-case shape/mesh sensitivity

Exploratory first/last masks×first/last full recordings, four pairs, both raw/IK; change one geometric setting at a time. No prediction/primary change. Diagnostic only, not replacement full 24-mask scores or proof of author settings.

| Changed setting | MPJPE change | MPJVE change | Other checks |
|---|---: |---: |---|
| Shared beta0→all 1 | −0.728 to−0.290 cm | −1.087 to−0.399 cm | Global angles unchanged; bone lengths change jerk |
| Enable pose blendshape | 0 | +0.032 to+0.105 cm | Joint angles/positions/jitter unchanged |

Changes=alternative minus primary; range covers four pairs×two branches. [Full exploratory results](results/dip_imu_only_ik_v1_20261116/mesh_convention_sensitivity.json). Shape effects exceed the overall~0.146 cm IK vertex gain in magnitude, but sampled cases are not overall differences or author-setting evidence.

## Paper numbers and remaining work

[Table2](https://arxiv.org/html/2304.12518v1#S6.SS3): DIP MPJRE23.9°, MPJPE9.7 cm, MPJVE12.1 cm, jitter1.9(100m/s³). Current primary22.16620°/9.42636 cm/11.82258 cm/1.22145 is close/lower, but does not prove strict reproduction or superiority; unresolved measurement choices can affect that comparison.

Historical AMASS differs: 25 masks(extra global), corpus/filter/retained-duration conventions, 65 completed epochs selecting59 versus paper24 masks/80. Do not subtract37.73 h retained frames from 63 h raw corpus as “missing hours.” DIP cannot erase initialization history.[Original audit](REPRODUCTION_AUDIT.md).

Next seek actual author IK/evaluation evidence; separately plan24-mask training if strict AMASS alignment is required. Existing raw/IK caches, 14 tests and independent audit support rechecks when evidence arrives. Consumer-device data, strict causal raw preprocessing and end-to-end latency are phase 2; CPU batch evaluation does not replace them. No test-driven learning-rate, joint or smoothing reselection occurred.
