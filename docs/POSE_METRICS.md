# IMUPoser evaluation metrics and numerical validation

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Completed: 2026-11-13. Four metrics, known-pose checks and cached-prediction rechecks are complete. Entry: [eval_pose_metrics.py](../tools/eval_pose_metrics.py); implementation: [paper_metrics.py](../tools/paper_metrics.py). This task measures existing predictions only, without training or network inference.

Definitions follow [IMUPoser §5.3 and Table2](https://arxiv.org/html/2304.12518v1). Discrete jerk, translation handling and default mesh switch follow the cited [TransPose FullMotionEvaluator](https://github.com/Xinyu-Yi/TransPose/blob/37be773ceceda49160717311a4f74a695817b8d3/articulate/evaluator.py). These choices are auditable, not proof of recovering every unpublished IMUPoser evaluation setting.

## Definitions

SMPL positions/input translations use meters; orientations use the global frame. Compute errors per frame/point, then average. Use all 24 SMPL joints including pelvis, and all 6,890 corresponding vertices.

| Field | Meaning/unit | Alignment and reduction |
|---|---|---|
| `mpjre_deg` | MPJRE: SO(3) geodesic error of global joint rotations, degrees | FK from local rotations; retain root rotation; average24 joints per frame |
| `mpjpe_cm` | MPJPE: joint Euclidean position error, cm | Subtract each pose's own pelvis; 24 joints including zero-error root |
| `mpjve_cm` | MPJVE: mesh-vertex Euclidean position error, cm | Same pelvis translation alignment; 6,890 vertices; neither velocity nor joint-position error |
| `jitter_m_s3` | Mean jerk of predicted joint trajectories, m/s³ | Global joints before spatial alignment, including supplied translation; not prediction minus GT |
| `jitter_100_m_s3` | Same jitter divided by100 | Matches paper table units(10^2 m/s^3); do not multiply by100 |
| `local_rotation_deg` | Local joint-rotation error, degrees | Supplementary diagnostic, not global MPJRE |
| `reference_jitter_m_s3` / `reference_jitter_100_m_s3` | GT trajectory's own jerk | Separate reference, not subtracted from prediction jitter |

Spatial errors align pelvis translation only: no heading, root rotation, scale, Procrustes or mesh-centroid alignment. Meter distances ×100 give centimeters. atan2 geodesics equal trace/acos on valid SO(3), with improved zero/small-angle stability. Clearly nonorthogonal, reflected or nonfinite matrices are rejected rather than silently repaired.

## Jitter timing

For uniformly sampled positions p_t:

```text
jerk[t] = (p[t+3] - 3*p[t+2] + 3*p[t+1] - p[t]) * fps^3
jitter = mean over valid time samples and 24 joints of ||jerk[t]||_2
```

A complete T-frame recording supplies `max(T-3,0)` jerk samples. Under four frames, jitter is JSON `null`, count0, not artificial zero jitter; spatial errors still contribute.

Identical prediction/GT does not imply zero jitter: matching accelerated-changing motion can have zero spatial/angular error and nonzero equal jerk. Conversely, a wrong static pose can have zero jitter; smoothness alone does not establish accuracy.

Mesh computation chunks do not break temporal trajectories. Cache `sequence_ids`, `start_frames`, `offsets` must continuously cover each recording from 0 before reconstruction and differentiation. Gaps/overlaps/duplicate windows are rejected. Never differentiate across recordings. With offsets alone, intervals are independent sequences; adjacency is not guessed. Single-sequence `lengths` trims padding; an unbounded padded batch cannot be treated as one trajectory.

Discontinuities between network windows within one continuous recording must appear in jitter. No extra smoothing or computational-boundary exclusions hide them.

## Shape and mesh

- Default: supplied SMPL mean shape, shared by prediction/GT. Current validation uses licensed male SMPL; asset SHA-256 is recorded.
- `--betas` accepts10 fixed shared coefficients, not per-frame shape.
- Pose blend shapes default off, matching inherited SMPL and inspected TransPose defaults. `--pose-blendshape` explicitly enables and records them; tests confirm mesh error can change while joint-position error does not.
- IMUPoser text does not specify this switch/all shape details. This is an explicit configurable convention, not confirmed author behavior. Paper comparison must hold it fixed and report sensitivity when needed.
- Optional NPZ `predicted_translation_m`/`target_translation_m` each use `[T,3]`, zero if omitted. Spatial metrics remove pelvis translation; jitter retains trajectory motion. Current IMUPoser caches have no predicted trajectory and use zero translation.

## Aggregation

Primary `frame_weighted` uses valid frames for angular/joint/vertex errors and valid T−3 stencils for jitter; these denominators differ. Auxiliary `sequence_equal` gives each defined sequence equal weight; sequences without jerk samples do not enter jitter means.

Keep per-recording frames, jerk counts, metrics and per-joint statistics. This tool measures one configuration at a time; one three-location mask is not the mean over paper device combinations. The paper does not fully specify cross-configuration weights; freeze the full benchmark aggregation separately.

## Usage

NPZ prediction/reference must be paired local poses: `[T,144]`, `[T,24,6]` or `[T,24,3,3]`. 6D follows repository column conventions; matrices must be valid SO(3). Scalar `fps`, if supplied, must match `--fps`; interpret all samples at that fixed interval.

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tests/test_paper_metrics.py

conda run --no-capture-output -n imuposer python tools/eval_pose_metrics.py \
  --input outputs/local_dip/dual_wrist_dev_epoch59_20261110/evaluation/predictions.npz \
  --pose-key pose_lw_rw_h \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --fps 25 --fk-chunk 32 \
  --output outputs/local_dip/metric_validation_20261113_recheck/metrics.json
```

Existing outputs cannot be overwritten. JSON records input/model/source hashes, actual imports, software versions, shape and mesh settings. CPU and licensed SMPL suffice: no checkpoint, training set or network inference. Prior tools/results are unchanged.

## Completed validation

**20 tests pass** in [test_paper_metrics.py](../tests/test_paper_metrics.py): 13 asset-independent and7 licensed-SMPL checks. Without SMPL, the latter explicitly skip. Seven existing sensor-metric regressions also pass.

| Analytic case | Expected and verified |
|---|---|
| Identical static poses | Four metrics and supplementary local error zero |
| Identical changing poses | Pose errors zero; predicted/GT jerk equal, possibly nonzero |
| Root Z rotation90° | Global MPJRE90°; average local error90/24°; spatial errors match static geometry |
| Rotate terminal joint only | Joint positions unchanged, MPJPE near0; vertices move, MPJVE nonzero |
| Joint shift0.01m / vertex shift0.02m | ×100 cm conversion, included root, distinct point denominators |
| Whole-body translation | Pelvis-aligned spatial errors zero; translational jerk retained |
| Constant/linear/quadratic trajectories | Third difference zero within floating tolerance |
| Single-axis p(t)=t^3 | Jerk6m/s³; double fps for same discrete array gives48m/s³ |
| Boundaries and short sequences | No cross-recording differentiation; short jitter null; exclude padding; reject invalid offsets/gaps/overlaps |
| Compute chunks and shape switches | Repeatable fixed-input results; shared fixed shape; explicit pose-blendshape setting |

End-to-end recheck uses the same frozen E04 development predictions: **11 recordings, 25,423 frames, 25,390 jerk samples**. Chunks32/11 give identical aggregate metrics. A third run with chunk 19 independently reconstructs NumPy distances, trace/acos and third differences: maximum discrepancy **0.00005493**, in the corresponding metric units, below0.0001. Small angular discrepancies reflect atan2 versus trace/acos floating-point handling.

| Metric | Same development cache, frame weighted |
|---|---: |
| MPJRE | 21.292074° |
| MPJPE | 9.451584 cm |
| MPJVE | 10.257982 cm |
| Prediction jitter | 78.290040m/s³ =0.782900×(10^2 m/s^3) |
| GT jitter | 82.186566m/s³ =0.821866×(10^2 m/s^3) |
| Supplementary local rotation | 19.802034° |

New MPJPE differs from frame-weighted E04 by about 0.000000011 cm. This table is frame weighted; the E04 teaching report's primary table is subject equal. Do not interpret different reductions as performance changes.

These numbers validate the measuring tool on AMASS-pretrained offline-window development caches. At this historical point, DIP fine-tuning, online inference and refinement were not complete; **do not directly compare with paper Table2**.

Evidence: [frozen measurement protocol](../outputs/local_dip/metric_validation_20261113/PROTOCOL.json), [chunk 32](../outputs/local_dip/metric_validation_20261113/metrics_chunk32_checked.json), [chunk 11](../outputs/local_dip/metric_validation_20261113/metrics_chunk11_checked.json), [independent validation](../outputs/local_dip/metric_validation_20261113/validation.json), [validation code](../outputs/local_dip/metric_validation_20261113/validate_metrics.py). Original outputs stay in ignored local storage; licensed data/model assets are not copied.

## Follow-up: 2026-11-16

This unchanged metric implementation now measures all 24-mask final-DIP before/after and IMU-only-IK comparisons, independently audited for caches/aggregation/chunks. See [fine-tuning](DIP_FINETUNING_COMPARISON.md), [IK](IMU_ONLY_IK.md), and [shape/mesh/reduction sensitivity](PAPER_EVALUATION_ALIGNMENT.md). This document preserves M01's earlier known-pose record; metric Python sources are unchanged.
