# Reproduce the paired sensor evaluation

This workflow evaluates the preselected Epoch 59 checkpoint on every processed DIP test frame with two masks. It does not train, refine with IK, select a checkpoint on DIP test results, or claim paper-equivalent performance. Existing evaluation scripts and their uncommitted changes are left intact.

The completed study is in `outputs/local_dip/sensor_pair_epoch59_20261027/`. Start with its `REPORT.md`, `analysis/summary.json`, `validation.json` and `final_validation.json`. Outputs remain ignored by Git; model/data assets are neither added nor published.

## Run

Use the existing `imuposer` Conda environment and a **fresh** results directory. Commands below reproduce the executed workflow with a new destination, preserving the completed run. Each stage refuses to overwrite its destination.

```bash
cd /home/alex/research/PoseEstimationWithWearables
STUDY_DIR=outputs/local_dip/sensor_pair_epoch59_20261027_reproduction

conda run --no-capture-output -n imuposer python tests/test_sensor_metrics.py

conda run --no-capture-output -n imuposer python tools/eval_sensor_pair.py \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_test.pt \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --reference-global outputs/local_dip/epoch59_full_metrics.json \
  --output-dir "$STUDY_DIR" --threads 2

conda run --no-capture-output -n imuposer python tools/analyze_sensor_pair.py \
  --run-dir "$STUDY_DIR" --bootstrap-draws 10000 --seed 20261027

conda run --no-capture-output -n imuposer python tools/diagnose_sensor_pair.py \
  --run-dir "$STUDY_DIR"

conda run --no-capture-output -n imuposer python tools/render_sensor_cases.py \
  --run-dir "$STUDY_DIR"

conda run --no-capture-output -n imuposer python tools/validate_sensor_pair.py \
  --run-dir "$STUDY_DIR"
```

The human-authored research interpretation in `REPORT.md` is a reviewed deliverable for the completed study; the commands regenerate the numerical tables, figures, animations, diagnostics and validation artifacts. They do not automatically generate narrative conclusions. To reproduce without the earlier global report, omit `--reference-global`; record that the historical baseline cross-check is then unavailable.

## Software and import assumptions

The executed environment was Python 3.10.21, PyTorch 2.7.0+cu128 (CPU inference), NumPy 1.23.5, PyTorch Lightning 1.9.5, SciPy 1.10.1, Matplotlib 3.7.5, Pillow 12.3.0 and chumpy 0.70. No dependency was upgraded or installed. Runtime CPU threads=2, batch size=1, `eval()` and `torch.inference_mode()`. No GPU is required.

The current Conda environment imports `imuposer` from the editable original checkout `/home/alex/research/IMUPoser/src/imuposer`, while the evaluation entry points are in the team repository. Its dataset implementation uses shared input storage and masks at fetch time. `DipTestSubset` pre-masks the retained input so that both that implementation and the team's inherited eager-masking implementation give the same features. Regression tests exercise this interaction. Check `provenance.json` and `diagnostics/training_source_audit.json` before using another environment. `source_snapshot/` preserves the evaluated workflow and the actual imported Python package, with hashes, without copying model/data assets.

## Files and metric conventions

| Artifact | Content |
|---|---|
| `provenance.json` | Exact argv, checkpoint/data/SMPL hashes, source hashes, import path, versions, configuration and coverage |
| `{global,lw_rp_h}_metrics.json` | Overall and per-window scalar and 24-joint position/local/global errors; GT hashes |
| `predictions.npz` | Cached predictions for both masks and identical GT, window offsets and keys; no raw IMU inputs |
| `validation.json` | Coverage/masking/pairing and every-window comparison with the prior global baseline |
| `analysis/overall.csv` | Frame-weighted and sequence-equal results; absolute and relative differences |
| `analysis/per_sequence.csv` | Sequence ID, frames, windows, both configurations and paired differences |
| `analysis/per_window.csv` | Exact window keys and paired scalar differences |
| `analysis/per_{joint,region}.csv` | Frame-weighted and sequence-equal joint/region metrics |
| `analysis/per_sequence_{joint,region}.csv` | Joint/region metrics per complete sequence |
| `analysis/per_window_{joint,region}.csv` | Joint/region metrics per matching window |
| `analysis/summary.json` | Improvement counts, cluster bootstrap, focus cases and exploratory motion descriptors |
| `analysis/figures/` | PNG/PDF sequence comparison, window delta histogram, joint and region comparisons |
| `analysis/failure_cases.csv` | Preidentified cases and post hoc extreme differences |
| `input_consistency.json`, `analysis/input_consistency_by_sensor.csv` | Input orientation/acceleration consistency proxies, not corruption labels |
| `diagnostics/training_source_audit.json` | Training-mask enumeration, actual source fingerprints and remaining historical uncertainty |
| `diagnostics/upstream_rotation_*.{csv,json}` | GT-assisted replacement of root/torso rotations; diagnostic only |
| `visuals/` | Four matched snapshot panels, two 125-frame/25 FPS GIFs, manifest |
| `final_validation.json` | Independent cached-window recomputation, CSV aggregation audit and asset/source hash rechecks |

Position errors use original SMPL FK, the same mean male shape and zero translation for prediction and GT, all 24 joints including the zero-error pelvis, and no rigid or root-rotation alignment. Local/global rotations use trace/acos SO(3) geodesics in degrees, with clamping. Per-window joint means exclude padding; aggregate by valid frames and then equal joints. Sequence-equal results first average valid frames within each sequence, then average its 19 sequence means.

`delta = lw_rp_h - global`, so negative favors three locations. Percentage change divides that absolute delta by the corresponding global baseline, **not** the average of individual percentage changes. Improvement counts use an absolute tie tolerance of `1e-5` in the reported metric units. Window counts give each window one vote, including partial tails; the separate improved-window-frame proportion accounts for unequal lengths.

Four primary groups partition all joints: arms `[16,17,18,19,20,21,22,23]`, legs `[1,2,4,5,7,8,10,11]`, torso `[0,3,6,9,13,14]`, head `[12,15]`. Thus head includes the neck and torso includes the zero-error root. Additional overlapping groups: upper body `[3,6,9,12,13,14,15,16,17,18,19,20,21,22,23]`, left arm `[16,18,20,22]`, right arm `[17,19,21,23]`. These are analysis definitions, not asserted paper groupings.

Bootstrap draws sample 19 paired sequences with replacement. For the frame-weighted estimand each draw recomputes the ratio of error sums to frame sums; the sequence-equal estimand averages the sampled sequence means. Percentile intervals are descriptive: sequences share at most two test subjects and should not be treated as 19 independent subjects. Frames/windows are never used as independent bootstrap samples.

The evaluation reruns global once because the old report lacks per-joint metrics and predictions. Its 203 scalar window results are checked against the existing baseline; later analysis and rendering reuse the cache without further inference. No prior experiment output, checkpoint, dataset or SMPL model is overwritten.
