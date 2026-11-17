# Reproduce the wrist and pocket development experiment

The completed local study is `outputs/local_dip/wrist_pocket_dev_epoch59_20261110/`. Read its `REPORT.md` or self-contained `REPORT.html` for the teaching report. The results use the fixed Epoch 59 checkpoint, all four wrist-side × pocket-side combinations with head input, and subjects s_07/s_08 reserved from DIP's training subject pool before development inference. No inference on `dip_test.pt`, training, or checkpoint selection is part of this workflow.

## Reservation and data identity

The original `dip_train.pt` has 41 sequences without subject labels. The reservation reconstructs acceleration, orientation, and pose preprocessing for every raw recording from s_07 and s_08. It requires bitwise equality of all three arrays to match the existing processed sequences. Every one of the 11 raw recordings matched, giving original sequence indices 30–40 and 25,423 frames. The other 30 sequence indices are excluded from this experiment. Subjects s_01–s_06 remain the proposed training pool for future DIP fine-tuning; this reservation does not automatically change the old fine-tuning loader.

The current installed global training loader excludes DIP files. This supports treating these subjects as development data for the AMASS baseline, although a complete historical training corpus audit is unavailable. Do not call them a new population test set.

## Run in a fresh directory

The tools refuse to overwrite stage outputs. Use the existing `imuposer` Conda environment and obtain DIP, the checkpoint, and SMPL separately through their existing licenses. The reproduction uses the original checkpoint SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058`.

```bash
cd /home/alex/research/PoseEstimationWithWearables
STUDY_DIR=outputs/local_dip/wrist_pocket_dev_epoch59_20261110_reproduction
mkdir "$STUDY_DIR"
cp outputs/local_dip/wrist_pocket_dev_epoch59_20261110/PROTOCOL.json "$STUDY_DIR/PROTOCOL.json"

conda run --no-capture-output -n imuposer python tests/test_wrist_pocket_dev.py
conda run --no-capture-output -n imuposer python tests/test_sensor_metrics.py

conda run --no-capture-output -n imuposer python tools/eval_wrist_pocket_dev.py reserve \
  --run-dir "$STUDY_DIR" \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --raw-dip /home/alex/research/IMUPoser/data/raw/DIP_IMU

conda run --no-capture-output -n imuposer python tools/eval_wrist_pocket_dev.py evaluate \
  --run-dir "$STUDY_DIR" \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl

conda run --no-capture-output -n imuposer python tools/analyze_wrist_pocket_dev.py analyze --run-dir "$STUDY_DIR"
conda run --no-capture-output -n imuposer python tools/analyze_wrist_pocket_dev.py render --run-dir "$STUDY_DIR"
conda run --no-capture-output -n imuposer python tools/analyze_wrist_pocket_dev.py validate --run-dir "$STUDY_DIR"
```

The report is an authored interpretation of the completed run. The analysis commands regenerate its numerical inputs and figures, rather than automatically generating scientific conclusions.

## Analysis conventions

Each mask contains exactly three of five supported locations. The head is fixed; one wrist and one pocket are retained. Missing acceleration triples and all nine missing orientation entries are zero; present inputs remain bitwise unchanged. Inference uses CPU, two threads, 125-frame independent BiLSTM windows at 25 FPS, and retains every partial final window.

Primary outcomes are left/right arm global rotation error and left/right wrist position error. Average valid frames within each subject, then weight the two subjects equally. Frame-weighted and sequence-equal summaries are secondary sensitivity views. No frame-level p values, population confidence claims, or test-based checkpoint search are used.

For any endpoint E, with L/R indicating wrist and pocket sides:

- Wrist effect R minus L = `((E(R,R)-E(L,R)) + (E(R,L)-E(L,L)))/2`.
- Pocket effect R minus L = `((E(L,R)-E(L,L)) + (E(R,R)-E(R,L)))/2`.
- Interaction = `(E(R,R)-E(L,R)) - (E(R,L)-E(L,L))`.

All errors are smaller-is-better, but the expected sign of the wrist contrast depends on which arm is measured. Positive for left-arm error and negative for right-arm error support direct wrist coverage. An interaction measures dependence on pocket side, not statistical significance.

Positions use inherited SMPL FK, mean male shape, zero root translation, all 24 joints including the zero-error pelvis, and no rigid or root-rotation alignment. Primary arm groups are `[16,18,20,22]` and `[17,19,21,23]`; wrists are joints 20/21. Global rotations include parent rotations; local rotations measure parent-relative joint rotations, except that root-local rotation is body orientation. These are offline baseline results, not paper-equivalent or consumer-device performance.

## Artifacts and checks

| File | Purpose |
|---|---|
| `PROTOCOL.json` | Plan saved before reservation and inference |
| `split_manifest.json` | Selected subjects, recording identities, original indices, all source hashes, and remaining sequence indices |
| `evaluation/provenance.json` | Exact command, imported source paths, source hashes, software versions, checkpoint and asset hashes |
| `evaluation/*_metrics.json` | Every window's 24-joint metrics for each configuration |
| `evaluation/predictions.npz` | Shared reference and all four cached predictions with offsets and identities |
| `analysis/overall.csv` | All configurations under three aggregation rules |
| `analysis/per_{window,sequence,subject}.csv` | Endpoint breakdowns |
| `analysis/per_joint.csv` | Every joint under each aggregation rule |
| `analysis/contrasts.csv` | Wrist, pocket, interaction, and simple effects at each level |
| `analysis/figures/` | PNG and PDF plots |
| `visuals/` | Examples selected by recording name before inference, at identical times and camera settings |
| `validation.json` | Independent cached-metric recomputation, aggregation/contrast checks, and unchanged asset/source hashes |
| `source_snapshot/` | Evaluated Python source only; no licensed model or data assets |
| `REPORT.md`, `REPORT.html` | Detailed findings and undergraduate research guidance |

The environment currently imports `imuposer` from the original editable checkout while the experiment tools are in the team repository. Provenance and the source snapshot capture both. Synthetic inputs in the regression tests check infrastructure only; every accuracy result comes from the real reserved DIP recordings.

Future fine-tuning must explicitly filter out development indices recorded in `split_manifest.json`; using the old loader on the entire `dip_train.pt` would destroy this separation. No model training or loader modification is included in this experiment.
