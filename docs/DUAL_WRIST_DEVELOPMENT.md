# Reproduce the head plus both wrists development comparison

E04 evaluates the fixed Epoch 59 checkpoint with `lw_rw_h` (left wrist, right wrist, head), then compares its predictions with all four frozen wrist/pocket configurations from E03. The completed study is [dual_wrist_dev_epoch59_20261110](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/REPORT.md). The [research log](RESEARCH_LOG.md) connects its question and conclusions to earlier work.

## Data and comparison

Reuse E03's reserved subjects s_07/s_08, exact-content split manifest, 11 recordings, 209 windows, and 25,423 valid frames. Keep the same checkpoint, SMPL asset, independent 125-frame CPU inference at 25 FPS, and partial tails. `dip_test.pt` is never opened. The four existing configurations reuse cached predictions; only `lw_rw_h` receives new inference. No training is performed.

The primary outcomes are mean left/right wrist position error and the larger of the two wrist mean errors. Aggregate joint errors within each subject using valid frames, give both subjects equal weight, then derive these outcomes. The worse-wrist measure is the maximum of the two aggregated wrist errors, rather than an average of per-frame maxima or subject maxima. Frame-weighted, sequence-equal, subject, sequence, window, and joint results are also exported.

Every paired delta is new configuration minus comparator. Negative means lower error. Both arm global errors, individual wrist errors, leg position, root orientation, and whole-body metrics remain visible. There is no application-specific acceptable leg-error increase; report the tradeoff without inventing a threshold after inspecting results.

## Reproduction commands

Use the existing `imuposer` Conda environment. The tools require the original E03 study and its referenced local assets and sources to remain available and unchanged. The recorded source imports span the team checkout and original editable checkout, as documented in [the E03 guide](WRIST_POCKET_DEVELOPMENT.md). This is a local reproduction workflow, not a standalone asset distribution.

Use a fresh results directory. All stages refuse to overwrite existing stage outputs. Copying the protocol reuses the frozen E04 plan; its timestamp describes the original plan, while the new evaluation provenance records the reproduction time.

```bash
cd /home/alex/research/PoseEstimationWithWearables
STUDY_DIR=outputs/local_dip/dual_wrist_dev_epoch59_20261110_reproduction
mkdir "$STUDY_DIR"
cp outputs/local_dip/dual_wrist_dev_epoch59_20261110/PROTOCOL.json "$STUDY_DIR/PROTOCOL.json"
cp outputs/local_dip/dual_wrist_dev_epoch59_20261110/BASELINE_RECORD.json "$STUDY_DIR/BASELINE_RECORD.json"

conda run --no-capture-output -n imuposer python tests/test_dual_wrist_dev.py
conda run --no-capture-output -n imuposer python tests/test_wrist_pocket_dev.py
conda run --no-capture-output -n imuposer python tests/test_sensor_metrics.py

conda run --no-capture-output -n imuposer python tools/dual_wrist_dev.py evaluate --run-dir "$STUDY_DIR"
conda run --no-capture-output -n imuposer python tools/dual_wrist_dev.py analyze --run-dir "$STUDY_DIR"
conda run --no-capture-output -n imuposer python tools/dual_wrist_dev.py render --run-dir "$STUDY_DIR"
conda run --no-capture-output -n imuposer python tools/dual_wrist_dev.py validate --run-dir "$STUDY_DIR"
```

The report is an authored interpretation. These commands reproduce inference, numerical analysis, figures, matched examples, and validation; they do not generate scientific conclusions automatically.

## Saved artifacts

| Artifact | Purpose |
|---|---|
| `PROTOCOL.json` | Question, hypotheses, outcomes, aggregation, comparisons, and decision rule saved before new-mask inference |
| `BASELINE_RECORD.json` | Hashes of all 46 original E03 files at the start of E04 |
| `evaluation/provenance.json` | Exact command, software, source/asset hashes, import paths, coverage, and reused configurations |
| `evaluation/lw_rw_h_metrics.json` | Per-window, per-joint position/local/global errors for the new mask |
| `evaluation/predictions.npz` | New predictions, references, offsets, and window identities |
| `evaluation/resources.json` | CPU workflow time and peak process memory |
| `source_snapshot/` | Evaluated Python sources and hashes without model/data assets |
| `analysis/overall.csv` | All five configurations under three aggregation rules |
| `analysis/per_{subject,sequence,window,joint}.csv` | Endpoint and joint breakdowns |
| `analysis/comparisons.csv` | All paired outcomes at overall, subject, sequence, and window levels |
| `analysis/summary.json` | Summaries and recording-level improvement counts |
| `analysis/figures/` | PNG and PDF comparisons |
| `visuals/` | Same fixed recordings and times as E03, now including both wrists |
| `validation.json` | Independent cached metric and aggregate recomputation, pairing checks, and unchanged hashes |
| `REPORT.md` | Findings, exceptions, and interpretation |

Synthetic regression tests check infrastructure only. All accuracy measurements come from the reserved DIP recordings. The study is an offline development comparison with two people, not a paper-equivalent benchmark or a consumer-device evaluation. All experiment artifacts remain local and ignored by Git.
