# Team handoff: IMUPoser reproduction

Snapshot: November 16, 2026. This repository contains the research implementation, frozen protocols, audit manifests, experiment reports, compact numerical results, figures, and validation tools. Training data, newly generated weights, and large prediction caches remain external.

## Read this first

1. [Research log](RESEARCH_LOG.md): experiment questions, methods, results, and decisions.
2. [Reproduction audit](REPRODUCTION_AUDIT.md) and [current paper alignment](PAPER_EVALUATION_ALIGNMENT.md): what matches the paper and what remains unresolved.
3. [Metric definitions](POSE_METRICS.md) and [rolling inference](ROLLING_INFERENCE.md): units, alignment, aggregation, and temporal dependencies.
4. [DIP training protocol](DIP_FINETUNING.md), [training run](DIP_TRAINING_RUN.md), and [four-metric comparison](DIP_FINETUNING_COMPARISON.md).
5. [IMU-only IK](IMU_ONLY_IK.md), [jitter diagnosis](IK_JITTER_DIAGNOSIS.md), and [independent Temporal IK experiment](TEMPORAL_IK_EXPERIMENT.md).
6. [Consumer-device roadmap](CONSUMER_DEVICE_ROADMAP.md): the next implementation stage.

Earlier sensor-placement experiments are retained in [sensor evaluation](SENSOR_EVALUATION.md), [wrist/pocket development](WRIST_POCKET_DEVELOPMENT.md), and [dual-wrist development](DUAL_WRIST_DEVELOPMENT.md). They are supplementary to the reproduction work.

## What is established

The final DIP model starts from the existing AMASS Epoch 59 checkpoint. Development training used s01–s06 with s07/s08 validation; its selected budget was two epochs. Final training restarted from the AMASS initialization with fresh optimizer state and trained on s01–s08 for two epochs, 162 updates. The frozen protocol uses 24 masks, Adam at 0.0003, batch 256, and seed 42. The held-out s09/s10 data did not select that budget.

The full comparison covers 19 held-out recordings and 24 masks, 24,078 frames per mask. Fine-tuning lowered all four aggregate metrics, with all four also lower in each of the 24 mask aggregates. MPJRE fell 25.76%, MPJPE 18.15%, mesh MPJVE 18.00%, and jitter 35.01%.

Framewise IK improves the three spatial errors slightly but raises jitter by 16.51% over the final network. Jitter analysis identifies pocket-containing masks and propagation through the pelvis as major contributors. Temporal IK is a separate experiment, with its weight selected on development data: it lowers jitter 4.42% relative to framewise IK and retains 98.58% of its vertex-error improvement, while remaining 11.36% above the network's jitter. Existing IK and no-IK results are preserved.

These are results under the recorded implementation. Remaining differences include historical 25-mask AMASS training, preprocessing and retained-data differences, incompletely recovered paper window/IK details, body-model conventions, and future information in centered input preprocessing. This does not yet establish strict reproduction of the paper's final numbers or accuracy on consumer devices.

## Code and assets

- `tools/`: inference, metrics, rolling windows, data audits, DIP fine-tuning, IK, Temporal IK, reports, and independent validators.
- `tests/`: numerical, temporal-boundary, aggregation, training-state, and optional licensed-asset integration checks.
- `docs/protocols/`: frozen training and evaluation metadata.
- `docs/reproduction_audit_20261111/`: traceable AMASS inventories and historical source snapshots.
- `docs/results/`: selected logs, tables, plots, manifests, runtime records, and independent validation reports.
- `.agents/skills/ml-training-recipes/`: the installed research skill, including its reference files.

The supporting source fixes used in the original working checkout are included here: configurable DataLoader workers, shared unmasked training windows with per-item masking, and resume support in the original AMASS launcher. These changes do not replace the frozen DIP launcher or alter its protocol.

Acquire AMASS, DIP-IMU, and the required SMPL assets under their respective licenses. Obtain trusted checkpoints separately; see [model card](MODEL_CARD.md) for the older Epoch 55 handoff and [DIP comparison](DIP_FINETUNING_COMPARISON.md) for current weight hashes. No new checkpoint or licensed dataset is published in this update. Existing upstream model assets are unchanged.

The evaluation was run from `/home/alex/research/IMUPoser` plus this team's `tools/` directory. Frozen records deliberately preserve those absolute paths and exact historical hashes. They describe the original run, not a checksum of every later checkout. Some experiment wrappers assume the recorded output layout. On another machine, inspect each entry point's `--help` and configure paths in a new run directory; do not overwrite frozen protocols or historical outputs. Large caches referenced by local-run manifests are external; the compact result folders have their own manifests for the files shipped here.

## Verify a checkout

Use Python 3.10 and install the package into a compatible environment:

```bash
python -m pip install -r requirements-inference.txt
python -m pip install -r requirements-visualization.txt
python -m pip install -e src/
PYTHONPATH="$PWD/src" python -m unittest discover -s tests -v
```

The inference requirements select CPU PyTorch. GPU training was validated in the separate `imuposer` environment with PyTorch 2.7.0+cu128 on an RTX 5060 Ti; runtime records are included with the experiments. Do not replace a working CUDA build with the CPU requirements when preparing GPU training.

Tests requiring external checkpoints or licensed assets explicitly skip when unavailable. Optional integration paths are controlled by `IMUPOSER_CHECKPOINT`, `IMUPOSER_TEST_CHECKPOINT`, `IMUPOSER_DIP_TEST`, `IMUPOSER_SMPL_MODEL`, and `IMUPOSER_SMPL`; the last two configure different test modules. Only use trusted checkpoints. Running this test suite does not start the formal 30-epoch development run or overwrite checkpoints.

Before publishing this snapshot, the complete suite passed **105 tests with no skips or failures** in the existing `imuposer` environment, using this checkout's `src/` via `PYTHONPATH`. The trusted Epoch 55 checkpoint, processed DIP test data, and licensed SMPL asset were configured for the optional integration checks. The four compact result manifests also verified successfully, covering 86 files. This checks the shared code and archived metadata; the completed full-dataset experiment audits are documented separately in their reports.

Full result validation additionally needs the original checkpoints, processed DIP data, SMPL asset, and prediction caches at the paths recorded in the experiment manifests. The `validate_*.py` tools and the reports describe those checks. The shipped validation JSON files record the completed historical audits; possession of those files alone is not a rerun.
