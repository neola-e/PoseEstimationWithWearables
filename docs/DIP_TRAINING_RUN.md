# Launch DIP fine-tuning in WSL

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Date: 2026-11-14. Entry point: [train_dip_finetuning.py](../tools/train_dip_finetuning.py). This dated task prepared and validated the launcher; Alex launches actual training in an ordinary WSL Terminal. The frozen protocol and Epoch 59 are unchanged; the earlier CPU smoke is not repeated.

The launcher follows [DIP_FINETUNING_V1](protocols/dip_finetuning_v1_20261114/PROTOCOL.json): Adam lr 3e-4, seed42, 24 masks, batch 256, 30 development epochs, selection by minimum independent-development combined validation loss. Exact ties retain the earlier completed epoch.

## Launch development training

First inspect free GPU memory with `nvidia-smi` in an ordinary WSL Terminal. At the prior preflight, 6,472/8,151 MiB was already occupied. The code does not terminate other programs or alter the batch size.

```bash
cd /home/alex/research/PoseEstimationWithWearables

conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage development \
  --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1
```

By default, read the frozen protocol, identity manifest, and source hashes in `outputs/local_dip/dip_finetuning_v1_20261114/`. Train s_01–s_06, validate s_07/s_08: 62 training and 20 validation batches per epoch. No test loading or automatic transition to final first-eight training.

Unavailable CUDA exits with no CPU fallback. CUDA OOM stops execution and records the error; batch, learning rate, precision and budget remain fixed. Check free memory before resuming. No temporary CLI epochs/batch/lr overrides are accepted.

A new output directory must not exist. Reuse requires explicit `--resume`. This directory is separate from frozen assets and existing checkpoints.

## Resume and result files

```bash
conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage development \
  --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1 \
  --resume
```

Resume verifies protocol, execution sources, Torch version, device, data and checkpoint hashes. Restore model, Adam, Python/NumPy/Torch RNG and independent training/validation DataLoader generators; continue after the last **fully committed epoch**. Incomplete epochs rerun; if epoch1 never completed, restart that epoch from the original Epoch 59. Changes to code, precision or training settings cannot silently reuse old state.

| File | Purpose |
|---|---|
| `train.log` | Live batch progress, epoch train/validation loss, device and memory; also printed to terminal |
| `history.csv`, `history.json` | Per-epoch pose/joint/combined loss, valid/padded frames, duration, best epoch |
| `best.ckpt` | Symlink to minimum-development-validation checkpoint |
| `last.ckpt` | Symlink to last fully committed epoch |
| `checkpoints/epoch_*.ckpt` | Separate complete model/Adam/RNG/sampler state per epoch; never overwrite/delete |
| `STATE.json` | Atomically committed last/best paths and hashes; authoritative resume state |
| `progress.json` | Completed epochs, step and best score |
| `completion.json` | Best epoch and final-stage K after the full budget completes |
| `failure.json` | Interruption/training exception, last committed epoch and resume guidance |
| `RUN.json`, `source_snapshot/` | Command, configuration, source hashes/snapshots; protocol copy preserves original bytes |

Write an independent checkpoint, atomically update `STATE.json`, then update links/history. Uncommitted files after interruption are not resume points; links/history can be reconstructed from committed checkpoints. Completed checkpoint bytes are unchanged. Thirty epochs require about 4 GB disk; best/last symlinks do not duplicate weights.

Monitor from another ordinary WSL Terminal:

```bash
tail -f /home/alex/research/PoseEstimationWithWearables/outputs/local_dip/dip_dev_v1/train.log
```

## Final first-eight training with selected budget

After all 30 development epochs finish:

```bash
conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage final \
  --device cuda:0 \
  --development-run outputs/local_dip/dip_dev_v1 \
  --output outputs/local_dip/dip_final_v1
```

Verify completed development and selected K against full history and best checkpoint. Reload **original AMASS Epoch 59**, reset Adam/seed42, and train s_01–s_08 for exactly K epochs. Do not continue from development-best weights or select on s_09/s_10. Final pool: 20,736 masked samples, 81 batches per epoch; emit `final.ckpt` symlink at completion.

Add `--resume` for final-stage recovery. s_07/s_08 enter training in this stage, a role change explicitly recorded in the frozen protocol.

Rolling four-metric tests, all-mask aggregation and IK are subsequent work, not automatically executed by this command. Validation loss does not replace paper MPJRE, MPJPE, mesh MPJVE and jitter.

## Completed code verification

**All 10 training-engine tests pass**: padded B×T weighting for unequal batches; validation disables gradients/dropout; exact tie selection; nonfinite-loss rejection; refusal to train without CUDA; refusal to resume a full run from the old smoke directory; immutable checkpoint/tamper checks; commit state/orphan handling; deterministic epoch resume with shuffle/dropout/Adam. Tiny synthetic CPU networks are used, not another actual-model smoke.

The following read-only entry check verifies frozen assets and real DIP identities/sample counts without creating training output or updating parameters:

```bash
conda run --no-capture-output -n imuposer python tools/train_dip_finetuning.py \
  --stage development --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1 --check-only
```

GPU tensors/model forward were verified in an ordinary WSL Terminal by [G01](GPU_DIAGNOSTIC.md). This Codex sandbox cannot access the GPU; this dated launcher review does not claim multi-epoch GPU training or change the frozen recipe to bypass memory/permission limits.
