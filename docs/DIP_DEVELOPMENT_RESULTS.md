# DIP development training results and final training budget

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Review date: 2026-11-15. Training executed by Alex in an ordinary WSL Terminal. Results and integrity reviewed with Codex assistance.

**Development training is complete; the frozen selection rule chooses a final budget of 2 epochs.** Next, restart from the original AMASS Epoch 59 and train s_01–s_08 for exactly two epochs, then compare before/after on s_09/s_10 under one evaluation protocol. This review did not launch final training or test inference.

## Training results

Run directory: `outputs/local_dip/dip_dev_v1`; run ID: `58592e9dbc11470f9990bacb20d6aa9f`. RTX 5060 Ti, cuda: 0, batch 256, 24 masks, Adam lr 3e-4; 30 epochs and 1,860 optimizer steps completed under the [frozen protocol](protocols/dip_finetuning_v1_20261114/PROTOCOL.json). Training: s_01–s_06; validation: s_07/s_08. Test data was not loaded.

| Completed epochs | Training combined loss | Validation combined loss | Interpretation |
|---|---: |---: |---|
| 1 | 0.02166910 | 0.02041846 | First development fine-tuning epoch |
| 2 | 0.01420470 | **0.01997489** | Minimum validation value across all 30 epochs |
| 30 | 0.00346574 | 0.02144157 | Last completed epoch; validation 7.34% above the minimum |

![Development training and validation curves](results/dip_dev_v1_20261115/learning_curve.png)

Training loss keeps decreasing while the independent development minimum occurs early, consistent with an overfitting tendency. This alone cannot distinguish subject-domain differences, dataset size, or input-configuration effects. The frozen learning rate, masks, loss, and epoch-selection rule are unchanged.

These values are pose plus FK-joint MSE including public-code padding terms. They are not position errors in cm and cannot be directly compared with AMASS validation loss. Four-metric improvement has not yet been measured in this dated review; low training loss does not establish paper reproduction.

## Budget selection and integrity

Checks cover all 30 ordered history rows, step count = 62 × completed epochs, finite losses, 80,070×24 valid training frames and 25,423×24 validation frames per epoch, all 20 Adam step counters equal to 1,860, and all 31 execution-source hashes. The full history exactly matches history embedded in the last checkpoint. Initialization and frozen-protocol hashes are unchanged.

| Artifact | File | SHA-256 |
|---|---|---|
| Best development checkpoint | `checkpoints/epoch_002_03a6585efa03.ckpt` | `4515e13ae95ee5ec49ff90d941c81149ec4c583ba293aadf21abe17de699ba16` |
| Last development checkpoint | `checkpoints/epoch_030_6da9e5d52b29.ckpt` | `298e8580decef1ca2ee9b08370a714116f336a5d652b8919770e30000aa21501` |

The best epoch has internal zero-based index 1, corresponding to **2 completed epochs**. Final-stage `--check-only` verifies 2 epochs, 20,736 training samples, 81 batches per epoch, batch 256, and no development validation or test loading: 162 expected updates. This check neither constructs/updates the training model nor creates the final training directory.

The actual run lasted from 2026-11-15 01:56:51 to 02:12:20 America/Chicago, about 15 minutes 29 seconds from RUN UTC start and completion timestamps. Recorded training plus validation compute totals 921.19 seconds. Peak PyTorch GPU allocated memory: 2,906.05 MiB, distinct from system-wide GPU use; peak process RSS: 2,147.19 MiB. Free GPU memory at startup: 6,887 MiB. This execution verifies the frozen batch 256 forward/backward/update loop on this device under those resource conditions.

Metadata copies: [verification.json](results/dip_dev_v1_20261115/verification.json), [30-epoch history](results/dip_dev_v1_20261115/history.csv), [read-only final-stage plan](results/dip_dev_v1_20261115/final_stage_check.json). Full review and figures: `outputs/local_dip/dip_dev_review_20261115/`. Original training records and checkpoints are unchanged.

## Launch final first-eight training

Run in an ordinary WSL Terminal:

```bash
cd /home/alex/research/PoseEstimationWithWearables

conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage final \
  --device cuda:0 \
  --development-run outputs/local_dip/dip_dev_v1 \
  --output outputs/local_dip/dip_final_v1
```

The launcher reads selected K=2, resets the original Epoch 59, Adam and seed42; no manual epoch override is needed. As predeclared, s_07/s_08 join training and are no longer independent validation for this final model. Final output: `outputs/local_dip/dip_final_v1/final.ckpt`. Development artifacts remain unchanged; add `--resume` to the same command after interruption.

Only afterward evaluate the fixed final model before/after under identical rolling windows, smoothing, body model, 24 masks, and four metrics. Test outcomes do not reselect this run's epochs or checkpoint.
