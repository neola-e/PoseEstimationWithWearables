# DIP fine-tuning protocol and engineering smoke

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Date: 2026-11-14. Frozen protocol, actual smoke and independent reload checks complete. Alex requested protocol freezing plus Epoch 59 data-loading/gradient/update/save/reload verification. Formal multi-epoch training and final tests are outside this smoke.

Preparation: [prepare_dip_finetuning.py](../tools/prepare_dip_finetuning.py); state/data: [dip_finetuning.py](../tools/dip_finetuning.py); smoke: [smoke_dip_finetuning.py](../tools/smoke_dip_finetuning.py); tests: [test_dip_finetuning.py](../tests/test_dip_finetuning.py).

## Frozen training protocol

[PROTOCOL.json](protocols/dip_finetuning_v1_20261114/PROTOCOL.json), ID `DIP_FINETUNING_V1`, SHA-256 `79bde74ead3f596c5806e3fa1d0cd5b07fcfb6260b0c9e81c218771ba50441ad`. Byte-identical protocol/data/source-hash metadata is versioned in `docs/protocols/dip_finetuning_v1_20261114/`. Original model sources, licensed data, Epoch 59 and existing research records are unchanged; architecture/state keys are retained.

[Paper §4.3/5.3](https://arxiv.org/html/2304.12518v1) specifies AMASS initialization, first-eight DIP training/last-two testing, architecture and losses. It does not fully freeze fine-tuning budget/internal validation/selection. The30-epoch cap, seed, development split and selection below are explicit local choices, not recovered author settings.

| Item | Frozen setting |
|---|---|
| Initialization | Original AMASS Epoch 59, step 165,060; SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058` |
| Trainable parameters | All model parameters; fresh Adam, no AMASS moments/steps |
| Optimizer | Adam lr `3e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay0, amsgrad/foreach off |
| Schedule/clipping/precision | No scheduler/clipping; float32; no AMP/TF32; immediately reject nonfinite loss/gradients |
| RNG | Python/NumPy/Torch seed42, deterministic algorithms, cuDNN benchmark off |
| Data/input | 25 Hz, 60D, lw/rw/lp/rp/h, acc/30; retain centered-five acceleration mean/orientation resampling, no double smoothing |
| Windows | Nonoverlapping125 frames, retain short tails; zero-pad input/target in batch; packed LSTM uses actual lengths |
| Masks | 24 sparse masks per motion window: 5 single, 10 double, 9 triple; no five-location global; retain60 channels |
| Loss | 144D column-convention6D pose MSE + all 24-joint SMPL FK position MSE, weights1:1; positions in meters |
| Padding | Explicitly retain public-code padding terms during training; exclude padding in four-metric reporting; do not silently change inherited recipe |
| Body | Existing mean male SMPL, zero translation, pose blend shapes off; asset hash in protocol |
| Formal batch | 256, shuffle=True, workers=0, drop_last=False; no automatic batch/lr change |
| Development budget | At most30 epochs, no early stop; full-development validation after each completed epoch |
| Selection | Minimum full-development combined validation loss; earlier completed epoch wins exact ties; weight validation batches by padded B×T, not equally across unequal lengths |
| Saving | Last/best each epoch; model, Adam, epoch/step, Python/NumPy/Torch RNG, sampler state, protocol hash |

### Exact24-mask order

Order is part of the protocol:

```text
lw_rw_h, rw_lp_rp, lw_rw_rp, lw_rp_h, rw_rp_h,
lw_lp_rp, lw_rw_lp, lw_lp_h, rw_lp_h,
lw_rw, lw_lp, lw_rp, lw_h, rw_lp, rw_rp, rw_h, lp_rp, lp_h, rp_h,
lw, rw, lp, rp, h
```

Epoch 59's historical25-mask pretraining remains a known difference. Removing global changes new DIP samples, not the initialization's past exposure.

## Subjects and data manifest

Split subjects before window/mask expansion so masks of the same motion cannot cross train/development. Reconstruct acc/ori/pose from local raw DIP using inherited filling/trimming/resampling/smoothing and bitwise match processed containers. Record raw hashes, three processed-content hashes and excluded-record reasons.

1. **Development: **s_01–s_06 training, s_07/s_08 validation. Reserved development subjects contribute no smoke/development gradients.
2. **Final paper split: **after selecting K completed epochs(1–30), reset to identical Epoch 59/fresh Adam and train s_01–s_08 exactly K epochs. s_07/s_08 explicitly become training and cannot remain independent validation for that model.
3. **Test: **s_09/s_10 only for locked before/after reporting, never hyperparameter/epoch/checkpoint selection. Preparation/smoke did not open their raw files or test container.

[data_manifest.json](protocols/dip_finetuning_v1_20261114/data_manifest.json) records all identities. All 41 first-eight processed recordings have unique raw-content matches. Of42 raw candidates, s_01/02.pkl fails inherited finite-value criteria; reason/hash retained. The11 reserved s_07/s_08 identities are unchanged.

| Stage | Recordings | Frames | Motion windows | Samples ×24 | Batches/epoch at256 |
|---|---: |---: |---: |---: |---: |
| s_01–s_06 development training | 30 | 80,070 | 655 | 15,720 | 62 |
| s_07/s_08 validation | 11 | 25,423 | 209 | 5,016 | 20 |
| First-eight final fixed-budget training | 41 | 105,493 | 864 | 20,736 | 81 |

Prior research discussions already used test observations; preserve this history rather than claim an untouched blind test.

## Before/after evaluation

Train on125-frame chunks; report with [audited rolling last-frame inference](ROLLING_INFERENCE.md): 125-window, stride1, short prefixes, no state carry across windows/recordings. Both models retain identical centered acceleration averaging with disclosed two-frame lookahead; trailing-five smoothing is separate sensitivity analysis.

Use [POSE_METRICS.md](POSE_METRICS.md): global MPJRE, pelvis-aligned MPJPE/mesh MPJVE, whole-recording jitter. Within each mask, spatial errors use valid-frame weighting, jitter T−3 weighting; then equally average24 masks. Keep per-recording/per-mask rows. Fix shared shape, joints/vertices, mesh switch, zero translation. IMU-only IK is a later independent component; this dated comparison uses raw networks.

Development validation selects the model; paper-reproduction claims require four metrics, a fixed protocol and differences, not lower training loss alone.

## Smoke checks and results

Smoke: **3 Adam steps, batch 8,24 samples**, exactly one per mask, varied training windows including a short tail. First verify training/validation identities, shapes, lengths; only training contributes gradients.

After step 3, independently reload model/optimizer: bitwise model/Adam states, matching same-input predictions, matching continuation from identical RNG and specified input. One extra update in uninterrupted and reloaded branches checks continuation without rewriting the saved step 3 checkpoint.

Checkpoint metadata:`kind=engineering_smoke_only`,`selection_eligible=false`,`epoch=-1`; not a completed formal epoch or best-model candidate.

Actual imuposer **CPU, 2 threads**: this session sees no CUDA, device_count0, which does not establish missing host hardware. Smoke runtime **6.27 s**, peak RSS **1,741 MiB**, excluding earlier raw matching; not a formal batch 256/GPU timing estimate.

| Check | Observation |
|---|---|
| Loaders | Only s_01–s_06 training/s_07, s_08 validation; identities/shapes/lengths/content hashes pass; no validation gradients or test loading |
| Masks | 24 samples cover every sparse mask exactly once |
| Step1 | Loss0.02971774; 951 valid, 49 padded frames; 20/20 parameter tensors have finite nonzero gradients and update |
| Step2 | Loss0.03434055; 839 valid, 161 padded; 20/20 tensors finite nonzero gradients/update |
| Step3 | Loss0.04248882; 1,000 valid, 0 padded; 20/20 tensors finite nonzero gradients/update |
| Change vs Epoch 59 | 20/20 tensors changed; max absolute0.000901447 |
| Model/Adam reload | Weights, 20 Adam moment groups/step counters bitwise identical; each step=3 |
| Reload predictions | Same probe max absolute difference0 |
| RNG/continuation | Same restored Python/NumPy/Torch RNG, input, loss, model, Adam and final RNG bitwise match |
| Independent process | Rebuild8 probe features directly from container without new dataset/collate/loss helpers; targets/lengths bitwise equal; initial and smoke predictions match caches exactly |
| Unit checks | Six pass: 24 masks, subject isolation, public-loader mask/tail agreement, shared-data immutability, analytic padding loss, three-RNG replay/state-difference detection |

The three losses use different motions/masks and dropout; their order does not establish convergence. The smoke verifies engineering, not development/test accuracy improvement or paper reproduction.

Saved [smoke_step3.ckpt](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/smoke_step3.ckpt), SHA-256 `10f9656573fb41c522bd5ca80c65d2830a24fe3e391e98de2590fd5802afda7c`. [Results](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/results.json), [gradients/updates](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/steps.json), [independent validation](../outputs/local_dip/dip_finetuning_v1_20261114/independent_validation.json).

`FREEZE_MANIFEST.json` covers original frozen sources/inputs; `smoke/ARTIFACT_MANIFEST.json` covers smoke artifacts. Final integrity lists, logs and validator snapshots stay in the run root. Checkpoints, predictions, original assets are ignored large files; metadata copies in docs/protocols do not replace run backups.

## Rerun

Use a nonexistent new directory; freeze, then smoke:

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tools/prepare_dip_finetuning.py \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --reservation outputs/local_dip/wrist_pocket_dev_epoch59_20261110/split_manifest.json \
  --raw-dip /home/alex/research/IMUPoser/data/raw/DIP_IMU \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --output outputs/local_dip/dip_finetuning_v1_rerun

conda run --no-capture-output -n imuposer python tools/smoke_dip_finetuning.py \
  --protocol-dir outputs/local_dip/dip_finetuning_v1_rerun \
  --device auto --threads 2

conda run --no-capture-output -n imuposer python -m unittest discover \
  -s tests -p test_dip_finetuning.py -v

conda run --no-capture-output -n imuposer python tools/validate_dip_finetuning_smoke.py \
  --protocol-dir outputs/local_dip/dip_finetuning_v1_20261114 \
  --output /tmp/dip_finetuning_smoke_recheck.json
```

`--device auto` chooses CUDA if available, otherwise CPU; explicit `--device cuda:0` fails without CUDA. Frozen source/input hashes must match; changed code/data cannot reuse an old protocol. Formal budget is frozen; this entry executes only an engineering smoke.
