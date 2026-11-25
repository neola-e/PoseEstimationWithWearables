# IMUPoser rolling-window inference and dependency audit

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Date: 2026-11-13. Implementation, full-development comparison and independent checks complete. Using one Epoch 59 checkpoint, distinguish original nonoverlapping chunks from per-frame rolling inference, trace each output's inputs, and quantify differences with the same four metrics. Weights, original training code and prior experiments remain unchanged.

Implementation: [window_inference.py](../tools/window_inference.py); comparison: [compare_window_inference.py](../tools/compare_window_inference.py); tests: [test_window_inference.py](../tests/test_window_inference.py); run: [window_comparison_epoch59_20261113](../outputs/local_dip/window_comparison_epoch59_20261113/).

## Paper description and explicit local choices

[Paper §5.3](https://arxiv.org/html/2304.12518v1) describes online125-sample history and “1-sample overlap”; [§4.2](https://arxiv.org/html/2304.12518v1) describes five-frame averaging. Output position, startup boundaries and filter direction are not fully specified.

**Stride1 with last-frame output is our explicit online convention, not a verified recovery of every author detail.** Adjacent full 125-frame windows at stride1 overlap124 frames. Literal one-frame overlap implies stride124, which with last-frame-only output cannot emit one pose per input frame. Inspected public source has no author online entry resolving this ambiguity; overlap and stride are distinct.

Frame IDs are zero based; `[start,stop)` excludes stop.

| Setting | Input | Retained output | Startup/end |
|---|---|---|---|
| Original `chunk` | `[125k,min(125k+125,T))` | All frames in chunk | Actual-length tail; early outputs see later chunk inputs |
| `rolling_last` | At t: `[max(0,t-124),t+1)` | Last frame only | t=0 uses one frame, growing to125; retain warmup, no future padding |
| Optional `repeat_first` startup | Same real history, prepend frame0 repeats to125 | Last position | Explicit alternative only; numerical comparison uses short prefixes |

Rerun both LSTM directions per window; no hidden-state carry. Reset for each recording. BiLSTM remains bidirectional within its window, but the last-frame window contains no input after t. A full-sequence BiLSTM pass or state carry is not equivalent.

`RollingPoseStream.push(frame)` accepts one frame and emits one pose; caller `reset()` at recording boundaries. Buffer at most125 pre-scaled/masked60D frames. Batched `infer_sequence` accelerates independent windows for offline checks; later batch windows do not affect earlier ones. Offline throughput is not device end-to-end latency.

## Smoothing and future dependencies

Inherited preprocessing smooths acceleration only; orientation matrices retain60→25 Hz entrywise linear interpolation. No extra orientation/output smoothing.

| Setting | 25 Hz acceleration support at t | Edges | Effect at online t |
|---|---|---|---|
| Centered mean | `[t-2,t+2]`, clipped to recording | Average available samples | Up to2 future frames; at least about 80 ms wait to release corresponding result |
| Trailing-five mean | `[t-4,t]`, clipped | Use available1–5 startup frames | No future dependency at this25 Hz stage; possible lag, not equivalent to original smoothing |

Removing model lookahead does not make preprocessed inputs causal from the raw stream. Original preprocessing also uses forward/backward NaN filling and a right-side source sample during60→25 Hz interpolation. Here `rolling_trailing` means **causal at the25 Hz stage after those upstream operations**; strict raw-sensor causality remains separate work.

To avoid double smoothing, reconstruct trailing inputs from authorized local raw DIP. Keep NaN filling, trimming/resampling, changing only acceleration averaging direction. First verify reconstructed centered acceleration bitwise equals frozen processed data for each recording. Orientation, target, mask, scale and checkpoint remain identical.

## Trace one output

Save `seq{ID}_{variant}_trace.csv` per recording/setting:

- `output_frame`: original processed-recording frame for the pose.
- `input_start/input_stop`: real25 Hz network feature interval.
- `model_output_index/model_length`: selected output and valid length; batch zeros are not valid history.
- `left_repeats_of_frame_zero`: startup repeats, always0 in the main experiment.
- `acc_source_start/acc_source_stop`: pre-smoothing25 Hz support after accounting for the filter; orientation uses `ori_source_start/ori_source_stop`.
- `available_at_25hz_frame/future_frames_at_25hz`: earliest complete-dependency frame and lookahead relative to output time. Not compute duration; excludes raw filling/interpolation/transport delays.

At t=125, rolling input `[1,126)`, output index124. Centered acceleration support `[0,128)`; trailing `[0,126)`. Original chunk input `[125,250)`, index0, may access frame249 and, with centered filtering, 251. Clip all intervals to recording boundaries. CSV plus filter rules reconstructs inputs without guessing cache/padding behavior.

## Frozen comparison

Pre-run [PROTOCOL.json](../outputs/local_dip/window_comparison_epoch59_20261113/PROTOCOL.json) fixes Epoch 59; all 11 exactly matched s_07/s_08 development recordings, 25,423 frames; mask `lw_rw_h`; 125-frame windows; stride1; short-prefix startup; no DIP fine-tuning or IK.

1. `chunk_centered`: recompute original chunk baseline.
2. `rolling_centered`: change model window only, isolating inference semantics.
3. `rolling_trailing`: additionally change acceleration filtering direction.

No s_09/s_10 test access, checkpoint selection or mask search. Use [POSE_METRICS.md](POSE_METRICS.md): 24 joints, 6,890 vertices, shared mean male SMPL, pose blend shapes off, zero translation, pelvis translation alignment only. Differentiate within recordings; retain chunk-boundary discontinuities in each recording's jitter.

Primary spatial reduction is frame weighted, jitter T−3 weighted; retain sequence-equal alternatives. Hash sources, checkpoint, SMPL, processed/raw data and split manifest. Actual package import is `/home/alex/research/IMUPoser/src/imuposer`; preserve package/tools source snapshots in the run.

## Results and verification

All 11 recordings, 25,423 frames completed fresh inference for all three variants, each with25,390 jerk samples. Lower is better:

| Inference/filter | MPJRE ° | MPJPE cm | MPJVE cm | Jitter, 100m/s³ |
|---|---: |---: |---: |---: |
| Chunk + centered | 21.292074 | 9.451584 | 10.257982 | 0.782900 |
| Rolling last + centered | 22.068137 | 9.811889 | 10.749727 | 0.859217 |
| Rolling last + trailing five | 22.526798 | 10.149600 | 11.133277 | 0.892934 |

With centered filtering fixed, rolling increases MPJRE0.776063°, MPJPE0.360305 cm, MPJVE **0.491745 cm(4.79%)**, jitter **9.75%**. Mean local-rotation difference between predictions: 7.082630°. Chunk predictions cannot simply be relabeled online. Rolling MPJVE improves in only 2/11 recordings; the first completed recording improved but did not represent the full result.

Within rolling, trailing filtering adds MPJVE **0.383550 cm(3.57%)**, jitter3.92%. Filter direction has measurable effects here; fixing lookahead is not cost-free. Window changes also alter context distribution, resets and selected outputs, so their entire effect cannot be attributed solely to future access. This does not prove trailing filtering worse under other training recipes.

Auxiliary sequence-equal MPJVE: 9.765306/10.433512/11.063694 cm. Full metrics, local error, reference jitter, per-recording results and both reductions: [metrics.json](../outputs/local_dip/window_comparison_epoch59_20261113/metrics.json). This is one three-location development mask, without DIP fine-tuning/IK/all 24 evaluation; not directly comparable with paper test scores.

Independent entry: [validate_window_comparison.py](../tools/validate_window_comparison.py). It derives interval boundaries from frame IDs without calling window/trace generators, and checks frozen E04 caches:

| Check | Result |
|---|---|
| All dependency intervals | 76,269 rows pass; no missing/duplicate frames or cross-recording windows |
| Actual-network trace reconstruction | 231 independent windows cover startup, 125 boundaries and ends; max6D difference1.222×10⁻⁶ |
| New chunks vs frozen E04 | Targets bitwise equal; 25,423 predicted frames max6D difference3.368×10⁻⁶ from batch numerics |
| Centered reconstruction | All 11 acceleration arrays bitwise match frozen processed data; no double smoothing |
| Chunk lookahead | 25,214 outputs use future model inputs, maximum 124 frames; 126 including centered filtering |
| Rolling + centered | Zero model lookahead; 25,412 outputs still depend on future pre-smoothing data, maximum 2 frames |
| Rolling + trailing | Zero pre-smoothing25 Hz lookahead; upstream filling/interpolation separate |
| Unit/model integration | Nine new tests pass: stream/batch equivalence, startup, boundaries, tails, filtering, future perturbation; seven sensor regressions pass |

**Tolerance history.** Initial random-input actual-model test used2×10⁻⁵6D tolerance; one single-frame/batched CPU packed-LSTM difference6.336×10⁻⁵ failed. Final cross-batch-shape tolerance is explicitly10⁻⁴. With batching fixed, future-perturbed historical outputs must and do match bitwise. The231 actual-DIP reconstructions retain2×10⁻⁵ threshold, with much smaller observed maxima. Original run sources remain; amended tests/logs are separate validation snapshots, not overwritten execution evidence.

CPU imuposer runtime about 765 s, peak RSS1,111 MiB: offline batch resources, not device latency. All 77 original [ARTIFACT_MANIFEST.json](../outputs/local_dip/window_comparison_epoch59_20261113/ARTIFACT_MANIFEST.json) files reverified. [Independent conclusions](../outputs/local_dip/window_comparison_epoch59_20261113/independent_validation.json); follow-up artifacts/final tests have separate `VALIDATION_MANIFEST.json`.

## Commands

Use a nonexistent output directory for reruns.

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tools/compare_window_inference.py \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --manifest outputs/local_dip/wrist_pocket_dev_epoch59_20261110/split_manifest.json \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --output outputs/local_dip/window_comparison_epoch59_rerun \
  --device cpu --threads 2 --batch-size 16
```

`--sequences` accepts frozen-development IDs only; omit for all 11. Existing licensed local assets suffice; no training-dependency changes.

```bash
conda run --no-capture-output -n imuposer python tools/validate_window_comparison.py \
  --run outputs/local_dip/window_comparison_epoch59_20261113 \
  --legacy-cache outputs/local_dip/dual_wrist_dev_epoch59_20261110/evaluation/predictions.npz \
  --output /tmp/window_comparison_validation_recheck.json

IMUPOSER_TEST_CHECKPOINT='/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  conda run --no-capture-output -n imuposer python -m unittest discover \
  -s tests -p test_window_inference.py -v
```
