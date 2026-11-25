# Framewise IK and Temporal IK in this project

Both methods refine poses produced by the same final DIP network. They adjust selected local joint rotations using available IMU orientations; they do not retrain the network or use ground truth at inference.

| Property | Framewise IK (I01) | Temporal IK (I02) |
|---|---|---|
| Data for output t | Network pose and available orientations at t | Network poses and orientations from max(0,t-4) through t |
| Objective | Match available IMU orientations | Match orientations plus penalize squared third differences of FK joint positions |
| Coupling | Each frame independent | Joint optimization inside each five-frame window |
| Emitted output | Current frame after ten Adam steps | Last frame after ten steps; previous emitted poses are not rewritten |
| State | Fresh Adam per frame | Fresh Adam per window, no state carry |
| Added future access | None | None; five samples span 160 ms of past history at 25 Hz |
| Post-output filter | None | None |
| Relation to paper | Paper-guided reconstruction, with explicit missing author details | Our separate temporal-regularization extension |

Conceptually:

```text
Framewise: minimize orientation error at t
Temporal:  minimize orientation error over the past/current window
           + lambda * joint-trajectory third-difference penalty
```

Framewise correction can switch direction between neighboring frames, adding high-frequency position changes even when each frame matches its measured orientation better. Temporal IK explicitly penalizes rapid changes in the joint trajectory during optimization. It does not simply average or low-pass-filter the final output.

In the completed I02 run, the window is five frames, Adam has ten steps at 0.001, and the regularizer weight is 0.001 selected on development data. The jerk-loss scale is 0.01 m; this optimization loss uses frame-based differences, while reported jitter includes the separate 25³/100 conversion. Short prefixes use only valid samples and complete four-frame stencils. Local degrees of freedom match I01; unoptimized entries remain unchanged.

The neural predictor's 125-frame rolling context is separate from this five-frame IK window. The extra IK stage accesses no future sample, but inherited centered preprocessing still has future dependencies; this does not establish strict causality of the entire sensor pipeline.

## Measured tradeoff on the full test set

Same final weights, all 19 held-out recordings and 24 masks:

| Metric | Network, no IK | Framewise IK | Temporal IK |
|---|---:|---:|---:|
| MPJRE, degrees | 22.46870 | 22.16620 | 22.16886 |
| MPJPE, cm | 9.54019 | 9.42636 | 9.42868 |
| Mesh MPJVE, cm | 11.96889 | 11.82258 | 11.82466 |
| Jitter, 100 m/s³ | 1.04836 | 1.22145 | 1.16742 |

Temporal lowers framewise jitter by 4.42% while retaining 98.58% of its mesh-error gain over the network. Its jitter remains 11.36% above the network. The spatial difference between the two IK methods is small, so static screenshots can look nearly identical; the time-series curves and matched slow replay explain the temporal effects more clearly.

Neither local-window regularization nor a fixed ten-step optimizer guarantees globally optimal smoothness of the emitted sequence. The three replay clips illustrate the existing results and do not select settings or replace the full-test scores.

[Framewise implementation/report](IMU_ONLY_IK.md) · [Temporal experiment](TEMPORAL_IK_EXPERIMENT.md) · [Jitter diagnosis](IK_JITTER_DIAGNOSIS.md) · [Matched replay](FINAL_MODEL_REPLAY.md)
