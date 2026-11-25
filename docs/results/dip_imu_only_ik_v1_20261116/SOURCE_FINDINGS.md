# IK and evaluation conventions: pre-execution evidence and decisions

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](../../RECORD_LANGUAGE.md).

Work date: 2026-11-16, America/Chicago. Saved before any new test-set IK results.

- IMUPoser v1 §4.4: constrain bones with absolute IMU orientation, MSE+Adam, 10 iterations/frame; wrist input optimizes shoulder/elbow. IK learning rate, complete head/hip indices, parameterization, state reset and target projection are unspecified. https://arxiv.org/html/2304.12518v1#S4.SS4
- FIGLAB/IMUPoser public directories/Python at `a0e9d33b9c57d6737d711185b62084e17684b0e2` lack the corresponding pose optimizer/paper evaluation loop. Math inverse_kinematics_R only converts global/local rotations. https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2
- Public preprocessing `ji_mask=[18,19,1,2,15,0]`; use first five. Raw DIP indices `[7,8,9,10,0,2]`; wrist devices correspond to forearm segments. Old `leaf_joints=[20,21,7,8,12]` does not match these inputs. Issue16 replies are other users' questions, not author confirmation. https://github.com/FIGLAB/IMUPoser/issues/16
- Cited AvatarPoser §3 gives IK lr 1e-3; public models/network.py defaults1e-1, five iterations, and hand-position optimization. Do not import that position objective or velocity-MPJVE definition into IMUPoser's vertex metric. Paper:https://static.siplab.org/papers/eccv2022-avatarposer.pdf ; code commit `0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0`.

Predeclared decision: retain final two-epoch checkpoint, F03 full 24-mask data/predictions, four metrics. Add only ten Adam steps/frame, lr 1e-3,3×3 orientation MSE. Jointly handle available sensors; optimize sensed segments/direct parents. Head/hip indices explicitly reconstructed. Local rotvec increments, FP64 optimize/FP32 output, no extra joint/temporal prior, unoptimized entries retained. Independent initialization/Adam per frame; sum per-frame MSE to preserve Adam-epsilon behavior across batching. Emit final update, no GT calibration/loss, test tuning, best-iterate selection or fallback.

Original orientations use entrywise matrix interpolation and may leave SO(3). Project only IK targets to nearest proper rotation; network inputs/processed files unchanged. First-eight training-pool checks diagnose fixed mapping/conventions without fitting calibration; results in train_conventions.json.

Map24 combinations to12 display classes by wrist/pocket/head counts, ignoring left/right. Keep equal24-mask primary mean; also report equal12 groups, equal recordings, subject reductions. Do not switch primary to whichever is closest to paper. Exact author aggregation, rolling overlap/startup, shape betas unresolved; historical25-mask/corpus/epoch differences retained. Raw/IK share canonical metrics; MPJVE always means mesh-vertex position error.
