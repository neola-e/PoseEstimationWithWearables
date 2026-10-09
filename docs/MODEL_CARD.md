# Team IMUPoser model card

Local reproduction of FIGLAB/IMUPoser (CHI 2023), for internal noncommercial research under the unchanged source LICENSE. Not a checkpoint certified by the original authors.

## Architecture and conventions

Original unchanged `IMUPoserModel`/`RNN`: input dropout 0.2, linear 60→512 with ReLU, two-layer bidirectional LSTM with 512 hidden units per direction, linear 1024→144. Inference disables the loss-only SMPL construction, loads network weights strictly, calls `eval()`, and runs in `torch.inference_mode()`.

25 FPS; sensor order left wrist, right wrist, left thigh/pocket, right thigh/pocket, head. Acceleration triples divided by 30 precede row-major global orientation matrices. Missing sensor acceleration and orientation blocks are zero. Outputs are 24 local SMPL rotations in column-based 6D, including global root orientation. Padding is excluded from inference/evaluation outputs using lengths. No translation or shape output. See IMU_INPUT_FORMAT.md.

## Provenance and training

Epoch 55, global step 154056, training validation metric 0.010310699231922626 (filename rounds to 0.01031). Original full checkpoint SHA-256: `a0e3dc1ca51897b2e0af89e7c04bf44e98061ffe008ab4f895816c876597a81b`. Portable inference-only checkpoint SHA-256: `fe697c88e36fddcc1aa556b6b158f1cbf17e403aa5953cfb0fc98ae49ca8fdc3`. All 20 tensor values are identical between formats. Neither checkpoint is committed or publicly published by this change.

Training uses AMASS synthetic IMUs; the global dataset excludes DIP files. The inspected script uses seed 42, a random 90/10 sample split, Adam at 3e-4, and pose MSE plus SMPL joint-position coordinate MSE. The historical loss reduces over padded tensors; the new evaluation masks padding. Random splitting can put correlated windows or sensor masks of one motion in both splits. Corpus versions and a complete training manifest were not independently audited.

## DIP subset check

Prior CPU evaluation used the first 125 frames from each processed test sequence ID 0/1/2: 375 valid frames, all five sensors. Pose MSE 0.0160473606; mean local joint angle 13.5087°; mean global joint angle 12.8796°; root-relative MPJPE 3.25846 cm; joint-coordinate MSE 0.000674497 m². FK uses the same mean male shape, zero translation, all 24 joints including the zero-error root, and no rigid/rotation alignment. See dip_subset_metrics.json.

These are prefix smoke checks, not full-dataset or paper-comparable benchmarks. Pose MSE is not the combined training validation metric. Test preprocessing selects subjects s_09/s_10, but the processed file did not retain subject/motion names.

## Limitations and integration

Bidirectional windows use future frames; training windows are 125 frames. Centered five-frame acceleration smoothing separately adds 80 ms future context. There is no zero-lookahead streaming guarantee or global trajectory estimate. Real sensors introduce domain shift through timing, gravity treatment, orientation direction, heading drift, pocket motion, mounting calibration, body shapes, devices, and unseen actions. Start with synchronized offline measurements, verify units and common global/segment frames, and compare a bounded-lookahead policy against this offline baseline.

CPU is the default. Full AMASS and SMPL assets are unnecessary for inference; evaluation uses separately licensed DIP, and position metrics/visualization use separately licensed SMPL. Do not commit datasets, checkpoints, or restricted assets. The source tree inherited a tracked SMPL model from upstream; this change neither adds nor modifies it, and its presence does not grant additional redistribution rights. Obtain assets through their licensors rather than treating that inherited copy as a license grant.

## Attribution and licensing

Retain the original LICENSE and citations: Mollyn et al., IMUPoser, CHI 2023 (DOI 10.1145/3544548.3581392); Huang et al., Deep Inertial Poser, SIGGRAPH Asia 2018; Loper et al., SMPL, SIGGRAPH Asia 2015; Mahmood et al., AMASS, ICCV 2019, as applicable. Checkpoint distribution remains subject to licensing review. No public model release or data upload is part of this change.
