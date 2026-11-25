# IMUPoser research progress and experiment log

Created: 2026-11-10. Last updated: 2026-11-19. Calendar dates use America/Chicago unless a source specifies UTC. Prepared for Alex with Codex assistance.

This is the running research notebook for reproducing IMUPoser and investigating how wearable sensor placement affects pose estimation. As of 2026-11-11, the priority is paper reproduction: first DIP quantitative evaluation, then consumer-device data and the real-time system. The [reproduction audit](REPRODUCTION_AUDIT.md) defines the scope, implementation gaps, data evidence, and retraining decision. Completed placement studies remain preserved as supplementary analysis.

Entries E00–E03 and the initial E04 proposal were reconstructed from saved reports, protocols, validation records, and our discussion. Their experiment dates are preserved where documented; unknown dates are marked unknown. E04's execution entry was added during the subsequent authorized experiment. Historical summaries do not imply that notes were written before those experiments. Experiment IDs below are notebook references assigned when this log was created.

## Current project status

| Area | Status as of 2026-11-19 | Evidence or next action |
|---|---|---|
| Reproduction scope and Epoch 59 audit | Completed | Audit A01 below; [paper comparison and decision](REPRODUCTION_AUDIT.md) |
| Four pose metrics and numerical checks | Completed | M01 below; [metric conventions and validation](POSE_METRICS.md) |
| Rolling inference and per-output dependencies | Implemented and checked | R01 below; [semantics, smoothing, and chunk comparison](ROLLING_INFERENCE.md); exact author overlap semantics and raw-stream preprocessing remain open |
| DIP fine-tuning protocol and engineering smoke | Frozen and validated | F01 below; [protocol, data identity, gradients, and checkpoint replay](DIP_FINETUNING.md); T02 now records completed development training |
| GPU access and execution route | Verified in ordinary WSL; isolated in Codex | G01 below; [GPU diagnosis](GPU_DIAGNOSTIC.md); user selected ordinary-terminal execution with permissions unchanged |
| Full DIP training launcher | Development and final stages exercised on GPU | T01/T02/F02 below; [WSL startup and resume commands](DIP_TRAINING_RUN.md) |
| DIP development training | Completed on GPU by Alex; final budget K=2 | T02 below; K=2 has now been used by completed F02 final training |
| Final first-eight DIP training | Completed on GPU by Alex | F02 below; 2 epochs, 162 steps, reporting checkpoint locked; F03 now records completed test comparison |
| Paired four-metric test comparison | Completed on CPU and independently validated | F03 below; [full results](DIP_FINETUNING_COMPARISON.md); all24 masks, all19 test recordings; four metrics lower for every mask aggregate |
| IMU-only orientation IK | Implemented, full24-mask evaluation and independent validation complete | I01 below; [implementation and measured tradeoff](IMU_ONLY_IK.md); three errors decrease, jitter increases |
| IK jitter source diagnosis | Completed across all19 recordings,24 masks and24 joints | J01 below; [attribution, rotations, frequency and component counterfactuals](IK_JITTER_DIAGNOSIS.md) |
| Temporal IK extension | Independent new experiment completed and validated | I02 below; [separate protocol and result](TEMPORAL_IK_EXPERIMENT.md); jitter4.42% belowI01,98.58% vertex gain retained, still11.36% above raw |
| Paper evaluation conventions | Audited, with unresolved author details explicitly recorded | [21-item alignment and sensitivity](PAPER_EVALUATION_ALIGNMENT.md); 24-to12 grouping verified; exact IK/shape/window/reduction settings remain uncertain |
| Paper-aligned DIP pipeline | Fine-tuning, rolling, IK and full four-metric baseline exercised | F03/I01 complete the DIP engineering chain; strict author-equivalent numerical reproduction remains unproven |
| CPU inference and visualization | Implemented and validated | [Handoff validation](VALIDATION.md); [model card](MODEL_CARD.md) |
| Final-model four-way offline replay | Three fixed clips rendered and independently validated | V01 below; [animations, numerical checks and interpretation](FINAL_MODEL_REPLAY.md); raw / framewise IK / Temporal IK / GT, same source frames |
| Five versus three locations | Completed; supplementary priority | Experiment E02 below; results preserved |
| Wrist side and pocket side | Completed; supplementary priority | Experiment E03 below; results preserved |
| Head plus both wrists | Completed; supplementary priority | Experiment E04 below; better wrist accuracy with worse leg accuracy |
| Team handoff | Research code/reports recorded in commit8519e9c; V01 additions currently uncommitted | [Team guide](TEAM_HANDOFF.md); original [PR 1](https://github.com/neola-e/PoseEstimationWithWearables/pull/1) history in E01; no new remote-state claim from this visualization task |
| Live consumer-device evaluation | Future work | Offline calibration, timing, and an explicit latency policy are needed |

## How to maintain this notebook

The structure adapts laboratory notebook practice: dated entries, objectives, methods, results, interpretation, and links to supporting records. See the [NCI Guide for Keeping Laboratory Records](https://techtransfer.cancer.gov/investigators/inventions/guide-lab-records) and [HHS ORI laboratory notebook guidance](https://ori.hhs.gov/education/products/wsu/data_lab.html).

1. Add an entry after each experiment, meaningful analysis, failure, or research decision. Record the work date and entry date separately if writing retrospectively, and identify who performed the work.
2. Save a protocol before new inference or training: question, hypothesis, data split, comparisons, primary outcomes, aggregation, and selection rules. Log deviations with their reasons.
3. Keep observations separate from explanations. Record negative results and exceptions, and label exploratory analyses and proposed work.
4. Link the exact command, code revision and any uncommitted source snapshot, environment, seeds, input hashes, metrics, and figures. Keep large artifacts in their run directory.
5. Preserve completed entries. Add dated corrections or follow-ups referencing the original entry instead of silently replacing the history. Refresh the status table and action list as work advances.
6. Keep this text log under version control when committing project work. Preserve run artifacts separately: the linked output directories are ignored by Git, so this file alone is not a backup of the experiments.
7. Use English for research entries, documentation, report/table labels and generated HTML. When translating historical records, preserve original bytes and hashes separately and distinguish the English presentation from original execution evidence; see [record language and provenance](RECORD_LANGUAGE.md).

## Shared experimental context

**Model versions.** The initial inference handoff uses Epoch 55, SHA-256 `a0e3dc1ca51897b2e0af89e7c04bf44e98061ffe008ab4f895816c876597a81b`. The later placement studies use a fixed Epoch 59 checkpoint, SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058`. Epoch 59's recorded training-validation metric is 0.0101725226; that combined training loss is not position error in centimeters. Do not combine the two checkpoints' results into a single benchmark.

**Inputs and inference.** Five supported locations are left wrist, right wrist, left pocket/thigh, right pocket/thigh, and head. Each supplies acceleration and orientation. Missing inputs are zeroed in both feature blocks. Earlier placement studies use CPU inference at 25 FPS in independent windows of at most 125 frames, retaining partial final windows and excluding padding from metrics. The bidirectional LSTM uses future frames inside each window. F03 instead uses explicitly recorded rolling125/stride1/last-frame output and prefix startup; inherited centered preprocessing still has future dependence.

**Metrics.** MPJPE means mean per-joint position error, reported in centimeters. Positions use inherited SMPL forward kinematics, a common mean male shape, zero root translation, and all 24 joints including the zero-error pelvis. Predicted root orientation is retained without rigid alignment. Local rotation measures parent-relative orientation; global rotation includes the ancestor chain. These are offline pose measures, not trajectory or live-device accuracy. Details: [input conventions](IMU_INPUT_FORMAT.md) and [sensor evaluation guide](SENSOR_EVALUATION.md).

**Data separation.** The original test preprocessing selects s_09/s_10. The follow-up reserves s_07/s_08 as development subjects, with identities reconstructed from raw recordings. F01 now enforces s_01–s_06 training and excludes reserved sequence indices 30–40 before mask expansion; its separately frozen final first-eight stage explicitly changes that reservation's role. The inherited loader does not enforce the reservation. Prior test observations already informed the research questions; further generalization claims need independent confirmation. A01 documents the audited initialization's current/historical data evidence.

**Execution provenance.** The evaluated environment imports `imuposer` from the original editable checkout at `/home/alex/research/IMUPoser/src/imuposer`, while study tools are in the team repository. Each study preserves the actual source paths, hashes, and evaluated source snapshot. Reproducing a study requires both sources to match its recorded provenance.

## Experiment and audit index

| ID | Work date | Question or milestone | Status |
|---|---|---|---|
| E00 | Date unknown; before the handoff | Establish a usable reproduction environment | Setup recorded |
| E01 | 2026-10-08 | Can teammates run pretrained inference and inspect outputs on CPU? | Completed and validated |
| E02 | 2026-10-27 | Does the selected three-location improvement generalize across the test split? | Completed and validated |
| E03 | 2026-11-10 | Does the arm penalty follow wrist coverage when pocket side is controlled? | Completed and validated |
| E04 | 2026-11-10 | Is a second wrist more useful than a pocket for the intended application? | Completed and validated; application tradeoff remains open |
| A01 | 2026-11-11 | What remains for paper reproduction, and does Epoch 59 require replacement? | Audit completed; retain weights now and document strict-recipe retraining conditions |
| M01 | 2026-11-13 | Can all four metrics measure the same predictions with explicit conventions and known-pose checks? | Implemented and validated; full paper pipeline remains pending |
| R01 | 2026-11-13 | How do explicit rolling last-frame inference and smoothing compare with independent chunks? | Implemented; all 11 development recordings and per-output dependencies checked |
| F01 | 2026-11-14 | Can Epoch 59 update on DIP, save/reload/replay, with a frozen subject-safe recipe? | Three-step CPU smoke passed; protocol frozen; full training pending |
| G01 | 2026-11-14 | Is CUDA failure a host installation problem or Codex device isolation? | Ordinary-terminal CUDA matmul/autograd and model/FK pass; Codex sandbox hides /dev/dxg |
| T01 | 2026-11-14 | Can Alex launch and resume the frozen full training stages from ordinary WSL? | Launcher ready; ten synthetic engine tests and actual-data read-only check pass; GPU run pending |
| T02 | 2026-11-15 | Which frozen development epoch budget should transfer to final first-eight training? | Thirty GPU epochs verified; best completed epoch2; final K=2, not yet launched in this review |
| F02 | 2026-11-15 | Is the final first-eight two-epoch model complete and ready for locked paired test evaluation? | Final training verified; paired24-mask test protocol frozen; evaluation pending ordinary WSL execution |
| F03 | 2026-11-15 | How do all four test metrics change after the fixed two-epoch DIP fine-tuning? | All456 paired jobs completed; four metrics decrease for all24 mask aggregates; independent validation passed |
| I01 | 2026-11-16 | Does IMU-only IK honor available orientations, and what changes under the remaining paper conventions? | 14 tests plus full456-pair independent validation pass; pose errors decrease around1.2–1.35%, jitter increases16.51%; unknown author settings preserved |
| J01 | 2026-11-16 | Where does I01 extra jitter arise, and is it associated with rotation high frequencies? | All10,944 joint-mask-recording cells verified; pelvis component is dominant; action and mask rankings/plots saved |
| I02 | 2026-11-16 | Can causal Temporal IK preserve spatial gains while reducing extra jitter? | Development-onlyλ0.001 selected; fixed all24 test completed;98.58% vertex gain retained and4.42% lower jitter thanI01, but not below raw |
| V01 | 2026-11-19 | What do the final network, framewise IK and Temporal IK outputs look like against identical-frame GT? | Three10-second four-panel animations plus0.25x jitter replay;750 source frames,1,000 encoded frames, independent pairing/FK/timing audit passed |
| L01 | 2026-11-19 | Can all readable project records be shared in English without losing historical evidence? | English migration of documentation, overview, tables, report generators and historical reading copies; lossless original-byte archive and translation audit |

## E00 Reproduction environment

**Work date:** Unknown. **Entry date:** 2026-11-10, retrospective. **Status:** Setup recorded.

**Objective and outcome.** Make the original project usable on WSL2 with an RTX 5060 Ti. The saved reproduction notes record Python 3.10, PyTorch 2.7, CUDA 12.8, successful GPU detection, imports, and model construction. Compatibility work included NumPy 1.23.5 and installing chumpy without build isolation.

**Evidence and limitation.** The [original reproduction log](../../IMUPoser/notes/reproduction_log.md) records environment setup, but its old “next step” predates the later experiments. The [model card](MODEL_CARD.md) and study provenance provide the later checkpoint and execution records. Historical training corpus and split membership remain incompletely audited; successful setup alone does not establish reproduction of the paper's results.

## E01 CPU inference handoff

**Work date:** 2026-10-08. **Entry date:** 2026-11-10, retrospective. **Status:** Completed and validated.

**Question.** Can a teammate load existing weights and run inference, evaluation, and visualization without retraining or loading AMASS?

**Method.** Use the Epoch 55 checkpoint in a fresh Python 3.10 CPU environment. Check original and portable checkpoint loading, input/output formats, variable lengths, a synthetic interface example, three DIP sequence prefixes, and a predicted-versus-reference GIF.

**Observed results.** All 13 tests passed with optional licensed assets; without those assets, nine passed and four explicitly skipped. Original and portable formats produced identical network tensors and synthetic predictions. The 375-frame DIP prefix check gave MPJPE **3.2585 cm** and local rotation error **13.5087 degrees**. The 125-frame GIF decoded successfully. Peak process memory for the fresh workflow was approximately **540 MiB**.

**Interpretation.** The inference interface and bounded evaluation workflow work on CPU. Three short prefixes are insufficient to estimate full-test accuracy, and synthetic inputs only test the interface.

**Project outcome.** The handoff became commit `8f9d9c5397185853b103c50ef6a81d615d0e0d0e` and [PR 1](https://github.com/neola-e/PoseEstimationWithWearables/pull/1), created 2026-10-09 UTC. Later studies use Epoch 59 rather than this handoff checkpoint.

**Evidence.** [Exact validation commands and outcomes](VALIDATION.md), [checkpoint provenance](MODEL_CARD.md), and [prefix metrics](dip_subset_metrics.json). The validation document's “not pushed” wording describes its historical validation stage; the PR link records the later handoff state.

## E02 Five locations versus three locations

**Work date:** 2026-10-27. **Entry date:** 2026-11-10, retrospective. **Status:** Completed and validated.

**Motivation and question.** A previously selected high-error window from Sequence 18 improved when two inputs were removed. Did that observation generalize across the full processed DIP test split?

**Comparison and method.** Keep Epoch 59 fixed. Pair all five locations (`global`) against left wrist + right pocket + head (`lw_rp_h`) on identical reference poses and windows. Evaluate all **19 sequences, 203 windows, and 24,078 valid frames** per configuration. Primary summaries weight valid frames; sequence-equal summaries check sensitivity. No training or checkpoint search is part of this comparison.

**Observed results.**

| Outcome | Five locations | Three locations | Change with three locations |
|---|---:|---:|---:|
| Whole-body MPJPE | 7.8866 cm | 9.4112 cm | +1.5246 cm, or +19.33% |
| Mean global rotation error | 20.9495 degrees | 24.8507 degrees | +3.9012 degrees |
| Right-arm position error | 12.81 cm | 19.95 cm | Worsened in all 19 sequences |

Three locations improved **32/203 windows**, but only **1/19 complete sequences**. Sequence 18's selected window improved from **19.44 to 13.20 cm**; this was the largest improvement among all 203 windows.

**Interpretation.** The selected improvement was real but exceptional. This fixed five-location baseline performs better overall. The dominant penalty occurs in the right arm, whose direct input was removed. Because the right wrist and left pocket disappeared together, this experiment could not separate wrist coverage from pocket effects or isolate sensor count alone.

**Follow-up.** Compare wrist side and pocket side at a fixed count of three locations on reserved development subjects. This question became E03.

**Validation and evidence.** Seven infrastructure tests passed. Independent recomputation of all **406 configuration-window predictions** matched saved joint metrics exactly. [Report](../outputs/local_dip/sensor_pair_epoch59_20261027/REPORT.md), [validation](../outputs/local_dip/sensor_pair_epoch59_20261027/final_validation.json), [run provenance](../outputs/local_dip/sensor_pair_epoch59_20261027/provenance.json), and [reproduction commands](SENSOR_EVALUATION.md).

## E03 Wrist side and pocket side on development subjects

**Work date:** 2026-11-10. **Entry date:** 2026-11-10, retrospective summary of the completed study. **Status:** Completed and validated.

**Question and saved hypothesis.** Does switching wrist input move the arm-error penalty to the opposite side, and does pocket side change that effect? If direct limb information matters, each arm should improve when its own wrist input is available. This prediction was saved in the run protocol before development inference.

**Design.** Hold head input, Epoch 59, reference poses, windows, and metrics fixed. Evaluate all four left/right wrist × left/right pocket combinations. Reserve subjects **s_07/s_08**, comprising **11 recordings, 25,423 valid frames, and 209 windows per configuration**. Match each recording's processed acceleration, orientation, and pose to raw-data preprocessing bit for bit. Use equal subject weighting as the primary summary. No new test-split inference or training was part of this study.

**Observed placement results.** Head input is present in every row. All values are errors; lower is better.

| Wrist input | Pocket input | Whole-body MPJPE cm | Left wrist position cm | Right wrist position cm |
|---|---|---:|---:|---:|
| Left | Right | 10.15 | 13.80 | 26.19 |
| Right | Right | 9.71 | 23.70 | 13.43 |
| Left | Left | 10.31 | 12.73 | 26.92 |
| Right | Left | 9.70 | 23.85 | 12.71 |

**Direct coverage effects.** Average equally over subjects and pocket choices, comparing an arm with and without its own wrist input.

| Outcome | Own wrist absent | Own wrist present | Relative error reduction |
|---|---:|---:|---:|
| Left wrist position | 23.77 cm | 13.26 cm | 44.2% |
| Right wrist position | 26.56 cm | 13.07 cm | 50.8% |
| Left arm global rotation | 50.66 degrees | 26.20 degrees | 48.3% |
| Right arm global rotation | 50.32 degrees | 24.31 degrees | 51.7% |

**Consistency and exceptions.** The expected arm pattern appeared in both subjects and at both pocket sides. Averaging over pockets within recordings, arm global rotation favored direct wrist coverage in **11/11 recordings on each side**; right-wrist position did so in **11/11**, and left-wrist position in **10/11**. The left-wrist exception shows that position and orientation can move in different directions.

**Pocket findings.** The average right-minus-left pocket effect on whole-body MPJPE was **−0.078 cm**, but was **−0.583 cm for s_07** and **+0.428 cm for s_08**. Right-pocket input lowered leg position error by about **1.00 cm** on average, while other regional outcomes could worsen. A small whole-body average does not imply that pocket placement has no effect.

**Interpretation.** The error penalty follows the missing wrist, supporting missing direct limb information as a leading explanation for E02's right-arm penalty. Whole-body averages conceal this redistribution: switching wrists changes whole-body MPJPE by only about **0.52 cm** on average over pockets. There is no robust overall placement winner, and roughly **13 cm** of wrist-position error remains with direct coverage. These two-subject DIP results do not establish performance for new people or consumer devices. E02 and E03 use different subjects and aggregation rules; their absolute scores do not measure improvement or regression against each other.

**Follow-up.** The report recommends testing head + both wrists to measure the arm-versus-lower-body tradeoff. E04 records that proposal; its results are pending.

**Execution update later on 2026-11-10.** Alex requested the proposed comparison. E04 is now completed; the execution entry below records its results without changing this original E03 interpretation.

**Validation and evidence.** Five new and seven existing infrastructure tests passed. All **836 cached configuration-window predictions** were independently recomputed. Maximum position-metric difference was **0.0000153 cm**, rotation metrics matched exactly, and contrasts matched exactly. [Saved protocol](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/PROTOCOL.json), [split manifest](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/split_manifest.json), [report](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/REPORT.md), [illustrated HTML report](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/REPORT.html), [numerical validation](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/validation.json), [report review](../outputs/local_dip/wrist_pocket_dev_epoch59_20261110/REPORT_REVIEW.json), and [reproduction guide](WRIST_POCKET_DEVELOPMENT.md).

## E04 Head plus both wrists

**Initial proposal and entry date:** 2026-11-10. **Status at proposal:** Proposed; not run. No results or final success thresholds were recorded at that stage. The initial plan is preserved below, followed by the execution entry.

**Question.** With three locations available, does replacing the pocket with the second wrist improve both arms enough to justify the lower-body cost for the intended application?

**Working hypothesis.** `lw_rw_h` may reduce the uninstrumented-arm penalty seen in E03, while losing pocket information may worsen leg pose or body orientation. This is a sensor-allocation tradeoff: adding a wrist and removing a pocket happen together.

**Draft protocol.** Keep Epoch 59 and the same s_07/s_08 reservation, 209 windows, reference hashes, and subject-equal aggregation. Compare the new supported mask `lw_rw_h` against all four E03 configurations. Reuse their validated predictions and run inference only for the new configuration.

**Outcomes to fix before execution.** For an arm-focused application, use mean left/right wrist position error as the primary summary and report the worse wrist separately. Retain both arm global rotation errors, leg position, root rotation, and whole-body MPJPE. Record the application priority and any acceptable leg-error increase before seeing results; without an agreed threshold, report the tradeoff without declaring a winner.

**Completion criteria.** Save the final protocol, obtain complete coverage, verify pairing and unchanged assets, independently recompute cached metrics, report both subjects and every selected outcome, then append results and the resulting research decision to this notebook.

### E04 execution and results

**Work and entry date:** 2026-11-10. **Requested by:** Alex. **Execution and documentation:** Codex assistance. **Status:** Completed and validated. Alex explicitly requested running head + both wrists and updating this log.

**Protocol and execution.** Saved the [E04 protocol](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/PROTOCOL.json) before new inference. Retained Epoch 59, s_07/s_08, all 11 recordings, 25,423 valid frames, and 209 windows. Only `lw_rw_h` received new inference. The four E03 configurations reused their 836 cached window predictions. No training or test-split inference occurred. No application error budget was supplied, so the frozen decision rule reports tradeoffs without a pass/fail declaration.

**Observed results.** Primary summaries give both subjects equal weight. The comparator ranges below cover the four wrist/pocket configurations, each with head input and three total locations.

| Outcome | Four wrist/pocket configurations | Head plus both wrists |
|---|---:|---:|
| Mean left/right wrist position error | 18.28–20.00 cm | 13.25 cm |
| Larger wrist mean position error | 23.70–26.92 cm | 13.45 cm |
| Leg position error | 8.59–9.82 cm | 12.17 cm |
| Root rotation error | 12.06–13.95 degrees | 14.87 degrees |
| Whole-body MPJPE | 9.70–10.31 cm | 9.45 cm |

Mean wrist error decreases by **5.03–6.75 cm (27.5–33.7%)**, while leg error increases by **2.35–3.59 cm** and root error by **0.92–2.81 degrees**, depending on comparator. Whole-body MPJPE improves by **0.25–0.86 cm** under this particular joint average. Left/right wrist errors with both inputs are **13.45/13.06 cm**; left/right arm global errors are **26.26/25.83 degrees**. Whole-body local rotation error slightly worsens to **19.80 degrees** from **19.04–19.61 degrees**, despite better whole-body global rotation error.

**Consistency and exceptions.** Both subjects show lower mean wrist error and higher leg error against all four configurations. Whole-body error on s_07 worsens against the two right-pocket configurations by 0.09/0.38 cm, whereas s_08 improves against all four. Mean/worse wrist errors improve in 11/11 recordings against left-wrist/pocket configurations and 10/11 against right-wrist/pocket configurations. The exception is s_08 `03.pkl`, sequence 38: mean wrist error is 12.16 cm with both wrists versus 9.41/10.16 cm with right wrist + right/left pocket. Already instrumented wrist errors do not improve against every comparator. Frame-weighted and sequence-equal summaries retain the primary aggregate tradeoff directions.

**Interpretation and research decision.** Both wrists largely remove the imbalance caused by leaving one arm without direct input, while the pocket contributes useful lower-body information. Head plus both wrists is a promising candidate for an arm-focused application on these recordings, but it does not dominate leg or root-orientation outcomes. Whether its benefit outweighs the cost remains an application decision. Approximately 13 cm wrist error and 26 degrees arm-orientation error remain. A proposed next diagnostic is to inspect residual torso/shoulder errors and sequence 38 using cached development predictions; it has not been run as a new experiment.

**Validation.** Four new and twelve existing infrastructure tests passed. Independent validation recomputed all **1,045 configuration-window predictions**, checked **20,205 CSV endpoints** and **15,300 paired delta rows**, and verified exact cache reference/window pairing. Maximum joint position difference was 0.0000153 cm; rotation errors matched exactly. All differences were below the 0.0001 metric tolerance. All 46 E03 files, evaluated sources, and asset hashes remained unchanged. New inference plus cache writing took 6.94 seconds with 1,046.59 MiB peak process memory; these are offline workflow measurements.

**Evidence.** [Detailed report](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/REPORT.md), [numerical validation](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/validation.json), [provenance](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/evaluation/provenance.json), [all five configurations](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/analysis/overall.csv), [paired comparisons](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/analysis/comparisons.csv), [tradeoff figure](../outputs/local_dip/dual_wrist_dev_epoch59_20261110/analysis/figures/sensor_allocation_tradeoff.png), and [reproduction guide](DUAL_WRIST_DEVELOPMENT.md).

## A01 Reproduction scope and training data audit

**Work and entry date:** 2026-11-11. **Requested by:** Alex. **Execution and documentation:** Codex assistance. **Status:** Completed. This is an audit and scope decision, with no training or model inference.

**Question.** Define a paper-versus-implementation comparison, trace the Epoch 59 training/data evidence, and decide whether the current weights must be discarded or retrained before further reproduction work.

**Scope decision.** Phase one targets DIP quantitative evaluation under the paper's pipeline. Phase two covers the consumer-device dataset, calibration, device-location handling, and real-time system. Existing sensor studies remain intact and receive lower priority until the baseline is complete. Missing phase-one components are reproducible DIP fine-tuning, clarified rolling-window semantics, IMU-driven rotation refinement, mesh/jitter metrics, and complete configuration aggregation.

**Data observations.** Current global-loader inputs are 18 AMASS processed files: 9,554 sequences, 3,395,598 frames at 25 FPS (37.7289 retained hours), and 31,293 independent windows including short tails. All 14,244 raw candidates were inventoried. Current preprocessing rules reject 4,686 recordings by frame rate and four by length. EKUT's 349 recordings and KIT's 4,232 recordings are all 100 Hz and therefore entirely excluded by the inherited 59/60/120 Hz filter. Other partial exclusions are documented. Accepted raw-recording counts match processed sequence counts for every subset; this is a count check, not raw-content identity matching.

**Preprocessing verification.** Reconstructed acc/ori/pose for all 20 current processed containers from existing intermediate files, giving 28,842 sequence-field checks, all bitwise equal. Processed file SHA-256, per-sequence content hashes, source hashes, and current file order are preserved. Both preprocessing scripts match the run's recorded Git baseline. Historical raw/archive versions and training-time file hashes remain unavailable.

**Training observations.** Epoch 59 has 165,060 steps and validation loss 0.010172522626817226. Recovered 65 epoch validation records directly from the local W&B binary; epoch 59 is the minimum and matches checkpoint callbacks and best_model.txt. The final checkpoint is epoch 64, step 178,815, with EarlyStopping wait_count=5 and stopped_epoch=64. The optimizer state records lr 3e-4, betas .9/.999, eps 1e-8, zero weight decay, and no scheduler. Some important settings, including batch size, scaling, window length, and 90/10 split, are class defaults rather than checkpoint instance fields; seed 42 is supported by source, while exact historical RNG and split indices are not retained.

**Consistency check.** Current enumeration gives 31,293 × 25 = 782,325 samples; the current split/batch rules yield 704,092 train samples, 78,233 validation samples, 2,751 train batches and 306 validation batches per epoch. Multiplying 2,751 by 60 and 65 reproduces both checkpoint step counts exactly. This supports the current inventory but does not certify the historical identity or order of every file.

**Mask finding.** Both current code and the recorded public Git baseline include 24 sparse configurations plus the extra five-location `global`. The extra mask adds one sample per motion window, 4.1667% more total samples than 24 masks, and nominally 4% of presplit exposure. `--combo_id global` sets five supported input slots; it does not restrict training to that mask. The accuracy effect of extra full-input training is unknown without a controlled training comparison.

**Decision.** Retain Epoch 59 as an audited initialization with known recipe differences. First complete evaluation and DIP fine-tuning. A strict paper-training reproduction later requires a separate run with explicit data coverage/version, 24 training masks, training budget, and selection rules; removing `global` only at evaluation or continuing the old checkpoint does not undo prior exposure. Do not select checkpoints or claim reproduction success from agreement with one test metric. Preserve s_07/s_08 for development unless a separately frozen final protocol explicitly changes their role.

**Artifacts and validation.** [Comparison and audit report](REPRODUCTION_AUDIT.md), [audit summary](reproduction_audit_20261111/audit_summary.json), [processed data manifest](reproduction_audit_20261111/processed_data_manifest.csv), [raw inventory](reproduction_audit_20261111/raw_amass_inventory.csv), [checkpoint metadata](reproduction_audit_20261111/checkpoint_metadata.json), [validation history](reproduction_audit_20261111/validation_history.csv), and [collection checks](reproduction_audit_20261111/collection_validation.json). Collection rechecked 31 input hashes and the original source hashes. Only metadata and code were copied; no data/model assets were copied. The added [audit utility](../tools/audit_reproduction.py) is read-only routine tooling; model, loader, preprocessing, and training implementations were unchanged. The W&B SDK's old DataStore reader was unavailable, so a read-only fragment/CRC reader recovered the records without changing dependencies.

## M01 Four metrics and known pose validation

**Work and entry date:** 2026-11-13. **Requested by:** Alex. **Execution and documentation:** Codex assistance. **Status:** Completed. The request was to build trusted MPJRE, MPJPE, mesh MPJVE, and jitter measurements before further reproduction work.

**Implementation.** Added [paper_metrics.py](../tools/paper_metrics.py), [eval_pose_metrics.py](../tools/eval_pose_metrics.py), and [test_paper_metrics.py](../tests/test_paper_metrics.py). Existing metric modules, prior studies, and model/training implementations were preserved. The CLI consumes cached NPZ predictions and paired references, without a checkpoint or new network inference.

**Conventions.** Global joint rotations define MPJRE; local rotation error remains a separate diagnostic. Spatial errors align pelvis translation only, retain root orientation, use all 24 joints including pelvis and all 6,890 vertices, and report centimeters. Prediction and reference share fixed shape; mean-shape SMPL and disabled pose blend shapes are the documented defaults, matching inherited/TransPose defaults rather than asserting an unspecified author flag. Pose blend shapes and fixed betas are explicit options. Jitter uses the unaligned joint trajectory, including supplied translations, third differences times fps cubed, and is reported in both m/s³ and units of 100 m/s³. Reference jerk is separate; it is not subtracted from prediction jitter.

**Boundary and aggregation checks.** Contiguous cached windows are reassembled within a recording; derivatives never cross recordings, gaps, overlaps, or padding. Mesh computation batches do not reset the temporal stencil. A sequence contributes T−3 valid jerk samples; shorter sequences yield null jitter and zero jerk count. Frame-weighted spatial means use T, while frame-weighted jerk uses T−3. Sequence-equal summaries are exported separately.

**Known-pose evidence.** Twenty new tests passed, including seven optional licensed-SMPL tests, and seven existing sensor-metric tests passed. Cases cover zero errors for identical poses, nonzero jerk for identical moving poses, 90-degree global-root versus local rotation effects, leaf rotations that change vertices without changing joints, centimeter conversion, translation alignment, analytic cubic jerk of 6 m/s³, fps-cubed scaling, shape/mesh options, padding, and sequence boundaries.

**Same-prediction verification.** Reused the E04 `lw_rw_h` development cache: 11 recordings, 25,423 frames, and 25,390 jerk samples. FK chunk sizes 32 and 11 gave identical aggregate metrics. An independent NumPy reduction using chunk size 19 checked spatial distances, trace/acos angles, complete-recording third differences, and aggregate denominators, with maximum difference 0.00005493 in the respective metric unit, below 0.0001 tolerance. The old frame-weighted MPJPE differed by approximately 0.000000011 cm; minor angular differences arise from stable atan2 versus legacy trace/acos.

**Measured cache values.** Frame-weighted MPJRE 21.292074 degrees; MPJPE 9.451584 cm; MPJVE 10.257982 cm; predicted jitter 78.290040 m/s³ (0.782900 in units of 100 m/s³); reference jitter 82.186566 m/s³; local rotation error 19.802034 degrees. These are measurement checks on existing offline development predictions, not a new test-set experiment or completed paper benchmark. E04's main report used subject-equal weighting, so its main table is not the same aggregation.

**Outcome and remaining work.** The requested measurement tool is ready. Explicit shape/blendshape choices and dataset/configuration averaging must accompany future reports. Rolling inference, DIP fine-tuning, IMU-only refinement, and all-configuration paper comparisons remain pending; no success claim follows from proximity to a paper score or low jitter alone.

**Evidence.** [Metric guide](POSE_METRICS.md), [protocol](../outputs/local_dip/metric_validation_20261113/PROTOCOL.json), [final measurements](../outputs/local_dip/metric_validation_20261113/metrics_chunk32_checked.json), [independent validation](../outputs/local_dip/metric_validation_20261113/validation.json), and [validation code](../outputs/local_dip/metric_validation_20261113/validate_metrics.py). No checkpoint/data asset or previous experiment output was overwritten.

## R01 Rolling inference and per output dependency audit

**Work and entry date:** 2026-11-13. **Requested by:** Alex. **Execution and documentation:** Codex assistance. **Status:** Completed. The objective was to implement explicit rolling-window inference, track the inputs for each output, and measure the difference from the previous independent 125-frame chunks.

**Frozen protocol.** The [saved protocol](../outputs/local_dip/window_comparison_epoch59_20261113/PROTOCOL.json) precedes new prediction. It retains Epoch 59, SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058`, and the exact s_07/s_08 development reservation: 11 complete recordings, 25,423 frames. The mask is fixed to `lw_rw_h`. No test data, model training, IK, sensor search, or checkpoint selection was used. The three comparisons are chunk + inherited centered acceleration mean, rolling last-frame + centered mean, and rolling last-frame + trailing five-frame mean. Other preprocessing and model inputs are held fixed.

**Window convention.** A rolling output at t uses `[max(0,t-124),t+1)` and takes only the last valid output. Stride is one frame, so adjacent full windows overlap by 124 frames. Startup uses available prefixes of length 1–125, without future padding, frame repetition, or discarded outputs. Both LSTM directions restart for every window; recordings do not share state. A bounded `RollingPoseStream` supports one-frame pushes, and independent-window batching is checked against that interface. This is a documented interpretation of online history, not recovered proof of the author's ambiguous “1-sample overlap” wording.

**Smoothing and causality.** Original acceleration smoothing is centered at t with up to two future 25 Hz samples, a nominal 80 ms availability delay. Trailing smoothing uses t−4 through t, with shortened startup averages. The trailing version is reconstructed before smoothing from hash-verified local raw recordings; centered reconstruction is bitwise equal to the existing processed acceleration for all recordings. Orientations and reference poses are unchanged. Existing bidirectional NaN filling and interpolation still precede this stage, so the trailing variant is causal at the resampled-input stage, not a certified raw-device pipeline.

**Results.** All variants use the same M01 body, alignment, mesh, sequence-boundary, and averaging conventions. Spatial means are frame weighted and jitter is weighted by T−3 samples per recording, 25,390 samples total.

| Variant | Global MPJRE ° | MPJPE cm | Mesh MPJVE cm | Jitter, 100 m/s³ |
|---|---:|---:|---:|---:|
| Chunk + centered mean | 21.292074 | 9.451584 | 10.257982 | 0.782900 |
| Rolling last + centered mean | 22.068137 | 9.811889 | 10.749727 | 0.859217 |
| Rolling last + trailing mean | 22.526798 | 10.149600 | 11.133277 | 0.892934 |

**Interpretation.** Rolling increases aggregate MPJVE by 0.491745 cm (4.79%) and jitter by 9.75% under the same smoothing. Only two recordings improve in MPJVE, despite improvement in the first completed recording. The mean local angular difference between paired chunk and rolling predictions is 7.082630 degrees. Changing to trailing smoothing adds another 0.383550 cm (3.57%) MPJVE. These observations establish a material inference-protocol difference on this initialization and development sample. The comparison also changes context placement and output selection, so the effect is not attributable solely to removing future input. Trailing smoothing changes the input distribution seen by a model trained with centered data; the result does not establish that all causally trained models will suffer the same cost.

**Verification.** All 76,269 trace rows pass independent interval reconstruction without calling the trace/window planning helpers. There are 231 real-network single-window reconstructions; their largest absolute 6D difference is 1.222e-6. All paired references match the frozen E04 cache bitwise, and recomputed chunk predictions differ by at most 3.368e-6. Chunk output uses model futures for 25,214 frames, up to 124 frames; centered filtering increases maximum pre-smoothing lookahead to 126. Rolling model lookahead is zero; centered filtering still uses futures for 25,412 outputs, up to two frames. Trailing has zero future dependency at that 25 Hz stage. Nine new analytic/integration tests and seven existing sensor-metric tests pass. All 77 original run artifact hashes pass independent verification.

**Numerical exception and correction.** The first random-input CPU checkpoint integration check failed its 2e-5 singleton-versus-batch 6D tolerance; the measured maximum was 6.336e-5. The final test uses 1e-4 for this change of packed-LSTM batch shape. The causal perturbation test now keeps batch partition fixed and requires bitwise-identical past outputs, which passes. The 231 actual-DIP spot checks retain their original 2e-5 limit. Original run source snapshots and later test/validator snapshots are separate and retained. No model or inference algorithm change was required for this tolerance adjustment.

**Runtime and evidence.** CPU in the existing `imuposer` environment; 765 seconds overall and peak RSS 1,111 MiB. This is batched offline throughput, not live-device latency. [Guide and commands](ROLLING_INFERENCE.md), [metrics and per-recording results](../outputs/local_dip/window_comparison_epoch59_20261113/metrics.json), [real-network spot checks](../outputs/local_dip/window_comparison_epoch59_20261113/validation.json), [independent audit](../outputs/local_dip/window_comparison_epoch59_20261113/independent_validation.json), [original artifact manifest](../outputs/local_dip/window_comparison_epoch59_20261113/ARTIFACT_MANIFEST.json), and final `VALIDATION_MANIFEST.json` preserve the evidence. CSV files in the run directory give one dependency row per output.

**Decision and remaining work.** The requested explicit rolling implementation and chunk-versus-rolling measurement are ready. Keep offline chunk and rolling numbers labeled separately; retain both smoothing options rather than silently adding 80 ms lookahead to an online claim. Author window/edge details, raw-stream causal preprocessing, DIP fine-tuning, IMU-only refinement, and all-configuration evaluation remain separate reproduction tasks. This development comparison does not reproduce the paper's full test benchmark.

## F01 Frozen DIP fine tuning protocol and checkpoint smoke

**Work and entry date:** 2026-11-14, America/Chicago. **Requested by:** Alex. **Execution and documentation:** Codex assistance. **Status:** Completed for the requested protocol freeze and engineering short run. No full development or final training was launched.

**Protocol before gradients.** `DIP_FINETUNING_V1` was frozen at 2026-11-14 06:41:54 UTC; the smoke execution started at 06:42:07 UTC. Its SHA-256 is `79bde74ead3f596c5806e3fa1d0cd5b07fcfb6260b0c9e81c218771ba50441ad`. The [versioned protocol mirror](protocols/dip_finetuning_v1_20261114/PROTOCOL.json), [data manifest](protocols/dip_finetuning_v1_20261114/data_manifest.json), and [source hashes](protocols/dip_finetuning_v1_20261114/SOURCE_HASHES.json) are bitwise copies of the frozen run metadata. Twenty-nine source files, including the actually imported original package, are snapshotted in the run directory. Original source/data/model assets and previous experiment outputs are preserved.

**Frozen recipe and selection.** Start from audited Epoch 59 with fresh Adam, lr 3e-4, betas .9/.999, eps 1e-8, no weight decay, scheduler, clipping, AMP, or TF32. Seed Python/NumPy/PyTorch with 42 and enable deterministic algorithms. Development training uses batch 256, at most 30 epochs without early stopping, all 24 sparse masks, and the inherited non-overlapping 125-frame training windows with tails. Loss remains 6D MSE plus SMPL FK joint MSE, weights 1:1, including public-code padding terms; scientific reporting excludes padding. Choose the smallest full-development validation combined loss, weighting batch means by padded B×T and resolving exact ties by earlier epoch. Never select from training loss or test metrics. The 30-epoch cap, internal split, seed, and selection policy are explicit reproduction choices rather than recovered author fine-tuning settings.

**Identity and separation.** All 41 processed recordings from s_01–s_08 were uniquely matched by bitwise raw-reconstructed acc/ori/pose. Of 42 raw candidates, s_01/02.pkl fails the inherited finite-value criterion and is listed with its hash. The original s_07/s_08 reservation is unchanged. Split subjects before mask expansion: s_01–s_06 give 30 recordings, 80,070 frames, 655 motion windows and 15,720 masked training samples; s_07/s_08 give 11 recordings, 25,423 frames, 209 windows and 5,016 validation samples. Formal batch256 would give 62 train and 20 validation batches per epoch. Test containers and s_09/s_10 raw recordings were not opened in this task.

**Future final stage.** After development selection fixes K completed epochs, reset to the same Epoch 59 and fresh optimizer/seed and train all first eight participants for exactly K epochs. This role change for s_07/s_08 is separately recorded; they cease to be independent validation for that final model. The combined first-eight pool has 864 windows, 20,736 masked samples and 81 batches per epoch at 256. The final checkpoint is the end of the fixed budget; s_09/s_10 do not select it. Pre/post reporting uses the same rolling last-frame convention, centered filter with disclosed lookahead, SMPL settings, all 24 masks, and four metrics; mask-level spatial means are frame weighted, jitter is T−3 weighted, and mask means are then averaged equally. IK remains separate. The initialization's historical 25-mask exposure remains a known difference.

**Actual smoke.** Loaded the original Epoch 59 and ran three Adam updates on CPU with 2 threads, batch8, using 24 varied training-window samples that cover every sparse mask exactly once and include a short tail. CUDA was unavailable to this session. Validation loader checks use only reserved examples and do not run gradients or choose a model. The short-run batch and budget are explicitly distinct from the future formal recipe.

| Optimizer step | Pose MSE | FK joint MSE, m² | Combined loss | Valid / padding frames |
|---|---:|---:|---:|---:|
| 1 | 0.02555345 | 0.00416430 | 0.02971774 | 951 / 49 |
| 2 | 0.02348259 | 0.01085796 | 0.03434055 | 839 / 161 |
| 3 | 0.02980795 | 0.01268087 | 0.04248882 | 1,000 / 0 |

All 20 parameter tensors have nonzero finite gradients and change at every step. Relative to initialization all 20 tensors change, with maximum absolute change 0.000901447. The rows use different motions/masks and dropout, so their loss order is not a convergence result.

**Checkpoint and continuation.** Saved `smoke/smoke_step3.ckpt`, SHA-256 `10f9656573fb41c522bd5ca80c65d2830a24fe3e391e98de2590fd5802afda7c`. Model and all Adam states/counters reload bitwise; every Adam counter is 3. Cached probe predictions have zero difference after reload. From identical RNG and the same prescribed continuation input, one uninterrupted update and one reloaded update produce bitwise-identical model weights, optimizer states, RNG, and loss. These two extra probe updates are verification branches; the saved step3 artifact is unchanged. Metadata says `engineering_smoke_only`, `selection_eligible=false`, `epoch=-1`, and zero completed formal epochs.

**Independent verification and tests.** A fresh process reconstructs eight probe features and targets directly from processed data without the new dataset/collate/loss helpers. Lengths and targets match bitwise; initial and saved-checkpoint predictions each have zero difference against their corresponding cache. It also checks all 20 Adam moment states, step counts, recipe metadata, subject isolation, and artifact hashes. Six unit tests pass, covering all sparse masks and tails against the inherited public loader, pre-mask subject separation, feature immutability, analytic padding loss, RNG replay, and state-difference rejection. No failures or protocol deviations were observed.

**Runtime and evidence.** The smoke itself took 6.27 seconds, peak RSS 1,741 MiB, excluding raw identity mapping. This does not estimate full-batch GPU training speed. [Guide](DIP_FINETUNING.md), [run configuration](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/RUN.json), [gradient/update records](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/steps.json), [results](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/results.json), and [independent validation](../outputs/local_dip/dip_finetuning_v1_20261114/independent_validation.json) preserve the evidence. The run's freeze manifest covers 32 files and the smoke manifest covers five artifacts; final logs and validation snapshot are sealed in the root `ARTIFACT_MANIFEST.json`.

**Conclusion and next action.** Data loading, gradients, optimizer updates, save/reload, and controlled continuation are verified; the requested future training settings are frozen. Full development/final training and accuracy comparison remain unexecuted. Before GPU execution, resolve the current session's CUDA visibility and implement/run the full epoch/validation loop against this frozen recipe. Training loss alone is not a paper-reproduction success criterion.

## G01 GPU visibility and CUDA execution route

**Work and entry date:** 2026-11-14. **Requested by:** Alex. **Execution:** Codex collected sandbox diagnostics and prepared the read-only preflight; Alex ran the identical script in an ordinary WSL terminal. **Status:** Host GPU computation verified and root cause diagnosed; the user chose to preserve current Codex permissions.

**Environment and comparison.** Both reports identify the same lyrixlegion WSL2 kernel, Ubuntu 26.04.1, Python 3.10.21 from the `imuposer` environment, PyTorch 2.7.0+cu128, and the same diagnostic script hash. CUDA_VISIBLE_DEVICES and related device-visibility variables are unset. In the ordinary terminal `/dev` is devtmpfs and `/dev/dxg` is present/readable/writable, with Seccomp=0. In Codex a private minimal tmpfs replaces `/dev`, dxg is absent, NoNewPrivs=1 and Seccomp=2. Its nvidia-smi fails with OS GPU-access blocking, Torch returns False/0, and driver cuInit returns 100. This directly locates the failure in the sandbox device boundary rather than an absent CUDA build.

**Ordinary-terminal hardware and software.** NVIDIA GeForce RTX 5060 Ti, 8,151 MiB reported memory, compute capability 12.0. Driver query returns 610.60; NVIDIA-SMI reports tool 610.47.03, KMD 610.60, CUDA UMD 13.3. Torch uses build CUDA12.8, reports one available GPU, and includes sm_120 in its architectures. Driver and framework version strings describe different components; actual computation verifies this combination works.

**Actual GPU tests.** The [new diagnostic](../tools/diagnose_cuda.py) creates a 256×256 matrix on cuda:0, multiplies by identity, checks values/device, and runs autograd with finite cuda:0 gradients. It loads the original Epoch59 with explicit cuda:0, forwards eight synthetic masked IMU frames, and evaluates SMPL FK on GPU. Parameters, network output and FK joints are all cuda:0; network shape is `[1,8,144]`. No DIP loader, optimizer step, parameter update, checkpoint write, or repeated CPU smoke occurs. The ordinary report is `passed_cuda_and_model`; the sandbox report is `blocked_cuda_unavailable` with the expected preflight exit2.

**Permissions and selected resolution.** Alex explicitly chose ordinary WSL execution and no permission change. A read-only CUDA query was also attempted using its historical command authorization and require_escalated; the platform rejected it before execution because granular sandbox_approval=false. That historical allowance cannot override this session restriction. No settings were changed and no alternative mechanism was used to bypass the boundary. Under the selected workflow, prepare/review code and read shared reports in Codex, and execute GPU jobs in the ordinary WSL terminal with explicit cuda:0. Direct Codex execution remains unavailable until a separately consented client policy allows per-command sandbox-execution approval. There is no evidence requiring driver, Torch, or CUDA reinstallation.

**Resource observation.** Before this preflight's GPU compute began, ordinary nvidia-smi already showed 6,472/8,151 MiB used and 97% utilization. The GPU's existing workload source was not established, and no process was terminated. Basic CUDA and model-device prerequisites pass, but eight-frame inference does not establish that frozen batch256 training fits the available memory. Recheck resource availability before launching the full trainer; do not silently change the frozen recipe.

**Integrity and artifacts.** Protocol copies, original Epoch59 and the saved smoke checkpoint have identical hashes across both reports. All 47 frozen F01 artifacts and 29 frozen source files were rechecked unchanged. [Diagnostic guide](GPU_DIAGNOSTIC.md), [sandbox report](../outputs/local_dip/gpu_diagnostic_20261114/codex_sandbox.json), [ordinary-terminal report](../outputs/local_dip/gpu_diagnostic_20261114/terminal.json), and [paired checks](../outputs/local_dip/gpu_diagnostic_20261114/comparison.json) preserve evidence. No formal training was launched. The final diagnostic artifact manifest and source snapshot are kept in that run directory.

**Next action.** Implement/run the frozen full epoch/validation loop from ordinary WSL with explicit CUDA selection, after checking memory availability. This resolves the execution route while keeping Codex isolated; it does not claim a GPU training run has already completed.

## T01 Full DIP training launcher handoff

**Work and entry date:** 2026-11-14. **Requested by:** Alex, who will launch the command in an ordinary WSL terminal. **Implementation and validation:** Codex assistance. **Status:** Training code and commands prepared; formal IMUPoser training was not executed by Codex.

**Implementation.** Added [train_dip_finetuning.py](../tools/train_dip_finetuning.py), [engine tests](../tests/test_train_dip_finetuning.py), and [startup guide](DIP_TRAINING_RUN.md). Original F01 protocol, source helpers, datasets, Epoch59 and smoke artifacts are preserved. The launcher loads the frozen protocol and rejects changed source/data hashes. CUDA is mandatory for actual CLI training; configuration-only checking can run on CPU without model construction or output-directory creation. Batch, lr, seed, precision and epoch budget are not user-overridable CLI knobs.

**Stages.** Development runs exactly the frozen 30-epoch cap without early stopping, on s01–s06 with s07/s08 validation. Each validation batch mean is weighted by padded B×T; smaller combined validation loss wins, and exact ties retain the earlier epoch. The separate final command requires completed development evidence, verifies the selected K completed epochs, resets to the original AMASS Epoch59 with fresh Adam/seed, and trains all first eight for exactly K epochs. Test data is never opened by either training stage, and final training does not begin automatically after development.

**State and outputs.** Each completed epoch writes a unique immutable checkpoint. Atomic STATE.json points to the canonical last/best files; best.ckpt and last.ckpt are regenerable symbolic links. Checkpoints include Adam, RNG, train/validation loader generators, epoch/step, selection state and full numerical history. Resume restores only committed complete epochs; it ignores uncommitted files and regenerates views, avoiding duplicate history. A process lock prevents concurrent writers. Original checkpoint bytes are not overwritten or deleted. GPU OOM/nonfinite errors stop with logs rather than modifying the recipe; partial epochs are not presented as completed.

**Validation.** Ten tiny synthetic CPU-engine tests pass, including weighted validation reduction, no-gradient/eval validation, best-epoch ties, nonfinite rejection, CUDA-only startup, rejection of an engineering-smoke resume directory without modifying it, immutable checkpoint/hashes, recovery from missing links/orphan files, final-stage K selection, and exact shuffled/dropout/Adam continuation across an epoch boundary. This is engine testing, not a repeat of real Epoch59 CPU smoke. The actual frozen-data --check-only invocation passes with 15,720 training and 5,016 validation samples, 62 training batches/epoch, batch256 and 30 epochs; training_started=false and no intended training directory created.

**Handoff and limits.** Proposed new development output is `outputs/local_dip/dip_dev_v1`; commands for startup, --resume, and final first-eight training are in the guide. Logs include per-batch progress and epoch losses, and history.csv/JSON retain full reductions. GPU tensor/model forward support was established by G01, but the complete launcher's GPU backward/full-batch memory behavior has not been executed in this sandbox. Check the pre-existing GPU workload before launching. Four-metric test reporting and IK remain later work. Validation logs and final code snapshots are retained in `outputs/local_dip/dip_training_launcher_validation_20261114/`.

## T02 Completed DIP development training and fixed final budget

**Work and entry date:** 2026-11-15. **Training execution:** Alex in ordinary WSL. **Integrity/result review:** Codex assistance. **Status:** Development run completed; final training remains pending. Run ID `58592e9dbc11470f9990bacb20d6aa9f`, directory `outputs/local_dip/dip_dev_v1`.

**Observed execution.** RTX5060Ti cuda:0 completed all 30 frozen development epochs, batch256, all24 masks, with 62 updates per epoch and 1,860 total steps. s01–s06 trained, s07/s08 validated; no test data opened. Run start 01:56:51 and completion 02:12:20 America/Chicago, about 15m29s wall time. Recorded train+validation compute totals 921.19s. Peak Torch GPU allocated is 2,906.05 MiB and peak process RSS 2,147.19 MiB. Startup free GPU memory was 6,887 MiB. The full-batch CUDA backward/update path is now exercised successfully under those resource conditions, superseding the execution uncertainty at T01 handoff.

**Selection.** The minimum full-development validation combined loss is 0.019974888814836028 at completed epoch2. Training loss at epochs1/2/30 is 0.02166910/0.01420470/0.00346574. Validation loss at epochs1/2/30 is 0.02041846/0.01997489/0.02144157. The last validation value is 7.34% above the minimum. Continued training fits the training pool without improving the reserved subjects after epoch2, consistent with an overfitting tendency; this is not a four-metric accuracy conclusion or proof of a single cause.

**Integrity review.** Verified all 30 history entries, finite losses, sequential step counts, train/validation valid-frame exposure, all31 execution source hashes, frozen protocol and original initialization hashes, and best/last checkpoint file hashes. The last checkpoint's full embedded history equals the external history, and all20 Adam counters equal 1,860. Best is `epoch_002_03a6585efa03.ckpt`, SHA `4515e13ae95ee5ec49ff90d941c81149ec4c583ba293aadf21abe17de699ba16`; last is `epoch_030_6da9e5d52b29.ckpt`, SHA `298e8580decef1ca2ee9b08370a714116f336a5d652b8919770e30000aa21501`. No checkpoint or training record was altered during review.

**Decision fixed by protocol.** Transfer K=2 completed epochs to the final first-eight stage. The actual final --check-only command passes: 20,736 mask samples, 81 batches/epoch, two epochs, expected162 total updates, no validation/test loading, no training started. Start from original AMASS Epoch59 with fresh Adam/seed42, as predeclared. s07/s08 become training data only for this final model. Test scores must not change K or select a different development checkpoint. No new training was launched by the reviewer.

**Evidence and remaining work.** [Results, learning curve and final command](DIP_DEVELOPMENT_RESULTS.md), [verification metadata](results/dip_dev_v1_20261115/verification.json), [history mirror](results/dip_dev_v1_20261115/history.csv), and [read-only final-stage plan](results/dip_dev_v1_20261115/final_stage_check.json). Review artifacts are preserved in `outputs/local_dip/dip_dev_review_20261115/`. Next execute final two-epoch first-eight training in ordinary WSL, then compare original and final checkpoints using the same rolling/four-metric test protocol. Loss numbers alone do not establish paper reproduction.

## F02 Final first eight training and paired evaluation preparation

**Work/entry date:** 2026-11-15. **Training execution:** Alex in ordinary WSL. **Review and evaluation tooling:** Codex assistance. **Status:** Final training completed and verified; paired test measurement prepared, no complete comparison result yet.

**Final model.** Run `dip_final_v1`, ID `448a01bc15f742b4be28ddbe4aab0e1f`, completes the predeclared two-epoch budget with all first-eight subjects, 20,736 masked samples, batch256, 81 updates/epoch and162 updates total. Both epochs use cuda:0 on RTX5060Ti. Combined training loss is 0.02032563 then0.01377506; there is no validation loss in this stage. The checkpoint is eligible for final reporting, records original AMASS epoch59/step165060, and all20 Adam counters equal162. All31 training execution source hashes match. Final checkpoint SHA-256 is `996a3e2df21ac675df8f777d975d0700f3836c1d835bb898044a714e6cb5f7d8`. No test data was opened during training.

**Fixed comparison.** New [comparison runner](../tools/compare_dip_finetuning.py) locks the original Epoch59 and the immutable final checkpoint before predictions. It exact-matches all19 test recordings to raw s09/s10: nine from s09, ten from s10, 24,078 frames total. The processed container matches the A01 audited SHA. Each model uses all24 sparse masks, rolling125/stride1/last output/prefix startup, existing centered acceleration filter, FP32 with TF32 disabled, no IK. Canonical unchanged M01 CPU metrics use shared mean male shape, pelvis translation-only spatial alignment, all24 joints/all6890 vertices, zero trajectory translation, and per-recording jerk. Spatial averages are frame weighted and jerk T−3 weighted within masks, then mask means are averaged equally. Each mask has24,021 jerk samples. There are456 paired mask-recording jobs and1,155,744 predicted output frames across both models.

**Execution route and tests.** Current Codex still cannot access CUDA. Under the user's prior unchanged-permissions choice, the prepared command runs GPU inference in ordinary WSL, with two spawned CPU metric workers. Caches and metric rows are resumable and linked to protocol/input hashes. Six aggregation/coverage tests pass; the actual SMPL metric worker also returns zero on an eight-frame synthetic identical-pose comparison. No model training or test-driven checkpoint selection is performed by this evaluation.

**Evidence and next action.** [Comparison guide](DIP_FINETUNING_COMPARISON.md), [frozen evaluation protocol](../outputs/local_dip/dip_finetune_comparison_v1/PROTOCOL.json), and [final training review](../outputs/local_dip/dip_finetune_comparison_v1/FINAL_TRAINING_REVIEW.json). Await ordinary-terminal execution and full result files, then independently check all paired caches, recompute aggregation, perform fixed CPU metric spot checks, and report actual pre/post differences. Earlier chunk or single-mask development values are not substituted for this pending full comparison.

**Execution follow-up.** With no ordinary-terminal prediction progress yet available, started an explicit CPU-only fallback in a separate `dip_finetune_comparison_cpu_v1` directory, preserving the original GPU comparison protocol and both checkpoints. Two parts use inference threads8/batch64 and canonical metric threads2. Both models share the same device/batching and numerical conventions, and a separately frozen evaluation protocol records CPU execution. This changes computation placement, not model training or metric definitions. CPU parts pause at a recording boundary if the preferred GPU pipeline starts producing predictions. Only a complete paired run will supply reported aggregate outcomes; no partial metric table is being treated as final.

**Completed follow-up, 2026-11-15.** The separately frozen CPU fallback has now completed and passed independent validation; F03 below supersedes the pending execution status above. Alex later reported ordinary-terminal launch, but the supplied terminal output was the already verified two-epoch training log and the prepared GPU evaluation directory still had no predictions at the final report check. No partial GPU output is mixed into the complete CPU result.

## F03 Paired four-metric test comparison after fixed two-epoch DIP fine-tuning

**Work and entry date:** 2026-11-15, America/Chicago. **Requested by:** Alex. **Training execution:** Alex in ordinary WSL on RTX5060Ti. **Evaluation and integrity review:** Codex assistance, CPU. **Status:** Complete and independently validated.

**Question and frozen comparison.** Does the final first-eight K=2 model improve the four predefined test measures relative to the original AMASS Epoch59? The checkpoints, subjects, all24 sparse masks, input processing, rolling window semantics, metric definitions and aggregation were fixed before predictions. No optimizer update, training-protocol change, test-driven checkpoint selection or budget change was performed. F02 records both model hashes and the162-step final training evidence.

**Actual execution.** Completed all456 mask-recording pairs, comprising two models ×24 masks ×24,078 frames =1,155,744 predicted output frames. s09 has9 recordings and s10 has10. Every processed reference exactly matched raw preprocessing. Both models used CPU FP32, batch64, rolling125/stride1/last output/prefix startup; two parts ran concurrently, each with8 inference threads and2 canonical metric threads. Elapsed times were72.2 and72.8 minutes. The preferred ordinary-terminal GPU route remains a separate run; no result mixing occurred. Device placement was explicitly frozen in the separate CPU evaluation protocol; training itself used GPU.

**Observed primary results.** Within masks, spatial errors weight frames and jitter weights T−3; the final summary averages all24 mask means equally. No padding or recording-boundary derivative is included. MPJRE is global rotation, MPJVE is mesh vertex position error, and jitter is predicted motion's third derivative rather than velocity error.

| Outcome | AMASS Epoch59 | Final DIP 2epochs | Relative decrease |
|---|---:|---:|---:|
| MPJRE, degrees | 30.26337 | 22.46870 | 25.76% |
| MPJPE, cm | 11.65586 | 9.54019 | 18.15% |
| MPJVE mesh vertices, cm | 14.59548 | 11.96889 | 18.00% |
| Jitter, 100 m/s³ | 1.61320 | 1.04836 | 35.01% |

All24 individual mask aggregates improve in all four metrics. Minimum mask-level relative decreases are17.18% MPJRE,8.81% MPJPE,9.07% MPJVE and8.57% jitter. Both test subjects separately improve on all four aggregate measures. These statements concern recording-aggregated values, not every individual frame or every recording.

**Independent validation.** Verified955 original artifact hashes, both checkpoints, SMPL, processed data and execution sources; all456 caches have bitwise-identical targets, correct protocol/mask/recording identities, expected frame/jerk counts and finite predictions. Independently reconstructed all mask/overall sums using math.fsum; maximum discrepancy2.22e-16. Eight preselected model×mask×recording metric rechecks use FK chunk11 instead of32, with maximum primary-metric difference1.574e-12 against threshold1e-4. Reference jitter is identical across masks/models. Six aggregation/coverage tests and the static-pose actual-SMPL worker check pass. The original artifact manifest is preserved; a separate validation manifest seals subsequent logs, verification source and report outputs.

**Interpretation and limits.** This is measured accuracy and smoothness improvement from the fixed DIP fine-tuning under the specified evaluation, consistently across configured masks. Lower jitter alone would not establish better pose accuracy. Two people and correlated mask/frame results do not establish statistical significance or broader consumer-device generalization. Historical25-mask AMASS exposure, unresolved exact paper window/startup wording, inherited future-dependent filtering/filling/interpolation, and omitted IMU-only IK still prevent claiming strict reproduction of the paper's final numbers. Earlier nonoverlapping or single-mask study scores are not substituted into this table.

**Evidence and next action.** [Results, figure and conventions](DIP_FINETUNING_COMPARISON.md), [versionable metadata](results/dip_final_2ep_20261115/MANIFEST.json), [paired comparison CSV](results/dip_final_2ep_20261115/comparison_with_changes.csv), [all24 mask details](results/dip_final_2ep_20261115/per_mask.csv), and [independent verification](results/dip_final_2ep_20261115/independent_validation.json). Full caches, rows, source snapshots and original/validation manifests remain in `outputs/local_dip/dip_finetune_comparison_cpu_v1/`. Next complete refinement and remaining paper-convention checks using the same locked baseline; do not choose a new K from these test scores. Existing placement studies remain supplementary.

## I01 IMU-only orientation refinement and paper-convention reconciliation

**Work/entry date:** 2026-11-16, America/Chicago. **Requested by:** Alex. **Execution and review:** Codex assistance; CPU, no training or fresh neural inference. **Status:** Implemented, full evaluation completed and independently validated.

**Objective and evidence fixed before testing.** Reconstruct the orientation-only IK described by IMUPoser4.4 on the immutable F03 final model/predictions, and audit remaining paper evaluation choices. The checked author release has no matching optimization/evaluation loop; inverse_kinematics_R only converts global/local rotations. Paper specifies MSE, Adam and10 iterations per frame plus shoulder/elbow optimization, but omits exact IK learning rate, full head/hip index lists, rotation variables and optimizer/output boundary rules. A separate frozen protocol records every reconstruction choice before new test IK results; neither scores nor proximity to the paper select settings.

**Recipe and isolation.** Use public synthesis sensor segments18/19/1/2/15 rather than stale leaf_joints20/21/7/8/12. Optimize respectively16/18,17/19,0/1,0/2,12/15; the head/hip lists are declared reconstruction conventions. Fixed Adam lr1e-3 comes from the cited AvatarPoser paper, not a recovered IMUPoser parameter; AvatarPoser code defaults1e-1/5steps and targets positions, so it is not an exact transferable implementation. Use local right-multiplicative rotation-vector increments, FP64 optimization/FP32 output, SVD-projected available orientation targets, fresh per-frame moments,10 updates and final iterate. All available constraints optimize jointly. No GT, GT calibration, hidden sensor, position loss, extra smoothing, best-iterate choice or rollback. Original dataset/network inputs and every unoptimized local6D entry stay unchanged. The solver consumes prediction[t] plus measured orientation[t]; existing upstream future dependence remains disclosed.

**Observed full-test results.** Same final two-epoch checkpoint and all24 masks ×19 recordings ×24,078frames/mask as F03. Completed456 pairs and577,872 refined frames in two parallel CPU parts, each2threads/frame_batch256; elapsed280.19 and282.94s. Canonical mean-male/zero-translation/no-pose-blendshape metrics and the frozen frame/T−3 then mask-equal reduction are unchanged.

| Metric | Final network, no IK | IMU-only IK | Change |
|---|---:|---:|---:|
| MPJRE, ° | 22.46870 | 22.16620 | 1.35% decrease |
| MPJPE (joints), cm | 9.54019 | 9.42636 | 1.19% decrease |
| MPJVE (mesh vertices), cm | 11.96889 | 11.82258 | 1.22% decrease |
| Jitter, 100 m/s³ | 1.04836 | 1.22145 | 16.51% increase |

All24 mask-level recording aggregates and both subject-level aggregates show the same direction: three pose errors decrease, jitter increases. MPJVE improves0.14630cm, about1.46mm. Available-IMU angular residual falls12.60312°→10.95081°; no direction-MSE increase occurs among577,872 mask-frames. This establishes optimization toward the IMU objective, not an across-the-board motion-quality improvement. Independent-frame corrections can increase high-order temporal changes; this is a plausible explanation rather than an isolated causal result. No test-directed tuning follows this result.

**Validation.** Ten analytic/numerical/measurement-isolation solver tests and four coverage/group-weight tests pass. Independent validation checks1,424 original artifacts,456 standalone solver inputs/outputs, protected source/model/data hashes, bitwise active measurements and absent zeros. No GT is present in standalone IK input files. Unoptimized local6D values are bitwise preserved over11,437,050 joint-frames. Independent aggregation differs by at most1.78e-15. Fixed first/last masks × first/last recordings re-run five boundary/late frames with batch1 instead of256: maximum saved-pose difference0. Full-recording raw/IK FK-chunk11 versus32 rechecks differ by at most1.194e-12 against1e-4. Original F03 baseline, checkpoints, processed data and training protocol remain untouched.

**Remaining convention audit.** Public preprocessing data has default beta ones, while the predeclared canonical measurement uses beta zero; interpolation also yields nonorthogonal orientation matrices. Train-only mapping sanity uses4,241 fixed samples from41 first-eight recordings, finding approximately1.50/1.40/3.36/2.52/0.37degree segment errors without fitting calibration. All24 masks collapse to12 mirror groups. Changing only overall reduction to12 group-equal yields MPJVE12.16981cm instead of primary11.82258cm; sequence/subject-equal alternatives are also saved. An explicitly exploratory four-pair geometry check finds shared beta ones changes vertex errors by−1.087 to−0.399cm and pose-blendshape-on by+0.032 to+0.105cm; those case ranges are not full-test scores or proof of author settings. Twenty-one alignment items distinguish source-backed definitions, reconstruction conventions, unresolved author details, known training differences and phase-two work.

**Interpretation, limits and decision.** DIP's fine-tuning→rolling→IMU-only IK→all24 four-metric engineering chain now runs with auditable evidence. Strict equivalence remains unproven because exact author IK/shape/window/reduction details and historical25-mask/AMASS/epoch differences remain. Similar or lower values than Table2 are insufficient proof. Preserve both raw and IK results including worse jitter. Additional participants/consumer devices and real-time causality/latency remain later work. Do not change training K or choose new IK settings from these test scores.

**Evidence.** [IK method/results](IMU_ONLY_IK.md), [paper alignment and sensitivity](PAPER_EVALUATION_ALIGNMENT.md), [frozen IK protocol](results/dip_imu_only_ik_v1_20261116/PROTOCOL.json), [24mask details](results/dip_imu_only_ik_v1_20261116/per_mask.csv), [12group details](results/dip_imu_only_ik_v1_20261116/per_group.csv), [independent validation](results/dip_imu_only_ik_v1_20261116/independent_validation.json), and [metadata manifest](results/dip_imu_only_ik_v1_20261116/MANIFEST.json). Full standalone inputs, poses, loss trajectories, rows and code/evidence snapshots remain in `outputs/local_dip/dip_imu_only_ik_v1_20261116/`; original and appended validation manifests are separate. A narrow .gitignore exception makes the three small docs/results metadata bundles (development, F03 and I01) versionable; model/motion assets remain ignored. No commit or push performed.

## J01 Jitter attribution, adjacent rotations and controlled pose-component checks

**Work/entry date:** 2026-11-16, America/Chicago. **Requested by:** Alex. **Execution/review:** Codex assistance. **Status:** Completed and independently validated; descriptive diagnosis of the existing I01 test results, no baseline tuning.

**Scope and measurements.** Reconstructed whole-recording joint trajectories with the same mean-body FK/chunk32 and25Hz conventions, for all19 recordings ×24 masks ×24 joints. Per-time norm differences sum exactly to I01's0.17309549(100m/s³) excess. Exported receiving-joint, mask, recording, category and joint-category tables,10,944 cell records, adjacent local/global rotation changes, correction rotation vectors and per-frame jerk. Derivatives never cross recordings. Category prefixes are explicitly mapped by the dataset overview to DIP AppendixF/Table6: upper-body/lower-body/locomotion/freestyle/interaction. No per-frame action labels were invented.

**Findings.** Right/left feet and ankles together account for37.68% of the net increase, despite unchanged local rotations. Seventeen pocket-containing masks account for95.95%. The largest mask increases include rw_rp(+0.29447),lw_rp(+0.28841),rw_lp(+0.28530). Upper-body category contributes40.88% and locomotion26.15% of net excess; locomotion has a slightly larger average increase per time sample, while upper-body has more weight. s10/01_a.pkl ands09/01_a.pkl are the largest recording contributors; s10/01_c.pkl has the largest per-time mean rise. Rankings distinguish severity from total contribution.

**Rotation and propagation evidence.** Pelvis mean adjacent local angle rises1.10943°→1.13892°(~2.66%), while its5–12.5Hz local angular-velocity power rises19.56%. Per-record linear detrending/Hann periodograms use relative-frame rotation vectors, avoiding absolute-axis-angleπ branch artifacts; a high-frequency band is not assumed to be pure noise. Pelvis correction components exhibit34,296 jumps larger than0.015rad. The largest-contribution case rw_lp on s10/01_a.pkl has correction switching near±0.01rad and additional jerk spikes, preserved in a time/frequency figure.

**Controlled counterfactuals.** Restoring only the original pelvis local rotation within the I01 poses reduces overall jitter1.22145→1.09072, eliminating75.53% of net excess in that intervention. Keeping only the pelvis IK correction on the raw pose gives1.19007. Restoring hips/arms/head gives1.20568/1.20584/1.22144. These are pose-component interventions, not newly optimized constraint-satisfying inference variants. Nonlinear norm and chain interactions prevent summing the component effects into independent causal shares. The findings identify pelvis/hip correction propagation to distal world positions; pelvis position jitter itself can remain zero.

**Verification/evidence.** Three analytic signal/accounting tests pass. Independent comparison of all10,944 cells against the old canonical per-joint rows has maximum difference1.42e-14; independent contribution accounting differs from the I01 increase by1.67e-16. All grouped totals match. [Report and figures](IK_JITTER_DIAGNOSIS.md), [independent validation](results/ik_jitter_diagnostics_v1_20261116/independent_validation.json), [19-recording detail](results/ik_jitter_diagnostics_v1_20261116/by_sequence.csv), [24 masks](results/ik_jitter_diagnostics_v1_20261116/by_mask.csv), [counterfactuals](results/ik_jitter_diagnostics_v1_20261116/counterfactual_results.json). Full arrays and source snapshots remain in `outputs/local_dip/ik_jitter_diagnostics_v1_20261116/` and `ik_jitter_counterfactuals_v1_20261116/`. Original I01 data, code and manifests are unchanged.

## I02 Independent causal Temporal IK regularization experiment

**Work/entry date:** 2026-11-16, America/Chicago. **Requested by:** Alex, explicitly as a new experiment preserving I01. **Execution/review:** Codex assistance; CPU. **Status:** Complete, independently validated; partial reduction of excess jitter.

**Method fixed before new temporal test results.** Optimize independent rolling5-frame histories[max(0,t−4),t] and emit onlyt. Sum available-IMU orientation MSE over valid history frames plusλ times squared third differences of all24 FK joint positions, normalized by0.01m. Same mean-body geometry, sensed segments/local DOFs,10Adam steps,lr1e-3,FP64 variables andFP32 output asI01. No GT in the optimizer, no added future frames, post-filter, cross-window state, or recording-boundary derivative. Valid short-prefix masks prevent fake startup jerk. Pose variables only are updated; model weights and training protocol stay frozen. This temporal loss is an extension experiment, not asserted to be the original paper method.

**Development-side selection.** Use the fixed development epoch2 trained on s01–s06, not the final model trained on the calibration people. Calibration uses s07/s08 all11 recordings, central96-frame panels with4 preceding pose outputs and12 prechosen mirror-group representatives:132 pairs/12,672 metric frames. Every pose still has original125-frame neural history. Freezeλ grid[0.001,0.01,0.1,1], geometry, budget and selection rule first: all3 pose errors no worse than raw, retain≥50% framewise-IK vertex gain, jitter below framewiseIK; minimize jitter among eligible, with a predeclared fallback ordering if none qualify. Onlyλ0.001 qualifies. Calibration never loads the test tensor container; no temporal test metrics selectλ. Zero temporal weight reproduces I01 outputs exactly in all132 real calibration pairs. The original test was previously analyzed, so this is not described as an untouched blind test.

**Fixed full-test outcomes.** Use the immutable final first-eight two-epoch model's cached network outputs, all24 masks and all19 s09/s10 recordings. Complete456 pairs/577,872 temporal output frames; two CPU parts take420.69/421.90s. Primary frame/T−3 then mask-equal reductions are unchanged.

| Metric | Network | I01 framewise IK | I02 Temporal IK |
|---|---:|---:|---:|
| MPJRE, degrees | 22.46870 | 22.16620 | 22.16886 |
| MPJPE, cm | 9.54019 | 9.42636 | 9.42868 |
| MPJVE vertices, cm | 11.96889 | 11.82258 | 11.82466 |
| Jitter,100m/s³ | 1.04836 | 1.22145 | 1.16742 |

Temporal reduces total jitter4.42% relative toI01 and retains98.58% of I01's MPJVE gain versus raw. It remains11.36% above raw jitter. In every one of24 mask aggregates, three pose errors remain below raw and jitter is belowI01, but jitter is below raw in0/24. Most pose errors are slightly higher thanI01; the MPJVE cost is0.00207cm(~0.021mm). Predeclared compromise criterion is met; the stronger goal of returning jitter to raw-network level is not met. No follow-up parameter choice is made from these test outcomes.

**Validation.** Six causal/geometry/zero-weight/boundary tests plus two interval-inference/selection tests pass, in addition to J01's3 signal tests. Mean-body FK integration differs from canonical FP32 by at most2.02e-7m. Independent audit verifies922 original test artifacts,456 pairs, allprotected original sources/models/data/calibration choices and11,437,050 frozen local joint-frames. Raw/I01 metric rows and target/prediction cache hashes remain unchanged. Selection is independently reconstructed. Aggregate discrepancy≤1.78e-15. Fixed first/last masks × first/last recordings independently re-run past-only scalar windows at frames0/1/3/4/124/125/last: maximum pose difference0. FKchunk11 versus32 complete-recording checks differ≤1.237e-12 against1e-4.

**Evidence and limitation.** [Independent experiment report](TEMPORAL_IK_EXPERIMENT.md), [frozen test protocol](results/temporal_ik_v1_20261116/PROTOCOL.json), [development selection](results/temporal_ik_v1_20261116/calibration_selection.json), [three-way metrics](results/temporal_ik_v1_20261116/comparison.csv), [joint jitter](results/temporal_ik_v1_20261116/joint_jitter.csv), [independent validation](results/temporal_ik_v1_20261116/independent_validation.json). Full calibration/test directories are `outputs/local_dip/temporal_ik_v1_20261116/`; original and appended validation manifests are distinct. The limited calibration panel and fixed10-step budget do not establish an optimal regularizer. Upstream preprocessing future dependence persists; this past-only IK does not add future access. Further iteration/weight/form changes require another development-selected experiment, preserving this result andI01.

## V01 Final-model four-way offline replay

**Work/entry date:** 2026-11-19, America/Chicago. **Requested by:** Alex. **Execution/review:** Codex. **Status:** Completed and independently validated; explanatory visualization of existing F03/I01/I02 test predictions, no model or optimizer updates.

**Question and fixed selection.** Inspect motion direction, exact frame alignment, local joint errors and extra jitter using one common mask, `rw_lp` (right wrist + left pocket). Before rendering, freeze s09/03_a.pkl seq5 central250 frames[478,728), s09/01_a.pkl seq0 central250 frames[864,1114), and archived J01's largest-contribution pair s10/01_a.pkl seq9[1246,1496), centered on recorded single-joint excess-jerk peak1371. The third clip is an upper-body failure example, selected exploratorily from the prior diagnosis; it is not a third action class or an unbiased performance sample. No new clip score or visual appearance changes the selection. Category names describe recordings, not verified per-frame actions.

**Inputs, geometry and display.** Reuse final two-epoch checkpoint lineage SHA996a3e2df21ac675df8f777d975d0700f3836c1d835bb898044a714e6cb5f7d8 and unchanged raw/framewise/Temporal caches. All four panels use the same source frame, mean male body, zero root translation, inherited mesh settings, one camera and a common scale across every case. The proper display rotation(X,Y,Z)→(X,−Z,Y) preserves handedness; there is no heading fitting, lag adjustment, interpolation or output smoothing. Gray dashed GT overlays and common0–40cm joint-error colors expose mismatches. Each clip has250 frames at25Hz; numerical jerk uses only247 complete in-clip stencils. Three original-speed GIFs use40ms/frame; the additional jitter slow replay uses the exact same250 frames at160ms/frame, without interpolation. Full curves are explanatory offline plots, not online-availability claims.

**Observed clip results.** MPJPE(network/framewise/Temporal) is7.82913/7.60993/7.61921cm for walking,10.94494/10.83118/10.83377cm for upper-body, and7.97335/7.83514/7.83759cm for the jitter example. Jitter is1.45791/1.74066/1.57023,1.09042/1.64525/1.38254, and1.20962/1.69896/1.55713 respectively, in100m/s³. Temporal reduces framewise jitter9.79%,15.97%,8.35% within these clips, while remaining above raw. These are clip summaries, not substitutes for I02's full-test4.42% result. All four canonical metrics are in the linked CSV.

**Joint errors and direction.** Left wrist/hand, unobserved by this mask, are the main position-error points. Upper-body network left-hand mean error50.60cm contrasts with whole-body10.94cm, and framewise IK leaves50.56cm. Direction checks find no overall180° root reversal in the selected frames; maximum root-angle error over all branches/clips is22.69°. Larger walking root steps also occur in GT. GT jitter is nonzero and exceeds network jitter in the walking clip, reinforcing that low jerk alone does not establish accuracy. These observations describe the fixed cases; they do not select a new mask or refinement policy.

**Verification and review.** Four new analytic checks cover fixed selection, proper display rotation, source-frame clocks and cubic-trajectory jerk/pelvis alignment. The full suite passes109 tests with no failures/skips using the team checkout source and trusted local integration assets. The independent process checks64 rendered artifacts,750 source frames and72,000 four-panel joint-frames; saved poses match original caches bitwise and GT matches processed DIP. FKchunk11 differs from saved FK by0m; independent MPJPE/jitter summaries differ by≤1.78e-15. All1,000 encoded GIF frames decode with correct40/160ms durations; five fixed frame images per clip match their quantized PNGs exactly, and slow/normal counterparts match. Protected original data, weights, protocol and baseline cache hashes are unchanged. Codex additionally visually inspects the15 fixed-time four-panel screenshots(0/62/125/187/249); this is distinguished from viewing every animated frame.

**Execution and evidence.** CPU rendering peak RSS838.88MiB; zero network forward calls and optimizer steps. Entry points are `tools/render_final_comparison.py prepare`, `render`, and `tools/validate_final_comparison.py --output`, using `outputs/local_dip/final_four_way_replay_v1_20261119`; [exact commands and protocol](FINAL_MODEL_REPLAY.md) record all arguments. Protocol stores base Git revision8519e9c, hashes and snapshots of the uncommitted execution sources, input/model/cache identities, selection rules and geometry. Large animations/poses stay local;20 compact reports/metadata/preview files plus their own manifest are mirrored under `docs/results/final_four_way_replay_v1_20261119/`. Original artifact/validation manifests are retained; final review state and an additional review manifest are separate.

**Interpretation and next action.** The replay explains the existing accuracy/jitter compromise and substantial unobserved-arm failures. It does not establish consumer-device accuracy, strict author-equivalent evaluation, global trajectory or real-time latency. Next implement an explicit real-device log adapter with units, mounting/world-frame calibration, timestamps and25Hz resampling, then replay one physical-device recording through the fixed model. [Animation/report](FINAL_MODEL_REPLAY.md), [clip metrics](results/final_four_way_replay_v1_20261119/clip_metrics.csv), [independent validation](results/final_four_way_replay_v1_20261119/independent_validation.json), [visual review](results/final_four_way_replay_v1_20261119/VISUAL_REVIEW.json), [compact manifest](results/final_four_way_replay_v1_20261119/MANIFEST.json).

## L01 English research records and presentation migration

**Work/entry date:** 2026-11-19, America/Chicago. **Requested by:** Alex, to support record keeping and group sharing. **Status:** Completed; English translation and provenance audit passed, separately from numerical experiments.

**Scope and method.** Translate the remaining Chinese documentation, research-log labels, result tables, report text, generated four-way replay overview, and corresponding readable local report/source snapshots. Also inspect JSON/logs for Chinese stored as escaped Unicode. Update report generators so future presentation uses English. Preserve each original file's exact bytes in a base64 archive, recording original and English SHA-256 values; do not attribute the new language to old experimental execution. Historical protocols/manifests retain their original hashes. The migration audit verifies both versions explicitly rather than silently replacing historical digests.

**Scientific boundary and validation.** No neural training, network inference, IK optimization, metric recomputation, checkpoint edit or GIF regeneration. Localized Python generators must have identical AST structure after replacing string constants, preserving computations and numerical constants. Numeric JSON values and numeric CSV cells must match, and readable project text is scanned for residual Chinese, including Unicode escapes in JSON/logs. Selected original checkpoints/data/caches, the original replay protocol and all four GIF hashes are checked separately. Full-suite results and the translation audit are archived with this migration.

The final migration covers 40 files, including Unicode-escaped labels in two JSON files and one log. Six localized source copies preserve their complete non-string AST; seven JSON/CSV copies preserve numerical data. The scan finds zero remaining Chinese text files. Thirty-two historical manifest entries match the losslessly archived pre-translation bytes; 62 scientific inputs, unchanged sources and animations match original hashes. All 109 unit/integration tests pass with no skips or failures. Original-byte restoration is checked in a new temporary directory.

**Sharing and replay.** [English overview](../outputs/local_dip/final_four_way_replay_v1_20261119/index.html) uses the same existing animations and links. [Language/provenance guide](RECORD_LANGUAGE.md), [translation map](record_translations_20261119/TRANSLATION_MAP.json), [lossless originals](record_translations_20261119/ORIGINAL_BYTES.json), [migration validation](record_translations_20261119/VALIDATION.json). A legacy byte-exact validator can correctly flag localized presentation/source files; use the translation audit and restore archived originals into a new directory when inspecting historical bytes. No old model result is reclassified as a fresh experiment.

**Method clarification.** [Framewise versus Temporal IK](IK_REFINEMENT_EXPLAINED.md) documents the same final network with two refinement methods: independent current-frame orientation correction versus joint past/current five-frame optimization with a third-difference position penalty, emitting only the current frame. Both use ten Adam updates and no added future access. I02 is a separate extension, not the original paper baseline; it lowers full-test framewise jitter by 4.42%, retaining 98.58% vertex-error gain, still 11.36% above raw-network jitter.

## Open questions and action tracking

| Priority | Item | Status and next step |
|---|---|---|
| Current engineering step | Replay real consumer-device logs on the PC | V01 final DIP four-way replay complete; next freeze device log schema, unit/frame/mounting calibration, timestamps and resampling before one physical recording |
| Current priority | Establish paper-aligned DIP quantitative evaluation | I01 IK and21-item convention audit completed; seek author settings for strict equivalence and separately plan any strict AMASS recipe |
| Later research | Does the E04 wrist benefit justify its leg and root-orientation costs? | Supplementary until reproduction baseline is complete |
| Later error diagnosis | Why does substantial error remain with both wrists? | Preserve cached predictions and the sequence 38 exception; revisit after complete baseline |
| Temporal stability extension | Can jitter return to raw-network level while preserving IK gains? | I02 achieves4.42% lower jitter thanI01 with98.58% vertex gain retained; still11.36% above raw. Further changes require separate development-side selection |
| Test evaluation | Measure the locked final model against original Epoch59 | F03 andI01 raw/IK full24-mask reports complete. IK improves three errors and worsens jitter in all24 aggregates; no test-driven recipe or K change |
| Reproducibility | Review newer study tools and guides for the team repository | Earlier work committed in8519e9c; V01 tools/log/compact results currently uncommitted,109-test suite and artifact validation pass |
| Generalization | Do the effects hold for additional people and consumer devices? | Future independent confirmation; F03 final-model comparison covers two test subjects |
| Live operation | What accuracy is retained under a usable latency policy? | R01 measures last-frame rolling and filter direction; raw-stream causal filling/resampling and end-to-end timing remain future work |

## Research discussion on 2026-11-10

Alex requested a project update, interpretation of the latest findings, and a persistent progress record. The discussion emphasized that sensor allocation changes which arm is accurate; a whole-body average can hide the tradeoff. The head + both wrists comparison remains a proposed next experiment. Today's documentation action establishes this notebook as the central progress record; it does not execute E04.

**Later follow-up on the same date.** Alex requested E04 execution. Its completed entry above supersedes the proposal status in this earlier discussion and records the measured arm-versus-leg tradeoff.

## Research discussion on 2026-11-11

Alex clarified that the immediate objective is reproducing the IMUPoser paper before pursuing extensions, then authorized the scope comparison and training/data audit. A01 records those completed tasks. Subsequent work should follow the phase-one reproduction checklist rather than continue sensor-allocation experiments by default.

## Research discussion on 2026-11-13

Alex requested complete metric definitions and known-pose validation, emphasizing that MPJVE is mesh vertex position error rather than velocity. M01 records the implementation and validation. The next phase-one work should use this measurement layer with explicit protocol choices; it does not require replacing the audited pretrained weights immediately.

## Template for the next entry

Copy this template when work begins. Use a new ID for a new experiment or a dated follow-up for an existing one.

```text
Experiment ID and title:
Work date and time with timezone:
Entry date and author or contributors:
Status: proposed / running / completed / validated / failed / inconclusive
Related experiment or decision:

Question and motivation:
Hypothesis and outcomes fixed before execution:
Data split, identities, exclusions, and hashes:
Checkpoint and hash:
Code revision, uncommitted snapshot, environment, and seeds:
Exact command or saved protocol/provenance link:
Comparisons, controls, metrics, and aggregation:
Run directory and artifact links:

Observed results with units and baselines:
Validation results:
Failures, deviations, and unexpected observations:
Interpretation and alternative explanations:
Limitations:
Decision and rationale, including who made it:
Next action and status:
```
