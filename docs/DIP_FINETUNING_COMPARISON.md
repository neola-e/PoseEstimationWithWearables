# DIP 微调前后四指标比较

完成日期：2026-11-15（America/Chicago）。原 AMASS Epoch 59 与最终两轮 DIP 微调模型已在相同流程下完成测试并通过独立核验。四项总体指标均下降，24 个传感器组合各自汇总后的四项指标也均下降。训练协议和 checkpoint 保持不变，没有根据测试结果重选模型。

**2026-11-16 后续：** 同一最终模型已加入 IMU-only IK 并完成全量核验，见 [IMU_ONLY_IK.md](IMU_ONLY_IK.md)；三项精度误差进一步小幅下降，jitter上升。本文原无IK结果及其冻结协议保持原记录，另见[当前论文口径核对](PAPER_EVALUATION_ALIGNMENT.md)。

## 主要结果

| 指标 | AMASS Epoch 59 | DIP 两轮微调后 | 下降幅度 | 改善组合数 |
|---|---:|---:|---:|---:|
| MPJRE (°) | 30.26337 | 22.46870 | 25.76% | 24/24 |
| MPJPE (cm) | 11.65586 | 9.54019 | 18.15% | 24/24 |
| MPJVE（网格顶点） (cm) | 14.59548 | 11.96889 | 18.00% | 24/24 |
| Jitter (100 m/s³) | 1.61320 | 1.04836 | 35.01% | 24/24 |

测试范围：s09/s10、全部 19 条录制、每个模型每个 mask 24,078 帧、全部 24 个稀疏 mask，排除五传感器 global。空间指标先在每个 mask 内按有效帧加权；jitter 按每条录制 T−3 个时间样本加权；最后对 24 个 mask 等权平均。相对下降为 `(微调前−微调后)/微调前×100%`。

旋转误差下降约 26%，关节与体表位置误差下降约 18%，预测动作的 jitter 下降约 35%。这表明在这套固定测试流程下，DIP 微调改善了姿态准确性和动作平滑性。Jitter 单独衡量平滑程度，不能替代准确性指标；它不表示相对 GT 的速度误差。本次两名测试者及相互关联的 mask 不足以支持统计显著性或广泛人群泛化结论。

![四指标比较](results/dip_final_2ep_20261115/four_metrics.png)

[数值及变化 CSV](results/dip_final_2ep_20261115/comparison_with_changes.csv) · [逐 mask CSV](results/dip_final_2ep_20261115/per_mask.csv) · [逐测试者 CSV](results/dip_final_2ep_20261115/per_subject.csv) · [完整汇总 JSON](results/dip_final_2ep_20261115/metrics.json)

## 模型与数据追溯

最终阶段 `dip_final_v1` 已由 Alex 在普通 WSL Terminal 使用 RTX 5060 Ti 完成：前八人、2 epochs、162 updates、batch256。预算 K=2 由开发集预先选定；最终阶段从原 AMASS 权重与全新 Adam/seed42 重新开始。训练 combined loss 从 0.02032563 至 0.01377506；训练没有打开测试集。核对全部 31 个训练执行源文件哈希和 20 个 Adam step counter 均通过。

| 模型 | 固定权重 | SHA-256 |
|---|---|---|
| 微调前 | AMASS Epoch 59 | `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058` |
| 微调后 | `dip_final_v1/checkpoints/epoch_002_9d8a49a977cb.ckpt` | `996a3e2df21ac675df8f777d975d0700f3836c1d835bb898044a714e6cb5f7d8` |

测试容器 SHA-256：`352ce4f1312fbc7d496ce3d3d3c7287f01389624b75296ce4e349868b958b475`。所有 19 条 processed 录制均与 raw 的 acc/ori/pose 逐位匹配；s09 有 9 条，s10 有 10 条。

实际完成目录：`outputs/local_dip/dip_finetune_comparison_cpu_v1/`。两个模型各有 577,872 个预测输出帧，合计 1,155,744 帧；456 个 mask×录制配对全部完成。训练用 GPU，本次配对推理与指标使用 CPU。Codex 的 GPU 隔离及用户选择的现有权限保持不变。

## 固定测量设置

[实际 CPU 评估协议](results/dip_final_2ep_20261115/PROTOCOL.json) 与 [原冻结训练协议](protocols/dip_finetuning_v1_20261114/PROTOCOL.json) 分开保存。

| 项目 | 两个模型共用的设置 |
|---|---|
| 输入 | 25 Hz，60 维，24 个稀疏 mask；加速度除以 30 |
| 滚动推理 | 125 帧历史、步长 1、只取末帧、起始用已有短前缀；每窗口和录制重置状态 |
| 平滑 | 保持 processed 原有居中五帧加速度平均，不再平滑；披露未来依赖 |
| 网络执行 | CPU、FP32、无 AMP；eval/inference mode；batch64；两个并行分片，每分片推理 8 线程 |
| 指标执行 | 原 `paper_metrics.py`，每分片 2 线程，FK chunk32；不更新网络参数 |
| 身体模型 | 相同 mean male SMPL，24 关节与全部 6,890 顶点；pose blend shapes 关闭、零轨迹平移 |
| MPJRE | 局部旋转经 FK 后的全局旋转角误差，度；局部旋转 debug 指标单独保留 |
| MPJPE / MPJVE | pelvis 平移对齐后的关节 / 网格顶点位置误差，cm；保留根旋转；MPJVE 不是速度 |
| Jitter | 预测关节轨迹三阶差分，乘 25³；每条录制独立计算，表格单位 100 m/s³ |
| 汇总与 refinement | mask 内按帧 / T−3 加权，再对 mask 等权；两个模型均无 IK |

每个 mask 有 24,021 个有效 jerk 时间样本。两个 CPU 分片并行完成，各耗时约 72.2 / 72.8 分钟，不含前期准备与后续独立核查。此前准备的普通 WSL GPU 协议位于 `dip_finetune_comparison_v1/`，截至报告生成时尚无共享预测缓存或完整结果；本报告没有混用其结果。

## 验证结果

- 新增 6 项汇总、覆盖、权重及输入一致性测试通过；8 帧合成静态相同姿态的实际 SMPL 指标计算均为零。此前 M01 的已知姿态数值检查继续适用。
- 独立验证全部 955 个原始结果文件哈希、模型 / SMPL / 数据 / 源码哈希和 456 个预测缓存，所有 target 与 processed 逐位一致；预测全为有限值，帧数与 T−3 分母正确。
- 使用独立 `math.fsum` 重新构建汇总，最大绝对差 `2.22e-16`。
- 事先固定首尾 mask × 首尾录制 × 两个模型，共 8 次指标重算；将 FK chunk 从 32 改为 11，四指标最大绝对差 `1.574e-12`，远低于事先阈值 `1e-4`。
- GT reference jitter 在全部 mask 与两个模型间一致。未重训、未改 checkpoint、未用测试分数选择预算。

[独立验证 JSON](results/dip_final_2ep_20261115/independent_validation.json) · [最终训练审查](results/dip_final_2ep_20261115/FINAL_TRAINING_REVIEW.json) · [结果文件校验清单](results/dip_final_2ep_20261115/MANIFEST.json)

## 复算入口与解释范围

实际评估入口：[compare_dip_finetuning_cpu.py](../tools/compare_dip_finetuning_cpu.py)，复用 [compare_dip_finetuning.py](../tools/compare_dip_finetuning.py) 的冻结配对及汇总逻辑；两个 CPU 分片分别为 `--part 0` 与 `--part 1`，同一目录可断点续算。独立验证使用 [validate_dip_finetuning_comparison.py](../tools/validate_dip_finetuning_comparison.py)，不进行网络推理或模型选择。指标及推理实现 [paper_metrics.py](../tools/paper_metrics.py) 和 [window_inference.py](../tools/window_inference.py) 保持原版本。

本结果完成了固定微调模型与原初始化的同口径对照。仍有 AMASS 初始化的历史 25-mask 暴露、原论文窗口 / 起始边界细节尚未完全找回、居中滤波及填补 / 插值的未来依赖、尚未加入 IMU-only IK 等差异。因此不能把本表直接宣称为论文最终数值的严格复现。后续应优先完成 refinement 和剩余口径核对；测试结果不回用于改变本次已冻结的训练预算。
