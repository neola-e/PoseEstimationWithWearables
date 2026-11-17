# IMU-only IK refinement：实现、全量评估与验证

完成日期：2026-11-16，America/Chicago。最终两轮 DIP 模型的 IMU-only IK 已实现，并在 s09/s10 的全部 19 条录制、24 个位置组合上完成对照和独立核验。**三个精度误差小幅下降，但 jitter 上升**；这一结果在各个 mask 的录制汇总中方向一致。该方法依据论文重建，未找回的作者设置已作为显式约定记录，不能称为作者实现的逐行复刻。

**2026-11-16 后续独立实验：** [J01抖动诊断](IK_JITTER_DIAGNOSIS.md)定位了pelvis修正传播；[I02 Temporal IK](TEMPORAL_IK_EXPERIMENT.md)保留98.58%的顶点精度收益并相对本baseline降低jitter4.42%，但仍高于无IK。本文I01数值、源码和运行目录保持原baseline记录。

## 同一模型下的主要结果

| 指标 | 无 IK | IMU-only IK | 相对变化 | 同方向的组合数 |
|---|---:|---:|---:|---:|
| MPJRE (°) | 22.46870 | 22.16620 | 下降 1.35% | 24/24 |
| MPJPE（关节） (cm) | 9.54019 | 9.42636 | 下降 1.19% | 24/24 |
| MPJVE（网格顶点） (cm) | 11.96889 | 11.82258 | 下降 1.22% | 24/24 |
| Jitter (100 m/s³) | 1.04836 | 1.22145 | 上升 16.51% | 24/24 |

主汇总继续使用 F03：mask 内空间误差按帧数加权，jitter 按每条录制 T−3 加权，再对全部24 mask 等权。每个 mask 24,078 帧、24,021 个 jerk 时间样本。MPJVE 是网格顶点位置误差；jitter 是预测关节位置三阶差分，不是速度误差，也不是与 GT jerk 的差。

MPJVE 从 11.96889 降到 11.82258 cm，改善约 1.46 mm（1.22%）；jitter 增加 16.51%。不能把本次 IK 描述成四指标全面改善。s09、s10 分别汇总后同样是三个精度指标下降、jitter 上升；两名受试者及相关 mask 不支持统计显著性或消费设备泛化结论。

![四指标对照](results/dip_imu_only_ik_v1_20261116/ik_four_metrics.png)

[主结果和变化 CSV](results/dip_imu_only_ik_v1_20261116/changes.csv) · [逐 mask](results/dip_imu_only_ik_v1_20261116/per_mask.csv) · [逐受试者](results/dip_imu_only_ik_v1_20261116/per_subject.csv) · [完整指标 JSON](results/dip_imu_only_ik_v1_20261116/metrics.json)

## 依据和明确补足的约定

[IMUPoser §4.4](https://arxiv.org/html/2304.12518v1#S4.SS4) 规定利用 IMU 的绝对方向，使用 MSE、Adam，每帧10次优化，并说明腕部优化肩与肘。所检查的 [公开仓库 commit a0e9d33](https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2) 未找到相应的姿态优化器和完整评估循环；`inverse_kinematics_R` 是 global/local 旋转转换。

论文未给出 IK 学习率、头/髋全部关节索引、旋转参数化、目标投影及状态重置的完整细节。本次在查看新测试结果之前固定以下设置，并保存 [协议](results/dip_imu_only_ik_v1_20261116/PROTOCOL.json) 与源码快照：

| 设置 | 本次实现 | 证据边界 |
|---|---|---|
| 次数与目标 | 每帧10次 Adam；可用 IMU 全局方向3×3矩阵 MSE；多个约束联合优化 | 与 IMUPoser 方法描述一致 |
| 学习率 | 1e-3；betas 0.9/0.999，eps1e-8，无 weight decay | 学习率来自被引用的 AvatarPoser 论文，未证实为 IMUPoser 作者值 |
| 旋转变量 | 围绕网络姿态的局部右乘旋转向量增量，零增量开始 | 重建约定；指数映射保证 SO(3) |
| 精度 | 优化 FP64、输出 FP32；保留原指标的 FP32 FK | 明确的执行约定 |
| 输出与状态 | 取第10次更新之后的姿态；每帧 fresh Adam，无跨帧状态 | 重建约定；不做 early-stop、最优迭代挑选或回退 |
| 局部自由度 | 只优化下表指定关节，其余 local 6D 条目逐位保持原值 | 肩肘有论文依据；头/髋具体列表为重建约定 |
| 目标方向 | 对原始插值矩阵做最近 SO(3) 投影，仅用于 IK | 原 processed 与网络输入保持不变；奇异/反射/缺失 active 数据拒绝 |
| 其他信息 | 不使用 GT 方向、GT 位置、GT 校准、隐藏 IMU、加速度目标或额外时间平滑 | 仅消费 prediction[t] 与可用 orientation[t] |

被引用的 [AvatarPoser §3](https://static.siplab.org/papers/eccv2022-avatarposer.pdf) 在论文中给出 IK lr1e-3，但其 [公开函数](https://github.com/eth-siplab/AvatarPoser/blob/0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0/models/network.py) 默认 lr1e-1、5次，优化的是手部位置。这里采用论文学习率作为预先声明的重建值，没有移植其位置目标；尤其没有沿用 AvatarPoser 中 MPJVE 的“速度”含义。

## 传感器与骨骼映射

| 输入位置 | 方向约束的 SMPL 段 | 优化的局部旋转 |
|---|---|---|
| 左腕 | 18，左前臂/肘段 | 16/18，肩与肘 |
| 右腕 | 19，右前臂/肘段 | 17/19，肩与肘 |
| 左口袋 | 1，左髋/大腿段 | 0/1，pelvis 与左髋 |
| 右口袋 | 2，右髋/大腿段 | 0/2，pelvis 与右髋 |
| 头 | 15，头 | 12/15，颈与头 |

依据公开合成的 `ji_mask=[18,19,1,2,15,0]`，而非旧 `leaf_joints=[20,21,7,8,12]`。腕部设备追踪前臂段，直接套用旧 leaf 列表会把口袋误当脚踝。raw DIP 索引保持 `[7,8,9,10,0]`。[Issue16](https://github.com/FIGLAB/IMUPoser/issues/16) 中是其他用户的提问，不能当作作者确认。

在 first8 训练池的41条录制上预先固定每25帧抽样，检查4,241个样本：IMU 与既定段的 GT 全局方向平均差为左腕1.50°、右腕1.40°、左髋3.36°、右髋2.52°、头0.37°。这只是输入/映射诊断；没有从 GT 拟合或修改任何校准矩阵。训练池部分插值方向不严格正交，支持显式区分网络原输入与 IK 旋转目标。[完整训练侧审计](results/dip_imu_only_ik_v1_20261116/train_conventions.json)。

## 全量执行与核验

使用最终 checkpoint `epoch_002_9d8a49a977cb.ckpt`，SHA-256 `996a3e2df21ac675df8f777d975d0700f3836c1d835bb898044a714e6cb5f7d8`。训练协议、两个原 checkpoint、processed 数据及 F03 结果均保持原状。模型是在 GPU 上训练的；本轮只在 CPU 对已有最终预测做姿态变量优化和指标测量，没有重训或重新进行网络推理。

两个 CPU 分片分别运行228个配对，各用2线程、frame_batch256，并行耗时约4分43秒。共新增577,872帧 refined 姿态。每个 `inputs/` NPZ 只包含原网络预测、可用 IMU 方向、mask/录制身份和哈希；不含 GT。`refined/` 保存结果、每帧11点损失轨迹及方向残差；指标阶段才单独读取原缓存中的 GT。

可用 IMU 的平均角残差由 12.60312° 降至 10.95081°，所有577,872个 mask-frame 的方向 MSE 均未增加，说明优化确实朝所定义的 IMU 目标移动。这不保证整条运动更平滑；三阶差分对逐帧修正敏感，是 jitter 上升的一种可能解释，当前并未单独隔离噪声、目标投影和优化离散性的因果贡献。

- 10项 IK 数值/隔离测试和4项解析汇总测试通过。
- 独立校验1,424个原始结果文件、全部456个输入/输出配对和所有受保护源文件/权重/数据哈希。
- 可用 IMU 输入与 processed 逐位一致，缺失通道为零；11,437,050个未优化 joint-frame 的原始6D条目逐位不变。
- 独立重算整体汇总，最大绝对差1.78e-15。
- 固定首尾mask×首尾录制的0/1/124/125/末帧，以 batch1 重跑10步，与原batch256保存结果最大差0。
- 同四个固定配对、两个模型重新测量完整录制，FK chunk11对比32，四指标最大绝对差1.194e-12，小于事先阈值1e-4。

[独立验证 JSON](results/dip_imu_only_ik_v1_20261116/independent_validation.json) · [元数据校验清单](results/dip_imu_only_ik_v1_20261116/MANIFEST.json) · [执行前证据和决定](results/dip_imu_only_ik_v1_20261116/SOURCE_FINDINGS.md)

实现：[imu_only_ik.py](../tools/imu_only_ik.py)；协调和断点续算：[eval_imu_only_ik.py](../tools/eval_imu_only_ik.py)；验证：[validate_imu_only_ik.py](../tools/validate_imu_only_ik.py)。完整运行目录为 `outputs/local_dip/dip_imu_only_ik_v1_20261116/`，含原始与追加验证 manifest。批处理吞吐不等于实机端到端延迟。

## 论文口径与下一步

[剩余评估口径核对](PAPER_EVALUATION_ALIGNMENT.md) 给出逐项证据、21项对照、12展示分组和平均/shape/mesh敏感性。DIP 的微调→滚动推理→IMU-only IK→四指标全24配置链路已跑通。严格数值复现还受原作者未公开细节和历史 AMASS 配方差异限制；测试结果没有用于重选本次学习率、关节、预算或额外平滑。现有无IK与有IK结果都保留，以便后续获得作者配置时直接核对。
