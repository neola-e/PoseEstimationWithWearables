# 论文评估口径核对：2026-11-16

当前已完成定义、数据、实现与结果的逐项核查；**可运行的 DIP 流程已包含微调、滚动推理、IMU-only IK 和全24配置四指标，但严格作者设置仍有未决项**。以下记录明确哪些由原文/公开代码支持，哪些只是本地固定约定，不通过接近论文分数来倒推作者做法。原2026-11-11审计及F03基线保留原记录。

[21项完整对照 CSV](results/dip_imu_only_ik_v1_20261116/paper_alignment.csv) · [IK 结果和实现](IMU_ONLY_IK.md) · [冻结设置](results/dip_imu_only_ik_v1_20261116/PROTOCOL.json)

## 已核对的口径

| 项目 | 当前确认 |
|---|---|
| 数据与预算 | first8最终训练、last2测试；开发集选K=2；固定checkpoint、固定19条录制；测试不选模型 |
| 输入与设备 | 原25Hz、acc/30、全局方向、缺失置零、全部24个1–3位置mask，排除五位置global |
| 方向段映射 | 公开合成ji_mask为18/19/1/2/15；未用旧leaf列表；训练侧检查只诊断，不拟合GT校准 |
| 指标 | 全局MPJRE、pelvis平移对齐的24关节MPJPE和6890顶点MPJVE；不对齐root旋转；MPJVE不是速度 |
| Jitter | 预测运动jerk，25³三阶差分，序列内T−3分母、单位除100；不跨录制边界或减GT |
| IK已实现部分 | 只用可用IMU方向，MSE+Adam，每帧10次；肩肘优化；无GT推理输入；保留raw对照 |
| 24→12展示类 | 忽略左右，按手部/口袋/头数量分组：3种单位置类、5种双位置类、4种三位置类；每类成员保存在per_group.csv |

定义与设备分组依据 [IMUPoser §3–6](https://arxiv.org/html/2304.12518v1)。离散jerk/mesh执行沿用此前已验证的工具，当前四指标源码未改动。

## 原作者细节仍无法确认

| 项目 | 本地固定决定 | 为什么仍是未决项 |
|---|---|---|
| IK学习率 | lr1e-3，来自引用的AvatarPoser论文 | IMUPoser未列值；AvatarPoser公开函数默认值和目标也与论文/本任务不同 |
| 头/髋优化列表与参数化 | 感知段+直接父关节，局部旋转向量增量；方向目标投影SO(3) | 没有完整作者IK实现可逐项比对；肩肘有文字依据，其余为已披露重建约定 |
| IK状态与输出 | 每帧fresh Adam，取第10次更新后结果，无迭代选择/回退 | 原文没有这些边界规则 |
| 滚动步长与起始 | 保留F03的stride1、末帧输出、短前缀 | 原文同时写125历史和1-sample overlap；公开源码未找到完整评估循环，不能据此证明作者所有语义 |
| 平滑及因果性 | 保留原acc居中5帧均值；ori矩阵插值；原NaN填补 | acc有2帧/80ms未来依赖；原填补、插值也未形成严格原始流因果证明。IK只消费同帧测量，不新增未来依赖 |
| 身体形状 | 主结果mean male、beta0；预测和GT共享 | processed DIP保存的是默认beta全1，非个体拟合形状；作者评估beta未公开 |
| Mesh开关 | 沿公开FK默认，pose blendshape关闭 | 作者实际评估开关未给出；灵敏度另行报告 |
| 最终平均方式 | mask内按帧/T−3，再24mask等权 | 论文展示12类，但未充分给出跨录制/受试者/组合的完整权重；不能把展示分组等同于整体加权 |

方法依据：[IMUPoser §4.4/5.3](https://arxiv.org/html/2304.12518v1#S4.SS4)，[公开代码 a0e9d33](https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2)，[AvatarPoser论文](https://static.siplab.org/papers/eccv2022-avatarposer.pdf)及[其IK函数](https://github.com/eth-siplab/AvatarPoser/blob/0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0/models/network.py)。公开数学函数inverse_kinematics_R只是旋转转换，不等于本文姿态优化器。

## 已测量的汇总敏感性

以下均为同一批最终IK预测，没有重训、重新推理、改shape或重新选择模型。主口径在查看结果前固定，不因辅助数值更接近论文而更换。

| 汇总 | MPJRE ° | MPJPE cm | MPJVE cm | Jitter，100 m/s³ |
|---|---:|---:|---:|---:|
| 24mask等权；mask内按帧/T−3（主） | 22.16620 | 9.42636 | 11.82258 | 1.22145 |
| 24mask等权；mask内序列等权（辅） | 22.04929 | 9.06523 | 11.41680 | 1.59500 |
| 12展示类等权（辅） | 22.74294 | 9.71844 | 12.16981 | 1.19235 |
| 24mask等权；两名受试者等权（辅） | 22.20689 | 9.45620 | 11.86733 | 1.21784 |

仅将24mask等权改成12展示类等权，MPJVE即从11.82258变为12.16981 cm，差约0.34722 cm。序列等权对jitter的影响也很明显。分组成员和各类明细见 [per_group.csv](results/dip_imu_only_ik_v1_20261116/per_group.csv)，所有平均口径及两个分支见 [aggregation_sensitivity.csv](results/dip_imu_only_ik_v1_20261116/aggregation_sensitivity.csv)。

## Shape 与 mesh 的固定案例敏感性

额外作探索性测量：首/末mask×首/末录制四个完整配对，对raw/IK两个分支分别改变一个几何设置，不改变任何预测或主口径。这是固定案例诊断，不是全24mask的替代总体分数，也不推断作者到底用了哪个设置。

| 单独改变的设置 | MPJPE变化范围 | MPJVE变化范围 | 其他核验 |
|---|---:|---:|---|
| 两侧共享beta由0改为全1 | −0.728至−0.290 cm | −1.087至−0.399 cm | 全局角度不变；jerk因骨长变化而改变 |
| 开启pose blendshape | 0 | +0.032至+0.105 cm | 关节角、关节位置和jitter均不变 |

变化为“替代设置减主设置”；范围覆盖四个配对×两个分支。[完整探索性结果](results/dip_imu_only_ik_v1_20261116/mesh_convention_sensitivity.json)。Shape差异的量级大于本次IK约0.146cm的整体MPJVE改善；但抽查案例不能当作总体差异或作者口径证明。

## 与论文数值的关系及剩余工作

[论文Table2](https://arxiv.org/html/2304.12518v1#S6.SS3) 报告DIP上的MPJRE23.9°、MPJPE9.7cm、MPJVE12.1cm、jitter1.9（100m/s³）。当前主结果22.16620°/9.42636cm/11.82258cm/1.22145，与这些数值接近或更低，不能据此声称严格复现或优于作者；上面的未决测量约定足以影响这种比较。

历史AMASS初始化仍有已知差异：25mask训练（多global）、语料/过滤及保留时长口径、实际65轮并选Epoch59，而论文描述24mask/80轮。37.73小时保留帧与论文63小时原始语料不宜直接相减当作缺失量。当前DIP微调不能抹掉初始化的训练历史。[原审计](REPRODUCTION_AUDIT.md)保留证据。

接下来优先补足作者实际IK/评估配置的证据，并在需要严格AMASS配方时另行规划24mask训练。已有raw/IK预测和14项测试、独立核查可直接用于新证据到来后的核对。消费者设备数据、严格实时因果预处理和端到端延迟属于第二阶段，本次CPU批量评估不代替它们。当前没有根据测试集重新挑选IK学习率、关节或平滑规则。
