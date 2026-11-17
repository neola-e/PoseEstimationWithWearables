# IMUPoser 评估指标与数值验证

完成日期：2026-11-13。状态：四项指标实现、已知姿态检查及缓存预测复核完成。入口为 [eval_pose_metrics.py](../tools/eval_pose_metrics.py)，计算模块为 [paper_metrics.py](../tools/paper_metrics.py)。本次只测量现有预测，没有训练或重新运行姿态网络。

定义依据为 [IMUPoser 第 5.3 节及 Table 2](https://arxiv.org/html/2304.12518v1)。Jitter 的离散公式、平移处理及 mesh 默认开关参照论文引用的 [TransPose FullMotionEvaluator](https://github.com/Xinyu-Yi/TransPose/blob/37be773ceceda49160717311a4f74a695817b8d3/articulate/evaluator.py)。这使细节可复核，但不代表已找回 IMUPoser 作者未公开的全部评估设置。

## 指标定义

所有 SMPL 位置与输入平移都以米为单位，方向为全局坐标系中的方向。对每帧、每个选定点先计算误差，再平均。当前选用全部 24 个 SMPL 关节，包括 pelvis，以及全部 6,890 个对应网格顶点。

| 输出字段 | 意义与单位 | 对齐和平均 |
|---|---|---|
| `mpjre_deg` | MPJRE：全局关节旋转的 SO(3) 测地角误差，度 | 先由局部旋转做 FK；不消除根旋转。每帧平均 24 个关节 |
| `mpjpe_cm` | MPJPE：关节欧氏位置误差，cm | 每个姿态减去自己的 pelvis 位置后比较，24 个关节含零误差 root |
| `mpjve_cm` | MPJVE：网格顶点欧氏位置误差，cm | 使用与 MPJPE 相同的 pelvis 平移对齐，平均全部 6,890 顶点；不是速度或关节位置误差 |
| `jitter_m_s3` | 预测关节轨迹的平均 jerk，m/s³ | 使用空间对齐前的全局关节位置，包含提供的平移；不与 GT 相减 |
| `jitter_100_m_s3` | 同一 jitter 除以 100 | 对应论文表格的 (10^2\,m/s^3) 单位；不是乘以 100 |
| `local_rotation_deg` | 局部关节旋转误差，度 | 仅作附加诊断，不替代全局 MPJRE |
| `reference_jitter_m_s3` / `reference_jitter_100_m_s3` | GT 轨迹自身的 jerk | 作为参考单独报告，不从预测 jitter 中扣除 |

空间误差只对齐 pelvis 平移，不做 heading、root 旋转、尺度或 Procrustes 对齐，也不对齐网格质心。厘米距离等于米距离乘以 100。使用 atan2 形式计算旋转测地角，与合法 SO(3) 上的 trace/acos 定义等价，并改善零角和小角的数值稳定性。对明显非正交、反射或非有限矩阵直接拒绝，不静默修复。

## Jitter 的时间约定

对于均匀采样的位置 (p_t)，本实现计算：

```text
jerk[t] = (p[t+3] - 3*p[t+2] + 3*p[t+1] - p[t]) * fps^3
jitter = mean over valid time samples and 24 joints of ||jerk[t]||_2
```

长度 T 的完整录制提供 `max(T-3, 0)` 个时间样本。长度不足四帧时 jitter 为 JSON `null`，计数为零，不伪装为零抖动；这种录制仍参与空间误差统计。

预测与 GT 相同并不意味着 jitter 为零：相同但有加加速度的运动，其空间和旋转误差均为零，而两者各自的 jitter 都可大于零。反之，完全不动的错误姿态可以有零 jitter，因此 jitter 低不能单独证明姿态准确。

计算 mesh 的批大小不会切断时间轨迹。已有缓存的 `sequence_ids`、`start_frames` 和 `offsets` 必须逐录制从零连续覆盖，才能拼回完整录制后计算 jerk；间隙、重叠和重复窗口会被拒绝。不同录制之间不求导。若只有 offsets，则每个区间视为独立序列，不猜测它们相邻。只有单序列的 `lengths` 时先去掉 padding；不支持把不带有效边界的 padded batch 当成一条轨迹。

独立网络窗口之间若产生跳变，只要它们属于同一连续录制，就应在 jitter 中体现。本工具不会额外平滑预测以降低指标，也不会用计算批边界掩盖这些跳变。

## 身体形状与 mesh 约定

- 默认使用提供的 SMPL 模型的 mean shape，预测与 GT 使用同一个身体。当前验证资产为已有 male SMPL，文件 SHA-256 随结果保存。
- 可通过 `--betas` 提供十个固定形状系数，仍共享给一对预测与 GT；不接受随帧变化的身体形状。
- pose blend shapes 默认为关闭，与继承的 SMPL 和所核查 TransPose evaluator 默认一致。可用 `--pose-blendshape` 显式开启，结果中记录这个设置。测试确认该开关可改变 mesh 误差而不改变关节位置误差。
- IMUPoser 正文没有明确该开关和全部 shape 细节，因此此默认是已公开、可切换的实现约定，不冒充已证实的作者设置。未来论文数字比较必须保持这一设置一致，必要时报告敏感性分析。
- 可在 NPZ 中提供 `predicted_translation_m`、`target_translation_m`，形状均为 `[T,3]`；省略时为零。空间指标消除其 pelvis 平移，jitter 则保留平移运动。当前 IMUPoser 缓存没有预测轨迹，所以采用零平移。

## 汇总规则

主输出 `frame_weighted` 对旋转、关节和顶点误差按有效帧数加权；jitter 按每条录制的有效三阶差分样本数 T−3 加权。这两类分母不能混用。`sequence_equal` 是辅助汇总，每条有定义的序列给同等权重；没有 jerk 样本的序列不进入 jitter 平均。

每条录制保留帧数、jerk 样本数、各指标及逐关节统计。当前工具一次测量一个配置，不把某个三位置配置的结果冒充论文全部设备组合的平均值。论文没有充分给出所有跨配置权重细节，后续完整基准应先固定配置汇总协议。

## 使用方式

输入 NPZ 中预测和参考必须已配对，支持 `[T,144]`、`[T,24,6]` 或 `[T,24,3,3]` 局部姿态。6D 旋转遵循仓库的列向量约定。矩阵输入必须是合法 SO(3)。若有标量 `fps` 字段，必须与 `--fps` 一致；所有时间样本按该固定采样间隔解释。

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tests/test_paper_metrics.py

conda run --no-capture-output -n imuposer python tools/eval_pose_metrics.py \
  --input outputs/local_dip/dual_wrist_dev_epoch59_20261110/evaluation/predictions.npz \
  --pose-key pose_lw_rw_h \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --fps 25 --fk-chunk 32 \
  --output outputs/local_dip/metric_validation_20261113_recheck/metrics.json
```

工具拒绝覆盖已有输出。JSON 包含输入、模型与源码哈希、实际导入路径、软件版本、shape 和 mesh 设置。仅有 CPU 和已有授权 SMPL 资产即可测量；无需 checkpoint、训练集或网络推理。旧实验工具及结果未改写。

## 已完成的验证

[test_paper_metrics.py](../tests/test_paper_metrics.py) 的 **20 项测试通过**，包括 13 项不依赖 SMPL 资产的检查和 7 项使用现有 SMPL 的检查；没有资产时后七项明确跳过。旧 sensor metrics 的 7 项回归测试也通过。

检查覆盖以下可解析情况：

| 情况 | 预期与检查结果 |
|---|---|
| 相同静态姿态 | 四项指标及附加局部误差均为零 |
| 相同但变化的姿态 | 姿态误差为零；预测/GT jitter 相同且可为非零 |
| 根关节绕 Z 轴 90° | 全局 MPJRE 为 90°；局部关节平均误差为 90/24°；空间误差与静态几何解析式一致 |
| 仅旋转末端关节 | 关节位置不动，MPJPE 近零；表面顶点移动，MPJVE 非零 |
| 关节位移 0.01 m / 顶点位移 0.02 m | 验证 ×100 的厘米转换、root 是否计入、关节与顶点分母不同 |
| 整体平移 | pelvis 对齐后的空间误差为零；平移中的 jerk 仍计入 jitter |
| 常数、线性、二次轨迹 | 三阶差分为零，允许浮点容差 |
| 单轴 (p(t)=t^3) | jerk 为 6 m/s³；同一离散数组的 fps 翻倍，数值变为 48 m/s³ |
| 时间边界与短序列 | 不跨录制差分；短序列返回 null；padding 排除；非法 offsets/间隙/重叠拒绝 |
| 计算批大小及形状开关 | 固定输入重复计算稳定；共享固定 shape；pose blendshape 设置显式记录 |

端到端使用 E04 的同一份冻结开发集预测：**11 条录制、25,423 帧、25,390 个 jerk 时间样本**。计算批大小 32 与 11 的所有汇总指标完全相同。第三次使用批大小 19，通过独立 NumPy 距离、trace/acos 和第三差分重算，最大差异为 **0.00005493**（对应指标单位），低于 0.0001 容差；角度细小差异来自稳定 atan2 与 trace/acos 的浮点处理。

| 指标 | 同一份开发集缓存的 frame-weighted 结果 |
|---|---:|
| MPJRE | 21.292074° |
| MPJPE | 9.451584 cm |
| MPJVE | 10.257982 cm |
| 预测 jitter | 78.290040 m/s³，即 0.782900 × (10^2\,m/s^3) |
| GT jitter | 82.186566 m/s³，即 0.821866 × (10^2\,m/s^3) |
| 附加局部旋转误差 | 19.802034° |

新 MPJPE 与旧 E04 同为 frame-weighted 的结果相差约 0.000000011 cm。这里使用 frame-weighted，而 E04 教学报告的主表使用 subject-equal；两种平均口径不要混为性能变化。

这些数字用于验证测量工具，仍来自 AMASS 预训练模型的离线窗口开发集缓存；没有完成 DIP 微调、在线推理或 refinement，**不直接对比论文 Table 2**。

证据：[冻结测量约定](../outputs/local_dip/metric_validation_20261113/PROTOCOL.json)、[批大小 32 结果](../outputs/local_dip/metric_validation_20261113/metrics_chunk32_checked.json)、[批大小 11 结果](../outputs/local_dip/metric_validation_20261113/metrics_chunk11_checked.json)、[独立验证结果](../outputs/local_dip/metric_validation_20261113/validation.json)、[独立验证代码](../outputs/local_dip/metric_validation_20261113/validate_metrics.py)。原始测量输出保存在忽略的本地目录，不复制授权数据或模型资产。

## 2026-11-16 后续测量

同一原指标实现现已用于全24mask的最终DIP微调前后比较及IMU-only IK对照，并通过独立缓存/汇总/分块核验，见[微调结果](DIP_FINETUNING_COMPARISON.md)、[IK结果](IMU_ONLY_IK.md)及[shape、mesh和平均方式敏感性](PAPER_EVALUATION_ALIGNMENT.md)。本文件保留早期M01已知姿态验证的原记录；Python指标源码未改动。
