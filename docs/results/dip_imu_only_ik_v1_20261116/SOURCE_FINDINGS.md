# IK 与评估口径：执行前证据和决定

工作日期：2026-11-16，America/Chicago。本文件在任何新测试集 IK 结果产生前保存。

- IMUPoser v1 §4.4：以 IMU 的绝对方向约束骨骼，MSE + Adam，每帧 10 次；腕部优化肩、肘。未给出 IK 学习率、全部头/髋索引、参数化、优化状态重置及目标投影细节。https://arxiv.org/html/2304.12518v1#S4.SS4
- FIGLAB/IMUPoser `a0e9d33b9c57d6737d711185b62084e17684b0e2` 的公开目录和 Python 源码未提供相应姿态 IK 优化器或论文评估循环；数学 inverse_kinematics_R 仅是 global/local 旋转转换。https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2
- 公开预处理 `ji_mask=[18,19,1,2,15,0]`，前五项用于本任务。raw DIP 使用 `[7,8,9,10,0,2]`；腕部设备对应前臂段。旧 `leaf_joints=[20,21,7,8,12]` 不适用于该输入。Issue16 的跟帖是其他用户提问，不能当作作者确认。https://github.com/FIGLAB/IMUPoser/issues/16
- 被引用的 AvatarPoser 论文 §3 的 IK 学习率是 1e-3；其公开代码 `models/network.py` 默认 1e-1、5次，且优化手位置。不能把那个位置目标或其 MPJVE（速度）的定义移用于 IMUPoser 的网格顶点指标。论文：https://static.siplab.org/papers/eccv2022-avatarposer.pdf ，代码 commit `0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0`。

本次预先决定：保持最终两轮 checkpoint、原 F03 全24mask测试数据/预测和四指标。只增加每帧10次 Adam、lr1e-3、3×3方向MSE。选择感知段及直接父关节，联合处理全部可用传感器；头/髋索引是明确披露的重建约定。用局部旋转向量增量，FP64优化、FP32输出；无额外关节/时间先验，未优化局部量保持原值。每帧初始化与 Adam 独立；批量 loss 对每帧MSE求和，避免batch size改变Adam epsilon的有效意义。取最终更新，无 GT 校准、GT损失、测试调参、最优迭代选择或回退。

原 ori 是矩阵逐元素插值，可能不严格位于 SO(3)。仅在 IK 目标中投影到最近旋转，原网络输入和 processed 文件保持不变。训练池 first8 的检查仅检验既定映射/数据约定，不拟合任何校准量；具体结果另见 train_conventions.json。

论文24组合到12展示类：忽略左右，按腕/口袋/头的数量分组。保留主汇总24mask等权，并输出12组等权、序列等权和按受试者的辅助汇总，不能凭哪个更接近论文而更换主规则。作者的精确聚合、rolling overlap/startup、shape betas仍待确认；已知AMASS历史25mask/训练语料与epoch差异保留。无IK和有IK均使用相同规范指标，MPJVE始终为网格顶点位置误差。

