# IMUPoser 论文复现目标与 Epoch 59 审计

审计日期：2026-11-11。对象：本地 `IMUPoserGlobalModel_global-10062026-021332` 训练运行及 Epoch 59。由 Alex 提出，Codex 协助核查。

**当前决定：保留 Epoch 59，暂不从头重训 AMASS。先补齐 DIP 评估协议与微调流程。** 该模型应标记为“公开代码配方的 AMASS 预训练基线，含已知论文差异”。若最终目标是严格对齐论文训练配方，24-mask、训练轮数和数据覆盖问题解决后，需要另开可追溯训练运行。仅删除评估中的 `global` 不能消除既有权重的训练差异。

本次完成目标定义、论文对照、数据清单、训练记录及 checkpoint 审计；没有启动训练或模型推理。已有 E02–E04 传感器分析保留为附加研究记录，当前优先级降低。

**2026-11-13 进展更新：** 四指标工具与数值验证现已完成，见 [POSE_METRICS.md](POSE_METRICS.md)。滚动末帧推理、逐帧依赖追踪及与分块的开发集定量比较也已完成，见 [ROLLING_INFERENCE.md](ROLLING_INFERENCE.md)。步长 1 为显式实现约定；论文的 overlap 表述、原始流预处理因果性仍待进一步核实。下文对照表和 CSV 保留 2026-11-11 的审计状态，不回写历史证据；DIP 微调、refinement 和完整配置评估仍待执行。

**2026-11-14 进展更新：** DIP 微调协议已冻结，全部前八人录制已按 raw 内容匹配；Epoch 59 的 3 步实际训练、梯度/参数检查、保存重载及确定性续训检查通过，见 [DIP_FINETUNING.md](DIP_FINETUNING.md)。本次为 CPU 工程短跑，未启动正式 30-epoch 开发训练或前八人最终训练，也没有新的 test 结果。24-mask 只应用于新微调，初始化的原 25-mask 差异仍保留。

**2026-11-16 进展更新：** 开发集30轮选K=2、前八人最终两轮GPU微调及全24配置对照已完成，见 [DIP_FINETUNING_COMPARISON.md](DIP_FINETUNING_COMPARISON.md)。IMU-only IK已依据论文重建，完成456组测试及独立核验：三项姿态误差小幅下降，jitter上升16.51%，见 [IMU_ONLY_IK.md](IMU_ONLY_IK.md)。[剩余评估口径核对](PAPER_EVALUATION_ALIGNMENT.md)记录21项证据与平均/shape/mesh敏感性；作者IK精确参数、窗口、shape和聚合细节仍未完整找回。下文保留原审计日期状态，不将当前工程跑通等同严格论文数值复现。

## 复现范围与完成标准

| 阶段 | 范围 | 交付与完成标准 |
|---|---|---|
| 第一阶段 | DIP 定量评估 | 固定预训练权重及其差异说明；完成有明确拆分和选择规则的 DIP 微调；实现并验证滚动推理、IMU 方向 refinement、四项指标和全部 24 个论文配置；输出每配置结果及与论文对应的汇总 |
| 第二阶段 | 消费设备数据与实时系统 | 在作者 IMUPoser Dataset 上评估；核对同步、校准和设备位置处理；再补实时输入、姿态输出、实际运行延迟与设备位置识别 |
| 暂缓 | 新架构、新传感器配置搜索、新的性能优化 | 完整基线建立后，根据明确失败模式再安排；原始报告和结果不覆盖、不删除 |

第一阶段的“完成”指协议、指标和差异可解释，不以碰巧达到某个论文数字为判断条件。使用 Epoch 59 得到的结果必须保留训练配方差异标签；若未修复这些差异，应称为带已知差异的复现，而非完全相同的训练重现。

论文证据来自 [IMUPoser v1，第 3–6 节](https://arxiv.org/html/2304.12518v1)。公开代码基线为运行记录中的 commit [`a0e9d33`](https://github.com/FIGLAB/IMUPoser/tree/a0e9d33b9c57d6737d711185b62084e17684b0e2)。下表将论文描述、当前可检查实现和历史证据分开；公开代码本身也不等于论文实验的完整冻结版本。

## 论文与当前实现对照

可筛选的表格版本：[paper_implementation_comparison.csv](reproduction_audit_20261111/paper_implementation_comparison.csv)。

| 项目 | 论文设置 | 当前设置及证据 | 差异与待办 |
|---|---|---|---|
| 模型 | 两层 BiLSTM；60 输入、512 hidden、144 输出 | checkpoint 的 20 个网络张量共 10,680,976 参数；当前原始架构匹配 | 架构可继续使用；锁定实际导入路径 |
| 训练语料 | AMASS；文中描述约 63 小时 | 当前 loader 选择 18 个 AMASS `.pt` 文件，合计 37.7289 小时的保留帧 | 原始语料时长和预处理后时长口径不同；不能把两者相减当作缺失量。记录版本、过滤与遗漏，详见数据审计 |
| 训练 mask | 24 个位置组合 | 当前及所记录 Git 基线都枚举 25 个，额外包含五位置 `global` | 属于论文与公开代码的配方差异；模型不是仅用 `global` 训练。严格 24-mask 对齐需新训练 |
| 采样与输入 | 25 FPS；全局加速度、方向；加速度缩放 30；缺失位置置零 | 当前代码与输入约定一致，保留前五个位置，丢弃第六个 pelvis 输入 | 固定通道顺序、单位和 mask；不能因使用三位置 mask 就把网络输入改成 36 维 |
| 合成与预处理 | 虚拟 SMPL 位置；5 帧平均平滑 | 两段预处理脚本与记录中的 Git HEAD 相同；当前中间数据可重建最终 acc/ori/pose | 实际是 acc 五帧平滑、ori 矩阵逐元素插值、pose 轴角插值后转矩阵。论文对逐张量处理未完整细化，需记录和澄清 |
| 帧率过滤 | 文中未给出当前文件过滤清单 | 第一段脚本只接受 int(fps) 为 59/60/120；120 下采样至 60；短序列丢弃 | EKUT/KIT 的缺失需按原始帧率核查，不能默认为下载失败；不要在本次审计中擅自重写预处理 |
| 训练窗口 | 125 帧非重叠窗口 | 当前代码为 125 帧，保留短尾，batch 内补零 | 当前训练 loss 没有按有效长度屏蔽 padding；标注与现有评估排除 padding 的区别 |
| 优化器 | Adam，lr 3e-4，batch 256 | checkpoint 保存 lr、betas、eps、weight decay；当前实现为 Adam。batch 256 为未序列化类默认值，但步数交叉核查一致 | 维持 Adam 配方，不套用新 skill 的 LLM 优化器建议；batch 的历史证据等级单独标注 |
| 训练轮数 | 80 epochs | 实际运行覆盖 epoch 0–64；最佳 epoch 59；epoch 64 早停 | 65 个已完成 epoch 与论文 80 不同；先保留基线，严格配方训练时单独固定预算 |
| 模型选择 | 主文未给出足以重建全部内部验证细节的说明 | 当前实现先展开 mask，再随机拆分样本 90/10；按 `validation_step_loss` 选最小值 | 不是独立动作/受试者验证；相同动作的不同 mask 可能跨训练/验证。历史 split indices 未保存 |
| DIP 微调 | s_01–s_08 训练；s_09/s_10 测试 | 当前研究使用的 Epoch 59 是记录为 GlobalModel 的 AMASS 预训练权重；该运行无 DIP 微调记录 | 下一阶段需要可复现微调入口、数据拆分和模型选择记录；不能把模块存在等同于微调已完成 |
| 推理 | 在线滚动 125 样本；文中另写 1-sample overlap | 当前 E02–E04 是独立、不重叠 chunk，并使用 chunk 内全部上下文 | 必须明确步长、输出帧、暖启动和未来上下文；不要未经确认把 overlap 文字直接等同于某个 stride |
| 旋转 refinement | IMU 方向约束优化；每帧 10 次迭代 | 当前研究结果为 raw network 输出；GT-assisted 替换只是诊断 | 另实现并测试实际可用 IMU 输入驱动的 refinement；禁止用 GT 当推理输入 |
| 旋转和位置指标 | 全局 MPJRE、root 对齐 MPJPE | 已有全局/局部角度和 root-relative MPJPE | 核对关节集合、root、shape 与加权；局部角度不能代替论文 MPJRE |
| 主指标及平滑性 | MPJVE 为网格顶点误差；另报 jitter | 当前研究未计算 MPJVE 和 jitter | 补 mesh 误差及 jerk 指标，明确单位、采样间隔、序列边界。MPJVE 不是速度误差 |
| 配置汇总 | 24 组合，结果按 12 类展示 | 当前主研究仅比较部分 mask，且包含论文设备预算外的五位置参照 | 导出全部 24 组合的每序列结果，再固定分组及平均规则；细节不足时保留待确认项 |
| 消费设备与实时 | 作者设备数据及系统验证 | 当前主要为 DIP 代理位置与离线推理 | 放入第二阶段；当前 GIF 和 CPU 工作流耗时不能代替实机精度或端到端延迟 |

## 当前数字为何不能直接对比论文

| 当前数字 | 实际含义 | 阻止直接比较的原因 |
|---|---|---|
| E02 五位置 MPJPE 7.8866 cm | Epoch 59，DIP test 全部保留帧，五输入，离线 chunk | 五位置参照与论文 1–3 位置配置口径不同，且缺少论文微调/在线/refinement 流程 |
| E02 三位置 MPJPE 9.4112 cm | 仅 `lw_rp_h`，同一 test split 的 raw network 结果 | 单配置不能代表全部论文组合；其余协议差异仍存在 |
| E04 两腕加头 MPJPE 9.4467 cm | s_07/s_08 开发数据，subject-equal 汇总 | 不同人物与 split；与论文 test 汇总数接近没有可比性 |
| 训练验证损失 0.0101725226 | 6D pose MSE 与 FK joint-position MSE 的合成目标 | 不是 cm 单位的 MPJPE，也不是论文主指标 MPJVE |

第一阶段先消除评估协议差异，再判断残余差异是否需要训练修正。测试集已参与前期错误分析，因此后续固定协议报告可以使用它，但不能描述为从未接触过的盲测。

## 证据等级与历史边界

| 等级 | 能确认的事实 | 证据 |
|---|---|---|
| 历史文件直接记录 | checkpoint epoch/step、序列化配置实例、优化器状态、callback、65 条验证记录、运行命令和软件清单 | [checkpoint metadata](reproduction_audit_20261111/checkpoint_metadata.json)、[run metadata](reproduction_audit_20261111/run_metadata.json)、[validation history](reproduction_audit_20261111/validation_history.csv) |
| 今天可重复检查 | 当前文件清单与哈希、张量形状/有限性、预处理重建、当前 mask 枚举、当前源码与 Git HEAD 的差异 | [数据清单](reproduction_audit_20261111/processed_data_manifest.csv)、[逐序列清单](reproduction_audit_20261111/processed_sequence_manifest.csv)、[重建核查](reproduction_audit_20261111/preprocessing_reconstruction.csv)、[源码差异](reproduction_audit_20261111/source_comparison.csv) |
| 交叉支持但非历史证明 | 当前 18 文件按现有规则的样本数正好解释历史训练步数 | [审计汇总](reproduction_audit_20261111/audit_summary.json)；这不能证明当时的每个文件与今天哈希相同 |
| 未恢复 | 当时逐文件哈希、文件顺序、随机拆分索引、随机数完整状态、每 mask 的 batch 暴露、AMASS 下载版本及全部未提交源码快照 | W&B 只记录 Git commit 和 Config 对象 repr；checkpoint 的 Config 类属性也未完全保存 |

因此，对“Epoch 59 使用了哪些子集”的最准确表述是：**当前训练目录、既有预处理日志和 checkpoint 步数支持下面的 18 子集清单；没有训练当时的文件哈希清单可逐文件认证历史身份。**

## AMASS 与预处理审计

当前原始目录包含配置中的 20 个子集；25 FPS 训练目录只有其中 18 个。`preprocess_full_retry.log` 明确记录 EKUT 和 KIT 为 “not supported”。原始帧率的全量统计见 [raw_amass_summary.json](reproduction_audit_20261111/raw_amass_summary.json)，逐录制候选清单见 [raw_amass_inventory.csv](reproduction_audit_20261111/raw_amass_inventory.csv)。原始 ZIP 的 CRC 仅用于清单描述，不冒充加密内容哈希；本次没有对全部原始压缩文件计算 SHA-256。

全量检查 **14,244** 个原始候选录制后，按当前规则有 **4,686** 个因帧率被排除、**4** 个因长度被排除，剩余 **9,554** 个。每个子集的保留数量都与当前 processed 数据一致；这是数量核对，不是逐条原始动作内容匹配。

| 过滤位置 | 原因与数量 |
|---|---|
| EKUT | 全部 349 条均为 100 Hz，当前代码不接受 |
| KIT | 全部 4,232 条均为 100 Hz，当前代码不接受 |
| Eyes_Japan_Dataset | 57 条为 250 Hz，排除；保留 693 条 |
| MPI_mosh | 39 条为 100 Hz，排除；保留 38 条 |
| TCD_handMocap | 9 条为 150 Hz，排除；保留 53 条 |
| CMU / BioMotionLab_NTroje | 分别有 3 / 1 条在转换至 60 FPS 后不超过 12 帧，排除 |

这些过滤来自与 Git HEAD 相同的预处理实现。若决定补进 100/150/250 Hz 数据，应形成新的数据版本和训练记录；不能把新语料悄悄替换后继续声称与旧运行相同。

| 当前参与全局训练的子集 | 序列 | 25 FPS 帧数 | 含短尾的 125 帧窗口 |
|---|---:|---:|---:|
| ACCAD | 252 | 40,262 | 443 |
| BioMotionLab_NTroje | 3,060 | 785,738 | 7,653 |
| BMLhandball | 649 | 153,314 | 1,564 |
| BMLmovi | 1,801 | 254,775 | 2,766 |
| CMU | 2,079 | 828,371 | 7,635 |
| DanceDB | 153 | 314,262 | 2,588 |
| DFaust_67 | 129 | 17,240 | 184 |
| Eyes_Japan_Dataset | 693 | 499,427 | 4,067 |
| HUMAN4D | 148 | 108,991 | 943 |
| HumanEva | 28 | 12,723 | 114 |
| MPI_HDM05 | 215 | 216,921 | 1,841 |
| MPI_Limits | 35 | 31,257 | 268 |
| MPI_mosh | 38 | 12,049 | 113 |
| SFU | 44 | 22,872 | 204 |
| SSM_synced | 30 | 2,819 | 39 |
| TCD_handMocap | 53 | 10,201 | 127 |
| TotalCapture | 37 | 61,665 | 511 |
| Transitions_mocap | 110 | 22,711 | 233 |
| **合计** | **9,554** | **3,395,598** | **31,293** |

上述 3,395,598 帧约为 **37.7289 小时**，不包含因 mask 复制形成的重复样本。`dip_train.pt` 为 41 条、105,493 帧；`dip_test.pt` 为 19 条、24,078 帧，两者不被当前 global train loader 选入训练文件。datamodule 的 setup 会构建 test dataset；构建或读取测试容器不等于把它交给训练/验证 loader。

本次对全部 20 个 processed 容器的 acc、ori、pose 共 **28,842 次序列字段检查**进行了第二段预处理重建，结果逐位相同。这说明今天的 25 FPS 数据与现存第一阶段中间数据、当前转换代码一致。它不证明第一阶段每条数据与原始 AMASS 录制的历史映射，也不证明 2026-10-06 当时的数据哈希。源脚本、当前文件 SHA-256 和逐序列字段哈希均已保存。

## 训练配置与 checkpoint 选择

| 项目 | 审计结果 | 证据边界 |
|---|---|---|
| 运行 | `wk8tfuyl`，启动于 2026-10-06 07:13:33 UTC | W&B 本地 metadata |
| 命令参数 | `--combo_id global --experiment IMUPoserGlobalModel --num_workers 0` | metadata；没有 `--fast_dev_run` 或 `--resume_from` 参数 |
| 权重 | Epoch 59；step 165,060；SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058` | 原始 checkpoint，非 Epoch 55 handoff 权重 |
| 配置实例 | GlobalModelIMUPoser；60 维输入；24 关节 6D；joint loss 开启；MSE；workers 0 | checkpoint 实例字段已序列化 |
| 优化器状态 | lr 0.0003；betas 0.9/0.999；eps 1e-8；weight decay 0；无 scheduler | 保存的参数组与 step；Adam 类型由当前及 Git 基线源码支持 |
| batch / 窗口 / 拆分 | 256 / 125 帧 / 90%–10% | 来自 Config 类属性与源码，不在实例字典中；以下训练步数核查提供一致性支持 |
| seed | 训练入口调用 seed_everything(42, workers=True) | 当前及 Git 基线代码；Config.torch_seed=0 不是该调用的 seed；完整历史 RNG 状态未恢复 |
| 训练 loss | pose MSE + FK position MSE；当前源码包含 padding | 不是位置误差 cm；当前验证方式也不是独立动作拆分 |
| 选择规则 | 最小化 validation_step_loss；保留最优五个 checkpoint | callback 的历史状态与 best_model.txt 相互一致 |
| 最佳值 | epoch 59，0.010172522626817226 | 65 条完整验证历史中的最小值，亦等于 callback best_model_score |
| 终止 | epoch 64；wait_count=5；stopped_epoch=64 | 最终 checkpoint 的 EarlyStopping 状态；0–64 共 65 epochs |
| 软件 | Python 3.10.21，PyTorch 2.7.0+cu128，Lightning 1.9.5，NumPy 1.23.5 | 当时 requirements 与 metadata；现代硬件兼容环境不同于原始 README |

W&B 的 `config.yaml` 中模型配置是 Python 对象 repr，不能从中恢复 batch 等所有超参数。记录的 Git commit 也不包含当时未提交的修改。审计保存了当前源码与该 Git HEAD 的逐文件哈希差异，避免把工作区源码直接当作完整历史快照。

![恢复的验证损失历史及 Epoch 59 选择位置](reproduction_audit_20261111/validation_curve.png)

图中的 loss 是该运行的模型选择指标，不是论文精度。绘图数据为恢复的 65 条历史记录，代码见 [plot_validation_history.py](reproduction_audit_20261111/plot_validation_history.py)。

## 样本数与步数交叉核查

按当前 loader 顺序、短尾保留、25 个 mask 和现有类默认值计算：

- 31,293 动作窗口 × 25 masks = **782,325** 个展开样本。
- `int(782325 × 0.9)` = **704,092** 个训练样本，验证 **78,233** 个。
- batch 256，末 batch 保留：训练 **2,751** batches/epoch，验证 **306** batches/epoch。
- `2,751 × 60 = 165,060`，与 epoch 59 checkpoint 完全一致。
- `2,751 × 65 = 178,815`，与最终 checkpoint 完全一致；训练输出的 `3057/3057` 亦等于 2,751 + 306。

W&B 的 trainer/global_step 在对应日志中分别为 165,059 和 178,814，checkpoint 则是完成更新后的计数，相差一。该差别与整个历史记录一致，不应据此判断缺失了训练步骤。

这些核查排除了明显的样本规模不一致，但不能唯一反推出每个文件或每个 split index。同样的总窗口数可能由不同内容组成。

## 25 masks 对复现口径的影响

当前字典由 **5 个单位置、10 个双位置、9 个三位置，以及 1 个五位置 `global`** 组成。去掉 `global` 后为 24 个论文预算内组合。记录的 Git 基线同样包含这个额外 mask，所以它不是本地最近几次传感器实验引入的训练修改。

`--combo_id global` 在该实现中用于建立五位置、60 维网络输入；dataset 内部仍逐一枚举所有 mask。它不表示模型只见过五位置输入。以后限制训练 mask 时应将“网络支持的输入槽位”与“训练枚举集合”分开，不要把所有配置的 `global` 字样一概删除。

| 可确定的影响 | 数量或含义 |
|---|---|
| 额外样本 | 每个原始动作窗口多一个副本；当前共多 31,293 个样本 |
| 总枚举规模 | 25 相对 24 增加 4.1667%；原始动作内容没有增加 |
| 拆分前名义曝光 | 每 mask 1/25；五位置输入占 4%；其余 24 合计 96% |
| 同一 motion window 的相关性 | mask 展开后随机拆分，存在不同 mask 跨训练/验证的风险 |
| 24-mask 假设计算 | 751,032 个样本；同一 90/10、batch 256 规则下约 2,641 train batches/epoch，而当前为 2,751 |
| 无法从记录确定 | 历史每 mask 的实际采样数，以及额外五位置训练对各稀疏配置精度的正负影响 |

因此，训练 recipe 有已知差异，但**没有证据说这个差异一定提高或损害稀疏输入精度**。要量化因果影响，需要控制其他变量的 24-vs-25 训练对照；这不是本次审计或下一步评估补齐的必要前置条件。

## 重训决定与下一步

1. **现在保留 Epoch 59。** 它是有完整优化器状态、验证选择证据和一致数据规模支持的可用初始化。没有发现需要立即丢弃权重的 NaN 或步数异常。保存原权重和旧结果，避免把训练 loss 当作复现成功证明。
2. **先补评估与 DIP 微调。** 固定指标、滚动窗口与 refinement 口径，再建立可重复微调和 checkpoint 选择。这样能判断评估协议差异，而不会混入一次昂贵的新预训练。
3. **若要求严格复现训练，之后另开新运行。** 在确认 AMASS 版本、100 Hz 数据处理、24-mask 集合、80-epoch 预算及验证规则后再训练。只续训现有 25-mask 权重到更多 epoch，不能让它等价于从未见过 `global` 的训练。
4. **开发与正式拆分分开记录。** 当前 s_07/s_08 保留给开发；如最终按论文把 s_01–s_08 合并微调，先冻结超参数和选择规则，另建 run，并明确 s_07/s_08 不再是独立验证集。当前旧 loader 不自动执行保留索引过滤。

| 优先级 | 下一项工作 | 完成条件 |
|---|---|---|
| P0 | 写清 DIP 评估协议中的未决细节 | 输出帧、窗口步长、暖启动、平滑、shape、root、汇总方式、jitter 单位全部明确；资料不足项标注来源与决定 |
| P0 | 补齐四指标的独立数值验证 | 已知姿态能验证；MPJVE 与 MPJPE、local 与 global 指标分开 |
| P1 | 建立 DIP 微调入口与数据过滤 | 使用明确训练/开发索引；不凭测试集选 checkpoint；完整记录实际配置和源快照 |
| P1 | 补滚动推理及 IMU-only refinement | 同 checkpoint 下分别验证开关差异；不得使用 GT 约束推理 |
| P1 | 完整评估 24 个配置 | 输出 per-sequence/per-mask 原始指标与论文对应汇总，保留权重配方差异标签 |
| P2 | 决定是否启动严格 AMASS 配方运行 | 数据覆盖、24-mask 和训练预算均已固定，且现有基线差异已定位 |

## 可追溯文件与复核命令

本次只读收集工具为 [audit_reproduction.py](../tools/audit_reproduction.py)。运行时只把统计、哈希和代码写到新目录，不复制数据或权重。已有原始/中间/25 FPS 数据和 checkpoint 的哈希在收集结束时重新检查。

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tools/audit_reproduction.py \
  --original-root /home/alex/research/IMUPoser \
  --output-dir docs/reproduction_audit_20261111_recheck
```

工具拒绝覆盖目标目录。全部证据索引见 [audit_summary.json](reproduction_audit_20261111/audit_summary.json)，收集校验见 [collection_validation.json](reproduction_audit_20261111/collection_validation.json)。源快照反映审计日，不追认成训练当时快照。W&B 二进制历史以只读方式解析并核对每个片段 CRC，读取格式参照 [W&B 官方 DataStore 实现](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py)；未更改训练环境依赖。

项目进度与既有实验见 [RESEARCH_LOG.md](RESEARCH_LOG.md)。本次范围是审计和复现计划，后续训练与评估待执行。
