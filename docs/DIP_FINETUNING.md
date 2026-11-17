# DIP 微调协议与短跑验证

日期：2026-11-14。状态：训练协议冻结、实际短跑和独立重载核查均已完成。由 Alex 指定执行两项工作：冻结训练协议，以及验证 Epoch 59 的数据加载、梯度更新、参数变化、checkpoint 保存和重载。正式多轮训练与最终测试不属于本次短跑。

准备入口：[prepare_dip_finetuning.py](../tools/prepare_dip_finetuning.py)。训练状态与数据工具：[dip_finetuning.py](../tools/dip_finetuning.py)。短跑入口：[smoke_dip_finetuning.py](../tools/smoke_dip_finetuning.py)。单元检查：[test_dip_finetuning.py](../tests/test_dip_finetuning.py)。

## 冻结的训练协议

机器可读协议为 [PROTOCOL.json](protocols/dip_finetuning_v1_20261114/PROTOCOL.json)，编号 `DIP_FINETUNING_V1`，SHA-256 为 `79bde74ead3f596c5806e3fa1d0cd5b07fcfb6260b0c9e81c218771ba50441ad`。与实验目录原件逐位相同的协议、数据清单和源码哈希保存在 `docs/protocols/dip_finetuning_v1_20261114/`，便于纳入版本控制。原始模型代码、licensed 数据、Epoch 59 和既有研究记录不改写；新入口保留相同模型架构和权重键名。

[论文第 4.3、5.3 节](https://arxiv.org/html/2304.12518v1) 给出 AMASS 初始化、DIP 前八人训练与后两人测试，以及训练架构和损失等信息。论文没有完整冻结 DIP 微调的预算、内部验证与模型选择细节；下表中的 30 epochs 上限、seed、开发划分与选择规则是本次明确约定，不能冒充找回的作者设置。

| 项目 | 冻结设置 |
|---|---|
| 初始化 | 原始 AMASS Epoch 59；step 165,060；SHA-256 `02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058` |
| 训练参数 | 全部模型参数参与更新；从新 Adam 状态开始，不继承 AMASS 的动量和步数 |
| 优化器 | Adam；lr `3e-4`；betas `(0.9,0.999)`；eps `1e-8`；weight decay 0；amsgrad/foreach 关闭 |
| 调度、裁剪、精度 | 无 scheduler、无梯度裁剪；float32；无 AMP/TF32；遇到非有限 loss/梯度立即报错 |
| 随机性 | Python、NumPy、PyTorch seed 均为 42；确定性算法开启；禁用 cuDNN benchmark |
| 数据和输入 | 25 Hz；60 维，位置顺序 lw/rw/lp/rp/h；加速度除以 30；保留继承的居中五帧加速度平均和方向重采样，不重复平滑 |
| 训练窗口 | 不重叠 125 帧，保留短尾；batch 内输入与目标补零；packed LSTM 接收实际长度 |
| Mask | 每个动作窗口枚举 24 个稀疏 mask：5 单位置、10 双位置、9 三位置；不加入五位置 global；通道数保持 60 |
| Loss | 144 维列向量 6D pose MSE＋全部 24 关节的 SMPL FK 位置 MSE，权重 1:1；位置以米为单位 |
| Padding 约定 | 训练 loss 明确保留公开代码的 padding 项；报告四指标时排除 padding。两者含义不同，不悄悄改变继承的训练配方 |
| 身体模型 | 现有 mean male SMPL、零平移、pose blend shapes 关闭；文件哈希进入协议 |
| 正式 batch | 256，shuffle=True，workers=0，drop_last=False；不自动改变 batch 或学习率 |
| 开发预算 | 最多 30 epochs，无早停，每个完整 epoch 做一次全开发集验证 |
| 开发 checkpoint 选择 | 选择最小全开发集 combined validation loss；相同值选较早完成的 epoch；验证批按 padded B×T 加权，避免不等长批直接等权 |
| 保存 | 每个 epoch 保存 last 和 best；记录模型、Adam、epoch/step、Python/NumPy/Torch RNG、采样器状态和协议哈希 |

### 明确的 24 个训练 mask

顺序也是协议的一部分：

```text
lw_rw_h, rw_lp_rp, lw_rw_rp, lw_rp_h, rw_rp_h,
lw_lp_rp, lw_rw_lp, lw_lp_h, rw_lp_h,
lw_rw, lw_lp, lw_rp, lw_h, rw_lp, rw_rp, rw_h, lp_rp, lp_h, rp_h,
lw, rw, lp, rp, h
```

Epoch 59 原有的 25-mask 预训练差异仍然存在；本次排除 global 只改变新的 DIP 微调样本，不能消除初始化的既有训练经历。

## 受试者与数据清单

先按受试者拆分，再生成窗口与 24 个 mask，避免同一动作的不同 mask 跨训练和开发集。使用本地 raw DIP 按继承的填补、裁剪、重采样及平滑规则重建 acc/ori/pose，与 processed 容器逐位匹配。每条录制保存 raw 文件哈希和三个 processed 内容哈希，排除项也保留原因。

1. **开发阶段：** s_01–s_06 训练，s_07/s_08 验证。原来保留的开发集不进入短跑或开发训练的梯度。
2. **最终论文划分阶段：** 开发阶段选定完成轮数 K（1–30）后，重新从同一个 Epoch 59 和新 Adam 开始，用 s_01–s_08 固定训练 K 轮。此时 s_07/s_08 的角色明确转为训练，不能再称为该最终模型的独立验证集。
3. **测试：** s_09/s_10 只用于固定模型的微调前后报告，不参与超参数、轮数或 checkpoint 选择。本次准备与短跑不打开 test 容器或这两人的 raw 文件。

完整身份清单和统计见 [data_manifest.json](protocols/dip_finetuning_v1_20261114/data_manifest.json)。前八人全部 41 条 processed 录制均找到唯一 raw 内容匹配；42 条 raw 候选中，s_01/02.pkl 因继承的有限值规则被排除，原因与文件哈希保留。原 s_07/s_08 的 11 条保留录制身份保持不变。

| 阶段 | 录制 | 帧数 | 动作窗口 | ×24 mask 后样本 | batch 256 的每轮批数 |
|---|---:|---:|---:|---:|---:|
| s_01–s_06 开发训练 | 30 | 80,070 | 655 | 15,720 | 62 |
| s_07/s_08 开发验证 | 11 | 25,423 | 209 | 5,016 | 20 |
| 前八人最终固定预算训练 | 41 | 105,493 | 864 | 20,736 | 81 |

现有测试结果曾参与此前研究讨论，因此后续也需保留这段历史，不能声称测试集从未被观察。

## 微调前后的评估约定

训练用 125 帧分块，报告用 [已核查的滚动末帧推理](ROLLING_INFERENCE.md)：窗口 125、步长 1、短前缀起始、状态不跨窗口或录制复用。前后都保留相同的居中加速度平滑，并披露其最多两帧的未来依赖；过去五帧平滑属于另列的敏感性分析。

四指标沿用 [POSE_METRICS.md](POSE_METRICS.md)：全局 MPJRE、pelvis 平移对齐的 MPJPE 和网格 MPJVE，以及按每条完整录制计算的 jitter。每个 mask 内按有效帧数加权空间误差、按 T−3 加权 jitter，然后对 24 个 mask 等权平均；逐录制、逐 mask 结果保留。前后都用相同 shape、关节/顶点集合、mesh 开关和零平移。IMU-only IK 尚属后续独立组件，本次仍比较原始网络。

开发 validation loss 用于选模型；论文复现结论需要四指标、固定协议与差异说明，不能由 training loss 降低单独推断。

## 短跑检查与结果

短跑设置为 **3 个 Adam 步、batch 8、24 个样本**，每个 mask 恰好一次；从不同训练窗口取样，并确保包含一个短尾窗口。先检查训练与验证 loader 的身份、形状和长度，只让训练数据参与梯度。

保存第 3 步后，使用独立加载的模型与优化器检查：模型参数和 Adam 状态逐位相等、同一输入的预测一致、从相同 RNG 与规定输入继续一步得到相同的模型/Adam/RNG 状态。额外一步分别在不中断与重载两个分支执行，用于续训检查；不会改写保存的第 3 步 checkpoint。

短跑 checkpoint 的 `kind=engineering_smoke_only`、`selection_eligible=false`、`epoch=-1`；它不计为完成一个正式 epoch，也不进入 best 模型选择。

实际使用现有 `imuposer` 环境的 **CPU、2 线程**。当前会话检测结果是 CUDA 不可见、GPU 数量为 0；这只描述本会话能力，不判定 PC 是否安装显卡。短跑本身约 **6.27 秒**，峰值 RSS **1,741 MiB**；此时间不含此前 raw 身份匹配，也不能外推正式 batch 256 或 GPU 的训练耗时。

| 检查 | 观测结果 |
|---|---|
| 训练与验证 loader | 分别仅包含 s_01–s_06 和 s_07/s_08；身份、形状、长度和内容哈希通过检查；验证数据不参与梯度；未打开 test 数据 |
| 24 个 mask | 24 个短跑样本恰好覆盖全部 24 种稀疏组合 |
| 第 1 步 | loss 0.02971774；951 有效帧、49 padding 帧；20/20 参数张量梯度非零、有限并发生更新 |
| 第 2 步 | loss 0.03434055；839 有效帧、161 padding 帧；20/20 参数张量梯度非零、有限并发生更新 |
| 第 3 步 | loss 0.04248882；1,000 有效帧、0 padding；20/20 参数张量梯度非零、有限并发生更新 |
| 相对 Epoch 59 的参数变化 | 20/20 张量改变；最大绝对变化 0.000901447 |
| 模型及 Adam 重载 | 模型权重、20 组 Adam 动量与 step 计数全部逐位相等；每组 Adam step=3 |
| 重载后的预测 | 同一 probe 的最大绝对差 0 |
| 随机性与续训 | 恢复相同 Python/NumPy/Torch RNG 后，同一指定输入继续一步的 loss、模型、Adam、最终 RNG 全部逐位一致 |
| 独立进程复核 | 不调用新 dataset/collate/loss 工具，直接从容器重建 8 个 probe 输入；目标与长度逐位相等；初始模型和短跑 checkpoint 的预测与缓存分别完全一致 |
| 单元检查 | 6 项通过，覆盖 24-mask 数量、受试者隔离、与公开 loader 的逐 mask/短尾一致性、共享数据不被污染、padding loss 的解析值、三类 RNG 恢复与状态差异检测 |

三个训练 loss 来自不同动作和 mask，包含 dropout；不能从这三行的大小顺序判断是否收敛。短跑证明工程链路可用，尚不提供开发集/测试集精度改善或论文复现成功的证据。

保存文件：[smoke_step3.ckpt](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/smoke_step3.ckpt)，SHA-256 `10f9656573fb41c522bd5ca80c65d2830a24fe3e391e98de2590fd5802afda7c`。完整检查见 [results.json](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/results.json)、[逐步梯度与更新](../outputs/local_dip/dip_finetuning_v1_20261114/smoke/steps.json)、[独立进程验证](../outputs/local_dip/dip_finetuning_v1_20261114/independent_validation.json)。

原冻结源码与输入清单为 `FREEZE_MANIFEST.json`；短跑文件为 `smoke/ARTIFACT_MANIFEST.json`。最终完整性清单、日志与独立验证器快照保存在实验根目录。模型/优化器 checkpoint、预测和原始资产属于忽略的大文件；`docs/protocols` 的元数据副本不替代实验目录备份。

## 复跑命令

输出目录必须不存在，正式复跑应换一个新目录。先冻结协议，再执行短跑。

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tools/prepare_dip_finetuning.py \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --reservation outputs/local_dip/wrist_pocket_dev_epoch59_20261110/split_manifest.json \
  --raw-dip /home/alex/research/IMUPoser/data/raw/DIP_IMU \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --output outputs/local_dip/dip_finetuning_v1_rerun

conda run --no-capture-output -n imuposer python tools/smoke_dip_finetuning.py \
  --protocol-dir outputs/local_dip/dip_finetuning_v1_rerun \
  --device auto --threads 2

conda run --no-capture-output -n imuposer python -m unittest discover \
  -s tests -p test_dip_finetuning.py -v

conda run --no-capture-output -n imuposer python tools/validate_dip_finetuning_smoke.py \
  --protocol-dir outputs/local_dip/dip_finetuning_v1_20261114 \
  --output /tmp/dip_finetuning_smoke_recheck.json
```

`--device auto` 在可用时使用 CUDA，否则使用 CPU；显式 `--device cuda:0` 在 CUDA 不可见时会报错。短跑会核对冻结源码和输入哈希，拒绝在源码或数据已改变时沿用旧协议。正式训练预算已冻结，本次入口执行的仅是工程短跑。
