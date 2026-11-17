# 在 WSL 启动 DIP 微调

日期：2026-11-14。入口为 [train_dip_finetuning.py](../tools/train_dip_finetuning.py)。本次只准备并验证启动代码，实际训练由 Alex 在普通 WSL Terminal 启动。冻结协议和 Epoch 59 不修改，已有 CPU 短跑不重复执行。

本入口沿用 [DIP_FINETUNING_V1](protocols/dip_finetuning_v1_20261114/PROTOCOL.json)：Adam lr 3e-4、seed42、24 个 mask、batch256、开发训练 30 epochs、按独立开发集的最低 combined validation loss 选择轮数。同分保留较早完成的轮数。

## 第一步启动开发训练

在普通 WSL Terminal 中先用 `nvidia-smi` 确认 GPU 的空闲显存。上次设备预检时已有 6,472/8,151 MiB 显存占用；代码不会自动结束其他程序或改变训练 batch。

```bash
cd /home/alex/research/PoseEstimationWithWearables

conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage development \
  --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1
```

默认读取 `outputs/local_dip/dip_finetuning_v1_20261114/` 的冻结协议、身份清单和源码哈希。s_01–s_06 训练，s_07/s_08 验证；每轮 62 个训练批、20 个验证批。不会加载 test 数据，也不会自动进入最终前八人训练。

CUDA 不可用会退出，禁止 CPU fallback。运行中出现 CUDA OOM 会停止并记录错误，batch、学习率、精度和预算保持原样；先确认显存可用性，再恢复。入口不接受临时覆盖 epochs/batch/lr 的命令行参数。

输出目录必须不存在；重复使用同一目录须显式加 `--resume`。新目录与原冻结目录和已有 checkpoint 分开。

## 断点恢复与结果文件

```bash
conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage development \
  --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1 \
  --resume
```

恢复时核对协议、执行源码、Torch 版本、设备、数据及 checkpoint 哈希。恢复模型、Adam、Python/NumPy/Torch RNG、独立的训练/验证 DataLoader generator；从最后一个**完整提交的 epoch** 继续。未完成的一轮会重跑；若第一轮尚未完成，从原 Epoch 59 重新开始这一轮。代码、精度或训练设置改变时不会静默沿用旧状态。

| 文件 | 用途 |
|---|---|
| `train.log` | 实时批进度、每轮 train/validation loss、设备及显存信息；同时输出到终端 |
| `history.csv`、`history.json` | 每轮 pose/joint/combined loss、有效/padding 帧数、耗时、最佳轮数 |
| `best.ckpt` | 开发集最低验证损失的 checkpoint，符号链接 |
| `last.ckpt` | 最后完整提交轮的 checkpoint，符号链接 |
| `checkpoints/epoch_*.ckpt` | 每轮独立的完整模型/Adam/RNG/采样状态，文件不覆盖或删除 |
| `STATE.json` | 原子提交的 last/best 位置和文件哈希；恢复时以此为准 |
| `progress.json` | 已完成 epoch、step 和最佳分数，供查看进度 |
| `completion.json` | 完成全部预算后给出 best_epoch 和最终阶段 K |
| `failure.json` | 中断或训练异常；包含最后完整提交轮数和恢复提示 |
| `RUN.json`、`source_snapshot/` | 本次执行命令、配置、源码哈希与快照；协议副本按原字节保留 |

每轮先写独立 checkpoint，再原子更新 `STATE.json`，最后更新链接与历史表。意外中断后，未提交文件不会被当作恢复点；链接和历史表可由已提交 checkpoint 重建。已有训练轮的 checkpoint 字节保持不变。保存 30 轮约需 4 GB 磁盘空间，best/last 符号链接不会复制权重数据。

可在另一个普通 WSL Terminal 查看：

```bash
tail -f /home/alex/research/PoseEstimationWithWearables/outputs/local_dip/dip_dev_v1/train.log
```

## 第二阶段按选定预算训练前八人

开发阶段完成 30 轮后，再启动：

```bash
conda run --no-capture-output -n imuposer python -u tools/train_dip_finetuning.py \
  --stage final \
  --device cuda:0 \
  --development-run outputs/local_dip/dip_dev_v1 \
  --output outputs/local_dip/dip_final_v1
```

入口核对开发阶段确已完成，并从完整历史与 best checkpoint 复核选定 K。然后重新加载**原 AMASS Epoch 59**，重置 Adam 和 seed42，用 s_01–s_08 训练恰好 K 轮。不会从开发微调的 best 权重继续训练，不会使用 s_09/s_10 选模型。最终 pool 为 20,736 个 mask 样本，每轮 81 批，完成时生成 `final.ckpt` 链接。

最终阶段的恢复命令是在这条命令末尾加 `--resume`。s_07/s_08 在该阶段进入训练；这项角色变化已在冻结协议中单独标明。

训练后的滚动四指标、所有 mask 的测试汇总和 IK refinement 是后续工作，不由本训练命令自动执行。当前训练验证损失不能替代论文 MPJRE、MPJPE、网格 MPJVE 和 jitter。

## 已完成的代码验证

新增训练引擎的 **10 项测试全部通过**，覆盖不等长批的 padded B×T 加权、验证关闭梯度/dropout、精确同分选择、非有限 loss 拒绝、CUDA 不可见时禁止训练、禁止将旧 smoke 目录用作完整训练恢复目录、不可变 checkpoint 与损坏检查、提交状态与孤立文件处理，以及带 shuffle/dropout/Adam 的跨 epoch 确定性恢复。全部使用微型合成 CPU 网络，不是重复真实 IMUPoser 短跑。

还执行了以下只读入口检查，核对冻结资产和真实 DIP 身份/样本数；不创建训练目录，不进行参数更新：

```bash
conda run --no-capture-output -n imuposer python tools/train_dip_finetuning.py \
  --stage development --device cuda:0 \
  --output outputs/local_dip/dip_dev_v1 --check-only
```

GPU tensor/模型前向已由 [G01](GPU_DIAGNOSTIC.md) 在普通 WSL 验证；当前 Codex 沙盒不能访问 GPU，因此本次没有声称完成该完整入口的 GPU 多轮训练，也未改变冻结协议以绕过显存或权限限制。
