# IMUPoser 滚动窗口推理与依赖核查

日期：2026-11-13。状态：实现、全开发集比较和独立核查完成。在同一个 Epoch 59 checkpoint 上，已区分原有非重叠分块和逐帧滚动推理，记录每个输出实际使用的输入，并用相同四指标量化差异。模型权重、原始训练代码和既有实验保持原样。

实现：[window_inference.py](../tools/window_inference.py)。比较入口：[compare_window_inference.py](../tools/compare_window_inference.py)。测试：[test_window_inference.py](../tests/test_window_inference.py)。实验目录：[window_comparison_epoch59_20261113](../outputs/local_dip/window_comparison_epoch59_20261113/)。

## 论文规定与本次实现约定

[论文第 5.3 节](https://arxiv.org/html/2304.12518v1) 描述在线使用 125 个样本的历史窗口，同时写了 “1-sample overlap”；[第 4.2 节](https://arxiv.org/html/2304.12518v1) 描述五帧均值平滑。正文没有完整说明输出位置、起始边界及平滑方向。

**本次将步长 1、取末帧作为明确的在线实现约定，不能称为已证实的作者全部细节。** 长度 125、步长 1 的相邻满窗口实际重叠 124 帧。若把论文文字按“只重叠一帧”理解，步长应为 124；如果这种窗口仅取末帧，则无法每帧产生一个新姿态。当前可检查源码没有解决这段文字歧义的作者在线入口，因此不把 overlap 和 stride 混为一谈。

所有帧号从 0 开始，区间 `[start, stop)` 的 stop 不包含在内。

| 设置 | 输入窗口 | 保留输出 | 起始和末尾 |
|---|---|---|---|
| 原有 `chunk` | `[125k, min(125k+125,T))` | 每块全部帧 | 尾块按实际长度运行；块内早期输出可使用块内后续输入 |
| `rolling_last` | 对帧 t：`[max(0,t-124), t+1)` | 仅窗口最后一帧对应的输出 | t=0 时只用一帧，逐渐增长到 125；不丢弃暖启动帧，不补未来帧 |
| 可选 `repeat_first` 暖启动 | 同一真实历史，左侧重复第 0 帧至 125 长度 | 最后位置 | 只提供为显式选项；本次定量比较使用短前缀，不使用此选项 |

每个窗口都重新运行两个方向的 LSTM，隐藏状态不跨窗口复用；每条录制独立开始。BiLSTM 在窗口内部依然双向，但末帧输出对应的窗口没有 t 之后的模型输入。不能用一次长序列 BiLSTM 或保留上一窗口状态代替此语义。

`RollingPoseStream.push(frame)` 提供真正一次接收一帧的入口；每次输出一个姿态。调用方在录制边界调用 `reset()`。缓存最多 125 个 60 维输入，输入已按原规则缩放和 mask。批量 `infer_sequence` 对独立窗口分批运行以加速离线核查，批中后面的窗口不参与前面窗口的计算；这类吞吐时间不能当成流式设备的端到端延迟。

## 平滑与未来依赖

继承的预处理实际只平滑加速度；方向矩阵仍是 60→25 Hz 的逐元素线性插值，没有追加方向或输出姿态平滑。

| 设置 | 25 Hz 加速度第 t 帧的来源 | 起始和末尾 | 对在线时刻 t 的影响 |
|---|---|---|---|
| 原有居中平均 | `[t-2, t+2]`，裁到录制范围内 | 不足五帧时按实际帧数平均 | 最多用未来两帧；至少需要约 80 ms 等待才能发布对应时刻的结果 |
| 过去五帧平均 | `[t-4, t]`，裁到录制范围内 | 起始使用已有 1–5 帧 | 在这一 25 Hz 阶段没有未来依赖；可能带来信号滞后，不能称为与原平滑等价 |

滚动模型窗口本身消除未来上下文，并不自动让已有预处理数据具有原始流因果性。原流程还包含前后向 NaN 填补，以及 60→25 Hz 插值时可能读取右侧原始样本。`rolling_trailing` 在本报告中特指**完成这些预处理后的 25 Hz 阶段因果**；原始传感器流的严格因果链仍需另行实现和验证。

为避免对已经平滑的数据再次平滑，本次过去五帧版本从已授权本地 raw DIP 录制重建：保持原 NaN 填补、裁剪、重采样规则，只改变加速度的均值窗口方向。每条录制都先验证居中版本与冻结 processed 文件的加速度逐位相等；方向、目标姿态、mask、缩放和 checkpoint 相同。

## 如何追踪一帧

每条录制、每种设置保存一份 `seq{ID}_{variant}_trace.csv`，逐帧记录：

- `output_frame`：该姿态对应的原录制帧号。
- `input_start/input_stop`：直接送入网络的真实 25 Hz 特征区间。
- `model_output_index/model_length`：所取的网络输出位置及有效长度；批处理补零不属于有效历史。
- `left_repeats_of_frame_zero`：暖启动重复次数，主实验全部为 0。
- `acc_source_start/acc_source_stop`：考虑五帧滤波后，网络加速度依赖的平滑前 25 Hz 区间；方向使用 `ori_source_start/ori_source_stop`。
- `available_at_25hz_frame/future_frames_at_25hz`：在该预处理阶段最早具备全部依赖的帧号，以及相对输出时刻的未来帧数。这不是计算耗时，也不包含上游原始数据填补、插值或传输延迟。

例如输出 t=125：滚动输入是 `[1,126)`，取输出索引 124。居中平滑依赖的加速度范围是 `[0,128)`；过去五帧平均为 `[0,126)`。原始分块模式则使用 `[125,250)`，取索引 0，因此可使用到第 249 帧，滤波后还可能使用到第 251 帧。所有范围都在录制边界裁剪。CSV 加上上述滤波规则足以重建每个输出的网络输入，不需要猜测缓存或 padding。

## 固定比较协议

运行前保存的 [PROTOCOL.json](../outputs/local_dip/window_comparison_epoch59_20261113/PROTOCOL.json) 锁定：Epoch 59；已有精确匹配的 s_07/s_08 全部 11 条开发录制、25,423 帧；固定 `lw_rw_h` 三位置；125 帧窗口；滚动步长 1；短前缀起始；无 DIP 微调、无 IK refinement。三个设置为：

1. `chunk_centered`：复算原来的分块基线。
2. `rolling_centered`：仅改变模型输入窗口，隔离推理语义的影响。
3. `rolling_trailing`：在滚动方式上再改变加速度平滑，隔离滤波方向的影响。

不打开 s_09/s_10 测试集，不选择 checkpoint 或搜索 mask。四指标使用 [POSE_METRICS.md](POSE_METRICS.md) 的相同工具：24 关节、6,890 顶点、相同 mean male SMPL、pose blend shapes 关闭、零平移、空间误差仅对齐 pelvis 平移。各录制单独计算 jerk 后汇总，分块边界产生的跳变保留在同一条录制的 jitter 中。

主汇总对空间误差按帧加权，jitter 按各录制 T−3 的有效差分数加权；同时保留序列等权结果。源码、checkpoint、SMPL、数据、raw 录制和划分清单均有哈希；实际导入的是原始 `/home/alex/research/IMUPoser/src/imuposer` 包，实验目录保留该包及新工具的运行源码快照。

## 验证和比较结果

全部 11 条录制、25,423 帧完成三种设置的全新网络推理；每种设置有 25,390 个 jerk 时间样本。主汇总如下，误差与 jitter 均为越低越好：

| 推理与平滑 | MPJRE ° | MPJPE cm | MPJVE cm | Jitter，100 m/s³ |
|---|---:|---:|---:|---:|
| 分块＋居中平均 | 21.292074 | 9.451584 | 10.257982 | 0.782900 |
| 滚动末帧＋居中平均 | 22.068137 | 9.811889 | 10.749727 | 0.859217 |
| 滚动末帧＋过去五帧平均 | 22.526798 | 10.149600 | 11.133277 | 0.892934 |

保持居中平滑时，滚动相比分块的 MPJRE 增加 0.776063°、MPJPE 增加 0.360305 cm、MPJVE 增加 **0.491745 cm（4.79%）**、jitter 增加 **9.75%**。两组预测之间的平均局部旋转差为 7.082630°，所以不能把分块预测改个标签当作在线结果。11 条录制中只有 2 条的滚动 MPJVE 更低；先完成的第一条录制呈改善，不能代表总结果。

在滚动设置下，过去五帧平均相对居中平均进一步增加 MPJVE **0.383550 cm（3.57%）**、jitter 增加 3.92%。这说明本模型和数据上的滤波方向改变有可测影响，不能把修复未来依赖当成零成本改动。窗口比较同时改变了上下文分布、重置位置和输出位置，不能将全部变化单独归因于未来信息；滤波比较也不证明过去平均在其他训练配方中必然更差。

辅助序列等权 MPJVE 分别为 9.765306、10.433512、11.063694 cm；完整四指标、局部误差、参考 jitter、逐序列结果及两种加权方式见 [metrics.json](../outputs/local_dip/window_comparison_epoch59_20261113/metrics.json)。当前仅为一个三位置 mask 的开发集方法核查，未做 DIP 微调、IK 或全部 24 配置评估，不能直接对照论文 test 数字。

独立验证入口：[validate_window_comparison.py](../tools/validate_window_comparison.py)。该验证器不调用窗口计划或 trace 生成函数，从帧号独立推导每一行边界，并对照原 E04 缓存：

| 核查 | 结果 |
|---|---|
| 三种设置全部输出的依赖区间 | 76,269 行全部通过；无漏帧、重复帧或跨录制窗口 |
| 真实网络的 trace 重建 | 231 个覆盖暖启动、125 帧边界和末尾的独立单窗口检查通过；最大 6D 差 1.222×10⁻⁶ |
| 新分块与冻结 E04 缓存 | 全部目标逐位相等；25,423 帧预测最大 6D 差 3.368×10⁻⁶，源于批量数值差异 |
| 居中预处理重建 | 全部 11 条加速度与冻结 processed 数据逐位相等；没有二次平滑 |
| 分块的未来依赖 | 25,214 个输出使用后续模型输入，最多 124 帧；加上居中滤波最多 126 帧 |
| 滚动＋居中平均 | 模型未来输入为零；25,412 个输出仍依赖平滑前未来数据，最多两帧 |
| 滚动＋过去五帧平均 | 在平滑前 25 Hz 阶段的未来依赖为零；上游 NaN 填补和插值仍另计 |
| 单元和模型集成测试 | 新增 9 项通过，含流式与批量一致性、暖启动、边界、尾块、滤波、未来扰动检查；既有 sensor metrics 7 项回归测试通过 |

**数值容差记录。** 初版随机输入的真实模型集成检查使用 2×10⁻⁵ 的 6D 容差，单帧与批量 CPU packed-LSTM 在一个输出出现 6.336×10⁻⁵ 差异，检查失败。最终对跨批量形状比较采用明确的 10⁻⁴ 容差；固定批划分后，未来扰动前的历史输出要求且达到逐位相等。真实 DIP 的 231 个重建检查仍保持 2×10⁻⁵ 门槛，实际最大差远低于门槛。原始运行源码快照保留，测试调整后的源码与日志另外存入验证快照，不覆盖运行证据。

运行在 `imuposer` 环境的 CPU 上，总计约 765 秒，峰值 RSS 1,111 MiB；这些是批量离线核查的资源记录，不是实机延迟测量。原始完整性清单 [ARTIFACT_MANIFEST.json](../outputs/local_dip/window_comparison_epoch59_20261113/ARTIFACT_MANIFEST.json) 的 77 个文件均通过复核。独立结论见 [independent_validation.json](../outputs/local_dip/window_comparison_epoch59_20261113/independent_validation.json)；后续验证文件及最终测试快照另有 `VALIDATION_MANIFEST.json`。

## 运行方式

输出目录必须尚不存在；复跑时使用新目录，避免覆盖记录。

```bash
cd /home/alex/research/PoseEstimationWithWearables
conda run --no-capture-output -n imuposer python tools/compare_window_inference.py \
  --checkpoint '/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  --data /home/alex/research/IMUPoser/data/processed_imuposer_25fps/dip_train.pt \
  --manifest outputs/local_dip/wrist_pocket_dev_epoch59_20261110/split_manifest.json \
  --smpl-model /home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
  --output outputs/local_dip/window_comparison_epoch59_rerun \
  --device cpu --threads 2 --batch-size 16
```

`--sequences` 仅接受冻结开发清单中的序列号，省略时覆盖全部 11 条录制。完整复跑只需要已有本地授权资产，无须修改训练环境依赖。

```bash
conda run --no-capture-output -n imuposer python tools/validate_window_comparison.py \
  --run outputs/local_dip/window_comparison_epoch59_20261113 \
  --legacy-cache outputs/local_dip/dual_wrist_dev_epoch59_20261110/evaluation/predictions.npz \
  --output /tmp/window_comparison_validation_recheck.json

IMUPOSER_TEST_CHECKPOINT='/home/alex/research/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10062026-021332/epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt' \
  conda run --no-capture-output -n imuposer python -m unittest discover \
  -s tests -p test_window_inference.py -v
```
