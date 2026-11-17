# IMUPoser GPU 可见性诊断

日期：2026-11-14。状态：普通 WSL 的 GPU 计算和 IMUPoser CUDA 前向验证通过；Codex 当前沙盒仍隔离 GPU，权限保持原样。目的：确认本机 NVIDIA GPU 能用于后续 DIP 微调，区分普通 WSL 与 Codex 沙盒的设备访问。冻结的微调协议、Epoch 59、短跑 checkpoint 和训练源码均不修改；不重复 CPU 短跑，不启动正式训练。

## 已确认的诊断

| 层次 | 结果与证据 |
|---|---|
| 运行环境 | Ubuntu 26.04.1，WSL2 内核 `6.18.33.2-microsoft-standard-WSL2`；主机名 lyrixlegion。Codex 命令额外运行于受限制的挂载与 seccomp 环境；不是原生 Windows Python |
| 普通 WSL Terminal | Alex 运行同一诊断脚本后生成现场报告：`nvidia-smi` 成功、`/dev/dxg` 存在并可读写、Torch CUDA 可用；下方有完整计算验证 |
| Codex 沙盒 | `/dev` 的最终挂载是精简 tmpfs，只有 null/zero/random/tty 等基础设备，没有 `/dev/dxg`；NoNewPrivs=1、Seccomp=2 |
| 沙盒内 nvidia-smi | 使用 `/usr/lib/wsl/lib/nvidia-smi`，报 `Failed to initialize NVML: GPU access blocked by the operating system` |
| Python/Conda | 默认交互环境是 base；显式 `conda run -n imuposer` 后执行文件为 `/home/alex/miniconda3/envs/imuposer/bin/python`，Python 3.10.21，环境选择正确 |
| PyTorch | 2.7.0+cu128，`torch.version.cuda=12.8`，`torch.backends.cuda.is_built()=True`；这是带 CUDA 的构建 |
| CUDA 相关包 | CUDA runtime 12.8.57，cuDNN 9.7.1.26，cuBLAS 12.8.3.14；WSL `libcuda.so.1` 和 NVML 库存在 |
| 可见性环境变量 | CUDA_VISIBLE_DEVICES、NVIDIA_VISIBLE_DEVICES、NVIDIA_DRIVER_CAPABILITIES、LD_LIBRARY_PATH 均未设置，没有发现用环境变量隐藏 GPU 的证据 |
| 沙盒内 Torch 查询 | `cuda.is_available=False`、`device_count=0`；driver `cuInit(0)` 返回 100，Torch 初始化失败 |
| 当前会话审批限制 | 工作区隔离保留；`approval_policy.granular.sandbox_approval=false`。尝试复用历史已批准的只读 CUDA 查询时，平台在执行前自动拒绝 `require_escalated`，理由为不允许提出升级审批。没有通过该通道执行命令，也没有修改权限或绕过隔离 |

**根因已定位到 Codex 的设备隔离层。** 普通 WSL 的 GPU 接口存在，而当前 Codex 的挂载命名空间将 `/dev` 替换后隐藏了该接口。CUDA 构建和用户态库都已安装，不能把沙盒里的“no NVIDIA driver”异常直接解释为宿主机没装显卡驱动。

普通 WSL 与沙盒报告的主机名、内核、Python 路径、Torch 版本、诊断脚本哈希及受保护资产哈希一致；主要差异是 `/dev` 挂载与设备节点、seccomp 限制。普通终端为原 devtmpfs，NoNewPrivs=0、Seccomp=0；沙盒为精简 tmpfs，NoNewPrivs=1、Seccomp=2。这是对同一环境的对照，不是把历史训练元数据当作当前 GPU 检测结果。

## 当前采用的处理方式

Alex 选择在普通 WSL Terminal 执行预检，保持现有 Codex 权限不变。诊断脚本为 [diagnose_cuda.py](../tools/diagnose_cuda.py)：

```bash
conda run --no-capture-output -n imuposer python \
  /home/alex/research/PoseEstimationWithWearables/tools/diagnose_cuda.py \
  --check-model --label ordinary_wsl_terminal \
  --output /home/alex/research/PoseEstimationWithWearables/outputs/local_dip/gpu_diagnostic_20261114/terminal.json
```

该脚本只写一个新 JSON 报告，拒绝覆盖已有报告。检查内容为：

1. 系统、环境、设备节点、nvidia-smi 与 PyTorch CUDA 构建信息。
2. 在 `cuda:0` 创建 256×256 张量，执行矩阵乘法和 autograd，核对结果、设备及有限性。
3. 从原始 Epoch 59 加载网络，在 GPU 上执行 8 帧合成 IMU 输入的前向与 SMPL FK，核对参数、输入、输出及关节张量都位于 `cuda:0`。
4. 前后核对原协议、协议元数据副本、原始 Epoch 59 和已有短跑 checkpoint 的哈希。没有 DIP 数据加载、模型参数更新、优化器步或 checkpoint 保存。

沙盒对照报告：[codex_sandbox.json](../outputs/local_dip/gpu_diagnostic_20261114/codex_sandbox.json)。该报告状态为 `blocked_cuda_unavailable`，退出码 2 明确表示设备预检未通过；受保护文件哈希保持不变。

## GPU 计算验证结果

普通终端报告：[terminal.json](../outputs/local_dip/gpu_diagnostic_20261114/terminal.json)，状态 `passed_cuda_and_model`。已从共享项目直接读取并检查，与用户粘贴的输出一致。

| 项目 | 现场结果 |
|---|---|
| NVIDIA GPU | GeForce RTX 5060 Ti，报告显存 8,151 MiB，计算能力 12.0 |
| 驱动 | `--query-gpu=driver_version` 为 610.60；NVIDIA-SMI 工具版本 610.47.03；表头 KMD 610.60、CUDA UMD 13.3 |
| Python | `/home/alex/miniconda3/envs/imuposer/bin/python`，Python 3.10.21，Conda 环境 imuposer |
| Torch CUDA | 2.7.0+cu128，编译 CUDA 12.8，is_available=True，device_count=1；架构列表包含 sm_120 |
| 实际矩阵计算 | 256×256 乘法及 autograd 通过；输入、输出、梯度设备均为 cuda:0 |
| IMUPoser | Epoch 59 网络的参数与前向输出均为 cuda:0；8 帧输出形状 `[1,8,144]` |
| SMPL FK | 由网络输出计算的关节位于 cuda:0，值有限 |
| 训练动作 | 0 个优化器步，没有加载 DIP 数据、改参数或写 checkpoint |
| 完整性 | 两份报告受保护资产哈希一致；冻结运行的 47 个文件与 29 个原源码文件复核无变化 |

驱动报告的 CUDA UMD 13.3 与 PyTorch 编译 CUDA 12.8 是不同层次的信息，不能把版本号不同本身判为故障；本次已通过实际计算验证兼容性。当前没有证据要求重装 PyTorch、CUDA runtime 或 Windows 驱动。

**显存条件仍需在正式训练前确认。** 报告开始时、执行本次 CUDA 检查之前，nvidia-smi 已显示 6,472/8,151 MiB 显存占用和 97% GPU 利用率。该占用不能归因于本次未启动的训练；报告未能确定是哪个 Windows/WSL 工作负载产生，不能把 WSL 进程表中列出的 Xwayland 当作唯一占用来源。本次没有终止任何程序。设备、架构与模型计算条件已具备，但此次 8 帧前向不证明冻结 batch256 在当前可用显存下能够训练。

## 修复路径和后续执行

当前方案使用普通 WSL Terminal 运行 GPU 任务，Codex 负责准备、检查代码并读取共享输出；这是 Alex 选择保留当前权限后的执行方式。后续启动训练时显式指定 `--device cuda:0`，设备不可用就报错，避免 `auto` 回退到 CPU。现有短跑入口已有该显式选项，并将输入、模型与 SMPL helper 移到同一 CUDA 设备；本次不调用短跑入口。

当前结论：**普通 WSL 已具备 GPU 执行前提；Codex 沙盒直接访问仍不可用。** 后续正式微调还需要完整的 epoch/validation 执行入口及可用显存核查；沿用已冻结的协议，不自动减小 batch、不重训 AMASS，也不把这次设备预检算作正式训练。

如果后续需要 Codex 直接启动 GPU 命令，应先由用户在客户端允许单条命令提出沙盒外审批，保留工作区隔离；再对具体 GPU 命令逐条授权。当前会话不具备这种审批能力，助手不能从项目代码中改变平台策略。尚无证据支持用某个项目路径白名单设置恢复被屏蔽的设备挂载，因此不建议猜测设备透传配置。

本次没有重装驱动、安装 CUDA Toolkit、修改 shell/Conda 环境、设置全局可见性变量、调整设备权限或关闭安全限制。WSL CUDA 使用 Windows 提供的驱动映射；若今后普通终端也失败，应另行检查 Windows 驱动与 WSL 状态，并在系统修改前单独获得同意。[NVIDIA CUDA on WSL 12.8 指南](https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html)

审批粒度与布尔开关的含义依据 [OpenAI 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml)。官方文档说明 sandbox_approval 控制沙盒升级审批能否出现；它不保证当前客户端可以在原会话中切换设备挂载，也不表示打开审批就自动授予所有命令权限。
