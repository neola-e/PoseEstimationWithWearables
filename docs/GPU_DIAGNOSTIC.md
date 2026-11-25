# IMUPoser GPU visibility diagnosis

English presentation updated 2026-11-19; original experiment dates and bytes are preserved in [the language/provenance guide](RECORD_LANGUAGE.md).

Date: 2026-11-14. Status: ordinary-WSL GPU computation and IMUPoser CUDA forward pass; the current Codex sandbox still isolates the GPU, permissions unchanged. Objective: verify local NVIDIA GPU suitability for DIP fine-tuning and distinguish ordinary WSL access from Codex device access. Frozen fine-tuning protocol, Epoch 59, smoke checkpoint and training sources remain unchanged; no repeated CPU smoke or formal training.

## Confirmed diagnosis

| Layer | Result and evidence |
|---|---|
| Runtime | Ubuntu26.04.1, WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, host lyrixlegion. Codex additionally uses restricted mounts/seccomp; this is not native Windows Python |
| Ordinary WSL Terminal | Alex ran the same diagnostic: successful `nvidia-smi`, readable/writable `/dev/dxg`, available Torch CUDA; computation checks below |
| Codex sandbox | Final `/dev` mount is minimal tmpfs containing null/zero/random/tty etc., without `/dev/dxg`; NoNewPrivs=1, Seccomp=2 |
| Sandbox nvidia-smi | `/usr/lib/wsl/lib/nvidia-smi` returns `Failed to initialize NVML: GPU access blocked by the operating system` |
| Python/Conda | Default interactive env is base; explicit `conda run -n imuposer` uses `/home/alex/miniconda3/envs/imuposer/bin/python`, Python 3.10.21, the intended environment |
| PyTorch | 2.7.0+cu128, `torch.version.cuda=12.8`, `torch.backends.cuda.is_built()=True`: CUDA-enabled build |
| CUDA packages | Runtime12.8.57, cuDNN9.7.1.26, cuBLAS12.8.3.14; WSL `libcuda.so.1` and NVML libraries exist |
| Visibility variables | CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES and LD_LIBRARY_PATH unset; no evidence of environment-variable GPU hiding |
| Sandbox Torch | `cuda.is_available=False`, `device_count=0`; driver `cuInit(0)` returns100 and initialization fails |
| Session approval limits | Workspace isolation retained; `approval_policy.granular.sandbox_approval=false`. Attempted reuse of a previously approved read-only CUDA query was automatically rejected before execution because escalation approval requests are disabled. No command ran through that route; no permission change or isolation bypass |

**Root cause is the Codex device-isolation boundary.** Ordinary WSL exposes the GPU interface, while the current Codex mount namespace replaces `/dev` and hides it. CUDA builds/user-space libraries are installed; the sandbox's “no NVIDIA driver” message does not establish a missing host driver.

Both reports match hostname, kernel, Python path, Torch version, diagnostic-source hash and protected-asset hashes. Principal differences are `/dev` mount/device nodes and seccomp: ordinary terminal devtmpfs, NoNewPrivs=0, Seccomp=0; sandbox minimal tmpfs, NoNewPrivs=1, Seccomp=2. This is a same-environment comparison, not historical training metadata presented as current GPU detection.

## Selected execution route

Alex chose ordinary WSL preflight with Codex permissions unchanged. Script: [diagnose_cuda.py](../tools/diagnose_cuda.py).

```bash
conda run --no-capture-output -n imuposer python \
  /home/alex/research/PoseEstimationWithWearables/tools/diagnose_cuda.py \
  --check-model --label ordinary_wsl_terminal \
  --output /home/alex/research/PoseEstimationWithWearables/outputs/local_dip/gpu_diagnostic_20261114/terminal.json
```

The script writes one new JSON report and refuses overwrite. It checks:

1. System/environment/device nodes, nvidia-smi and Torch CUDA build.
2. A256×256 CUDA tensor multiplication and autograd on `cuda:0`, verifying devices/finiteness.
3. Original Epoch 59 network plus eight synthetic IMU frames on GPU and SMPL FK; parameters/input/output/joints all on `cuda:0`.
4. Before/after hashes of original protocol, metadata copy, Epoch 59 and existing smoke checkpoint. No DIP loading, parameter updates, optimizer steps or checkpoint saving.

Sandbox control: [codex_sandbox.json](../outputs/local_dip/gpu_diagnostic_20261114/codex_sandbox.json), status `blocked_cuda_unavailable`, exit2 explicitly indicating failed device preflight; protected hashes unchanged.

## GPU computation verification

Ordinary-terminal report: [terminal.json](../outputs/local_dip/gpu_diagnostic_20261114/terminal.json), status `passed_cuda_and_model`; directly read from shared storage and consistent with pasted user output.

| Item | Observed result |
|---|---|
| NVIDIA GPU | GeForce RTX5060Ti; reported8,151 MiB; compute capability12.0 |
| Driver | `--query-gpu=driver_version`: 610.60; NVIDIA-SMI utility610.47.03; header KMD610.60, CUDA UMD13.3 |
| Python | `/home/alex/miniconda3/envs/imuposer/bin/python`, Python 3.10.21, Conda imuposer |
| Torch CUDA | 2.7.0+cu128, build CUDA12.8, is_available=True, device_count=1; architectures include sm_120 |
| Computation | 256×256 matmul/autograd pass; inputs/outputs/gradients all cuda: 0 |
| IMUPoser | Epoch 59 parameters/forward output cuda: 0; eight-frame shape `[1,8,144]` |
| SMPL FK | Joints computed from network output on cuda: 0 and finite |
| Training actions | Zero optimizer steps; no DIP loading, parameter change or checkpoint write |
| Integrity | Protected hashes match across reports; 47 frozen-run files and29 original source files rechecked unchanged |

Driver CUDA UMD13.3 and Torch build CUDA12.8 describe different layers; differing numbers alone do not diagnose failure. Actual computation passed. No evidence requires reinstalling Torch, CUDA runtime or Windows drivers.

**Check free memory before formal training.** Before this CUDA check, nvidia-smi already showed6,472/8,151 MiB used and97% GPU utilization. This was not the not-yet-started training. The report does not identify the responsible Windows/WSL workload; Xwayland in the WSL process list is not proof of sole ownership. No process was terminated. Device/architecture/model computation prerequisites pass; eight-frame forward does not prove frozen batch 256 training fits currently free memory.

## Resolution and subsequent execution

Use ordinary WSL for GPU work; Codex prepares/reviews code and reads shared outputs, honoring Alex's permission choice. Specify `--device cuda:0`; fail if unavailable instead of `auto` CPU fallback. The existing smoke entry supports this and moves inputs/model/SMPL helper to the same CUDA device; it was not invoked here.

Conclusion at this date: **ordinary WSL meets GPU execution prerequisites; direct Codex-sandbox access remains unavailable.** Formal fine-tuning still needs the full epoch/validation launcher and available-memory check. Preserve the frozen recipe: no automatic smaller batch, AMASS retraining, or counting this preflight as training.

Direct Codex GPU execution would first require the user to enable command-specific external-sandbox approval in the client, retaining workspace isolation, then approve the actual command. This session cannot request that approval; project code cannot change platform policy. No evidence supports a project-path allowlist restoring hidden device mounts; do not guess device passthrough settings.

No driver/runtime installation, shell/Conda edit, global visibility variable, device-permission change or blanket safety-disable occurred. WSL CUDA uses Windows driver mapping. If ordinary WSL later fails, separately inspect Windows driver/WSL state and obtain approval before system changes. [NVIDIA CUDA on WSL12.8 guide](https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html)

Approval-category semantics follow the [OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml): sandbox_approval controls whether escalation prompts are allowed. It does not guarantee live device-mount changes or automatically authorize every command.
