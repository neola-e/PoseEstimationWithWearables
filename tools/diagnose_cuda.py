"""Read-only GPU preflight: environment, CUDA matmul, optional model/FK forward.

Does not run DIP training, update parameters, change configuration, install
packages, or write checkpoints. Only a new JSON report is written. Use the SAME
command in a normal WSL terminal and in Codex to compare device visibility.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        return {'argv': argv, 'returncode': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': argv, 'error': str(e)}


def node(path):
    try:
        s = Path(path).stat()
        return {'exists': True, 'character_device': stat.S_ISCHR(s.st_mode),
                'mode': stat.filemode(s.st_mode), 'readable': os.access(path, os.R_OK),
                'writable': os.access(path, os.W_OK)}
    except OSError as e:
        return {'exists': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-model', action='store_true', help='Bounded inference only; no optimizer or DIP loader')
    parser.add_argument('--label', default='unspecified', help='For example codex_sandbox or ordinary_wsl_terminal')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; choose a new report name')
    frozen = ROOT/'outputs/local_dip/dip_finetuning_v1_20261114'
    protocol_file = frozen/'PROTOCOL.json'
    protocol = json.loads(protocol_file.read_text())
    protected = [protocol_file, ROOT/'docs/protocols/dip_finetuning_v1_20261114/PROTOCOL.json',
                 Path(protocol['checkpoint']['path']), frozen/'smoke/smoke_step3.ckpt']
    hashes_before = {str(path): digest(path) for path in protected}
    if hashes_before[str(protected[2])] != protocol['checkpoint']['sha256']:
        raise ValueError('Original Epoch59 hash differs from frozen protocol')
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    mounts = read_text('/proc/self/mountinfo') or ''
    status = read_text('/proc/self/status') or ''
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'label': args.label,
              'argv': sys.argv, 'script_sha256': digest(__file__),
              'environment': {'uname': platform.uname()._asdict(), 'os_release': read_text('/etc/os-release'),
                              'is_wsl': 'microsoft' in platform.release().lower(),
                              'container_marker': read_text('/run/systemd/container'),
                              'docker_marker_exists': Path('/.dockerenv').exists(),
                              'ssh_environment_present': any(os.environ.get(k) for k in ['SSH_CONNECTION','SSH_TTY']),
                              'python_executable': sys.executable, 'python_version': platform.python_version(),
                              'python_prefix': sys.prefix,
                              'variables': {k:os.environ.get(k) for k in ['CONDA_PREFIX','CONDA_DEFAULT_ENV','CUDA_VISIBLE_DEVICES',
                                  'NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','CUDA_HOME','CUDA_PATH','LD_LIBRARY_PATH']},
                              'device_nodes': {p:node(p) for p in ['/dev/dxg','/dev/nvidia0','/dev/nvidiactl']},
                              'dev_mounts': [line for line in mounts.splitlines() if len(line.split())>4 and line.split()[4]=='/dev'],
                              'process_restrictions': [line for line in status.splitlines() if line.startswith(('NoNewPrivs:','Seccomp:','Seccomp_filters:','CapEff:'))]},
              'nvidia_smi': command([smi]),
              'gpu_query': command([smi,'--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
              'checks': {'cuda_matmul': 'not_run', 'model_cuda_forward': 'not_run' if args.check_model else 'not_requested'},
              'no_training_or_optimizer_steps': True}
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
                           'cuda_built': torch.backends.cuda.is_built(),
                           'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}
        if not report['torch']['cuda_available']:
            try:
                library = ctypes.CDLL('libcuda.so.1')
                library.cuInit.argtypes, library.cuInit.restype = [ctypes.c_uint], ctypes.c_int
                report['driver_init_code'] = library.cuInit(0)
            except Exception as e:
                report['driver_library_error'] = str(e)
            try:
                torch.cuda.init()
            except Exception as e:
                report['cuda_init_error'] = str(e)
            report['status'] = 'blocked_cuda_unavailable'
        else:
            device = torch.device('cuda:0')
            props = torch.cuda.get_device_properties(device)
            report['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                             'compute_capability': [props.major,props.minor], 'torch_architectures': torch.cuda.get_arch_list()}
            # Small real GPU kernels, including autograd. No model parameter or
            # optimizer update. The identity product has an exact expected value.
            a = torch.arange(256*256,device=device,dtype=torch.float32).reshape(256,256)/65536
            a.requires_grad_(True)
            b = torch.eye(256,device=device)
            c = a @ b
            torch.cuda.synchronize(device)
            if c.device != device or not torch.equal(a,c):
                raise ValueError('CUDA matrix product device/value check failed')
            c.square().mean().backward()
            if a.grad is None or a.grad.device != device or not torch.isfinite(a.grad).all():
                raise ValueError('CUDA autograd check failed')
            report['checks']['cuda_matmul'] = {'passed': True,'input_device':str(a.device),
                'output_device':str(c.device),'gradient_device':str(a.grad.device),'shape':[256,256]}
            if args.check_model:
                from runtime import PosePredictor
                from imuposer.math import r6d_to_rotation_matrix
                from imuposer.smpl.parametricModel import ParametricModel
                predictor = PosePredictor(protocol['checkpoint']['path'],device='cuda:0',threads=2)
                body = ParametricModel(protocol['smpl_model']['path'],device=device)
                x = torch.zeros(1,8,60,device=device)
                x[:,:,15:].reshape(1,8,5,3,3)[:,:,[0,1,4]] = torch.eye(3,device=device)
                with torch.inference_mode():
                    y = predictor.model(x,torch.tensor([8],dtype=torch.long))
                    joints = body.forward_kinematics(r6d_to_rotation_matrix(y).reshape(-1,216),calc_mesh=False)[1]
                if (any(p.device != device for p in predictor.model.parameters()) or y.device != device
                        or joints.device != device or not torch.isfinite(y).all() or not torch.isfinite(joints).all()):
                    raise ValueError('IMUPoser or SMPL failed CUDA device/finite check')
                report['checks']['model_cuda_forward'] = {'passed': True,'parameter_device':str(next(predictor.model.parameters()).device),
                    'output_device':str(y.device),'fk_joint_device':str(joints.device),'output_shape':list(y.shape),
                    'frames':8,'optimizer_steps':0,'data':'synthetic masked input; no DIP loading'}
            report['status'] = 'passed_cuda_and_model' if args.check_model else 'passed_cuda_compute'
    except Exception as e:
        report['status'],report['error'] = 'failed_gpu_preflight',repr(e)
    hashes_after = {str(path):digest(path) for path in protected}
    report['protected_files'] = hashes_after
    report['protected_files_unchanged'] = hashes_before == hashes_after
    if not report['protected_files_unchanged']:
        report['status'] = 'failed_integrity_check'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str(args.output),'torch':report.get('torch'),
                      'gpu':report.get('gpu'),'checks':report['checks'],'protected_files_unchanged':report['protected_files_unchanged']},indent=2))
    return 0 if report['status'].startswith('passed_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
