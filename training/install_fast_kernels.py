#!/usr/bin/env python3
"""User-run CPU installer for isolated HF training kernels; never upgrades the base venv."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'.cache/fast_kernels'
PURE=('fla-core==0.6.0','flash-linear-attention==0.6.0','opt-einsum==3.4.0')
CUDA=('nvidia-cuda-nvcc==13.0.88','nvidia-cuda-crt==13.0.88','nvidia-nvvm==13.0.88',
      'nvidia-cuda-runtime==13.0.96','nvidia-cuda-cccl==13.0.85')
CORE=('torch','transformers','vllm','triton','peft','safetensors')


def signature():
    import torch
    return dict(versions={p:md.version(p) for p in CORE},torch_version=torch.__version__,cuda=torch.version.cuda,
                abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),python=f'cp{sys.version_info.major}{sys.version_info.minor}',machine=platform.machine())


def wheel_url(info):
    cuda=info['cuda'].split('.')[0];torch_version='.'.join(info['torch_version'].split('+')[0].split('.')[:2])
    name=f'causal_conv1d-1.7.0+cu{cuda}torch{torch_version}cxx11abi{str(info["abi"]).upper()}-{info["python"]}-{info["python"]}-linux_{info["machine"]}.whl'
    return 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/'+name


def fetch(url,path):
    path=Path(path)
    req=urllib.request.Request(url,headers={'User-Agent':'ADL-final-kernel-installer/1'})
    with urllib.request.urlopen(req,timeout=60) as response,path.with_suffix(path.suffix+'.part').open('wb') as stream:
        size=int(response.headers.get('Content-Length','0'));done=0;began=time.monotonic();last=0
        while chunk:=response.read(1024*1024):
            stream.write(chunk);done+=len(chunk);elapsed=time.monotonic()-began
            if elapsed-last>=2:
                eta=f'{(size-done)/(done/elapsed):.0f}s' if size and elapsed else 'estimating'
                print(f'Download {done/2**20:.1f}/{size/2**20:.1f}MiB | elapsed{elapsed:.0f}s | ETA{eta}',flush=True);last=elapsed
    path.with_suffix(path.suffix+'.part').replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command,env,logfile,label):
    print(f'{label}: starting; ETA estimating; log={logfile}',flush=True)
    began=time.monotonic();last=0;offset=0
    with Path(logfile).open('w') as stream:
        process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while True:
                done=process.poll() is not None
                with Path(logfile).open() as reader:reader.seek(offset);new=reader.read();offset=reader.tell()
                if new:print(new,end='',flush=True)
                elapsed=time.monotonic()-began
                if elapsed-last>=15:print(f'{label} elapsed{elapsed:.0f}s; compiling/downloading; ETA estimating.',flush=True);last=elapsed
                if done:break
                time.sleep(1)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if process.returncode:raise RuntimeError(f'{label} failed; inspect {logfile}')


def toolkit_root(prefix):
    candidates=[p.parent.parent for p in Path(prefix).rglob('nvcc') if p.parent.name=='bin']
    for p in candidates:
        if (p/'include/cuda_runtime.h').exists() and ((p/'lib').exists() or (p/'lib64').exists()):return p
    raise RuntimeError(f'CUDA13 compiler/header layout not found below {prefix}; no mismatched CUDA12 build attempted')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true')
    parser.add_argument('--prefix',type=Path,default=PREFIX);a=parser.parse_args();info=signature()
    if info['machine']!='x86_64' or info['cuda']!='13.0':raise ValueError('This reviewed installer targets the current x86_64/cu130 environment')
    plan=dict(prefix=str(a.prefix.resolve()),base_environment_unchanged=True,signature=info,packages=PURE,
        cuda_build_packages=CUDA,causal_wheel=wheel_url(info),gpu_work=False,
        fallback='Compile an exact recorded upstream commit using a localCUDA13 toolchain,CPU jobs<=2; no GPU auto-detection.')
    if not a.install:print(json.dumps(plan,indent=2));print('Plan only. Add --install to download/install; no GPU work.');return plan
    prefix=a.prefix.resolve();prefix.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(prefix).free<3*2**30:raise RuntimeError('Kernel installation needs at least3GiB free space')
    if (prefix/'installed.json').exists():raise ValueError('Prefix already completed; use its env.sh or choose a new --prefix')
    site=prefix/'site-packages';downloads=prefix/'downloads';downloads.mkdir(exist_ok=True)
    (prefix/'tmp').mkdir(exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'8.9','PIP_DISABLE_PIP_VERSION_CHECK':'1',
        'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        'PYTHONPATH':str(site)+os.pathsep+os.environ.get('PYTHONPATH',''),
        'PIP_CACHE_DIR':str(prefix/'pip-cache'),'TMPDIR':str(prefix/'tmp')}
    receipt=dict(plan,status='installing',downloads={},started=time.time())
    (prefix/'install_plan.json').write_text(json.dumps(receipt,indent=2)+'\n')
    pip=downloads/'pip.pyz';receipt['downloads'][str(pip)]=fetch('https://bootstrap.pypa.io/pip/pip.pyz',pip)
    base=[sys.executable,str(pip),'install','--no-deps','--no-build-isolation','--target',str(site)]
    extras=[]
    try:md.version('wheel')
    except md.PackageNotFoundError:extras=['wheel==0.45.1']
    run(base+list(PURE)+extras,env,prefix/'01_fla_install.log','1/3 FLA installation')
    # Never let pip choose another torch/CUDA ABI wheel.
    url=wheel_url(info);wheel=downloads/url.rsplit('/',1)[1]
    try:
        receipt['downloads'][str(wheel)]=fetch(url,wheel)
        run(base+[str(wheel)],env,prefix/'02_causal_install.log','2/3 exact causal-conv1d wheel')
        receipt['causal_method']='upstream_exact_wheel'
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        compiler=prefix/'cuda13'
        run([sys.executable,str(pip),'install','--no-deps','--target',str(compiler),*CUDA],env,prefix/'02_cuda_install.log','2/3 localCUDA13 toolchain')
        cuda=toolkit_root(compiler)
        version=subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True)
        if 'release 13.0' not in version:raise ValueError('Compiler and torch CUDA versions differ')
        lib=cuda/'lib64' if (cuda/'lib64').exists() else cuda/'lib'
        if not (lib/'libcudart.so').exists():
            runtime=list(lib.glob('libcudart.so.*'))
            if runtime:(lib/'libcudart.so').symlink_to(runtime[0].name)
        commit=json.loads(urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/Dao-AILab/causal-conv1d/commits/main',headers={'User-Agent':'ADL-final-kernel-installer/1'}),timeout=60).read())['sha']
        if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('Invalid upstream commit')
        archive=downloads/f'causal-conv1d-{commit}.tar.gz'
        receipt['downloads'][str(archive)]=fetch(f'https://github.com/Dao-AILab/causal-conv1d/archive/{commit}.tar.gz',archive)
        env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env['PATH'],CAUSAL_CONV1D_FORCE_BUILD='TRUE')
        env['CPATH']=os.pathsep.join(str(p) for p in (cuda/'include',cuda/'include/cccl') if p.exists())+os.pathsep+env.get('CPATH','')
        run(base+[str(archive)],env,prefix/'03_causal_build.log','3/3 causal-conv1d source build')
        receipt.update(causal_method='source_build',causal_commit=commit,cuda_home=str(cuda))
    if signature()!=info:raise RuntimeError('Core environment unexpectedly changed')
    receipt.update(status='installed_not_gpu_verified',finished=time.time(),installed_packages={d.metadata['Name']:d.version for d in md.distributions(path=[str(site)])})
    (prefix/'installed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Activation is explicit and reversible; the current vLLM market does not source it.
    (prefix/'env.sh').write_text('export PYTHONPATH='+shlex.quote(str(site))+'${PYTHONPATH:+:$PYTHONPATH}\n')
    print(f'Installed kernels in {site}; base Torch/vLLM unchanged. For later HF training: source {prefix}/env.sh',flush=True)
    print('No GPU correctness/speed claim yet; no training or market started. Installation receipts and logs saved.',flush=True)
    return receipt


if __name__=='__main__':main()
