#!/usr/bin/env python3
"""User-launched official Seller baseline. --cpu-check never queries or uses GPUs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare(cfg):
    from huggingface_hub import try_to_load_from_cache
    from training.official_seller_data import audit_data,generate,sha
    from training.official_seller_train import batch_indices,frozen_cases
    if (cfg['epochs'],cfg['thinking'],cfg['rank'],cfg['alpha'],cfg['effective_batch_size'])!=(1,True,16,32,8):
        raise ValueError('Baseline fixes one subset epoch, ON, rank16/alpha32 and global batch8')
    if cfg.get('initial_adapter'):raise ValueError('Start from original base; no Buyer or TA adapter initialization')
    folder=ROOT/cfg['data_dir']
    if not folder.exists():
        print('CPU: curating/tokenizing official Seller data; progress below. No GPU work.',flush=True)
        generate(cfg)
    data,manifest=audit_data(cfg);cfg=copy.deepcopy(cfg)
    batches=batch_indices(len(data['train']),cfg['effective_batch_size'],cfg['seed'])
    cfg.update(train_examples=len(data['train']),validation_examples=len(data['validation']),max_steps=len(batches),
               warmup_steps=max(1,math.ceil(len(batches)*cfg['warmup_ratio'])),
               checkpoint_steps=sorted({max(1,math.ceil(len(batches)*q)) for q in (.25,.5,1.)}),
               data_manifest_sha256=sha(folder/'manifest.json'))
    cases=frozen_cases(cfg,batches)
    paths=[Path(__file__),ROOT/'training/official_seller_data.py',ROOT/'training/official_seller_train.py',
           ROOT/'training/buyer_operations_train.py',ROOT/'training/buyer_surgical_train.py',ROOT/'training/official_buyer_train.py',
           ROOT/'training/reasoning_checkpoint.py',ROOT/'training/buyer_hf_check.py',ROOT/'training/decision_ablation.py',
           ROOT/'training/basic_checks.py',ROOT/'training/local_resources.py']
    paths+=list(ROOT.glob('adlmarket/*.py'))+[ROOT/'training/configs/seller_official_v1.yaml',
           ROOT/'training/persona_versions/seller_checkpoint1_v1/persona_1.md',*Path(ROOT/'training/persona_versions/seller_checkpoint1_v2').glob('persona_*.md')]
    cfg['sources']={str(p.resolve()):sha(p) for p in paths}
    cfg['native_cases_sha256']=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    index=try_to_load_from_cache(cfg['base_model'],'model.safetensors.index.json',revision=cfg['revision'])
    if not isinstance(index,str):raise ValueError('Pinned base missing; run training/download.sh first')
    files=set(json.loads(Path(index).read_text())['weight_map'].values())
    if any(not (Path(index).parent/f).is_file() for f in files):raise ValueError('Incomplete cached base')
    from importlib.metadata import version
    cfg['versions']={name:version(name) for name in ('torch','transformers','peft','safetensors','vllm')}
    receipt=ROOT/'.cache/fast_kernels/installed.json'
    if not receipt.is_file():raise ValueError('Fast-kernel overlay missing; install with training/install_fast_kernels.py --install')
    cfg['kernel_receipt_sha256']=sha(receipt)
    cfg['objective']='Ordinary native assistant completion CE; original official reasoning and tool targets; prompt/history masked; no special tool-name loss.'
    return cfg,batches,cases,manifest


def check_run(folder):
    from training.official_seller_data import sha,audit_data
    cfg=json.loads((Path(folder)/'train_config.json').read_text())
    if any(sha(p)!=h for p,h in cfg['sources'].items()):raise ValueError('Frozen Seller training source changed')
    if sha(ROOT/cfg['data_dir']/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Frozen dataset changed')
    if sha(ROOT/'.cache/fast_kernels/installed.json')!=cfg['kernel_receipt_sha256']:raise ValueError('Kernel overlay receipt changed')
    from importlib.metadata import version
    if any(version(name)!=v for name,v in cfg['versions'].items()):raise ValueError('Training core environment changed')
    cases=json.loads((Path(folder)/'native_cases.json').read_text())
    if hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()!=cfg['native_cases_sha256']:raise ValueError('Frozen probes changed')
    data,_=audit_data(cfg)
    from training.official_seller_train import batch_indices
    if json.loads((Path(folder)/'batch_plan.json').read_text())['batches']!=batch_indices(len(data['train']),cfg['effective_batch_size'],cfg['seed']):
        raise ValueError('Training schedule changed')
    return cfg


def launch(folder,cfg,env):
    from training.basic_checks import read_new
    command=[sys.executable,'-u']
    if cfg['world_size']>1:command+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={cfg["world_size"]}','--']
    command += [str(Path(__file__).resolve()),'--worker','--run',str(folder)]
    log_path=folder/'runner.log';offsets={};began=time.monotonic();last=0
    print(f'GPU loading/preflight/base evaluation → training → checkpoint evaluations. Log:{log_path}; startup ETA estimating.',flush=True)
    with log_path.open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(log_path,offsets):print(line,flush=True)
                now=time.monotonic()
                if done or now-last>=20:
                    print(f'[Seller pipeline] elapsed{now-began:.0f}s | per-stage progress/ETA above; first work/loading ETA estimating.',flush=True);last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=proc.returncode,elapsed_seconds=time.monotonic()-began),indent=2)+'\n')
    if proc.returncode:raise RuntimeError(f'Seller worker exit{proc.returncode}; inspect {log_path}. No automatic restart.')


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'training/configs/seller_official_v1.yaml')
    p.add_argument('--gpus',default='0,1');p.add_argument('--wait',action='store_true',help='Poll resource availability before launch; not a reservation')
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'training/runs');p.add_argument('--worker',action='store_true');p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.worker:
        from training.official_seller_train import worker
        return worker(a.run)
    if a.resume:
        folder=a.resume.resolve();cfg=check_run(folder)
        if (folder/'training_finished.json').exists():
            from training.official_seller_train import write_review
            write_review(folder);print(f'Already finished; review:{folder}/seller_training_review.md; no GPU work.');return folder
        completed=[p for p in (folder/'checkpoints').glob('step_*') if (p/'checkpoint_complete.json').is_file()]
        cfg['start_step']=max((json.loads((p/'checkpoint_complete.json').read_text())['step'] for p in completed),default=0)
        if cfg['start_step']:
            from training.decision_ablation import check_fingerprint
            cp=folder/'checkpoints'/f'step_{cfg["start_step"]:04d}'
            check_fingerprint(cp,json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints'])
            required=['optimizer.pt',*[f'rng_rank{r}.pt' for r in range(cfg['world_size'])]]
            if any(not (cp/n).is_file() for n in required):raise ValueError('No exact-resume optimizer/RNG state')
    else:
        print('CPU: official Seller source, group splits, current personas/tools and pinned base checks...',flush=True)
        cfg,batches,cases,manifest=prepare(yaml.safe_load(a.config.read_text()))
        print(f'CPU ready: train{cfg["train_examples"]}, heldout{cfg["validation_examples"]}; {cfg["max_steps"]}steps/one subset epoch; '
              f'checkpoints{cfg["checkpoint_steps"]}; native checks{len(cases)} per version; no gold reasoning supplied.',flush=True)
        print(f'Train actions:{manifest["splits"]["train"]["actions"]}; max{manifest["splits"]["train"]["max_tokens"]}tokens, history≤2; native reasoning≤384tokens.',flush=True)
    if a.cpu_check:print('PASS CPU only; no GPU query, services or model execution.',flush=True);return cfg
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    began=time.monotonic()
    while True:
        try:cards,skipped,selection=training_gpu_plan(a.gpus,required=1);break
        except RuntimeError as error:
            if not a.wait or 'No GPU has enough free memory after reserve' not in str(error):raise
            print(f'Waiting for18GiB training budget +2GiB reserve; elapsed{time.monotonic()-began:.0f}s, ETA unknown. {error}',flush=True);time.sleep(15)
    if a.resume and len(cards)!=cfg['world_size']:raise ValueError('Exact resume must keep GPU count; use --gpus with the original number of usable cards')
    cfg.update(world_size=len(cards),devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    cache,note=runtime_cache(ROOT)
    if note:print(note,flush=True)
    if not a.resume:
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=8.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_official_',dir=parent))
        (folder/'batch_plan.json').write_text(json.dumps(dict(batches=batches),indent=1)+'\n')
        (folder/'native_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=1)+'\n')
        import shutil
        shutil.copytree(ROOT/'training/persona_versions/seller_checkpoint1_v2',folder/'personas')
        (folder/'agent.yaml').write_text('reasoning: true\n')
    else:
        import shutil
        if shutil.disk_usage(folder).free<8*1024**3:raise RuntimeError('Less than8GiB free; no resumed GPU worker launched')
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    fast=ROOT/'.cache/fast_kernels/site-packages';tile=ROOT/'.cache/fast_kernels/tilelang-cache';tile.mkdir(parents=True,exist_ok=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(d['index'] for d in cards),'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','TILELANG_CACHE_DIR':str(tile),
         'PYTORCH_ALLOC_CONF':'expandable_segments:True','PYTHONPATH':str(fast)+os.pathsep+str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    print(f'Run:{folder}; actual physical GPUs:{[d["index"] for d in cards]}; DDP world{len(cards)}, global batch8 unchanged.',flush=True)
    print('Fast-kernel dispatch and longest forward/backward must pass on GPU first. Total ETA unknown until measured; no market or deployment follows automatically.',flush=True)
    launch(folder,cfg,env)
    from training.official_seller_train import write_review
    status=write_review(folder)
    print(f'Completed:{status["status"]}. Read {folder}/seller_training_review.md; candidate={status["development_candidate"]}. No automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped by user; checkpoints/logs retained.',flush=True);raise SystemExit(130)
