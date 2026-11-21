#!/usr/bin/env python3
"""User-launched bounded Seller repair. --cpu-check does not query or use GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare():
    import yaml
    from importlib.metadata import version
    from training.seller_consistency_data import DATA,PARENT,SEED,audit,prepare as prepare_data,write_review
    from training.official_buyer_data import sha
    from training.train_official_seller import check_run as check_parent
    if not DATA.exists():prepare_data()
    manifest=audit();write_review(DATA,manifest);check_parent(PARENT.parent.parent)
    cfg=yaml.safe_load((ROOT/'training/configs/seller_official_v1.yaml').read_text())
    cfg.update(manifest['recipe']);cfg.update(experiment='seller_consistency_v2',seed=SEED,data_dir=str(DATA),parent_adapter=str(PARENT),
        data_manifest_sha256=sha(DATA/'manifest.json'),weight_decay=.01,history_turns=6,max_train_examples=288,
        objective='Ordinary native completion CE; original replay + separately labelled engine controls; no new think format or special token weights.')
    paths=[Path(__file__),ROOT/'training/seller_consistency_train.py',ROOT/'training/buyer_operations_train.py',ROOT/'training/buyer_surgical_train.py',
        ROOT/'training/official_seller_train.py',ROOT/'training/official_buyer_train.py',ROOT/'training/reasoning_checkpoint.py',
        ROOT/'training/decision_ablation.py',ROOT/'training/buyer_hf_check.py',ROOT/'training/basic_checks.py',ROOT/'training/local_resources.py']
    cfg['sources']={str(p.resolve()):sha(p) for p in paths}
    cfg['versions']={name:version(name) for name in ('torch','transformers','peft','safetensors','vllm')}
    receipt=ROOT/'.cache/fast_kernels/installed.json'
    if not receipt.is_file():raise ValueError('Fast-kernel overlay missing; no silent slow training')
    cfg['kernel_receipt_sha256']=sha(receipt)
    return cfg,manifest


def check_run(folder):
    from importlib.metadata import version
    from training.official_buyer_data import sha
    from training.seller_consistency_data import audit
    folder=Path(folder);cfg=json.loads((folder/'train_config.json').read_text())
    for p,h in cfg['sources'].items():
        if sha(p)!=h:raise ValueError(f'Frozen repair source changed: {p}')
    if any(version(n)!=v for n,v in cfg['versions'].items()):raise ValueError('Core environment changed')
    if sha(ROOT/'.cache/fast_kernels/installed.json')!=cfg['kernel_receipt_sha256']:raise ValueError('Fast-kernel receipt changed')
    data=Path(cfg['data_dir']);audit(data)
    if sha(data/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Repair manifest changed')
    if sha(folder/'native_cases.json')!=sha(data/'evaluation_cases.json'):raise ValueError('Repair cases changed')
    if (cfg['learning_rate'],cfg['max_steps'],cfg['warmup_steps'],cfg['effective_batch_size'],cfg['checkpoint_steps'])!=(5e-6,36,3,8,[18,36]):raise ValueError('Bounded recipe changed')
    return cfg


def launch(folder,cfg,env):
    from training.basic_checks import read_new
    command=[sys.executable,'-u']
    if cfg['world_size']>1:command+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={cfg["world_size"]}','--']
    command+=[str(Path(__file__).resolve()),'--worker','--run',str(folder)]
    log_path=folder/'runner.log';offsets={};began=time.monotonic();last=0
    with log_path.open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(log_path,offsets):print(line,flush=True)
                now=time.monotonic()
                if done or now-last>=20:
                    print(f'[Seller repair] elapsed {now-began:.0f}s | stage progress/ETA above | initialization ETA estimating; log {log_path}',flush=True);last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=proc.returncode,elapsed_seconds=time.monotonic()-began),indent=2)+'\n')
    if proc.returncode:raise RuntimeError(f'Seller repair exit{proc.returncode}; inspect {log_path}. No automatic restart.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--wait',action='store_true')
    parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--resume',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'training/runs')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--run',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        from training.seller_consistency_train import worker
        return worker(args.run)
    print('CPU: checking repaired data, parent step12, unchanged v2 personas, source fingerprints and kernel receipt.',flush=True)
    if args.resume:
        folder=args.resume.resolve();cfg=check_run(folder)
        if (folder/'training_finished.json').is_file():
            from training.seller_consistency_train import write_review
            write_review(folder);print(f'Already finished. Read {folder}/seller_consistency_review.md; no GPU work.');return folder
        checkpoints=[p for p in (folder/'checkpoints').glob('step_*') if (p/'checkpoint_complete.json').is_file()]
        cfg['start_step']=max((int(p.name.split('_')[1]) for p in checkpoints),default=0)
        if cfg['start_step']:
            from training.decision_ablation import check_fingerprint
            cp=folder/'checkpoints'/f'step_{cfg["start_step"]:04d}'
            check_fingerprint(cp,json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints'])
            if any(not (cp/n).is_file() for n in ['optimizer.pt',*[f'rng_rank{r}.pt' for r in range(cfg['world_size'])]]):raise ValueError('Incomplete optimizer/RNG resume state')
            if not (folder/'native_step_0000.json').is_file():raise ValueError('Missing parent behavior control')
    else:cfg,manifest=prepare()
    if args.cpu_check:
        print('PASS CPU only: 216 official replay +72 engine controls, 36 steps, parent+18/36 evaluations; no GPU query or model execution.',flush=True);return cfg
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    # Check disk before entering a potentially long resource wait.
    parent,note=output_parent(ROOT,args.output_dir,min_free_gib=8.)
    if note:print(note,flush=True)
    began=time.monotonic()
    while True:
        try:cards,skipped,selection=training_gpu_plan(args.gpus,required=1);break
        except RuntimeError as error:
            resource='No GPU has enough free memory after reserve' in str(error) or ('Requested 1 usable GPUs, found 0' in str(error))
            if not args.wait or not resource:raise
            print(f'Waiting for 18GiB budget +2GiB reserve; elapsed {time.monotonic()-began:.0f}s; ETA unknown. {error}',flush=True);time.sleep(15)
    if args.resume and len(cards)!=cfg['world_size']:raise ValueError('Exact resume requires the original GPU count')
    cfg.update(world_size=len(cards),devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    cache,note=runtime_cache(ROOT)
    if note:print(note,flush=True)
    if not args.resume:
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_consistency_',dir=parent))
        shutil.copyfile(Path(cfg['data_dir'])/'evaluation_cases.json',folder/'native_cases.json')
        shutil.copytree(ROOT/'training/persona_versions/seller_checkpoint1_v2',folder/'personas')
    if shutil.disk_usage(folder).free<8*1024**3:raise ValueError('Less than 8GiB disk space remains')
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    fast=ROOT/'.cache/fast_kernels/site-packages';tile=ROOT/'.cache/fast_kernels/tilelang-cache';tile.mkdir(parents=True,exist_ok=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','TILELANG_CACHE_DIR':str(tile),
        'PYTORCH_ALLOC_CONF':'expandable_segments:True','PYTHONPATH':str(fast)+os.pathsep+str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
        'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    print(f'Run: {folder}; physical GPUs {[c["index"] for c in cards]}; world {len(cards)}; fixed global batch8.',flush=True)
    print('Full-context backward preflight → parent step12 30 decisions → 18 steps →30 decisions/gate → at most18 further steps+30 decisions. No automatic market/deployment.',flush=True)
    launch(folder,cfg,env)
    from training.seller_consistency_train import write_review
    result=write_review(folder)
    print(f'Completed: {result["status"]}. Read {folder}/seller_consistency_review.md; no automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped by user; logs/checkpoints retained.',flush=True);raise SystemExit(130)
