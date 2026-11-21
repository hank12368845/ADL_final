#!/usr/bin/env python3
"""Prepare or launch a small latest-persona safety continuation, user-controlled GPUs."""
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
SOURCES=('training/train_buyer_latest_brakes.py','training/buyer_latest_brakes_train.py','training/buyer_latest_brakes_data.py',
    'training/buyer_operations_train.py','training/official_buyer_train.py','training/official_buyer_data.py',
    'training/step84_repair_plan.py','training/buyer_hf_check.py','training/reasoning_checkpoint.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cfg):
    from training.buyer_latest_brakes_train import check_data
    cfg=dict(cfg);train,val,batches,m=check_data(cfg)
    if cfg['max_steps']!=36 or cfg['warmup_steps']!=3 or cfg['effective_batch_size']!=8 or not cfg['thinking']:
        raise ValueError('This bounded experiment fixes 36 steps, warmup3, batch8 and thinking ON')
    if cfg['checkpoint_steps']!=[12,24,36] or cfg['learning_rate']!=5e-6:raise ValueError('Bounded safety recipe changed')
    if cfg['max_length']!=12288 or cfg['seed']!=m['seed']:raise ValueError('Length or seed differs from frozen data recipe')
    parent_cfg=json.loads((Path(m['parent_adapter']).parent.parent/'train_config.json').read_text())
    for key in ('base_model','revision','rank','alpha','dropout','target_modules'):cfg[key]=parent_cfg[key]
    cfg.update(parent_adapter=m['parent_adapter'],parent_fingerprints=m['parent_fingerprints'],
        data_manifest_sha256=sha(ROOT/cfg['data_dir']/'manifest.json'),training_sources={p:sha(ROOT/p) for p in SOURCES},
        objective='Native assistant completion CE only, no auxiliary decision weighting',world_size=2)
    from importlib.metadata import version
    cfg['versions']={name:version(name) for name in ('torch','transformers','peft','accelerate','safetensors','vllm')}
    if cfg['rank']>32:raise ValueError('Invalid adapter rank')
    return cfg,m,batches


def resume(cfg,checkpoint):
    from training.decision_ablation import check_fingerprint
    cp=Path(checkpoint).resolve();old_run=cp.parent.parent;old=json.loads((old_run/'train_config.json').read_text())
    record=json.loads((cp/'checkpoint_complete.json').read_text());step=record['step']
    for key in ('data_manifest_sha256','training_sources','base_model','revision','parent_fingerprints','rank','alpha','dropout',
                'target_modules','learning_rate','warmup_steps','max_steps','effective_batch_size','checkpoint_steps','seed','max_length','versions'):
        if cfg[key]!=old[key]:raise ValueError(f'Exact resume differs: {key}')
    if step not in (12,24):raise ValueError('Resume only a completed intermediate checkpoint')
    check_fingerprint(cp,record['adapter_fingerprints'])
    if any(not (cp/name).is_file() for name in ('optimizer.pt','rng_rank0.pt','rng_rank1.pt')):raise ValueError('Incomplete optimizer/RNG state')
    scores=json.loads((old_run/'checkpoint_scores.json').read_text())
    cfg.update(resume_checkpoint=str(cp),start_step=step,baseline_summary=record['baseline_summary'],
        previous_scores=[s for s in scores if s['step']<=step],reference_run=old.get('reference_run',str(old_run)))
    return cfg


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='0,1')
    p.add_argument('--config',type=Path,default=ROOT/'training/configs/buyer_latest_brakes_v1.yaml')
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'training/runs');a=p.parse_args()
    print('CPU: latest persona/data/source checks and fixed 3-buy / 3-offer / 2-wait batches...',flush=True)
    cfg,m,batches=prepare(yaml.safe_load(a.config.read_text()))
    if a.resume:cfg=resume(cfg,a.resume)
    print(f'Prepared: {m["train_examples"]} examples (144 official rehearsal + 144 engine controls), 36 steps; LR5e-6 warmup3; checkpoints12/24/36; mean {m["mean_tokens"]:.0f} tokens.',flush=True)
    print(f'Native checks: {m["evaluation_calls"]} cases/checkpoint including current latest regression + new held-out controls + seen subset; step0 reuses37 labelled parent outputs. No persona changes.',flush=True)
    if a.cpu_check:print('PASS: CPU only; no GPU query, training or model inference.');return cfg
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    cards,skipped,selection=training_gpu_plan(a.gpus,2)
    if len(cards)!=2 or any(c['budget_gib']<18 for c in cards):raise ValueError('Need two cards with >=18 GiB startup budget each after reserve')
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=3.);cache_env,cache_note=runtime_cache(ROOT)
    for message in (note,cache_note):
        if message:print(message,flush=True)
    cfg.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_latest_brakes_',dir=parent))
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copytree(ROOT/cfg['data_dir']/'personas',folder/'personas');(folder/'agent.yaml').write_text('reasoning: true\n')
    (folder/'batch_plan.json').write_text(json.dumps(batches)+'\n')
    print(f'GPU devices {[c["index"] for c in cards]}; output/log {folder}; load/preflight/step0 ETA estimating.',flush=True)
    env={**os.environ,**cache_env,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'PYTHONUNBUFFERED':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'training/buyer_latest_brakes_train.py'),'--session',str(folder)]
    began=time.monotonic()
    with (folder/'runner.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=proc.stdout.read1(4096):
                text=chunk.decode(errors='replace');sys.stdout.write(text);sys.stdout.flush();log.write(text);log.flush()
            code=proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps({'exit':code,'elapsed_seconds':time.monotonic()-began}))
    if code:raise SystemExit(f'FAIL: safety worker exit={code}; inspect {folder}/runner.log')
    from training.buyer_latest_brakes_train import write_review
    print(f'Completed. Read {write_review(folder)}',flush=True)
    return folder


if __name__=='__main__':main()
