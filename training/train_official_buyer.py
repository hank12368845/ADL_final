#!/usr/bin/env python3
"""User-launched official Buyer baseline. CPU check never queries or starts GPUs."""
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
SOURCES=('training/train_official_buyer.py','training/official_buyer_train.py','training/official_buyer_data.py',
         'training/buyer_operations_train.py','training/reasoning_checkpoint.py','training/buyer_hf_check.py',
         'training/buyer_diagnostics.py','training/diagnostic_scores.py','training/official_buyer_report.py','training/buyer_operations_data.py',
         'training/buyer_operations_v2_data.py','training/buyer_operations_v2_1_data.py',
         *tuple(f'training/persona_versions/buyer_official_v1/persona_{i}.md' for i in (1,2,3)))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(cfg):
    from training.official_buyer_train import check_data,batch_indices,freegen_cases,official_cases
    from huggingface_hub import try_to_load_from_cache
    if cfg['epochs']!=1 or not cfg['thinking'] or cfg['rank']>32 or cfg['rank']<1:raise ValueError('This baseline fixes one full epoch, ON, rank<=32')
    if cfg.get('initial_adapter'):raise ValueError('Official baseline starts from the original base, not a repaired adapter')
    train,val,manifest=check_data(cfg);cfg=copy.deepcopy(cfg)
    batches=batch_indices(len(train),cfg['effective_batch_size'],cfg['seed'])
    cfg['max_steps']=len(batches);cfg['warmup_steps']=max(1,math.ceil(len(batches)*cfg['warmup_ratio']))
    cfg['checkpoint_steps']=sorted({max(1,math.ceil(len(batches)*q)) for q in (.25,.5,.75,1.)})
    cfg['data_manifest_sha256']=sha(ROOT/cfg['data_dir']/'manifest.json')
    cfg['training_sources']={p:sha(ROOT/p) for p in SOURCES}
    from importlib.metadata import version,PackageNotFoundError
    cfg['versions']={}
    for name in ('torch','transformers','peft','accelerate','safetensors','pyarrow','huggingface_hub','vllm','flash-linear-attention','causal-conv1d'):
        try:cfg['versions'][name]=version(name)
        except PackageNotFoundError:cfg['versions'][name]=None
    cfg['world_size']=2;cfg['objective']='Native full assistant completion CE only; no extra tool-name loss'
    index=try_to_load_from_cache(cfg['base_model'],'model.safetensors.index.json',revision=cfg['revision'])
    if not isinstance(index,str):raise ValueError('Pinned base not cached; run training/download.sh')
    weights=json.loads(Path(index).read_text())
    if any(not (Path(index).parent/name).exists() for name in set(weights['weight_map'].values())):raise ValueError('Incomplete cached base weights')
    cases=freegen_cases();heldout=official_cases(cfg)
    first_ids={train.metadata[i]['id'] for batch in batches[:cfg['checkpoint_steps'][0]] for i in batch}
    train_cases=official_cases(cfg,split='train',allowed_ids=first_ids)
    cfg['native_cases_sha256']=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    cfg['official_native_cases_sha256']=hashlib.sha256(json.dumps(heldout,sort_keys=True).encode()).hexdigest()
    cfg['train_native_cases_sha256']=hashlib.sha256(json.dumps(train_cases,sort_keys=True).encode()).hexdigest()
    cfg['train_examples']=len(train);cfg['validation_examples']=len(val)
    cfg['padding_examples']=len(batches)*cfg['effective_batch_size']-len(train)
    cfg['official_data_mode']='Only actual short teacher reasoning targets; OFF-only source actions excluded, not supplied with invented reasoning.'
    return cfg,batches,cases,heldout,train_cases,manifest


def resume_info(cfg,checkpoint,batches,cases):
    checkpoint=Path(checkpoint).resolve();old=json.loads((checkpoint.parent.parent/'train_config.json').read_text())
    complete=json.loads((checkpoint/'checkpoint_complete.json').read_text());step=complete['step']
    keys=('base_model','revision','data_dir','rank','alpha','dropout','learning_rate','weight_decay','warmup_steps','max_steps',
          'effective_batch_size','seed','max_length','thinking','data_manifest_sha256','native_cases_sha256','official_native_cases_sha256','train_native_cases_sha256','training_sources','world_size',
          'target_modules','validation_max','probe_max_tokens','checkpoint_steps','gradient_checkpointing')
    if any(cfg[k]!=old[k] for k in keys):raise ValueError('Resume recipe/data/source changed')
    if not 0<step<cfg['max_steps']:raise ValueError('Checkpoint must precede the full-epoch target')
    previous=json.loads((checkpoint.parent.parent/'batch_plan.json').read_text())
    if previous['batches'][:step]!=batches[:step]:raise ValueError('Resume batch prefix differs')
    for name in ('adapter_model.safetensors','optimizer.pt','rng_rank0.pt','rng_rank1.pt'):
        if not (checkpoint/name).is_file():raise ValueError('Incomplete exact-resume checkpoint')
    from training.decision_ablation import check_fingerprint
    check_fingerprint(checkpoint,complete['adapter_fingerprints'])
    return {'resume_checkpoint':str(checkpoint),'start_step':step,'baseline_behavior':complete['baseline_behavior'],'reference_run':old.get('reference_run',str(checkpoint.parent.parent))}


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'training/configs/buyer_official_v1.yaml')
    p.add_argument('--gpus',default='0,1');p.add_argument('--output-dir',type=Path,default=ROOT/'training/runs')
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    print('CPU: official source/splits, full-epoch coverage, native cases and pinned base checks...',flush=True)
    cfg,batches,cases,heldout,train_cases,data=prepare(cfg)
    if a.resume:cfg.update(resume_info(cfg,a.resume,batches,cases))
    print(f'Official ON train={cfg["train_examples"]}, validation={cfg["validation_examples"]}; one epoch={cfg["max_steps"]} steps, '
          f'batch={cfg["effective_batch_size"]}, warmup={cfg["warmup_steps"]}, checkpoints={cfg["checkpoint_steps"]}; padding replay={cfg["padding_examples"]}.',flush=True)
    print(f'Objective: {cfg["objective"]}. Native HF checks per evaluation: {len(cases)} current-rule + {len(heldout)} official held-out + {len(train_cases)} training-subset = {len(cases)+len(heldout)+len(train_cases)} calls; no gold reasoning supplied.',flush=True)
    if a.cpu_check:print('PASS: CPU only; no GPU query or model inference.',flush=True);return cfg
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    devices,skipped,selection=training_gpu_plan(a.gpus,2)
    if len(devices)!=2:raise ValueError('Official baseline requires two usable GPUs')
    cfg.update(devices=devices,gpu_selection=selection,skipped_gpus=skipped)
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=10.);cache_env,cache_note=runtime_cache(ROOT)
    for message in (note,cache_note):
        if message:print(message,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_official_',dir=parent))
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (folder/'batch_plan.json').write_text(json.dumps({'world_size':2,'effective_batch_size':cfg['effective_batch_size'],'batches':batches},indent=1))
    (folder/'native_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=1))
    (folder/'official_native_cases.json').write_text(json.dumps(heldout,ensure_ascii=False,indent=1))
    (folder/'train_native_cases.json').write_text(json.dumps(train_cases,ensure_ascii=False,indent=1))
    import shutil
    shutil.copytree(ROOT/'training/persona_versions/buyer_official_v1',folder/'personas');(folder/'agent.yaml').write_text('reasoning: true\n')
    print(f'GPU devices: {devices}; NEW base LoRA; output/log/checkpoints: {folder}',flush=True)
    print('Loading / memory preflight / baseline validation ETA estimating. Training tqdm shows step+micro, elapsed, speed and ETA; native checks have their own progress.',flush=True)
    if a.plan_only:print('PASS: resources planned; no training started.');return folder
    env={**os.environ,**cache_env,'CUDA_VISIBLE_DEVICES':','.join(d['index'] for d in devices),'PYTHONUNBUFFERED':'1',
         'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
         'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'training/official_buyer_train.py'),'--session',str(folder)]
    started=time.monotonic()
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
    (folder/'wrapper_result.json').write_text(json.dumps({'exit':code,'elapsed_seconds':time.monotonic()-started}))
    if code:raise SystemExit(f'FAIL: official worker exit={code}; inspect {folder}/runner.log')
    status=json.loads((folder/'finished.json').read_text())
    from training.official_buyer_report import write_review
    review=write_review(folder)
    print(f'Read first: {review}',flush=True)
    print(f'Finished status={status["status"]}; development candidate={status["development_candidate"]}. Read {folder}/finished.json and checkpoint_scores.json; behavior and later market validation required.',flush=True)
    return folder


if __name__=='__main__':main()
