#!/usr/bin/env python3
"""User-launched two-GPU re-evaluation with corrected EOS; never trains."""
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


def prepare_recheck(run,steps):
    from training.official_buyer_train import check_data
    from training.decision_ablation import check_fingerprint
    run=Path(run).resolve();cfg=json.loads((run/'train_config.json').read_text())
    if json.loads((run/'finished.json').read_text())['status']!='complete':raise ValueError('Expected completed original training run')
    check_data(cfg);plan=[{'step':0,'checkpoint':None}]
    for step in sorted(set(steps)):
        if step<=0:raise ValueError('Checkpoint steps must be positive; base included automatically')
        cp=run/'checkpoints'/f'step_{step:04d}';record=json.loads((cp/'checkpoint_complete.json').read_text())
        if record['step']!=step:raise ValueError('Checkpoint step mismatch')
        check_fingerprint(cp,record['adapter_fingerprints']);plan.append({'step':step,'checkpoint':str(cp)})
    hashes={}
    for name,key in (('native_cases.json','native_cases_sha256'),('official_native_cases.json','official_native_cases_sha256'),('train_native_cases.json','train_native_cases_sha256')):
        rows=json.loads((run/name).read_text());digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        if digest!=cfg[key]:raise ValueError('Original frozen cases changed')
        hashes[name]=digest
    for key in ('resume_checkpoint','start_step','baseline_behavior','reference_run'):cfg.pop(key,None)
    cfg.update(evaluation_only=True,evaluation_plan=plan,original_run=str(run),original_training_sources=cfg['training_sources'],
               recheck_sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                   ('training/recheck_official_buyer.py','training/official_buyer_train.py','training/official_buyer_report.py')})
    return cfg,hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--steps',type=int,nargs='+',default=[84,112]);p.add_argument('--gpus',default='0,1')
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--output-dir',type=Path,default=ROOT/'results/basic_checks')
    a=p.parse_args();cfg,hashes=prepare_recheck(a.run,a.steps)
    calls=sum(len(json.loads((Path(cfg['original_run'])/n).read_text())) for n in hashes)*len(cfg['evaluation_plan'])
    print(f'Recheck: base + steps {[e["step"] for e in cfg["evaluation_plan"][1:]]}, {calls} native calls, SAME original cases, max_new_tokens={cfg["probe_max_tokens"]}; zero training updates.',flush=True)
    if a.cpu_check:print('PASS: checkpoint fingerprints, data and frozen cases; no GPU queries or inference.');return
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    cards,skipped,selection=training_gpu_plan(a.gpus,2)
    if len(cards)!=2:raise ValueError('Recheck requires two usable GPUs')
    cfg.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=2.);cache_env,cache_note=runtime_cache(ROOT)
    for message in (note,cache_note):
        if message:print(message,flush=True)
    out=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_official_buyer_recheck_',dir=parent))
    (out/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for name in hashes:shutil.copy2(Path(cfg['original_run'])/name,out/name)
    print(f'Output/log: {out}; loading ETA estimating. Per-version validation/native progress includes elapsed, speed, ETA.',flush=True)
    env={**os.environ,**cache_env,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),
        'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'OMP_NUM_THREADS':'4','PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'training/official_buyer_train.py'),'--session',str(out)]
    began=time.monotonic()
    with (out/'runner.log').open('w') as log:
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
    (out/'wrapper_result.json').write_text(json.dumps({'exit':code,'elapsed_seconds':time.monotonic()-began,'training_updates':0}))
    if code:raise SystemExit(f'FAIL: recheck exit={code}; inspect {out}/runner.log')
    from training.official_buyer_report import write_review
    print(f'Completed. Read: {write_review(out)}',flush=True)


if __name__=='__main__':main()
