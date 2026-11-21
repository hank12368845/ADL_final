#!/usr/bin/env python3
"""One command: dollar-line training, step6 stop gate, conditional step12 evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCES=('training/train_buyer_dollar_lines.py','training/buyer_dollar_train.py','training/buyer_dollar_data.py',
    'training/buyer_dollar_lines.py','training/buyer_check_lines.py','training/buyer_self_checks.py',
    'training/buyer_operations_train.py','training/official_buyer_data.py','training/official_buyer_train.py',
    'training/step84_repair_plan.py','training/buyer_hf_check.py','training/buyer_latest_brakes_train.py','training/reasoning_checkpoint.py')


def prepare(cfg):
    from importlib.metadata import version
    from training.buyer_dollar_data import check_data
    from training.buyer_latest_brakes_data import sha
    cfg=dict(cfg);_,_,batches,m=check_data(cfg)
    for k in ('max_steps','warmup_steps','checkpoint_steps','effective_batch_size','learning_rate','seed','thinking'):
        if cfg[k]!=m[k]:raise ValueError(f'Bounded dollar recipe changed: {k}')
    if cfg['max_length']!=12288:raise ValueError('Unexpected context bound')
    parent=json.loads((Path(m['parent_adapter']).parent.parent/'train_config.json').read_text())
    for k in ('base_model','revision','rank','alpha','dropout','target_modules'):cfg[k]=parent[k]
    if cfg['rank']>32:raise ValueError('Invalid submission rank')
    cfg.update(parent_adapter=m['parent_adapter'],parent_fingerprints=m['parent_fingerprints'],world_size=2,
        objective='mean(full check including CASH and closing think) + mean(field values) + .25 mean(native response)',
        data_manifest_sha256=sha(ROOT/cfg['data_dir']/'manifest.json'),training_sources={p:sha(ROOT/p) for p in SOURCES},
        versions={p:version(p) for p in ('torch','transformers','peft','safetensors','vllm')})
    return cfg,m,batches


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='0,1');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'training/configs/buyer_dollar_lines_v1.yaml');p.add_argument('--output-dir',type=Path)
    a=p.parse_args();print('CPU: frozen latest inputs, dollar targets, check/action/state agreement and staged stop checks...',flush=True)
    cfg,m,batches=prepare(yaml.safe_load(a.config.read_text()))
    print(f'48 existing cases, no new inputs. LR1e-5, warmup2, max12 steps; independent full-check loss; mean {m["mean_tokens"]:.0f} tokens.',flush=True)
    print('Step6: seen12 ONLY. Fail -> stop. Pass -> continue to12 and evaluate all73. Cached parent baseline73; no model calls at step0.',flush=True)
    if a.cpu_check:print('PASS: CPU only; no GPU work.');return cfg
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    cards,skipped,selection=training_gpu_plan(a.gpus,2)
    if len(cards)!=2 or any(c['budget_gib']<18 for c in cards):raise ValueError('Need two GPUs with >=18GiB startup budget after headroom')
    parent,note=output_parent(ROOT,a.output_dir or ROOT/'training/runs',min_free_gib=3.)
    cache,cache_note=runtime_cache(ROOT)
    for msg in (note,cache_note):
        if msg:print(msg,flush=True)
    cfg.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_dollar_lines_',dir=parent))
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n');(folder/'batch_plan.json').write_text(json.dumps(batches)+'\n')
    (folder/'personas').mkdir()
    for i in (1,2,3):shutil.copy2(ROOT/cfg['data_dir']/f'persona_{i}.md',folder/'personas'/f'persona_{i}.md')
    (folder/'agent.yaml').write_text('reasoning: true\n')
    print(f'GPUs {[c["index"] for c in cards]}; output/log {folder}; loading/preflight ETA estimating.',flush=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'training/buyer_dollar_train.py'),'--session',str(folder)]
    began=time.monotonic()
    with (folder/'runner.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=process.stdout.read1(4096):
                message=chunk.decode(errors='replace');sys.stdout.write(message);sys.stdout.flush();log.write(message);log.flush()
            code=process.wait()
        except BaseException:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=code,elapsed_seconds=time.monotonic()-began)))
    if code:raise SystemExit(f'FAIL: worker exit={code}; inspect {folder}/runner.log')
    finished=json.loads((folder/'finished.json').read_text())
    print(f'Completed: {finished["status"]}, step{finished["step"]}. Read {folder}/dollar_review.md; no automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':main()
