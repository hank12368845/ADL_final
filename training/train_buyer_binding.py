#!/usr/bin/env python3
"""User-launched two-GPU preflight -> vLLM parent -> one-pass SFT -> checkpoint evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.buyer_binding_data import DATA,generate,check_data,sha


def prepare(data=DATA):
    from importlib.metadata import version
    m=generate(data);check_data(data)
    from training.buyer_binding_eval import validate_wire
    validate_wire(json.loads((Path(data)/'evaluation_cases.json').read_text()))
    sources=[ROOT/f'training/{p}.py' for p in ('train_buyer_binding','buyer_binding_data','buyer_binding_train','buyer_binding_eval',
        'buyer_operations_train','reasoning_checkpoint','official_buyer_train','chat_payload','local_resources')]
    sources+=list((ROOT/'adlmarket').glob('*.py'))+[ROOT/'configs/models.yaml',ROOT/'configs/basic_buyer.yaml']
    return dict(experiment='buyer_binding_v1',data_dir=str(Path(data).resolve()),data_manifest_sha256=sha(Path(data)/'manifest.json'),
        base_model=m['base_model'],revision=m['revision'],parent_adapter=m['parent_adapter'],parent_fingerprints=m['parent_fingerprints'],
        **m['model_recipe'],seed=m['seed'],thinking=True,learning_rate=1e-5,warmup_steps=4,max_steps=48,
        checkpoint_steps=[24,48],max_length=20480,effective_batch_size=8,world_size=2,
        objective='Ordinary native assistant completion CE; prompt masked; no auxiliary loss or forced format',
        sources={str(p.resolve()):sha(p) for p in sources},
        versions={name:version(name) for name in ('torch','transformers','peft','safetensors','vllm')})


def worker_command(folder,kind,labels=None):
    command=[sys.executable,'-u']
    if kind in ('preflight','train'):
        # torchrun allows option abbreviation and otherwise consumes our --run.
        command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2','--']
    command+=[str(Path(__file__).resolve()),'--worker',kind,'--run',str(folder)]
    if labels:command+=['--labels',','.join(labels)]
    return command


def stage(folder,cfg,kind,env,labels=None):
    from training.basic_checks import read_new
    from training.local_resources import training_gpu_plan
    selection=','.join(d['index'] for d in cfg['devices'])
    cards,skipped,snapshot=training_gpu_plan(selection,required=2)
    if [d['index'] for d in cards]!=[d['index'] for d in cfg['devices']]:
        raise ValueError('Physical GPU routes changed between stages')
    cfg.update(devices=cards,skipped_gpus=skipped)
    cfg.setdefault('stage_resource_checks',[]).append(dict(stage=kind,selection=snapshot))
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    command=worker_command(folder,kind,labels)
    print(f'Stage {kind}: loading / warm-up ETA estimating. Log:{folder}/{kind}.log',flush=True)
    began=time.monotonic();offsets={};last=0
    with (folder/f'{kind}.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(folder/f'{kind}.log',offsets):print(line,flush=True)
                now=time.monotonic()
                if done or now-last>=15:
                    print(f'[{kind}] elapsed{now-began:.0f}s | worker progress/ETA above; loading or first work ETA estimating.',flush=True);last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if proc.returncode:raise RuntimeError(f'{kind} exit{proc.returncode}; inspect {folder}/{kind}.log')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--data-dir',type=Path,default=DATA);p.add_argument('--output-dir',type=Path,default=ROOT/'training/runs')
    p.add_argument('--cpu-check',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--evaluate',type=Path)
    p.add_argument('--worker',choices=('preflight','train','eval_parent','eval_candidates'));p.add_argument('--run',type=Path);p.add_argument('--labels')
    a=p.parse_args()
    if a.worker:
        if a.worker in ('preflight','train'):
            from training.buyer_binding_train import worker
            return worker(a.run,a.worker=='preflight')
        from training.buyer_binding_eval import evaluate
        return evaluate(a.run,a.labels.split(','))
    if a.resume or a.evaluate:
        from training.buyer_binding_train import check_run
        folder=(a.resume or a.evaluate).resolve();cfg=check_run(folder)
    else:
        print('CPU: native-engine replay, dataset integrity, original personas and split audit; ETA estimating.',flush=True)
        cfg=prepare(a.data_dir);m=json.loads((Path(cfg['data_dir'])/'manifest.json').read_text())
        print(f'CPU ready:384examples/48steps,5board+2offer+1wait per batch,one pass; mean{m["mean_tokens"]:.0f},max{m["max_tokens"]} input+target tokens; think<= {m["max_think_tokens"]}.',flush=True)
        if a.cpu_check:print('PASS CPU only; no GPU query, services or model execution.',flush=True);return cfg
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_binding_',dir=parent))
    from training.local_resources import training_gpu_plan,runtime_cache
    cards,skipped,selection=training_gpu_plan(a.gpus,required=2);cache,note=runtime_cache(ROOT)
    if note:print(note,flush=True)
    if (a.resume or a.evaluate) and [d['index'] for d in cfg['devices']]!=[d['index'] for d in cards]:
        raise ValueError('Resume/evaluation must keep the same physical GPU routes')
    cfg.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection,port=a.port)
    if a.resume:
        completed=[n for n in (24,48) if (folder/'checkpoints'/f'step_{n:04d}'/'checkpoint_complete.json').exists()]
        if not completed:raise ValueError('No complete checkpoint; restart a new run')
        cfg['start_step']=max(completed)
        from training.decision_ablation import check_fingerprint
        cp=folder/'checkpoints'/f'step_{cfg["start_step"]:04d}'
        check_fingerprint(cp,json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints'])
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (folder/'batch_plan.json').write_text((Path(cfg['data_dir'])/'batch_plan.json').read_text())
    print(f'Run:{folder}; GPU{[d["index"] for d in cards]}; parentstep84 + checkpoint24/48 evaluation90each (270calls); no market or deployment.',flush=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(d['index'] for d in cards),'PYTHONUNBUFFERED':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
        'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    began=time.monotonic()
    if not a.evaluate:
        if not a.resume:stage(folder,cfg,'preflight',env)
        if not (folder/'evaluation_parent.json').exists():stage(folder,cfg,'eval_parent',env,['parent'])
        if not (folder/'training_finished.json').exists():stage(folder,cfg,'train',env)
    elif not (folder/'evaluation_parent.json').exists():
        raise ValueError('Missing frozen parent baseline; do not compare unpaired candidates')
    labels=[str(n) for n in (24,48) if (folder/'checkpoints'/f'step_{n:04d}'/'checkpoint_complete.json').exists() and not (folder/f'evaluation_{n}.json').exists()]
    if labels:stage(folder,cfg,'eval_candidates',env,labels)
    from training.buyer_binding_eval import write_review
    result=write_review(folder)
    (folder/'finished.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-began,
        complete=result['complete'],development_candidate=result['development_candidate'],automatic_promotion=False),indent=2)+'\n')
    print(f'Completed.Read {folder}/binding_review.md; candidate={result["development_candidate"]}. No automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':main()
