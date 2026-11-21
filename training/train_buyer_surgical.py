#!/usr/bin/env python3
"""User-launched two-GPU step84 surgical continuation: preflight, baseline,20steps,automatic evaluation."""
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
from training.prepare_buyer_surgical import DATA,PARENT,audit_folder,sha


def prepare(data=DATA):
    from importlib.metadata import version
    m=audit_folder(data)
    from training.buyer_surgical_eval import validate_wire
    cases=json.loads((Path(data)/'evaluation_cases.json').read_text());validate_wire(cases)
    recipe=json.loads((Path(data)/'recipe.json').read_text())
    expected=dict(max_steps=20,effective_batch_size=4,learning_rate=5e-6,warmup_steps=2,checkpoint_steps=[10,20],world_size=2,rank=16,alpha=32,thinking=True)
    if any(recipe[k]!=v for k,v in expected.items()):raise ValueError('Frozen surgical recipe differs')
    parent_cfg=json.loads((PARENT.parent.parent/'train_config.json').read_text())
    sources=[ROOT/f'training/{name}.py' for name in ('train_buyer_surgical','buyer_surgical_train','buyer_surgical_eval','prepare_buyer_surgical',
        'buyer_binding_eval','buyer_binding_data','buyer_operations_train','reasoning_checkpoint','official_buyer_train','chat_payload','local_resources','decision_ablation')]
    sources+=list((ROOT/'adlmarket').glob('*.py'))+[ROOT/'configs/models.yaml',ROOT/'configs/basic_buyer.yaml']
    return dict(experiment='buyer_surgical_v1',data_dir=str(Path(data).resolve()),data_manifest_sha256=sha(Path(data)/'manifest.json'),
        base_model='Qwen/Qwen3.5-4B',revision=m['model_revision'],parent_adapter=str(PARENT),parent_fingerprints=m['parent_fingerprints'],
        **expected,dropout=parent_cfg['dropout'],target_modules=parent_cfg['target_modules'],seed=m['seed'],max_length=recipe['max_length'],
        activation_offload_from_tokens=16000,mlp_chunk_tokens=2048,objective='Ordinary native completion CE; faithful replay56/repair24; no thinking format or cap',
        sources={str(p.resolve()):sha(p) for p in sources},versions={name:version(name) for name in ('torch','transformers','peft','safetensors','vllm')})


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
    worker_env=dict(env)
    if kind in ('preflight','train'):worker_env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    print(f'Stage {kind}: loading / warm-up ETA estimating. Log:{folder}/{kind}.log',flush=True)
    began=time.monotonic();offsets={};last=0
    with (folder/f'{kind}.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
            from training.buyer_surgical_train import worker
            return worker(a.run,a.worker=='preflight')
        from training.buyer_surgical_eval import evaluate
        return evaluate(a.run,a.labels.split(','))
    if a.resume or a.evaluate:
        from training.buyer_surgical_train import check_run
        folder=(a.resume or a.evaluate).resolve();cfg=check_run(folder)
    else:
        print('CPU: native-engine replay, dataset integrity, original personas and split audit; ETA estimating.',flush=True)
        cfg=prepare(a.data_dir);m=json.loads((Path(cfg['data_dir'])/'manifest.json').read_text())
        print(f'CPU ready:80exposures/20steps,batch4,replay56+repair24; max{m["max_input_target_tokens"]} full input+target tokens; original thinking preserved.',flush=True)
        if a.cpu_check:print('PASS CPU only; no GPU query, services or model execution.',flush=True);return cfg
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_surgical_',dir=parent))
    from training.local_resources import training_gpu_plan,runtime_cache
    cards,skipped,selection=training_gpu_plan(a.gpus,required=2);cache,note=runtime_cache(ROOT)
    if note:print(note,flush=True)
    if (a.resume or a.evaluate) and [d['index'] for d in cfg['devices']]!=[d['index'] for d in cards]:
        raise ValueError('Resume/evaluation must keep the same physical GPU routes')
    cfg.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection,port=a.port)
    if a.resume:
        completed=[n for n in (10,20) if (folder/'checkpoints'/f'step_{n:04d}'/'checkpoint_complete.json').exists()]
        if not completed:raise ValueError('No complete checkpoint; restart a new run')
        cfg['start_step']=max(completed)
        from training.decision_ablation import check_fingerprint
        cp=folder/'checkpoints'/f'step_{cfg["start_step"]:04d}'
        check_fingerprint(cp,json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints'])
    (folder/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (folder/'batch_plan.json').write_text((Path(cfg['data_dir'])/'batch_plan.json').read_text())
    print(f'Run:{folder}; GPU{[d["index"] for d in cards]}; parentstep84 + checkpoint10/20 evaluation102each (306calls); no market or deployment.',flush=True)
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
    labels=[str(n) for n in (10,20) if (folder/'checkpoints'/f'step_{n:04d}'/'checkpoint_complete.json').exists() and not (folder/f'evaluation_{n}.json').exists()]
    if labels:stage(folder,cfg,'eval_candidates',env,labels)
    from training.buyer_surgical_eval import write_review
    result=write_review(folder)
    (folder/'finished.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-began,
        complete=result['complete'],development_candidate=result['development_candidate'],automatic_promotion=False),indent=2)+'\n')
    print(f'Completed.Read {folder}/surgical_review.md; candidate={result["development_candidate"]}. No automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':main()
