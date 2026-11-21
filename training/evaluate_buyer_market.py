#!/usr/bin/env python3
"""Public full-market comparison of the stable buyer checkpoint and plain base ON."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT_CANDIDATE=ROOT/'training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032'
ARMS=('candidate','base_on')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(directory,seed=0,rotate=True):
    import yaml
    from adlmarket.casting import plan_markets
    from adlmarket.setups import market_run,deep_merge
    from training.decision_ablation import check_fingerprint,fingerprint
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    candidate=DEFAULT_CANDIDATE
    cfg=json.loads((candidate.parent.parent/'train_config.json').read_text())
    reference=json.loads((ROOT/'training/data/buyer_symmetric_repair_v1/manifest.json').read_text())
    if Path(reference['parent_adapter'])!=candidate:raise ValueError('Archived reference points at another candidate')
    check_fingerprint(candidate,reference['parent_fingerprints'])
    if json.loads((candidate/'training_step.json').read_text())!={'step':32,'thinking':True}:raise ValueError('Unexpected candidate step/mode')
    source=ROOT/cfg['data_dir'];data=json.loads((source/'manifest.json').read_text())
    if sha(source/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Candidate training data changed')
    personas=[(candidate.parent.parent/'personas'/f'persona_{p}.md').read_text() for p in (1,2,3)]
    for i,text in enumerate(personas,1):
        if text!=(source/'personas'/f'persona_{i}.md').read_text() or len(text)>8000:raise ValueError('Candidate personas changed or overlength')
    economy=yaml.safe_load((ROOT/'configs/economy.yaml').read_text())
    market=deep_merge(economy['market'],economy['phase2']['market'])
    common=yaml.safe_load((ROOT/'configs/buyer_scored_short.yaml').read_text())
    common.update(name='buyer_full_public_review',seed=seed,market=market,rotate=rotate,repeats=1,
        save_trajectories=True,save_prompts=True,parallel_markets=1)
    common['models']=str(ROOT/'configs/models.yaml')
    configs={};plans={};snapshots={}
    for label in ARMS:
        folder=directory/label/'buyer';folder.mkdir(parents=True)
        for i,text in enumerate(personas,1):(folder/f'persona_{i}.md').write_text(text)
        agent={'id':'example_student_buyer','owner':'example_student','role':'buyer','model':'base',
               'reasoning':True,'lora':str(candidate) if label=='candidate' else None}
        (folder/'agent.yaml').write_text(yaml.safe_dump(agent,sort_keys=False))
        run_cfg=copy.deepcopy(common);run_cfg['buyers'][0]=str(folder)
        path=directory/label/'market.yaml';path.write_text(yaml.safe_dump(run_cfg,sort_keys=False));configs[label]=str(path)
        loaded=market_run(path,root=ROOT)
        if len(loaded.sellers)!=5 or len(loaded.buyers)!=10:raise ValueError('Incorrect market size')
        ours=[a for a in loaded.buyers if a.owner!='ta']
        if {a.id for a in ours}!={f'example_student_buyer_p{p}' for p in (1,2,3)}:raise ValueError('Persona expansion mismatch')
        for i,a in enumerate(ours):
            if not a.reasoning or a.persona_body()!=personas[i].strip():raise ValueError('Wrong persona or reasoning mode')
            spec=loaded.models[a.model]
            if label=='candidate' and (not spec.lora or Path(spec.lora.path)!=candidate):raise ValueError('Candidate adapter not resolved')
            if label=='base_on' and spec.lora is not None:raise ValueError('Base control must have no adapter')
        seats=plan_markets(loaded.config.casting,loaded.sellers,loaded.buyers,seed,pads=loaded.pads)
        plans[label]=[{'market':p.market_id,'seed_id':p.seed_id,'rotation':p.rotation,
                       'sellers':[a.id for a in p.sellers],'buyers':[a.id for a in p.buyers]} for p in seats]
        snapshots[label]=[(a.id,a.role,a.owner,a.reasoning,a.persona_body(),
                           fingerprint(loaded.models[a.model].lora.path) if loaded.models[a.model].lora else None)
                          for a in loaded.everyone()]
    if plans['candidate']!=plans['base_on']:raise ValueError('Paired market seats/seeds differ')
    # Only the three target buyers' adapters may differ; opponents and all personas are identical.
    for a,b in zip(snapshots['candidate'],snapshots['base_on']):
        if a[:5]!=b[:5] or a[2]=='ta' and a[5]!=b[5]:raise ValueError('Paired opponent/persona changed')
    markets=5 if rotate else 1
    if any(len(p)!=markets for p in plans.values()):raise ValueError('Missing rotations')
    manifest={'experiment':'buyer_public_market_review_v1','candidate':str(candidate),'candidate_fingerprints':fingerprint(candidate),
        'thinking':True,'seed':seed,'rotate':rotate,'markets_per_arm':markets,'total_markets':2*markets,
        'actions_per_arm':450*markets,'total_actions':900*markets,'configs':configs,'plans':plans,
        'persona_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in personas],
        'sources':{name:sha(ROOT/name) for name in ('configs/models.yaml','configs/economy.yaml','configs/buyer_scored_short.yaml',
                  'training/evaluate_buyer_market.py','training/buyer_market_report.py','training/basic_checks.py','training/market_load_balance.py')},
        'inputs_sha256':{str(p):sha(p) for label in ARMS for p in (directory/label).rglob('*') if p.is_file()},
        'description':f'Public items; toy TA opponents; 5 sellers x 10 buyers; 5 rounds x 6 turns; {markets} stock rotation(s) per model. Same three training personas, seed, seats, opponents and rules; candidate LoRA versus plain base ON.',
        'status':'prepared_no_gpu_work'}
    (directory/'suite.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (directory/'state.json').write_text(json.dumps({'completed':{},'status':'prepared'},indent=2)+'\n')
    return manifest


def progress_text(arm_index,count,manifest,elapsed):
    done=arm_index*manifest['actions_per_arm']+count;total=manifest['total_actions']
    speed=done/elapsed if done and elapsed>0 else 0
    eta=time.strftime('%H:%M:%S',time.gmtime(max(0,int((total-done)/speed)))) if speed else 'estimating'
    elapsed_text=time.strftime('%H:%M:%S',time.gmtime(elapsed))
    finished_markets=done//450
    return f'[Overall] {done}/{total} actions | {finished_markets}/{manifest["total_markets"]} markets | elapsed {elapsed_text} | {speed:.2f} actions/s | ETA {eta}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--single-market',action='store_true',help='One full 30-turn market per model; default includes all five stock rotations')
    p.add_argument('--prepare-only',action='store_true',help='CPU configuration/adapter checks only; no GPU query or services')
    p.add_argument('--mock',action='store_true',help='CPU pipeline check only; scores are not model performance')
    p.add_argument('--output-dir',type=Path);p.add_argument('--resume',type=Path,help='Keep completed arms; rerun an incomplete arm from its start')
    p.add_argument('--report',type=Path,help='CPU report for existing completed arms')
    a=p.parse_args()
    from training.buyer_market_report import write_report
    if a.report:write_report(a.report.resolve());return
    if a.resume:
        directory=a.resume.resolve();manifest=json.loads((directory/'suite.json').read_text())
        if manifest['experiment']!='buyer_public_market_review_v1':raise ValueError('Wrong resume suite')
        for name,expected in manifest['sources'].items():
            if sha(ROOT/name)!=expected:raise ValueError(f'Resume source changed: {name}')
        from training.decision_ablation import check_fingerprint
        check_fingerprint(manifest['candidate'],manifest['candidate_fingerprints'])
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5)
        if note:print(note,flush=True)
        directory=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_market_review_',dir=parent))
        print('CPU: freeze candidate, personas, rules, opponents and paired stock rotations...',flush=True)
        manifest=prepare(directory,a.seed,not a.single_market)
    print(manifest['description'],flush=True)
    print(f'Suite: {directory}; {manifest["total_markets"]} full markets, {manifest["total_actions"]} actions.',flush=True)
    if a.prepare_only:
        print('PASS: CPU preparation; no GPU query, inference or service started.',flush=True);return directory
    for name,expected in manifest['inputs_sha256'].items():
        if sha(name)!=expected:raise ValueError(f'Frozen suite input changed: {name}')
    state=json.loads((directory/'state.json').read_text());previous_elapsed=state.get('elapsed_seconds',0.);started=time.monotonic();state['status']='running'
    # Save how this user invocation was launched, separate from immutable prepared inputs.
    (directory/'launch.json').write_text(json.dumps({'gpus':a.gpus,'port':a.port,'mock':a.mock,'mode':'mock' if a.mock else 'gpu'},indent=2))
    (directory/'state.json').write_text(json.dumps(state,indent=2))
    print('Loading / warm-up ETA estimating. Both per-market and overall progress will follow.',flush=True)
    with (directory/'suite.log').open('a') as log:
        for index,label in enumerate(ARMS):
            if label in state['completed']:
                meta=json.loads((Path(state['completed'][label]['run_dir'])/'metadata.json').read_text())
                if meta.get('mock_llm')!=a.mock:raise ValueError('Cannot mix mock and real arms')
                print(f'Skip completed arm: {label}',flush=True);continue
            print(f'Arm {index+1}/2: {label}',flush=True)
            parent=directory/label/'output';parent.mkdir(exist_ok=True)
            command=[sys.executable,'-u',str(ROOT/'training/basic_checks.py'),'run','--mode','mock' if a.mock else 'gpu','--role','buyer',
                '--config',manifest['configs'][label],'--gpus',a.gpus,'--port',str(a.port),'--output-dir',str(parent)]
            env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    match=re.match(r'^\[[#-]+\] (\d+)/(\d+)',line)
                    if match:
                        if int(match[2])!=manifest['actions_per_arm']:raise ValueError('Child action horizon differs from frozen suite')
                        status=progress_text(index,int(match[1]),manifest,previous_elapsed+time.monotonic()-started)
                        print(status,flush=True);log.write(status+'\n');log.flush()
                code=process.wait()
            except BaseException:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                state.update(status='interrupted',elapsed_seconds=previous_elapsed+time.monotonic()-started);(directory/'state.json').write_text(json.dumps(state,indent=2));raise
            if code:
                state.update(status='failed',failed_arm=label,exit=code,elapsed_seconds=previous_elapsed+time.monotonic()-started);(directory/'state.json').write_text(json.dumps(state,indent=2))
                raise SystemExit(f'Arm {label} failed ({code}); inspect {parent} and {directory}/suite.log')
            runs=sorted(parent.glob('*/run_market_*'))
            if not runs:raise ValueError('Child produced no run directory')
            run=runs[-1];meta=json.loads((run/'metadata.json').read_text())
            if meta.get('status')!='finished' or meta.get('mock_llm')!=a.mock:raise ValueError('Invalid completed arm')
            state['completed'][label]={'run_dir':str(run),'config_sha256':sha(manifest['configs'][label])}
            state['elapsed_seconds']=previous_elapsed+time.monotonic()-started
            (directory/'state.json').write_text(json.dumps(state,indent=2));write_report(directory)
    state.update(status='complete',elapsed_seconds=previous_elapsed+time.monotonic()-started)
    (directory/'state.json').write_text(json.dumps(state,indent=2));write_report(directory)
    print(f'Completed. Read first: {directory}/market_review.md',flush=True)
    return directory


if __name__=='__main__':main()
