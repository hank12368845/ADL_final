#!/usr/bin/env python3
"""Sequential user-launched seller+buyer queues, with persistent overnight status."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.diagnose_step84_market import check_sources
from training.local_resources import output_parent

SELLER=ROOT/'results/basic_checks/20261121-005859_seller_personas_1f4wxek7'
BUYER=ROOT/'results/basic_checks/20261121-000742_step10_economy_a2txwbqx'


def validate_inputs(seller,buyer):
    p=json.loads((seller/'compare_plan.json').read_text())
    if p.get('mock') or p.get('serving_mode')=='tensor_parallel_2':raise ValueError('Seller must be the real replica experiment')
    for arm in p['arms'].values():check_sources(json.loads((Path(arm)/'suite.json').read_text()))
    m=json.loads((buyer/'suite.json').read_text());check_sources(m)
    if m.get('phase')!=2 or m.get('total_markets')!=15 or m.get('total_actions')!=6750:
        raise ValueError('Buyer is not the full step10 Phase2 economy')
    for meta in buyer.glob('run_economy_*/metadata.json'):
        if json.loads(meta.read_text()).get('mock_llm'):raise ValueError('Buyer cannot be mock')


def run_job(command,emit):
    proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1,start_new_session=True)
    try:
        for line in proc.stdout:emit(line.rstrip('\n'))
        return proc.wait()
    except BaseException:
        os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=60)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:proc.stdout.close()


def execute_plan(folder,plan,emit):
    result_file=folder/'overnight_status.json'
    state=json.loads(result_file.read_text()) if result_file.exists() else dict(jobs={})
    for job in plan['jobs']:
        name=job['name'];previous=state['jobs'].get(name,{})
        if previous.get('status')=='success':emit(f'Skip already successful {name}');continue
        state['jobs'][name]=dict(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),output=job['output'])
        result_file.write_text(json.dumps(state,indent=2)+'\n')
        emit(f'=== {name}: queued/running; output:{job["output"]} ===')
        try:code=run_job(job['command'],emit)
        except KeyboardInterrupt:
            state['jobs'][name].update(status='interrupted',exit=130)
            result_file.write_text(json.dumps(state,indent=2)+'\n');raise
        state['jobs'][name].update(status='success' if code==0 else 'interrupted' if code==130 else 'failed',exit=code,
                                  finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        result_file.write_text(json.dumps(state,indent=2)+'\n')
        if code==130:raise KeyboardInterrupt
        if code:emit(f'{name} failed(exit{code}); recorded. Continue the independent next evaluation.')
    lines=['# Overnight evaluation status','', '| Task | Status | Exit | Result folder |','|---|---|---:|---|']
    for job in plan['jobs']:
        s=state['jobs'][job['name']];lines.append(f'| {job["name"]} | {s["status"]} | {s["exit"]} | {job["output"]} |')
    lines+=['','Seller: persona_comparison.md; Buyer: economy_review.md and official run_economy standings.',
            'Waiting for other GPU jobs has no predictable ETA. No training, deployment or automatic model promotion.']
    (folder/'overnight_review.md').write_text('\n'.join(lines)+'\n')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seller-run',type=Path,default=SELLER);p.add_argument('--buyer-run',type=Path,default=BUYER)
    p.add_argument('--gpus',default='0,1');p.add_argument('--max-gpus',type=int,choices=[1,2],default=2)
    p.add_argument('--port',type=int,default=8100);p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'overnight_plan.json').read_text())
        validate_inputs(Path(plan['seller_run']),Path(plan['buyer_run']))
    else:
        seller=a.seller_run.resolve();buyer=a.buyer_run.resolve();validate_inputs(seller,buyer)
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=10.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_overnight_',dir=parent))
        jobs=[]
        for name,script,output in [('seller_personas','queue_seller_personas.py',seller),('buyer_economy','queue_buyer_economy.py',buyer)]:
            jobs.append(dict(name=name,output=str(output),command=[sys.executable,'-u',str(ROOT/'training'/script),
                '--resume',str(output),'--gpus',a.gpus,'--max-gpus',str(a.max_gpus),'--port',str(a.port)]))
        plan=dict(seller_run=str(seller),buyer_run=str(buyer),jobs=jobs,
            total_actions=7650,training_updates=0,automatic_promotion=False)
        (folder/'overnight_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(f'Overnight plan:{folder}; Seller900actions -> Buyer6750actions; sequential,not competing.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query/wait/job launch.');return folder
    lock_path=ROOT/'.cache/overnight_evaluations.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another overnight controller is running; do not start a second one')
        with (folder/'overnight.log').open('a') as log:
            def emit(line):print(line,flush=True);log.write(line+'\n');log.flush()
            state=execute_plan(folder,plan,emit)
    failed=any(s['status']!='success' for s in state['jobs'].values())
    print(f'Overnight finished; see {folder}/overnight_review.md',flush=True)
    if failed:raise SystemExit(1)
    return folder


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nOvernight controller stopped; completed markets remain saved.',flush=True);raise SystemExit(130)
