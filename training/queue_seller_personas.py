#!/usr/bin/env python3
"""Wait for enough free VRAM, then run the frozen seller persona comparison.

This is polling, not a cluster reservation. Other jobs are never stopped.
Only the user starts this script; CPU tests replace all GPU queries and launches.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.local_resources import query_gpus,select_gpus,output_parent
from training.basic_checks import duration
from training.compare_seller_personas import prepare_pair
from training.gpu_startup_errors import startup_memory_races


def port_free(port):
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:sock.bind(('127.0.0.1',port))
        return True
    except OSError:return False


def wait_ready(allowed,max_gpus,port,poll=15,stable_checks=2,max_wait_minutes=0,
               emit=print,query=None,pause=None,clock=None,port_check=None):
    query=query or query_gpus;pause=pause or time.sleep;clock=clock or time.monotonic
    port_check=port_check or port_free
    began=clock();stable=0;previous=None;checks=0
    while True:
        checks+=1;rows=query();available={r['index']:r for r in rows}
        missing=set(allowed)-set(available)
        if missing:raise ValueError(f'Unknown physical GPU indices:{sorted(missing)}')
        ready=[];status=[]
        for index in allowed:
            row=available[index]
            try:
                devices,_=select_gpus([row],index,reserve_gib=2.,min_budget_gib=14.)
                ready.extend(devices);status.append(f'GPU{index} free={row["free_gib"]:.2f}GiB ready')
            except RuntimeError:
                status.append(f'GPU{index} free={row["free_gib"]:.2f}GiB busy')
        # Select the freest feasible card first. Do not wait for a second card.
        ready=sorted(ready,key=lambda r:(-r['free_gib'],r['index']))[:max_gpus]
        selected=tuple(sorted(r['index'] for r in ready))
        ports_ready=bool(selected) and all(port_check(port+i) for i in range(len(selected)))
        if selected and ports_ready:
            stable=stable+1 if selected==previous else 1
        else:stable=0
        previous=selected if ports_ready else None
        elapsed=clock()-began
        emit(f'[QUEUE {checks}] {"; ".join(status)} | selected={",".join(selected) or "none"} '
             f'| stable={stable}/{stable_checks} | ports={"ready" if ports_ready else "waiting"} '
             f'| waited {duration(elapsed)} | queue ETA unknown; next check {poll:g}s')
        if stable>=stable_checks:return list(selected)
        if max_wait_minutes and elapsed>=max_wait_minutes*60:raise TimeoutError('GPU queue wait limit reached')
        pause(poll)


def complete(folder):
    plan=json.loads((folder/'compare_plan.json').read_text())
    for arm in plan['arms'].values():
        p=Path(arm)/'wrapper_result.json'
        if not p.exists():return False
        r=json.loads(p.read_text())
        if r.get('exit')!=0 or r.get('actions')!=450:return False
    return True


def launch(folder,selected,port,emit):
    command=[sys.executable,'-u',str(ROOT/'training/compare_seller_personas.py'),
             '--resume',str(folder),'--gpus',','.join(selected),'--port',str(port)]
    emit('Launching:'+ ' '.join(command));started=time.time()
    proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1,start_new_session=True)
    insufficient=False
    try:
        for line in proc.stdout:
            emit(line.rstrip('\n'))
            if 'No GPU has enough free memory after reserve' in line:insufficient=True
        code=proc.wait()
        races=startup_memory_races(folder,started) if code and not insufficient else []
        if races:
            emit('vLLM startup VRAM race detected; no market actions were taken. Return to queue.')
            with (Path(folder)/'queue_resource_failures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(),evidence=races))+'\n')
        return code,insufficient or bool(races)
    except BaseException:
        os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=45)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:proc.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1',help='Allowed physical indices; default0,1')
    p.add_argument('--max-gpus',type=int,choices=[1,2],default=2,help='Use two when both fit; otherwise start one')
    p.add_argument('--port',type=int,default=8100)
    p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--stable-checks',type=int,default=2)
    p.add_argument('--max-wait-minutes',type=float,default=0,help='0 waits indefinitely')
    p.add_argument('--seed',type=int,default=4);p.add_argument('--resume',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();allowed=[s.strip() for s in a.gpus.split(',')]
    if not allowed or any(not x.isdigit() for x in allowed) or len(set(allowed))!=len(allowed):p.error('Use distinct physical GPU indices')
    if not math.isfinite(a.poll_seconds) or not 1<=a.poll_seconds<=60:p.error('poll-seconds must be1..60')
    if a.stable_checks<1 or not math.isfinite(a.max_wait_minutes) or a.max_wait_minutes<0:p.error('Invalid wait limits')
    if not 1<=a.port<=65535-a.max_gpus+1:p.error('Invalid port range')
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'compare_plan.json').read_text())
        if plan.get('mock'):raise ValueError('This queue only launches real models; cannot resume a mock')
    else:
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_personas_',dir=parent))
        print('CPU: freezing v1/v2 and preparing the full-market comparison before waiting...',flush=True)
        prepare_pair(folder,a.seed)
    print(f'Queue output:{folder}; allowed={allowed}, maxGPUs={a.max_gpus}; no GPU reservation.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query, wait or service.');return folder
    with (folder/'queue.log').open('a') as log:
        def emit(line):
            print(line,flush=True);log.write(line+'\n');log.flush()
        while True:
            selected=allowed[:1] if complete(folder) else wait_ready(allowed,a.max_gpus,a.port,
                a.poll_seconds,a.stable_checks,a.max_wait_minutes,emit=emit)
            if shutil.disk_usage(folder).free<5*1024**3:raise RuntimeError('Less than5GiB free for results; launch cancelled')
            with (folder/'queue_attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),selected_gpus=selected,port=a.port))+'\n')
            code,insufficient=launch(folder,selected,a.port,emit)
            if code==0:return folder
            if not insufficient:raise RuntimeError(f'Comparison exit{code}; inspect {folder}/queue.log and arm runner logs')
            emit('VRAM changed before a model could start. Requeue; completed arms are preserved.')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nQueue stopped. Resume the printed output folder to continue.',flush=True);raise SystemExit(130)
