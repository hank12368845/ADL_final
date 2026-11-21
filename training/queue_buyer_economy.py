#!/usr/bin/env python3
"""User-launched VRAM wait for the official step10 Phase2 economy."""
import argparse
import json
import math
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
from training.queue_seller_personas import wait_ready
from training.evaluate_step10_economy import prepare
from training.diagnose_step84_market import check_sources
from training.gpu_startup_errors import startup_memory_races


def finished(folder):
    runs=list(Path(folder).glob('run_economy_*'))
    if len(runs)>1:raise ValueError('More than one economy run')
    if not runs:return False
    m=json.loads((runs[0]/'metadata.json').read_text())
    if m.get('mock_llm'):raise ValueError('Cannot resume a mock as real evaluation')
    return m.get('status')=='finished'


def launch(folder,selected,port,emit):
    cmd=[sys.executable,'-u',str(ROOT/'training/evaluate_step10_economy.py'),
         '--resume',str(folder),'--gpus',','.join(selected),'--port',str(port)]
    emit('Launching:'+ ' '.join(cmd));insufficient=False;started=time.time()
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1,start_new_session=True)
    try:
        for line in proc.stdout:
            emit(line.rstrip('\n'))
            if 'No GPU has enough free memory after reserve' in line:insufficient=True
        code=proc.wait()
        races=startup_memory_races(folder,started) if code and not insufficient else []
        if races:
            emit('vLLM startup VRAM race detected before market actions; requeue.')
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
    from training.local_resources import output_parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1');p.add_argument('--max-gpus',type=int,choices=[1,2],default=2)
    p.add_argument('--port',type=int,default=8100);p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--stable-checks',type=int,default=2);p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int,default=3);p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();allowed=[s.strip() for s in a.gpus.split(',')]
    if not allowed or any(not s.isdigit() for s in allowed) or len(set(allowed))!=len(allowed):p.error('Use distinct GPU indices')
    if not math.isfinite(a.poll_seconds) or not 1<=a.poll_seconds<=60 or a.stable_checks<1:p.error('Invalid polling settings')
    if not 1<=a.port<=65535-a.max_gpus+1:p.error('Invalid port range')
    if a.resume:
        folder=a.resume.resolve();m=json.loads((folder/'suite.json').read_text());check_sources(m)
    else:
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=10.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_economy_queue_',dir=parent))
        m=prepare(folder,a.seed)
    print(f'Buyer queue:{folder}; fixed step10,15 markets/6750actions,allowed GPUs={allowed}',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query/wait/service.');return folder
    finished(folder) # reject mock before requesting GPUs
    with (folder/'queue.log').open('a') as log:
        def emit(line):print(line,flush=True);log.write(line+'\n');log.flush()
        while True:
            selected=allowed[:1] if finished(folder) else wait_ready(allowed,a.max_gpus,a.port,
                a.poll_seconds,a.stable_checks,emit=emit)
            if shutil.disk_usage(folder).free<2*1024**3:raise RuntimeError('Insufficient result disk space')
            with (folder/'queue_attempts.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),selected_gpus=selected,port=a.port))+'\n')
            code,insufficient=launch(folder,selected,a.port,emit)
            if code==0:return folder
            if code==130:raise KeyboardInterrupt
            if not insufficient:raise RuntimeError(f'Buyer evaluation exit{code}; inspect {folder}/queue.log')
            emit('VRAM changed before launch; requeue and keep completed official markets.')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nBuyer queue stopped; resume the printed folder.',flush=True);raise SystemExit(130)
