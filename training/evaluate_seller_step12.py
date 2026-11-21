#!/usr/bin/env python3
"""One additional step12 market; reuse completed, verified starter/step46 controls."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter

import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import compare_seller_models as paired
from training import compare_seller_personas as legacy
from training.diagnose_step84_market import check_sources,sha
from training.decision_ablation import fingerprint,check_fingerprint
from training.seller_market_observations import analyze

CONTROL=ROOT/'results/basic_checks/20261121-151510_seller_models_87fz7bkq'
CANDIDATE=ROOT/'training/runs/20261121-135008_seller_official_9d95_ubx/checkpoints/step_0012'


def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def cache_arm(arm):
    from adlmarket.setups import market_run
    arm=Path(arm).resolve();m=json.loads((arm/'suite.json').read_text());check_sources(m)
    w=json.loads((arm/'wrapper_result.json').read_text())
    if w['exit'] or w['actions']!=450:raise ValueError('Cached control incomplete')
    run=Path(w['run_dir']);loaded=market_run(run/'config.yaml',root=ROOT);paired.validate(loaded,m)
    if paired.initial_signature(loaded)!=m['initial_signature']:raise ValueError('Cached initial state differs')
    for path,h in m['runtime_inputs'].items():
        if sha(path)!=h:raise ValueError('Cached runtime config changed')
    review=analyze(run)
    paths=[arm/'suite.json',arm/'wrapper_result.json',run/'config.yaml',run/'metadata.json']
    paths += [p for market in (run/'markets').iterdir() for p in (market/'events.jsonl',market/'final.json',market/'scores.json')]
    return dict(arm=str(arm),initial_signature=m['initial_signature'],mock=review['mock'],
                files={str(p):sha(p) for p in paths}),review,loaded


def prepare(folder,control=CONTROL,mock=False):
    from adlmarket.setups import market_run
    from adlmarket.config import check_adapter
    from training.train_official_seller import check_run
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    original=json.loads((Path(control)/'compare_plan.json').read_text());cached={};loaded={}
    if original['experiment']!='seller_starter_step46_v1':raise ValueError('Expected completed starter/step46 control')
    for label in ('starter','step46'):
        cached[label],review,loaded[label]=cache_arm(original['arms'][label])
        if review['mock']!=mock:raise ValueError('Cannot mix real and mock controls/candidate')
    if cached['starter']['initial_signature']!=cached['step46']['initial_signature']:raise ValueError('Cached worlds differ')
    cfg=check_run(CANDIDATE.parent.parent);complete=json.loads((CANDIDATE/'checkpoint_complete.json').read_text())
    if complete['step']!=12 or json.loads((CANDIDATE/'training_step.json').read_text())!=dict(step=12,thinking=True):raise ValueError('Wrong candidate')
    check_fingerprint(CANDIDATE,complete['adapter_fingerprints'])
    check_adapter('seller_step12',CANDIDATE,cfg['base_model'],{'model.layers.':'model.language_model.layers.'})
    arm=folder/'step12';m=legacy.prepare_arm(arm,'v2',original['seed'])
    path=arm/'seller/agent.yaml';a=yaml.safe_load(path.read_text());a['lora']=str(CANDIDATE);path.write_text(yaml.safe_dump(a,sort_keys=False))
    source_cfg=yaml.safe_load(Path(m['config']).read_text());source_cfg['name']='seller_step12_cached_controls'
    Path(m['config']).write_text(yaml.safe_dump(source_cfg,sort_keys=False));new=market_run(m['config'],root=ROOT)
    m.update(experiment='seller_step12_cached_controls_v1',arm='step12',adapter=str(CANDIDATE),adapter_fingerprints=complete['adapter_fingerprints'],
             market_rules=new.config.market.model_dump(mode='json'),initial_signature=paired.initial_signature(new))
    paths=[Path(__file__),ROOT/'training/compare_seller_models.py',ROOT/'training/seller_market_observations.py',ROOT/'training/queue_seller_personas.py',ROOT/'training/gpu_startup_errors.py',
           ROOT/'training/train_official_seller.py',ROOT/'training/official_seller_train.py',ROOT/'training/official_seller_data.py',
           CANDIDATE/'checkpoint_complete.json',CANDIDATE.parent.parent/'train_config.json']
    m['sources'].update({str(p.resolve()):sha(p) for p in paths})
    m['inputs']={str(p):sha(p) for p in arm.rglob('*') if p.is_file() and p.name!='suite.json'}
    m['adapters']={str(Path(new.models[a.model].lora.path).resolve()):fingerprint(new.models[a.model].lora.path)
                   for a in new.everyone() if new.models[a.model].lora}
    paired.validate(new,m);check_sources(m)
    if m['initial_signature']!=original['initial_signature'] or paired.opponent_signature(new)!=paired.opponent_signature(loaded['starter']):
        raise ValueError('New step12 setup does not match cached controls')
    dump(arm/'suite.json',m)
    plan=dict(experiment=m['experiment'],seed=original['seed'],cached=cached,arm=str(arm),initial_signature=m['initial_signature'],
              new_actions=450,cached_actions=900,total_reviewed_actions=1350,mock=mock,automatic_promotion=False,training_updates=0)
    dump(folder/'plan.json',plan);return plan


def verify_cached(plan):
    for label,c in plan['cached'].items():
        for path,h in c['files'].items():
            if sha(path)!=h:raise ValueError(f'Cached {label} result changed; do not reuse')


def play(arm,m,effective,cache,mock):
    from training.basic_checks import read_new,duration
    check_sources(m)
    if any(sha(p)!=h for p,h in m['runtime_inputs'].items()):raise ValueError('Runtime config changed')
    cmd=[sys.executable,'-u',str(ROOT/'run_market.py'),'--config',str(effective),'--mock-llm' if mock else '--launch-vllm']
    env={**os.environ,**cache,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and p!=str(ROOT/'.cache/fast_kernels/site-packages'))
    started=time.monotonic();first=None;last=0;offsets={};count=0;targets=Counter();run=None;position='loading/warm-up'
    with (arm/'runner.log').open('w') as log,(arm/'progress.log').open('w') as progress:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(arm/'runner.log',offsets):print(line,flush=True)
                found=list(arm.glob('run_market_*'));run=found[0] if found else None
                if run:
                    for p in run.glob('markets/*/events.jsonl'):
                        for line in read_new(p,offsets):
                            r=json.loads(line)
                            if r.get('type')=='market_start':first=first or time.monotonic()
                            if r.get('type')=='turn':
                                count+=1
                                if r['agent'] in paired.IDS:targets[r['agent']]+=1
                                position=f'r{r["round"]}/5 t{(r["tick"]-1)%6+1}/6'
                    if not count:
                        for p in run.glob('vllm/*.log'):
                            lines=read_new(p,offsets)
                            if lines:print(f'[vLLM {p.name}] {lines[-1][-180:]}',flush=True)
                now=time.monotonic()
                if done or now-last>=5:
                    speed=count/(now-first) if first and now-first>=5 else 0
                    eta=duration((450-count)/speed) if speed else 'estimating'
                    line=(f'step12 [{"#"*(count*24//450)}{"-"*(24-count*24//450)}] NEW {count}/450 | cached900/900 | '
                          f'review {900+count}/1350 | {position} | '+ ' '.join(f'p{i}:{targets[paired.IDS[i-1]]}/30' for i in (1,2,3))+
                          f' | elapsed {duration(now-started)} | {speed:.2f}actions/s | remaining ETA~{eta}')
                    print(line,flush=True);progress.write(line+'\n');progress.flush();last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    result=dict(exit=proc.returncode,run_dir=str(run) if run else None,actions=count,mock=mock,elapsed_seconds=time.monotonic()-started)
    dump(arm/'wrapper_result.json',result)
    if proc.returncode or count!=450:raise RuntimeError(f'step12 incomplete; inspect{arm}/runner.log')


def execute(folder,plan,selection,port,wait,mock):
    from training.queue_seller_personas import wait_ready
    from training.gpu_startup_errors import startup_memory_races
    verify_cached(plan);arm=Path(plan['arm']);m=json.loads((arm/'suite.json').read_text());check_sources(m)
    if paired.completed(arm):print('Skip completed step12:450/450; CPU report only.',flush=True);return
    while True:
        selected=wait_ready(selection.split(','),2,port) if wait and not mock else selection.split(',')
        if shutil.disk_usage(folder).free<5*1024**3:raise RuntimeError('Less than5GiB results space')
        for p in list(arm.glob('run_market_*'))+([arm/'routing_agents'] if (arm/'routing_agents').exists() else []):
            p.rename(p.with_name(f'archived_{time.time_ns()}_{p.name}'))
        started=time.time()
        try:
            effective,cache=paired.configure(arm,m,','.join(selected),port,mock);play(arm,m,effective,cache,mock);return
        except RuntimeError as error:
            races=startup_memory_races(folder,started);preflight='No GPU has enough free memory after reserve' in str(error)
            if not wait or mock or not(races or preflight):raise
            with (folder/'resource_requeues.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),preflight=preflight,evidence=races))+'\n')
            print('Startup VRAM race; return to queue. No completed control is rerun.',flush=True)


def report(folder):
    from adlmarket.setups import market_run
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text());verify_cached(plan);results={}
    for label,c in plan['cached'].items():_,results[label],_=cache_arm(c['arm'])
    arm=Path(plan['arm']);m=json.loads((arm/'suite.json').read_text());check_sources(m)
    if not paired.completed(arm):raise ValueError('step12 not complete')
    w=json.loads((arm/'wrapper_result.json').read_text());run=Path(w['run_dir']);loaded=market_run(run/'config.yaml',root=ROOT);paired.validate(loaded,m)
    if paired.initial_signature(loaded)!=plan['initial_signature']:raise ValueError('Played step12 world differs')
    if any(sha(p)!=h for p,h in m['runtime_inputs'].items()):raise ValueError('Step12 runtime changed')
    results['step12']=analyze(run);dump(arm/'seller_model_review.json',results['step12'])
    if len({r['mock'] for r in results.values()})!=1:raise ValueError('Mixed mock/real results')
    real=all(not r['mock'] and r['infrastructure_clean'] for r in results.values())
    lines=['# Seller存檔選擇：starter／step12／step46','',f'真模型有效比較={real}；固定seed{plan["seed"]}、v2三persona、相同初始世界及對手。',
           'starter／step46沿用已完成且逐筆核帳的兩場；只新增step12一場5×10／30turn，共450新行動。',
           '', '| persona | starter profit | step12 profit | step46 profit | step12−starter | 售出 starter/12/46 | 未售 starter/12/46 |',
           '|---|---:|---:|---:|---:|---|---|']
    for aid in paired.IDS:
        a,b,c=[results[k]['sellers'][aid] for k in ('starter','step12','step46')]
        lines.append(f'| {aid} | {a["profit_cents"]/100:.2f} | {b["profit_cents"]/100:.2f} | {c["profit_cents"]/100:.2f} | {(b["profit_cents"]-a["profit_cents"])/100:+.2f} | {a["units_sold"]}/{b["units_sold"]}/{c["units_sold"]} | {a["units_unsold"]}/{b["units_unsold"]}/{c["units_unsold"]} |')
    lines+=['', '| persona | 單件毛利 starter/12/46 | 同價重掛 starter/12/46 | 截斷 starter/12/46 | budget wait starter/12/46 |', '|---|---|---|---|---|']
    for aid in paired.IDS:
        r=[results[k]['sellers'][aid] for k in ('starter','step12','step46')]
        margins='/'.join('—' if v['mean_unit_margin'] is None else f'{v["mean_unit_margin"]:.2f}' for v in r)
        repeated='/'.join(str(len(v['quote_diagnostics']['unchanged_live_listings'])) for v in r)
        lines.append(f'| {aid} | {margins} | {repeated} | {"/".join(str(v["truncated_turns"]) for v in r)} | {"/".join(str(v["budget_waits"]) for v in r)} |')
    lines+=['', 'Step12較早不代表必定較好；這是既有checkpoint選擇，沒有新訓練。原seed5結果已參與選模，不能再當未見泛化證據。',
            '三persona同場互相影響、只一個seed及stock rotation、public商品／toy對手，不能保證正式排名。每角色只交一個LoRA，不把不同persona的不同checkpoint最高值拼成可提交成績。',
            '無貨set_price／同價重掛是低效動作，與真工具拒絕分開；持續board仍可成交，截斷／budget wait不是必然停止所有銷售，不能直接編造成美元損失。',
            '庫存判定讀實際送給模型的觀測並以交易重建交叉檢查，不直接用可能已被同tick成交改動的event you.stock；JSON保留原event診斷及差異證據。',
            'JSON含各persona逐商品價格／grade實現毛利／round profit、評論、工具錯誤、截斷與GPU負載。不自動部署、換persona、加seed、測step23或續訓。']
    result=dict(complete=True,valid_real_comparison=real,seed=plan['seed'],results=results,new_actions=450,cached_actions=900,
                automatic_promotion=False,training_updates=0)
    dump(folder/'seller_checkpoint_comparison.json',result);(folder/'seller_checkpoint_comparison.md').write_text('\n'.join(lines)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8100);p.add_argument('--wait',action='store_true')
    p.add_argument('--control',type=Path,default=CONTROL);p.add_argument('--output-dir',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',type=Path);p.add_argument('--mock',action='store_true');a=p.parse_args()
    if a.report:return report(a.report.resolve())
    ids=[s.strip() for s in a.gpus.split(',')]
    if not 1<=len(ids)<=2 or len(set(ids))!=len(ids) or any(not s.isdigit() for s in ids):p.error('Use1or2distinctphysicalGPUindices')
    if not 1<=a.port<=65534:p.error('Invalidport')
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'plan.json').read_text())
        if plan['experiment']!='seller_step12_cached_controls_v1' or plan['mock']!=a.mock:raise ValueError('Wrongexperimentormockmode')
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_step12_',dir=parent))
        print('CPU: verify cached900actions; freeze step12 same-world single market. ETA estimating.',flush=True)
        plan=prepare(folder,a.control,a.mock)
    print(f'Output:{folder}; NEW450actions only; cachedstarter/step46 preserved. No training/deployment.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query/wait/model inference.');return folder
    execute(folder,plan,','.join(ids),a.port,a.wait,a.mock);result=report(folder)
    print(f'Completed.Read {folder}/seller_checkpoint_comparison.md; no automatic next GPU work.',flush=True);return result


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; resume printed folder. Cached controls unchanged.',flush=True);raise SystemExit(130)
