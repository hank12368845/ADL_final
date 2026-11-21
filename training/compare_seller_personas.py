#!/usr/bin/env python3
"""User-launched v1/v2 seller persona comparison, same starter and full5x10 markets."""
import argparse
from collections import Counter, defaultdict
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adlmarket.config import load_agent, persona_variants
from adlmarket.setups import market_run, deep_merge
from adlmarket.casting import plan_markets
from training.basic_checks import read_new,duration
from training.diagnose_step84_market import sha,check_sources
from training.decision_ablation import fingerprint
from training.official_buyer_data import MODEL_REVISION
from training.step84_market_report import cents,turn_stats

IDS=[f'example_student_seller_p{i}' for i in [1,2,3]]
ADAPTER=ROOT/'agents/example_student/seller/checkpoint'


def write_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def initial_state(loaded):
    # The buyer pair helper hashes all initial engine facts and agent personas.
    # Exclude only the target seller personas, which are this experiment's variable.
    from training.compare_buyer_markets import initial_signature
    view=loaded.model_copy(deep=True)
    for a in view.sellers:
        if a.owner!='ta':a.persona=None;a.persona_text='';a.personas=None
    return initial_signature(view)


def validate(loaded,m):
    plans=plan_markets(loaded.config.casting,loaded.sellers,loaded.buyers,loaded.config.seed,pads=loaded.pads)
    if len(plans)!=1 or (len(loaded.sellers),len(loaded.buyers))!=(5,10):raise ValueError('Expected one5x10 market')
    r=loaded.config.market.rounds
    if (r.total,r.turns_per_round,r.output_tokens_per_round,loaded.config.market.history_turns)!=(5,6,8192,6):
        raise ValueError('Formal horizon/history/token limits changed')
    students=[a for a in loaded.sellers if a.owner!='ta']
    if {a.id for a in students}!=set(IDS):raise ValueError('Wrong student personas')
    for a in students:
        spec=loaded.models[a.model]
        if not a.reasoning or a.persona_body()!=m['personas'][a.id[-1]].strip():raise ValueError('Persona/thinking changed')
        if not spec.lora or Path(spec.lora.path).resolve()!=ADAPTER:raise ValueError('Not fixed starter adapter')
        if (spec.temperature,spec.top_p,spec.max_tokens)!=(.7,.95,8192):raise ValueError('Sampling changed')
    return plans


def prepare_arm(folder,label,seed):
    folder=Path(folder);folder.mkdir(parents=True)
    version=ROOT/f'training/persona_versions/seller_checkpoint1_{label}'
    frozen=json.loads((version/'manifest.json').read_text())
    if fingerprint(ADAPTER)!=frozen['baseline_adapter_fingerprints']:raise ValueError('Starter adapter changed')
    target=folder/'seller';target.mkdir();personas={}
    for i in ['1','2','3']:
        name=f'persona_{i}.md';p=version/name
        if sha(p)!=frozen['files'][name]['sha256']:raise ValueError('Frozen persona modified')
        personas[i]=p.read_text();(target/name).write_text(personas[i])
    (target/'agent.yaml').write_text(yaml.safe_dump(dict(id='example_student_seller',owner='example_student',
        role='seller',controller='llm',model='base',reasoning=True,lora=str(ADAPTER)),sort_keys=False))
    opponents=folder/'opponents';opponents.mkdir();entries={}
    for role,number in [('seller',2),('buyer',10)]:
        pool=persona_variants(load_agent(ROOT/f'agents/_ta/{role}/agent.yaml'),3);entries[role]=[]
        for n in range(number):
            a=pool[n%len(pool)];data=a.model_dump(mode='json',exclude_none=True)
            for key in ['source','persona','personas']:data.pop(key,None)
            data.update(id=f'ta_{role}_seat{n+1}',persona_text=a.persona_body(),
                        lora=str((Path(a.source).parent/a.lora).resolve()))
            path=opponents/f'{role}_{n+1}.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False));entries[role].append(str(path))
    official=yaml.safe_load((ROOT/'configs/economy.yaml').read_text())
    cfg=dict(name=f'seller_persona_{label}',seed=seed,models=str(ROOT/'configs/models.yaml'),
        sellers=[str(target),*entries['seller']],buyers=entries['buyer'],personas_per_student=3,
        rotate=False,repeats=1,save_trajectories=True,save_prompts=True,output_dir=str(folder),
        max_workers=15,parallel_markets=1,market=deep_merge(official['market'],official['phase1']['market']))
    config=folder/'market.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    loaded=market_run(config,root=ROOT)
    paths=[*ROOT.joinpath('adlmarket').glob('*.py'),Path(__file__),ROOT/'run_market.py',
           ROOT/'training/basic_checks.py',ROOT/'training/local_resources.py',ROOT/'training/market_load_balance.py',
           ROOT/'training/diagnose_step84_market.py',ROOT/'training/compare_buyer_markets.py',
           ROOT/'training/step84_market_report.py',ROOT/'training/decision_ablation.py',ROOT/'training/official_buyer_data.py',
           ROOT/'configs/models.yaml',ROOT/'configs/economy.yaml',ROOT/'scenarios/public/catalog.yaml',
           ROOT/'scenarios/public/items_example.json',version/'manifest.json',*version.glob('persona_*.md')]
    manifest=dict(arm=label,seed=seed,config=str(config),personas=personas,revision=MODEL_REVISION,
        sources={str(p.resolve()):sha(p) for p in paths},
        inputs={str(p):sha(p) for p in folder.rglob('*') if p.is_file()},
        adapters={str(Path(loaded.models[a.model].lora.path).resolve()):fingerprint(loaded.models[a.model].lora.path)
                  for a in loaded.everyone() if loaded.models[a.model].lora},
        versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','vllm','pydantic','PyYAML','openai']},
        training_updates=0,total_actions=450,target_actions_per_persona=30)
    validate(loaded,manifest);manifest['initial_signature']=initial_state(loaded);check_sources(manifest)
    write_json(folder/'suite.json',manifest);return manifest


def prepare_pair(folder,seed=4):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arms={label:prepare_arm(folder/label,label,seed) for label in ['v1','v2']}
    if arms['v1']['initial_signature']!=arms['v2']['initial_signature']:raise ValueError('Initial facts/opponents differ')
    plan=dict(experiment='seller_persona_v1_v2',seed=seed,arms={k:str(folder/k) for k in arms},
              initial_signature=arms['v1']['initial_signature'],total_actions=900,automatic_promotion=False,training_updates=0)
    write_json(folder/'compare_plan.json',plan);return plan


def configure(folder,m,gpus,port,mock=False):
    from training.local_resources import query_gpus,select_gpus,gpu_models,runtime_cache
    from training.market_load_balance import balance_ta_routes,validate_equivalent
    cfg=yaml.safe_load(Path(m['config']).read_text());before=market_run(m['config'],root=ROOT)
    devices,skipped,cache=[],[],{}
    if not mock:
        devices,skipped=select_gpus(query_gpus(),gpus,reserve_gib=2.,min_budget_gib=14.)
        print(f'GPU plan:{devices}; skipped:{skipped}',flush=True)
        if len(devices)==1:print('SINGLE-GPU FALLBACK; only one card fits.',flush=True)
        cache,note=runtime_cache(ROOT)
        if note:print(note,flush=True)
    models=yaml.safe_load((ROOT/'configs/models.yaml').read_text());balance=None
    if devices:
        models=gpu_models(models,devices,port)
        if len(devices)==2:balance=balance_ta_routes(cfg,before,models,folder,ROOT,devices,port)
    for s in models['serving'].values():s['extra_args']+=['--revision',m['revision'],'--tokenizer-revision',m['revision']]
    mp=folder/'models.yaml';mp.write_text(yaml.safe_dump(models,sort_keys=False));cfg['models']=str(mp)
    effective=folder/'effective_market.yaml';effective.write_text(yaml.safe_dump(cfg,sort_keys=False))
    after=market_run(effective,root=ROOT);validate(after,m);validate_equivalent(before,after)
    if initial_state(after)!=m['initial_signature']:raise ValueError('Routing changed initial engine state')
    routes={a.id:after.models[a.model].url_for(after.models[a.model].model) for a in after.everyone()}
    if len(devices)==2 and len(set(routes.values()))!=2:raise ValueError('Second GPU has no requests')
    write_json(folder/'resource_plan.json',dict(devices=devices,skipped=skipped,routes=routes,balance=balance,mock=mock))
    m['runtime_inputs']={str(p):sha(p) for p in [mp,effective,*folder.joinpath('routing_agents').rglob('*')] if p.is_file()}
    write_json(folder/'suite.json',m);return effective,cache


def play(folder,m,effective,cache,mock,offset):
    check_sources(m)
    for p,digest in m['runtime_inputs'].items():
        if sha(p)!=digest:raise ValueError(f'Runtime file changed:{p}')
    cmd=[sys.executable,'-u',str(ROOT/'run_market.py'),'--config',str(effective),'--mock-llm' if mock else '--launch-vllm']
    env={**os.environ,**cache,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and p!=str(ROOT/'.cache/fast_kernels/site-packages'))
    start=time.monotonic();first=None;last=0;offsets={};count=0;targets=Counter();run=None;position='loading / warm-up'
    with (folder/'runner.log').open('w') as log,(folder/'progress.log').open('w') as progress:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(folder/'runner.log',offsets):print(line,flush=True)
                found=list(folder.glob('run_market_*'));run=found[0] if found else None
                if run:
                    for p in run.glob('markets/*/events.jsonl'):
                        for line in read_new(p,offsets):
                            r=json.loads(line)
                            if r.get('type')=='market_start':first=first or time.monotonic()
                            if r.get('type')=='turn':
                                count+=1
                                if r['agent'] in IDS:targets[r['agent']]+=1
                                position=f'r{r["round"]}/5 t{(r["tick"]-1)%6+1}/6'
                    if count==0:
                        for p in run.glob('vllm/*.log'):
                            lines=read_new(p,offsets)
                            if lines:print(f'[vLLM {p.name}] {lines[-1][-180:]}',flush=True)
                now=time.monotonic()
                if done or now-last>=5:
                    speed=count/(now-first) if first and now-first>=5 else 0
                    eta=duration((900-offset-count)/speed) if speed else 'estimating'
                    line=(f'{m["arm"]} [{"#"*(count*24//450)}{"-"*(24-count*24//450)}] {count}/450 | overall {offset+count}/900 | {position} | '
                          +' '.join(f'p{i}:{targets[IDS[i-1]]}/30' for i in [1,2,3])
                          +f' | elapsed {duration(now-start)} | {speed:.2f} actions/s | overall ETA~{eta}')
                    print(line,flush=True);progress.write(line+'\n');progress.flush();last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    result=dict(exit=proc.returncode,run_dir=str(run) if run else None,actions=count,mock=mock,elapsed_seconds=time.monotonic()-start)
    write_json(folder/'wrapper_result.json',result)
    if proc.returncode or count!=450:raise RuntimeError(f'{m["arm"]} incomplete; inspect {folder}/runner.log')
    return run


def analyze(run):
    from adlmarket.market import load_catalog
    meta=json.loads((run/'metadata.json').read_text())
    if meta.get('status')!='finished':raise ValueError('Market not finished')
    folders=list((run/'markets').iterdir())
    if len(folders)!=1:raise ValueError('Expected exactly one full market')
    p=folders[0];final=json.loads((p/'final.json').read_text());scores={r['agent']:r for r in json.loads((p/'scores.json').read_text())}
    if (len(final['sellers']),len(final['buyers']))!=(5,10):raise ValueError('Wrong market size')
    rows=defaultdict(list);seen=Counter();load=defaultdict(Counter)
    with (p/'events.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('type')!='turn':continue
            seen[(r['agent'],r['tick'])]+=1
            if r['agent'] in IDS:rows[r['agent']].append(r)
            log=r.get('llm') or {}
            if log.get('base_url'):
                load[log['base_url']]['calls']+=1;load[log['base_url']]['output_tokens']+=log.get('output_tokens_spent',0)
    ids={a['id'] for a in final['sellers']+final['buyers']}
    if set(seen)!={(aid,tick) for aid in ids for tick in range(1,31)} or set(seen.values())!={1}:raise ValueError('Incomplete450-action coverage')
    catalog={q.id:q.ref_price for q in load_catalog(ROOT/'scenarios/public/catalog.yaml')}
    sellers={}
    for s in final['sellers']:
        aid=s['id']
        if aid not in IDS:continue
        trades=[t for t in final['trades'] if t['seller']==aid];revenue=cost=units=reference=low_units=low_savings=0;products=defaultdict(Counter)
        for t in trades:
            pid=t['product'];qty=t['qty'];unit_cost=s['low_cost'][pid] if t['grade']=='low' else s['cost'][pid]
            revenue+=t['price']*qty;cost+=unit_cost*qty;units+=qty;reference+=catalog[pid]*qty
            if t['grade']=='low':low_units+=qty;low_savings+=(s['cost'][pid]-unit_cost)*qty
            q=products[pid];q['units']+=qty;q['revenue_cents']+=t['price']*qty;q['profit_cents']+=(t['price']-unit_cost)*qty
            q['list_price_cents']=catalog[pid];q['low_units']+=qty if t['grade']=='low' else 0
        if revenue!=s['revenue'] or cost!=s['cost_of_sold'] or revenue-cost!=cents(scores[aid]['raw']):raise ValueError('Seller ledger mismatch')
        actions=sorted(rows[aid],key=lambda r:r['tick']);stats=turn_stats(actions);problems=[];below=0
        for r in actions:
            args=r.get('args') or {}
            if r.get('tool') in ['set_price','make_offer']:
                table=s['low_cost'] if args.get('grade')=='low' else s['cost'];c=table.get(args.get('product_id'))
                if c is not None and cents(args.get('price',0))<=c:below+=1
            if not r['result'].get('ok') or (r.get('llm') or {}).get('stop_reason')=='length':
                problems.append(dict(tick=r['tick'],tool=r.get('tool'),args=args,result=r['result'],reasoning=(r.get('llm') or {}).get('reasoning')))
        for rnd in range(1,6):
            if sum((r.get('llm') or {}).get('output_tokens_spent',0) for r in actions if r['round']==rnd)>8192:raise ValueError('Round token budget exceeded')
        sellers[aid]=dict(profit_cents=revenue-cost,revenue_cents=revenue,cost_cents=cost,units_sold=units,
            units_unsold=sum(s['stock'].values()),sales_to_students=sum(t['qty'] for t in trades if t['buyer'].startswith('example_student')),
            sales_price_to_list_ratio=revenue/reference if reference else None,low_grade_units=low_units,
            low_grade_cost_saving_cents=low_savings,by_product={k:dict(v) for k,v in products.items()},
            actions=dict(Counter(r.get('tool') or 'NO_TOOL' for r in actions)),
            errors=dict(Counter(r['result'].get('error') for r in actions if not r['result'].get('ok'))),
            below_or_at_cost_quotes=below,truncated_turns=stats['truncated_turns'],budget_waits=stats['budget_waits'],
            output_tokens=stats['spent_output_tokens'],problems=problems)
    if set(sellers)!=set(IDS):raise ValueError('Missing student sellers')
    return dict(complete=True,mock=meta.get('mock_llm',False),sellers=sellers,
        infrastructure_clean=all(not s.get('model_failures',0) for s in scores.values()),
        actual_server_load={k:dict(v) for k,v in load.items()})


def report(folder):
    plan=json.loads((folder/'compare_plan.json').read_text());arms={}
    for label,path in plan['arms'].items():
        arm=Path(path);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        w=json.loads((arm/'wrapper_result.json').read_text());run=Path(w['run_dir'])
        loaded=market_run(run/'config.yaml',root=ROOT);validate(loaded,m)
        if loaded.config.seed!=plan['seed'] or initial_state(loaded)!=plan['initial_signature']:raise ValueError('Played initial setup differs')
        arms[label]=analyze(run);write_json(arm/'seller_review.json',arms[label])
    real=all(not a['mock'] and a['infrastructure_clean'] for a in arms.values())
    lines=['# Seller persona v1/v2完整市場比較','',f'真模型有效對照={real}；seed={plan["seed"]}；各一場5×10／30turn，合計900行動。',
           '相同starter identity adapter、Thinking ON、商品、成本、初始庫存、對手與解碼，僅三份seller persona不同。',
           '','| Persona | v1 profit $ | v2 profit $ | 差額 $ | 售出units v1/v2 | 未售units v1/v2 | 截斷 v1/v2 |',
           '|---|---:|---:|---:|---|---|---|']
    for aid in IDS:
        a,b=arms['v1']['sellers'][aid],arms['v2']['sellers'][aid]
        lines.append(f'| {aid} | {a["profit_cents"]/100:.2f} | {b["profit_cents"]/100:.2f} | {(b["profit_cents"]-a["profit_cents"])/100:+.2f} | {a["units_sold"]}/{b["units_sold"]} | {a["units_unsold"]}/{b["units_unsold"]} | {a["truncated_turns"]}/{b["truncated_turns"]} |')
    totals={k:sum(s['profit_cents'] for s in a['sellers'].values()) for k,a in arms.items()}
    lines+=['',f'三persona本場帳目合計v1={totals["v1"]/100:.2f}、v2={totals["v2"]/100:.2f}；不是課程persona彙整。',
        'seller_review.json另列逐商品成交價／list price、實現毛利、low grade成本節省、未售庫存、工具錯誤、截斷與雙卡實際負載。',
        '低品質成本節省僅在相同成交價／數量下比較高低成本，不是已識別的策略因果收益；商品組合改變也會影響平均價格。',
        '目前無seller LoRA訓練；mock流程不算模型成績。單seed公共商品與toy買家只作診斷，不保證正式高分。',
        '沒有自動選新版／部署／訓練；先看各persona的profit與成交代價，再決定採用哪版。']
    result=dict(complete=True,valid_real_comparison=real,seed=plan['seed'],arms=arms,total_profit_cents=totals,
        automatic_promotion=False,training_updates=0)
    write_json(folder/'persona_comparison.json',result);(folder/'persona_comparison.md').write_text('\n'.join(lines)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--seed',type=int,default=4);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--mock',action='store_true');parser.add_argument('--resume',type=Path)
    parser.add_argument('--report',type=Path);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.report:return report(args.report.resolve())
    if args.resume:
        folder=args.resume.resolve();plan=json.loads((folder/'compare_plan.json').read_text())
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,args.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_personas_',dir=parent))
        print('CPU preparation: frozen personas,starter weights and matched full markets; ETA estimating.',flush=True)
        plan=prepare_pair(folder,args.seed)
    print(f'Prepared:{folder}; v1 then v2,900actions; no training.',flush=True)
    if args.prepare_only:print('PASS CPU only; no GPU query/service.');return folder
    if 'mock' in plan and plan['mock']!=args.mock:raise ValueError('Cannot mix mock and real resume')
    plan['mock']=args.mock;write_json(folder/'compare_plan.json',plan)
    for index,(label,path) in enumerate(plan['arms'].items()):
        arm=Path(path);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        previous=json.loads((arm/'wrapper_result.json').read_text()) if (arm/'wrapper_result.json').exists() else {}
        if previous.get('exit')==0 and previous.get('actions')==450:
            print(f'Skip completed {label}:450/450',flush=True);continue
        for p in list(arm.glob('run_market_*'))+([arm/'routing_agents'] if (arm/'routing_agents').exists() else []):
            p.rename(p.with_name(f'archived_{time.time_ns()}_{p.name}'))
        effective,cache=configure(arm,m,args.gpus,args.port,args.mock)
        play(arm,m,effective,cache,args.mock,index*450)
    result=report(folder)
    print(f'Completed.Read {folder}/persona_comparison.md; no automatic training or deployment.',flush=True)
    return result


if __name__=='__main__':main()
