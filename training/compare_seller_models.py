#!/usr/bin/env python3
"""Fixed v2 personas: starter versus official Seller step46 in paired full markets.

User launches GPU work. --prepare-only and --report never query/start GPUs.
Old persona and training experiment sources remain untouched.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import compare_seller_personas as previous
from training.compare_buyer_markets import initial_signature
from training.diagnose_step84_market import check_sources,sha
from training.decision_ablation import fingerprint,check_fingerprint

IDS=previous.IDS
STARTER=previous.ADAPTER
CANDIDATE=ROOT/'training/runs/20261121-135008_seller_official_9d95_ubx/checkpoints/step_0046'


def write_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def candidate_check(path=CANDIDATE):
    from training.train_official_seller import check_run
    from adlmarket.config import check_adapter
    path=Path(path).resolve();run=path.parent.parent;cfg=check_run(run)
    complete=json.loads((path/'checkpoint_complete.json').read_text())
    if (complete['step'],json.loads((path/'training_step.json').read_text()))!=(46,dict(step=46,thinking=True)):
        raise ValueError('Expected frozen official Seller step46, Thinking ON')
    if cfg['experiment']!='seller_official_v1':raise ValueError('Not the official Seller baseline')
    check_fingerprint(path,complete['adapter_fingerprints'])
    check_adapter('paired_seller_candidate',path,cfg['base_model'],{'model.layers.':'model.language_model.layers.'})
    return dict(adapter=str(path),fingerprints=complete['adapter_fingerprints'],training_run=str(run),
                checkpoint_complete_sha256=sha(path/'checkpoint_complete.json'),training_config_sha256=sha(run/'train_config.json'))


def validate(loaded,m):
    from adlmarket.casting import plan_markets
    from adlmarket.market import load_catalog
    from adlmarket.schema import build_items_schema
    if (len(loaded.sellers),len(loaded.buyers))!=(5,10):raise ValueError('Expected5sellers×10buyers')
    rules=loaded.config.market;r=rules.rounds
    if (r.total,r.turns_per_round,r.output_tokens_per_round,rules.history_turns)!=(5,6,8192,6):raise ValueError('Formal horizon/history/token limits changed')
    if rules.model_dump(mode='json')!=m['market_rules']:raise ValueError('Market rules changed')
    seats=plan_markets(loaded.config.casting,loaded.sellers,loaded.buyers,loaded.config.seed,pads=loaded.pads)
    if len(seats)!=1 or loaded.config.seed!=m['seed']:raise ValueError('Wrong seed or extra market')
    schema=build_items_schema('seller_pair_check',rules.schema_,rules.schema_.items_file(0),load_catalog(ROOT/rules.catalog),ROOT,5)
    if len(schema.plans)!=24:raise ValueError('Expected24public products')
    targets=[a for a in loaded.sellers if a.owner!='ta']
    if {a.id for a in targets}!=set(IDS):raise ValueError('Expected3student Seller personas')
    for a in targets:
        spec=loaded.models[a.model]
        if not a.reasoning or a.persona_body()!=m['personas'][a.id[-1]].strip():raise ValueError('v2 persona or Thinking ON changed')
        if not spec.lora or Path(spec.lora.path).resolve()!=Path(m['adapter']).resolve():raise ValueError('Wrong Seller adapter')
        if (spec.lora.base_model,spec.temperature,spec.top_p,spec.max_tokens)!=('Qwen/Qwen3.5-4B',.7,.95,8192):raise ValueError('Model or decoding changed')
        if spec.lora_rename!={'model.layers.':'model.language_model.layers.'}:raise ValueError('Missing HF-to-vLLM LoRA name mapping')
    return seats


def opponent_signature(loaded):
    rows=[]
    for a in loaded.everyone():
        if a.owner!='ta':continue
        spec=loaded.models[a.model]
        rows.append(dict(id=a.id,role=a.role,persona=a.persona_body(),reasoning=a.reasoning,controller=a.controller,
                         model=spec.model,temperature=spec.temperature,top_p=spec.top_p,max_tokens=spec.max_tokens,
                         lora=fingerprint(spec.lora.path) if spec.lora else None))
    return sorted(rows,key=lambda a:a['id'])


def prepare_pair(folder,seed=5,candidate=CANDIDATE):
    from adlmarket.setups import market_run
    if seed==4:raise ValueError('seed4 was already used for persona development; use a new seed')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True);provenance=candidate_check(candidate);arms={};opponents=[]
    for label,adapter in [('starter',STARTER),('step46',Path(candidate).resolve())]:
        arm=folder/label
        # Reuse the audited opponent construction only, before freezing our own
        # adapter-variable experiment. Do not monkeypatch the old comparison.
        m=previous.prepare_arm(arm,'v2',seed)
        agent_path=arm/'seller/agent.yaml';agent=yaml.safe_load(agent_path.read_text());agent['lora']=str(adapter)
        agent_path.write_text(yaml.safe_dump(agent,sort_keys=False))
        cfg=yaml.safe_load(Path(m['config']).read_text());cfg['name']='seller_model_'+label
        Path(m['config']).write_text(yaml.safe_dump(cfg,sort_keys=False));loaded=market_run(m['config'],root=ROOT)
        m.update(experiment='seller_starter_step46_v1',arm=label,adapter=str(adapter.resolve()),
                 adapter_fingerprints=fingerprint(adapter),candidate_provenance=provenance,market_rules=loaded.config.market.model_dump(mode='json'))
        sources=[Path(__file__),ROOT/'training/queue_seller_personas.py',ROOT/'training/gpu_startup_errors.py',
                 ROOT/'training/train_official_seller.py',ROOT/'training/official_seller_data.py',ROOT/'training/official_seller_train.py',
                 Path(provenance['training_run'])/'train_config.json',Path(provenance['adapter'])/'checkpoint_complete.json']
        m['sources'].update({str(p.resolve()):sha(p) for p in sources})
        m['inputs']={str(p):sha(p) for p in arm.rglob('*') if p.is_file() and p.name!='suite.json'}
        m['adapters']={str(Path(loaded.models[a.model].lora.path).resolve()):fingerprint(loaded.models[a.model].lora.path)
                       for a in loaded.everyone() if loaded.models[a.model].lora}
        m['initial_signature']=initial_signature(loaded);validate(loaded,m);check_sources(m);opponents.append(opponent_signature(loaded))
        write_json(arm/'suite.json',m);arms[label]=m
    if arms['starter']['initial_signature']!=arms['step46']['initial_signature'] or opponents[0]!=opponents[1]:
        raise ValueError('Persona, initial market facts or opponents differ between arms')
    plan=dict(experiment='seller_starter_step46_v1',seed=seed,arms={k:str(folder/k) for k in arms},
              initial_signature=arms['starter']['initial_signature'],total_actions=900,target_actions=180,
              candidate_provenance=provenance,automatic_promotion=False,training_updates=0)
    write_json(folder/'compare_plan.json',plan);return plan


def configure(arm,m,selection,port,mock=False):
    from adlmarket.setups import market_run
    from training.local_resources import query_gpus,select_gpus,gpu_models,runtime_cache
    from training.market_load_balance import balance_ta_routes,validate_equivalent
    cfg=yaml.safe_load(Path(m['config']).read_text());before=market_run(m['config'],root=ROOT);devices=[];skipped=[];cache={};balance=None
    if not mock:
        devices,skipped=select_gpus(query_gpus(),selection,reserve_gib=2.,min_budget_gib=14.)
        print(f'GPU plan:{devices}; skipped:{skipped}',flush=True)
        if len(devices)==1:print('SINGLE-GPU: one eligible card. Rules/decoding unchanged.',flush=True)
        cache,note=runtime_cache(ROOT)
        if note:print(note,flush=True)
    models=yaml.safe_load((ROOT/'configs/models.yaml').read_text())
    if devices:
        models=gpu_models(models,devices,port)
        if len(devices)==2:balance=balance_ta_routes(cfg,before,models,arm,ROOT,devices,port)
    for serving in models['serving'].values():
        if serving['max_model_len']!=64000:raise ValueError('Formal context changed')
        serving['extra_args']+=['--revision',m['revision'],'--tokenizer-revision',m['revision']]
    mp=arm/'models.yaml';mp.write_text(yaml.safe_dump(models,sort_keys=False));cfg['models']=str(mp)
    effective=arm/'effective_market.yaml';effective.write_text(yaml.safe_dump(cfg,sort_keys=False))
    after=market_run(effective,root=ROOT);validate(after,m);validate_equivalent(before,after)
    if initial_signature(after)!=m['initial_signature']:raise ValueError('GPU routing changed initial conditions')
    routes={a.id:after.models[a.model].url_for(after.models[a.model].model) for a in after.everyone()}
    if len(devices)==2 and len(set(routes.values()))!=2:raise ValueError('Second GPU has no assigned calls')
    write_json(arm/'resource_plan.json',dict(devices=devices,skipped=skipped,routes=routes,balance=balance,mock=mock))
    m['runtime_inputs']={str(p):sha(p) for p in [mp,effective,*arm.joinpath('routing_agents').rglob('*')] if p.is_file()}
    write_json(arm/'suite.json',m);return effective,cache


def quote_diagnostics(rows,aid):
    known={};zero=[];unchanged=[];last=[];first={};stocked_waits=0
    for r in sorted(rows,key=lambda r:r['tick']):
        obs=r['observation'];stock=obs['you']['stock'];tool=r.get('tool');args=r.get('args') or {};pid=args.get('product_id')
        if tool=='wait' and any(stock.values()):stocked_waits+=1
        if tool not in ('set_price','make_offer'):continue
        if stock.get(pid,0)<args.get('qty',1):
            zero.append(dict(tick=r['tick'],product=pid,tool=tool,other_stock_available=any(n>0 for p,n in stock.items() if p!=pid),engine_result=r['result']))
        if r['round']==5 and obs['turns_left_this_round']==1:last.append(r['tick'])
        if tool=='set_price':
            key=(previous.cents(args.get('price',0)),args.get('grade','high'))
            board=next((b for b in obs['board'] if (b['seller'],b['product_id'])==(aid,pid)),None)
            if board and known.get(pid)==key and previous.cents(board['price'])==key[0]:unchanged.append(dict(tick=r['tick'],product=pid))
            if r['result'].get('ok'):known[pid]=key
        if r['result'].get('ok') and stock.get(pid,0)>0:first.setdefault(pid,r['tick'])
    return dict(observed_insufficient_stock_quotes=zero,unchanged_live_listings=unchanged,last_turn_quotes=last,
                waits_with_stock=stocked_waits,first_stocked_quote_tick=first)


def analyze(run):
    base=previous.analyze(run);market=next((run/'markets').iterdir());final=json.loads((market/'final.json').read_text())
    events=[json.loads(l) for l in (market/'events.jsonl').read_text().splitlines()];turns=[r for r in events if r.get('type')=='turn']
    by_id={s['id']:s for s in final['sellers']}
    for aid,row in base['sellers'].items():
        rows=[r for r in turns if r['agent']==aid];trades=[t for t in final['trades'] if t['seller']==aid];s=by_id[aid]
        row['quote_diagnostics']=quote_diagnostics(rows,aid)
        row['mean_unit_margin']=row['profit_cents']/100/row['units_sold'] if row['units_sold'] else None
        row['profit_by_round']={str(r):sum((t['price']-(s['low_cost'][t['product']] if t['grade']=='low' else s['cost'][t['product']]))*t['qty'] for t in trades if t['round']==r)/100 for r in range(1,6)}
        row['reviews']=[r for r in final.get('reviews',[]) if r['seller']==aid]
        row['actual_api_models']=dict(Counter((r.get('llm') or {}).get('model') for r in rows if (r.get('llm') or {}).get('model')))
    return base


def report(folder):
    from adlmarket.setups import market_run
    plan=json.loads((folder/'compare_plan.json').read_text());arms={}
    for label,path in plan['arms'].items():
        arm=Path(path);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        w=json.loads((arm/'wrapper_result.json').read_text())
        if w['exit'] or w['actions']!=450:raise ValueError('Cannot compare an incomplete market')
        run=Path(w['run_dir']);loaded=market_run(run/'config.yaml',root=ROOT);validate(loaded,m)
        if initial_signature(loaded)!=plan['initial_signature']:raise ValueError('Played setup differs')
        for path,h in m.get('runtime_inputs',{}).items():
            if sha(path)!=h:raise ValueError('Played runtime config changed')
        arms[label]=analyze(run);write_json(arm/'seller_model_review.json',arms[label])
    real=all(not a['mock'] and a['infrastructure_clean'] for a in arms.values())
    lines=['# Seller starter vs step46：固定v2完整市場','',f'真模型有效對照={real}；seed={plan["seed"]}；各5seller×10buyer、5round×6turn，共900行動。',
           '兩場相同初始庫存／成本／買家效用／預算／座位、對手與v2三persona，只換三學生seller共享的LoRA。Thinking ON、temp0.7／top_p0.95、64Kcontext／6history、8192每round。',
           '', '| persona | starter profit | step46 profit | 差額 | 售出 starter/46 | 未售 starter/46 | 單件毛利 starter/46 |',
           '|---|---:|---:|---:|---|---|---|']
    for aid in IDS:
        a,b=(arms[l]['sellers'][aid] for l in ('starter','step46'))
        margin=lambda r:'—' if r['mean_unit_margin'] is None else f'{r["mean_unit_margin"]:.2f}'
        lines.append(f'| {aid} | {a["profit_cents"]/100:.2f} | {b["profit_cents"]/100:.2f} | {(b["profit_cents"]-a["profit_cents"])/100:+.2f} | {a["units_sold"]}/{b["units_sold"]} | {a["units_unsold"]}/{b["units_unsold"]} | {margin(a)}/{margin(b)} |')
    lines+=['', '| persona | 觀測無足量庫存仍報價 starter/46 | 原掛牌同價同grade重掛 starter/46 | 截斷 starter/46 | 額度wait starter/46 |', '|---|---|---|---|---|']
    for aid in IDS:
        a,b=(arms[l]['sellers'][aid] for l in ('starter','step46'));x,y=a['quote_diagnostics'],b['quote_diagnostics']
        lines.append(f'| {aid} | {len(x["observed_insufficient_stock_quotes"])}/{len(y["observed_insufficient_stock_quotes"])} | {len(x["unchanged_live_listings"])}/{len(y["unchanged_live_listings"])} | {a["truncated_turns"]}/{b["truncated_turns"]} | {a["budget_waits"]}/{b["budget_waits"]} |')
    lines+=['','無庫存set_price可被引擎接受；此處是觀測已知無法供貨的行動，與真正工具拒絕分開。同價重掛定義須同時看到本人既有掛牌及先前確認grade，不能把所有重複商品動作都算錯。',
            'JSON另有逐商品實際價量／grade毛利、逐round profit、首次掛牌tick、低grade成本差與評論、真正工具錯誤／模型失敗、raw API model名稱、實際各端點負載。',
            '同價重掛、等待與無貨報價不直接換算為確定損失；低grade成本節省已在profit內，不再加分，也不是策略因果效果。',
            '每persona單場、單rotation、同一public商品及toy對手；新seed不是private商品泛化證據。同seed控制初始條件，後續行為自然分歧，非逐token相同或顯著性證據。',
            'mock不算收益證據；沒有自動挑checkpoint、混搭persona、部署、再訓練或追加市場。']
    result=dict(complete=True,valid_real_comparison=real,seed=plan['seed'],arms=arms,total_actions=900,
                automatic_promotion=False,training_updates=0,candidate=plan['candidate_provenance'])
    write_json(folder/'seller_model_comparison.json',result);(folder/'seller_model_comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def completed(arm):
    p=arm/'wrapper_result.json'
    if not p.exists():return False
    w=json.loads(p.read_text());return w.get('exit')==0 and w.get('actions')==450


def execute(folder,plan,selection='0,1',port=8100,mock=False,wait=False):
    from training.queue_seller_personas import wait_ready
    from training.gpu_startup_errors import startup_memory_races
    allowed=[x.strip() for x in selection.split(',')]
    for index,(label,path) in enumerate(plan['arms'].items()):
        arm=Path(path);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        if completed(arm):print(f'Skip completed {label}:450/450',flush=True);continue
        while True:
            selected=wait_ready(allowed,2,port) if wait and not mock else allowed
            if shutil.disk_usage(folder).free<5*1024**3:raise RuntimeError('Less than5GiB result space; cancelled')
            for p in list(arm.glob('run_market_*'))+([arm/'routing_agents'] if (arm/'routing_agents').exists() else []):
                p.rename(p.with_name(f'archived_{time.time_ns()}_{p.name}'))
            started=time.time()
            try:
                effective,cache=configure(arm,m,','.join(selected),port,mock)
                previous.play(arm,m,effective,cache,mock,index*450)
                break
            except RuntimeError as error:
                races=startup_memory_races(folder,started)
                preflight='No GPU has enough free memory after reserve' in str(error)
                if not wait or mock or not (preflight or races):raise
                evidence=dict(time=time.time(),arm=label,preflight=preflight,evidence=races)
                with (folder/'resource_requeues.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')
                print('VRAM changed during startup; requeue this arm, preserving completed markets. Queue ETA unknown.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8100);p.add_argument('--seed',type=int,default=5)
    p.add_argument('--wait',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--mock',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--report',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.report:return report(a.report.resolve())
    allowed=a.gpus.split(',')
    if not 1<=len(allowed)<=2 or any(not x.strip().isdigit() for x in allowed) or len(set(x.strip() for x in allowed))!=len(allowed):p.error('Use one or two distinct physical GPU indices')
    if not 1<=a.port<=65534:p.error('Invalid port range')
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'compare_plan.json').read_text())
        if plan['experiment']!='seller_starter_step46_v1':raise ValueError('Wrong experiment folder')
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_models_',dir=parent))
        print('CPU: freezingv2 personas, adapters and matched formal markets; ETA estimating.',flush=True)
        plan=prepare_pair(folder,a.seed)
    print(f'Output:{folder}; starter then step46;900actions; no training or deployment.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query/service/wait.');return folder
    if 'mock' in plan and plan['mock']!=a.mock:raise ValueError('Cannot mix mock and real markets')
    plan['mock']=a.mock;write_json(folder/'compare_plan.json',plan)
    execute(folder,plan,a.gpus,a.port,a.mock,a.wait)
    result=report(folder)
    print(f'Completed.Read {folder}/seller_model_comparison.md; no automatic next GPU work.',flush=True);return result


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; completed arms retained. Resume the printed folder.',flush=True);raise SystemExit(130)
