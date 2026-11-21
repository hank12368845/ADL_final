"""Fixed same-input vLLM parent/checkpoint evaluation, official sampling and token budget."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import json
from pathlib import Path
import re
import time

from training.buyer_binding_data import ROOT,DATA,candidates


def validate_wire(cases):
    from training.chat_payload import api_messages
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    for c in cases:
        ChatCompletionRequest.model_validate(dict(model='Qwen/Qwen3.5-4B',messages=api_messages(c['messages']),
            tools=c['tools'],tool_choice='auto',max_tokens=c['observation'].get('output_tokens_left_this_round',8192),
            temperature=.7,top_p=.95,chat_template_kwargs={'enable_thinking':True}))
    return len(cases)


def assess(c,tool,args,log,t=None):
    from training.official_buyer_data import cents,schema_error
    errors=[];trade=None;obs=c['observation'];you=obs['you'];high_gain=0
    if tool is None:errors.append('no_tool_call')
    else:
        error=schema_error(tool,args,c['tools'])
        if error:errors.append(error)
    if not errors and tool=='send_message' and args.get('to') not in c['situation']['buyers']+c['situation']['sellers']:
        errors.append('unknown_recipient')
    if not errors and tool=='post_review' and not any(r.get('trade_id',r.get('id'))==args.get('trade_id') for r in obs.get('your_unreviewed_trades',[])):
        errors.append('review_not_in_unreviewed_list')
    if not errors and tool in ('buy_listed','accept_offer'):
        if tool=='buy_listed':
            trade=next((r for r in obs.get('board',[]) if (r['seller'],r['product_id'])==(args.get('seller_id'),args.get('product_id')) and r['in_stock']>=args.get('qty',1)),None)
            qty=args.get('qty',1)
            if trade is None:errors.append('not_current_board')
        else:
            trade=next((r for r in obs.get('offers_to_you',[]) if r['offer_id']==args.get('offer_id') and r['open_until_round']>=obs['round']),None)
            qty=trade['qty'] if trade else 0
            if trade is None:errors.append('not_current_offer')
        if trade:
            pid=trade['product_id'];value=cents(you['utility_of_next_unit'].get(pid,0));price=cents(trade['price'])*qty
            if qty!=1:errors.append('qty_not_one')
            if price>cents(you['cash']):errors.append('unaffordable')
            if value<=price:errors.append('nonpositive_visible_trade')
            if you['units_bought'].get(pid,0) and value==0:errors.append('already_owned_trade')
            high_gain=value-price
    truncated=log.get('stop_reason')=='length'
    if truncated:errors.append('truncated')
    if log.get('outcome') not in (None,'accepted'):errors.append('infrastructure_error')
    if log.get('arguments_repaired') or log.get('parsed_from_text'):errors.append('policy_repair')
    reason=log.get('reasoning') or ''
    sentences=[s.strip() for s in re.split(r'[\n.!?]+',reason) if len(s.strip())>=12]
    repeats=max(Counter(sentences).values(),default=0)
    valid=not errors
    return dict(case=c['name'],suite=c['suite'],family=c.get('family',c.get('mode')),persona_index=c['persona_index'],
        positive=c['positive'],valid=valid,captured=bool(c['positive'] and valid and trade and high_gain>0),
        negative_safe=bool(not c['positive'] and valid),tool=tool,arguments=args,errors=errors,
        truncated=truncated,cash_error='unaffordable' in errors,owned_error='already_owned_trade' in errors,
        pair_error=any(e in errors for e in ('not_current_board','not_current_offer')),
        reasoning_tokens=len(t.encode(reason,add_special_tokens=False)) if t else None,
        repeated_sentence_max=repeats,loop=repeats>=8,log=log)


def stats(rows):
    return dict(calls=len(rows),valid=sum(r['valid'] for r in rows),positive_total=sum(r['positive'] for r in rows),
        captured=sum(r['captured'] for r in rows),negative_safe=sum(r['negative_safe'] for r in rows),
        cash_errors=sum(r['cash_error'] for r in rows),owned_errors=sum(r['owned_error'] for r in rows),
        pair_errors=sum(r['pair_error'] for r in rows),truncated=sum(r['truncated'] for r in rows),
        no_tool=sum(r['tool'] is None for r in rows),loops=sum(r['loop'] for r in rows),
        max_think_tokens=max((r['reasoning_tokens'] or 0 for r in rows),default=0))


def summaries(rows):
    return {suite:{**stats([r for r in rows if r['suite']==suite]),
        'personas':{str(p):stats([r for r in rows if r['suite']==suite and r['persona_index']==p]) for p in (1,2,3)}}
        for suite in ('legacy36','binding_validation','stress','seen_train','market_regression')}


def gates(parent,child):
    checks={}
    for suite in ('legacy36','binding_validation','market_regression'):
        a,b=parent[suite],child[suite]
        checks[suite+'_persona_capture_retained']=all(b['personas'][str(p)]['captured']>=a['personas'][str(p)]['captured'] for p in (1,2,3))
        checks[suite+'_persona_negative_safe_retained']=all(b['personas'][str(p)]['negative_safe']>=a['personas'][str(p)]['negative_safe'] for p in (1,2,3))
        checks[suite+'_cash_owned_retained']=all(b['personas'][str(p)][k]<=a['personas'][str(p)][k] for p in (1,2,3) for k in ('cash_errors','owned_errors'))
    checks['new_pair_errors_not_worse']=child['binding_validation']['pair_errors']<=parent['binding_validation']['pair_errors']
    checks['stress_cash_owned_not_worse']=all(child['stress'][k]<=parent['stress'][k] for k in ('cash_errors','owned_errors'))
    checks['closure_no_regression']=all(child[s]['personas'][str(p)][k]<=parent[s]['personas'][str(p)][k] for s in parent for p in (1,2,3) for k in ('truncated','loops'))
    checks['measurable_main_improvement']=any(child[s][k]<parent[s][k] for s in ('binding_validation','market_regression') for k in ('pair_errors','truncated','loops')) or any(child[s]['valid']>parent[s]['valid'] for s in ('binding_validation','market_regression'))
    return checks


def make_policy(case,spec,client):
    from adlmarket.config import AgentSpec
    from adlmarket.policies import LLMPolicy
    from adlmarket.engine import with_reasoning
    body=json.loads(case['messages'][-1]['content'])
    policy=LLMPolicy(AgentSpec(id=case['situation']['your_id'],role='buyer',model='base',reasoning=True),
        with_reasoning(spec,True),client,case['messages'][0]['content'],case['tools'],6,1000,seed=0,save_prompts=True)
    turns=[]
    for m in case['messages'][1:-1]:
        if m['role']=='user':turns.append([])
        if not turns:raise ValueError('Invalid history boundary')
        turns[-1].append(copy.deepcopy(m))
    policy.history=turns;policy.private_state=body.get('your_private_state','');policy._last_result=body.get('last_action_result')
    policy.budget=case['observation'].get('output_tokens_left_this_round',8192)
    return policy


def evaluate(folder,labels):
    from transformers import AutoTokenizer
    from adlmarket.setups import market_run
    from adlmarket.serving import VLLMCluster,register_adapters
    from adlmarket.client import ModelClient
    from training.chat_payload import api_messages
    from training.buyer_binding_train import check_run
    folder=Path(folder);cfg=check_run(folder)
    cases=json.loads((Path(cfg['data_dir'])/'evaluation_cases.json').read_text())
    validate_wire(cases)
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    run=market_run(ROOT/'configs/basic_buyer.yaml',root=ROOT)
    base=run.models['base'];template=run.serving['base'];adapter=run.models[run.buyers[0].model]
    run.models={};run.serving={};names={}
    for i,d in enumerate(cfg['devices']):
        url=f'http://localhost:{cfg["port"]+i}/v1';server=f'binding_base_{i}'
        run.models[server]=base.model_copy(update=dict(base_url=url,replicas=[url]))
        run.serving[server]=template.model_copy(deep=True,update=dict(gpus=[d['index']],port=cfg['port']+i,gpu_memory_utilization=d['fraction'],
             extra_args=template.extra_args+['--revision',cfg['revision'],'--tokenizer-revision',cfg['revision']]))
        for label in labels:
            path=cfg['parent_adapter'] if label=='parent' else str(folder/'checkpoints'/f'step_{int(label):04d}')
            if label!='parent':
                from training.decision_ablation import check_fingerprint
                record=json.loads((Path(path)/'checkpoint_complete.json').read_text())
                check_fingerprint(path,record['adapter_fingerprints'])
            spec=adapter.model_copy(deep=True);spec.lora.path=path
            name=f'binding_{folder.name[-8:]}_{label}_{i}'
            run.models[name]=spec.model_copy(update=dict(model=name,base_url=url,replicas=[url],temperature=.7,top_p=.95,max_tokens=8192))
            names[label,i]=name
    real=ModelClient(api_key='EMPTY',timeout=300,retry_limit=2)
    class Wire:
        def complete(self,**kwargs):
            kwargs['messages']=api_messages(kwargs['messages']);return real.complete(**kwargs)
    jobs=[(c,label) for c in cases for label in labels];began=time.monotonic();outputs=[]
    def call(job):
        c,label=job;rank=c['evaluation_rank'];policy=make_policy(c,run.models[names[label,rank]],Wire())
        d=policy.decide(copy.deepcopy(c['observation']))
        if d.infra:raise RuntimeError(f'Invalid infrastructure result:{d.log}')
        r=assess(c,d.tool,d.args,d.log,t)
        r.update(label=label,gpu=cfg['devices'][rank]['index'],rank=rank,
            initial_messages_sha256=hashlib.sha256(json.dumps(api_messages(c['messages']),sort_keys=True).encode()).hexdigest(),
            messages_sha256=hashlib.sha256(json.dumps(d.log.get('messages'),sort_keys=True).encode()).hexdigest())
        return r
    print(f'Formal vLLM: labels{labels}, {len(jobs)} decisions;8192 budget,temp0.7,original input; loading ETA estimating.',flush=True)
    with VLLMCluster(run,folder/('vllm_'+('_'.join(labels)))):
        register_adapters(run)
        with ThreadPoolExecutor(max_workers=4) as pool,(folder/('evaluation_'+('_'.join(labels))+'.jsonl')).open('w') as stream:
            pending=[pool.submit(call,j) for j in jobs]
            for done,future in enumerate(as_completed(pending),1):
                r=future.result();stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();outputs.append(r)
                elapsed=time.monotonic()-began;speed=done/elapsed
                print(f'Eval {done}/{len(jobs)} | elapsed{elapsed:.0f}s | {speed:.3f}calls/s | ETA{(len(jobs)-done)/speed:.0f}s',flush=True)
    for label in labels:
        rows=[r for r in outputs if r['label']==label]
        if len(rows)!=len(cases) or len({r['case'] for r in rows})!=len(cases):raise ValueError('Incomplete evaluation')
        target=folder/f'evaluation_{label}.json'
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(summary=summaries(rows),decisions=rows),ensure_ascii=False,indent=1)+'\n')
        temporary.replace(target)
    write_review(folder)


def write_review(folder):
    folder=Path(folder);records={label:json.loads((folder/f'evaluation_{label}.json').read_text()) for label in ('parent','24','48') if (folder/f'evaluation_{label}.json').exists()}
    parent=records.get('parent',{}).get('summary');checks={label:gates(parent,r['summary']) for label,r in records.items() if label!='parent' and parent}
    eligible=[label for label,g in checks.items() if all(g.values())]
    chosen=max(eligible,key=lambda label:records[label]['summary']['binding_validation']['valid']) if eligible else None
    result=dict(complete=set(records)=={'parent','24','48'},gates=checks,development_candidate=str(folder/'checkpoints'/f'step_{int(chosen):04d}') if chosen else None,
        automatic_promotion=False,automatic_market=False,summaries={label:r['summary'] for label,r in records.items()})
    (folder/'binding_review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Step84 binding / closure continuation','',
        'Official-world engine trajectories; original personas/input; one384-example pass,48steps. Parent and checkpoints use identical vLLM0.7/0.95 and8192 budget, not a1200-token fuse.',
        'This is frozen-snapshot operation evaluation, not a market surplus or final-grade estimate. Stress quality priors are not engine probabilities.','',
        '| Version | Suite | Valid | Capture | Negative safe | Cash | Owned | Pair errors | Truncated | Loops | Max think tokens |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in records.items():
        for suite,s in r['summary'].items():
            lines.append(f'| {label} | {suite} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | {s["negative_safe"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["pair_errors"]} | {s["truncated"]} | {s["loops"]} | {s["max_think_tokens"]} |')
    lines+=['','Gates:','```json',json.dumps(checks,indent=2),'```',f'Development candidate:{result["development_candidate"]}',
            'No automatic deployment, next epoch, persona change or full market. Raw evaluation JSON contains all persona/family details and reasoning.']
    (folder/'binding_review.md').write_text('\n'.join(lines)+'\n');return result
