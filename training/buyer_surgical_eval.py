"""Fixed native vLLM evaluation for surgical continuation; no promotion to agents."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import json
from pathlib import Path
import time

from training.prepare_buyer_surgical import ROOT,allowed_options,cooldown_pair
from training.buyer_binding_eval import validate_wire,make_policy,stats as base_stats,assess as base_assess

SUITES=('legacy36','binding_validation','stress','prior_synthetic_probe','training_market_seen','surgical_validation')


def assess(c,tool,args,log,t=None):
    r=base_assess(c,tool,args,log,t)
    if tool=='buy_listed' and cooldown_pair(c)==(args.get('seller_id'),args.get('product_id')):
        r['errors'].append('persona_cooldown');r.update(valid=False,captured=False,negative_safe=False)
    options=allowed_options(c)
    selected=next((o for o in options if (tool=='buy_listed' and o['source']=='board' and (o['seller'],o['product'])==(args.get('seller_id'),args.get('product_id')))
        or (tool=='accept_offer' and o['source']=='offers_to_you' and o['offer_id']==args.get('offer_id'))),None)
    r['captured_visible_lower_bound_cents']=selected['gain_lower_bound_cents'] if selected and r['captured'] else 0
    r['best_visible_lower_bound_cents']=options[0]['gain_lower_bound_cents'] if options else 0
    return r


def stats(rows):
    s=base_stats(rows)
    s['captured_visible_lower_bound_cents']=sum(r.get('captured_visible_lower_bound_cents',0) for r in rows)
    s['best_visible_lower_bound_cents']=sum(r.get('best_visible_lower_bound_cents',0) for r in rows)
    s['output_tokens']=sum(r['log'].get('output_tokens_spent',0) for r in rows)
    return s


def summaries(rows):
    return {suite:{**stats([r for r in rows if r['suite']==suite]),
        'personas':{str(p):stats([r for r in rows if r['suite']==suite and r['persona_index']==p]) for p in (1,2,3)}} for suite in SUITES}


def gates(parent,child):
    checks={}
    for suite in ('legacy36','binding_validation','surgical_validation'):
        for key,op in [('captured',lambda a,b:b>=a),('negative_safe',lambda a,b:b>=a),
                       ('cash_errors',lambda a,b:b<=a),('owned_errors',lambda a,b:b<=a),
                       ('captured_visible_lower_bound_cents',lambda a,b:b>=a)]:
            checks[suite+'_'+key+'_retained']=all(op(parent[suite]['personas'][str(p)][key],child[suite]['personas'][str(p)][key]) for p in (1,2,3))
    checks['source_errors_not_worse']=all(child[s]['pair_errors']<=parent[s]['pair_errors'] for s in ('binding_validation','surgical_validation'))
    checks['stress_brakes_retained']=all(child['stress']['personas'][str(p)][k]<=parent['stress']['personas'][str(p)][k] for p in (1,2,3) for k in ('cash_errors','owned_errors'))
    checks['closure_not_worse']=all(child[s]['personas'][str(p)][k]<=parent[s]['personas'][str(p)][k] for s in SUITES for p in (1,2,3) for k in ('truncated','loops'))
    checks['targeted_error_reduced']=any(child[s][k]<parent[s][k] for s in ('binding_validation','surgical_validation','training_market_seen') for k in ('pair_errors','truncated','loops'))
    return checks


def evaluate(folder,labels):
    from transformers import AutoTokenizer
    from adlmarket.setups import market_run
    from adlmarket.serving import VLLMCluster,register_adapters
    from adlmarket.client import ModelClient
    from training.chat_payload import api_messages
    from training.buyer_surgical_train import check_run
    folder=Path(folder);cfg=check_run(folder)
    cases=json.loads((Path(cfg['data_dir'])/'evaluation_cases.json').read_text())
    validate_wire(cases)
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    run=market_run(ROOT/'configs/basic_buyer.yaml',root=ROOT)
    base=run.models['base'];template=run.serving['base'];adapter=run.models[run.buyers[0].model]
    run.models={};run.serving={};names={}
    for i,d in enumerate(cfg['devices']):
        url=f'http://localhost:{cfg["port"]+i}/v1';server=f'surgical_base_{i}'
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
            name=f'surgical_{folder.name[-8:]}_{label}_{i}'
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
    folder=Path(folder)
    records={label:json.loads((folder/f'evaluation_{label}.json').read_text()) for label in ('parent','10','20') if (folder/f'evaluation_{label}.json').exists()}
    parent=records.get('parent',{}).get('summary')
    checks={label:gates(parent,record['summary']) for label,record in records.items() if label!='parent' and parent}
    eligible=[label for label,g in checks.items() if all(g.values())]
    chosen=max(eligible,key=lambda label:records[label]['summary']['binding_validation']['captured_visible_lower_bound_cents']) if eligible else None
    result=dict(complete=set(records)=={'parent','10','20'},gates=checks,
        development_candidate=str(folder/'checkpoints'/f'step_{int(chosen):04d}') if chosen else None,
        automatic_promotion=False,automatic_market=False,summaries={label:r['summary'] for label,r in records.items()})
    (folder/'surgical_review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Step84局部修補驗收','',
        '20steps,80exposures:56faithful step84 replay+24repairs;original personas and thinking style. Same native vLLM0.7/0.95,8192 remaining round budget.',
        'training_market_seen is contaminated by the authorized replay/repair training. Other suites are development checks, not a sealed test or market grade.',
        'Visible lower-bound gains below sum independent snapshot choices; they are NOT realized surplus, market regret, or guaranteed fills.','',
        '| Version | Suite | Valid | Capture | Cash | Owned | Pair | Trunc | Repeat | Max think | Captured visible gain $ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,record in records.items():
        for suite,s in record['summary'].items():
            lines.append(f'| {label} | {suite} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["pair_errors"]} | {s["truncated"]} | {s["loops"]} | {s["max_think_tokens"]} | {s["captured_visible_lower_bound_cents"]/100:.2f} |')
    lines+=['','```json',json.dumps(checks,indent=2),'```',f'Development candidate:{result["development_candidate"]}',
        'No automatic deployment, persona change, extra epoch or full market. Inspect raw evaluation_parent/10/20.json and actual gains before any promotion.']
    (folder/'surgical_review.md').write_text('\n'.join(lines)+'\n')
    return result

