#!/usr/bin/env python3
"""CPU-only surgical SFT curation: faithful step84 replay plus audited local repairs."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.buyer_binding_data import MARKET,PARENT,Scene,sha,candidates

DATA=ROOT/'training/data/buyer_surgical_v1'
CONTROL_FAMILIES=('offer_without_board','first_loss_later_gain','cooldown_active','cooldown_expired')
SEED=20261124


def dumps(obj):return json.dumps(obj,ensure_ascii=False,separators=(',',':'))


def last_attempt(case):
    """Formal harness usually places results in history, not last_action_result."""
    messages=case['messages'];body=json.loads(messages[-1]['content'])
    previous=next((m for m in reversed(messages[1:-1]) if m['role']=='assistant' and m.get('tool_calls')),None)
    result=body.get('last_action_result')
    if result is None:
        tool=next((m for m in reversed(messages[1:-1]) if m['role']=='tool'),None)
        result=json.loads(tool['content']) if tool else None
    if previous is None:return None,None,result
    fn=previous['tool_calls'][0]['function'];args=fn['arguments']
    return fn['name'],json.loads(args) if isinstance(args,str) else args,result


def cooldown_pair(case):
    name,args,result=last_attempt(case)
    if name=='buy_listed' and (result or {}).get('error') in ('no_stock','not_listed'):
        return args['seller_id'],args['product_id']
    return None


def allowed_options(case):
    blocked=cooldown_pair(case)
    return [r for r in candidates(case) if r['qty']==1 and
            not(r['source']=='board' and (r['seller'],r['product'])==blocked)]


def function(answer):
    fn=answer['tool_calls'][0]['function'];args=fn['arguments']
    return fn['name'],json.loads(args) if isinstance(args,str) else args


def audit_answer(case):
    from training.official_buyer_data import schema_error,cents
    tool,args=function(case['answer']);error=schema_error(tool,args,case['tools'])
    if error:raise ValueError((case['name'],error))
    if tool not in ('buy_listed','accept_offer','wait'):raise ValueError('Unexpected target tool')
    if tool=='wait':
        if allowed_options(case):raise ValueError('Wait discards a known positive option')
        return
    if tool=='buy_listed' and cooldown_pair(case)==(args['seller_id'],args['product_id']):
        raise ValueError('Target violates original persona one-turn cooldown')
    obs=case['observation'];you=obs['you']
    if tool=='buy_listed':
        row=next((r for r in obs['board'] if (r['seller'],r['product_id'])==(args['seller_id'],args['product_id']) and r['in_stock']>=args['qty']),None)
        qty=args['qty']
    else:
        row=next((r for r in obs.get('offers_to_you',[]) if r['offer_id']==args['offer_id'] and r['open_until_round']>=obs['round']),None)
        qty=row['qty'] if row else 0
    if row is None or qty!=1:raise ValueError('Missing current native transaction')
    pid=row['product_id'];price=cents(row['price'])*qty
    if price>cents(you['cash']) or price>=cents(you['utility_of_next_unit'].get(pid,0)):
        raise ValueError('Target loses money or exceeds cash')


def repair_answer(case):
    """Local natural-language correction; no mandatory checklist or think-length target."""
    options=allowed_options(case);obs=case['observation'];you=obs['you']
    if not options:
        if case.get('family')!='cooldown_wait':raise ValueError('Unexpected all-wait repair')
        state=dict(held=sorted(p for p,n in you['units_bought'].items() if n),cash=you['cash'],pick=None,
                   unowned=None,live=None,afford=None,gain=None,bad=[],pending=None,last_buy=None)
        return dict(role='assistant',content='',reasoning_content='The last failed seller/product pair is excluded for this turn. The other current listings are not profitable. I will wait for the one-turn cooldown to expire, then check the new observation.',
            tool_calls=[dict(id='answer',type='function',function=dict(name='wait',arguments=dict(private_state=dumps(state))))])
    best=options[0];pid=best['product']
    if best['source']=='offers_to_you':
        tool='accept_offer';args=dict(offer_id=best['offer_id'])
        reason=(f'The current binding offer {best["offer_id"]} is for {pid} from {best["seller"]} at '
                f'${best["total_cents"]/100:.2f}. It is open and affordable with ${you["cash"]:.2f}. '
                'It does not need a public listing; use accept_offer for this offer, not buy_listed. ')
    else:
        tool='buy_listed';args=dict(seller_id=best['seller'],product_id=pid,qty=1)
        reason=(f'The current board has {pid} from {best["seller"]} at ${best["total_cents"]/100:.2f}. ')
        family=case.get('family','')
        if family=='first_loss_later_gain':
            first=obs['board'][0]
            reason=(f'The first listing, {first["seller"]}/{first["product_id"]} at ${first["price"]:.2f}, '
                    'is not worthwhile. There are other listings to compare. '+reason)
        if family=='cooldown_active':reason+='The last failed pair is excluded for this turn; this is a different eligible pair. '
        elif family=='cooldown_expired':reason+='One intervening turn has passed since the failed attempt, so that one-turn cooldown has expired. '
        elif family=='real_loop_fix':reason+='Ignore the unavailable historical item and choose this live listing. '
        elif family=='real_pair_fix':reason+='Copy this seller and product from the same row, rather than the stale pair in history. '
        reason+=f'Cash is ${you["cash"]:.2f}; the current next-unit value is ${you["utility_of_next_unit"][pid]:.2f}. '
    reason+=f'The applicable quality lower-bound surplus is ${best["gain_lower_bound_cents"]/100:.2f}>0. I will take it now.'
    held=sorted(p for p,n in you['units_bought'].items() if n)
    state=dict(held=held,cash=you['cash'],pick=pid,unowned=pid not in held,live=True,afford=True,
               gain=best['gain_lower_bound_cents']/100,bad=[],pending=None,last_buy=[best['seller'],pid] if tool=='buy_listed' else None)
    args['private_state']=dumps(state)
    return dict(role='assistant',content='',reasoning_content=reason,
                tool_calls=[dict(id='answer',type='function',function=dict(name=tool,arguments=args))])


def real_market_rows():
    actors={r['agent']:r for r in map(json.loads,(MARKET/'agents.jsonl').read_text().splitlines())}
    final=json.loads((MARKET/'final.json').read_text());trades={t['id']:t for t in final['trades']}
    replay=[];repairs=[];excluded=[]
    for r in map(json.loads,(MARKET/'events.jsonl').read_text().splitlines()):
        if r.get('type')!='turn' or not r['agent'].startswith('example_student_buyer'):continue
        log=r.get('llm') or {}
        if not log.get('messages'):continue
        actor=actors[r['agent']];name=f'real_{r["agent"]}_tick{r["tick"]}'
        c=dict(id=name,name=name,group='training_real_market_seed0',trajectory_id='real_e0_m0',
               persona_index=int(r['agent'][-1]),source_tick=r['tick'],messages=copy.deepcopy(log['messages']),
               situation=actor['situation'],observation=r['observation'],tools=actor['tools'],
               market_source=str(MARKET),positive=True)
        tr=trades.get(r['result'].get('data',{}).get('trade_id'))
        if tr and tr['utility']>tr['price']*tr['qty'] and r['result']['ok']:
            args=json.loads(log['raw_arguments'])
            c.update(family='faithful_step84_replay',source='step84_market_success_unchanged',
                realized_surplus_cents=tr['utility']-tr['price']*tr['qty'],
                answer=dict(role='assistant',content=log.get('text') or '',reasoning_content=log.get('reasoning') or '',
                    tool_calls=[dict(id='answer',type='function',function=dict(name=r['tool'],arguments=args))]))
            if log['stop_reason']=='length' or not c['answer']['reasoning_content']:
                excluded.append(dict(name=name,reason='missing_or_truncated_reasoning'));continue
            try:audit_answer(c)
            except ValueError as e:
                excluded.append(dict(name=name,reason=str(e)));continue
            replay.append(c)
        else:
            args=r.get('args') or {}
            absent=(r['tool']=='buy_listed' and not any((b['seller'],b['product_id'])==(args.get('seller_id'),args.get('product_id')) and b['in_stock']>=1 for b in r['observation']['board']))
            loop=log.get('stop_reason')=='length'
            if not(absent or loop) or not allowed_options(c):continue
            c.update(family='real_loop_fix' if loop else 'real_pair_fix',source='step84_real_failure_local_correction',
                     original_reasoning=log.get('reasoning') or '',original_tool=r['tool'],original_args=r['args'])
            c['answer']=repair_answer(c);audit_answer(c);repairs.append(c)
    return replay,repairs,excluded


def control(source,seed,persona,text,tokenizer,family):
    """Full native engine state; real failed call, intervening turn and publication."""
    from adlmarket.market import Listing
    from training.official_buyer_data import cents
    from adlmarket.prompts import observation_message
    s=Scene(source,seed,persona,text,tokenizer);m=s.market;a,b=s.sellers[:2]
    values={pid:table[0] for pid,table in m.buyers[s.aid].utility.items()}
    fixed=sorted((pid for pid in values if not m.schema.plan(pid).quality_varies),key=lambda p:-values[p])
    target,other,bg=fixed[:3];price=max(2,values[target]//5)
    if seed%2:
        m.listings[a,bg]=Listing(1)
        assert s.past('buy_listed',dict(seller_id=a,product_id=bg,qty=1))['ok']
        m.listings.clear()
        s.past('send_message',dict(to=b,message='Please send a binding offer if you have a better price.'))
    if family.startswith('cooldown_'):
        failed=s.past('buy_listed',dict(seller_id=a,product_id=target,qty=1))
        assert failed['error']=='not_listed'
        if family=='cooldown_expired':s.past('wait',{})
    # Realistic full board, all background quotes above current marginal value.
    products=list(m.products)
    for i in range(28):
        pid=products[i%len(products)];seller=s.sellers[(i//len(products)+i)%len(s.sellers)]
        s.post(seller,pid,values[pid]+100)
    if family=='offer_without_board':
        # No public row for the offer pair: the source channel must determine the tool.
        m.listings.pop((b,target),None);m._pending_listings.pop((b,target),None)
        result=m.apply(b,'make_offer',dict(buyer_id=s.aid,product_id=target,price=price/100,qty=1)).as_dict()
        if not result['ok']:raise ValueError(result)
    elif family=='cooldown_active':
        s.post(a,target,price);s.post(b,other,max(2,values[other]//4))
    else:s.post(a,target,price)
    m.end_turn()
    # observe shuffles rows itself; choose a reachable first-loss presentation.
    for _ in range(100):
        obs=s.observe()
        if family!='first_loss_later_gain' or cents(obs['board'][0]['price'])>=cents(obs['you']['utility_of_next_unit'].get(obs['board'][0]['product_id'],0)):break
    else:raise ValueError('Unable to obtain first-loss presentation')
    name=f'surgical_{seed}_p{persona}_{family}'
    c=dict(id=name,name=name,group=source['group'],trajectory_id=f'engine_{seed}',persona_index=persona,
           family=family,source='official_world_current_engine_control',observation=obs,situation=s.situation,
           tools=list(m.tools['buyer'].values()),messages=[dict(role='system',content=s.system),
             *[msg for turn in s.history[-6:] for msg in turn],dict(role='user',content=observation_message(obs,s.note,s.last))],positive=True)
    c['answer']=repair_answer(c);audit_answer(c)
    tool,args=function(c['answer']);before=m.buyers[s.aid].gained_utility
    clone=copy.deepcopy(m);result=clone.apply(s.aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
    if not result['ok']:raise ValueError(result)
    gain=clone.buyers[s.aid].gained_utility-before-round(result.get('data',{}).get('total',0)*100)
    if tool!='wait' and gain<=0:raise ValueError('Control does not profit')
    c['positive']=tool!='wait'
    c.update(engine_result=result,teacher_actual_gain_cents=gain,
             engine_budget_cents=m.buyers[s.aid].budget,engine_spent_cents=m.buyers[s.aid].spent)
    if family=='offer_without_board' and tool!='accept_offer':raise ValueError('Offer was lost')
    if family=='first_loss_later_gain' and tool!='buy_listed':raise ValueError('Did not continue scanning')
    return c


def encode(tokenizer,c):
    # Match vLLM's API-to-template conversion without changing frozen raw messages.
    messages=copy.deepcopy(c['messages'])
    for message in messages:
        for call in message.get('tool_calls',[]):
            if isinstance(call['function']['arguments'],str):
                call['function']['arguments']=json.loads(call['function']['arguments'])
    prompt=tokenizer.apply_chat_template(messages,tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    full=tokenizer.apply_chat_template(messages+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
    p=tokenizer.encode(prompt,add_special_tokens=False);ids=tokenizer.encode(full,add_special_tokens=False)
    if not full.startswith(prompt) or ids[:len(p)]!=p or len(ids)>64000:raise ValueError('Native boundary/context failure')
    tool,_=function(c['answer'])
    return dict(id=c['id'],group=c['group'],family=c['family'],label_action=tool,persona_index=c['persona_index'],
        input_ids=ids,prompt_length=len(p),target_length=len(ids)-len(p),thinking=True,
        reasoning_tokens=len(tokenizer.encode(c['answer']['reasoning_content'],add_special_tokens=False)))


def audit_folder(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    for section in ('sources','files'):
        for filename,digest in manifest[section].items():
            path=Path(filename) if section=='sources' else folder/filename
            if sha(path)!=digest:raise ValueError(f'{section} changed:{path}')
    for filename,digest in manifest['parent_fingerprints'].items():
        if sha(PARENT/filename)!=digest:raise ValueError('Parent adapter changed')
    rows=[json.loads(l) for l in (folder/'train.jsonl').read_text().splitlines()]
    tokens=[json.loads(l) for l in (folder/'train.tokens.jsonl').read_text().splitlines()]
    batches=json.loads((folder/'batch_plan.json').read_text())
    for c in rows:
        audit_answer(c)
        body=json.loads(c['messages'][-1]['content'])
        if any(body.get(k)!=v for k,v in c['observation'].items()):raise ValueError('Current input/observation mismatch')
        if 'engine_budget_cents' in c and round(c['observation']['you']['cash']*100)!=c['engine_budget_cents']-c['engine_spent_cents']:
            raise ValueError('Engine cash ledger mismatch')
    if len(tokens)!=len(rows):raise ValueError('Missing token rows')
    counts=Counter(i for b in batches for i in b)
    if len(batches)!=20 or any(len(b)!=4 for b in batches) or sum(counts.values())!=80:raise ValueError('Wrong exposure budget')
    if sum(n for i,n in counts.items() if rows[i]['family']=='faithful_step84_replay')!=56:raise ValueError('Not70/30 exposure')
    if set(counts)!=set(range(len(rows))):raise ValueError('Unique sample not exposed')
    if set(manifest['train_groups'])&set(manifest['validation_groups']):raise ValueError('Split leakage')
    return manifest


def generate(output=DATA):
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.buyer_latest_brakes_data import representative_worlds
    output=Path(output)
    if output.exists():raise ValueError('Refuse overwrite; use a new output directory')
    t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',local_files_only=True)
    texts={i:(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text() for i in (1,2,3)}
    replay,repairs,excluded=real_market_rows()
    if len(repairs)!=13:raise ValueError(f'Expected13 known failures, got{len(repairs)}; audit sources')
    official={s:[json.loads(l) for l in (ROOT/f'training/data/buyer_official_v1/{s}.jsonl').read_text().splitlines()] for s in ('train','validation')}
    worlds={s:representative_worlds(official[s],6) for s in official}
    # Thirteen real failures plus eleven controls; avoid expanding into another broad curriculum.
    jobs=[('offer_without_board',p) for p in (1,2,3,1,2)]
    jobs += [('first_loss_later_gain',p) for p in (1,2,3)]
    jobs += [('cooldown_active',1),('cooldown_expired',2),('cooldown_wait',3)]
    for i,(family,p) in enumerate(tqdm(jobs,desc='CPU repair controls',unit='cases')):
        repairs.append(control(worlds['train'][i%6],SEED+i,p,texts[p],t,family))
    assert len(repairs)==24
    heldout=[]
    for i,family in enumerate(CONTROL_FAMILIES):
        for p in (1,2,3):heldout.append(control(worlds['validation'][(i+p)%6],SEED+10000+i*10+p,p,texts[p],t,family))
    frozen=json.loads((ROOT/'training/data/buyer_binding_v1/evaluation_cases.json').read_text())
    for c in frozen:
        if c['suite']=='market_regression':c['suite']='training_market_seen'
        if c['suite']=='seen_train':c['suite']='prior_synthetic_probe'
    for c in heldout:c['suite']='surgical_validation'
    evaluation=frozen+heldout
    rows=replay+repairs;encoded=[encode(t,c) for c in tqdm(rows,desc='CPU native tokenization',unit='cases')]
    # Exact70/30 over80 exposures; each five-step window also has14 replay/6 repair.
    rng=random.Random(SEED);replay_pools={p:[i for i,c in enumerate(replay) if c['persona_index']==p] for p in (1,2,3)}
    schedule=[]
    for p,n in ((1,20),(2,18),(3,18)):
        pool=replay_pools[p];chosen=[]
        while len(chosen)<n:
            order=pool.copy();rng.shuffle(order);chosen+=order
        schedule+=chosen[:n]
    rng.shuffle(schedule);repair_order=list(range(len(replay),len(rows)));rng.shuffle(repair_order)
    batches=[]
    ri=fi=0
    for i in range(20):
        nr=2 if i%5==4 else 3;nf=4-nr
        batch=schedule[ri:ri+nr]+repair_order[fi:fi+nf];ri+=nr;fi+=nf
        rng.shuffle(batch);batches.append(batch)
    for i,c in enumerate(evaluation):c['evaluation_rank']=i%2
    recipe=dict(parent_adapter=str(PARENT),rank=16,alpha=32,thinking=True,max_steps=20,checkpoint_steps=[10,20],
        effective_batch_size=4,world_size=2,learning_rate=5e-6,warmup_steps=2,optimizer='fresh AdamW',scheduler='cosine',
        objective='ordinary native completion CE; no extra weights, think cap, prefix or new format',
        max_length=max(len(x['input_ids']) for x in encoded),automatic_deployment=False,automatic_market=False,
        implementation_status='CPU data and recipe only; GPU launcher is not created or started by this script')
    sources=[Path(__file__),ROOT/'training/buyer_binding_data.py',ROOT/'training/buyer_latest_brakes_data.py',
        ROOT/'training/step84_market_report.py',MARKET/'events.jsonl',MARKET/'final.json',MARKET/'agents.jsonl',
        *[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)],
        ROOT/'training/data/buyer_binding_v1/evaluation_cases.json',
        *[ROOT/f'training/data/buyer_official_v1/{s}.jsonl' for s in official],*(ROOT/'adlmarket').glob('*.py')]
    manifest=dict(version='buyer_surgical_v1',cpu_only=True,model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        versions={name:version(name) for name in ('torch','transformers','peft','vllm')},
        seed=SEED,unique_replay=len(replay),unique_repairs=len(repairs),excluded_replay=excluded,
        total_exposures=80,replay_exposures=56,repair_exposures=24,replay_reuse=dict(Counter(rows[i]['id'] for b in batches for i in b if i<len(replay))),
        training_actions=dict(Counter(function(rows[i]['answer'])[0] for b in batches for i in b)),
        repair_families=dict(Counter(r['family'] for r in repairs)),
        train_groups=sorted({r['group'] for r in rows}),validation_groups=sorted({r['group'] for r in heldout}),
        evaluation_suites=dict(Counter(c['suite'] for c in evaluation)),
        mean_input_target_tokens=sum(len(r['input_ids']) for r in encoded)/len(encoded),
        max_input_target_tokens=max(len(r['input_ids']) for r in encoded),
        scheduled_input_target_tokens=sum(len(encoded[i]['input_ids']) for b in batches for i in b),
        replay_target_tokens=sum(encoded[i]['target_length'] for b in batches for i in b if i<len(replay)),
        repair_target_tokens=sum(encoded[i]['target_length'] for b in batches for i in b if i>=len(replay)),
        replay_think_tokens=dict(min=min(r['reasoning_tokens'] for r in encoded[:len(replay)]),max=max(r['reasoning_tokens'] for r in encoded[:len(replay)])),
        sources={str(p.resolve()):sha(p) for p in sources},
        parent_fingerprints={f:sha(PARENT/f) for f in ('adapter_config.json','adapter_model.safetensors')},
        limitations=['70/30 is sample exposure, not token share.','Only one real source market; no successful offer replay.',
                    'Previous six real market probes are now training-market seen, not held-out.',
                    'Twenty steps and causal claims about prior regressions are hypotheses, not guarantees.'])
    if set(manifest['train_groups'])&set(manifest['validation_groups']):raise ValueError('Source leakage')
    output.mkdir(parents=True)
    for name,obj in [('batch_plan.json',batches),('evaluation_cases.json',evaluation),('recipe.json',recipe),('manifest.json',manifest)]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
    for name,objects in [('train.jsonl',rows),('train.tokens.jsonl',encoded),('validation.jsonl',heldout)]:
        (output/name).write_text(''.join(dumps(r)+'\n' for r in objects))
    manifest['files']={p.name:sha(p) for p in output.iterdir() if p.name!='manifest.json'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    audit_folder(output)
    print(f'CPU prepared:{output};{len(replay)} faithful replay +24 repairs;80 exposures,20steps,70/30. Max tokens:{recipe["max_length"]}. No GPU work.',flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DATA)
    p.add_argument('--audit',type=Path,help='Read-only CPU hash, label, ledger and exposure audit')
    a=p.parse_args()
    if a.audit:
        result=audit_folder(a.audit);print(f'PASS CPU audit:{a.audit}; no GPU work.');return result
    return generate(a.output_dir)


if __name__=='__main__':main()
