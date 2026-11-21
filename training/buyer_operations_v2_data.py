#!/usr/bin/env python3
"""Paired operation decisions, original-engine checks, balanced histories; CPU only."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.buyer_operations_data import BASE, REVISION, fresh_market, choose as _choose_legacy, personas, assistant, dump

FAMILIES = ('board_cash', 'board_value', 'board_owned', 'board_stock',
            'offer_cash', 'offer_value', 'offer_owned', 'offer_qty')
SOURCES = ('training/buyer_operations_v2_data.py', 'training/buyer_operations_data.py',
           'training/tool_supervision.py')


def choose(situation, obs):
    # The legacy helper passes cash through the engine's strictly-positive PRICE
    # converter. Zero cash is valid buyer state, so handle it without changing v1.
    if obs['you']['cash']==0:
        bad=sorted({t['seller'] for t in obs.get('your_unreviewed_trades',[]) if t.get('grade')=='low'})
        return 'wait',{'private_state':dump({'bad':bad,'pending':None,'last_buy':None})}
    return _choose_legacy(situation,obs)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def history_state(group, persona):
    """Real engine buy, accept and wait history shared by both sides of a pair."""
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message
    from adlmarket.tools import dollars
    market, _ = fresh_market(group)
    aid = f'buyer_{persona}'
    situation = market.situation(aid)
    current = market.tick
    sellers, products = list(market.sellers), list(market.products)
    history = []; previous = None; note = ''
    kinds=('buy_listed','accept_offer','wait'); rotation=(group+persona)%len(kinds)
    kinds=kinds[rotation:]+kinds[:rotation]
    for i, kind in enumerate(kinds):
        market.tick = current-3+i; market.round = (market.tick-1)//6+1
        market.listings.clear()
        pid, seller = products[2 if kind=='buy_listed' else 4], sellers[i%2]
        price = max(1, market.buyers[aid].utility[pid][0]//20)
        if kind=='buy_listed':
            market.listings[seller,pid] = Listing(price)
            args = {'seller_id':seller,'product_id':pid,'qty':1}
        elif kind=='accept_offer':
            result = market.apply(seller,'make_offer',{'buyer_id':aid,'product_id':pid,'price':dollars(price),'qty':1}).as_dict()
            if not result['ok']: raise ValueError(result)
            args = {'offer_id':result['data']['offer_id']}
        else: args = {}
        obs = market.observe(aid,turns_left=6-(market.tick-1)%6)
        obs['output_tokens_left_this_round'] = 8192
        expected, checked = choose(situation,obs)
        if expected!=kind: raise ValueError(f'History action is not the visible reference: {expected}/{kind}')
        # Use the visible teacher's correct minimal memory, with actual engine outcomes.
        args['private_state'] = checked['private_state']
        before = observation_message(obs,note,previous)
        result = market.apply(aid,kind,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
        if not result['ok']: raise ValueError(result)
        call_id = f'past_{i}'
        history.extend([{'role':'user','content':before}, assistant(kind,args,call_id),
                        {'role':'tool','tool_call_id':call_id,'content':dump(result)}])
        previous, note = result,args['private_state']
    market.tick = current; market.round = (current-1)//6+1
    market.listings.clear()
    return market,situation,history,previous,note


def make_pair(group, family, persona, history_dropout=.4):
    """Counterfactual current-state fixtures; negative labels never enter the prompt."""
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message, system_prompt
    from adlmarket.tools import dollars, to_cents
    if family not in FAMILIES: raise ValueError(family)
    market,situation,history,previous,note = history_state(group,persona)
    rng = random.Random(f'{group}:{family}:{persona}')
    aid = situation['your_id']; pid = list(market.products)[1]
    seller = list(market.sellers)[rng.randrange(5)]
    if family.endswith('value') and market.buyers[aid].utility[pid][0]>=market.buyers[aid].cash:
        eligible=[p for p in market.products if not market.schema.plan(p).quality_varies
                  and not market.buyers[aid].held.get(p) and market.buyers[aid].utility[p][0]<market.buyers[aid].cash]
        if not eligible: raise ValueError('No product for an affordable negative-value control')
        pid=rng.choice(eligible)
    value = market.buyers[aid].utility[pid][0]
    price = max(2, int(value*(.65 if family=='offer_qty' else rng.uniform(.25,.55))))
    boundary = family.endswith(('cash','value')) and rng.random()<.25
    margin = 1 if boundary else max(100,int(value*rng.uniform(.10,.35)))
    if family.endswith('value'):
        margin=min(margin,market.buyers[aid].cash-value)
        boundary=margin==1
    drop = rng.random()<history_dropout
    prompt = system_prompt('buyer',personas()[persona],situation,6,
                           quality=market.spec.schema_.quality,excess=True,round_tokens=8192)
    pair_id = f'g{group}_{family}_p{persona+1}'
    records = []
    for positive in (True,False):
        test = copy.deepcopy(market); buyer = test.buyers[aid]
        quote = price if positive or not family.endswith('value') else value+margin
        qty = 2 if not positive and family=='offer_qty' else 1
        if not positive:
            if family.endswith('cash'): buyer.cash = max(0,price-margin)
            if family.endswith('owned'):
                buyer.held[pid]=['high']; buyer.gained_utility += value
            if family=='board_stock': test.sellers[seller].stock[pid]=0
        if family.startswith('board'):
            test.listings[seller,pid] = Listing(quote)
            attempted = ('buy_listed',{'seller_id':seller,'product_id':pid,'qty':1})
        else:
            made = test.apply(seller,'make_offer',{'buyer_id':aid,'product_id':pid,'price':dollars(quote),'qty':qty}).as_dict()
            if not made['ok']: raise ValueError(made)
            attempted = ('accept_offer',{'offer_id':made['data']['offer_id']})
        obs = test.observe(aid,turns_left=6-(test.tick-1)%6)
        obs['output_tokens_left_this_round'] = 8192
        tool,args = choose(situation,obs)
        target = attempted[0] if positive else 'wait'
        if tool!=target: raise ValueError(f'{pair_id}/{positive}: expected {target}, got {tool}')
        check = copy.deepcopy(test); before = check.buyers[aid].gained_utility
        result = check.apply(aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
        if not result['ok']: raise ValueError(result)
        paid = to_cents(result['data']['total']) if positive else 0
        surplus = check.buyers[aid].gained_utility-before-paid
        if positive and surplus<=0: raise ValueError('Positive label has no realized surplus')
        negative_check = None
        if not positive:
            check=copy.deepcopy(test); before=check.buyers[aid].gained_utility
            bad=check.apply(aid,*attempted).as_dict()
            net=check.buyers[aid].gained_utility-before-to_cents(bad['data']['total']) if bad['ok'] else None
            if bad['ok'] and net>=0: raise ValueError('Negative trade is economically profitable')
            if family.endswith('value') and not bad['ok']:
                raise ValueError('Negative-value controls must be legal and affordable, isolating surplus')
            negative_check={'result':bad,'realized_surplus_cents':net}
        messages = [{'role':'system','content':prompt}, *([] if drop else copy.deepcopy(history)),
                    {'role':'user','content':observation_message(obs,note,previous)}]
        records.append({'id':f'{pair_id}_{"positive" if positive else "negative"}',
            'pair_id':pair_id,'group':group,'persona_index':persona+1,'family':family,
            'task':f'{family}_{"positive" if positive else "negative"}', 'positive':positive,
            'label_action':tool, 'history_mode':'fresh' if drop else 'recorded',
            'boundary':boundary,'messages':messages,'tools':list(test.tools['buyer'].values()),
            'answer':assistant(tool,args),'observation':obs,'situation':situation,
            'engine_result':result,'realized_surplus_cents':surplus,
            'negative_trade_check':negative_check,
            'teacher_version':'v2',
            'provenance':'synthetic_original_engine_counterfactual_v2; fixed-quality decision curriculum'})
    return records


def encode_record(tokenizer, record):
    from training.tool_supervision import encode_target
    fn=record['answer']['tool_calls'][0]['function']
    encoded=encode_target(tokenizer,record['messages'],record['tools'],fn['name'],fn['arguments'])
    return {**{k:record[k] for k in ('id','pair_id','group','task','family','positive','label_action','history_mode','persona_index')},
            **{k:encoded[k] for k in ('input_ids','prompt_length','target_length','tool_name_positions')}}


def diagnostic_case(record):
    return {'name':record['id'],'task':record['task'],'origin':{'split':'validation','group':record['group'],'pair_id':record['pair_id']},
            **{k:record[k] for k in ('messages','tools','observation','situation')},
            'contexts':['recorded'],'state_mode':'operations','teacher_version':'v2',
            'expected_tool':record['label_action']}


def generate(output,groups=40,seed=20261119,history_dropout=.4,max_length=12288):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    if groups<20 or not 0<=history_dropout<=1: raise ValueError('Need >=20 groups and dropout in [0,1]')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
    order=list(range(groups)); random.Random(seed).shuffle(order)
    n=max(2,groups//10)
    assignments={'validation':order[:n],'test':order[n:2*n],'train':order[2*n:]}
    manifest={'version':'buyer_operations_v2','base_model':BASE,'revision':REVISION,'thinking':False,
              'seed':seed,'groups':assignments,'split_unit':'market before persona/pair expansion',
              'history_dropout':history_dropout,'max_length':max_length,
              'sources':{p:file_sha(ROOT/p) for p in SOURCES},'splits':{},
              'curriculum':'fixed-quality, original engine; controlled cash/ownership/quote/stock interventions, not full rollouts',
              'raw_action_mix':'buy:accept:wait = 1:1:2; sampler uses 4:2:2 per global batch'}
    (output/'personas').mkdir()
    for i,p in enumerate(personas(),1): (output/'personas'/f'persona_{i}.md').write_text(p+'\n')
    probes=[]; full_validation=[]
    for split,indices in assignments.items():
        counts=Counter(); modes=Counter(); boundaries=Counter(); history_tools=Counter(); last_history_tools=Counter(); lengths=[]
        raw_path=output/f'{split}.jsonl'; token_path=output/f'{split}.tokens.jsonl'
        jobs=[(g,f,p) for g in indices for f in FAMILIES for p in range(3)]
        with raw_path.open('w') as raw, token_path.open('w') as tokens:
            for g,f,p in tqdm(jobs,desc=f'CPU v2 {split} pairs',unit='pairs'):
                for record in make_pair(seed+1009*g,f,p,history_dropout):
                    encoded=encode_record(tokenizer,record)
                    if len(encoded['input_ids'])>max_length:
                        raise ValueError(f'Overlength pair {record["id"]}; never drop only one side or truncate facts')
                    raw.write(dump(record)+'\n'); tokens.write(dump(encoded)+'\n')
                    counts[record['label_action']]+=1; modes[record['history_mode']]+=1
                    boundaries['boundary' if record['boundary'] else 'broad_or_nonprice']+=1
                    lengths.append(len(encoded['input_ids']))
                    past=[]
                    for message in record['messages']:
                        if message['role']=='assistant':
                            action=message['tool_calls'][0]['function']['name']
                            history_tools[action]+=1;past.append(action)
                    if past:last_history_tools[past[-1]]+=1
                    if split=='validation' and p==0:
                        case=diagnostic_case(record); full_validation.append(case)
                        if g in indices[:2] and f in ('board_cash','board_value','offer_cash','offer_value'):
                            probes.append(case)
        manifest['splits'][split]={'raw':sum(counts.values()),'tokenized':len(lengths),'actions':dict(counts),
              'contexts':dict(modes),'history_tools':dict(history_tools),'last_history_tools':dict(last_history_tools),'boundaries':dict(boundaries),
              'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),
              'raw_sha256':file_sha(raw_path),'tokens_sha256':file_sha(token_path)}
    (output/'probe_cases.json').write_text(json.dumps(probes,ensure_ascii=False,indent=1))
    (output/'validation_cases.json').write_text(json.dumps(full_validation,ensure_ascii=False,indent=1))
    manifest['probe_cases']=len(probes);manifest['validation_cases']=len(full_validation)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'training/data/buyer_operations_v2')
    parser.add_argument('--groups',type=int,default=40);parser.add_argument('--seed',type=int,default=20261119)
    parser.add_argument('--history-dropout',type=float,default=.4)
    args=parser.parse_args()
    generate(args.output_dir,args.groups,args.seed,args.history_dropout)


if __name__=='__main__':main()
