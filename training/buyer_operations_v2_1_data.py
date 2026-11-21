#!/usr/bin/env python3
"""CPU cash-boundary curriculum with paired board/offers and both history contexts."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.buyer_operations_data import BASE,REVISION,personas,assistant,dump
from training.buyer_operations_v2_data import (FAMILIES,history_state,make_pair as v2_pair,
    choose,file_sha,encode_record,diagnostic_case)

VERSION='buyer_operations_v2_1'
VARIANTS=('equal','above_one','wide','zero')
SOURCES=('training/buyer_operations_v2_1_data.py','training/buyer_operations_v2_data.py',
         'training/buyer_operations_data.py','training/tool_supervision.py')


def contexts(records):
    result=[]
    for mode in ('fresh','recorded'):
        for source in records:
            row=copy.deepcopy(source)
            row['pair_id']=source['pair_id']+'_'+mode
            row['id']=row['pair_id']+('_positive' if row['positive'] else '_negative')
            row['history_mode']=mode;row['curriculum_version']=VERSION
            if mode=='fresh':row['messages']=[row['messages'][0],row['messages'][-1]]
            result.append(row)
    return result


def cash_pair(group,persona,channel,variant):
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message,system_prompt
    from adlmarket.tools import dollars,to_cents
    if channel not in ('board','offer') or variant not in VARIANTS:raise ValueError('Invalid cash control')
    market,sit,history,previous,note=history_state(group,persona)
    aid=sit['your_id'];pid=list(market.products)[1];seller=list(market.sellers)[0]
    value=market.buyers[aid].utility[pid][0]
    # Same financial terms across channel, persona and history treatment.
    price=int(value*random.Random(f'{group}:cash_price').uniform(.25,.65))
    ample=market.buyers[aid].cash
    positive_cash={'equal':price,'above_one':price+1,'wide':ample,'zero':ample}[variant]
    negative_cash={'equal':price-1,'above_one':price-1,'wide':max(1,price//2),'zero':0}[variant]
    if not 0<=negative_cash<price<=positive_cash:raise ValueError('Cash boundary ordering failed')
    prompt=system_prompt('buyer',personas()[persona],sit,6,
        quality=market.spec.schema_.quality,excess=True,round_tokens=8192)
    family=f'{channel}_cash';pair=f'g{group}_{family}_{variant}_p{persona+1}'
    rows=[]
    for positive,cash in ((True,positive_cash),(False,negative_cash)):
        state=copy.deepcopy(market);state.buyers[aid].cash=cash
        if channel=='board':
            state.listings[seller,pid]=Listing(price)
            attempt=('buy_listed',{'seller_id':seller,'product_id':pid,'qty':1})
        else:
            made=state.apply(seller,'make_offer',{'buyer_id':aid,'product_id':pid,'price':dollars(price),'qty':1}).as_dict()
            if not made['ok']:raise ValueError(made)
            attempt=('accept_offer',{'offer_id':made['data']['offer_id']})
        obs=state.observe(aid,turns_left=6-(state.tick-1)%6);obs['output_tokens_left_this_round']=8192
        tool,args=choose(sit,obs)
        if tool!=(attempt[0] if positive else 'wait'):raise ValueError('Cash teacher mismatch')
        check=copy.deepcopy(state);before=check.buyers[aid].gained_utility
        outcome=check.apply(aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
        if not outcome['ok']:raise ValueError(outcome)
        surplus=check.buyers[aid].gained_utility-before-(to_cents(outcome['data']['total']) if positive else 0)
        if positive and surplus<=0:raise ValueError('No positive cash-control surplus')
        negative=None
        if not positive:
            rejected=copy.deepcopy(state).apply(aid,*attempt).as_dict()
            if rejected.get('error')!='insufficient_cash':raise ValueError(rejected)
            negative={'result':rejected,'realized_surplus_cents':None}
        rows.append({'id':pair+('_positive' if positive else '_negative'),'pair_id':pair,
            'group':group,'persona_index':persona+1,'family':family,'cash_variant':variant,
            'economic_id':f'g{group}_cash_{variant}','positive':positive,'label_action':tool,
            'task':f'{family}_{variant}_{"positive" if positive else "negative"}',
            'boundary':variant in ('equal','above_one'),'history_mode':'recorded',
            'messages':[{'role':'system','content':prompt},*copy.deepcopy(history),
                        {'role':'user','content':observation_message(obs,note,previous)}],
            'tools':list(state.tools['buyer'].values()),'answer':assistant(tool,args),
            'observation':obs,'situation':sit,'engine_result':outcome,
            'realized_surplus_cents':surplus,'negative_trade_check':negative,'teacher_version':'v2',
            'provenance':'original-engine v2.1 cash controls; current-state interventions, not complete rollouts'})
    return rows


def make_records(group,family,persona,cash_variant=None):
    if family.endswith('cash'):
        rows=cash_pair(group,persona,family.split('_')[0],cash_variant or VARIANTS[group%4])
    else:
        rows=v2_pair(group,family,persona,0.)
        for row in rows:row.update(cash_variant='standard',economic_id=row['pair_id'])
    return contexts(rows)


def splits():
    seed=20261119;order=list(range(40));random.Random(seed).shuffle(order)
    return seed,{'validation':order[:4],'test':order[4:8],'train':order[8:]}


def encode_row(tokenizer,row):
    encoded=encode_record(tokenizer,row)
    encoded.update({k:row[k] for k in ('cash_variant','economic_id','curriculum_version')})
    return encoded


def legacy_probe():
    seed,groups=splits();result=[]
    for g in groups['validation'][:2]:
        for family in ('board_cash','board_value','offer_cash','offer_value'):
            result.extend(diagnostic_case(r) for r in v2_pair(seed+1009*g,family,0,.4))
    original=ROOT/'training/data/buyer_operations_v2/probe_cases.json'
    if original.exists() and json.loads(original.read_text())!=result:
        raise ValueError('Legacy regression probe changed')
    return result


def generate(output):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    seed,groups=splits()
    reference=ROOT/'training/data/buyer_operations_v2/manifest.json'
    if reference.exists():
        old=json.loads(reference.read_text())
        if old['seed']!=seed or old['groups']!=groups:raise ValueError('V2 split boundaries changed')
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
    manifest={'version':VERSION,'base_model':BASE,'revision':REVISION,'thinking':False,
        'seed':seed,'groups':groups,'sources':{s:file_sha(ROOT/s) for s in SOURCES},'splits':{},
        'max_length':12288,'source':'synthetic original-engine controls; no TA adapters or private API',
        'history_contexts':'both fresh and recorded for every pair',
        'split_unit':'same pre-expansion market split as v2; no validation/test training',
        'cash_variants':list(VARIANTS),'note':'One variant per market, shared across persona/channel; all four variants covered in each split and the targeted probe.'}
    (output/'personas').mkdir()
    for i,text in enumerate(personas(),1):(output/'personas'/f'persona_{i}.md').write_text(text+'\n')
    for split,indices in groups.items():
        counts=Counter();modes=Counter();variants=Counter();lengths=[]
        raw=output/f'{split}.jsonl';tokens=output/f'{split}.tokens.jsonl'
        with raw.open('w') as a,tokens.open('w') as b:
            jobs=[(g,f,p) for g in indices for f in FAMILIES for p in range(3)]
            for g,f,p in tqdm(jobs,desc=f'CPU v2.1 {split}',unit='bundles'):
                variant=None if split=='train' else VARIANTS[indices.index(g)%4]
                for row in make_records(seed+1009*g,f,p,variant):
                    encoded=encode_row(tokenizer,row)
                    if len(encoded['input_ids'])>12288:raise ValueError(f'Overlength: {row["id"]}')
                    a.write(dump(row)+'\n');b.write(dump(encoded)+'\n')
                    counts[row['label_action']]+=1;modes[row['history_mode']]+=1
                    variants[row['cash_variant']]+=1;lengths.append(len(encoded['input_ids']))
        manifest['splits'][split]={'raw':len(lengths),'tokenized':len(lengths),'actions':dict(counts),
            'contexts':dict(modes),'cash_variants':dict(variants),'max_tokens':max(lengths),
            'raw_sha256':file_sha(raw),'tokens_sha256':file_sha(tokens)}
    legacy=legacy_probe();targeted=[];group=seed+1009*groups['validation'][2]
    for channel in ('board','offer'):
        for variant in VARIANTS:
            for r in contexts(cash_pair(group,0,channel,variant)):
                case=diagnostic_case(r)
                case['origin'].update({k:r[k] for k in ('cash_variant','history_mode','economic_id')})
                targeted.append(case)
    (output/'legacy_probe_cases.json').write_text(json.dumps(legacy,ensure_ascii=False,indent=1))
    (output/'cash_probe_cases.json').write_text(json.dumps(targeted,ensure_ascii=False,indent=1))
    manifest.update(legacy_probe_cases=len(legacy),cash_probe_cases=len(targeted),
                    legacy_probe_sha256=file_sha(output/'legacy_probe_cases.json'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'training/data/buyer_operations_v2_1')
    generate(parser.parse_args().output_dir)
