#!/usr/bin/env python3
"""Official-only Buyer SFT data: source-faithful contexts, checked targets, group holdouts."""
import argparse
from collections import Counter,defaultdict
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE='Qwen/Qwen3.5-4B';MODEL_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
VERSION='buyer_official_sft_v1';MARKER='YOUR SITUATION (fixed for this market; your own numbers are private):'
ALLOWED={'buy_listed','accept_offer','send_message','post_to_room','post_review','wait'}
MULTI=re.compile(r'\b(?:[2-9]\d*|two|three|four|five|multiple|several)\s+(?:units?|pieces?|bottles?)\b|\b(?:second|third)\s+unit\b|#\s*[23]\b|\bbulk\b',re.I)


def dump(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cents(x):return int(Decimal(str(x))*100)
def unpack(x):return json.loads(x) if isinstance(x,str) else x


def read_official(folder):
    import pyarrow.parquet as pq
    folder=Path(folder);source=json.loads((folder/'source.json').read_text())
    for name,info in source['files'].items():
        if sha(folder/name)!=info['sha256']:raise ValueError(f'Official source changed: {name}')
    for batch in pq.ParquetFile(folder/'data/train-00000-of-00001.parquet').iter_batches(batch_size=32):
        for row in batch.to_pylist():
            row['messages']=[unpack(m) for m in row['messages']]
            for t in row['tools']:
                props=unpack(t['function']['parameters']['properties'])
                # Some source exports expand a union of argument fields and fill unused schemas with null.
                t['function']['parameters']['properties']={k:v for k,v in props.items() if v is not None}
            yield row


def situation(system):return json.loads(system.split(MARKER,1)[1])


def economic_group(sit):
    # Agent ordering/personas do not make two copies of the same economy independent.
    obj={'products':sorted(sit['products'],key=lambda p:p['product_id']),
         'budget':sit['your_budget'],'sellers':sorted(sit['sellers']),'buyers':sorted(sit['buyers']),
         'overview':sorted(sit.get('market_overview',[]),key=lambda p:p['product_id'])}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:24]


def normalize(message,historical=False):
    m=copy.deepcopy(message)
    if m['role']=='assistant':
        if historical:m.pop('reasoning_content',None)
        m['content']=m.get('content') or ''
        for call in m.get('tool_calls') or []:
            call['function']['arguments']=unpack(call['function']['arguments'])
            if not isinstance(call['function']['arguments'],dict):raise ValueError('Non-object tool arguments')
            call['function']['arguments']={k:v for k,v in call['function']['arguments'].items() if v is not None}
    return m


def schema_error(tool,args,tools):
    schema=next((t['function']['parameters'] for t in tools if t['function']['name']==tool),None)
    if schema is None:return 'unknown_tool'
    props=schema.get('properties',{})
    if set(schema.get('required',[]))-set(args) or set(args)-set(props):return 'argument_keys'
    for key,val in args.items():
        p=props[key];typ=p.get('type');ok=(isinstance(val,str) if typ=='string' else type(val)==int if typ=='integer' else type(val) in (int,float) if typ=='number' else isinstance(val,dict) if typ=='object' else True)
        if not ok:return 'argument_types'
        if 'enum' in p and val not in p['enum']:return 'argument_enum'
        if type(val) in (int,float) and (val<p.get('minimum',float('-inf')) or val>p.get('maximum',float('inf'))):return 'argument_range'
        if isinstance(val,str) and len(val)>min(p.get('maxLength',1000),1000):return 'argument_length'
    return None


def fixed_positive_available(obs,sit):
    varies={p['product_id'] for p in sit.get('market_overview',[]) if p.get('quality_varies')}
    for channel in ('board','offers_to_you'):
        for r in obs.get(channel,[]):
            pid=r['product_id'];qty=r.get('qty',1)
            if pid in varies or qty!=1 or obs['you']['units_bought'].get(pid,0):continue
            if channel=='board' and r['in_stock']<1:continue
            if channel=='offers_to_you' and r['open_until_round']<obs['round']:continue
            if cents(obs['you']['cash'])>=cents(r['price']) and cents(obs['you']['utility_of_next_unit'].get(pid,0))>cents(r['price']):return True
    return False


def low_multiplier(value):
    if isinstance(value,(int,float)):return float(value)
    match=re.search(r'(\d+(?:\.\d+)?)\s*(?:x|×)',str(value))
    if not match:raise ValueError('Cannot parse official low-grade multiplier')
    return float(match[1])


def target_error(name,args,obs,sit,result,next_obs):
    if name not in ALLOWED:return 'unsupported_current_tool'
    if not result or result.get('ok') is not True:return 'failed_or_missing_tool_result'
    if not 1<=obs['round']<=5:return 'after_round5'
    if name in ('send_message','post_to_room'):
        if obs['round']==5 and obs['turns_left_this_round']==1:return 'message_after_last_action'
        if MULTI.search(args.get('message','')):return 'explicit_multiunit_request'
    if name=='wait' and fixed_positive_available(obs,sit):return 'wait_with_affordable_fixed_positive'
    if name not in ('buy_listed','accept_offer'):return None
    if name=='buy_listed':
        trade=next((r for r in obs['board'] if (r['seller'],r['product_id'])==(args.get('seller_id'),args.get('product_id')) and r['in_stock']>=1),None)
        qty=args.get('qty',1)
    else:
        trade=next((r for r in obs.get('offers_to_you',[]) if r['offer_id']==args.get('offer_id') and r['open_until_round']>=obs['round']),None)
        qty=trade['qty'] if trade else 0
    if not trade:return 'missing_current_trade'
    if qty!=1:return 'multiunit_trade'
    pid=trade['product_id']
    if obs['you']['units_bought'].get(pid,0)>0:return 'already_owned_trade'
    price=cents(trade['price']);value=cents(obs['you']['utility_of_next_unit'].get(pid,0))
    if price>cents(obs['you']['cash']):return 'unaffordable_trade'
    if value<=price:return 'nonpositive_visible_trade'
    if next_obs:
        tid=result.get('data',{}).get('trade_id')
        received=next((t for t in next_obs.get('your_unreviewed_trades',[]) if t.get('trade_id',t.get('id'))==tid),None)
        if received and received.get('grade')=='low' and round(value*low_multiplier(sit.get('low_grade_utility',.4)))<=price:return 'observed_low_nonpositive_trade'
    return None


def encode(tokenizer,row,max_length=12288):
    answer=row['answer'];reason=answer['reasoning_content']
    nreason=len(tokenizer.encode(reason,add_special_tokens=False))
    if nreason>512:return None,'reasoning_over_512_tokens'
    messages=copy.deepcopy(row['messages']);dropped=0
    while True:
        prompt=tokenizer.apply_chat_template(messages,tools=row['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        full=tokenizer.apply_chat_template(messages+[answer],tools=row['tools'],tokenize=False,add_generation_prompt=False,enable_thinking=True)
        if not full.startswith(prompt):raise ValueError('Native template prefix mismatch')
        prefix=tokenizer.encode(prompt,add_special_tokens=False);ids=tokenizer.encode(full,add_special_tokens=False)
        if ids[:len(prefix)]!=prefix:raise ValueError('Native target token boundary mismatch')
        if len(ids)<=max_length:break
        starts=[i for i,m in enumerate(messages) if m['role']=='user']
        if len(starts)<2:return None,'system_and_current_overlength'
        messages=[messages[0],*messages[starts[1]:]];dropped+=1
    if len(ids)-len(prefix)>1024:return None,'completion_over_1024_tokens'
    # No synthetic reasoning and no separate tool-name weight. Whole native assistant target is supervised.
    row['messages']=messages;row['history_turns_dropped_for_length']=dropped
    encoded={k:row[k] for k in ('id','group','trajectory_id','label_action')}
    encoded.update(input_ids=ids,prompt_length=len(prefix),target_length=len(ids)-len(prefix),thinking=True,
                   reasoning_tokens=nreason,history_turns_shown=sum(m['role']=='user' for m in messages)-1)
    return encoded,None


def split_groups(rows,seed):
    pools=defaultdict(set)
    for row in rows:pools[len(row['situation']['products'])].add(row['group'])
    assignment={};rng=random.Random(seed)
    for shape,groups in sorted(pools.items()):
        order=sorted(groups);rng.shuffle(order)
        if len(order)<3:raise ValueError(f'Only {len(order)} economic groups for {shape} products; cannot make disjoint holdouts')
        n=max(1,len(order)//10)
        for split,part in (('validation',order[:n]),('test',order[n:2*n]),('train',order[2*n:])):
            for group in part:assignment[group]=split
    return assignment


def generate(raw,output,seed=20261120):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Refuse overwrite; use a NEW output directory')
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,local_files_only=True)
    retained=[];stats=Counter();roles=Counter();seen=set();thinking_modes=Counter();economies=set();raw_actions=Counter()
    for episode in tqdm(read_official(raw),total=2299,desc='CPU official trajectories',unit='trajectories'):
        roles[episode['role']]+=1
        if episode['role']!='buyer':continue
        ms=episode['messages'];sit=situation(ms[0]['content']);group=economic_group(sit);economies.add(group)
        steps=[];history=[];obs=None
        for i,m in enumerate(ms):
            if m['role']!='assistant':continue
            stats['buyer_decisions']+=1;raw_actions['reasoning' if m.get('reasoning_content') else 'no_reasoning']+=1
            if i==0 or ms[i-1]['role']!='user':stats['skip_nonstandard_turn_order']+=1;continue
            try:obs=unpack(ms[i-1]['content']);answer=normalize(m)
            except (ValueError,TypeError):stats['skip_malformed_observation_or_args']+=1;continue
            current_start=i-1
            past=[j for j in range(1,current_start) if ms[j]['role']=='user'][-6:]
            start=past[0] if past else current_start
            try:messages=[copy.deepcopy(ms[0]),*[normalize(x,historical=True) for x in ms[start:i]]]
            except (ValueError,TypeError):stats['skip_bad_history_arguments']+=1;continue
            calls=answer.get('tool_calls') or []
            if len(calls)!=1:stats['skip_not_one_native_call']+=1;continue
            fn=calls[0]['function'];name=fn['name'];args=fn['arguments']
            err=schema_error(name,args,episode['tools'])
            if err:stats['skip_'+err]+=1;continue
            result=None
            if i+1<len(ms) and ms[i+1]['role']=='tool':
                try:result=unpack(ms[i+1]['content'])
                except ValueError:pass
            next_obs=None
            for message in ms[i+1:i+4]:
                if message['role']=='user':
                    try:next_obs=unpack(message['content'])
                    except ValueError:pass
                    break
            err=target_error(name,args,obs,sit,result,next_obs)
            if err:stats['skip_'+err]+=1;continue
            reason=answer.get('reasoning_content')
            if not isinstance(reason,str) or not reason.strip():stats['skip_missing_real_reasoning_for_on']+=1;continue
            if len(reason)>8192:stats['skip_reasoning_over_8192_chars']+=1;continue
            key=hashlib.sha256(dump([messages,episode['tools'],answer]).encode()).hexdigest()
            if key in seen:stats['skip_exact_duplicate']+=1;continue
            row={'id':episode['id']+f'_turn{i}','trajectory_id':episode['id'],'group':group,
                 'label_action':name,'messages':messages,'tools':copy.deepcopy(episode['tools']),
                 'answer':answer,'observation':obs,'situation':sit,'source_message_index':i,
                 'source_outcome':episode['outcome'],'original_tool_result':result,'curriculum_version':VERSION}
            encoded,err=encode(tokenizer,row)
            if err:stats['skip_'+err]+=1;continue
            seen.add(key);row['encoded']=encoded;retained.append(row);stats['retained']+=1
    assignment=split_groups(retained,seed);output.mkdir(parents=True)
    manifests={};groups={split:[] for split in ('train','validation','test')}
    from contextlib import ExitStack
    with ExitStack() as stack:
        raw_files={s:stack.enter_context((output/f'{s}.jsonl').open('w')) for s in groups}
        tok_files={s:stack.enter_context((output/f'{s}.tokens.jsonl').open('w')) for s in groups}
        counts={s:Counter() for s in groups};lengths=defaultdict(list);history=defaultdict(Counter);trajectories=defaultdict(set)
        for row in retained:
            split=assignment[row['group']];encoded=row.pop('encoded');raw_files[split].write(dump(row)+'\n');tok_files[split].write(dump(encoded)+'\n')
            counts[split][row['label_action']]+=1;lengths[split].append(len(encoded['input_ids']));history[split][encoded['history_turns_shown']]+=1;trajectories[split].add(row['trajectory_id'])
    for group,split in assignment.items():groups[split].append(group)
    for split in groups:
        if not lengths[split]:raise ValueError('Empty official split')
        ordered=sorted(lengths[split]);manifests[split]={'raw':len(ordered),'tokenized':len(ordered),'actions':dict(counts[split]),'trajectories':len(trajectories[split]),
            'tokens':sum(ordered),'mean_tokens':round(sum(ordered)/len(ordered),1),'p50_tokens':ordered[len(ordered)//2],'p95_tokens':ordered[int(.95*len(ordered))],'max_tokens':max(ordered),
            'history_turns':dict(history[split]),'raw_sha256':sha(output/f'{split}.jsonl'),'tokens_sha256':sha(output/f'{split}.tokens.jsonl')}
    manifest={'version':VERSION,'base_model':BASE,'revision':MODEL_REVISION,'thinking':True,'seed':seed,'groups':groups,
        'source':json.loads((raw/'source.json').read_text()),'sources':{'training/official_buyer_data.py':sha(ROOT/'training/official_buyer_data.py')},
        'raw_roles':dict(roles),'raw_buyer_actions':dict(raw_actions),'economic_groups_before_filter':len(economies),'curation_counts':dict(stats),'splits':manifests,
        'supervision':'Official-only successful targets with actual teacher reasoning; completion CE; no invented examples/reasoning or auxiliary tool-name loss.',
        'domain_shift':'All official buyer utility tables are old multiunit worlds with variable horizon. Original system, situation, observations and persona remain source-faithful. Targets exclude owned/multiunit trades and rounds>5; not claimed identical to final grading.',
        'thinking_limits':{'teacher_reasoning_tokens':512,'assistant_target_tokens':1024,'max_sequence':12288},
        'split_unit':'Conservative public market signature: products/list prices, common budget, participant IDs and public overview. Private utility tables/personas do not split a shared market. Entire groups/trajectories stay in one split.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'curation':dict(stats),'splits':manifests},indent=2),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=ROOT/'training/data/official_raw');p.add_argument('--output',type=Path,default=ROOT/'training/data/buyer_official_v1')
    a=p.parse_args();generate(a.raw,a.output)
