#!/usr/bin/env python3
"""Engine-backed training counterfactuals for current seller and offer-ID membership."""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.buyer_operations_data import dump,assistant
from training.buyer_operations_v2_data import history_state,choose,file_sha
from training.buyer_reasoning_data import encode_row,money
from training.board_repair_data import board_target
from training.decision_ablation import fingerprint

VERSION='buyer_id_repair_v1'
DEFAULT_PARENT=ROOT/'training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032'


def board_unit(group,persona,mode,negative_kind,index):
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message
    from adlmarket.tools import dollars,to_cents
    market,situation,history,previous,note=history_state(group,persona)
    aid=situation['your_id'];pid=list(market.products)[1]
    old_buy=next(m['tool_calls'][0]['function'] for m in history
                 if m['role']=='assistant' and m['tool_calls'][0]['function']['name']=='buy_listed')
    old_seller=old_buy['arguments']['seller_id']
    other=next(s for s in market.sellers if s!=old_seller)
    sellers=[old_seller,other]
    ample=market.buyers[aid].cash
    if negative_kind=='value' and ample<=market.buyers[aid].utility[pid][0]+1:
        eligible=[p for p in market.products if p not in market.quality_products
                  and not market.buyers[aid].held.get(p) and market.buyers[aid].utility[p][0]+1<ample]
        if not eligible:raise ValueError('No affordable fixed-quality value control')
        pid=eligible[0]
    value=market.buyers[aid].utility[pid][0];price=max(2,value*2//5)
    if ample<price+(index%2):raise ValueError('Need affordable tight and ample controls')
    records=[]
    for seat,seller in enumerate(sellers):
        for condition in ('tight','ample','negative'):
            state=copy.deepcopy(market);state.listings.clear();buyer=state.buyers[aid]
            quote=price;buyer.cash=price+(index%2) if condition=='tight' else ample
            if condition=='negative':
                if negative_kind=='cash':buyer.cash=price-1 if (index//4)%2==0 else 0
                elif negative_kind=='owned':buyer.held[pid]=['high'];buyer.gained_utility+=value
                elif negative_kind=='stock':state.sellers[seller].stock[pid]=0
                elif negative_kind=='value':quote=value+1
                else:raise ValueError('Unknown negative kind')
            state.listings[seller,pid]=Listing(quote)
            obs=state.observe(aid,turns_left=6-(state.tick-1)%6)
            obs['output_tokens_left_this_round']=8192
            notices=[m for m in obs['new_messages'] if m['channel']=='notice']
            if not notices or obs['offers_to_you']:raise ValueError('Need real notices with no current binding offers')
            tool,args=choose(situation,obs);positive=condition!='negative'
            if tool!=('buy_listed' if positive else 'wait'):raise ValueError('Counterfactual teacher mismatch')
            check=copy.deepcopy(state);before=check.buyers[aid].gained_utility
            result=check.apply(aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
            if not result['ok']:raise ValueError(result)
            surplus=check.buyers[aid].gained_utility-before-(to_cents(result['data']['total']) if positive else 0)
            if positive and surplus<=0:raise ValueError('Positive counterfactual must realize surplus')
            wrong_seller=sellers[1-seat]
            wrong=copy.deepcopy(state).apply(aid,'buy_listed',{'seller_id':wrong_seller,'product_id':pid,'qty':1}).as_dict()
            if wrong['ok']:raise ValueError('Noncurrent seller must not be executable')
            fake=copy.deepcopy(state).apply(aid,'accept_offer',{'offer_id':notices[0]['id']}).as_dict()
            if fake['ok']:raise ValueError('Notice IDs must not be binding offers')
            if not positive:
                rejected=copy.deepcopy(state);before=rejected.buyers[aid].gained_utility
                attempted=rejected.apply(aid,'buy_listed',{'seller_id':seller,'product_id':pid,'qty':1}).as_dict()
                bad_surplus=rejected.buyers[aid].gained_utility-before-to_cents(attempted['data']['total']) if attempted['ok'] else None
                if attempted['ok'] and bad_surplus>=0:raise ValueError('Negative control must reject or lose surplus')
            else:attempted=None;bad_surplus=None
            original_system=history_state_system(situation,persona)
            pair=f'g{group}_board_id_p{persona+1}_{mode}_s{seat}_{condition}'
            row={'id':pair,'pair_id':pair,'group':group,'persona_index':persona+1,
                 'family':'board_id' if positive else f'board_{negative_kind}',
                 'cash_variant':condition,'economic_id':f'g{group}_board_id_{mode}_{condition}',
                 'task':'board_id_positive' if positive else 'board_id_negative','positive':positive,
                 'label_action':tool,'history_mode':mode,'curriculum_version':VERSION,'thinking':True,
                 'messages':[{'role':'system','content':original_system},
                             *([] if mode=='fresh' else copy.deepcopy(history)),
                             {'role':'user','content':observation_message(obs,note,previous)}],
                 'tools':list(state.tools['buyer'].values()),'answer':assistant(tool,args),
                 'observation':obs,'situation':situation,'engine_result':result,
                 'realized_surplus_cents':surplus,'historical_seller':old_seller,
                 'current_seller':seller,'current_seller_is_historical':seller==old_seller,
                 'wrong_seller_engine_check':wrong,'notice_as_offer_engine_check':fake,
                 'negative_trade_check':{'result':attempted,'realized_surplus_cents':bad_surplus},
                 'provenance':'training-market engine-backed current-seller counterfactual; real historical notices'}
            text=board_target(row)
            if obs['board']:
                first,rest=text.split('\n',1)
                text=first+'\nSOURCE: current board; offers_to_you_count=0; use this same seller/product tuple.\n'+rest
            else:text='BOARD: no current stocked row; offers_to_you_count=0. ACTION: wait.'
            row['answer']['reasoning_content']=text
            records.append(row)
    return records


def history_state_system(situation,persona):
    from adlmarket.prompts import system_prompt
    from training.buyer_reasoning_data import personas
    from adlmarket.config import MarketSpec
    return system_prompt('buyer',personas()[persona],situation,6,
                         quality=MarketSpec().schema_.quality,excess=True,round_tokens=8192)


def offer_replay(source,seen,seed):
    pairs=defaultdict(dict)
    with (source/'train.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['family'].startswith('offer'):pairs[r['pair_id']][r['positive']]=r
    rng=random.Random(seed);selected=[]
    families=sorted({s[True]['family'] for s in pairs.values()})
    for family in families:
        for mode in ('fresh','recorded'):
            available=[s for s in pairs.values() if s[True]['family']==family and s[True]['history_mode']==mode
                       and not any(r['id'] in seen for r in s.values())]
            rng.shuffle(available)
            for sides in available[:4]:
                for sign in (True,False):
                    r=copy.deepcopy(sides[sign]);r['curriculum_version']=VERSION
                    r['provenance']='unchanged original reasoning-v1 offer replay';selected.append(r)
    if len(selected)!=64:raise ValueError('Need 32 positive/negative offer replay pairs')
    return selected


def generate(source,output,parent,seed=20261122):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    source,output,parent=Path(source).resolve(),Path(output).resolve(),Path(parent).resolve()
    old=json.loads((source/'manifest.json').read_text());cfg=json.loads((parent.parent.parent/'train_config.json').read_text())
    if old['version']!='buyer_operations_reasoning_v1' or cfg['data_version']!='buyer_board_repair_v1':raise ValueError('Unexpected source/parent')
    if not json.loads((parent/'training_step.json').read_text())['thinking']:raise ValueError('Parent must be thinking ON')
    parent_manifest=json.loads((ROOT/cfg['data_dir']/'manifest.json').read_text())
    if file_sha(ROOT/cfg['data_dir']/'manifest.json')!=cfg['data_manifest_sha256']:
        raise ValueError('Parent data manifest changed')
    if parent_manifest['groups']!=old['groups']:raise ValueError('Training market split changed')
    for name,expected in old['sources'].items():
        if file_sha(ROOT/name)!=expected:raise ValueError('Frozen source changed')
    if file_sha(source/'train.jsonl')!=old['splits']['train']['raw_sha256']:
        raise ValueError('Offer replay source changed')
    seen=set()
    for run in (parent.parent.parent,Path(parent_manifest['parent_adapter']).parent.parent):
        for rank in (0,1):
            seen.update(json.loads(s)['id'] for s in (run/f'samples_rank{rank}.jsonl').read_text().splitlines())
    order=list(old['groups']['train']);random.Random(seed).shuffle(order)
    records=[]
    for index,g in enumerate(tqdm(order[:16],desc='CPU ID counterfactual markets',unit='markets')):
        group=old['seed']+1009*g;persona=index%3
        if (group+persona)%3==0:persona=(persona+1)%3
        for mode in ('fresh','recorded'):
            records.extend(board_unit(group,persona,mode,('cash','owned','stock','value')[index%4],index))
    records.extend(offer_replay(source,seen,seed))
    if len(records)!=256 or len({r['id'] for r in records})!=256:raise ValueError('Expected 256 unique sample IDs')
    if Counter(r['label_action'] for r in records)!=Counter({'buy_listed':128,'accept_offer':32,'wait':96}):raise ValueError('Action mix changed')
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    lengths=[];rationales=[]
    with (output/'train.jsonl').open('w') as raw,(output/'train.tokens.jsonl').open('w') as tokens:
        for r in tqdm(records,desc='CPU encode ID repair',unit='examples'):
            encoded=encode_row(tokenizer,r)
            if len(encoded['input_ids'])>12288:raise ValueError('Overlength ID example')
            raw.write(dump(r)+'\n');tokens.write(dump(encoded)+'\n')
            lengths.append(len(encoded['input_ids']));rationales.append(len(encoded['reasoning_positions']))
    splits={'train':{'raw':256,'tokenized':256,'actions':dict(Counter(r['label_action'] for r in records)),
                     'max_tokens':max(lengths),'max_reasoning_tokens':max(rationales),
                     'raw_sha256':file_sha(output/'train.jsonl'),'tokens_sha256':file_sha(output/'train.tokens.jsonl')}}
    for split in ('validation','test'):
        for suffix,key in (('jsonl','raw_sha256'),('tokens.jsonl','tokens_sha256')):
            if file_sha(source/f'{split}.{suffix}')!=old['splits'][split][key]:raise ValueError('Frozen holdout changed')
            shutil.copy2(source/f'{split}.{suffix}',output/f'{split}.{suffix}')
        splits[split]=old['splits'][split]
    shutil.copytree(source/'personas',output/'personas');shutil.copy2(source/'cash_probe_cases.json',output/'cash_probe_cases.json')
    (output/'agent.yaml').write_text('reasoning: true\n')
    sources={**old['sources'],'training/board_repair_data.py':file_sha(ROOT/'training/board_repair_data.py'),
             'training/id_repair_data.py':file_sha(ROOT/'training/id_repair_data.py')}
    manifest={'version':VERSION,'base_model':cfg['base_model'],'revision':cfg['revision'],'thinking':True,
              'seed':old['seed'],'selection_seed':seed,'groups':old['groups'],'sources':sources,'splits':splits,
              'parent_adapter':str(parent),'parent_fingerprints':fingerprint(parent),
              'cash_probe_sha256':file_sha(output/'cash_probe_cases.json'),
              'histories':dict(Counter(r['history_mode'] for r in records)),
              'personas':dict(Counter(r['persona_index'] for r in records)),
              'note':'Engine-backed training-only state counterfactuals and offer replay; same raw JSON schema/personas; '
                     'holdouts copied unchanged. Programmatic SFT, not model-generated preference data.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'CPU ID repair ready: {output}; 256 examples; historical and new sellers both valid when currently listed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'training/data/buyer_reasoning_v1')
    p.add_argument('--output',type=Path,default=ROOT/'training/data/buyer_id_repair_v1')
    p.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    a=p.parse_args();generate(a.source,a.output,a.parent)
