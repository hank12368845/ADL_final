#!/usr/bin/env python3
"""CPU-only availability audit of official seller data; no training or GPU calls."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.official_buyer_data import read_official,situation,unpack,schema_error,cents


def audit():
    counts=Counter();actions=Counter();eligible=Counter();problems=Counter();groups=set();shapes=Counter();trajectories=0
    for row in read_official(ROOT/'training/data/official_raw'):
        if row['role']!='seller':continue
        trajectories+=1;messages=row['messages'];sit=situation(messages[0]['content']);shapes[len(sit['products'])]+=1
        public={k:sit.get(k) for k in ['products','sellers','buyers','market_overview']}
        groups.add(hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest())
        obs=None
        for i,m in enumerate(messages):
            if m['role']=='user':
                try:obs=unpack(m['content'])
                except (ValueError,TypeError):obs=None
            if m['role']!='assistant':continue
            counts['assistant_decisions']+=1;calls=m.get('tool_calls') or []
            if len(calls)!=1 or not obs:problems['missing_observation_or_single_tool']+=1;continue
            call=calls[0];name=call['function']['name'];actions[name]+=1
            try:
                args={k:v for k,v in unpack(call['function']['arguments']).items() if v is not None}
                error=schema_error(name,args,row['tools'])
            except (ValueError,TypeError,KeyError,AttributeError):error='malformed_arguments'
            if not error:
                result=next((unpack(n['content']) for n in messages[i+1:i+3]
                             if n['role']=='tool' and n.get('tool_call_id')==call['id']),None)
                if not result or result.get('ok') is not True:error='unsuccessful_or_missing_result'
            if not error and (name not in {'set_price','make_offer','send_message','post_to_room','wait'} or obs.get('round',99)>5):
                error='outside_current_tools_or_horizon'
            if not error and name in {'set_price','make_offer'}:
                pid=args.get('product_id');grade=args.get('grade','high');stock=obs.get('you',{}).get('stock',{}).get(pid,0)
                cost_table=sit.get('your_low_grade_unit_cost',{}) if grade=='low' else sit.get('your_unit_cost',{})
                if pid not in cost_table or stock<1:error='no_stock_or_invalid_grade_cost'
                elif cents(args.get('price',0))<=cents(cost_table[pid]):error='nonpositive_unit_margin'
                elif name=='make_offer' and (args.get('buyer_id') not in sit['buyers'] or args.get('qty',1)>stock):
                    error='offer_buyer_or_quantity'
            if error:problems[error]+=1;continue
            counts['schema_success_current_horizon_margin_candidates']+=1;eligible[name]+=1
            if (m.get('reasoning_content') or '').strip():counts['candidates_with_native_reasoning']+=1
    return dict(cpu_only=True,source=json.loads((ROOT/'training/data/official_raw/source.json').read_text()),
        seller_trajectories=trajectories,public_market_groups_before_split=len(groups),
        product_count_distribution_trajectories=dict(shapes),counts=dict(counts),raw_actions=dict(actions),
        candidate_actions=dict(eligible),exclusion_counts=dict(problems),limits=[
            'Availability audit only; not a tokenized training set or trained model.',
            'Source rules include older market horizons and other differences; current seller compatibility needs further review.',
            'Reasoning presence does not establish its length or correctness.',
            'Success/margin candidates do not prove high-profit strategy.',
            'No training or GPU inference was run.'])


if __name__=='__main__':
    result=audit();output=ROOT/'docs/SELLER_CHECKPOINT1_DATA_AUDIT.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source','limits']},ensure_ascii=False,indent=2))
    print(f'Saved:{output}; CPU audit only.')
