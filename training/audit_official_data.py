#!/usr/bin/env python3
"""Reproducible CPU inventory of the pinned official dataset before any model training."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.official_buyer_data import read_official,situation,economic_group


def audit(raw):
    roles=Counter();actions=defaultdict(Counter);modes=defaultdict(Counter);shapes=defaultdict(Counter)
    groups=defaultdict(set);economic_roles=defaultdict(set);reason_sizes=[]
    for row in read_official(raw):
        role=row['role'];roles[role]+=1;ms=row['messages'];assist=[m for m in ms if m['role']=='assistant'];thought=sum(bool(m.get('reasoning_content')) for m in assist)
        modes[role]['no_thinking' if thought==0 else 'all_thinking' if thought==len(assist) else 'mixed']+=1
        if role=='buyer':
            sit=situation(ms[0]['content']);n=len(sit['products']);shapes[n]['trajectories']+=1;shapes[n]['single_unit_tables']+=all(len(x)==1 for x in sit['your_utility_per_unit'].values());shapes[n]['fixed_horizon']+=isinstance(sit['rounds'],int)
            shapes[n]['private_values']+='utilities are private and differ' in ms[0]['content'];groups[n].add(economic_group(sit))
        for i,m in enumerate(ms):
            if m['role']!='assistant':continue
            actions[role]['decisions']+=1;calls=m.get('tool_calls') or [];actions[role]['real_reasoning']+=bool(m.get('reasoning_content'))
            if role=='buyer' and m.get('reasoning_content'):reason_sizes.append(len(m['reasoning_content']))
            if len(calls)!=1:actions[role]['not_one_native_call']+=1;continue
            actions[role]['tool_'+calls[0]['function']['name']]+=1
            if i+1<len(ms) and ms[i+1]['role']=='tool':
                result=json.loads(ms[i+1]['content']);actions[role]['tool_success']+=result.get('ok') is True
                if not result.get('ok'):actions[role]['error_'+str(result.get('error'))]+=1
            else:actions[role]['missing_tool_result']+=1
    ordered=sorted(reason_sizes)
    return {'roles':dict(roles),'actions':{r:dict(v) for r,v in actions.items()},'trajectory_modes':{r:dict(v) for r,v in modes.items()},
        'buyer_market_shapes':{str(n):dict(v) for n,v in shapes.items()},'conservative_public_groups':{str(n):len(v) for n,v in groups.items()},
        'buyer_reasoning_chars':{'min':ordered[0],'p50':ordered[len(ordered)//2],'p95':ordered[int(.95*len(ordered))],'max':ordered[-1]},
        'findings':['Official data cover genuine multi-candidate markets and negotiation, unlike previous one-candidate synthetic training.',
                    'All buyer source utility tables allow multiple marginal units and all source horizons are variable; none is an exact final-rule episode.',
                    'Some buyers have private values. Grouping by private utility alone would risk splitting copies of one public market.',
                    'Source tool properties/arguments contain null union padding in some exports; the loader removes null schema fields and null arguments.',
                    'No official training run or model quality claim follows from this inventory.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=ROOT/'training/data/official_raw');p.add_argument('--output',type=Path,default=ROOT/'docs/official_dataset_audit.json');a=p.parse_args()
    result=audit(a.raw);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
