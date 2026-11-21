#!/usr/bin/env python3
"""CPU board-first SFT repair: training-only states, unchanged observations/personas."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.buyer_reasoning_data import cents,money,encode_row,personas
from training.buyer_operations_v2_data import choose,file_sha
from training.buyer_operations_data import dump
from training.decision_ablation import fingerprint

VERSION='buyer_board_repair_v1'
DEFAULT_PARENT=ROOT/'training/runs/20261119-020532_buyer_operations_acyaikwa/checkpoints/step_0032'


def board_target(row):
    obs=row['observation'];you=obs['you']
    rows=[b for b in obs.get('board',[]) if b['in_stock']>=1]
    tool,args=choose(row['situation'],obs)
    if tool!=row['label_action'] or args!=row['answer']['tool_calls'][0]['function']['arguments']:
        raise ValueError('Visible teacher disagrees with the original engine-checked target')
    if not rows:
        if tool!='wait':raise ValueError('No board candidate must not buy')
        return 'BOARD: no current listing with stock>=1. CHECK: no available board target; ACTION: wait.'
    if len(rows)!=1 or obs.get('offers_to_you'):raise ValueError('Expected one board-only candidate')
    b=rows[0];pid=b['product_id'];seller=b['seller'];price=cents(b['price']);cash=cents(you['cash'])
    value=cents(you['utility_of_next_unit'].get(pid,0));owned=you['units_bought'].get(pid,0)
    if pid in {p['product_id'] for p in row['situation']['market_overview'] if p['quality_varies']}:
        raise ValueError('This repair curriculum isolates fixed-quality execution')
    expected='buy_listed' if not owned and cash>=price and value>price else 'wait'
    if tool!=expected:raise ValueError('Board arithmetic/ownership does not explain the label')
    action=f'buy_listed(seller_id={seller}, product_id={pid}, qty=1)' if tool=='buy_listed' else 'wait'
    return (f'BOARD: seller_id={seller}, product_id={pid}, price={money(price)}, in_stock={b["in_stock"]}.\n'
            f'CHECK this product: owned={owned}, cash={money(cash)}, next_value={money(value)}, '
            f'cash-price={money(cash-price)}, value-price={money(value-price)}; ACTION: {action}.')


def select_rows(rows,seed,seen_ids):
    pairs=defaultdict(dict)
    for r in rows:pairs[r['pair_id']][r['positive']]=r
    pool=defaultdict(list)
    for sides in pairs.values():
        if set(sides)!={True,False}:raise ValueError('Incomplete source pair')
        if any(r['id'] in seen_ids for r in sides.values()):continue
        a=sides[True];pool[a['family'],a['history_mode'],a['cash_variant']].append(sides)
    rng=random.Random(seed);result=[];used=set()
    def take(family,mode,variant,index):
        options=[s for s in pool[family,mode,variant] if s[True]['pair_id'] not in used]
        persona=(index+len(used))%3+1
        preferred=[s for s in options if s[True]['persona_index']==persona]
        selected=rng.choice(preferred or options);used.add(selected[True]['pair_id']);return selected
    for family in sorted({r['family'] for r in rows}):
        board=family.startswith('board_');n=8 if board else 4
        for mode in ('fresh','recorded'):
            for i in range(n):
                variant=('equal','above_one','wide','zero')[i%4] if family.endswith('cash') else 'standard'
                sides=take(family,mode,variant,i);result.extend([sides[True],sides[False]])
            if board:
                for i in range(8):
                    variant=('equal','above_one','wide','zero')[i%4] if family.endswith('cash') else 'standard'
                    result.append(take(family,mode,variant,i)[True])
    if len(result)!=256 or len({r['id'] for r in result})!=256:raise ValueError('Expected 256 unique repair examples')
    if Counter(r['label_action'] for r in result)!=Counter({'buy_listed':128,'accept_offer':32,'wait':96}):
        raise ValueError('Repair action quota mismatch')
    return result


def generate(source,output,parent,seed=20261121):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    source,output,parent=Path(source).resolve(),Path(output).resolve(),Path(parent).resolve()
    old=json.loads((source/'manifest.json').read_text());parent_run=parent.parent.parent
    cfg=json.loads((parent_run/'train_config.json').read_text())
    if old['version']!='buyer_operations_reasoning_v1' or cfg['data_manifest_sha256']!=file_sha(source/'manifest.json'):
        raise ValueError('Expected the unchanged reasoning-v1 parent data')
    if not json.loads((parent/'training_step.json').read_text())['thinking']:raise ValueError('Parent must be trained ON')
    for name,expected in old['sources'].items():
        if file_sha(ROOT/name)!=expected:raise ValueError('Frozen reasoning source changed')
    for split in ('train','validation','test'):
        for kind,key in (('jsonl','raw_sha256'),('tokens.jsonl','tokens_sha256')):
            if file_sha(source/f'{split}.{kind}')!=old['splits'][split][key]:raise ValueError('Source data changed')
    seen=set()
    for rank in (0,1):
        seen.update(json.loads(s)['id'] for s in (parent_run/f'samples_rank{rank}.jsonl').read_text().splitlines()
                    if json.loads(s)['step']<=32)
    with (source/'train.jsonl').open() as stream:rows=[json.loads(s) for s in stream]
    selected=select_rows(rows,seed,seen)
    train_groups={old['seed']+1009*g for g in old['groups']['train']}
    if any(r['group'] not in train_groups for r in selected):raise ValueError('Repair examples must come from training markets')
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    lengths=[];rationale=[]
    with (output/'train.jsonl').open('w') as raw,(output/'train.tokens.jsonl').open('w') as tokens:
        for original in tqdm(selected,desc='CPU board-first repair',unit='examples'):
            row=copy.deepcopy(original);row['curriculum_version']=VERSION
            if row['family'].startswith('board'):row['answer']['reasoning_content']=board_target(row)
            row['repair_source']='programmatic board-first demonstration' if row['family'].startswith('board') else 'unchanged offer reasoning replay'
            encoded=encode_row(tokenizer,row)
            if len(encoded['input_ids'])>12288:raise ValueError('Overlength repair example')
            raw.write(dump(row)+'\n');tokens.write(dump(encoded)+'\n')
            lengths.append(len(encoded['input_ids']));rationale.append(len(encoded['reasoning_positions']))
    splits={'train':{'raw':256,'tokenized':256,'actions':dict(Counter(r['label_action'] for r in selected)),
                     'max_tokens':max(lengths),'max_reasoning_tokens':max(rationale),
                     'raw_sha256':file_sha(output/'train.jsonl'),'tokens_sha256':file_sha(output/'train.tokens.jsonl')}}
    for split in ('validation','test'):
        for suffix in ('jsonl','tokens.jsonl'):shutil.copy2(source/f'{split}.{suffix}',output/f'{split}.{suffix}')
        splits[split]=old['splits'][split]
    shutil.copytree(source/'personas',output/'personas')
    shutil.copy2(source/'cash_probe_cases.json',output/'cash_probe_cases.json')
    (output/'agent.yaml').write_text('reasoning: true\n')
    manifest={'version':VERSION,'base_model':cfg['base_model'],'revision':cfg['revision'],'thinking':True,
              'seed':old['seed'],'selection_seed':seed,'groups':old['groups'],'splits':splits,
              'source_manifest_sha256':file_sha(source/'manifest.json'),
              'sources':{**old['sources'],'training/board_repair_data.py':file_sha(ROOT/'training/board_repair_data.py')},
              'parent_adapter':str(parent),'parent_fingerprints':fingerprint(parent),
              'cash_probe_sha256':file_sha(output/'cash_probe_cases.json'),
              'histories':dict(Counter(r['history_mode'] for r in selected)),
              'personas':dict(Counter(r['persona_index'] for r in selected)),
              'source_samples_excluded':len(seen),'input_changes':False,
              'note':'Stage A full-answer SFT demonstrations, not self-generated drafts or preference training. '
                     'Parent-seen rows excluded; validation/test and evaluation prompts copied unchanged.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'CPU repair ready: {output}; 256 examples; input and offer targets unchanged.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'training/data/buyer_reasoning_v1')
    p.add_argument('--output',type=Path,default=ROOT/'training/data/buyer_board_repair_v1')
    p.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    a=p.parse_args();generate(a.source,a.output,a.parent)
