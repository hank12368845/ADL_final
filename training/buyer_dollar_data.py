"""Relabel existing48 inputs only; freeze dollar lines and staged evaluation data."""
from collections import Counter
import copy
import json
from pathlib import Path

from training.buyer_latest_brakes_data import ROOT, sha

VERSION = 'buyer_dollar_lines_v1'
DATA = ROOT/'training/data'/VERSION
SOURCE = ROOT/'training/data/buyer_self_checks_v1'
BASELINE = ROOT/'training/runs/20261120-130432_buyer_self_checks_rueclguj/native_step_0000.json'


def generate(output=DATA):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.buyer_self_checks_data import check_data as original_check
    from training.buyer_dollar_lines import encode, assess, teach
    if Path(output).exists(): raise ValueError('Refuse overwrite existing dollar data')
    _, _, batches, old = original_check({'data_dir': str(SOURCE.relative_to(ROOT))})
    t = AutoTokenizer.from_pretrained(old['base_model'], revision=old['revision'], local_files_only=True)
    output = Path(output); original_train = [json.loads(l) for l in (SOURCE/'train.jsonl').read_text().splitlines()]
    original_val = [json.loads(l) for l in (SOURCE/'validation.jsonl').read_text().splitlines()]
    train = []; val = []; tokens = []; val_tokens = []
    for row in tqdm(original_train+original_val, desc='CPU dollar targets', unit='cases'):
        new, tok = encode(t, row)
        if new['messages'] != row['messages'] or new['answer']['tool_calls'] != row['answer']['tool_calls']:
            raise ValueError('Original input or target tool changed')
        if row in original_train: train.append(new); tokens.append(tok)
        else: val.append(new); val_tokens.append(tok)
    lookup = {r['id']: r for r in train+val}
    cases = json.loads((SOURCE/'evaluation_cases.json').read_text())
    for c in cases:
        if c['suite'].startswith('self_check_'):
            updated = teach(c); c['answer'] = updated['answer']
    old_outputs = {r['case']: r for r in json.loads(BASELINE.read_text())['decisions']}
    baseline = []
    for c in tqdm(cases, desc='CPU cached parent scoring', unit='cases'):
        r = old_outputs[c['name']]
        baseline.append(assess(c, r['text'], t, r['truncated'], c['evaluation_rank'], True))
    if len(cases) != 73 or len(train) != 48 or len(val) != 24: raise ValueError('Frozen cohort size changed')
    for c in cases:
        if c['suite']=='self_check_seen' and c['messages'] != lookup[c['id']]['messages']:
            raise ValueError('Seen input mismatch')
    output.mkdir(parents=True)
    for name, rows in (('train',train),('train.tokens',tokens),('validation',val),('validation.tokens',val_tokens)):
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for name, obj in (('batch_plan.json',batches),('evaluation_cases.json',cases),('baseline_decisions.json',baseline)):
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
    for i in (1,2,3): (output/f'persona_{i}.md').write_text((SOURCE/f'persona_{i}.md').read_text())
    sources = ('training/buyer_dollar_data.py','training/buyer_dollar_lines.py','training/buyer_check_lines.py',
        'training/buyer_self_checks.py','training/official_buyer_data.py','training/official_buyer_train.py',
        *(f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)))
    m = dict(version=VERSION, thinking=True, max_steps=12, warmup_steps=2, checkpoint_steps=[6,12],
        learning_rate=1e-5, effective_batch_size=8, seed=20261122,
        parent_adapter=old['parent_adapter'], parent_fingerprints=old['parent_fingerprints'],
        base_model=old['base_model'], revision=old['revision'], groups=old['groups'], train_examples=48,
        validation_examples=24, baseline_cached_calls=73, step6_gpu_calls=12, step12_gpu_calls=73,
        mean_tokens=sum(len(r['input_ids']) for r in tokens)/48, max_tokens=max(len(r['input_ids']) for r in tokens),
        max_check_tokens=max(len(r['check_positions']) for r in tokens),
        files={name:sha(output/name) for name in ('train.jsonl','train.tokens.jsonl','validation.jsonl','validation.tokens.jsonl',
            'batch_plan.json','evaluation_cases.json','baseline_decisions.json','persona_1.md','persona_2.md','persona_3.md')},
        sources={name:sha(ROOT/name) for name in sources},
        provenance={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE/'manifest.json',SOURCE/'train.jsonl',
            SOURCE/'validation.jsonl',SOURCE/'batch_plan.json',SOURCE/'evaluation_cases.json',BASELINE)},
        note='No new inputs/worlds/personas: existing48 relabelled with one current candidate dollar line. First6 steps cover all48 once; gate only seen12, no zero-cash examples in that seen subset. If pass, continue to12 and assess all73; no expansion or promotion.')
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    return m


def check_data(cfg):
    from training.buyer_operations_train import TokenRows
    from training.decision_ablation import check_fingerprint
    folder=ROOT/cfg['data_dir'];m=json.loads((folder/'manifest.json').read_text())
    if m['version'] != VERSION or not m['thinking']: raise ValueError('Wrong dollar data version')
    for name,digest in {**m['sources'],**m['provenance']}.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Dollar source/provenance changed: {name}')
    for name,digest in m['files'].items():
        if sha(folder/name)!=digest: raise ValueError(f'Dollar data changed: {name}')
    check_fingerprint(m['parent_adapter'],m['parent_fingerprints'])
    if set(m['groups']['train'])&set(m['groups']['validation']): raise ValueError('Source group leakage')
    train=TokenRows(folder/'train.tokens.jsonl',12288);val=TokenRows(folder/'validation.tokens.jsonl',12288)
    batches=json.loads((folder/'batch_plan.json').read_text())
    if len(train)!=48 or len(val)!=24 or len(batches)!=12: raise ValueError('Wrong bounded pilot size')
    for offset in (0,6):
        if Counter(i for b in batches[offset:offset+6] for i in b)!=Counter(range(48)):
            raise ValueError('Incomplete full pass through48')
    for b in batches:
        if len(b)!=8 or Counter(train[i]['label_action'] for i in b)!={'buy_listed':2,'accept_offer':2,'wait':4}:
            raise ValueError('Unbalanced positive/negative batch')
        pairs=Counter(train[i]['pair_id'] for i in b)
        if len(pairs)!=4 or set(pairs.values())!={2}: raise ValueError('Incomplete contrast pairs')
    return train,val,batches,m


if __name__=='__main__': print(json.dumps(generate(),indent=2))
