"""Bounded 48-case learnability pilot, drawn only from frozen official-train controls."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random

from training.buyer_latest_brakes_data import ROOT, sha
from training.buyer_self_checks import encode, target_check, truth

VERSION = 'buyer_self_checks_v1'
DATA = ROOT/'training/data'/VERSION
SOURCE = ROOT/'training/data/buyer_latest_brakes_v1'
SEED = 20261122


def generate(output=DATA):
    import yaml
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.train_buyer_latest_brakes import prepare
    if Path(output).exists():
        raise ValueError('Refuse overwrite; choose a new output directory')
    cfg, old, _ = prepare(yaml.safe_load((ROOT/'training/configs/buyer_latest_brakes_v1.yaml').read_text()))
    rows = [json.loads(l) for l in (SOURCE/'train.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for r in rows:
        if r.get('mode'):
            groups[r['group']].append(r)
    def kind(block):
        r = next(r for r in block if not r['positive'])
        if r['mode'] != 'cash':
            return r['mode']
        facts = truth(r); target = target_check(r, facts)
        if facts['cash'] == 0:
            return 'zero_cash'
        if target['total'] - facts['cash'] == 1:
            return 'cent_cash'
        return 'other_cash'
    chosen = []
    for label in ('cent_cash', 'zero_cash', 'owned', 'value'):
        group = next(g for g, block in groups.items() if kind(block) == label)
        chosen.extend(copy.deepcopy(groups[group]))
    if len(chosen) != 48:
        raise ValueError('Expect four complete worlds × three personas × two channels × two sides')
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
    train = []; tokens = []
    for r in tqdm(chosen, desc='CPU self-check labels', unit='cases'):
        facts = truth(r); selected = target_check(r, facts)
        eligible = [c for c in facts['checks'] if c['eligible']]
        if selected['eligible'] != r['positive'] or len(eligible) != int(r['positive']):
            raise ValueError('Teacher eligibility and original engine proof disagree')
        row, token = encode(tokenizer, r); train.append(row); tokens.append(token)
    pairs = defaultdict(list)
    for i, r in enumerate(train):
        pairs[r['pair_id']].append(i)
    by_channel = {c: [p for p in pairs.values() if train[p[0]]['channel'] == c] for c in ('board', 'offer')}
    rng = random.Random(SEED); batches = []
    for epoch in range(2):
        pools = copy.deepcopy(by_channel)
        for pool in pools.values():
            rng.shuffle(pool)
        for _ in range(6):
            batch = [i for c in ('board', 'offer') for _ in range(2) for i in pools[c].pop()]
            rng.shuffle(batch); batches.append(batch)
    # Seen: each persona/channel positive+negative, alternating cent and owned worlds.
    seen = []
    for persona in (1, 2, 3):
        for channel in ('board', 'offer'):
            mode = 'owned' if (persona + (channel == 'offer')) % 2 else 'cash'
            block = next(p for p in pairs.values() if train[p[0]]['persona_index'] == persona and train[p[0]]['channel'] == channel and
                train[p[0]]['mode'] == mode and (mode != 'cash' or kind([train[i] for i in p]) == 'cent_cash'))
            for i in block:
                r = copy.deepcopy(train[i]); r['suite'] = 'self_check_seen'; r['name'] = 'checks_seen_' + r['id']; seen.append(r)
    validation = [json.loads(l) for l in (SOURCE/'validation.jsonl').read_text().splitlines()]
    val_tokens = []
    for i, r in enumerate(validation):
        r, token = encode(tokenizer, r); validation[i] = r; val_tokens.append(token)
        r['suite'] = 'self_check_validation'
    cases = json.loads((SOURCE/'evaluation_cases.json').read_text())
    cases = [r for r in cases if r['suite'] in ('known', 'fresh', 'official')] + validation + seen
    # Each complete control pair stays on the same physical GPU across checkpoints.
    for c in cases:
        if c['suite'].startswith('self_check_'):
            c['evaluation_rank'] = (c['persona_index'] + (c['channel'] == 'offer')) % 2
    output = Path(output); output.mkdir(parents=True)
    for name, rows in (('train', train), ('train.tokens', tokens), ('validation', validation), ('validation.tokens', val_tokens)):
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (output/'batch_plan.json').write_text(json.dumps(batches)+'\n')
    (output/'evaluation_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1)+'\n')
    sources = ['training/buyer_self_checks.py', 'training/buyer_self_checks_data.py',
        'training/data/buyer_latest_brakes_v1/manifest.json', *old['sources']]
    for i in (1, 2, 3):
        (output/f'persona_{i}.md').write_text((ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text())
    manifest = dict(version=VERSION, thinking=True, seed=SEED, parent_adapter=old['parent_adapter'],
        parent_fingerprints=old['parent_fingerprints'], base_model=cfg['base_model'], revision=cfg['revision'],
        train_examples=48, validation_examples=24, evaluation_calls=len(cases), cached_parent_calls=37,
        groups={'train': sorted({r['group'] for r in train}), 'validation': sorted({r['group'] for r in validation})},
        sources={name: sha(ROOT/name) for name in sources},
        source_files={str(p.relative_to(ROOT)): sha(p) for p in (SOURCE/'train.jsonl', SOURCE/'validation.jsonl', SOURCE/'evaluation_cases.json')},
        files={name: sha(output/name) for name in ('train.jsonl', 'train.tokens.jsonl', 'validation.jsonl', 'validation.tokens.jsonl', 'batch_plan.json', 'evaluation_cases.json', 'persona_1.md', 'persona_2.md', 'persona_3.md')},
        mean_tokens=sum(len(t['input_ids']) for t in tokens)/len(tokens), max_tokens=max(len(t['input_ids']) for t in tokens),
        max_reasoning_tokens=max(t['reasoning_tokens'] for t in tokens+val_tokens),
        note='Learnability pilot: 48 unique controlled states from official-train worlds, two explicit passes (96 exposures), not a new official-data baseline. No oracle at generation, no observation/persona changes, no automatic promotion.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def check_data(cfg):
    from training.buyer_operations_train import TokenRows
    from training.decision_ablation import check_fingerprint
    folder = ROOT/cfg['data_dir']; m = json.loads((folder/'manifest.json').read_text())
    if m['version'] != VERSION or not m['thinking']:
        raise ValueError('Wrong self-check data')
    for name, digest in {**m['sources'], **m['source_files']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Source/persona changed: {name}')
    for name, digest in m['files'].items():
        if sha(folder/name) != digest:
            raise ValueError(f'Frozen data changed: {name}')
    if set(m['groups']['train']) & set(m['groups']['validation']):
        raise ValueError('Source group leakage')
    check_fingerprint(m['parent_adapter'], m['parent_fingerprints'])
    train = TokenRows(folder/'train.tokens.jsonl', 12288); val = TokenRows(folder/'validation.tokens.jsonl', 12288)
    batches = json.loads((folder/'batch_plan.json').read_text())
    if len(train) != 48 or len(val) != 24 or len(batches) != 12:
        raise ValueError('Pilot size changed')
    for offset in (0, 6):
        if Counter(i for b in batches[offset:offset+6] for i in b) != Counter(range(48)):
            raise ValueError('An epoch does not cover all 48 cases exactly once')
    for b in batches:
        if len(b) != 8 or Counter(train[i]['label_action'] for i in b) != {'buy_listed': 2, 'accept_offer': 2, 'wait': 4}:
            raise ValueError('Unbalanced contrast batch')
    return train, val, batches, m


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
