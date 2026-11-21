#!/usr/bin/env python3
"""Training-only, engine-checked cash/ID controls with symmetric board/offer reasoning."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.buyer_operations_data import dump
from training.buyer_operations_v2_data import file_sha
from training.buyer_operations_v2_1_data import cash_pair, contexts, make_records
from training.buyer_reasoning_data import cents, money, encode_row, replace_persona
from training.id_repair_data import board_unit, DEFAULT_PARENT
from training.decision_ablation import fingerprint

VERSION = 'buyer_symmetric_repair_v1'


def symmetric_target(row):
    obs, sit = row['observation'], row['situation']
    board, offers = obs.get('board', []), obs.get('offers_to_you', [])
    prefix = f'SOURCES board={len(board)}, offer={len(offers)}; check BOTH.'
    if len(board)+len(offers) > 1:
        raise ValueError('Single-candidate controls only; mixed markets need separate validation')
    if not board and not offers:
        return prefix+' No current candidate. ACTION wait.', 'wait'
    item = (board or offers)[0]
    source = 'board' if board else 'offer'
    pid = item['product_id']; qty = item.get('qty', 1)
    seller = item['seller']
    identity = f'seller={seller}, product={pid}'
    if offers:
        identity += f', offer_id={item["offer_id"]}'
    cash = cents(obs['you']['cash']); total = cents(item['price'])*qty
    value = cents(obs['you']['utility_of_next_unit'].get(pid, 0))
    varies = {p['product_id'] for p in sit['market_overview'] if p['quality_varies']}
    if pid in varies and not item.get('certified_high'):
        value = value*4//10
    owned = obs['you']['units_bought'].get(pid, 0)
    available = item['in_stock'] >= 1 if board else item['open_until_round'] >= obs['round']
    affordable = cash >= total
    good = affordable and value > total and not owned and available and qty == 1
    action = ('buy_listed' if board else 'accept_offer') if good else 'wait'
    text = (f'{prefix} SOURCE {source}: {identity}. '
            f'C={money(cash)}, P={money(total)}, C-P={money(cash-total)}, C>=P={str(affordable).lower()}. '
            f'V={money(value)}, V-P={money(value-total)}, owned={owned}, qty={qty}, available={str(available).lower()}. '
            f'ACTION {action}.')
    return text, action


def adapt(row, pair_id):
    row = copy.deepcopy(row)
    row['pair_id'] = pair_id
    row['id'] = pair_id+('_positive' if row['positive'] else '_negative')
    row['curriculum_version'] = VERSION; row['thinking'] = True
    row['messages'] = replace_persona(row['messages'], row['persona_index'])
    text, action = symmetric_target(row)
    if action != row['label_action']:
        raise ValueError(f'Symmetric arithmetic/engine disagreement: {row["id"]}')
    row['answer']['reasoning_content'] = text
    row['provenance'] += '; uniform source/cash/value/action demonstration, not sampled model reasoning'
    return row


def records_for(source_manifest, seed):
    """96 cash pairs plus 16 board-ID and 16 noncash-offer pairs = 256 records."""
    order = list(source_manifest['groups']['train']); random.Random(seed).shuffle(order)
    groups = [source_manifest['seed']+1009*g for g in order]
    records = []
    for channel in ('board', 'offer'):
        index = 0
        for repeat in range(2):
            for persona in range(3):
                for variant in ('equal', 'above_one', 'wide', 'zero'):
                    for row in contexts(cash_pair(groups[index], persona, channel, variant)):
                        records.append(adapt(row, row['pair_id']+f'_sym{repeat}'))
                    index += 1
    # Both historical and different sellers are executable when CURRENTLY listed.
    distribution = (0, 1, 2, 1, 2, 0, 1, 2)
    for i, persona in enumerate(distribution):
        eligible = [g for g in groups if (g+persona)%3 != 0]
        group = eligible[i % len(eligible)]
        mode = 'fresh' if i%2 == 0 else 'recorded'
        kind = ('cash', 'owned', 'stock', 'value')[i%4]
        bundle = board_unit(group, persona, mode, kind, i)
        for seat in (0, 1):
            pair_id = f'g{group}_board_id_p{persona+1}_{mode}_s{seat}_sym{i}'
            for condition in ('ample', 'negative'):
                row = next(r for r in bundle if r['current_seller_is_historical'] == (seat == 0)
                           and r['cash_variant'] == condition)
                records.append(adapt(row, pair_id))
        # Preserve ownership/value/quantity boundaries in offers as well as cash controls.
        family = ('offer_owned', 'offer_value', 'offer_qty')[i%3]
        for row in make_records(groups[(24+i)%len(groups)], family, persona):
            records.append(adapt(row, row['pair_id']+f'_sym{i}'))
    if len(records) != 256 or len({r['id'] for r in records}) != 256:
        raise ValueError('Expected 256 unique IDs')
    if Counter(r['label_action'] for r in records) != Counter(buy_listed=64, accept_offer=64, wait=128):
        raise ValueError('Action mix mismatch')
    return records


def generate(source, output, parent, seed=20261123):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    source, output, parent = (Path(p).resolve() for p in (source, output, parent))
    old = json.loads((source/'manifest.json').read_text())
    cfg = json.loads((parent.parent.parent/'train_config.json').read_text())
    if old['version'] != 'buyer_operations_reasoning_v1' or cfg['data_version'] != 'buyer_board_repair_v1':
        raise ValueError('Expected frozen reasoning source and safer board-repair parent')
    if not json.loads((parent/'training_step.json').read_text())['thinking']:
        raise ValueError('Parent must use thinking ON')
    for name, expected in old['sources'].items():
        if file_sha(ROOT/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    records = records_for(old, seed)
    allowed = {old['seed']+1009*g for g in old['groups']['train']}
    if any(r['group'] not in allowed for r in records):
        raise ValueError('Non-training market in repair data')
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
    lengths = []; reasoning = []
    with (output/'train.jsonl').open('w') as raw, (output/'train.tokens.jsonl').open('w') as tokens:
        for row in tqdm(records, desc='CPU encode symmetric repair', unit='examples'):
            encoded = encode_row(tokenizer, row)
            if len(encoded['input_ids']) > 12288:
                raise ValueError('Overlength example; never truncate facts')
            raw.write(dump(row)+'\n'); tokens.write(dump(encoded)+'\n')
            lengths.append(len(encoded['input_ids'])); reasoning.append(len(encoded['reasoning_positions']))
    splits = {'train': {'raw': len(records), 'tokenized': len(records),
                       'actions': dict(Counter(r['label_action'] for r in records)),
                       'max_tokens': max(lengths), 'max_reasoning_tokens': max(reasoning),
                       'raw_sha256': file_sha(output/'train.jsonl'), 'tokens_sha256': file_sha(output/'train.tokens.jsonl')}}
    for split in ('validation', 'test'):
        for suffix, key in (('jsonl', 'raw_sha256'), ('tokens.jsonl', 'tokens_sha256')):
            if file_sha(source/f'{split}.{suffix}') != old['splits'][split][key]:
                raise ValueError('Frozen holdout changed')
            shutil.copy2(source/f'{split}.{suffix}', output/f'{split}.{suffix}')
        splits[split] = old['splits'][split]
    shutil.copytree(source/'personas', output/'personas')
    shutil.copy2(source/'cash_probe_cases.json', output/'cash_probe_cases.json')
    (output/'agent.yaml').write_text('reasoning: true\n')
    coverage = {f'{channel}/p{p}': dict(Counter(r['label_action'] for r in records
                 if r['family'].startswith(channel+'_') and r['persona_index'] == p))
                for channel in ('board', 'offer') for p in (1, 2, 3)}
    sources = {**old['sources'], **{name: file_sha(ROOT/name) for name in
               ('training/symmetric_repair_data.py', 'training/id_repair_data.py', 'training/board_repair_data.py')}}
    manifest = {'version': VERSION, 'base_model': cfg['base_model'], 'revision': cfg['revision'], 'thinking': True,
                'seed': old['seed'], 'selection_seed': seed, 'groups': old['groups'], 'sources': sources, 'splits': splits,
                'source_manifest_sha256': file_sha(source/'manifest.json'),
                'parent_adapter': str(parent), 'parent_fingerprints': fingerprint(parent),
                'cash_probe_sha256': file_sha(output/'cash_probe_cases.json'), 'coverage': coverage,
                'histories': dict(Counter(r['history_mode'] for r in records)),
                'unique_message_inputs': len({dump(r['messages']) for r in records}),
                'note': 'Training-only engine controls; both sources share target structure. '
                        'Cash-only pairs, ID replay and offer negatives. Holdouts/personas unchanged. '
                        'Single candidate, not full markets; SFT demonstration, not preference/RL training.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'coverage': coverage, 'histories': manifest['histories']}, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'training/data/buyer_reasoning_v1')
    p.add_argument('--output', type=Path, default=ROOT/'training/data/buyer_symmetric_repair_v1')
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    a = p.parse_args(); generate(a.source, a.output, a.parent)
