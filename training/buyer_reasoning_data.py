#!/usr/bin/env python3
"""Original observations, brief visible-state arithmetic targets, native thinking ON."""
import argparse
from collections import Counter
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.buyer_operations_data import BASE, REVISION, dump, personas as old_personas
from training.buyer_operations_v2_1_data import SOURCES as OLD_SOURCES, cash_pair, contexts
from training.buyer_operations_v2_data import diagnostic_case, file_sha

VERSION = 'buyer_operations_reasoning_v1'
INSTRUCTION = ('Before the tool call, think in one or two short sentences: read CURRENT cash C, '
               'compute total P=price*qty and C-P, then check ownership, availability and value minus P. '
               'C-P must be nonnegative; exact equality is affordable. If negative, wait even for a cheap binding offer. '
               'Keep reasoning brief and finish with exactly one native tool call.')
SOURCES = (*OLD_SOURCES, 'training/buyer_reasoning_data.py')


def personas():
    return [p+'\n'+INSTRUCTION for p in old_personas()]


def cents(value):
    return int((Decimal(str(value))*100).to_integral_exact())


def money(value):
    return format(Decimal(value)/100, '.2f')


def reasoning_target(row):
    """Derive only from information visible in the current prompt, never outcomes."""
    obs, situation = row['observation'], row['situation']
    you = obs['you']; cash = cents(you['cash'])
    offers = obs.get('offers_to_you', [])
    board = obs.get('board', [])
    if len(offers)+len(board) > 1:
        raise ValueError('This curriculum requires one visible candidate')
    if not offers and not board:
        return f'Cash C={money(cash)}; no current listing or binding offer is available. Wait.', 'wait'
    option = (offers or board)[0]
    tool = 'accept_offer' if offers else 'buy_listed'
    pid = option['product_id']; qty = option.get('qty', 1)
    total = cents(option['price'])*qty
    value = cents(you['utility_of_next_unit'].get(pid, 0))
    varies = {p['product_id'] for p in situation['market_overview'] if p['quality_varies']}
    if pid in varies and not option.get('certified_high'):
        value = value*4//10
    owned = you['units_bought'].get(pid, 0)
    available = option.get('open_until_round', obs['round']) >= obs['round'] if offers else option['in_stock'] >= 1
    eligible = cash >= total and qty == 1 and not owned and available and value > total
    target = tool if eligible else 'wait'
    text = (f'For {pid}, cash C={money(cash)}, total P={money(cents(option["price"]))}*{qty}={money(total)}, '
            f'C-P={money(cash-total)}; next-unit value={money(value)}, value-P={money(value-total)}. '
            f'Owned={owned}, qty={qty}, available={"yes" if available else "no"}; '
            f'{"all checks pass: "+tool if eligible else "at least one check fails: wait"}.')
    return text, target


def replace_persona(messages, persona_index):
    result = copy.deepcopy(messages)
    content = result[0]['content']
    opening, closing = '<<<PERSONA_SECTION>>>', '<<<END_PERSONA_SECTION>>>'
    before, rest = content.split(opening, 1)
    _, after = rest.split(closing, 1)
    result[0]['content'] = before+opening+'\n'+personas()[persona_index-1]+'\n'+closing+after
    return result


def transform(row):
    new = copy.deepcopy(row)
    new['messages'] = replace_persona(row['messages'], row['persona_index'])
    reasoning, target = reasoning_target(row)
    if target != row['label_action']:
        raise ValueError(f'Visible arithmetic disagrees with the engine-checked teacher: {row["id"]}')
    new['answer']['reasoning_content'] = reasoning
    new['curriculum_version'] = VERSION
    new['thinking'] = True
    return new


def encode_row(tokenizer, row):
    prompt = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
    full = tokenizer.apply_chat_template(row['messages']+[row['answer']], tools=row['tools'], tokenize=False,
                                        add_generation_prompt=False, enable_thinking=True)
    if not prompt.endswith('<think>\n') or not full.startswith(prompt):
        raise ValueError('Native thinking template prefix mismatch')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded['input_ids'][:len(prefix)] != prefix:
        raise ValueError('Thinking target token boundary mismatch')
    text = full[len(prompt):]
    close = text.index('</think>')
    if text[:close].strip() != row['answer']['reasoning_content']:
        raise ValueError('Reasoning is not rendered before the tool call')
    match = re.search(r'<function=([^>]+)>', text)
    if not match or match.group(1) != row['label_action'] or match.start() <= close:
        raise ValueError('Expected exactly one native tool after reasoning')
    positions = []; reasoning_positions = []
    for i, (lo, hi) in enumerate(encoded['offset_mapping'][len(prefix):]):
        a, b = lo-len(prompt), hi-len(prompt)
        if a < match.end(1) and b > match.start(1): positions.append(i)
        if a < close and b > 0: reasoning_positions.append(i)
    if not positions or not reasoning_positions or len(reasoning_positions) > 192:
        raise ValueError('Missing decision/reasoning span or overlong scratchpad')
    keys = ('id','pair_id','group','task','family','positive','label_action','history_mode',
            'persona_index','cash_variant','economic_id','curriculum_version')
    return {**{k: row[k] for k in keys}, 'input_ids': encoded['input_ids'], 'prompt_length': len(prefix),
            'target_length': len(encoded['input_ids'])-len(prefix), 'tool_name_positions': positions,
            'reasoning_positions': reasoning_positions, 'thinking': True}


def generate(source, output):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source/'manifest.json').read_text())
    if old['version'] != 'buyer_operations_v2_1' or old['thinking']:
        raise ValueError('Expected the frozen OFF v2.1 source curriculum')
    for name, expected in old['sources'].items():
        if file_sha(ROOT/name) != expected: raise ValueError('Frozen source generator changed')
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    manifest = {'version': VERSION, 'base_model': BASE, 'revision': REVISION, 'thinking': True,
                'seed': old['seed'], 'groups': old['groups'], 'splits': {}, 'max_length': 12288,
                'sources': {p: file_sha(ROOT/p) for p in SOURCES},
                'source_manifest_sha256': file_sha(source/'manifest.json'),
                'source': 'v2.1 original-engine labels; programmatic brief visible-state arithmetic, no external LLM',
                'note': 'Only static persona and assistant reasoning targets change; observations and histories retain original facts.'}
    (output/'personas').mkdir()
    for i, text in enumerate(personas(), 1): (output/'personas'/f'persona_{i}.md').write_text(text+'\n')
    (output/'agent.yaml').write_text('reasoning: true\n')
    for split, info in old['splits'].items():
        if file_sha(source/f'{split}.jsonl') != info['raw_sha256']: raise ValueError('Source records changed')
        actions = Counter(); lengths = []; rationale_lengths = []
        raw_path, token_path = output/f'{split}.jsonl', output/f'{split}.tokens.jsonl'
        with (source/f'{split}.jsonl').open() as stream, raw_path.open('w') as raw, token_path.open('w') as tokens:
            for line in tqdm(stream, total=info['raw'], desc=f'CPU reasoning {split}', unit='examples'):
                row = transform(json.loads(line)); encoded = encode_row(tokenizer, row)
                if len(encoded['input_ids']) > 12288: raise ValueError('Overlength reasoning example')
                raw.write(dump(row)+'\n'); tokens.write(dump(encoded)+'\n')
                actions[row['label_action']] += 1
                lengths.append(len(encoded['input_ids'])); rationale_lengths.append(len(encoded['reasoning_positions']))
        manifest['splits'][split] = {'raw': len(lengths), 'tokenized': len(lengths), 'actions': dict(actions),
                                   'max_tokens': max(lengths), 'max_reasoning_tokens': max(rationale_lengths),
                                   'raw_sha256': file_sha(raw_path), 'tokens_sha256': file_sha(token_path)}
    cases = []
    for group in (old['groups']['validation'][2], old['groups']['validation'][0]):
        for channel in ('board','offer'):
            for variant in ('equal','above_one','wide','zero'):
                for row in contexts(cash_pair(old['seed']+1009*group, 0, channel, variant)):
                    row = transform(row); case = diagnostic_case(row)
                    case['origin'].update({k: row[k] for k in ('cash_variant','history_mode','economic_id')})
                    cases.append(case)
    path = output/'cash_probe_cases.json'; path.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    manifest['cash_probe_sha256'] = file_sha(path); manifest['cash_probe_cases'] = len(cases)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'CPU data complete: {output}; thinking ON, original observations; no GPU work.', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'training/data/buyer_operations_v2_1')
    parser.add_argument('--output', type=Path, default=ROOT/'training/data/buyer_reasoning_v1')
    args = parser.parse_args(); generate(args.source, args.output)
