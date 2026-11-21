#!/usr/bin/env python3
"""Reproducible operation SFT data; CPU generation, original-engine target checks."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TASKS = ('live_pair', 'owned_high', 'cash_limit', 'no_profit', 'empty',
         'binding_offer', 'loss_multiunit_offer', 'failed_pair', 'known_low', 'stale_memory')
COMMON = '''Respond with exactly one native tool call, even on the last turn.
Use the CURRENT observation, overriding old actions, messages and notes.
For buy_listed copy seller_id and product_id from ONE current board row with stock>=1; qty=1.
Skip items already owned, items with no positive next-unit utility, and purchases exceeding current cash.
Accept only current, unexpired offers with qty=1. Chat is not an offer.
For fixed/certified quality use next-unit utility V. For uncertified variable quality use low_grade_utility*V as a conservative bound for this operation-training stage.
Require positive bound minus total price. Choose the largest such gain; if none exists call wait.
Previous failure: re-read the current board and choose an eligible pair; do not repeat a stale pair from history.
Include private_state as a short JSON STRING: {"bad":[],"pending":null,"last_buy":null}.
bad stores confirmed low sellers; pending stores a live negotiation or null. last_buy is [seller_id,product_id] on buy_listed, otherwise null. Do not copy held/cash into memory; read them from observation.
Do not disclose your values or true cash in messages.''' 
STYLES = (
    'Buyer p1: prioritize the largest executable gain; ties favor lower spending.',
    'Buyer p2: compare binding offers and board purchases together; execute a good deal promptly.',
    'Buyer p3: track confirmed low deliveries; distrust quality promises, use the stated quality bound.')


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def personas():
    return [COMMON + '\n' + style for style in STYLES]


def assistant(tool, args, call_id='answer'):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': tool, 'arguments': args}}]}


def normalized(messages):
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls', []):
            if isinstance(call['function'].get('arguments'), str):
                call['function']['arguments'] = json.loads(call['function']['arguments'])
    return result


def encode(tokenizer, record):
    """Mask everything except the final assistant completion, including past mistakes."""
    messages = normalized(record['messages'])
    prompt = tokenizer.apply_chat_template(messages, tools=record['tools'], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [record['answer']], tools=record['tools'],
                                         tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError('Official template completion is not a continuation of its generation prompt')
    ids = tokenizer.encode(full, add_special_tokens=False)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError('Token boundary mismatch; refusing an incorrect loss mask')
    return {'id': record['id'], 'group': record['group'], 'task': record['task'],
            'input_ids': ids, 'prompt_length': len(prefix), 'target_length': len(ids)-len(prefix)}


def fresh_market(seed):
    from adlmarket.config import AgentSpec, MarketSpec
    from adlmarket.market import Market, Product
    from adlmarket.schema import ProductPlan, Schema
    rng = random.Random(seed)
    products, plans = {}, []
    for i in range(24):
        pid = 'P' + f'{rng.getrandbits(28):07x}'
        value = rng.randint(600, 12000)
        cost = rng.randint(50, max(51, value // 2))
        products[pid] = Product(pid, f'{rng.choice(["Serum", "Cream", "Lotion", "Shampoo"])} {i}',
                                'Beauty', rng.randint(cost, value), brand=f'Brand {i%5}',
                                summary='Beauty product for routine care.')
        plans.append(ProductPlan(pid, 'balanced', 1.5, None, (value,), cost,
                                 quality_varies=(i in (0,3)), low_cost=max(1, cost//2) if i in (0,3) else None))
    sellers = [AgentSpec(id=f'seller_{rng.getrandbits(24):06x}', role='seller', model='base') for _ in range(5)]
    buyers = [AgentSpec(id=f'buyer_{i}', role='buyer', owner='synthetic', model='base') for i in range(10)]
    spec = MarketSpec()
    schema = Schema(f'operations_{seed}', tuple(plans), rng.randint(10000, 50000), .4)
    market = Market.generate(schema.id, spec, schema, products, sellers, buyers, rng)
    market.tick = rng.randint(7, 29)
    market.round = (market.tick-1)//6+1
    return market, rng


def choose(situation, obs):
    """Visible-only conservative reference policy; not claimed to be strategically optimal."""
    from adlmarket.tools import to_cents
    you = obs['you']
    cash = to_cents(you['cash'])
    varies = {r['product_id'] for r in situation['market_overview'] if r['quality_varies']}
    candidates = []
    options = [('buy_listed', b) for b in obs.get('board', []) if b['in_stock'] >= 1]
    options += [('accept_offer', o) for o in obs.get('offers_to_you', [])
                if o['qty'] == 1 and o['open_until_round'] >= obs['round']]
    for tool, row in options:
        pid = row['product_id']
        if you['units_bought'].get(pid, 0):
            continue
        value = to_cents(you['utility_of_next_unit'].get(pid, 0))
        if pid in varies and not row.get('certified_high'):
            value = round(value * .4)
        price = to_cents(row['price'])
        if 0 <= price <= cash and value > price:
            args = ({'seller_id': row['seller'], 'product_id': pid, 'qty': 1} if tool == 'buy_listed'
                    else {'offer_id': row['offer_id']})
            candidates.append((value-price, -price, row['seller'], pid, tool, args))
    if candidates:
        *_, tool, args = max(candidates, key=lambda x: x[:4])
    else:
        tool, args = 'wait', {}
    bad = sorted({t['seller'] for t in obs.get('your_unreviewed_trades', []) if t.get('grade') == 'low'})
    state = {'bad': bad, 'pending': None,
             'last_buy': [args['seller_id'], args['product_id']] if tool == 'buy_listed' else None}
    return tool, {**args, 'private_state': dump(state)}


def make_record(group_seed, task, persona_index):
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message, system_prompt
    from adlmarket.tools import dollars
    market, rng = fresh_market(group_seed)
    initial = copy.deepcopy(market)
    current_tick = market.tick
    aid = f'buyer_{persona_index}'
    sit = market.situation(aid)  # market-start facts, before any purchases
    buyer = market.buyers[aid]
    sellers, pids = list(market.sellers), list(market.products)
    pid, alternate = pids[1], pids[2]
    value = buyer.utility[pid][0]
    price = max(1, value//3)
    history, previous_result, private = [], None, ''
    # Random distractors; selection cannot rely on one fixed row position or ID.
    for product in pids[3:]:
        if rng.random() < .55:
            seller = rng.choice(sellers)
            market.listings[seller, product] = Listing(max(1, round(buyer.utility[product][0]*rng.uniform(.3, 1.4))))
    market.listings[sellers[0], pid] = Listing(price)
    if task in ('empty', 'binding_offer', 'loss_multiunit_offer'):
        market.listings.clear()
    if task in ('no_profit', 'loss_multiunit_offer'):
        for (seller, product), listing in market.listings.items():
            listing.price = buyer.utility[product][0] + rng.randint(1, 1000)
    if task == 'cash_limit':
        buyer.cash = price - 1
    if task == 'known_low':
        variable = pids[0]
        market.listings[sellers[0], variable] = Listing(1, 'low')
        result = market.apply(aid, 'buy_listed', {'seller_id': sellers[0], 'product_id': variable, 'qty': 1}).as_dict()
        assert result['ok']
        market.listings[sellers[0], variable].price = buyer.utility[variable][0] // 2
        # Another variable-quality product is still unowned; the known-low seller's
        # inflated quote must not inherit its high-grade valuation.
        market.listings[sellers[0], pids[3]] = Listing(buyer.utility[pids[3]][0]//2, 'low')
    if task == 'owned_high':
        market.tick = current_tick-1; market.round = (market.tick-1)//6+1
        before = market.observe(aid, turns_left=4)
        old_args = {'seller_id': sellers[0], 'product_id': pid, 'qty': 1}
        previous_result = market.apply(aid, 'buy_listed', old_args).as_dict()
        assert previous_result['ok']
        history = [{'role': 'user', 'content': observation_message(before, '', None)},
                   assistant('buy_listed', old_args, 'past'),
                   {'role': 'tool', 'tool_call_id': 'past', 'content': dump(previous_result)}]
        private = dump({'held': [], 'cash': sit['your_budget'], 'last_buy': [sellers[0], pid]})
    if task == 'stale_memory':
        market.tick = current_tick-1; market.round = (market.tick-1)//6+1
        before = market.observe(aid, turns_left=6-(market.tick-1)%6)
        private = dump({'held': [pid], 'cash': 0, 'last_buy': [sellers[0], pid]})
        previous_result = market.apply(aid, 'wait', {}).as_dict()
        history = [{'role': 'user', 'content': observation_message(before, '', None)},
                   assistant('wait', {'private_state': private}, 'past'),
                   {'role': 'tool', 'tool_call_id': 'past', 'content': dump(previous_result)}]
        market.listings.clear()
        market.listings[sellers[0], pid] = Listing(price)
    if task == 'failed_pair':
        market.tick = current_tick-1; market.round = (market.tick-1)//6+1
        before = market.observe(aid, turns_left=4)
        market.sellers[sellers[0]].stock[pid] = 0
        old_args = {'seller_id': sellers[0], 'product_id': pid, 'qty': 1}
        previous_result = market.apply(aid, 'buy_listed', old_args).as_dict()
        assert previous_result.get('error') == 'no_stock'
        history = [{'role': 'user', 'content': observation_message(before, '', None)},
                   assistant('buy_listed', old_args, 'past'),
                   {'role': 'tool', 'tool_call_id': 'past', 'content': dump(previous_result)}]
        private = dump({'bad': [], 'pending': None, 'last_buy': [sellers[0], pid]})
    if task in ('binding_offer', 'loss_multiunit_offer'):
        qty = 1 if task == 'binding_offer' else 2
        offer_price = price if qty == 1 else value*3//4
        result = market.apply(sellers[1], 'make_offer', {'buyer_id': aid, 'product_id': pid,
                              'price': dollars(offer_price), 'qty': qty}).as_dict()
        assert result['ok']
    if task == 'live_pair':
        market.listings.pop((sellers[0], pid), None)
        market.listings[sellers[1], pid] = Listing(price)
        market.listings[sellers[0], alternate] = Listing(buyer.utility[alternate][0] + 1)
    market.tick = current_tick; market.round = (current_tick-1)//6+1
    # Some histories contain up to six complete turns. Earlier empty-board waits are
    # valid engine turns, not shortened observations; the last turn may contain a mistake.
    existing = len(history)//3
    count = max(0, rng.choice([0, 1, 3, 6])-existing)
    prefix = []
    for i in range(count):
        initial.tick = current_tick-existing-count+i
        initial.round = (initial.tick-1)//6+1
        past_obs = initial.observe(aid, turns_left=6-(initial.tick-1)%6)
        past_obs['output_tokens_left_this_round'] = 8192
        wait_args = {'private_state': dump({'bad': [], 'pending': None, 'last_buy': None})}
        result = initial.apply(aid, 'wait', {}).as_dict()
        call_id = f'prefix_{i}'
        prefix.extend([{'role': 'user', 'content': observation_message(past_obs, '', None)},
                       assistant('wait', wait_args, call_id),
                       {'role': 'tool', 'tool_call_id': call_id, 'content': dump(result)}])
    history = prefix + history
    obs = market.observe(aid, turns_left=6-(market.tick-1)%6)
    obs['output_tokens_left_this_round'] = 8192
    tool, args = choose(sit, obs)
    # Only labels see the engine's private state; choose() above gets visible facts only.
    test = copy.deepcopy(market)
    before_utility = test.buyers[aid].gained_utility
    result = test.apply(aid, tool, {k: v for k, v in args.items() if k != 'private_state'}).as_dict()
    if not result['ok']:
        raise ValueError(f'Illegal teacher action: {task} {result}')
    if tool in ('buy_listed', 'accept_offer'):
        if test.buyers[aid].gained_utility-before_utility <= round(result['data']['total']*100):
            raise ValueError('Teacher purchase has nonpositive realized surplus')
    prompt = system_prompt('buyer', personas()[persona_index], sit, 6,
                           quality=market.spec.schema_.quality, excess=True, round_tokens=8192)
    messages = [{'role': 'system', 'content': prompt}, *history,
                {'role': 'user', 'content': observation_message(obs, private, previous_result)}]
    rid = f'g{group_seed}_{task}_p{persona_index+1}'
    return {'id': rid, 'group': group_seed, 'task': task, 'messages': messages,
            'tools': list(market.tools['buyer'].values()), 'answer': assistant(tool, args),
            'observation': obs, 'situation': sit, 'engine_result': result,
            'provenance': 'synthetic_original_engine_visible_only_teacher_v1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'training/data/buyer_operations_v1')
    parser.add_argument('--groups', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--max-length', type=int, default=12288)
    parser.add_argument('--raw-only', action='store_true')
    args = parser.parse_args()
    if args.groups < 10:
        raise ValueError('Need >=10 independent groups for train/validation/test')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'personas').mkdir()
    for i, text in enumerate(personas(), 1):
        (output/'personas'/f'persona_{i}.md').write_text(text+'\n')
    tokenizer = None
    if not args.raw_only:
        from transformers import AutoTokenizer
        print('Loading cached official tokenizer / chat template (CPU only)...', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    from tqdm import tqdm
    order = list(range(args.groups))
    random.Random(args.seed).shuffle(order)
    n_eval = max(1, args.groups//10)
    assignments = {'validation': order[:n_eval], 'test': order[n_eval:2*n_eval], 'train': order[2*n_eval:]}
    manifest = {'version': 'buyer_operations_v1', 'seed': args.seed, 'groups': assignments,
                'base_model': BASE, 'revision': REVISION, 'max_length': args.max_length,
                'thinking': False, 'source': 'synthetic only; no TA weights/rollouts or diagnostic cases',
                'split_unit': 'market seed, before persona/task expansion', 'splits': {}}
    manifest['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['persona_sha256'] = [hashlib.sha256(text.encode()).hexdigest() for text in personas()]
    for split, groups in assignments.items():
        counts, skipped = Counter(), Counter()
        lengths, replay = [], []
        raw_path, token_path = output/f'{split}.jsonl', output/f'{split}.tokens.jsonl'
        with raw_path.open('w') as raw, token_path.open('w') as encoded:
            jobs = [(g, task, p) for g in groups for task in TASKS for p in range(3)]
            for g, task, p in tqdm(jobs, desc=f'CPU {split}', unit='examples'):
                record = make_record(args.seed + 1009*g, task, p)
                raw.write(dump(record)+'\n')
                counts[task] += 1
                if tokenizer:
                    token = encode(tokenizer, record)
                    if len(token['input_ids']) > args.max_length:
                        skipped[task] += 1
                    else:
                        lengths.append(len(token['input_ids']))
                        encoded.write(dump(token)+'\n')
                if split == 'test' and task not in {c['task'] for c in replay}:
                    replay.append({'name': record['id'], 'task': task,
                                   'origin': {'synthetic': True, 'group': record['group']},
                                   'messages': record['messages'], 'tools': record['tools'],
                                   'observation': record['observation'], 'situation': record['situation'],
                                   'contexts': ['recorded'], 'state_mode': 'operations'})
        manifest['splits'][split] = {'raw': sum(counts.values()), 'by_task': dict(counts),
                                     'tokenized': len(lengths), 'overlength': dict(skipped),
                                     'max_tokens': max(lengths, default=0),
                                     'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                     'tokens_sha256': hashlib.sha256(token_path.read_bytes()).hexdigest()}
        if replay:
            (output/'test_cases.json').write_text(dump(replay))
        if tokenizer and sum(skipped.values()) > .1*sum(counts.values()):
            raise ValueError('More than 10% overlength: adjust data design; never silently truncate current facts')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['splits'], indent=2), flush=True)
    print(f'CPU data complete: {output}', flush=True)


if __name__ == '__main__':
    main()
