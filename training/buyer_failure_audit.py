#!/usr/bin/env python3
"""CPU audit of current-board matching, explicit memory, quality and actual GPU routes.

Exports a small frozen replay set. Retrospective trades are audit labels only;
the replay prompt contains exactly the original agent-visible messages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit_failures(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = {'markets': {}, 'load': {}, 'note':
               'Retrospective diagnosis, not an agent input, optimal-policy label or final grade. '
               'A valid current row can lose a same-turn race. Private offers do not reserve stock.'}
    replay = []
    load = defaultdict(lambda: {'calls': 0, 'output_tokens': 0, 'agents': set()})
    for folder in sorted((run_dir / 'markets').iterdir()):
        if not (folder / 'final.json').exists():
            continue
        final = json.loads((folder / 'final.json').read_text())
        agents = {a['agent']: a for a in rows(folder / 'agents.jsonl')}
        trades = final['trades']
        report = {'buyers': {}, 'extra_seller_deliveries': {}}
        for seller in sorted({t['seller'] for t in trades if 'extra' in t['seller']}):
            report['extra_seller_deliveries'][seller] = dict(Counter(
                t['grade'] for t in trades if t['seller'] == seller))
        selected = set()
        previous = {}
        for row in rows(folder / 'events.jsonl'):
            if row.get('type') != 'turn':
                continue
            aid, llm = row['agent'], row.get('llm') or {}
            if llm:
                cost = load[llm['base_url']]
                cost['calls'] += 1
                cost['output_tokens'] += llm.get('usage', {}).get('output_tokens', 0)
                cost['agents'].add(aid)
            if agents[aid]['owner'] == 'ta' or row['role'] != 'buyer':
                continue
            out = report['buyers'].setdefault(aid, {
                'turns': 0, 'explicit_state': 0, 'valid_json_state': 0,
                'correct_held_cash': 0, 'missing_state_ticks': [],
                'missing_board_ticks': [], 'repeated_failed_pair_ticks': [],
                'duplicate_attempt_ticks': [], 'no_stock': [], 'not_listed': 0})
            out['turns'] += 1
            args, obs = row.get('args') or {}, row['observation']
            raw = llm.get('raw_arguments')
            try:
                raw = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                raw = None
            explicit = raw.get('private_state') if isinstance(raw, dict) else None
            if isinstance(explicit, str):
                out['explicit_state'] += 1
                try:
                    state = json.loads(explicit)
                except ValueError:
                    state = None
                if isinstance(state, dict):
                    out['valid_json_state'] += 1
                    held = sorted(p for p, n in obs['you']['units_bought'].items() if n > 0)
                    if isinstance(state.get('held'), list) and set(map(str, state['held'])) == set(held) and state.get('cash') == obs['you']['cash']:
                        out['correct_held_cash'] += 1
            else:
                out['missing_state_ticks'].append(row['tick'])
            error = row['result'].get('error')
            out['not_listed'] += error == 'not_listed'
            pair = (args.get('seller_id'), args.get('product_id'))
            board = next((b for b in obs.get('board', []) if
                          (b['seller'], b['product_id']) == pair and b['in_stock'] >= 1), None)
            if row['tool'] == 'buy_listed':
                if board is None:
                    out['missing_board_ticks'].append(row['tick'])
                if obs['you']['units_bought'].get(pair[1], 0) >= 1:
                    out['duplicate_attempt_ticks'].append(row['tick'])
                prev = previous.get(aid)
                if prev and prev['tool'] == 'buy_listed' and prev['result'].get('error') in ('no_stock', 'not_listed'):
                    prev_pair = (prev['args'].get('seller_id'), prev['args'].get('product_id'))
                    if pair == prev_pair:
                        out['repeated_failed_pair_ticks'].append(row['tick'])
            if error == 'no_stock':
                winners = [t for t in trades if t['tick'] == row['tick'] and
                           (t['seller'], t['product']) == pair]
                exhausted = board is not None and sum(t['qty'] for t in winners) >= board['in_stock']
                out['no_stock'].append({
                    'tick': row['tick'], 'pair': pair, 'current_row': board,
                    'classification': 'same_tick_stock_exhausted' if exhausted else
                                      'pair_missing_from_board' if board is None else 'unresolved_with_visible_stock',
                    'same_tick_trades': [{'trade': t['id'], 'buyer': t['buyer'], 'qty': t['qty']} for t in winners]})
            candidates = []
            if row['tool'] == 'buy_listed' and board is None:
                candidates.append(f'{aid}_missing_pair')
            if not isinstance(explicit, str) and aid.endswith('_p2'):
                candidates.append('p2_missing_memory')
            if row['tick'] in out['duplicate_attempt_ticks'] and aid.endswith('_p2'):
                candidates.append('p2_owned_product')
            if error == 'no_stock' and board is not None and aid.endswith('_p1'):
                candidates.append('p1_valid_row_race_control')
            for name in candidates:
                if name in selected or not llm.get('messages'):
                    continue
                selected.add(name)
                replay.append({'name': f'{folder.name}_{name}',
                               'origin': {'folder': str(folder), 'agent': aid, 'tick': row['tick']},
                               'messages': llm['messages'], 'observation': obs,
                               'situation': agents[aid]['situation'], 'tools': agents[aid]['tools'],
                               'contexts': ['recorded', 'fresh']})
            previous[aid] = row
        for aid, out in report['buyers'].items():
            ts = [t for t in trades if t['buyer'] == aid]
            extra = [t for t in ts if 'extra' in t['seller']]
            out['extra_trades'] = len(extra)
            out['extra_grades'] = dict(Counter(t['grade'] for t in extra))
            variable = {r['product_id'] for r in agents[aid]['situation']['market_overview'] if r['quality_varies']}
            out['extra_variable_quality_trades'] = sum(t['product'] in variable for t in extra)
            out['extra_surplus'] = round(sum(t['utility'] - t['price'] * t['qty'] for t in extra) / 100, 2)
            out['low_trades'] = [{**t, 'surplus_dollars': (t['utility'] - t['price'] * t['qty']) / 100}
                                 for t in ts if t['grade'] == 'low']
            out['no_stock_counts'] = dict(Counter(x['classification'] for x in out['no_stock']))
        summary['markets'][folder.name] = report
    summary['load'] = {url: {**value, 'agents': sorted(value['agents'])} for url, value in load.items()}
    (run_dir / 'buyer_failure_audit.json').write_text(json.dumps(summary, indent=2))
    (run_dir / 'failure_replay_cases.json').write_text(json.dumps(replay, ensure_ascii=False))
    print(f'Failure audit: {run_dir}/buyer_failure_audit.json; replay cases={len(replay)}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    audit_failures(parser.parse_args().run_dir)
