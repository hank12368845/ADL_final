"""Read existing transcripts to measure prompt compliance; never repair actions."""
from collections import Counter
import json
from pathlib import Path
import re


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze(folder):
    folder = Path(folder)
    agents = {a['agent']: a for a in rows(folder / 'agents.jsonl')}
    final = json.loads((folder / 'final.json').read_text())
    output = {'agents': {}, 'probe_listings': [], 'probe_low_deliveries': 0}
    for r in rows(folder / 'events.jsonl'):
        if r.get('type') != 'turn':
            continue
        aid, tool, args, obs = r['agent'], r.get('tool'), r.get('args') or {}, r['observation']
        if aid.startswith('quality_probe') and tool in ('set_price', 'make_offer'):
            output['probe_listings'].append({'tick': r['tick'], 'product': args.get('product_id'),
                                            'grade': args.get('grade'), 'ok': r['result'].get('ok')})
        if agents[aid]['controller'] != 'llm':
            continue
        out = output['agents'].setdefault(aid, {'turns': 0, 'violations': Counter(), 'low_trades': 0, 'low_loss': 0})
        out['turns'] += 1
        bad = out['violations']
        if not tool:
            bad['no_tool_call'] += 1
        raw = (r.get('llm') or {}).get('raw_arguments')
        state = None
        try:
            raw = json.loads(raw) if isinstance(raw, str) else raw
            text = raw.get('private_state') if isinstance(raw, dict) else None
            if not isinstance(text, str):
                bad['missing_private_state'] += 1
            else:
                if len(text) > 800:
                    bad['state_over_800'] += 1
                state = json.loads(text)
                if not isinstance(state, dict):
                    raise ValueError('state must encode an object')
        except (ValueError, TypeError):
            state = None
            bad['invalid_state_json'] += 1
        if agents[aid]['role'] == 'seller':
            if state is not None:
                actual = {p: n for p, n in obs['you']['stock'].items() if n > 0}
                if state.get('stock') != actual:
                    bad['stock_snapshot_mismatch'] += 1
            if tool in ('set_price', 'make_offer'):
                pid = args.get('product_id')
                if obs['you']['stock'].get(pid, 0) < args.get('qty', 1):
                    bad['insufficient_current_stock'] += 1
                key = 'your_low_grade_unit_cost' if args.get('grade') == 'low' else 'your_unit_cost'
                cost = agents[aid]['situation'].get(key, {}).get(pid)
                if cost is None or args.get('price', 0) <= cost:
                    bad['nonpositive_margin_or_invalid_grade'] += 1
            continue
        held = {p for p, n in obs['you']['units_bought'].items() if n >= 1}
        if state is not None:
            snapshot = state.get('held')
            if not isinstance(snapshot, list) or set(map(str, snapshot)) != held:
                bad['held_snapshot_mismatch'] += 1
            if state.get('cash') != obs['you']['cash']:
                bad['cash_snapshot_mismatch'] += 1
            if any(k not in state for k in ('pick', 'unowned', 'live', 'afford', 'gain', 'bad', 'pending')):
                bad['state_missing_check_fields'] += 1
        if tool == 'buy_listed':
            if args.get('qty') != 1:
                bad['qty_not_one'] += 1
            if args.get('product_id') in held:
                bad['duplicate_attempt'] += 1
            if not any(b['seller'] == args.get('seller_id') and b['product_id'] == args.get('product_id') and b['in_stock'] >= 1 for b in obs.get('board', [])):
                bad['not_on_current_board'] += 1
        if tool == 'accept_offer':
            offer = next((o for o in obs.get('offers_to_you', []) if o['offer_id'] == args.get('offer_id')), None)
            if not offer or offer['qty'] != 1 or offer['product_id'] in held:
                bad['invalid_or_duplicate_offer'] += 1
        if aid.endswith('_p2') and tool == 'send_message':
            text = args.get('message', '')
            if re.search(r'valuation|utility|budget|\d\s*%|0\.75\s*\*', text, re.I):
                bad['private_info_warning'] += 1
            products = [p for p in obs['you']['utility_of_next_unit'] if re.search(r'\b'+re.escape(p)+r'\b', text)]
            if len(products) == 1:
                cap = .75 * obs['you']['utility_of_next_unit'][products[0]]
                if any(float(q) > cap+.001 for q in re.findall(r'\$(\d+(?:\.\d+)?)', text)):
                    bad['dollar_quote_above_cap'] += 1
    for t in final['trades']:
        if t['seller'].startswith('quality_probe') and t['grade'] == 'low':
            output['probe_low_deliveries'] += 1
        buyer = output['agents'].get(t['buyer'])
        if buyer is not None and t['grade'] == 'low':
            buyer['low_trades'] += 1
            buyer['low_loss'] += max(0, t['price']*t['qty']-t['utility']) / 100
    output['note'] = 'Prompt diagnostics, not enforced guards or strategic grading. Message leakage warnings are heuristic; dollar-cap checks cover explicit dollar quotes only. Mock does not test persona adherence.'
    return output
