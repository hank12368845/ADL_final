"""Strict matched-cash decision report, from existing CPU-scored native tool calls."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path


def case_pairs(cases):
    """Ensure every counterfactual changes only current cash, not history or price."""
    pairs = defaultdict(dict)
    for case in cases:
        origin = case['origin']
        if origin.get('cash_variant') not in ('equal', 'above_one', 'wide', 'zero'):
            raise ValueError('Use the frozen cash suite')
        positive = case['expected_tool'] != 'wait'
        if positive in pairs[origin['pair_id']]:
            raise ValueError('Duplicate cash pair side')
        pairs[origin['pair_id']][positive] = case
    for sides in pairs.values():
        if set(sides) != {True, False}:
            raise ValueError('Incomplete cash pair')
        a, b = sides[True], sides[False]
        def normalize(case):
            observation = copy.deepcopy(case['observation'])
            del observation['you']['cash']
            messages = copy.deepcopy(case['messages'])
            current = json.loads(messages[-1]['content'])
            if messages[-1]['role'] != 'user' or any(current.get(k) != v for k, v in case['observation'].items()):
                raise ValueError('Current prompt and scoring observation differ')
            if set(current)-set(case['observation'])-{'last_action_result', 'your_private_state'}:
                raise ValueError('Unexpected current prompt metadata')
            del current['you']['cash']
            messages[-1]['content'] = json.dumps(current, sort_keys=True)
            return messages, case['tools'], observation, case['contexts']
        if normalize(a) != normalize(b):
            raise ValueError('Counterfactual changes more than current cash')
        tool = a['expected_tool']
        if tool == 'buy_listed':
            price = a['observation']['board'][0]['price']
        elif tool == 'accept_offer':
            offer = a['observation']['offers_to_you'][0]
            price = offer['price']*offer['qty']
        else:
            raise ValueError('Positive cash case must be a trade')
        if not a['observation']['you']['cash'] >= price > b['observation']['you']['cash']:
            raise ValueError('Cash pair does not straddle affordability')
    return dict(pairs)


def _counts(pairs):
    positive = [a for a, b in pairs]
    negative = [b for a, b in pairs]
    captured = sum(r['operation_capture'] for r in positive)
    def correct_wait(r):
        return r['tool'] == 'wait' and r['action_ok'] and r['policy_ok']
    attempts = sum(r['tool'] in ('buy_listed', 'accept_offer') for r in positive+negative)
    return {'pairs': len(pairs), 'positive_captured': captured,
            'negative_correct_waits': sum(correct_wait(r) for r in negative),
            'both_correct': sum(a['operation_capture'] and correct_wait(b) for a, b in pairs),
            'same_tool_both_sides': sum(a['tool'] == b['tool'] for a, b in pairs),
            'wait_both_sides': sum(a['tool'] == b['tool'] == 'wait' for a, b in pairs),
            'unaffordable': sum('unaffordable' in r['reviewed_evaluation']['errors'] for r in negative),
            'trade_attempts': attempts, 'trade_precision': captured/attempts if attempts else None,
            'positive_tools': dict(Counter(r['tool'] for r in positive)),
            'negative_tools': dict(Counter(r['tool'] for r in negative)),
            'memory_failures': sum(not r['state_ok'] for r in positive+negative)}


def write_report(folder):
    folder = Path(folder)
    cases = json.loads((folder/'cases.json').read_text())
    paired_cases = case_pairs(cases)
    by_name = {c['name']: c for c in cases}
    plan = json.loads((folder/'plan.json').read_text())
    labels = set(plan['adapters'])
    expected = {(model, persona, c['name'], context, seed)
                for model in labels for persona in plan['personas'] for c in cases
                for context in c['contexts'] for seed in plan['seeds']}
    seen = set(); rows = []; cash_pairs = defaultdict(dict); prompts = defaultdict(dict)
    for line in (folder/'scored_decisions.jsonl').read_text().splitlines():
        row = json.loads(line)
        model, persona = row['persona'].split('/', 1)
        key = (model, persona, row['case'], row['context'], row['seed'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate cash evaluation result')
        if not row['service_ok'] or row['scoring_version'] != 'diagnostic_operations_v1':
            raise ValueError('Invalid service/rubric; no behavior conclusion')
        seen.add(key); rows.append(row)
        case = by_name[row['case']]
        positive = case['expected_tool'] != 'wait'
        if positive != row['reviewed_evaluation']['positive_reference']:
            raise ValueError('Cash case and scored reference disagree')
        cash_pairs[(model, persona, case['origin']['pair_id'], row['context'], row['seed'])][positive] = row
        prompts[(persona, row['case'], row['context'], row['seed'])][model] = row
    if seen != expected or len(rows) != plan['calls']:
        raise ValueError('Incomplete cash evaluation')
    for values in prompts.values():
        if len({(r['input_sha256'], r['gpu']) for r in values.values()}) != 1:
            raise ValueError('Models used different prompts/hardware')
    grouped = defaultdict(list); details = defaultdict(list)
    for (model, persona, pair_id, context, seed), sides in cash_pairs.items():
        if set(sides) != {True, False}:
            raise ValueError('Missing decision pair side')
        a, b = sides[True], sides[False]
        if a['gpu'] != b['gpu']:
            raise ValueError('Cash sides used different GPUs')
        case = paired_cases[pair_id][True]
        channel = 'offer' if case['expected_tool'] == 'accept_offer' else 'board'
        pair = (a, b); grouped[model].append(pair)
        for dimension, value in (('channel', channel), ('persona', persona),
                                 ('history', case['origin']['history_mode']),
                                 ('variant', case['origin']['cash_variant'])):
            details[(model, dimension, value)].append(pair)
        details[(model, 'channel_variant', channel+'/'+case['origin']['cash_variant'])].append(pair)
    result = {'version': 'cash_decision_pairs_v1', 'calls': len(rows),
              'validation_market_groups': sorted({c['origin']['group'] for c in cases}),
              'gpu_calls': dict(Counter(str(r['gpu']) for r in rows)),
              'models': {model: {'overall': _counts(grouped[model]),
                       **{dimension: {value: _counts(sample) for (m, d, value), sample in details.items()
                                      if m == model and d == dimension}
                          for dimension in ('channel', 'persona', 'history', 'variant', 'channel_variant')}}
                         for model in sorted(labels)},
              'experiment_manifest': plan.get('experiment_manifest'),
              'note': f'{len({c["origin"]["group"] for c in cases})} held-out validation market(s), '
                      'repeated personas/seeds and some duplicate variant inputs. '
                      'Both-correct counts are paired cash decisions, not independent markets or surplus. '
                      'Memory formatting is reported separately and does not define trade capture.'}
    (folder/'cash_decision_pairs.json').write_text(json.dumps(result, indent=2))
    lines = ['# Cash decision pairs', '',
             'Only current cash changes within each pair; price, value, tools and history match. '
             'Both sides and all models use the same GPU.', '',
             '| Model / channel | Positive captured | Negative correct wait | Both correct | Unaffordable |',
             '|---|---:|---:|---:|---:|']
    for model, metrics in result['models'].items():
        for channel, counts in [('all', metrics['overall']), *metrics['channel'].items()]:
            n = counts['pairs']
            lines.append(f'| {model} / {channel} | {counts["positive_captured"]}/{n} | '
                         f'{counts["negative_correct_waits"]}/{n} | {counts["both_correct"]}/{n} | '
                         f'{counts["unaffordable"]} |')
    lines += ['', 'Read JSON for each persona, history, variant and channel × variant.', '', result['note']]
    (folder/'cash_decision_pairs.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return result
