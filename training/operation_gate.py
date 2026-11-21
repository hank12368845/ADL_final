#!/usr/bin/env python3
"""CPU evidence table for identical-prompt base/candidate operation diagnostics."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path


def compare(folder, reference='base'):
    if reference not in ('base','step8'):raise ValueError('Unknown comparison reference')
    folder = Path(folder)
    cases_path=folder/'cases.json'
    cases={c['name']:c for c in json.loads(cases_path.read_text())} if cases_path.exists() else {}
    with (folder/'scored_decisions.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    pairs = defaultdict(dict)
    for row in rows:
        label, persona = row['persona'].split('/', 1)
        if label not in (reference, 'candidate'):
            raise ValueError('This gate compares the named reference and candidate with identical personas')
        key = (persona, row['case'], row['context'], row['seed'])
        if label in pairs[key]:
            raise ValueError(f'Duplicate observation/seed/model: {key}')
        pairs[key][label] = row
    grouped = defaultdict(list)
    for key, pair in pairs.items():
        if set(pair) != {reference, 'candidate'}:
            raise ValueError(f'Missing paired result: {key}')
        a, b = pair[reference], pair['candidate']
        if a['input_sha256'] != b['input_sha256'] or a['gpu'] != b['gpu']:
            raise ValueError(f'Inputs/hardware differ within pair: {key}')
        if a['scoring_version'] != 'diagnostic_operations_v1' or b['scoring_version'] != a['scoring_version']:
            raise ValueError('Use the operation rubric for both models')
        grouped[key[0]].append((a, b))
    if not grouped:
        raise ValueError('No paired outcomes')
    table = {}
    for persona, outcomes in sorted(grouped.items()):
        result = {'paired_calls': len(outcomes)}
        changes = Counter()
        for label, index in [(reference, 0), ('candidate', 1)]:
            sample = [pair[index] for pair in outcomes]
            opportunities = [r for r in sample if r['reviewed_evaluation'].get('positive_reference')]
            negatives = [r for r in sample if not r['reviewed_evaluation'].get('positive_reference')]
            attempts = [r for r in sample if r['tool'] in ('buy_listed','accept_offer')]
            result[label] = {
                'service_failures': sum(not r['service_ok'] for r in sample),
                'invalid_actions': sum(not r['action_ok'] for r in sample),
                'policy_failures': sum(not r['policy_ok'] for r in sample),
                'state_failures': sum(not r['state_ok'] for r in sample),
                'positive_opportunities': len(opportunities),
                'captured_opportunities': sum(r['operation_capture'] for r in opportunities),
                'waits_on_positive_opportunities': sum(r['tool']=='wait' for r in opportunities),
                'negative_opportunities':len(negatives),
                'correct_negative_waits':sum(r['tool']=='wait' and r['action_ok'] and r['policy_ok'] for r in negatives),
                'negative_trade_attempts':sum(r['tool'] in ('buy_listed','accept_offer') for r in negatives),
                'trade_attempts':len(attempts),
                'trade_precision':sum(r['operation_capture'] for r in opportunities)/len(attempts) if attempts else None,
                'tools': dict(Counter(r['tool'] for r in sample)),
                'errors': dict(Counter(e for r in sample for e in r['reviewed_evaluation']['errors']))}
            for channel,tool in [('board','buy_listed'),('offer','accept_offer')]:
                subset=[r for r in opportunities if cases.get(r['case'],{}).get('expected_tool')==tool]
                result[label][f'{channel}_positive_opportunities']=len(subset)
                result[label][f'{channel}_captured']=sum(r['operation_capture'] for r in subset)
        for a, b in outcomes:
            before = a['action_ok'] and a['policy_ok']
            after = b['action_ok'] and b['policy_ok']
            changes['fixed' if after and not before else 'regressed' if before and not after else 'tie'] += 1
        result['paired_action_policy'] = dict(changes)
        a, b = result[reference], result['candidate']
        if a['service_failures'] or b['service_failures']:
            result['finding'] = 'service_failures_need_review'
        elif a['captured_opportunities']>0 and b['positive_opportunities']>0 and b['waits_on_positive_opportunities']==b['positive_opportunities']:
            result['finding'] = 'lost_all_captures_by_waiting'
        elif b['invalid_actions'] <= a['invalid_actions'] and b['policy_failures'] <= a['policy_failures'] and b['state_failures'] <= a['state_failures'] and b['captured_opportunities'] >= a['captured_opportunities'] and (b['invalid_actions'] < a['invalid_actions'] or b['policy_failures'] < a['policy_failures'] or b['captured_opportunities'] > a['captured_opportunities']):
            result['finding'] = 'improved_on_this_probe'
        elif b['invalid_actions'] >= a['invalid_actions'] and b['policy_failures'] >= a['policy_failures'] and b['captured_opportunities'] <= a['captured_opportunities'] and (b['invalid_actions'] > a['invalid_actions'] or b['policy_failures'] > a['policy_failures'] or b['captured_opportunities'] < a['captured_opportunities']):
            result['finding'] = 'regressed_on_this_probe'
        else:
            result['finding'] = 'mixed_or_inconclusive'
        table[persona] = result
    valid=not any(v[label]['service_failures'] for v in table.values() for label in (reference,'candidate'))
    output = {'personas': table, 'valid_evaluation':valid,'finding_rule_version':'operation_gate_v2','reference_model':reference,
              'note': 'Small paired diagnostic evidence, not a significance test or market score. '
              'An early negative result does not prove LoRA cannot learn. Inspect error types before extending training.'}
    (folder/'operation_gate.json').write_text(json.dumps(output, indent=2))
    lines = ['# Operation learning evidence', '',
             f'| Persona | Invalid {reference} → candidate | Policy errors {reference} → candidate | Memory errors {reference} → candidate | Captured positive opportunities {reference} → candidate | Offer capture {reference} → candidate | Correct negative waits {reference} → candidate | Finding |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    if not valid:
        lines[2:2]=['**INVALID EVALUATION: service failures prevent comparison of model behavior.**','']
    for persona, value in table.items():
        a, b = value[reference], value['candidate']
        lines.append(f'| {persona} | {a["invalid_actions"]} → {b["invalid_actions"]} | '
                     f'{a["policy_failures"]} → {b["policy_failures"]} | '
                     f'{a["state_failures"]} → {b["state_failures"]} | '
                     f'{a["captured_opportunities"]}/{a["positive_opportunities"]} → '
                     f'{b["captured_opportunities"]}/{b["positive_opportunities"]} | '
                     f'{a["offer_captured"]}/{a["offer_positive_opportunities"]} → '
                     f'{b["offer_captured"]}/{b["offer_positive_opportunities"]} | '
                     f'{a["correct_negative_waits"]}/{a["negative_opportunities"]} → '
                     f'{b["correct_negative_waits"]}/{b["negative_opportunities"]} | {value["finding"]} |')
    lines += ['', output['note']]
    (folder/'operation_gate.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic_run', type=Path)
    parser.add_argument('--reference',choices=['base','step8'],default='base')
    args=parser.parse_args()
    compare(args.diagnostic_run,args.reference)
