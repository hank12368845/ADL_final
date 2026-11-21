"""Behavior and token-cost gate before continuing a thinking-ON pilot to epochs."""
from collections import Counter
import json
import math
from pathlib import Path

from training.decision_ablation import digest, fingerprint


def compute(folder):
    folder = Path(folder).resolve()
    plan = json.loads((folder/'plan.json').read_text())
    provenance = plan['experiment_manifest']
    if not plan.get('thinking') or provenance['experiment'] != 'buyer_operations_reasoning_v1':
        raise ValueError('Gate requires the thinking-ON reasoning experiment')
    report = json.loads((folder/'cash_decision_pairs.json').read_text())
    if set(report['models']) != {'base_on','candidate'}: raise ValueError('Compare base ON and trained ON')
    rows = [json.loads(s) for s in (folder/'scored_decisions.jsonl').read_text().splitlines()]
    if len(rows) != plan['calls'] or any(not r['service_ok'] or not r.get('thinking') for r in rows):
        raise ValueError('Incomplete or mismatched thinking results')
    token_stats = {}
    for model in report['models']:
        sample = [r for r in rows if r['adapter'] == model]
        tokens = sorted(r.get('usage',{}).get('output_tokens',0)+r.get('usage',{}).get('reasoning_tokens',0) for r in sample)
        token_stats[model] = {'calls':len(sample),'mean_generated_tokens':sum(tokens)/len(tokens),
                             'p95_generated_tokens':tokens[math.ceil(.95*len(tokens))-1],
                             'max_generated_tokens':max(tokens),
                             'reasoning_nonempty':sum(bool(r.get('reasoning')) for r in sample),
                             'truncated':sum(r['stop_reason']=='length' for r in sample),
                             'missing_token_usage':sum(not r.get('usage') for r in sample)}
    baseline, candidate = report['models']['base_on'], report['models']['candidate']
    a, b = baseline['channel']['offer'], candidate['channel']['offer']
    n = b['pairs']; stats = token_stats['candidate']
    checks = {
        'offer_both_correct_increases_by_at_least_4': b['both_correct'] >= a['both_correct']+4,
        'offer_both_correct_at_least_25_percent': b['both_correct'] >= math.ceil(.25*n),
        'offer_unaffordable_decreases': b['unaffordable'] < a['unaffordable'],
        'offer_positive_recall_at_least_90_percent': b['positive_captured'] >= math.ceil(.9*n),
        'overall_positive_capture_preserved_with_tolerance_2': candidate['overall']['positive_captured'] >= baseline['overall']['positive_captured']-2,
        'each_persona_pair_correctness_preserved': all(v['both_correct'] >= baseline['persona'][p]['both_correct'] for p,v in candidate['persona'].items()),
        'no_truncated_calls': stats['truncated']==0,
        'p95_output_with_thinking_at_most_1024': stats['p95_generated_tokens']<=1024,
        'reasoning_visible_on_at_least_90_percent': stats['reasoning_nonempty']>=math.ceil(.9*stats['calls']),
        'token_usage_available': stats['missing_token_usage']==0 and stats['mean_generated_tokens']>0}
    return {'version':'reasoning_pilot_gate_v1','passed':all(checks.values()),'checks':checks,
            'token_stats':token_stats,'candidate_checkpoint':provenance['candidate_checkpoint'],
            'candidate_fingerprint':provenance['candidate_fingerprint'],
            'training_manifest_sha256':provenance['training_manifest_sha256'],
            'evidence_folder':str(folder),'scored_sha256':digest(folder/'scored_decisions.jsonl'),
            'pairs_sha256':digest(folder/'cash_decision_pairs.json'),
            'note':'Development thresholds, not course grades. Two held-out validation markets, repeated personas/seeds. '
                   'p95<=1024 is a six-turn token-cost screen, not an actual round-budget test; full markets remain necessary.'}


def write_report(folder):
    folder = Path(folder); result = compute(folder)
    (folder/'reasoning_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Thinking ON pilot gate','',f'Continue to epochs: **{result["passed"]}**','',
             *[f'- {name}: {value}' for name,value in result['checks'].items()], '',
             'Output tokens include thinking; see JSON for mean / p95 / maximum and truncation.', '', result['note']]
    (folder/'reasoning_gate.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return result


def validate_continuation(path, cfg, checkpoint):
    path = Path(path).resolve(); saved = json.loads(path.read_text()); actual = compute(path.parent)
    if actual != saved or not actual['passed']: raise ValueError('Pilot behavior/token gate did not pass')
    if actual['training_manifest_sha256'] != cfg['data_manifest_sha256']:
        raise ValueError('Epoch continuation uses different training data')
    if fingerprint(checkpoint) != actual['candidate_fingerprint']:
        raise ValueError('Resume adapter is not the behavior-tested candidate')
    return {'path':str(path),'sha256':digest(path),'passed':True}
