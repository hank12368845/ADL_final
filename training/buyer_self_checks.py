"""Visible-state scratchpad supervision and strict, free-generation field scoring.

This module never rewrites observations or supplies facts during inference.
The oracle is used only for offline labels and scoring controlled single-unit cases.
"""
from collections import Counter, defaultdict
import copy
import json
import re

FIELDS = ('cash', 'total', 'held', 'next_value', 'adjusted_value', 'affordable', 'eligible')


def identity(row):
    return row['source'], row['seller'], row['product'], row.get('offer')


def truth(case):
    from training.official_buyer_data import cents
    obs = case['observation']; you = obs['you']; cash = cents(you['cash'])
    varies = {r['product_id']: r.get('quality_varies', False) for r in case['situation']['market_overview']}
    checks = []
    for source, rows in (('board', obs['board']), ('offer', obs.get('offers_to_you', []))):
        for row in rows:
            pid = row['product_id']; qty = row.get('qty', 1)
            held = bool(you['units_bought'].get(pid, 0))
            value = cents(you['utility_of_next_unit'].get(pid, 0))
            # These selected p3 fixtures have no variable-grade history; q=2/3.
            # Other personas' targets are fixed quality. Never use hidden received grade.
            adjusted = value * 4 // 5 if varies[pid] and case['persona_index'] == 3 and not row.get('certified_high') else value
            total = cents(row['price']) * qty
            live = row.get('open_until_round', obs['round']) >= obs['round'] and row.get('in_stock', 1) >= qty
            checks.append(dict(source=source, seller=row['seller'], product=pid,
                **({'offer': row['offer_id']} if source == 'offer' else {}),
                total=total, held=held, next_value=value, adjusted_value=adjusted,
                affordable=cash >= total, eligible=not held and qty == 1 and live and cash >= total and adjusted > total))
    if len({identity(r) for r in checks}) != len(checks):
        raise ValueError('Duplicate current candidate identity')
    return dict(cash=cash, checks=checks)


def target_check(case, facts):
    matches = [r for r in facts['checks'] if r['product'] == case['target_product'] and r['source'] == case['channel']]
    if len(matches) != 1:
        raise ValueError('Target must uniquely identify one current candidate')
    return matches[0]


def teach(case):
    result = copy.deepcopy(case)
    result['answer']['reasoning_content'] = json.dumps(truth(case), separators=(',', ':'), ensure_ascii=False)
    return result


def encode(tokenizer, case):
    from training.official_buyer_data import encode as native_encode
    row = teach(case)
    token, error = native_encode(tokenizer, row)
    if error:
        raise ValueError(f'{case["id"]}: {error}; never trim current observation')
    full = tokenizer.apply_chat_template(row['messages'] + [row['answer']], tools=row['tools'], tokenize=False,
        add_generation_prompt=False, enable_thinking=True)
    prompt = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    reason = row['answer']['reasoning_content']
    if full[len(prompt):len(prompt)+len(reason)] != reason:
        raise ValueError('Scratchpad boundary changed')
    raw = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if raw['input_ids'] != token['input_ids']:
        raise ValueError('Offset/token alignment changed')
    facts = truth(row); target = identity(target_check(row, facts)); masks = []
    starts = [m.start() for m in re.finditer(r'\{"source":', reason)]
    for m in re.finditer(r'"(cash|total|held|next_value|adjusted_value|affordable|eligible)":(-?\d+|true|false)', reason):
        field = m[1]
        candidate = max((i for i, pos in enumerate(starts) if pos <= m.start()), default=-1)
        a, b = len(prompt)+m.start(2), len(prompt)+m.end(2)
        positions = [i-token['prompt_length'] for i, (lo, hi) in enumerate(raw['offset_mapping']) if hi > a and lo < b]
        if not positions or any(i < 0 or i >= token['target_length'] for i in positions):
            raise ValueError('Field supervision outside assistant completion')
        masks.append(dict(field=field, target=field == 'cash' or identity(facts['checks'][candidate]) == target,
            positions=positions, value=json.loads(m[2])))
    if Counter(m['field'] for m in masks) != {'cash': 1, **{f: len(facts['checks']) for f in FIELDS if f != 'cash'}}:
        raise ValueError('Missing field supervision')
    token.update(field_spans=masks, persona_index=row['persona_index'], channel=row['channel'],
        mode=row['mode'], positive=row['positive'], source=row['source'], pair_id=row['pair_id'])
    return row, token


def losses(model, row, device):
    """Balance semantic fields, including positive target, independently of JSON length.

    No tool-name multiplier. Non-target distractors cannot swamp target eligibility.
    Prompt/history labels are absent. Scalar loss is not the behavior acceptance gate.
    """
    import torch
    ids = torch.tensor([row['input_ids']], device=device, dtype=torch.long); p = row['prompt_length']
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
        logits_to_keep=torch.arange(p-1, ids.shape[1]-1, device=device), use_cache=False)
    targets = ids[:, p:]
    if out.logits.shape[:2] != targets.shape:
        raise ValueError('Assistant prediction alignment changed')
    ce = torch.nn.functional.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]), targets.reshape(-1), reduction='none')
    groups = defaultdict(lambda: defaultdict(list))
    for span in row['field_spans']:
        groups[span['field']][span['target']].append(ce[torch.tensor(span['positions'], device=device)].mean())
    by_field = {}
    for field in FIELDS:
        sides = [torch.stack(values).mean() for values in groups[field].values()]
        if not sides:
            raise ValueError('Missing field loss')
        by_field[field] = torch.stack(sides).mean()
    field_loss = torch.stack(list(by_field.values())).mean()
    return dict(total=field_loss + .25 * ce.mean(), completion=ce.mean(), fields=field_loss,
        **{'field_'+k: v for k, v in by_field.items()})


def score(case, text):
    """Strict observable scratchpad scoring; malformed/missing fields count as errors.

    This output format is a training convention, NOT an engine legality requirement.
    No gold reasoning, eligibility, or target identity is inserted into model input.
    """
    expected = truth(case); wanted = identity(target_check(case, expected))
    parsed = None; error = None
    try:
        reason = text.split('</think>', 1)[0].strip()
        if reason.startswith('<think>'):
            reason = reason[len('<think>'):].strip()
        if '</think>' not in text:
            raise ValueError('missing reasoning terminator')
        parsed = json.loads(reason)
        if not isinstance(parsed, dict) or not isinstance(parsed.get('checks'), list):
            raise ValueError('expected cash + checks JSON')
        lookup = {identity(r): r for r in parsed['checks']}
        if len(lookup) != len(parsed['checks']):
            raise ValueError('duplicate candidate')
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc); parsed = None; lookup = {}
    def equal(got, gold):
        return type(got) is type(gold) and got == gold
    cash = parsed is not None and equal(parsed.get('cash'), expected['cash'])
    per_candidate = []
    for row in expected['checks']:
        got = lookup.get(identity(row), {})
        per_candidate.append(dict(identity=list(identity(row)), target=identity(row) == wanted,
            correct={f: equal(got.get(f), row[f]) for f in FIELDS if f != 'cash'}))
    coverage = parsed is not None and set(lookup) == {identity(r) for r in expected['checks']}
    exact = cash and coverage and all(all(r['correct'].values()) for r in per_candidate)
    target = next(r for r in per_candidate if r['target'])
    return dict(format_valid=parsed is not None, format_error=error, cash_correct=cash, candidate_coverage=coverage,
        exact=exact, target_fields=dict(cash=cash, **target['correct']), candidates=per_candidate,
        expected=expected, generated=parsed)


def summarize(rows):
    result = {'calls': len(rows), 'format_valid': 0, 'coverage': 0, 'all_fields_exact': 0,
        'target_fields': {f: 0 for f in FIELDS}, 'all_candidate_fields': {f: {'correct': 0, 'total': 0} for f in FIELDS},
        'positive_total': 0, 'captured': 0, 'negative_total': 0, 'negative_safe': 0,
        'cash_errors': 0, 'owned_errors': 0, 'no_tool': 0, 'truncated': 0, 'tool_valid_when_fields_exact': 0}
    pairs = defaultdict(list)
    for r in rows:
        s = r['self_check']; result['format_valid'] += s['format_valid']; result['coverage'] += s['candidate_coverage']
        result['all_fields_exact'] += s['exact']
        for f in FIELDS:
            result['target_fields'][f] += s['target_fields'][f]
            vals = [s['cash_correct']] if f == 'cash' else [c['correct'][f] for c in s['candidates']]
            result['all_candidate_fields'][f]['correct'] += sum(vals)
            result['all_candidate_fields'][f]['total'] += len(vals)
        result['positive_total'] += r['positive'] is True; result['captured'] += r['captured']
        result['negative_total'] += r['positive'] is False
        result['negative_safe'] += r['positive'] is False and r['valid']
        result['cash_errors'] += r['cash_error']; result['owned_errors'] += 'already_owned_trade' in r['errors']
        result['no_tool'] += r['tool'] is None; result['truncated'] += r['truncated']
        result['tool_valid_when_fields_exact'] += s['exact'] and r['valid']
        pairs[r['pair_id']].append(r)
    complete = [p for p in pairs.values() if len(p) == 2 and {r['positive'] for r in p} == {True, False}]
    result['pairs'] = len(complete)
    result['both_fields_and_tool_correct_pairs'] = sum(all(r['self_check']['exact'] and r['valid'] and (not r['positive'] or r['captured']) for r in p) for p in complete)
    return result
