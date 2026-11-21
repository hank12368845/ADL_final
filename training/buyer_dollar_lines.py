"""Dollar-denominated one-candidate checks, independently supervised and scored."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import copy
import json
import re

from training.buyer_check_lines import ALIASES, ASSIGN, ROW, key, reasoning, parse as parse_cents
from training.buyer_self_checks import FIELDS, truth, target_check

MONEY = {'cash', 'total', 'val', 'adj'}


def dollar(n):
    return format(Decimal(n)/100, '.2f')


def parse(text):
    """Compare exact dollar decimals in integer internal units; no rounded credit.

    The model writes dollars, never converts units. A prediction of 1959 dollars
    remains wrong when cash is 19.59; unsupported precision fails only that field.
    """
    def convert(match):
        if match[1].lower() not in MONEY: return match[0]
        try:
            amount = Decimal(match[2])*100
            if not amount.is_finite() or abs(amount) > 10**12 or amount != amount.to_integral_value():
                raise ValueError('not a cent-exact dollar amount')
            return match[1]+'='+str(int(amount))
        except (InvalidOperation, ValueError): return match[1]+'=INVALID_DOLLARS'
    return parse_cents(ASSIGN.sub(convert, reasoning(text)))


def render(case):
    facts = truth(case); r = target_check(case, facts)
    fields = ' '.join(alias+'='+(str(int(r[f])) if type(r[f]) is bool else dollar(r[f])) for alias, f in ALIASES.items())
    return 'CASH='+dollar(facts['cash'])+' | ROW='+':'.join(key(r))+' '+fields


def teach(case):
    row = copy.deepcopy(case); row['answer']['reasoning_content'] = render(case)
    validate_gold(row)
    return row


def state_errors(case, tool, args):
    """Persona consistency, NOT additional engine legality requirements."""
    from training.official_buyer_data import cents
    errors = []; you = case['observation']['you']
    if not isinstance(args,dict): return ['state_not_json_object']
    try:
        s = json.loads(args.get('private_state', ''))
        if not isinstance(s, dict): raise ValueError()
    except (ValueError, TypeError, AttributeError): return ['state_not_json_object']
    def money_matches(got,expected):
        try: return type(got) in (int,float) and cents(got)==expected
        except (ValueError,OverflowError,InvalidOperation): return False
    if not money_matches(s.get('cash'),cents(you['cash'])): errors.append('state_cash')
    held = {p for p, n in you['units_bought'].items() if n}
    if not isinstance(s.get('held'), list) or not all(isinstance(p,str) for p in s['held']) or set(s['held']) != held: errors.append('state_held')
    if tool in ('buy_listed', 'accept_offer'):
        r = target_check(case, truth(case))
        if s.get('pick') != r['product']: errors.append('state_pick')
        for name, expected in (('unowned', not r['held']), ('live', True), ('afford', r['affordable'])):
            if type(s.get(name)) is not bool or s[name] != expected: errors.append('state_'+name)
        if not money_matches(s.get('gain'),r['adjusted_value']-r['total']): errors.append('state_gain')
    elif tool == 'wait':
        if any(s.get(f) is not None for f in ('pick', 'unowned', 'live', 'afford', 'gain')): errors.append('state_wait_flags')
    return errors


def validate_gold(case):
    from training.official_buyer_train import official_snapshot_review
    fn = case['answer']['tool_calls'][0]['function']; tool, args = fn['name'], fn['arguments']
    facts = truth(case); r = target_check(case, facts)
    if r['eligible'] != case['positive'] or sum(c['eligible'] for c in facts['checks']) != int(case['positive']):
        raise ValueError('Controlled target eligibility changed')
    expected = ('buy_listed' if case['channel'] == 'board' else 'accept_offer') if case['positive'] else 'wait'
    if tool != expected: raise ValueError('Check verdict disagrees with target tool')
    if tool == 'buy_listed' and (args['seller_id'], args['product_id'], args['qty']) != (r['seller'], r['product'], 1):
        raise ValueError('Board tool is not the checked row')
    if tool == 'accept_offer' and args['offer_id'] != r['offer']: raise ValueError('Offer tool is not the checked offer')
    if state_errors(case, tool, args): raise ValueError('Gold private_state disagrees with current state')
    if not official_snapshot_review(case, tool, args)['valid_snapshot']: raise ValueError('Unsafe gold tool')
    return True


def encode(tokenizer, case):
    from training.official_buyer_data import encode as native_encode
    row = teach(case); token, error = native_encode(tokenizer, row)
    if error: raise ValueError(f'{row["id"]}: {error}; current input will not be truncated')
    prompt = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full = tokenizer.apply_chat_template(row['messages']+[row['answer']], tools=row['tools'], tokenize=False, enable_thinking=True)
    raw = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if raw['input_ids'] != token['input_ids'] or not full.startswith(prompt+render(row)):
        raise ValueError('Native dollar target/token boundary changed')
    p = token['prompt_length']; a = len(prompt); b = full.index('<tool_call>', a)
    check = [i-p for i, (lo, hi) in enumerate(raw['offset_mapping']) if hi > a and lo < b]
    tool = [i for i in range(token['target_length']) if i not in check]
    spans = []
    mapping = {'cash': 'cash', **ALIASES}
    for match in ASSIGN.finditer(render(row)):
        if match[1].lower() not in mapping: continue
        start, end = a+match.start(2), a+match.end(2)
        positions = [i-p for i, (lo, hi) in enumerate(raw['offset_mapping']) if hi > start and lo < end]
        if not positions or not set(positions) <= set(check): raise ValueError('Wrong dollar field mask')
        spans.append(dict(field=mapping[match[1].lower()], positions=positions))
    if {s['field'] for s in spans} != set(FIELDS) or not check or not tool or check[0] != 0:
        raise ValueError('Incomplete independent line/tool supervision')
    token.update(check_positions=check, response_positions=tool, field_spans=spans,
        **{k: row[k] for k in ('persona_index', 'channel', 'mode', 'positive', 'pair_id')})
    return row, token


def losses(model, row, device):
    import torch
    ids = torch.tensor([row['input_ids']], device=device, dtype=torch.long); p = row['prompt_length']
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
        logits_to_keep=torch.arange(p-1, ids.shape[1]-1, device=device), use_cache=False)
    if out.logits.shape[:2] != ids[:, p:].shape: raise ValueError('Misaligned dollar targets')
    ce = torch.nn.functional.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]), ids[:, p:].reshape(-1), reduction='none')
    mean = lambda positions: ce[torch.tensor(positions, device=device)].mean()
    line, response = mean(row['check_positions']), mean(row['response_positions'])
    fields = {s['field']: mean(s['positions']) for s in row['field_spans']}
    field = torch.stack(list(fields.values())).mean()
    return dict(total=line+field+.25*response, line=line, fields=field, response=response,
        **{'field_'+k: v for k, v in fields.items()})


def score(case, text):
    facts = truth(case); target = target_check(case, facts); parsed = parse(text)
    matches = [r for r in parsed['rows'] if r['key'] == key(target)]
    got = matches[0]['fields'] if len(matches) == 1 else {}
    correct = {'cash': type(parsed['cash']) is int and parsed['cash'] == facts['cash']}
    correct.update({f: type(got.get(f)) is type(target[f]) and got.get(f) == target[f] for f in FIELDS if f != 'cash'})
    allowed = {key(r) for r in facts['checks']}; unknown = [r['key'] for r in parsed['rows'] if r['key'] not in allowed]
    started = re.match(r'^CASH\s*=', reasoning(text), re.IGNORECASE) is not None
    return dict(self_started=started, candidate_found=len(matches)==1, target_fields=correct,
        exact=started and not unknown and all(correct.values()), unknown_rows=unknown,
        expected=dict(cash=facts['cash'], candidate=target), parsed=parsed)


def assess(case, text, tokenizer, truncated, rank, cached=False):
    from training.step84_repair_plan import assess as native_assess
    r = native_assess(case, 'latest', text, tokenizer, truncated, rank, cached)
    r.update(persona_index=case['persona_index'], mode=case.get('mode'))
    if case['suite'].startswith('self_check_'):
        r['line_check'] = score(case, text)
        r['state_errors'] = state_errors(case, r['tool'], r['arguments'])
        eligible = r['line_check']['parsed']['rows']
        chosen = [p['fields'].get('eligible') for p in eligible if p['key'] == key(target_check(case, truth(case)))]
        flag = chosen[0] if len(chosen)==1 else None
        r['line_action_consistent'] = bool(r['valid'] and ((flag is True and r['captured']) or (flag is False and r['tool'] not in ('buy_listed', 'accept_offer'))))
    return r


def summarize(rows):
    result = dict(calls=len(rows), self_started=0, all_fields_exact=0, target_fields={f: 0 for f in FIELDS},
        captured=0, positive_total=0, negative_safe=0, negative_total=0, cash_errors=0, owned_errors=0,
        no_tool=0, truncated=0, state_consistent=0, line_action_consistent=0, cash_negative_safe=0,
        owned_negative_safe=0, cash_negative_total=0, owned_negative_total=0)
    for r in rows:
        c = r['line_check']; result['self_started'] += c['self_started']; result['all_fields_exact'] += c['exact']
        for f in FIELDS: result['target_fields'][f] += c['target_fields'][f]
        result['captured'] += r['captured']; result['positive_total'] += r['positive'] is True
        result['negative_total'] += r['positive'] is False; result['negative_safe'] += r['positive'] is False and r['valid']
        result['cash_errors'] += r['cash_error']; result['owned_errors'] += 'already_owned_trade' in r['errors']
        result['no_tool'] += r['tool'] is None; result['truncated'] += r['truncated']
        result['state_consistent'] += not r['state_errors']; result['line_action_consistent'] += r['line_action_consistent']
        for mode in ('cash', 'owned'):
            subset = r['positive'] is False and r['mode'] == mode
            result[mode+'_negative_total'] += subset; result[mode+'_negative_safe'] += subset and r['valid']
    return result


def step6_gate(summary):
    return dict(seen12_complete=summary['calls']==12, self_started12=summary['self_started']==12,
        fields_exact12=summary['all_fields_exact']==12, retain6_positive=summary['captured']==6,
        cash3_safe=summary['cash_negative_total']==3 and summary['cash_negative_safe']==3,
        held3_safe=summary['owned_negative_total']==3 and summary['owned_negative_safe']==3,
        zero_error_or_truncation=all(summary[k]==0 for k in ('cash_errors','owned_errors','no_tool','truncated')),
        consistent_state_and_action=summary['state_consistent']==12 and summary['line_action_consistent']==12)
