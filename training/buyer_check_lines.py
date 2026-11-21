"""Optional compact scratchpad format; CPU preparation only, no new training recipe.

Money uses integer cents. Board identity MUST include seller; an offer ID identifies
its seller through the current observation. Punctuation errors are local to fields.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import re

from training.buyer_self_checks import FIELDS, truth, target_check

ALIASES = {'total': 'total', 'held': 'held', 'val': 'next_value', 'adj': 'adjusted_value',
    'afford': 'affordable', 'eligible': 'eligible'}
ASSIGN = re.compile(r'\b([A-Za-z_]+)\s*=\s*([^\s|,;]+)')
ROW = re.compile(r'\bROW\s*=\s*([^\s|,;]+)', re.IGNORECASE)


def reasoning(text):
    body = text.split('</think>', 1)[0]
    for boundary in ('<tool_call>', '<function='):
        body = body.split(boundary, 1)[0]
    return body.removeprefix('<think>').strip()


def value(raw, boolean=False):
    if boolean:
        if raw.lower() in ('true', '1'): return True
        if raw.lower() in ('false', '0'): return False
        raise ValueError('expected 0/1 or true/false')
    try: number = Decimal(raw)
    except InvalidOperation: raise ValueError('expected integer cents') from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError('expected integer cents')
    return int(number)


def parse(text):
    body = reasoning(text); issues = []; cash = None
    entries = list(ASSIGN.finditer(body)); cash_fields = [m for m in entries if m[1].lower() == 'cash']
    if len(cash_fields) == 1:
        try: cash = value(cash_fields[0][2])
        except ValueError as exc: issues.append('cash: '+str(exc))
    else: issues.append('cash: missing or duplicated')
    headers = list(ROW.finditer(body)); rows = []
    for index, head in enumerate(headers):
        parsed = dict(key=None, fields={}, errors={}); key = head[1].split(':')
        if len(key) == 3 and key[0].lower() in ('board', 'offer') and all(key):
            parsed['key'] = (key[0].lower(), key[1], key[2])
        else: parsed['errors']['row'] = 'expected board:seller:product or offer:offer_id:product'
        segment = body[head.end():headers[index+1].start() if index+1 < len(headers) else len(body)]
        assignments = [m for m in ASSIGN.finditer(segment) if m[1].lower() in ALIASES]
        counts = Counter(ALIASES[m[1].lower()] for m in assignments)
        for m in assignments:
            field = ALIASES[m[1].lower()]
            if counts[field] != 1:
                parsed['errors'][field] = 'duplicated field'; continue
            try: parsed['fields'][field] = value(m[2], field in ('held', 'affordable', 'eligible'))
            except ValueError as exc: parsed['errors'][field] = str(exc)
        rows.append(parsed)
    return dict(cash=cash, rows=rows, issues=issues)


def key(row):
    return row['source'], row.get('offer', row['seller']), row['product']


def render(facts):
    """Training-target candidate ONLY; not appended to evaluation inputs."""
    parts = [f'CASH={facts["cash"]}']
    for row in facts['checks']:
        parts.append('ROW='+':'.join(key(row))+' '+' '.join(
            alias+'='+str(int(row[field]) if type(row[field]) is bool else row[field]) for alias, field in ALIASES.items()))
    return ' | '.join(parts)


def score(case, text):
    facts = truth(case); parsed = parse(text); wanted = key(target_check(case, facts))
    cash_correct = type(parsed['cash']) is int and parsed['cash'] == facts['cash']
    keys = Counter(r['key'] for r in parsed['rows']); rows = []
    for expected in facts['checks']:
        matches = [r for r in parsed['rows'] if r['key'] == key(expected)]
        generated = matches[0]['fields'] if len(matches) == 1 else {}
        rows.append(dict(key=key(expected), target=key(expected) == wanted, correct={
            f: type(generated.get(f)) is type(expected[f]) and generated.get(f) == expected[f] for f in FIELDS if f != 'cash'}))
    coverage = set(keys) == {key(r) for r in facts['checks']} and all(n == 1 for n in keys.values())
    return dict(cash_correct=cash_correct, coverage=coverage, rows=rows, parsed=parsed,
        target_fields=dict(cash=cash_correct, **next(r['correct'] for r in rows if r['target'])),
        exact=cash_correct and coverage and all(all(r['correct'].values()) for r in rows))
