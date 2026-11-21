"""Official seller targets, audited current-rule inputs and whole-market holdouts.

No generated answers or GPU work. The official worlds are older and smaller; the
input adaptation and target exclusions are recorded, not claimed to be grading data.
"""
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from functools import lru_cache

from training.official_buyer_data import BASE, MODEL_REVISION, ROOT, dump, normalize, read_official, sha, situation, schema_error, cents

VERSION = 'seller_official_v1'
PERSONAS = ROOT/'training/persona_versions/seller_checkpoint1_v2'
ALLOWED = {'set_price', 'make_offer', 'send_message', 'post_to_room', 'wait'}


def observation(text):
    # A few exports append human annotations after the JSON. Preserve the source
    # messages for histories; only parse the first complete observation object.
    value = json.JSONDecoder().raw_decode(text.lstrip())[0] if isinstance(text, str) else text
    if not isinstance(value, dict) or 'you' not in value or 'round' not in value:
        raise ValueError('Not a marketplace observation')
    return value


@lru_cache(maxsize=1)
def current_spec():
    import yaml
    from adlmarket.config import MarketSpec
    from adlmarket.setups import deep_merge
    data = yaml.safe_load((ROOT/'configs/economy.yaml').read_text())
    return MarketSpec.model_validate(deep_merge(data['market'], data['phase1']['market']))


def current_tools(sit):
    from adlmarket.tools import build_tools
    return build_tools('seller', current_spec(), bool(sit.get('your_low_grade_unit_cost')))


def current_system(sit, persona):
    from adlmarket.prompts import system_prompt
    spec = current_spec()
    return system_prompt('seller', persona, sit, spec.history_turns,
                         quality=spec.schema_.quality if sit.get('your_low_grade_unit_cost') else None,
                         overview=True, board=True, buyer_values='shared',
                         round_tokens=8192)


def group_id(sit):
    obj = {k: sorted(sit[k]) for k in ('sellers', 'buyers')}
    obj['products'] = sorted(sit['products'], key=lambda p: p['product_id'])
    obj['overview'] = sorted(sit.get('market_overview', []), key=lambda p: p['product_id'])
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:24]


def snapshot_review(sit, obs, tool, args, tools):
    """Snapshot legality and positive quote margin, not optimality or realized profit."""
    errors = []
    if tool not in ALLOWED:
        return dict(valid=False, errors=['unsupported_or_missing_tool'], profitable_quote=False)
    error = schema_error(tool, args, tools)
    if error:
        return dict(valid=False, errors=[error], profitable_quote=False)
    quote = tool in ('set_price', 'make_offer')
    if quote:
        pid = args.get('product_id'); grade = args.get('grade', 'high')
        qty = args.get('qty', 1); stock = obs['you'].get('stock', {}).get(pid, 0)
        costs = sit.get('your_low_grade_unit_cost', {}) if grade == 'low' else sit['your_unit_cost']
        if stock < qty: errors.append('no_stock_or_excess_qty')
        if pid not in costs: errors.append('unknown_product_or_grade')
        elif cents(args['price']) <= cents(costs[pid]): errors.append('nonpositive_margin')
        if round(float(args['price'])*100, 6) != round(float(args['price'])*100):
            errors.append('price_precision')
        if tool == 'make_offer':
            if args['buyer_id'] not in sit['buyers']: errors.append('unknown_buyer')
            if len(obs.get('your_open_offers', [])) >= 8: errors.append('too_many_offers')
        if tool == 'set_price' and obs['round'] == 5 and obs['turns_left_this_round'] == 1:
            errors.append('listing_after_last_buy_turn')
        if tool == 'make_offer' and obs['round'] == 5 and obs['turns_left_this_round'] == 1:
            errors.append('offer_without_next_accept_turn')
    if tool == 'send_message' and args['to'] not in sit['sellers']+sit['buyers']:
        errors.append('unknown_recipient')
    if tool in ('send_message', 'post_to_room') and obs['round'] == 5 and obs['turns_left_this_round'] == 1:
        errors.append('message_after_last_action')
    return dict(valid=not errors, errors=errors, profitable_quote=quote and not errors)


def target_error(sit, obs, answer, tools, result):
    if not result or result.get('ok') is not True: return 'failed_or_missing_result'
    if not 1 <= obs['round'] <= 5 or not 1 <= obs.get('turns_left_this_round', 0) <= 6:
        return 'outside_current_horizon'
    calls = answer.get('tool_calls') or []
    if len(calls) != 1: return 'not_one_tool'
    fn = calls[0]['function']; name = fn['name']; args = fn['arguments']
    review = snapshot_review(sit, obs, name, args, tools)
    if not review['valid']: return review['errors'][0]
    if name == 'make_offer' and args.get('qty') != 1: return 'multiunit_offer_old_demand'
    if name == 'wait' and any(obs['you'].get('stock', {}).values()):
        remaining={p for p,n in obs['you']['stock'].items() if n}
        posted={b['product_id'] for b in obs.get('board',[]) if b.get('seller')==sit['your_id'] and b.get('in_stock',0)>0}
        final=obs['round']==5 and obs['turns_left_this_round']==1
        if not final and not remaining.issubset(posted):return 'wait_with_unlisted_stock'
    reason = answer.get('reasoning_content')
    if not isinstance(reason, str) or not reason.strip(): return 'no_native_reasoning'
    if re.search(r'no public (?:price )?board|(?:round|rounds)\s*(?:[6-9]|1\d)\b|\b(?:8|10|12|16)[ -]round', reason, re.I):
        return 'reasoning_old_rule_reference'
    return None


def split_groups(rows, seed):
    """Assign an entire public market, independent of role/persona/private costs."""
    shapes = defaultdict(set)
    for r in rows: shapes[len(r['situation']['products'])].add(r['group'])
    rng = random.Random(seed); assignment = {}
    for shape, groups in sorted(shapes.items()):
        order = sorted(groups); rng.shuffle(order)
        if len(order) < 3:
            # Never split one rare market across roles/trajectories to fabricate
            # a same-shape holdout. Keep scarce 24-product worlds in train and
            # report that this shape lacks an independent official holdout.
            assignment.update({g:'train' for g in order})
            continue
        n = max(1, len(order)//10)
        for label, part in [('validation', order[:n]), ('test', order[n:2*n]), ('train', order[2*n:])]:
            assignment.update({g: label for g in part})
    return assignment


def select_rows(rows, maximum, seed):
    """Randomized within actions, spread across trajectories, bound wait/chat share."""
    rng = random.Random(seed); pools = defaultdict(list)
    for row in rows: pools[row['label_action']].append(row)
    for pool in pools.values(): rng.shuffle(pool)
    selected = []; fractions = {'set_price': .50, 'make_offer': .35, 'send_message': .08, 'post_to_room': .02, 'wait': .05}
    def take(action, count):
        by_traj = defaultdict(list)
        for r in pools[action]: by_traj[r['trajectory_id']].append(r)
        order = sorted(by_traj); rng.shuffle(order)
        got = []
        while len(got) < count and any(by_traj.values()):
            for tid in order:
                if by_traj[tid] and len(got) < count: got.append(by_traj[tid].pop())
        ids = {r['id'] for r in got}; pools[action] = [r for r in pools[action] if r['id'] not in ids]
        selected.extend(got)
    for action, fraction in fractions.items(): take(action, int(maximum*fraction))
    for action in ('make_offer', 'set_price'): take(action, maximum-len(selected))
    rng.shuffle(selected)
    return selected


def encode(tokenizer, row, max_length):
    answer = row['answer']; reason_tokens = len(tokenizer.encode(answer['reasoning_content'], add_special_tokens=False))
    if reason_tokens > 384: return None, 'reasoning_over_384_tokens'
    messages = copy.deepcopy(row['messages']); dropped = 0
    while True:
        prompt = tokenizer.apply_chat_template(messages, tools=row['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        full = tokenizer.apply_chat_template(messages+[answer], tools=row['tools'], tokenize=False, add_generation_prompt=False, enable_thinking=True)
        if not full.startswith(prompt): raise ValueError('Native template prefix mismatch')
        prefix = tokenizer.encode(prompt, add_special_tokens=False); ids = tokenizer.encode(full, add_special_tokens=False)
        if ids[:len(prefix)] != prefix: raise ValueError('Native token boundary mismatch')
        if len(ids) <= max_length: break
        starts = [i for i,m in enumerate(messages) if m['role'] == 'user']
        if len(starts) < 2: return None, 'system_current_overlength'
        messages = [messages[0], *messages[starts[1]:]]; dropped += 1
    if len(ids)-len(prefix) > 768: return None, 'completion_over_768_tokens'
    row['messages'] = messages; row['history_turns_dropped_for_length'] = dropped
    row['history_turns_shown'] = sum(m['role'] == 'user' for m in messages)-1
    tok = {k:row[k] for k in ('id','trajectory_id','group','label_action','persona_index')}
    tok.update(input_ids=ids, prompt_length=len(prefix), target_length=len(ids)-len(prefix), reasoning_tokens=reason_tokens)
    return tok, None


def generate(cfg):
    from transformers import AutoTokenizer
    from adlmarket.market import load_catalog
    from tqdm import tqdm
    output = ROOT/cfg['data_dir']
    if output.exists(): raise ValueError('Refuse to overwrite prepared data; audit existing manifest or choose new data_dir')
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=MODEL_REVISION, local_files_only=True)
    summaries={p.id:p.summary for p in load_catalog(ROOT/current_spec().catalog)}
    persona_text = {i:(PERSONAS/f'persona_{i}.md').read_text().strip() for i in (1,2,3)}
    counts = Counter(); rows = []; seen = set()
    for episode in tqdm(read_official(ROOT/'training/data/official_raw'), total=2299, desc='CPU official Seller audit', unit='trajectories'):
        if episode['role'] != 'seller': continue
        source_sit = situation(episode['messages'][0]['content']); group = group_id(source_sit)
        sit = copy.deepcopy(source_sit); sit.update(rounds=5, turns_per_round=6)
        # Match the official current engine's product_text=summary. Keep every
        # original numeric/private fact; this is input adaptation, not new targets.
        for product in sit['products']:
            if summaries.get(product['product_id']):
                product.pop('description',None);product['summary']=summaries[product['product_id']]
        persona = int(hashlib.sha256(episode['id'].encode()).hexdigest(),16)%3+1
        system = current_system(sit, persona_text[persona]); tools = current_tools(sit); ms = episode['messages']
        for i,m in enumerate(ms):
            if m['role'] != 'assistant': continue
            counts['source_seller_decisions'] += 1
            if i < 1 or ms[i-1]['role'] != 'user': counts['skip_turn_order'] += 1; continue
            try:
                obs = observation(ms[i-1]['content']); answer = normalize(m)
                nxt = ms[i+1] if i+1<len(ms) else {}
                result = json.loads(nxt['content']) if nxt.get('role') == 'tool' else None
                calls = answer.get('tool_calls') or []
                if len(calls) != 1 or nxt.get('tool_call_id') != calls[0].get('id'): result = None
                error = target_error(sit, obs, answer, tools, result)
            except (ValueError, TypeError, KeyError, AttributeError): error = 'malformed_source'
            if error: counts['skip_'+error] += 1; continue
            starts = [j for j in range(1,i) if ms[j]['role']=='user'][-(cfg['history_turns']+1):]
            try: history = [normalize(x, historical=True) for x in ms[starts[0]:i]]
            except (ValueError, TypeError): counts['skip_history'] += 1; continue
            row = dict(id=episode['id']+f'_turn{i}', trajectory_id=episode['id'], group=group,
                       persona_index=persona, label_action=answer['tool_calls'][0]['function']['name'],
                       situation=sit, observation=obs, messages=[dict(role='system',content=system),*history],
                       tools=tools, answer=answer, source_message_index=i, source_outcome=episode['outcome'],
                       original_tool_result=result, original_rounds=source_sit['rounds'])
            key = hashlib.sha256(dump([row['messages'],answer]).encode()).hexdigest()
            if key in seen: counts['skip_exact_duplicate'] += 1; continue
            encoded,error = encode(tokenizer,row,cfg['max_length'])
            if error: counts['skip_'+error] += 1; continue
            seen.add(key); row['_encoded'] = encoded; rows.append(row); counts['eligible'] += 1
    assignment = split_groups(rows,cfg['seed']); output.mkdir(parents=True)
    splits = {}; groups = {}
    for split in ('train','validation','test'):
        candidates = [r for r in rows if assignment[r['group']] == split]
        maximum = cfg['max_train_examples'] if split=='train' else cfg['max_holdout_examples']
        selected = select_rows(candidates, maximum, cfg['seed']+len(split))
        if len(selected) < (64 if split=='train' else 8): raise ValueError(f'Insufficient official {split}: {len(selected)}')
        with (output/f'{split}.jsonl').open('w') as raw, (output/f'{split}.tokens.jsonl').open('w') as tokens:
            for row in selected:
                tok=row.pop('_encoded'); raw.write(dump(row)+'\n'); tokens.write(dump(tok)+'\n'); row['_encoded']=tok
        lengths=sorted(len(r['_encoded']['input_ids']) for r in selected)
        splits[split]=dict(examples=len(selected),eligible_before_cap=len(candidates),actions=dict(Counter(r['label_action'] for r in selected)),
                           personas=dict(Counter(r['persona_index'] for r in selected)),trajectories=len({r['trajectory_id'] for r in selected}),
                           market_groups=len({r['group'] for r in selected}),mean_tokens=sum(lengths)/len(lengths),max_tokens=max(lengths),
                           product_counts=dict(Counter(len(r['situation']['products']) for r in selected)),
                           history_turns=dict(Counter(r['history_turns_shown'] for r in selected)),
                           raw_sha256=sha(output/f'{split}.jsonl'),tokens_sha256=sha(output/f'{split}.tokens.jsonl'))
        groups[split]=sorted({r['group'] for r in selected})
    paths=[Path(__file__),ROOT/'training/official_buyer_data.py',ROOT/'configs/economy.yaml',ROOT/current_spec().catalog,
           *ROOT.glob('adlmarket/*.py'),*PERSONAS.glob('persona_*.md'),PERSONAS/'manifest.json']
    manifest=dict(version=VERSION,base_model=BASE,revision=MODEL_REVISION,thinking=True,seed=cfg['seed'],groups=groups,splits=splits,
                  preparation_config={k:cfg[k] for k in ('max_length','history_turns','max_train_examples','max_holdout_examples','seed')},
                  sources={str(p.resolve()):sha(p) for p in paths},curation_counts=dict(counts),
                  source=json.loads((ROOT/'training/data/official_raw/source.json').read_text()),
                  adaptation='Current engine Seller rules/tools + original official private/numeric facts, catalog one-line product summaries and current v2 personas; horizon5x6; only native successful reasoning targets. History at most2, oldest complete turns dropped to8192, current observation never truncated. No synthesized target.',
                  limitations='Original worlds include6/12/24 products and old demand/horizon. Shapes with fewer than3 eligible independent markets stay in train, without a same-shape official holdout; current24-product probes are development only. Historical actions and teacher prose may retain old-world habits. Quote validity is not realized sale or optimal profit.')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def audit_data(cfg):
    from training.buyer_operations_train import TokenRows
    folder=ROOT/cfg['data_dir'];m=json.loads((folder/'manifest.json').read_text())
    if m['version']!=VERSION or m['revision']!=MODEL_REVISION: raise ValueError('Wrong Seller dataset')
    if any(sha(p)!=h for p,h in m['sources'].items()): raise ValueError('Seller preparation source/persona changed')
    if any(cfg[k]!=v for k,v in m['preparation_config'].items()): raise ValueError('Dataset preparation recipe changed')
    sets={s:set(m['groups'][s]) for s in m['groups']}
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]): raise ValueError('Market leakage')
    data={}
    for split in sets:
        for suffix,key in [('jsonl','raw_sha256'),('tokens.jsonl','tokens_sha256')]:
            if sha(folder/f'{split}.{suffix}')!=m['splits'][split][key]: raise ValueError('Seller split changed')
        data[split]=TokenRows(folder/f'{split}.tokens.jsonl',cfg['max_length'])
        if any(r['group'] not in sets[split] for r in data[split].metadata): raise ValueError('Wrong market in split')
    return data,m
