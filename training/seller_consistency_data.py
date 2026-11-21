"""CPU-only Seller consistency repair; immutable v1, fixed market splits, audited controls.

Official completions are retained verbatim or quarantined, never silently repaired.
Controlled examples are separately labelled and engine checked; no profitability claim.
"""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT_PATH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_PATH))

from training.official_buyer_data import ROOT,BASE,MODEL_REVISION,normalize,read_official,dump,sha,cents
from training.official_seller_data import observation,current_spec,current_system,encode,select_rows

V1=ROOT/'training/data/seller_official_v1'
DATA=ROOT/'training/data/seller_consistency_v2'
PARENT=ROOT/'training/runs/20261121-135008_seller_official_9d95_ubx/checkpoints/step_0012'
SEED=20261123
ROUND=re.compile(r'\b(?:rounds?\s*|R)(\d+)(?=\b|T\d)|\b(\d+)[ -]round',re.I)


def horizon_findings(text,round_=1,check_relative=True):
    """Conservative risk screening, not a claim of full natural-language correctness."""
    findings=[]
    for m in ROUND.finditer(text or ''):
        n=int(m[1] or m[2])
        if n<=5:continue
        before=text[max(0,m.start()-100):m.start()];after=text[m.end():m.end()+90]
        if re.search(r'\b(?:no|not|never|without)\s+(?:(?:a|another|any|sixth)\s+)?$',before,re.I):continue
        # Expiry may legitimately be R6 even though the market ends at R5.
        expiry=bool(re.search(r'exp(?:ires?|iry)?(?:\s+(?:the|end|of|round))*\s*$',before,re.I))
        expiry=expiry or bool(re.search(r'(?:until|till|through)(?:\s+(?:the|end|of|round))*\s*$',before,re.I)
                              and re.search(r'\b(?:offer\w*|open|o\d+)\b',before,re.I))
        parenthesized_offer=bool(re.search(r'(?:o\d+|offer)[^;\n]{0,65}\(\s*$',before,re.I))
        if n<=6 and (expiry or parenthesized_offer):
            findings.append(dict(kind='allowed_offer_expiry',round=n,context=before[-90:]+m[0]+after[:60]));continue
        findings.append(dict(kind='future_round_risk',round=n,context=before[-90:]+m[0]+after[:60]))
    if check_relative and round_==5 and re.search(r'\bnext round\b',text or '',re.I):
        for m in re.finditer(r'\bnext round\b',text,re.I):
            before=text[max(0,m.start()-35):m.start()]
            if not re.search(r'\b(?:no|not|never|without)\s+(?:a\s+)?$',before,re.I):
                findings.append(dict(kind='next_round_after_close_risk',context=text[max(0,m.start()-70):m.end()+90]))
    return findings


def stock_assertions(note):
    """Only explicit own-inventory blocks; do not treat price/offer quantities as stock."""
    result=[]
    for marker in re.finditer(r'\b(?:stock(?:\s+left)?|inventory|left)\s*:',note or '',re.I):
        tail=note[marker.end():]
        stop=re.search(r'\b(?:next|open|offers?|plan|profit|costs?|board|rating|demand|if)\s*[:=]?|\.(?=\s+[A-Z])',tail,re.I)
        block=tail[:stop.start()] if stop else tail
        for m in re.finditer(r'\b(P\d+)\s*(?:[x:=]\s*|\(\s*)(\d+)(?=[\s,;@)]|\.(?:\s|$)|$)|\b(P\d+)\s+(\d+)(?=\s*(?:u\b|@|[,;]|\.(?:\s|$)|$))',block,re.I):
            result.append((m[1] or m[3],int(m[2] or m[4])))
    return result


def own_texts(answer):
    yield 'reasoning',answer.get('reasoning_content') or ''
    # Public negotiation may bluff; do not enforce truth/horizon on messages.
    for call in answer.get('tool_calls') or []:
        args=call['function']['arguments']
        if isinstance(args,str):args=json.loads(args)
        yield 'private_state',args.get('private_state','')


def row_findings(row):
    risks=[];allowed=[];checked=[];current=row['observation'];round_=current['round']
    for field,text in own_texts(row['answer']):
        for f in horizon_findings(text,round_):
            (allowed if f['kind']=='allowed_offer_expiry' else risks).append(dict(field='target.'+field,**f))
        if field=='private_state':
            for pid,n in stock_assertions(text):
                checked.append(dict(product=pid,claimed=n,current=current['you']['stock'].get(pid)))
                if current['you']['stock'].get(pid)!=n:risks.append(dict(kind='target_inventory_assertion_mismatch',product=pid,claimed=n,current=current['you']['stock'].get(pid)))
    historical=None
    for m in row['messages'][1:]:
        if m['role']=='user':
            historical=observation(m['content'])
            # Stale quantity is normal input; the target must follow CURRENT stock.
            # A carried note saying "End R4; next round" is valid on arrival
            # in R5. Its writing time is not necessarily the input's round.
            for f in horizon_findings(historical.get('your_private_state',''),historical['round'],check_relative=False):
                (allowed if f['kind']=='allowed_offer_expiry' else risks).append(dict(field='input.own_private_state',**f))
        elif m['role']=='assistant':
            for field,text in own_texts(m):
                for f in horizon_findings(text,(historical or current)['round']):
                    (allowed if f['kind']=='allowed_offer_expiry' else risks).append(dict(field='history.'+field,**f))
    return dict(risks=risks,allowed_expiries=allowed,inventory_assertions_checked=checked,
                caveat='Risk quarantine, not a semantic accuracy score; ordinary stale input quantity and public bluff are not errors.')


def engine_for(row):
    from adlmarket.market import Market,Product,SellerBook,BuyerBook,Offer
    from adlmarket.schema import Schema,ProductPlan
    sit,obs=row['situation'],row['observation'];aid=sit['your_id'];cost={p:cents(c) for p,c in sit['your_unit_cost'].items()}
    low={p:cents(c) for p,c in sit.get('your_low_grade_unit_cost',{}).items()}
    products=[Product(id=p['product_id'],title=p['title'],category=p.get('type',''),ref_price=cents(p['list_price'])) for p in sit['products']]
    plans=tuple(ProductPlan(p.id,'control',1.,None,(),cost[p.id],p.id in low,low.get(p.id)) for p in products)
    schema=Schema('seller_snapshot_control',plans,0,.4)
    sellers=[SellerBook(s,dict(obs['you']['stock']) if s==aid else {p.id:0 for p in products},dict(cost),dict(low),initial_stock=dict(sit['your_stock'])) for s in sit['sellers']]
    # Counterparty utilities/cash are not used by quote/wait validation and never
    # inserted into the model input. These are snapshot tool checks, not markets.
    buyers=[BuyerBook(b,0,0,{}) for b in sit['buyers']]
    market=Market('seller_snapshot_control',current_spec(),schema,products,sellers,buyers);market.round=obs['round']
    for o in obs.get('your_open_offers',[]):
        market.offers[o['offer_id']]=Offer(o['offer_id'],aid,o['buyer'],o['product_id'],cents(o['price']),o['qty'],obs['round'],o['open_until_round'],o.get('grade','high'))
    return market


def verify_target(row):
    from training.official_buyer_data import schema_error
    calls=row['answer'].get('tool_calls') or []
    if len(calls)!=1:raise ValueError('Exactly one native target call required')
    fn=calls[0]['function'];name=fn['name'];args=fn['arguments'];err=schema_error(name,args,row['tools'])
    if err:raise ValueError((row['id'],err))
    obs,sit=row['observation'],row['situation'];pid=args.get('product_id')
    if any(type(n)!=int or n<0 or n>sit['your_stock'].get(p,0) for p,n in obs['you']['stock'].items()):raise ValueError('Current stock exceeds initial allocation or is invalid')
    if name in ('set_price','make_offer'):
        if obs['you']['stock'].get(pid,0)<args.get('qty',1):raise ValueError('Target ignores current stock')
        costs=sit.get('your_low_grade_unit_cost',{}) if args.get('grade')=='low' else sit['your_unit_cost']
        if cents(args['price'])<=cents(costs[pid]):raise ValueError('Target has nonpositive margin')
        if obs['round']==5 and obs['turns_left_this_round']==1:raise ValueError('Late quote has no next buying turn')
    result=engine_for(row).apply(sit['your_id'],name,args)
    if not result.ok:raise ValueError((row['id'],result.as_dict()))


def encode_row(tokenizer,row,maximum=32768):
    # Same native completion CE and EOS as v1; only complete old history turns
    # may be trimmed. No newly required think/JSON/checklist format.
    encoded,error=encode(tokenizer,row,maximum)
    if error:raise ValueError((row['id'],error))
    encoded.update(family=row['family'],source_kind=row['source_kind'],source_sample_id=row['source_sample_id'])
    return encoded


def control(source,family,index,persona=None):
    r=copy.deepcopy(source);r['source_kind']='engine_control';r['source_sample_id']=source.get('source_sample_id',source['id'])
    r['family']=family;r['id']=source['id']+'__'+family;r['controlled_intervention']=True
    if persona is not None:
        r['persona_index']=persona
        text=(ROOT/f'training/persona_versions/seller_checkpoint1_v2/persona_{persona}.md').read_text()
        r['messages'][0]['content']=current_system(r['situation'],text)
    obs=r['observation'];sit=r['situation'];fn=source['answer']['tool_calls'][0]['function'];a=fn['arguments']['product_id']
    alternatives=[p for p in sit['products'] if p['product_id']!=a and sit['your_stock'].get(p['product_id'],0)>=1 and cents(p['list_price'])>cents(sit['your_unit_cost'][p['product_id']])]
    b=max(alternatives,key=lambda p:cents(p['list_price'])-cents(sit['your_unit_cost'][p['product_id']]))
    price=fn['arguments']['price'];grade=fn['arguments'].get('grade','high');name=fn['name']
    args={k:v for k,v in fn['arguments'].items() if k not in ('private_state','message')}
    if name=='make_offer':args['qty']=1
    obs['you']['stock']={p:0 for p in obs['you']['stock']};obs['your_open_offers']=[]
    obs['board']=[p for p in obs.get('board',[]) if p['seller']!=sit['your_id']]
    obs['new_messages']=[];obs['output_tokens_left_this_round']=8192
    stale=f'Old plan: {a} has 3 units; list {a} at {price:.2f}. Check current observation before acting.'
    obs['your_private_state']=stale
    if family=='live_inventory':
        obs['you']['stock'][a]=1
        reason=f'The current stock has one {a}. The earlier note says three, so I use the current quantity. Quote one unit at {price:.2f}, above its grade cost.'
        note=f'Current {a}=1; quote {price:.2f}; no sale is assumed.'
    elif family=='soldout_switch':
        pid=b['product_id'];obs['you']['stock'][pid]=1;name='set_price';args=dict(product_id=pid,price=b['list_price'],grade='high')
        reason=f'{a} is sold out in the current stock. The old note is stale. {pid} still has one unit; post it at {b["list_price"]:.2f}, above cost.'
        note=f'Current {a}=0; {pid}=1. Use current stock rather than the old plan.'
    elif family=='soldout_wait':
        name='wait';args={};reason='All current stock quantities are zero. The old note is stale. There is nothing left to supply, so wait.'
        note='Current stock is empty. No further sale or replenishment is assumed.'
    elif family=='final_round':
        obs['round']=5;obs['turns_left_this_round']=1 if index%2==0 else 5;obs['you']['stock'][a]=1
        obs['your_private_state']=f'Old plan: wait until R6 to lower {a}. This note is outdated; the market has five rounds.'
        if index%2==0:
            name='wait';args={};reason='This is the last market turn. Existing board terms can still sell, but new terms arrive too late. Wait; there is no later round.'
            note='Final market turn. No later round is planned.'
        else:
            reason=f'Round five is active with five turns left. Current {a} stock is one. Quote now at {price:.2f}; do not defer action beyond the five-round market.'
            note=f'R5 still active; current {a}=1; quote {price:.2f} now.'
    else:raise ValueError(family)
    args['private_state']=note
    r['answer']=dict(role='assistant',content='',reasoning_content=reason,tool_calls=[dict(id='answer',type='function',function=dict(name=name,arguments=args))])
    r['label_action']=name;r['expected_quote']=name in ('set_price','make_offer');r['messages'][-1]['content']=json.dumps(obs,ensure_ascii=False)
    # Deliberately stale INPUT memory is labelled, unlike accidental old-world
    # plans in official replay. Only the correct output is supervised.
    r['input_memory_is_intentionally_stale']=True
    for field,text in own_texts(r['answer']):
        if any(f['kind']!='allowed_offer_expiry' for f in horizon_findings(text,obs['round'])):raise ValueError('Control target has a horizon conflict')
    verify_target(r);return r


def prepare(output=DATA):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.official_seller_data import audit_data
    import yaml
    old_cfg=yaml.safe_load((ROOT/'training/configs/seller_official_v1.yaml').read_text());_,old_manifest=audit_data(old_cfg)
    output=Path(output).resolve()
    if output.exists():raise ValueError('Refuse overwrite; audit existing data or choose a new output')
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,local_files_only=True)
    original={s:[json.loads(l) for l in (V1/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation','test')}
    kept={};quarantine=[];allowed=[]
    for split,rows in original.items():
        kept[split]=[]
        for row in rows:
            review=row_findings(row)
            if review['risks']:quarantine.append(dict(split=split,id=row['id'],**review));continue
            verify_target(row);r=copy.deepcopy(row);r.update(source_kind='official_replay',source_sample_id=row['id'],family='official_replay')
            kept[split].append(r)
            if review['allowed_expiries']:allowed.append(dict(split=split,id=row['id'],findings=review['allowed_expiries']))
    if len(kept['train'])<216 or len(kept['validation'])<6:raise ValueError(f'Insufficient clean official pool:{ {s:len(v) for s,v in kept.items()} }')
    # Rebuild some FULL six-turn contexts from the original source, without
    # contaminating v1 files. A risk in the extra history excludes that long variant.
    by_id={r['trajectory_id']:None for r in kept['train']+kept['validation']}
    for episode in read_official(ROOT/'training/data/official_raw'):
        if episode['id'] in by_id:by_id[episode['id']]=episode
    long=[]
    for r in tqdm(kept['train'],desc='CPU audit full Seller history',unit='rows'):
        source=by_id[r['trajectory_id']];i=r['source_message_index'];ms=source['messages'];starts=[j for j in range(1,i) if ms[j]['role']=='user'][-7:]
        if len(starts)<5:continue
        variant=copy.deepcopy(r);variant['messages']=[r['messages'][0],*[normalize(m,historical=True) for m in ms[starts[0]:i]]]
        if row_findings(variant)['risks']:continue
        encoded=encode_row(tokenizer,variant)
        if len(encoded['input_ids'])>=10000:variant['_encoded']=encoded;long.append(variant)
    rng=random.Random(SEED);rng.shuffle(long);chosen_long=long[:18]
    reserved={r['source_sample_id'] for r in chosen_long}
    native=chosen_long+select_rows([r for r in kept['train'] if r['id'] not in reserved],216-len(chosen_long),SEED)
    if len(native)!=216:raise ValueError('Native replay allocation incomplete')
    # Eighteen independent source snapshots × four conditional variants. Spread
    # group/trajectory and persona; only train-world facts enter these examples.
    candidates=[r for r in kept['train'] if r['label_action'] in ('set_price','make_offer') and len(r['situation']['products'])>1]
    rng.shuffle(candidates);contexts=[];used=set()
    for r in candidates:
        if r['trajectory_id'] in used:continue
        used.add(r['trajectory_id']);contexts.append(r)
        if len(contexts)==18:break
    if len(contexts)!=18:raise ValueError('Insufficient distinct official source trajectories for repair')
    controls=[]
    for i,r in enumerate(contexts):
        controls.extend(control(r,f,i,persona=i%3+1) for f in ('live_inventory','soldout_switch','soldout_wait','final_round'))
    train=native+controls
    for r in tqdm(train,desc='CPU tokenize audited Seller targets',unit='rows'):
        if '_encoded' not in r:r['_encoded']=encode_row(tokenizer,r)
    rng.shuffle(native);batches=[]
    # Exactly 6 original official targets + 2 controlled targets per global batch.
    for i in range(36):batches.append([train.index(r) for r in native[6*i:6*i+6]]+list(range(216+2*i,216+2*i+2)))
    # Evaluation worlds remain in their original market split. No sealed-test
    # labels or known seed5 failures are used to construct training targets.
    val_contexts=[r for r in kept['validation'] if r['label_action'] in ('set_price','make_offer')]
    if len(val_contexts)<3:raise ValueError('Missing held-out quote contexts')
    cases=[]
    for i,r in enumerate(val_contexts[:3]):
        for family in ('live_inventory','soldout_switch','soldout_wait','final_round'):
            c=control(r,family,i,persona=i+1);c.update(name='unseen_'+c['id']+f'_p{i+1}',kind='heldout_controls');cases.append(c)
    parent_cases=json.loads((PARENT.parent.parent/'native_cases.json').read_text())
    cases += [dict(c,kind='positive_regression',expected_quote=True) for c in parent_cases if c['kind']=='current_rule' and c.get('quote_opportunity')]
    cases += [dict(name='official_'+r['id'],kind='official_heldout',messages=r['messages'],tools=r['tools'],situation=r['situation'],observation=r['observation'],persona_index=r['persona_index']) for r in kept['validation'][:6]]
    from training.seller_market_observations import decision_stock
    # Three real development contexts: no teacher target and no changed input.
    for source_dir,label,aid,tick in [
        ('results/basic_checks/20261121-183332_seller_step12_k88_h5cq/step12','horizon_loop','example_student_seller_p1',26),
        ('results/basic_checks/20261121-183332_seller_step12_k88_h5cq/step12','stale_inventory','example_student_seller_p2',20),
        ('results/basic_checks/20261121-151510_seller_models_87fz7bkq/step46','unfinished_tool','example_student_seller_p3',16)]:
        w=json.loads((ROOT/source_dir/'wrapper_result.json').read_text());run=Path(w['run_dir']);market=next((run/'markets').iterdir());final=json.loads((market/'final.json').read_text())
        rows=[json.loads(l) for l in (market/'events.jsonl').read_text().splitlines()];event=next(r for r in rows if r.get('type')=='turn' and r['agent']==aid and r['tick']==tick)
        from training.official_buyer_data import situation
        messages=event['llm']['messages'];sit=situation(messages[0]['content']);obs=observation(messages[-1]['content'])
        stock,_=decision_stock(event,next(s for s in final['sellers'] if s['id']==aid),final['trades'])
        if obs['you']['stock']!=stock:raise ValueError('Real probe stock mismatch')
        tools=engine_for(dict(situation=sit,observation=obs)).tool_schemas('seller')
        cases.append(dict(name='real_'+label,kind='real_market_development',messages=messages,tools=tools,situation=sit,observation=obs,persona_index=int(aid[-1]),max_new_tokens=min(8192,event['llm']['max_tokens']),source=str(market/'events.jsonl')))
    output.mkdir(parents=True)
    for split,rows in [('train',train),('validation',kept['validation']),('test',kept['test'])]:
        for r in rows:
            if '_encoded' not in r:r['_encoded']=encode_row(tokenizer,r)
        with (output/f'{split}.jsonl').open('w') as raw,(output/f'{split}.tokens.jsonl').open('w') as tokens:
            for r in rows:
                v={k:x for k,x in r.items() if k!='_encoded'};raw.write(dump(v)+'\n');tokens.write(dump(r['_encoded'])+'\n')
    dump_files={'batch_plan.json':batches,'evaluation_cases.json':cases,'quarantine.json':quarantine,'allowed_expiries.json':allowed}
    for name,value in dump_files.items():(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    paths=[Path(__file__),ROOT/'training/official_seller_data.py',ROOT/'training/official_buyer_data.py',ROOT/'training/seller_market_observations.py',
           ROOT/'docs/SELLER_HORIZON_DATA_AUDIT.json',V1/'manifest.json',ROOT/'training/data/official_raw/source.json',
           *V1.glob('*.jsonl'),*ROOT.glob('adlmarket/*.py'),
           *ROOT.glob('training/persona_versions/seller_checkpoint1_v2/persona_*.md')]
    paths += [Path(c['source']) for c in cases if c.get('source')]
    from training.decision_ablation import fingerprint
    manifest=dict(version='seller_consistency_v2',seed=SEED,base_model=BASE,revision=MODEL_REVISION,parent_adapter=str(PARENT),parent_fingerprints=fingerprint(PARENT),
        original_manifest_sha256=sha(V1/'manifest.json'),sources={str(p.resolve()):sha(p) for p in paths},
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},groups=old_manifest['groups'],
        clean_official_pool={s:len(v) for s,v in kept.items()},quarantined=len(quarantine),legal_expiry_rows=len(allowed),
        training=dict(examples=288,native=216,controlled=72,steps=36,batch=8,per_batch_native=6,per_batch_control=2,
                      actions=dict(Counter(r['label_action'] for r in train)),long_native=len(chosen_long),
                      max_tokens=max(len(r['_encoded']['input_ids']) for r in train),history=dict(Counter(r.get('history_turns_shown',0) for r in train))),
        evaluation=dict(calls=len(cases),kinds=dict(Counter(c['kind'] for c in cases)),real_market_cases_are_development_only=True),
        recipe=dict(rank=16,alpha=32,dropout=0.,thinking=True,learning_rate=5e-6,warmup_steps=3,max_steps=36,effective_batch_size=8,checkpoint_steps=[18,36],
                    mlp_chunk_tokens=2048,activation_offload_from_tokens=16000,max_length=32768),
        limitations=['Official replay targets unchanged; risk screen is not complete semantic proof.',
                     'Stock/terminal controls are labelled interventions in official worlds, not official teacher trajectories or optimal pricing.',
                     'Passing CPU checks does not establish model ability, profit improvement, or CUDA memory feasibility.'])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');return audit(output)


def audit(folder=DATA):
    from training.buyer_operations_train import TokenRows
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    if any(sha(p)!=h for p,h in m['sources'].items()):raise ValueError('Consistency source changed; use a new version')
    if any(sha(folder/p)!=h for p,h in m['files'].items()):raise ValueError('Consistency data changed')
    from training.decision_ablation import check_fingerprint
    check_fingerprint(PARENT,m['parent_fingerprints'])
    groups=m['groups']
    if any(set(groups[a])&set(groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Split leakage')
    train=[json.loads(l) for l in (folder/'train.jsonl').read_text().splitlines()];batches=json.loads((folder/'batch_plan.json').read_text())
    if len(train)!=288 or len(batches)!=36 or sorted(i for b in batches for i in b)!=list(range(288)):raise ValueError('Exposure plan changed')
    for split in ('train','validation','test'):
        original={r['id']:r for r in (json.loads(l) for l in (V1/f'{split}.jsonl').read_text().splitlines())}
        rows=[json.loads(l) for l in (folder/f'{split}.jsonl').read_text().splitlines()]
        for r in rows:
            if r['group'] not in groups[split]:raise ValueError('Wrong source market split')
            if r['source_sample_id'] not in original:raise ValueError('Untraceable original sample')
            source=original[r['source_sample_id']]
            if (r['group'],r['trajectory_id'])!=(source['group'],source['trajectory_id']):raise ValueError('Source lineage changed')
            verify_target(r)
            if r['source_kind']=='official_replay':
                if r['answer']!=source['answer']:raise ValueError('Official completion was rewritten')
                if row_findings(r)['risks']:raise ValueError('Unresolved official consistency risk')
            else:
                for field,text in own_texts(r['answer']):
                    if any(f['kind']!='allowed_offer_expiry' for f in horizon_findings(text,r['observation']['round'])):raise ValueError('Control target horizon conflict')
    for batch in batches:
        if Counter(train[i]['source_kind'] for i in batch)!=dict(official_replay=6,engine_control=2):raise ValueError('Batch mix changed')
    TokenRows(folder/'train.tokens.jsonl',32768)
    return m


def write_review(folder,manifest):
    folder=Path(folder);m=manifest
    lines=['# Seller CPU consistency review','',
        f"CPU checks passed. Clean official pool: {m['clean_official_pool']}; quarantined rows: {m['quarantined']}.",
        f"Legal offer-expiry rows retained in the clean pool: {m['legal_expiry_rows']}.",
        f"Training: {m['training']['native']} unchanged official targets + {m['training']['controlled']} explicitly labelled engine controls.",
        f"Actions: {m['training']['actions']}; long native contexts: {m['training']['long_native']}; max tokens: {m['training']['max_tokens']}.",
        f"Evaluation cases: {m['evaluation']['kinds']}.",'',
        'Original source files, adapters, personas and market engine remain unchanged.',
        'Historical note quantities may legitimately be stale; only explicit target inventory assertions are compared with current stock.',
        'A carried R4 note saying next round is not incorrectly rejected on arrival in R5. Explicit R6/R7 action plans remain incompatible.',
        'Legal offer expiry into R6 and public negotiation bluff are preserved.',
        'The 75%/25% mix is a bounded repair proposal, not a known optimal recipe. Pricing quality is not established by positive margins.',
        'No GPU work or model improvement is established by this report. Long-context CUDA preflight is still required.']
    (folder/'cpu_review.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path,default=DATA)
    args=parser.parse_args()
    print('CPU-only Seller source audit/tokenization; no GPU query or inference.',flush=True)
    result=audit(args.output) if args.audit_only or (args.output/'manifest.json').is_file() else prepare(args.output)
    write_review(args.output,result)
    print(f'PASS CPU consistency. Read {args.output}/cpu_review.md',flush=True)
