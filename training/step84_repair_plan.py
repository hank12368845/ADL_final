"""Fixed step84: test the user's current personas against the historical short control."""
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/basic_checks/20261120-010510_official_buyer_recheck_c9xqfgss'
ARMS=('baseline','latest')
LATEST=ROOT/'agents/example_student/buyer'


def intervention(case,arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    c=copy.deepcopy(case)
    if arm=='baseline':return c
    system=c['messages'][0]['content'];before,rest=system.split('<<<PERSONA_SECTION>>>',1)
    _,after=rest.split('<<<END_PERSONA_SECTION>>>',1)
    text=c['latest_persona']
    if len(text)>8000:raise ValueError('Latest persona exceeds 8000 chars')
    c['messages'][0]['content']=before+'<<<PERSONA_SECTION>>>\n'+text.strip()+'\n<<<END_PERSONA_SECTION>>>'+after
    return c


def fresh_cases(personas):
    from adlmarket.market import Listing
    from adlmarket.prompts import observation_message,system_prompt
    from adlmarket.tools import dollars,to_cents
    from training.buyer_operations_v2_data import history_state,choose
    result=[]
    for index in range(6):
        group=20266120+1009*index;pindex=index//2;channel=('board','offer')[index%2]
        market,sit,history,previous,note=history_state(group,pindex)
        aid=sit['your_id'];seller=list(market.sellers)[0];buyer=market.buyers[aid]
        candidates=sorted((pid for pid in market.products if not buyer.held.get(pid) and not market.schema.plan(pid).quality_varies),
                          key=lambda pid:buyer.utility[pid][0],reverse=True)
        first,second=candidates[:2];price=max(2,min(buyer.utility[first][0],buyer.utility[second][0])//3)
        negative_cash=(price-1,0,price//2)[pindex]
        positive_cash=(price,price+1,price*2)[pindex]
        held=next(pid for pid,n in buyer.held.items() if n);bad=candidates[-1]
        pair=f'new_g{group}_{channel}_p{pindex+1}'
        for positive,cash in ((True,positive_cash),(False,negative_cash)):
            state=copy.deepcopy(market);state.buyers[aid].cash=cash
            state.listings.clear();state.offers.clear()
            # Two real choices plus owned/negative-surplus distractors. All have real stock.
            state.listings[seller,held]=Listing(1)
            state.listings[seller,bad]=Listing(state.buyers[aid].utility[bad][0]+1)
            for pid in (first,second):
                if channel=='board':state.listings[seller,pid]=Listing(price)
                else:
                    made=state.apply(seller,'make_offer',{'buyer_id':aid,'product_id':pid,'price':dollars(price),'qty':1}).as_dict()
                    if not made['ok']:raise ValueError(made)
            obs=state.observe(aid,turns_left=6-(state.tick-1)%6);obs['output_tokens_left_this_round']=8192
            tool,args=choose(sit,obs)
            if positive and tool not in ('buy_listed','accept_offer') or not positive and tool!='wait':raise ValueError('Invalid new cash pair')
            clone=copy.deepcopy(state);before=clone.buyers[aid].gained_utility
            outcome=clone.apply(aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
            if not outcome['ok']:raise ValueError(outcome)
            gain=clone.buyers[aid].gained_utility-before-(to_cents(outcome['data']['total']) if positive else 0)
            if positive and gain<=0:raise ValueError('New positive is not profitable')
            if not positive:
                if channel=='board':attempt=('buy_listed',{'seller_id':seller,'product_id':first,'qty':1})
                else:attempt=('accept_offer',{'offer_id':next(o['offer_id'] for o in obs['offers_to_you'] if o['product_id']==first)})
                rejection=copy.deepcopy(state).apply(aid,*attempt).as_dict()
                if rejection.get('error')!='insufficient_cash':raise ValueError(rejection)
            else:rejection=None
            prompt=system_prompt('buyer',personas[pindex],sit,6,quality=state.spec.schema_.quality,excess=True,round_tokens=8192)
            result.append({'name':pair+('_positive' if positive else '_negative'),'suite':'fresh','pair_id':pair,'positive':positive,
                'channel':channel,'persona_index':pindex+1,'evaluation_rank':index%2,'messages':[{'role':'system','content':prompt},
                    {'role':'user','content':observation_message(obs,note,previous)}],
                'tools':list(state.tools['buyer'].values()),'observation':obs,'situation':sit,
                'engine_check':{'positive_surplus_cents':gain if positive else None,'negative_trade_result':rejection},
                'expected_tool':tool,'origin':{'synthetic':True,'group':group,'counterfactual_current_cash':True}})
    return result


def prepare():
    from training.decision_ablation import check_fingerprint
    cfg=json.loads((REFERENCE/'train_config.json').read_text());run=Path(cfg['original_run'])
    checkpoint=run/'checkpoints/step_0084';record=json.loads((checkpoint/'checkpoint_complete.json').read_text())
    check_fingerprint(checkpoint,record['adapter_fingerprints'])
    if record['step']!=84:raise ValueError('Wrong fixed checkpoint')
    for name,key in (('native_cases.json','native_cases_sha256'),('official_native_cases.json','official_native_cases_sha256')):
        data=json.loads((REFERENCE/name).read_text())
        if hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()!=cfg[key]:raise ValueError('Reference cases changed')
    known=json.loads((REFERENCE/'native_cases.json').read_text());official=json.loads((REFERENCE/'official_native_cases.json').read_text())
    old=json.loads((REFERENCE/'native_step_0084.json').read_text());cache={r['case']:r for r in old['decisions']}
    for c in known:c.update(suite='known',positive=c['expected_tool']!='wait',pair_id=c['origin']['pair_id'],channel='board' if '_board_' in c['name'] else 'offer')
    for i,c in enumerate(official):c.update(suite='official',evaluation_rank=cache[c['name']]['gpu_rank'],persona_index=i%3+1)
    personas=[(run/'personas'/f'persona_{i}.md').read_text() for i in (1,2,3)]
    latest=[(LATEST/f'persona_{i}.md').read_text() for i in (1,2,3)]
    if any(len(text)>8000 for text in latest):raise ValueError('Latest persona exceeds 8000 chars')
    fresh=fresh_cases(personas);cases=known+fresh+official;jobs=[]
    for c in cases:c['latest_persona']=latest[c['persona_index']-1]
    for arm in ARMS:
        for c in cases:
            cached=cache[c['name']] if arm=='baseline' and c['suite']!='fresh' else None
            jobs.append({'arm':arm,'case':c['name'],'rank':c['evaluation_rank'],'cached':cached})
    sources=('training/step84_repair_plan.py','training/check_step84_repair.py','training/official_buyer_train.py',
             'training/official_buyer_data.py','training/buyer_hf_check.py','training/buyer_operations_train.py',
             'training/buyer_operations_v2_data.py','training/buyer_operations_data.py',
             *tuple(str((LATEST/f'persona_{i}.md').relative_to(ROOT)) for i in (1,2,3)))
    plan={'experiment':'step84_latest_persona_check_v1','checkpoint':str(checkpoint),
        'checkpoint_fingerprints':record['adapter_fingerprints'],'base_model':cfg['base_model'],'revision':cfg['revision'],
        'model_recipe':{k:cfg[k] for k in ('rank','alpha','dropout','target_modules')},'reference':str(REFERENCE),
        'reference_native_sha256':hashlib.sha256((REFERENCE/'native_step_0084.json').read_bytes()).hexdigest(),
        'thinking':True,'max_new_tokens':1024,'do_sample':False,'training_updates':0,
        'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'arms':list(ARMS),'cases':cases,'jobs':jobs,'personas':{'baseline':personas,'latest':latest},
        'latest_persona_source':str(LATEST),'arm_meanings':{'baseline':'Historical short experimental personas; official snapshots retain original source persona.',
            'latest':'Exact current user personas from agents/example_student/buyer, no appended rules.'},
        'gpu_calls':sum(j['cached'] is None for j in jobs),'cached_calls':sum(j['cached'] is not None for j in jobs),
        'limits':'Development ablation, not sealed test or market surplus. Only persona section changes; original observations/tools and checkpoint remain fixed.'}
    return plan


def assess(case,arm,text,tokenizer,truncated,rank,cached=False):
    from training.buyer_hf_check import parse_raw
    from training.official_buyer_train import official_snapshot_review
    parsed=parse_raw(tokenizer,text,case['tools'])
    checked=official_snapshot_review(case,parsed['tool'],parsed['arguments'])
    if parsed['parsed_calls']!=1:checked={'valid_snapshot':False,'errors':['not_exactly_one_tool'],'positive_trade':False}
    valid=checked['valid_snapshot'] and not truncated
    return {'arm':arm,'case':case['name'],'suite':case['suite'],'rank':rank,'cached':cached,'tool':parsed['tool'],
        'arguments':parsed['arguments'],'errors':checked['errors'],'valid':valid,'truncated':truncated,
        'positive':case.get('positive'),'pair_id':case.get('pair_id'),'channel':case.get('channel'),
        'captured':case.get('positive') is True and valid and checked['positive_trade'],
        'cash_error':'unaffordable' in checked['errors'],'review':parsed['tool']=='post_review',
        'text':text,'generated_tokens':len(tokenizer.encode(text,add_special_tokens=False))}


def summarize(rows):
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for suite in ('known','fresh','official'):
            selected=[r for r in rows if r['arm']==arm and r['suite']==suite]
            if not selected:continue
            pairs=defaultdict(dict)
            for r in selected:
                if r['pair_id']:pairs[r['pair_id']][r['positive']]=r
            stats[arm][suite]={'calls':len(selected),'valid':sum(r['valid'] for r in selected),
                'positive_total':sum(r['positive'] is True for r in selected),'captured':sum(r['captured'] for r in selected),
                'cash_errors':sum(r['cash_error'] for r in selected),'no_tool':sum(r['tool'] is None for r in selected),
                'truncated':sum(r['truncated'] for r in selected),'missed_to_review':sum(r['positive'] is True and r['review'] for r in selected),
                'both_safe_and_captured':sum(set(p)=={True,False} and p[True]['captured'] and p[False]['valid'] for p in pairs.values()),
                'board_captured':sum(r['captured'] and r['channel']=='board' for r in selected)}
    c=stats.get('latest',{});base=stats.get('baseline',{})
    gates={'known_capture_at_least_6_of_8':c.get('known',{}).get('captured',0)>=6,
        'known_board_at_least_3_of_4':c.get('known',{}).get('board_captured',0)>=3,
        'fresh_capture_at_least_5_of_6':c.get('fresh',{}).get('captured',0)>=5,
        'zero_cash_errors':all(c.get(s,{}).get('calls') and c[s]['cash_errors']==0 for s in ('known','fresh')),
        'all_current_rule_actions_legal':all(c.get(s,{}).get('calls') and c[s]['valid']==c[s]['calls'] for s in ('known','fresh')),
        'zero_no_tool_or_truncation':all(c.get(s,{}).get('calls') and c[s]['no_tool']==0 and c[s]['truncated']==0 for s in ('known','fresh')),
        'official_validity_not_below_baseline':c.get('official',{}).get('valid',0)>=base.get('official',{}).get('valid',99)}
    complete=len(rows)==74
    return {'stats':stats,'gates':gates,'complete':complete,'eligible_for_small_market':complete and all(gates.values()),'automatic_promotion':False}


def write_report(folder,rows):
    folder=Path(folder);result=summarize(rows)
    (folder/'repair_review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Step84: current user personas versus historical short control','',
        'Fixed weights, thinking ON, original observation formatting. latest uses the existing user personas without added rules. baseline is the historical short experimental control, not the latest user persona. Cached baseline decisions are labelled; no training updates.',
        'Negative cases accept ANY legal nonpurchase action, not only wait. Snapshot legality is not market surplus.','',
        '| Arm | Suite | Legal | Positive capture | Board capture | Cash errors | Missed to review | Both safe and captured |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,suites in result['stats'].items():
        for suite,s in suites.items():lines.append(f'| {arm} | {suite} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | {s["board_captured"]} | {s["cash_errors"]} | {s["missed_to_review"]} | {s["both_safe_and_captured"]} |')
    lines+=['','Predeclared small-market gate:']+[f'- {key}: {bool(value)}' for key,value in result['gates'].items()]
    lines+=['',f'Complete: {result["complete"]}; eligible for small-market comparison: {result["eligible_for_small_market"]}.',
        'No automatic deployment, retraining or full-market run. Compare raw decisions and per-persona errors before changing training.']
    (folder/'repair_review.md').write_text('\n'.join(lines)+'\n')
    return result
