"""Latest-persona safety SFT: official rehearsal plus engine-checked paired brakes."""
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084'
REFERENCE=ROOT/'results/basic_checks/20261120-020252_step84_repair_ewp82xy6'
OFFICIAL=ROOT/'training/data/buyer_official_v1'
VERSION='buyer_latest_brakes_v1'
SEED=20261121


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_personas():return [(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text() for i in (1,2,3)]


def replace_persona(messages,text):
    result=copy.deepcopy(messages);before,rest=result[0]['content'].split('<<<PERSONA_SECTION>>>',1)
    _,after=rest.split('<<<END_PERSONA_SECTION>>>',1)
    result[0]['content']=before+'<<<PERSONA_SECTION>>>\n'+text.strip()+'\n<<<END_PERSONA_SECTION>>>'+after
    return result


def reference_trade(row):
    fn=row['answer']['tool_calls'][0]['function'];a=fn['arguments'];obs=row['observation']
    if fn['name']=='buy_listed':return next(b for b in obs['board'] if (b['seller'],b['product_id'])==(a['seller_id'],a['product_id']))
    return next(o for o in obs['offers_to_you'] if o['offer_id']==a['offer_id'])


def teach(case,tool,args,trade=None):
    """Reason from CURRENT facts, in cents, before choosing the native function."""
    from training.official_buyer_data import cents
    you=case['observation']['you'];cash=cents(you['cash']);held=sorted(p for p,n in you['units_bought'].items() if n)
    if trade is None:
        options=[*case['observation']['board'],*case['observation'].get('offers_to_you',[])]
        trade=next((o for o in options if o['product_id']==case.get('target_product')),options[-1])
    pid=trade['product_id'];qty=trade.get('qty',1);total=cents(trade['price'])*qty
    value=cents(you['utility_of_next_unit'].get(pid,0));owned=you['units_bought'].get(pid,0)
    varies=next(o.get('quality_varies',False) for o in case['situation']['market_overview'] if o['product_id']==pid)
    # Variable-quality training targets are p3 with no prior variable deliveries: q=2/3, E=.8V.
    if varies and case['persona_index']!=3:raise ValueError('Only p3 quality controls are generated')
    bound=value*4//5 if varies and not trade.get('certified_high') else value
    reason=(f'Current {"offer" if "offer_id" in trade else "board"} {trade.get("offer_id",trade["seller"]+"/"+pid)}: '
        f'{pid}, held={owned}, current next-unit value={value} cents. Cash={cash} cents; total={total} cents; cash-total={cash-total} cents. ')
    reason+=f'Cash>=total is {str(cash>=total).lower()}. '
    if not owned:reason+=f'{"One adjustment (q=2/3, E=0.8V): value="+str(bound) if varies else "Fixed-quality value="+str(bound)} cents. '
    if tool=='wait':
        why='already held, so no positive next-unit value' if owned else 'cash is below total' if cash<total else 'current value is not above total'
        reason+=f'{why}; no other eligible purchase. Finish now with wait.'
        pick=None;gain=None
    else:
        if owned or cash<total or bound<=total or qty!=1:raise ValueError('Unsafe positive target')
        reason+=f'Unowned, affordable including equality, gain={bound-total} cents>0. Execute {tool} now.'
        pick=pid;gain=(bound-total)/100
    state={'held':held,'cash':you['cash'],'pick':pick,'unowned':True if pick else None,'live':True if pick else None,
        'afford':True if pick else None,'gain':gain,'bad':[],'pending':None,
        'last_buy':[args['seller_id'],pid] if tool=='buy_listed' else None}
    arguments={'private_state':json.dumps(state,separators=(',',':')),**{k:v for k,v in args.items() if k!='private_state'}}
    return {'role':'assistant','content':'','reasoning_content':reason,'tool_calls':[{'id':'answer','type':'function','function':{'name':tool,'arguments':arguments}}]}


def market_from_official(row,seed):
    from adlmarket.market import Market,Product
    from adlmarket.config import AgentSpec,MarketSpec
    from adlmarket.schema import ProductPlan,Schema
    from training.official_buyer_data import cents,low_multiplier
    sit=row['situation'];rng=random.Random(seed);products={};plans=[]
    quality={r['product_id']:r.get('quality_varies',False) for r in sit['market_overview']}
    for item in sit['products']:
        pid=item['product_id'];value=cents(sit['your_utility_per_unit'][pid][0]);cost=max(1,value//4)
        products[pid]=Product(pid,item['title'],'Beauty',cents(item['list_price']),brand=item.get('brand',''),
            product_type=item.get('type',''),description=item.get('description',''),summary=item.get('summary',item.get('description','')))
        plans.append(ProductPlan(pid,'balanced',1.5,None,(value,),cost,quality_varies=quality[pid],low_cost=max(1,cost//2) if quality[pid] else None))
    sellers=[AgentSpec(id=f'seller_{rng.getrandbits(24):06x}',role='seller',model='base') for _ in range(5)]
    buyers=[AgentSpec(id=f'buyer_{i}',role='buyer',model='base') for i in range(10)]
    schema=Schema(f'brakes_{seed}',tuple(plans),cents(sit['your_budget']),low_multiplier(sit['low_grade_utility']))
    market=Market.generate(schema.id,MarketSpec(),schema,products,sellers,buyers,rng)
    market.tick=13;market.round=3
    return market


def pairs(source,seed,pindex,mode,personas):
    from adlmarket.market import Listing
    from adlmarket.prompts import system_prompt,observation_message
    from adlmarket.tools import dollars,to_cents
    from training.buyer_operations_v2_data import choose
    market=market_from_official(source,seed);aid=f'buyer_{pindex}';seller=list(market.sellers)[0]
    sit=market.situation(aid)
    fixed=sorted((p for p in market.products if not market.schema.plan(p).quality_varies),key=lambda p:market.buyers[aid].utility[p][0],reverse=True)
    if len(fixed)<4:raise ValueError('Need fixed-quality items for controlled safety targets')
    target,background,decoy,other=fixed[:4]
    variable=sorted((p for p in market.products if market.schema.plan(p).quality_varies),key=lambda p:market.buyers[aid].utility[p][0],reverse=True)
    if pindex==2 and variable:target=variable[0]
    value=market.buyers[aid].utility[target][0]
    price=max(2,value//5 if market.schema.plan(target).quality_varies else value//3);variant=seed%4
    out=[]
    # Add one real past purchase so reviews and a misleading initial value remain present.
    market.listings[seller,background]=Listing(1)
    past=market.apply(aid,'buy_listed',{'seller_id':seller,'product_id':background,'qty':1}).as_dict()
    if not past['ok']:raise ValueError(past)
    market.listings.clear()
    for channel in ('board','offer'):
        pair=f'{source["group"]}_{seed}_p{pindex+1}_{mode}_{channel}'
        for positive in (True,False):
            state=copy.deepcopy(market);buyer=state.buyers[aid]
            cash=(price,price+1,price*2,price+100)[variant]
            if mode=='cash' and not positive:cash=(price-1,0,price//2,max(0,price-100))[variant]
            buyer.cash=cash
            if mode=='owned' and not positive:buyer.held[target]=['high']
            quote=buyer.utility[target][0]+1 if mode=='value' and not positive else price
            if mode=='value':buyer.cash=max(cash,quote+100)
            if channel=='board':state.listings[seller,target]=Listing(quote)
            else:
                made=state.apply(seller,'make_offer',{'buyer_id':aid,'product_id':target,'price':dollars(quote),'qty':1}).as_dict()
                if not made['ok']:raise ValueError(made)
            state.listings[seller,background]=Listing(1)
            state.listings[seller,decoy]=Listing(buyer.utility[decoy][0]+1)
            obs=state.observe(aid,turns_left=6);obs['output_tokens_left_this_round']=8192
            tool,args=choose(sit,obs)
            expected=('buy_listed' if channel=='board' else 'accept_offer') if positive else 'wait'
            if tool!=expected:
                raise ValueError('Engine fixture teacher mismatch')
            clone=copy.deepcopy(state);before=clone.buyers[aid].gained_utility
            result=clone.apply(aid,tool,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
            if not result['ok']:raise ValueError(result)
            gain=clone.buyers[aid].gained_utility-before-(to_cents(result['data']['total']) if positive else 0)
            if positive and gain<=0:raise ValueError('Nonpositive positive target')
            unsafe=None
            if not positive:
                if channel=='board':attempt=('buy_listed',{'seller_id':seller,'product_id':target,'qty':1})
                else:attempt=('accept_offer',{'offer_id':next(o['offer_id'] for o in obs['offers_to_you'] if o['product_id']==target)})
                bad_state=copy.deepcopy(state);before_bad=bad_state.buyers[aid].gained_utility
                bad_result=bad_state.apply(aid,*attempt).as_dict()
                bad_gain=bad_state.buyers[aid].gained_utility-before_bad-to_cents(bad_result['data']['total']) if bad_result['ok'] else None
                if mode=='cash' and bad_result.get('error')!='insufficient_cash':raise ValueError('Cash control did not reject')
                if bad_gain is not None and bad_gain>=0:raise ValueError('Negative control is not a loss')
                unsafe={'result':bad_result,'surplus_cents':bad_gain}
            name=pair+('_positive' if positive else '_negative')
            case={'name':name,'id':name,'pair_id':pair,'positive':positive,'channel':channel,'mode':mode,
                'group':source['group'],'trajectory_id':'engine_'+str(seed),'persona_index':pindex+1,'target_product':target,
                'suite':'repair_validation','evaluation_rank':(pindex+(channel=='offer'))%2,'observation':obs,'situation':sit,'tools':list(state.tools['buyer'].values()),
                'messages':[{'role':'system','content':system_prompt('buyer',personas[pindex],sit,6,quality=state.spec.schema_.quality,excess=True,round_tokens=8192)},
                    {'role':'user','content':observation_message(obs,'',past)}],
                'source':'official_world_engine_counterfactual','official_source_id':source['id'],
                'engine_result':result,'positive_surplus_cents':gain if positive else None,'negative_trade_check':unsafe}
            case['answer']=teach(case,tool,args);case['label_action']=tool
            out.append(case)
    return out


def representative_worlds(rows,count):
    pools=defaultdict(list)
    for r in rows:pools[r['group']].append(r)
    order=sorted(pools,key=lambda g:(-len(pools[g][0]['situation']['products']),g));random.Random(SEED).shuffle(order)
    if len(order)<count:raise ValueError('Not enough independent source worlds')
    # Ensure the scarce 24-product groups remain represented, then fill other shapes.
    largest=sorted(order,key=lambda g:-len(pools[g][0]['situation']['products']))[:min(2,count)]
    selected=largest+[g for g in order if g not in largest][:count-len(largest)]
    return [pools[g][0] for g in selected]


def rehearsal(rows,personas,tokenizer):
    from training.official_buyer_train import official_snapshot_review
    from training.official_buyer_data import encode,cents
    pools=defaultdict(list)
    for r in rows:
        if r['label_action'] not in ('buy_listed','accept_offer'):continue
        t=reference_trade(r)
        if next(o['quality_varies'] for o in r['situation']['market_overview'] if o['product_id']==t['product_id']):continue
        if r['label_action']=='accept_offer' and 4*cents(t['price'])>3*cents(r['observation']['you']['utility_of_next_unit'][t['product_id']]):continue
        pools[r['label_action']].append(r)
    result=[];rng=random.Random(SEED)
    for action in ('buy_listed','accept_offer'):
        rng.shuffle(pools[action]);used=Counter();selected=[]
        while len(selected)<24:
            if not pools[action]:raise ValueError('Not enough official replay sources fitting latest persona')
            row=min(pools[action],key=lambda r:used[r['group']]);pools[action].remove(row);used[row['group']]+=1;selected.append(row)
            original=row;trial=[]
            for pindex,text in enumerate(personas):
                r=copy.deepcopy(original);r.update(id=original['id']+f'_latest_p{pindex+1}',source='official_rehearsal',persona_index=pindex+1,
                    suite='train',positive=True,channel='board' if action=='buy_listed' else 'offer')
                r['messages']=replace_persona(original['messages'],text)
                fn=original['answer']['tool_calls'][0]['function']
                if not official_snapshot_review(r,action,fn['arguments'])['positive_trade']:raise ValueError('Unsafe official rehearsal')
                r['answer']=teach(r,action,fn['arguments'],reference_trade(original))
                _,error=encode(tokenizer,r)
                if error:break
                trial.append(r)
            if len(trial)==3:result.extend(trial)
            else:selected.pop()
    return result


def generate(output):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.official_buyer_data import encode
    from training.decision_ablation import check_fingerprint
    from training.step84_repair_plan import intervention
    output=Path(output)
    if output.exists():raise ValueError('Refuse overwrite; use a new output directory')
    personas=latest_personas()
    if any(len(t)>8000 for t in personas):raise ValueError('Persona over limit')
    original_cfg=json.loads((PARENT.parent.parent/'train_config.json').read_text())
    cp=json.loads((PARENT/'checkpoint_complete.json').read_text());check_fingerprint(PARENT,cp['adapter_fingerprints'])
    train_source=[json.loads(l) for l in (OFFICIAL/'train.jsonl').read_text().splitlines()]
    val_source=[json.loads(l) for l in (OFFICIAL/'validation.jsonl').read_text().splitlines()]
    worlds=representative_worlds(train_source,12);val_worlds=representative_worlds(val_source,6)
    modes=('cash','owned','cash','value','cash','owned','cash','cash','owned','cash','cash','cash')
    targeted=[]
    cash_index=0
    for index,source in enumerate(worlds):
        seed=SEED+101*index
        if modes[index]=='cash':
            seed+=(cash_index%4-seed%4)%4;cash_index+=1
        for pindex in range(3):targeted.extend(pairs(source,seed,pindex,modes[index],personas))
    validation=[]
    for pindex in range(3):
        for which,mode in enumerate(('cash','owned')):
            seed=SEED+10000+101*(pindex*2+which)
            if mode=='cash':seed+=(pindex-seed%4)%4
            validation.extend(pairs(val_worlds[pindex*2+which],seed,pindex,mode,personas))
    tokenizer=AutoTokenizer.from_pretrained(original_cfg['base_model'],revision=original_cfg['revision'],local_files_only=True)
    rows=rehearsal(train_source,personas,tokenizer)+targeted
    encoded=[];val_encoded=[]
    for row in tqdm(rows+validation,desc='CPU latest safety targets',unit='examples'):
        token,error=encode(tokenizer,row)
        if error:raise ValueError(f'{row["id"]}: {error}; current observation will not be truncated')
        if token['reasoning_tokens']>160:raise ValueError('Safety rationale should be short')
        token.update({k:row.get(k) for k in ('source','pair_id','positive','persona_index','channel','mode')})
        (val_encoded if row in validation else encoded).append(token)
    # Two official buy + two official offer + one complete board pair + one offer pair.
    lookup={r['id']:i for i,r in enumerate(rows)};pools={a:[r for r in rows if r['source']=='official_rehearsal' and r['label_action']==a] for a in ('buy_listed','accept_offer')}
    rng=random.Random(SEED)
    for pool in pools.values():rng.shuffle(pool)
    blocks=[]
    for index,source in enumerate(worlds):
        order=list(range(1,4));rng.shuffle(order)
        for persona_index in order:
            block=[r for r in targeted if r['group']==source['group'] and r['persona_index']==persona_index]
            blocks.append(block)
    batches=[]
    for block in blocks:
        batch=[lookup[pools[a].pop()['id']] for a in ('buy_listed','accept_offer') for _ in range(2)]
        batch += [lookup[r['id']] for r in sorted(block,key=lambda r:(r['positive'],r['channel']))]
        batches.append(batch)
    if len(batches)!=36 or Counter(i for b in batches for i in b)!=Counter(range(288)):raise ValueError('Incomplete training coverage')
    original_plan=json.loads((REFERENCE/'plan.json').read_text());original_rows=json.loads((REFERENCE/'decisions.json').read_text())
    if original_plan['personas']['latest']!=personas:raise ValueError('Latest personas changed since parent diagnostic; baseline cache is no longer valid')
    cached={r['case']:r for r in original_rows if r['arm']=='latest'}
    cases=[]
    for c in original_plan['cases']:
        c=intervention(c,'latest');c['cached_parent']=cached[c['name']];cases.append(c)
    cases+=validation
    first_ids={rows[i]['id'] for b in batches[:12] for i in b}
    seen=[]
    for persona_index in (1,2,3):
        for positive in (True,False):
            channel=('board','offer','board')[persona_index-1]
            r=next(r for r in targeted if r['id'] in first_ids and r['persona_index']==persona_index and r['positive']==positive and r['mode']=='cash' and r['channel']==channel)
            r=copy.deepcopy(r);r['suite']='seen_train';r['evaluation_rank']=persona_index%2;r['name']='seen_'+r['name'];seen.append(r)
    cases+=seen
    output.mkdir(parents=True)
    def dump(name,data):
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    dump('train.jsonl',rows);dump('train.tokens.jsonl',encoded);dump('validation.jsonl',validation);dump('validation.tokens.jsonl',val_encoded)
    (output/'evaluation_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=1)+'\n')
    (output/'batch_plan.json').write_text(json.dumps(batches)+'\n')
    (output/'personas').mkdir()
    for i,text in enumerate(personas,1):(output/'personas'/f'persona_{i}.md').write_text(text)
    sources=('training/buyer_latest_brakes_data.py','training/official_buyer_data.py','training/step84_repair_plan.py',
        'training/buyer_operations_v2_data.py','training/buyer_operations_data.py',
        *tuple(f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)))
    manifest={'version':VERSION,'thinking':True,'base_model':original_cfg['base_model'],'revision':original_cfg['revision'],
        'parent_adapter':str(PARENT),'parent_fingerprints':cp['adapter_fingerprints'],'seed':SEED,
        'sources':{name:sha(ROOT/name) for name in sources},'official_train_sha256':sha(OFFICIAL/'train.jsonl'),
        'official_validation_sha256':sha(OFFICIAL/'validation.jsonl'),'reference_decisions_sha256':sha(REFERENCE/'decisions.json'),
        'train_examples':288,'train_sources':dict(Counter(r['source'] for r in rows)),
        'train_actions':dict(Counter(r['label_action'] for r in rows)),'validation_examples':24,
        'groups':{'train':sorted({r['group'] for r in rows}),'validation':sorted({r['group'] for r in validation})},
        'mean_tokens':sum(len(r['input_ids']) for r in encoded)/len(encoded),'max_tokens':max(len(r['input_ids']) for r in encoded),
        'max_target_tokens':max(r['target_length'] for r in encoded),'steps':36,'global_batch':8,'per_batch_actions':{'buy_listed':3,'accept_offer':3,'wait':2},
        'evaluation_calls':len(cases),'cached_parent_calls':37,'seen_cases':6,
        'files':{name:sha(output/name) for name in ('train.jsonl','train.tokens.jsonl','validation.jsonl','validation.tokens.jsonl','evaluation_cases.json','batch_plan.json')},
        'note':'Half official fixed-quality successful trades, half original-engine counterfactual controls based on official train worlds. Targets use program-derived short rationale, not claimed original teacher reasoning. No development failures copied into training; sealed official test untouched.'}
    if set(manifest['groups']['train'])&set(manifest['groups']['validation']):raise ValueError('Group leakage')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    import sys,argparse
    sys.path.insert(0,str(ROOT))
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'training/data/buyer_latest_brakes_v1')
    print(json.dumps(generate(p.parse_args().output_dir),indent=2))
