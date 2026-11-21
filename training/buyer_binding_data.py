"""Official-world, current-engine trajectories for seller/product binding and short closure."""
from collections import Counter, defaultdict
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/buyer_binding_v1'
PARENT=ROOT/'training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084'
REFERENCE=ROOT/'results/basic_checks/20261120-153410_base_step84__fme0dw_/plan.json'
MARKET=ROOT/'results/basic_checks/20261120-160558_step84_market_st8d6o1p/run_market_20261120-160605/markets/e0_m0'
FAMILIES=('binding','pivot_seller','pivot_product','new_listing','offer','wait','rehearsal_board','rehearsal_offer')
SEED=20261123


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def candidates(case):
    from training.step84_market_report import visible_options
    return visible_options(case['observation'],case['situation'],.4)


def teach(case):
    from training.official_buyer_data import cents
    obs=case['observation'];you=obs['you'];options=candidates(case)
    best=options[0] if options else None
    held=sorted(p for p,n in you['units_bought'].items() if n)
    if best:
        pid=best['product'];seller=best['seller'];price=best['total_cents'];gain=best['gain_lower_bound_cents']/100
        tool='accept_offer' if best['source']=='offers_to_you' else 'buy_listed'
        args={'offer_id':best['offer_id']} if tool=='accept_offer' else {'seller_id':seller,'product_id':pid,'qty':1}
        reason=(f'Current {"offer "+best["offer_id"] if tool=="accept_offer" else "board row"}: seller={seller}, product={pid}, '
                f'total=${price/100:.2f}. Cash=${you["cash"]:.2f}; affordable={cents(you["cash"])>=price}. '
                f'Held={you["units_bought"].get(pid,0)}; current next high value=${you["utility_of_next_unit"].get(pid,0):.2f}. '
                f'Even the applicable quality lower bound gives gain=${gain:.2f}>0. ')
        if (case.get('last_result') or {}).get('ok') is False:
            reason+='Recheck this current row rather than copying the failed old pair. '
        reason+=f'Execute {tool} now.'
    else:
        tool='wait';args={};pid=None;gain=None
        reason=f'Cash=${you["cash"]:.2f}; held={",".join(held) or "none"}. No current affordable positive quality-bound trade. Old catalog prices and failed historical pairs are not live listings. Wait now.'
    state=dict(held=held,cash=you['cash'],pick=pid,unowned=(pid not in held) if pid else None,
               live=True if pid else None,afford=True if pid else None,gain=gain,bad=[],pending=None,
               last_buy=[args['seller_id'],pid] if tool=='buy_listed' else None)
    args={'private_state':dump(state),**args}
    return {'role':'assistant','content':'','reasoning_content':reason,'tool_calls':[
        {'id':'answer','type':'function','function':{'name':tool,'arguments':args}}]}


class Scene:
    """Only real engine transactions change holdings/cash; all earlier turns are retained."""
    def __init__(self,source,seed,persona,text,tokenizer,budget=None):
        from adlmarket.market import Market
        from adlmarket.config import AgentSpec,MarketSpec
        from training.buyer_latest_brakes_data import market_from_official
        base=market_from_official(source,seed)
        names=(['ta_seller_p2','ta_seller_extra_2','ta_seller_p3','ta_seller_extra_3','ta_seller_p1'] if random.Random(seed+17).getrandbits(1) else
               [f'vendor_{seed%97}_p2',f'vendor_{seed%97}_extra_2',f'vendor_{seed%97}_p3',f'vendor_{seed%97}_extra_3',f'vendor_{seed%97}_p1'])
        schema=replace(base.schema,id=f'binding_{seed}',budget=base.schema.budget if budget is None else budget)
        spec=MarketSpec(channels={'board':True,'dm':True,'seller_room':False,'buyer_room':True})
        self.market=Market.generate(schema.id,spec,schema,base.products,
            [AgentSpec(id=s,role='seller',model='base') for s in names],
            [AgentSpec(id=f'buyer_{i}',role='buyer',model='base') for i in range(10)],random.Random(seed))
        self.market.tick=1;self.market.round=1
        self.aid=f'buyer_{persona-1}';self.sellers=names;self.history=[];self.note='';self.last=None;self.spent=0
        self.tokenizer=tokenizer;self.persona=persona;self.source=source;self.seed=seed
        self.situation=self.market.situation(self.aid)
        from adlmarket.prompts import system_prompt
        self.system=system_prompt('buyer',text,self.situation,6,quality=spec.schema_.quality,
             excess=True,round_tokens=8192,buyer_values=spec.buyer_values,board=True)

    def observe(self):
        o=self.market.observe(self.aid,turns_left=6-(self.market.tick-1)%6)
        o['output_tokens_left_this_round']=8192-self.spent
        return o

    def post(self,seller,pid,price,grade='high'):
        result=self.market.apply(seller,'set_price',dict(product_id=pid,price=price/100,grade=grade)).as_dict()
        if not result['ok']:raise ValueError(result)

    def advance(self):
        self.market.end_turn();old=self.market.round;self.market.tick+=1;self.market.round=(self.market.tick-1)//6+1
        if self.market.round!=old:self.spent=0

    def past(self,tool,args,before_apply=None):
        from adlmarket.prompts import observation_message
        obs=self.observe();user={'role':'user','content':observation_message(obs,self.note,self.last)}
        if before_apply:before_apply()
        result=self.market.apply(self.aid,tool,args).as_dict()
        answer={'role':'assistant','content':'','tool_calls':[{'id':f'past{self.market.tick}','type':'function',
                 'function':{'name':tool,'arguments':copy.deepcopy(args)}}]}
        prompt=self.tokenizer.apply_chat_template([{'role':'system','content':self.system},user],tools=list(self.market.tools['buyer'].values()),
              tokenize=False,add_generation_prompt=True,enable_thinking=True)
        full=self.tokenizer.apply_chat_template([{'role':'system','content':self.system},user,{**answer,'reasoning_content':'Complete this one action.'}],
              tools=list(self.market.tools['buyer'].values()),tokenize=False,enable_thinking=True)
        assert full.startswith(prompt)
        self.spent+=len(self.tokenizer.encode(full[len(prompt):],add_special_tokens=False))
        self.history.append([user,answer,{'role':'tool','tool_call_id':answer['tool_calls'][0]['id'],'content':dump(result)}])
        self.last=result;self.note=args.get('private_state','');self.advance()
        return result

    def case(self,family):
        from adlmarket.prompts import observation_message
        obs=self.observe()
        c=dict(name=f'binding_{self.seed}_p{self.persona}_{family}',group=self.source['group'],
            trajectory_id=f'engine_{self.seed}',persona_index=self.persona,family=family,source='official_world_native_engine_rollout',
            official_source_id=self.source['id'],observation=obs,situation=self.situation,tools=list(self.market.tools['buyer'].values()),
            messages=[{'role':'system','content':self.system},*[m for turn in self.history[-6:] for m in turn],
                      {'role':'user','content':observation_message(obs,self.note,self.last)}],last_result=self.last)
        c['answer']=teach(c);fn=c['answer']['tool_calls'][0]['function'];c['label_action']=fn['name'];c['positive']=fn['name']!='wait'
        clone=copy.deepcopy(self.market);before=clone.buyers[self.aid].gained_utility
        result=clone.apply(self.aid,fn['name'],{k:v for k,v in fn['arguments'].items() if k!='private_state'}).as_dict()
        if not result['ok']:raise ValueError(result)
        actual_gain=clone.buyers[self.aid].gained_utility-before-round(result.get('data',{}).get('total',0)*100)
        if c['positive'] and actual_gain<=0:raise ValueError('Teacher trade not profitable')
        book=self.market.buyers[self.aid]
        if book.cash!=book.budget-book.spent or book.gained_utility!=sum(t.utility for t in self.market.trades if t.buyer==self.aid):
            raise ValueError('Engine cash/holding history inconsistent')
        c.update(id=c['name'],engine_result=result,teacher_actual_gain_cents=actual_gain,
                 engine_budget_cents=book.budget,engine_spent_cents=book.spent)
        return c


def make_case(source,seed,persona,text,t,family):
    from adlmarket.market import Listing
    from training.official_buyer_data import cents
    values={p:cents(v[0]) for p,v in source['situation']['your_utility_per_unit'].items()}
    quality={r['product_id']:r['quality_varies'] for r in source['situation']['market_overview']}
    fixed=sorted((p for p in values if not quality[p]),key=lambda p:-values[p])
    if len(fixed)<4:raise ValueError('Need four fixed-quality products')
    index=random.Random(seed+37).randrange(min(3,len(fixed)-2))
    fixed=[fixed[index],*[p for p in fixed if p!=fixed[index]]]
    target,other,bg,bg2=fixed[:4];price=max(2,values[target]//5)
    budget=None
    if family.startswith('cash_'):
        remaining=(price,price-1,0)[persona-1];budget=remaining+1
    if family=='budget_choice':budget=price+1
    s=Scene(source,seed,persona,text,t,budget);m=s.market;a,b=s.sellers[:2]
    # Real short/long histories, with two successful purchases and communication, not wait-only prefixes.
    if family in ('cash_board','cash_offer','budget_choice'):
        m.listings[a,bg]=Listing(1);s.past('buy_listed',dict(seller_id=a,product_id=bg,qty=1));m.listings.clear()
    elif seed%2 and family not in ('quality_upgrade',):
        m.listings[a,bg]=Listing(1)
        assert s.past('buy_listed',dict(seller_id=a,product_id=bg,qty=1))['ok']
        assert m.apply(b,'make_offer',dict(buyer_id=s.aid,product_id=bg2,price=.01,qty=1)).ok
        assert s.past('accept_offer',dict(offer_id=list(m.offers)[-1]))['ok']
        s.past('send_message',dict(to=a,message='Please send any updated offer through the binding tools.'))
        s.past('post_to_room',dict(message='Compare currently visible sellers before buying.'))
        if 'post_review' in m.tools['buyer']:s.past('post_review',dict(trade_id=m.trades[0].id,rating=5))
        else:s.past('send_message',dict(to=b,message='Please list available stock.'))
        s.past('wait',{})
        m.listings.clear()
    # Populate interleaved current rows across similar IDs using real seller set_price actions.
    for j,pid in enumerate(fixed[:min(6,len(fixed))]):
        s.post(s.sellers[j%5],pid,values[pid]+100)
    if family in ('pivot_seller','pivot_product','wait'):
        m.listings.clear();m.listings[a,target]=Listing(price)
        competitors=[x for x in m.buyers if x!=s.aid]
        while m.sellers[a].stock[target]>1:
            q=m.sellers[a].stock[target]-1
            r=m.apply(competitors[0],'buy_listed',dict(seller_id=a,product_id=target,qty=q)).as_dict()
            if not r['ok']:raise ValueError(r)
        def consume():
            r=m.apply(competitors[1],'buy_listed',dict(seller_id=a,product_id=target,qty=1)).as_dict()
            if not r['ok']:raise ValueError(r)
        failed=s.past('buy_listed',dict(seller_id=a,product_id=target,qty=1),consume)
        if failed.get('error')!='no_stock':raise ValueError('Expected genuine stock race')
        m.listings.clear()
        if family=='pivot_seller':s.post(b,target,price+1)
        if family=='pivot_product':s.post(b,other,max(1,values[other]//4))
    elif family=='new_listing':
        m.listings.clear()
        failed=s.past('buy_listed',dict(seller_id=a,product_id=target,qty=1))
        if failed.get('error')!='not_listed':raise ValueError(failed)
        s.post(a,target,price)
    elif family in ('offer','rehearsal_offer','cash_offer'):
        result=m.apply(b,'make_offer',dict(buyer_id=s.aid,product_id=target,price=price/100,qty=1)).as_dict()
        if not result['ok']:raise ValueError(result)
    elif family in ('quality_floor','quality_risk','quality_upgrade'):
        variable=next((p for p in sorted(values,key=lambda p:-values[p]) if quality[p]),None)
        if variable is None:raise ValueError('No quality variable product')
        if family=='quality_upgrade':
            m.listings.clear();m.listings[a,variable]=Listing(1,grade='low')
            assert s.past('buy_listed',dict(seller_id=a,product_id=variable,qty=1))['ok']
            s.post(b,variable,max(1,values[variable]//10))
            s.post(a,other,max(1,values[other]//4))
        else:
            s.post(b,variable,max(1,values[variable]//10 if family=='quality_floor' else values[variable]*3//5))
            s.post(a,other,max(1,values[other]*3//4))
    elif family=='budget_choice':
        s.post(a,target,price);s.post(b,other,max(1,price*3//4))
    else:s.post(b,target,price)
    if family in ('binding','pivot_seller','new_listing','offer','rehearsal_board','rehearsal_offer'):
        # Multiple profitable choices: both row matching and price/utility comparison matter.
        s.post(s.sellers[2],target,max(price+1,values[target]*11//20))
        s.post(a,other,max(1,values[other]-max(1,min(values[other],values[target])//10)))
    m.end_turn()  # Publish prices without fabricating a buyer turn.
    # Other overpriced rows are retained as cross-seller distractors.
    if family=='pivot_product':
        s.post(a,target,values[target]+100);m.end_turn()
    if family=='wait':
        for j,pid in enumerate(fixed[:4]):s.post(s.sellers[j%5],pid,values[pid]+100)
        m.end_turn()
    c=s.case(family)
    if family in FAMILIES and c['label_action']!=('wait' if family=='wait' else 'accept_offer' if family in ('offer','rehearsal_offer') else 'buy_listed'):
        raise ValueError(f'Family/action mismatch:{family}/{c["label_action"]}')
    return c


def encode(t,c,max_length=20480):
    start=t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    full=t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
    assert start.endswith('<think>\n') and full.startswith(start)
    p=t.encode(start,add_special_tokens=False);ids=t.encode(full,add_special_tokens=False)
    if ids[:len(p)]!=p or len(ids)>max_length:raise ValueError(f'Unmodified input too long:{c["name"]}/{len(ids)}')
    reason=len(t.encode(c['answer']['reasoning_content'],add_special_tokens=False))
    if reason>300:raise ValueError('Teacher think over300tokens')
    return dict(id=c['id'],group=c['group'],label_action=c['label_action'],family=c['family'],persona_index=c['persona_index'],
        input_ids=ids,prompt_length=len(p),target_length=len(ids)-len(p),reasoning_tokens=reason,thinking=True)


def generate(output=DATA):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from training.buyer_latest_brakes_data import representative_worlds
    output=Path(output)
    if output.exists():return check_data(output)[-1]
    ref=json.loads(REFERENCE.read_text());parent_cfg=json.loads((PARENT.parent.parent/'train_config.json').read_text())
    t=AutoTokenizer.from_pretrained(ref['base_model'],revision=ref['revision'],local_files_only=True)
    files={split:ROOT/f'training/data/buyer_official_v1/{split}.jsonl' for split in ('train','validation')}
    official={s:[json.loads(l) for l in p.read_text().splitlines()] for s,p in files.items()}
    worlds={s:representative_worlds(official[s],16 if s=='train' else 6) for s in files}
    texts={str(i):(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text() for i in (1,2,3)}
    if texts!=ref['original_personas']:raise ValueError('Current original personas changed')
    all_rows={};tokens={}
    for split in files:
        rows=[];encoded=[]
        jobs=[(wi,world,p,family) for wi,world in enumerate(worlds[split]) for p in (1,2,3) for family in FAMILIES]
        for wi,world,p,family in tqdm(jobs,desc=f'CPU native engine {split}',unit='cases'):
            seed=SEED+(0 if split=='train' else 100000)+wi*101+p*10+FAMILIES.index(family)
            c=make_case(world,seed,p,texts[str(p)],t,family);c['suite']='binding_'+split;c['evaluation_rank']=len(rows)%2
            encoded.append(encode(t,c));rows.append(c)
        all_rows[split]=rows;tokens[split]=encoded
    pools={action:[i for i,c in enumerate(all_rows['train']) if c['label_action']==action] for action in ('buy_listed','accept_offer','wait')}
    rng=random.Random(SEED)
    for pool in pools.values():rng.shuffle(pool)
    batches=[]
    for _ in range(48):
        batch=[pools[a].pop() for a,n in (('buy_listed',5),('accept_offer',2),('wait',1)) for _ in range(n)]
        rng.shuffle(batch);batches.append(batch)
    assert not any(pools.values())
    evaluation=[]
    for p in (1,2,3):
        for fi,family in enumerate(FAMILIES):
            group=worlds['validation'][(fi+p)%6]['group']
            c=next(c for c in all_rows['validation'] if c['persona_index']==p and c['family']==family and c['group']==group)
            evaluation.append(copy.deepcopy(c))
    stress=('cash_board','cash_offer','budget_choice','quality_risk','quality_floor','quality_upgrade')
    for p in (1,2,3):
        for j,family in enumerate(stress):
            c=make_case(worlds['validation'][(p+j)%6],SEED+200000+p*100+j,p,texts[str(p)],t,family)
            c['suite']='stress';evaluation.append(c)
    for p in (1,2,3):
        for family in ('binding','offer'):
            c=copy.deepcopy(next(c for c in all_rows['train'] if c['persona_index']==p and c['family']==family))
            c['suite']='seen_train';evaluation.append(c)
    for i,c in enumerate(evaluation):c['evaluation_rank']=i%2
    old=[copy.deepcopy(c) for c in ref['cases']]
    for c in old:c['suite']='legacy36';c.pop('prompt_sha256',None)
    evaluation+=old
    actors={r['agent']:r for r in (json.loads(l) for l in (MARKET/'agents.jsonl').read_text().splitlines())}
    wanted={(f'example_student_buyer_p{p}',tick) for p,tick in ((2,3),(3,23),(2,20),(2,21),(3,2),(1,30))}
    real=[]
    with (MARKET/'events.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r.get('type')!='turn' or (r['agent'],r['tick']) not in wanted:continue
            actor=actors[r['agent']];name=f'real_{r["agent"]}_tick{r["tick"]}'
            c=dict(name=name,id=name,group='reserved_real_market_seed0',trajectory_id='real_e0_m0',
                persona_index=int(r['agent'][-1]),family='real_closure' if r['tick'] in (3,23) else 'real_pair' if r['tick']!=30 else 'real_held',
                suite='market_regression',messages=r['llm']['messages'],observation=r['observation'],situation=actor['situation'],tools=actor['tools'],source='reserved_real_market_snapshot')
            c['last_result']=json.loads(c['messages'][-1]['content']).get('last_action_result')
            c['positive']=bool(candidates(c));c['answer']=teach(c);c['evaluation_rank']=len(real)%2;real.append(c)
    if len(real)!=6:raise ValueError('Missing reserved real-market snapshots')
    evaluation+=real
    assert len(evaluation)==90 and Counter(c['evaluation_rank'] for c in evaluation)=={0:45,1:45}
    output.mkdir(parents=True)
    for split in files:
        (output/f'{split}.jsonl').write_text(''.join(dump(c)+'\n' for c in all_rows[split]))
        (output/f'{split}.tokens.jsonl').write_text(''.join(dump(c)+'\n' for c in tokens[split]))
    (output/'evaluation_cases.json').write_text(json.dumps(evaluation,ensure_ascii=False)+'\n')
    (output/'batch_plan.json').write_text(dump(batches)+'\n')
    sources=[Path(__file__),ROOT/'training/buyer_latest_brakes_data.py',ROOT/'training/step84_market_report.py',ROOT/'training/official_buyer_data.py',REFERENCE,
             *files.values(),MARKET/'agents.jsonl',MARKET/'events.jsonl',*[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)],*(ROOT/'adlmarket').glob('*.py')]
    m=dict(version='buyer_binding_v1',thinking=True,seed=SEED,parent_adapter=str(PARENT),parent_fingerprints=ref['cached_adapter_fingerprints'],
        base_model=ref['base_model'],revision=ref['revision'],model_recipe={k:parent_cfg[k] for k in ('rank','alpha','dropout','target_modules')},
        train_examples=384,validation_examples=len(all_rows['validation']),evaluation_calls=90,max_length=20480,
        training_steps=48,checkpoint_steps=[24,48],batch_size=8,learning_rate=1e-5,warmup_steps=4,
        groups={s:sorted({c['group'] for c in all_rows[s]}) for s in files},
        mean_tokens=sum(len(c['input_ids']) for c in tokens['train'])/384,max_tokens=max(len(c['input_ids']) for c in tokens['train']),
        max_think_tokens=max(c['reasoning_tokens'] for c in tokens['train']),
        labels=dict(Counter(c['label_action'] for c in all_rows['train'])),
        sources={str(p.resolve()):sha(p) for p in sources},files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        note='Official product/value worlds replayed under current engine rules. No raw old-world horizon or artificial current-cash override. '
             '384examples,one complete pass,48steps. Formal vLLM parent/step24/step48 evaluation90each,including6reserved real snapshots. No deployment or automatic market.')
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    check_data(output);return m


def check_data(folder=DATA):
    from training.buyer_operations_train import TokenRows
    from training.decision_ablation import check_fingerprint
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    if m['version']!='buyer_binding_v1':raise ValueError('Wrong binding data')
    for path,h in m['sources'].items():
        if sha(path)!=h:raise ValueError(f'Data source changed:{path}')
    for name,h in m['files'].items():
        if sha(folder/name)!=h:raise ValueError(f'Data changed:{name}')
    check_fingerprint(m['parent_adapter'],m['parent_fingerprints'])
    if set(m['groups']['train'])&set(m['groups']['validation']):raise ValueError('Market source leakage')
    train=TokenRows(folder/'train.tokens.jsonl',20480);val=TokenRows(folder/'validation.tokens.jsonl',20480)
    batches=json.loads((folder/'batch_plan.json').read_text())
    assert len(train)==384 and len(batches)==48 and Counter(i for b in batches for i in b)==Counter(range(384))
    for b in batches:
        assert Counter(train[i]['label_action'] for i in b)=={'buy_listed':5,'accept_offer':2,'wait':1}
    return train,val,batches,m
