#!/usr/bin/env python3
"""144-call checkpoint/history/private-state diagnosis. GPU runs are user-started."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CONTEXTS=('recorded','fresh','clean_fresh')
LABELS=('base','step4','step8')
DEFAULT_RUN=ROOT/'training/runs/20261118-193743_buyer_operations_6yy413cv'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fixture(source, kind):
    """Build a controlled observation with an original-engine backing state.

    Historical messages and nonbinding notes come from held-out records. Cash
    interventions are diagnostic state fixtures, not claimed as full rollouts.
    """
    from adlmarket.market import Listing
    from adlmarket.tools import to_cents, dollars
    from training.buyer_operations_data import fresh_market, choose
    market,_=fresh_market(source['group'])
    aid=source['situation']['your_id']
    market.round=source['observation']['round']
    # fresh_market(seed) reproduces the source fixture's current tick.
    market.listings.clear()
    buyer=market.buyers[aid]
    buyer.cash=to_cents(source['observation']['you']['cash'])
    if kind.startswith('board_'):
        original=source['observation']['board']
        if len(original)!=1 or source['observation']['offers_to_you']:
            raise ValueError('Board contrast needs exactly one listing and no offers')
        row=original[0];pid=row['product_id'];seller=row['seller'];price=to_cents(row['price'])
        market.listings[seller,pid]=Listing(price)
        if kind=='board_cash_negative':buyer.cash=price-1
        attempted=('buy_listed',{'seller_id':seller,'product_id':pid,'qty':1})
    else:
        offers=source['observation']['offers_to_you']
        if len(offers)!=1 or source['observation']['board']:
            raise ValueError('Offer contrast needs exactly one offer and no board')
        row=offers[0];pid=row['product_id'];seller=row['seller'];price=to_cents(row['price'])
        if kind=='offer_price_negative':price=buyer.utility[pid][0]+1
        if buyer.cash<price:raise ValueError('Offer price contrast must isolate value, not cash')
        made=market.apply(seller,'make_offer',{'buyer_id':aid,'product_id':pid,'price':dollars(price),'qty':1}).as_dict()
        if not made['ok']:raise ValueError(made)
        if made['data']['offer_id']!=row['offer_id']:raise ValueError('Offer identity changed')
        attempted=('accept_offer',{'offer_id':row['offer_id']})
    if market.schema.plan(pid).quality_varies:
        raise ValueError('These controls require fixed-quality items; clearing memory must not hide quality evidence')
    obs=market.observe(aid,turns_left=source['observation']['turns_left_this_round'])
    obs['output_tokens_left_this_round']=source['observation']['output_tokens_left_this_round']
    expected,args=choose(source['situation'],obs)
    target={'board_positive':'buy_listed','board_cash_negative':'wait',
            'offer_positive':'accept_offer','offer_price_negative':'wait'}[kind]
    if expected!=target:raise ValueError(f'Control has unexpected target: {kind}/{expected}')
    if kind.endswith('positive') and obs!=source['observation']:
        different=[k for k in obs if obs[k]!=source['observation'].get(k)]
        raise ValueError(f'Positive control failed to reproduce held-out observation: {different}')
    test=copy.deepcopy(market)
    before=test.buyers[aid].gained_utility
    result=test.apply(aid,expected,{k:v for k,v in args.items() if k!='private_state'}).as_dict()
    if not result['ok']:raise ValueError(f'Expected action rejected: {result}')
    if target!='wait' and test.buyers[aid].gained_utility-before<=to_cents(result['data']['total']):
        raise ValueError('Positive control has no positive realized surplus')
    negative_check=None
    if target=='wait':
        test=copy.deepcopy(market);before=test.buyers[aid].gained_utility
        rejected=test.apply(aid,*attempted).as_dict()
        if kind=='board_cash_negative' and rejected.get('error')!='insufficient_cash':raise ValueError(rejected)
        net=None
        if rejected['ok']:
            net=test.buyers[aid].gained_utility-before-to_cents(rejected['data']['total'])
            if net>=0:raise ValueError('Negative control accidentally profitable')
        negative_check={'result':rejected,'realized_surplus_cents':net}
    messages=copy.deepcopy(source['messages'])
    body=json.loads(messages[-1]['content'])
    body.update(obs)
    messages[-1]['content']=json.dumps(body,ensure_ascii=False)
    if len(messages)<=2:raise ValueError('Recorded controls must actually contain history')
    return {'name':f'g{source["group"]}_{kind}','task':kind,
            'origin':{'split':'validation','group':source['group'],'source_id':source['id'],
                      'intervention':{'board_cash_negative':'current cash set one cent below listed price',
                                      'offer_price_negative':'binding offer and its notice quote value plus one cent'}.get(kind,'none')},
            'messages':messages,'observation':obs,'situation':copy.deepcopy(source['situation']),
            'tools':copy.deepcopy(source['tools']),'contexts':list(CONTEXTS),'state_mode':'operations',
            'expected_tool':expected,'reference_arguments':args,'engine_result':result,
            'negative_trade_check':negative_check,'private_state_present':bool(body['your_private_state'])}


def build_cases(data_dir):
    data_dir=Path(data_dir)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    group_indices=manifest['groups']['validation'][:2]
    if len(group_indices)!=2:raise ValueError('Need two held-out validation market groups')
    groups=[manifest['seed']+1009*g for g in group_indices]
    forbidden={manifest['seed']+1009*g for split in ('train','test') for g in manifest['groups'][split]}
    if set(groups)&forbidden:raise ValueError('Validation controls overlap train/test groups')
    raw=data_dir/'validation.jsonl'
    expected_hash=manifest.get('splits',{}).get('validation',{}).get('raw_sha256')
    if expected_hash and digest(raw)!=expected_hash:raise ValueError('Validation data checksum mismatch')
    selected={}
    with raw.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['group'] in groups and row['id'].endswith('_p1') and row['task'] in ('stale_memory','binding_offer'):
                selected[row['group'],row['task']]=row
    cases=[]
    for group in groups:
        for task,kind in [('stale_memory','board_positive'),('stale_memory','board_cash_negative'),
                          ('binding_offer','offer_positive'),('binding_offer','offer_price_negative')]:
            cases.append(fixture(selected[group,task],kind))
    return cases,{'source_split':'validation','groups':groups,'validation_sha256':digest(raw),
                  'cases':8,'models':list(LABELS),'contexts':list(CONTEXTS),'seeds':[0,1],
                  'persona':'operations_p1','calls':144,'training_updates':0,
                  'note':'Development counterfactual controls; reserve untouched test groups for later evaluation.'}


def prepare_inputs(data_dir, run_dir, output_dir):
    from adlmarket.config import check_adapter
    cases,manifest=build_cases(data_dir)
    run_dir=Path(run_dir).resolve();output_dir=Path(output_dir).resolve()
    adapters={'base':None,'step4':str(run_dir/'checkpoints/step_0004'),
              'step8':str(run_dir/'checkpoints/step_0008')}
    fingerprints={}
    for label,path in adapters.items():
        if path is None:continue
        check_adapter(label,Path(path),'Qwen/Qwen3.5-4B',{'model.layers.':'model.language_model.layers.'})
        fingerprints[label]={'config_sha256':digest(Path(path)/'adapter_config.json'),
                             'weights_sha256':digest(Path(path)/'adapter_model.safetensors')}
    manifest['adapters']=fingerprints
    manifest['generator_sha256']=digest(Path(__file__))
    output_dir.mkdir(parents=True,exist_ok=True)
    for filename,value in [('cases.json',cases),('adapters.json',adapters)]:
        text=json.dumps(value,ensure_ascii=False,indent=1)+'\n';path=output_dir/filename
        if path.exists() and path.read_text()!=text:
            raise ValueError(f'Derived inputs changed; preserve this experiment and use a new --inputs-dir: {path}')
        path.write_text(text)
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return output_dir


def write_report(folder):
    folder=Path(folder)
    plan=json.loads((folder/'plan.json').read_text())
    cases={c['name']:c for c in json.loads((folder/'cases.json').read_text())}
    rows=[json.loads(line) for line in (folder/'scored_decisions.jsonl').open() if line.strip()]
    expected={(label,name,context,seed) for label in LABELS for name,c in cases.items()
              for context in c['contexts'] for seed in plan['seeds']}
    indexed={}
    for row in rows:
        key=(row['adapter'],row['case'],row['context'],row['seed'])
        if key in indexed:raise ValueError(f'Duplicate matrix outcome: {key}')
        if row['outcome']!='accepted':raise ValueError('Service failure prevents ablation interpretation')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('Missing or unexpected ablation cells')
    for name,c in cases.items():
        for context in c['contexts']:
            for seed in plan['seeds']:
                matched=[indexed[label,name,context,seed] for label in LABELS]
                if len({r['input_sha256'] for r in matched})!=1 or len({r['gpu'] for r in matched})!=1:
                    raise ValueError('Checkpoint comparison changed input or GPU')
        for label in LABELS:
            for seed in plan['seeds']:
                if len({indexed[label,name,ctx,seed]['gpu'] for ctx in CONTEXTS})!=1:
                    raise ValueError('History comparison changed GPU')
    def correct(row):
        return row['action_ok'] and row['policy_ok'] and row['tool']==cases[row['case']]['expected_tool']
    table={}
    for label in LABELS:
        table[label]={}
        for context in CONTEXTS:
            sample=[r for r in rows if r['adapter']==label and r['context']==context]
            positive=[r for r in sample if cases[r['case']]['expected_tool']!='wait']
            negative=[r for r in sample if cases[r['case']]['expected_tool']=='wait']
            attempts=[r for r in sample if r['tool'] in ('buy_listed','accept_offer')]
            valid_trades=sum(r['operation_capture'] for r in attempts)
            table[label][context]={'calls':len(sample),'tools':dict(Counter(r['tool'] for r in sample)),
                'decision_correct':sum(correct(r) for r in sample),
                'positive_opportunities':len(positive),'captured_opportunities':sum(r['operation_capture'] for r in positive),
                'board_captured':sum(r['operation_capture'] for r in positive if cases[r['case']]['task']=='board_positive'),
                'offer_captured':sum(r['operation_capture'] for r in positive if cases[r['case']]['task']=='offer_positive'),
                'trade_attempts':len(attempts),'valid_positive_trades':valid_trades,
                'trade_precision':valid_trades/len(attempts) if attempts else None,
                'negative_cases':len(negative),'correct_waits':sum(r['tool']=='wait' for r in negative),
                'false_trade_attempts':sum(r['tool'] in ('buy_listed','accept_offer') for r in negative),
                'state_failures':sum(not r['state_ok'] for r in sample)}
    effects={}
    for label in LABELS:
        effects[label]={}
        for before,after in [('recorded','fresh'),('fresh','clean_fresh')]:
            changes=Counter();identical_disagreements=0
            for name,c in cases.items():
                if after=='clean_fresh' and not c['private_state_present']:continue
                for seed in plan['seeds']:
                    a,b=indexed[label,name,before,seed],indexed[label,name,after,seed]
                    changes['fixed' if correct(b) and not correct(a) else 'regressed' if correct(a) and not correct(b) else 'tie']+=1
                    if a['input_sha256']==b['input_sha256'] and (a['tool'],a['arguments'])!=(b['tool'],b['arguments']):identical_disagreements+=1
            effects[label][f'{before}_to_{after}']={**dict(changes),'matched_pairs':sum(changes.values()),
                                                    'identical_input_output_disagreements':identical_disagreements}
    checkpoint_effects={}
    for before,after in [('base','step4'),('step4','step8')]:
        checkpoint_effects[f'{before}_to_{after}']={}
        for context in CONTEXTS:
            changes=Counter()
            for name in cases:
                for seed in plan['seeds']:
                    a,b=indexed[before,name,context,seed],indexed[after,name,context,seed]
                    changes['fixed' if correct(b) and not correct(a) else 'regressed' if correct(a) and not correct(b) else 'tie']+=1
            checkpoint_effects[f'{before}_to_{after}'][context]=dict(changes)
    identity_controls={}
    for label in LABELS:
        same=tool_disagreements=argument_disagreements=0
        for name,c in cases.items():
            if c['private_state_present']:continue
            for seed in plan['seeds']:
                a,b=indexed[label,name,'fresh',seed],indexed[label,name,'clean_fresh',seed]
                if a['input_sha256']!=b['input_sha256']:raise ValueError('Blank-note identity control changed input')
                same+=1;tool_disagreements+=a['tool']!=b['tool'];argument_disagreements+=a['arguments']!=b['arguments']
        identity_controls[label]={'same_input_pairs':same,'tool_disagreements':tool_disagreements,
                                  'argument_disagreements':argument_disagreements}
    output={'valid_evaluation':True,'calls':len(rows),'gpu_calls':dict(Counter(r['gpu'] for r in rows)),
            'table':table,'paired_context_effects':effects,'paired_checkpoint_effects':checkpoint_effects,
            'blank_note_identity_controls':identity_controls,
            'limits':'Two validation groups, one persona, two seeds. Diagnostic-only history/state removal is not a grading-time modification. '
                     'Directional evidence does not prove a sole cause. Global waiting still requires backend parity checks before attributing it solely to weights.'}
    (folder/'history_ablation.json').write_text(json.dumps(output,indent=2))
    lines=['# Checkpoint / history / private-state ablation','',
           '| Model | Context | Correct decisions | Captured positive trades | Trade precision | Correct negative waits | Memory errors |',
           '|---|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        for context in CONTEXTS:
            r=table[label][context];precision='N/A' if r['trade_precision'] is None else f'{100*r["trade_precision"]:.1f}%'
            lines.append(f'| {label} | {context} | {r["decision_correct"]}/{r["calls"]} | '
                         f'{r["captured_opportunities"]}/{r["positive_opportunities"]} | {precision} | '
                         f'{r["correct_waits"]}/{r["negative_cases"]} | {r["state_failures"]} |')
    lines+=['',output['limits'],'','Paired context effects (memory effects use only nonempty-note cases):','',
            json.dumps(effects,indent=2),'','Paired checkpoint effects:','',json.dumps(checkpoint_effects,indent=2),
            '','Blank-note repeated-input controls:','',json.dumps(identity_controls,indent=2)]
    (folder/'history_ablation.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'training/data/buyer_operations_v1')
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--inputs-dir',type=Path,default=ROOT/'training/data/buyer_history_ablation_v1')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    inputs=prepare_inputs(args.data_dir,args.run_dir,args.inputs_dir)
    command=['buyer_diagnostics.py','--versions','operations','--persona-indices','1',
             '--adapters-file',str(inputs/'adapters.json'),'--cases-file',str(inputs/'cases.json'),
             '--experiment-manifest',str(inputs/'manifest.json'),
             '--analysis','history_ablation','--seeds','0,1','--gpus',args.gpus,'--port',str(args.port)]
    if args.prepare_only:command.append('--prepare-only')
    if args.output_dir:command+=['--output-dir',str(args.output_dir)]
    print(f'Derived controls: {inputs}; 8 cases x 3 checkpoints x 3 contexts x 2 seeds = 144 calls.',flush=True)
    from training.buyer_diagnostics import main as diagnose
    sys.argv=command
    diagnose()


if __name__=='__main__':main()
