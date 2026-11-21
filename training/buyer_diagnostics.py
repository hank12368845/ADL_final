#!/usr/bin/env python3
"""Frozen-observation buyer diagnostics; GPU mode is launched by the user only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NORMAL = ROOT/'results/basic_checks/20261118-153235_gpu_buyer_5jc31470/run_market_20261118-153236/markets/e0_m0'
LOW = ROOT/'results/basic_checks/20261118-153611_gpu_buyer_h5vack5s/run_market_20261118-153611/markets/e0_m0'
OLD = ROOT/'results/basic_checks/20261117-140057_gpu_buyer_oqzp78_o/run_market_20261117-140057/markets/e0_m0'


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def cases():
    """Freeze observed inputs; labels stay outside the model prompt."""
    specs = [('held_high',NORMAL,'example_student_buyer_p1',10),
             ('missing_listing',LOW,'example_student_buyer_p3',24),
             ('known_low',LOW,'example_student_buyer_p2',23),
             ('empty_final',OLD,'example_student_buyer_p1',27)]
    result=[]
    for name,folder,aid,tick in specs:
        row=next(r for r in read_rows(folder/'events.jsonl') if r.get('type')=='turn' and r['agent']==aid and r['tick']==tick)
        agent=next(r for r in read_rows(folder/'agents.jsonl') if r['agent']==aid)
        messages=copy.deepcopy(row['llm']['messages'])
        assert json.loads(messages[-1]['content'])['you']==row['observation']['you']
        result.append({'name':name,'origin':{'folder':str(folder),'agent':aid,'tick':tick},
                       'messages':messages,'observation':row['observation'],
                       'situation':agent['situation'],'tools':agent['tools'],
                       'contexts':['recorded','fresh']})
    base=copy.deepcopy(result[0])
    for name,qty in [('binding_counteroffer',1),('multiunit_offer',2)]:
        case=copy.deepcopy(base); case['name']=name
        case['origin']={'synthetic':True,'based_on':base['origin'],
                        'change':'Final-turn offer-only state with unowned fixed-quality P72'}
        obs=case['observation'];obs.update(round=5,turns_left_this_round=1,new_messages=[],board=[],your_unreviewed_trades=[])
        obs['you']={'cash':case['situation']['your_budget'],'units_bought':{},
                    'utility_of_next_unit':{'P72':78.0},'utility_so_far':0,'units_by_grade':{}}
        obs['offers_to_you']=[{'offer_id':'diagnostic_offer','seller':case['situation']['sellers'][0],
                              'buyer':case['situation']['your_id'],'product_id':'P72',
                              'price':30.0,'qty':qty,'open_until_round':5}]
        obs['output_tokens_left_this_round']=8192
        body={**obs,'your_private_state':'','last_action_result':{'ok':True}}
        case['messages']=[case['messages'][0],{'role':'user','content':json.dumps(body)}]
        case['contexts']=['fresh'];result.append(case)
    return result


def replace_persona(messages, persona, context):
    if context not in ('recorded','fresh','clean_fresh'):
        raise ValueError(f'Unknown diagnostic context: {context}')
    msgs=copy.deepcopy(messages)
    system=msgs[0]['content']
    before,rest=system.split('<<<PERSONA_SECTION>>>',1)
    _,after=rest.split('<<<END_PERSONA_SECTION>>>',1)
    msgs[0]['content']=before+'<<<PERSONA_SECTION>>>\n'+persona+'\n<<<END_PERSONA_SECTION>>>'+after
    if context=='recorded':return msgs
    msgs=[msgs[0],msgs[-1]]
    if context=='clean_fresh':
        body=json.loads(msgs[-1]['content'])
        body['your_private_state']=''
        msgs[-1]['content']=json.dumps(body,ensure_ascii=False)
    return msgs


def interleave_gpu_jobs(jobs):
    """Alternate queued GPU destinations while preserving paired-input assignment."""
    queues=defaultdict(deque)
    for job in jobs:queues[job[3]].append(job)
    ordered=[]
    while any(queues.values()):
        for gpu in sorted(queues):
            if queues[gpu]:ordered.append(queues[gpu].popleft())
    return ordered


def evaluate(case, tool, args):
    """Check action sets, not a single supposed optimum. Quality is judged ex ante."""
    obs=case['observation'];you=obs['you'];errors=[];opportunity=[]
    operation_flags={}
    if case.get('state_mode')=='operations':
        if case.get('teacher_version')=='v2':
            from training.buyer_operations_v2_data import choose
        else:
            from training.buyer_operations_data import choose
        reference,_=choose(case['situation'],obs)
        operation_flags={'positive_reference':reference!='wait'}
    if not tool:
        return {'errors':['no_native_tool_call'],'opportunity':[],**operation_flags}
    schema=next((t['function'] for t in case['tools'] if t['function']['name']==tool),None)
    if schema is None:return {'errors':['unknown_tool'],'opportunity':[],**operation_flags}
    required=schema['parameters']['required'];props=schema['parameters']['properties']
    if any(k not in args for k in required) or set(args)-set(props):errors.append('tool_argument_keys')
    if case.get('state_mode')=='operations':
        for k,value in args.items():
            prop=props.get(k,{})
            expected=prop.get('type')
            valid=(isinstance(value,str) if expected=='string' else
                   type(value)==int if expected=='integer' else
                   type(value) in (int,float) if expected=='number' else True)
            if not valid or isinstance(value,str) and len(value)>prop.get('maxLength',100000) or type(value) in (int,float) and (value<prop.get('minimum',float('-inf')) or value>prop.get('maximum',float('inf'))):
                errors.append('tool_argument_types')
    state=args.get('private_state')
    if not isinstance(state,str):errors.append('missing_state_string')
    else:
        if len(state)>800:errors.append('state_over_800')
        try:
            state=json.loads(state)
            held={p for p,n in you['units_bought'].items() if n>0}
            if not isinstance(state,dict):raise ValueError()
            if case.get('state_mode')=='operations':
                pair=[args.get('seller_id'),args.get('product_id')] if tool=='buy_listed' else None
                if not isinstance(state.get('bad'),list) or any(not isinstance(x,str) for x in state['bad']) or 'pending' not in state or state['pending'] is not None and not isinstance(state['pending'],dict) or 'last_buy' not in state or state['last_buy']!=pair:
                    errors.append('operations_memory')
            else:
                snapshot=state.get('held')
                if not isinstance(snapshot,list) or set(map(str,snapshot))!=held:errors.append('held_snapshot')
                if state.get('cash')!=you['cash']:errors.append('cash_snapshot')
                if any(k not in state for k in ['pick','unowned','live','afford','gain','bad','pending']):errors.append('missing_check_fields')
        except (ValueError,TypeError):errors.append('state_not_json_object')
    if 'tool_argument_types' in errors:
        return {'errors':sorted(set(errors)),'opportunity':opportunity,**operation_flags}
    trade=None
    if tool=='buy_listed':
        if args.get('qty')!=1:errors.append('qty_not_one')
        trade=next((r for r in obs.get('board',[]) if r['seller']==args.get('seller_id') and r['product_id']==args.get('product_id') and r['in_stock']>=1),None)
        if trade is None:errors.append('not_on_current_board')
        else:trade={**trade,'qty':args.get('qty',1)}
    elif tool=='accept_offer':
        trade=next((o for o in obs['offers_to_you'] if o['offer_id']==args.get('offer_id')),None)
        if trade is None:errors.append('not_current_offer')
        else:
            if trade['qty']!=1:errors.append('offer_multiunit_policy')
            if trade['open_until_round']<obs['round']:errors.append('offer_expired')
    if trade:
        pid=trade['product_id'];v=you['utility_of_next_unit'].get(pid,0)
        if isinstance(state,dict) and case.get('state_mode')!='operations':
            if state.get('pick')!=pid or any(not isinstance(state.get(k),bool) for k in ['unowned','live','afford']) or type(state.get('gain')) not in (int,float):
                errors.append('incomplete_trade_checklist')
        if you['units_bought'].get(pid,0)>0:errors.append('persona_forbids_rebuy')
        if v<=0:errors.append('no_marginal_high_utility')
        if trade['price']*trade.get('qty',1)>you['cash']:errors.append('unaffordable')
        if trade['price']*trade.get('qty',1)>=v:errors.append('nonpositive_best_grade_surplus')
        varies=next((r.get('quality_varies',False) for r in case['situation']['market_overview'] if r['product_id']==pid),False)
        bad={r['seller'] for r in obs.get('your_unreviewed_trades',[]) if r.get('grade')=='low'}
        if varies and trade['seller'] in bad and trade['price']>=.4*v:errors.append('ignores_known_low_risk')
        if case.get('state_mode')=='operations' and varies and not trade.get('certified_high') and trade['price']*trade.get('qty',1)>=.4*v:
            errors.append('nonpositive_quality_bound')
        if isinstance(state,dict) and case.get('state_mode')!='operations':
            truth={'unowned':you['units_bought'].get(pid,0)==0,
                   'live':trade.get('open_until_round',obs['round'])>=obs['round'],
                   'afford':trade['price']*trade.get('qty',1)<=you['cash']}
            if any(state.get(k) is not value for k,value in truth.items()):
                errors.append('trade_check_flags_mismatch')
            if not varies or trade.get('certified_high'):
                gain=state.get('gain')
                expected=v-trade['price']*trade.get('qty',1)
                if type(gain) not in (int,float) or abs(gain-expected)>.011:
                    errors.append('fixed_quality_gain_mismatch')
    if case['name'] in ['empty_final','multiunit_offer'] and tool!='wait':errors.append('expected_wait')
    if case['name']=='binding_counteroffer' and tool!='accept_offer':opportunity.append('missed_only_positive_final_offer')
    result={'errors':sorted(set(errors)),'opportunity':opportunity,**operation_flags}
    if operation_flags:
        if operation_flags['positive_reference'] and tool not in ('buy_listed','accept_offer'):
            opportunity.append('missed_positive_operation')
    return result


def summarize(folder):
    records=list(read_rows(folder/'decisions.jsonl'));groups={}
    for row in records:
        key=f'{row["persona"]}/{row["context"]}'
        group=groups.setdefault(key,{'cases':0,'errors':{},'opportunity':{}});group['cases']+=1
        for category in ['errors','opportunity']:
            for issue in row['evaluation'][category]:group[category][issue]=group[category].get(issue,0)+1
    (folder/'summary.json').write_text(json.dumps(groups,indent=2))
    lines=['# Frozen buyer diagnostics','', 'This measures frozen-input decisions, not market profit or course grade.','']
    for key,value in groups.items():lines += [f'## {key}', '',json.dumps(value,ensure_ascii=False),'']
    (folder/'summary.md').write_text('\n'.join(lines))


def worker(folder):
    from adlmarket.setups import market_run
    from adlmarket.serving import VLLMCluster,register_adapters
    from adlmarket.client import ModelClient
    from adlmarket.engine import with_reasoning
    from training.chat_payload import api_messages, validate_payloads
    run=market_run(ROOT/'configs/basic_buyer.yaml',root=ROOT)
    plan=json.loads((folder/'plan.json').read_text()); devices=plan['devices']
    template=run.serving['base'];adapter=run.models[run.buyers[0].model]
    base=run.models['base'];run.models={};run.serving={};names={}
    adapters=plan.get('adapters',{'identity':adapter.lora.path})
    for i,d in enumerate(devices):
        name=f'diagnostic_buyer_{i}';server=f'diagnostic_base_{i}'
        url=f'http://localhost:{plan["port"]+i}/v1'
        run.models[server]=base.model_copy(update={'base_url':url,'replicas':[url]})
        run.serving[server]=template.model_copy(update={'gpus':[d['index']],'port':plan['port']+i,'gpu_memory_utilization':d['fraction']})
        for label,path in adapters.items():
            model_name=f'{name}_{label}'
            if path is None:
                names[i,label]=server
            else:
                spec=adapter.model_copy(deep=True)
                spec.lora.path=path
                run.models[model_name]=spec.model_copy(update={'model':model_name,'base_url':url,'replicas':[url]})
                names[i,label]=model_name
    frozen=json.loads((folder/'cases.json').read_text());jobs=[]
    cash_pair_indices={}
    if plan.get('analysis')=='cash_decision':
        from training.cash_decision_report import case_pairs
        cash_pair_indices={pair_id:i for i,pair_id in enumerate(case_pairs(frozen))}
    preflight=validate_payloads(frozen,plan['personas'])
    (folder/'request_preflight.json').write_text(json.dumps(preflight,indent=2))
    for ci,case in enumerate(frozen):
        for context in case['contexts']:
            for si,seed in enumerate(plan['seeds']):
                # A condition uses the same GPU for every persona, avoiding hardware as a paired difference.
                route=cash_pair_indices[case['origin']['pair_id']] if cash_pair_indices else ci
                gpu=(route+si)%len(devices)
                for persona,text in plan['personas'].items():
                    for label in adapters:jobs.append((case,context,seed,gpu,persona,text,label))
    jobs=interleave_gpu_jobs(jobs)
    if plan.get('calls',len(jobs))!=len(jobs):raise ValueError('Planned/model job count mismatch')
    (folder/'job_routes.json').write_text(json.dumps([
        {'case':case['name'],'context':context,'seed':seed,'gpu':devices[gpu]['index'],
         'persona':persona,'adapter':label} for case,context,seed,gpu,persona,text,label in jobs],indent=2))
    client=ModelClient(api_key='EMPTY',timeout=120,retry_limit=2)
    def call(job):
        case,context,seed,gpu,persona,text,label=job
        spec=with_reasoning(run.models[names[gpu,label]],plan.get('thinking',False))
        messages=api_messages(replace_persona(case['messages'],text,context))
        call_start=time.monotonic()
        response=client.complete(model=spec.model,messages=messages,tools=case['tools'],
             max_tokens=min(8192,case['observation'].get('output_tokens_left_this_round',8192)),
             temperature=spec.temperature,top_p=spec.top_p,seed=seed,base_url=spec.base_url,
             api_key='EMPTY',extra_body=spec.extra_body)
        action=response.tool_call
        tool=action.name if action and not action.arguments_error and response.stop_reason!='length' else None
        args=action.arguments if tool else {}
        return {'case':case['name'],'context':context,'seed':seed,'persona':f'{label}/{persona}' if len(adapters)>1 else persona,'adapter':label,'gpu':devices[gpu]['index'],
                'tool':tool,'arguments':args,'outcome':response.outcome,'error':response.error,
                'text':response.text,'raw_arguments':action.raw_arguments if action else None,
                'arguments_error':action.arguments_error if action else None,
                'stop_reason':response.stop_reason,'evaluation':evaluate(case,tool,args),
                'thinking':plan.get('thinking',False),'reasoning':getattr(response,'reasoning',None),
                'usage':getattr(response,'usage',{}),'elapsed_seconds':time.monotonic()-call_start,
                'input_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()}
    with VLLMCluster(run,folder/'vllm'):
        register_adapters(run)
        (folder/'inference_started.json').write_text(json.dumps({'time':time.time()}))
        print(f'Replay inference started: {len(jobs)} calls',flush=True)
        (folder/'evaluation_status.json').write_text(json.dumps({'status':'running','expected_calls':len(jobs)}))
        with ThreadPoolExecutor(max_workers=4) as pool, (folder/'decisions.jsonl').open('w') as stream:
            futures=[pool.submit(call,j) for j in jobs]
            try:
                for completed,future in enumerate(as_completed(futures),1):
                    row=future.result()
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    if row['outcome']!='accepted':
                        (folder/'evaluation_status.json').write_text(json.dumps({
                            'status':'invalid','completed_calls':completed,'expected_calls':len(jobs),
                            'reason':'model request failed; no performance conclusion',
                            'outcome':row['outcome'],'error':row['error']},indent=2))
                        raise RuntimeError(f'Model request failed; evaluation invalid: {row["error"]}')
            except BaseException:
                for pending in futures:pending.cancel()
                raise
    summarize(folder)
    from training.diagnostic_scores import write_scores
    write_scores(folder)
    if set(adapters) in ({'base','candidate'},{'step8','candidate'}) and all(c.get('state_mode')=='operations' for c in frozen):
        from training.operation_gate import compare
        compare(folder,reference='step8' if 'step8' in adapters else 'base')
    if plan.get('analysis')=='history_ablation':
        from training.buyer_history_ablation import write_report
        write_report(folder)
    if plan.get('analysis')=='cash_decision':
        from training.cash_decision_report import write_report
        write_report(folder)
        if plan.get('experiment_manifest',{}).get('experiment')=='buyer_operations_reasoning_v1':
            from training.buyer_reasoning_gate import write_report as reasoning_report
            reasoning_report(folder)
    (folder/'evaluation_status.json').write_text(json.dumps({'status':'complete','completed_calls':len(jobs),'expected_calls':len(jobs)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--prepare-only',action='store_true',help='Freeze and validate inputs; no GPU query or server')
    parser.add_argument('--thinking',action='store_true',help='Enable native thinking; recorded in every result')
    parser.add_argument('--versions',default='v4',help='Comma-separated v2,v3,v4, or operations for the short SFT personas')
    parser.add_argument('--adapter',type=Path,help='Candidate trained buyer PEFT adapter')
    parser.add_argument('--adapters-file',type=Path,help='JSON map of adapter labels to paths; null means plain base')
    parser.add_argument('--experiment-manifest',type=Path,help='CPU-generated provenance and checkpoint fingerprints')
    parser.add_argument('--persona-indices',default='1,2,3',help='Distinct persona indices from 1,2,3')
    parser.add_argument('--analysis',choices=['standard','history_ablation','cash_decision'],default='standard')
    parser.add_argument('--compare-base',action='store_true',help='Compare candidate to plain base with IDENTICAL personas/inputs')
    parser.add_argument('--cases-file',type=Path,help='Frozen cases exported by buyer_failure_audit.py')
    parser.add_argument('--rescore',type=Path,help='CPU-only scoring of an existing diagnostic run')
    parser.add_argument('--gpus',default='auto');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--seeds',default='0,1');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();os.chdir(ROOT)
    if args.worker:worker(args.worker);return
    from training.local_resources import output_parent,query_gpus,select_gpus,runtime_cache
    if args.rescore:
        from training.diagnostic_scores import write_scores
        write_scores(args.rescore.resolve());return
    frozen=json.loads(args.cases_file.read_text()) if args.cases_file else cases();personas={}
    if not frozen or len({c['name'] for c in frozen})!=len(frozen):
        raise ValueError('Use nonempty cases with distinct names')
    versions=args.versions.split(',')
    if len(set(versions))!=len(versions) or any(v not in ['v2','v3','v4','operations','operations_reasoning'] for v in versions) or any(v.startswith('operations') for v in versions) and len(versions)>1:
        raise ValueError('Use comma-separated v2,v3,v4, OR operations alone')
    if versions in (['operations'],['operations_reasoning']):
        from training.buyer_operations_data import personas as operation_personas
        if versions==['operations_reasoning']:
            if not args.thinking: raise ValueError('Reasoning personas require --thinking')
            from training.buyer_reasoning_data import personas as operation_personas
        for c in frozen:c['state_mode']='operations'
    persona_indices=[int(i) for i in args.persona_indices.split(',')]
    if not persona_indices or len(set(persona_indices))!=len(persona_indices) or not set(persona_indices)<={1,2,3}:
        raise ValueError('Use distinct persona indices from 1,2,3')
    for version in versions:
        for i in persona_indices:
            if version.startswith('operations'):
                personas[f'{version}_p{i}']=operation_personas()[i-1]
            else:
                path=ROOT/f'agents/example_student/buyer/persona_{i}.md' if version=='v4' else ROOT/f'training/persona_versions/buyer_{version}/persona_{i}.md'
                personas[f'{version}_p{i}']=path.read_text().strip()
    adapters={}
    if args.adapters_file:
        if args.adapter or args.compare_base:raise ValueError('Use adapters-file OR adapter/compare-base')
        from adlmarket.config import check_adapter
        adapters=json.loads(args.adapters_file.read_text())
        if not isinstance(adapters,dict) or not adapters:raise ValueError('Empty/invalid adapters map')
        for label,path in adapters.items():
            if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in label):
                raise ValueError('Adapter labels must contain only letters, digits or underscores')
            if path is not None:
                path=Path(path).resolve()
                check_adapter(label,path,'Qwen/Qwen3.5-4B',{'model.layers.':'model.language_model.layers.'})
                adapters[label]=str(path)
    if args.compare_base and not args.adapter:raise ValueError('--compare-base requires --adapter')
    if args.adapter:
        from adlmarket.config import check_adapter
        candidate=args.adapter.resolve()
        check_adapter('diagnostic_candidate',candidate,'Qwen/Qwen3.5-4B',{'model.layers.':'model.language_model.layers.'})
        adapters={'candidate':str(candidate)}
        if args.compare_base:adapters={'base':None,**adapters}
    seeds=[int(x) for x in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Use distinct seeds')
    if args.analysis=='history_ablation':
        if set(adapters)!={'base','step4','step8'} or len(personas)!=1 or len(frozen)!=8 or any(set(c['contexts'])!={'recorded','fresh','clean_fresh'} for c in frozen):
            raise ValueError('History ablation requires base/step4/step8, one persona and eight three-context controls')
    if args.analysis=='cash_decision':
        from training.cash_decision_report import case_pairs
        case_pairs(frozen)
    from training.chat_payload import validate_payloads
    print('CPU preflight: validating chat payloads against the vLLM request schema...',flush=True)
    preflight=validate_payloads(frozen,personas)
    parent,note=output_parent(ROOT,args.output_dir)
    if note:print(note,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_diagnostic_',dir=parent))
    (folder/'cases.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=1))
    total=sum(len(c['contexts']) for c in frozen)*len(personas)*len(seeds)*max(1,len(adapters))
    plan={'seeds':seeds,'personas':personas,'port':args.port,'calls':total,'devices':[],
          'thinking':args.thinking,
          'request_preflight':preflight,'analysis':args.analysis}
    if args.experiment_manifest:
        plan['experiment_manifest']=json.loads(args.experiment_manifest.read_text())
    if adapters:plan['adapters']=adapters
    print(f'Frozen cases={len(frozen)}, personas={len(personas)}, calls={total}; output: {folder}',flush=True)
    if args.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan,indent=2));print('PASS: inputs frozen; no GPU work started.');return folder
    snapshot=query_gpus()
    devices,skipped=select_gpus(snapshot,args.gpus)
    plan.update(devices=devices,skipped=skipped,gpu_snapshot=snapshot,gpu_selection=args.gpus)
    requested=len(args.gpus.split(',')) if args.gpus!='auto' else len(devices)
    if len(devices)!=requested:
        (folder/'plan.json').write_text(json.dumps(plan,indent=2))
        raise SystemExit(f'Requested {requested} GPUs but only {len(devices)} fit; no model servers started. '
                         f'Skipped: {skipped}. Plan: {folder}/plan.json')
    cache_env,note=runtime_cache(ROOT)
    plan.update(devices=devices,skipped=skipped,cache_env=cache_env)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2))
    print('GPU assignments:',[(d['index'],d['fraction']) for d in devices],flush=True)
    print('Startup ETA estimating; cache:',cache_env['VLLM_CACHE_ROOT'],flush=True)
    env={**os.environ,**cache_env,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    started=time.monotonic();play_start=None;last_log={}
    with (folder/'runner.log').open('w') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(folder)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                count=(folder/'decisions.jsonl').read_bytes().count(b'\n') if (folder/'decisions.jsonl').exists() else 0
                if play_start is None and (folder/'inference_started.json').exists():
                    try:play_start=json.loads((folder/'inference_started.json').read_text())['time']
                    except ValueError:pass
                elapsed=time.monotonic()-started
                measured=time.time()-play_start if play_start else 0
                speed=count/measured if count and measured>=5 else 0
                eta=f'{(total-count)/speed:.0f}s' if speed else 'estimating'
                filled=min(24,int(24*count/total));print(f'[{"#"*filled}{"-"*(24-filled)}] {count}/{total} | elapsed {elapsed:.0f}s | {speed:.2f} calls/s | ETA {eta}',flush=True)
                if not count:
                    for log in (folder/'vllm').glob('*.log'):
                        lines=log.read_text(errors='replace').splitlines()
                        if lines and lines[-1]!=last_log.get(log):print(f'[{log.name}] {lines[-1][-220:]}',flush=True);last_log[log]=lines[-1]
                if proc.poll() is not None:break
                time.sleep(5)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if proc.returncode:
        print('\n'.join((folder/'runner.log').read_text(errors='replace').splitlines()[-30:]))
        raise SystemExit(f'FAIL: {folder}/runner.log (exit {proc.returncode})')
    print(f'Completed. Diagnostic folder: {folder}\nSummary: {folder}/summary.md',flush=True)
    if args.analysis=='history_ablation':print(f'Read first: {folder}/history_ablation.md',flush=True)
    if args.analysis=='cash_decision':print(f'Read first: {folder}/cash_decision_pairs.md',flush=True)
    return folder


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
