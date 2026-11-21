#!/usr/bin/env python3
"""Small reason-prefix intervention on existing brakes checkpoints; never trains."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'training/runs/20261120-024106_buyer_latest_brakes_vjsgyqu9'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fact_reason(case):
    """Truthful visible facts only: no teacher action, tool names or future outcomes."""
    from training.official_buyer_data import cents
    obs=case['observation'];you=obs['you'];cash=cents(you['cash'])
    held=sorted(p for p,n in you['units_bought'].items() if n)
    varies={r['product_id']:r.get('quality_varies',False) for r in case['situation']['market_overview']}
    lines=[f'CURRENT cash={cash} cents. CURRENT held IDs={json.dumps(held)}.']
    for row in [*obs['board'],*obs.get('offers_to_you',[])]:
        pid=row['product_id'];qty=row.get('qty',1);price=cents(row['price'])*qty;owned=you['units_bought'].get(pid,0)
        value=cents(you['utility_of_next_unit'].get(pid,0))
        # Selected fixture p3 worlds have no variable-grade history: q=2/3, low multiplier=.4.
        expected=value*4//5 if varies[pid] and case['persona_index']==3 and not row.get('certified_high') else value
        live=row.get('open_until_round',obs['round'])>=obs['round'] and row.get('in_stock',1)>=1
        eligible=not owned and qty==1 and live and price<=cash and expected>price
        source='offer '+row['offer_id'] if 'offer_id' in row else 'board '+row['seller']+'/'+pid
        lines.append(f'{source}: product={pid}; held_count={owned}; CURRENT next_value={value} cents; '
            f'adjusted_value={expected} cents; total={price} cents; cash-total={cash-price} cents; '
            f'cash>=total={str(cash>=price).lower()}; gain={expected-price} cents; eligible={str(eligible).lower()}.')
    result='\n'.join(lines)
    if any(name in result for name in ('buy_listed','accept_offer','wait','<tool_call>')):raise ValueError('Action leaked into fact prefix')
    return result


def prepare(run):
    from training.buyer_latest_brakes_train import check_data
    from training.decision_ablation import check_fingerprint
    run=Path(run).resolve();cfg=json.loads((run/'train_config.json').read_text());_,_,_,manifest=check_data(cfg)
    if json.loads((run/'finished.json').read_text())['status']!='complete':raise ValueError('Need completed brakes run')
    allcases=json.loads((ROOT/cfg['data_dir']/'evaluation_cases.json').read_text())
    selected=[c for c in allcases if c['suite']=='seen_train' or c['suite']=='repair_validation' and c['mode']=='owned' and c['channel']=='board']
    if len(selected)!=12:raise ValueError('Expected six seen cases and six held-out ownership cases')
    free={};adapters={'parent':manifest['parent_adapter'],'repair36':str(run/'checkpoints/step_0036')};fingerprints={}
    for label,step in (('parent',0),('repair36',36)):
        rows=json.loads((run/f'native_step_{step:04d}.json').read_text())['decisions'];lookup={r['case']:r for r in rows}
        free[label]={c['name']:lookup[c['name']] for c in selected}
        cp=Path(adapters[label]);complete=json.loads((cp/'checkpoint_complete.json').read_text())
        check_fingerprint(cp,complete['adapter_fingerprints']);fingerprints[label]=complete['adapter_fingerprints']
    # This selection's grade-varying candidates are exactly the controlled p3 fixtures.
    for c in selected:c['fact_reason']=fact_reason(c)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    for c in selected:
        prompt=tokenizer.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'):raise ValueError('Unexpected reasoning boundary')
        c['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    jobs=[{'adapter':label,'case':c['name'],'rank':c['evaluation_rank']} for label in adapters for c in selected]
    if Counter(j['rank'] for j in jobs)!={0:12,1:12}:raise ValueError('Unbalanced job route')
    return {'experiment':'brakes_reason_tool_diagnostic_v1','run':str(run),'base_model':cfg['base_model'],'revision':cfg['revision'],
        'model_recipe':{k:cfg[k] for k in ('rank','alpha','dropout','target_modules')},'adapters':adapters,'fingerprints':fingerprints,
        'cases':selected,'free_cached':free,'jobs':jobs,'new_gpu_calls':24,'max_new_tokens':1024,'training_updates':0,
        'sources':{p:sha(ROOT/p) for p in ('training/diagnose_buyer_brakes.py','training/buyer_operations_train.py',
             'training/reasoning_checkpoint.py','training/official_buyer_train.py','training/step84_repair_plan.py','training/buyer_hf_check.py')},
        'evidence_sha256':{str(run/f'native_step_{s:04d}.json'):sha(run/f'native_step_{s:04d}.json') for s in (0,36)},
        'limits':'Controlled reasoning-prefix intervention; not grading inference, training, proof of hidden reasoning, or market surplus. A changed prefix may itself change behavior.'}


def write_report(folder,plan):
    folder=Path(folder);rows=[]
    for f in sorted(folder.glob('decisions_rank*.jsonl')):rows.extend(json.loads(l) for l in f.read_text().splitlines() if l)
    if len(rows)!=24 or len({(r['adapter'],r['case']) for r in rows})!=24:raise ValueError('Incomplete diagnostic')
    expected={(j['adapter'],j['case']):j['rank'] for j in plan['jobs']}
    if {(r['adapter'],r['case']) for r in rows}!=set(expected) or any(r['rank']!=expected[r['adapter'],r['case']] for r in rows):raise ValueError('Diagnostic cases or routes changed')
    table={};cases={c['name']:c for c in plan['cases']}
    for label in plan['adapters']:
        table[label]={}
        for condition in ('free_cached','fact_prefix'):
            selected=list(plan['free_cached'][label].values()) if condition=='free_cached' else [r for r in rows if r['adapter']==label]
            table[label][condition]={'calls':12,'positive_captured':sum(r['captured'] for r in selected),
                'negative_safe':sum(not cases[r['case']]['positive'] and r['valid'] for r in selected),
                'cash_errors':sum(r['cash_error'] for r in selected),'owned_errors':sum('already_owned_trade' in r['errors'] for r in selected),
                'no_tool':sum(r['tool'] is None for r in selected),'truncated':sum(r['truncated'] for r in selected)}
    result={'summary':table,'training_updates':0,'limits':plan['limits']}
    (folder/'reason_tool_review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Reason versus tool diagnostic','',plan['limits'],'',
        '| Adapter | Condition | Positive capture /6 | Negative safe /6 | Cash errors | Owned errors | No tool | Truncated |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for label,conditions in table.items():
        for condition,s in conditions.items():lines.append(f'| {label} | {condition} | {s["positive_captured"]} | {s["negative_safe"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["no_tool"]} | {s["truncated"]} |')
    lines+=['','Fact prefixes contain current cash, holdings, values and eligibility, but no teacher tool name or function syntax.',
        'Recovery with supplied facts supports prioritizing fact/constraint generation; persistent failures show tool-selection weakness also remains.',
        'Neither outcome uniquely identifies a training cause. Do not claim model improvement from oracle-supplied facts.']
    (folder/'reason_tool_review.md').write_text('\n'.join(lines)+'\n')


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from training.buyer_operations_train import adapter_model
    from training.reasoning_checkpoint import load_weights
    from training.official_buyer_train import generation_stop_ids
    from training.step84_repair_plan import assess
    from training.decision_ablation import check_fingerprint
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text())
    for name,digest in {**{str(ROOT/k):v for k,v in plan['sources'].items()},**plan['evidence_sha256']}.items():
        if sha(name)!=digest:raise ValueError('Diagnostic source/evidence changed')
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(plan['devices'][rank]['fraction'],rank)
    print(f'worker{rank}: loading base and fixed adapters; ETA estimating.',flush=True)
    base=AutoModelForCausalLM.from_pretrained(plan['base_model'],revision=plan['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa')
    model=adapter_model(base,plan['model_recipe']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    t=AutoTokenizer.from_pretrained(plan['base_model'],revision=plan['revision'],local_files_only=True);eos=generation_stop_ids(model,t)
    cases={c['name']:c for c in plan['cases']};jobs=[j for j in plan['jobs'] if j['rank']==rank];current=None;began=time.monotonic()
    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream,torch.inference_mode():
        for i,job in enumerate(jobs,1):
            if current!=job['adapter']:
                current=job['adapter'];check_fingerprint(plan['adapters'][current],plan['fingerprints'][current]);load_weights(model,plan['adapters'][current])
            c=cases[job['case']];prompt=t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            if hashlib.sha256(prompt.encode()).hexdigest()!=c['prompt_sha256']:raise ValueError('Prompt changed')
            supplied=c['fact_reason']+'\n</think>\n\n';ids=t(prompt+supplied,add_special_tokens=False,return_tensors='pt').to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=plan['max_new_tokens'],do_sample=False,use_cache=True,
                eos_token_id=eos,pad_token_id=t.pad_token_id,return_dict_in_generate=False)
            tail=out[0,ids.input_ids.shape[1]:];raw=t.decode(tail,skip_special_tokens=False)
            truncated=len(tail)>=plan['max_new_tokens'] and int(tail[-1]) not in eos
            r=assess(c,'latest',raw,t,truncated,rank);r.update(adapter=current,supplied_reason=c['fact_reason'],generated_tokens=len(tail))
            stream.write(json.dumps(r)+'\n');stream.flush()
            elapsed=time.monotonic()-began;speed=i/elapsed
            print(f'[worker{rank}] {i}/12 | {current} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {(12-i)/speed:.0f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--gpus',default='0,1')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/basic_checks');a=p.parse_args()
    if a.worker:return worker(a.worker)
    print('CPU: select12 paired cases, verify adapters/latest inputs, construct fact-only reasoning prefixes...',flush=True)
    plan=prepare(a.run)
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=2.)
    if note:print(note,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_brakes_reason_tool_',dir=parent))
    if a.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');print(f'PASS: CPU only; 24 planned GPU calls, zero training. Prepared {folder}');return folder
    cards,skipped,selection=training_gpu_plan(a.gpus,2);plan.update(devices=cards,gpu_selection=selection,skipped_gpus=skipped)
    (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');cache,cache_note=runtime_cache(ROOT)
    if cache_note:print(cache_note,flush=True)
    print(f'Output/log {folder}; 24 new calls, same case GPU, zero training; load ETA estimating.',flush=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4',
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker',str(folder)]
    began=time.monotonic();last=-1
    with (folder/'runner.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=process.stdout.read1(4096):
                text=chunk.decode(errors='replace');sys.stdout.write(text);sys.stdout.flush();log.write(text);log.flush()
                done=sum(len(f.read_text().splitlines()) for f in folder.glob('decisions_rank*.jsonl'))
                if done!=last:
                    elapsed=time.monotonic()-began;speed=done/elapsed;eta=f'{(24-done)/speed:.0f}s' if speed else 'estimating'
                    line=f'[Overall] {done}/24 | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n';print(line,end='',flush=True);log.write(line);log.flush();last=done
            code=process.wait()
        except BaseException:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps({'exit':code,'elapsed_seconds':time.monotonic()-began,'training_updates':0}))
    if code:raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    write_report(folder,plan);print(f'Completed. Read {folder}/reason_tool_review.md',flush=True)
    return folder


if __name__=='__main__':main()
