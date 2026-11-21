#!/usr/bin/env python3
"""Fixed step84, compact diagnostic personas, seen12 + development24; no training."""
from collections import Counter
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'training/runs/20261120-141718_buyer_dollar_lines_sgkzzv1q'
PERSONAS=ROOT/'training/persona_versions/buyer_compact_check_v1'
SOURCES=('training/evaluate_compact_buyer.py','training/buyer_dollar_lines.py','training/buyer_check_lines.py',
    'training/buyer_self_checks.py','training/buyer_operations_train.py','training/reasoning_checkpoint.py',
    'training/official_buyer_train.py','training/step84_repair_plan.py','training/buyer_hf_check.py',
    *(f'training/persona_versions/buyer_compact_check_v1/persona_{i}.md' for i in (1,2,3)))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_persona(case,original,replacement):
    c=copy.deepcopy(case);before,body=c['messages'][0]['content'].split('<<<PERSONA_SECTION>>>',1)
    old,after=body.split('<<<END_PERSONA_SECTION>>>',1)
    if old.strip()!=original.strip():raise ValueError('Cached case is not current original persona')
    if not 0<len(replacement)<=8000:raise ValueError('Invalid compact persona length')
    c['messages'][0]['content']=before+'<<<PERSONA_SECTION>>>\n'+replacement.strip()+'\n<<<END_PERSONA_SECTION>>>'+after
    return c


def prompt(t,c):
    text=t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    if not text.endswith('<think>\n'):raise ValueError('Expected empty native think')
    return text


def assess_case(c,text,t,truncated,rank,cached=False,condition='compact_check'):
    from training.step84_repair_plan import assess
    from training.buyer_dollar_lines import state_errors
    r=assess(c,condition,text,t,truncated,rank,cached)
    r.update(persona_index=c['persona_index'],mode=c.get('mode'),condition=condition,
        state_errors=state_errors(c,r['tool'],r['arguments']))
    return r


def prepare(run):
    from transformers import AutoTokenizer
    from training.buyer_dollar_data import check_data
    from training.decision_ablation import check_fingerprint
    run=Path(run).resolve();cfg=json.loads((run/'train_config.json').read_text())
    _,_,_,m=check_data(cfg)
    if sha(ROOT/cfg['data_dir']/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Data manifest changed')
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    originals={str(i):(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text() for i in (1,2,3)}
    compact={str(i):(PERSONAS/f'persona_{i}.md').read_text() for i in (1,2,3)}
    source=ROOT/cfg['data_dir']/'evaluation_cases.json';baseline_path=run/'native_step_0000.json'
    original_cases=[c for c in json.loads(source.read_text()) if c['suite'] in ('self_check_seen','self_check_validation')]
    old={r['case']:r for r in json.loads(baseline_path.read_text())['decisions']}
    cases=[];baseline=[];lengths=[]
    for c in original_cases:
        new=replace_persona(c,originals[str(c['persona_index'])],compact[str(c['persona_index'])])
        a,b=prompt(t,c),prompt(t,new)
        new['original_prompt_sha256']=hashlib.sha256(a.encode()).hexdigest();new['prompt_sha256']=hashlib.sha256(b.encode()).hexdigest()
        lengths.append(dict(case=c['name'],original=len(t.encode(a,add_special_tokens=False)),compact=len(t.encode(b,add_special_tokens=False))))
        saved=old[c['name']];baseline.append(assess_case(c,saved['text'],t,saved['truncated'],c['evaluation_rank'],True,'original_cached'))
        cases.append(new)
    if Counter(c['suite'] for c in cases)!={'self_check_seen':12,'self_check_validation':24}:raise ValueError('Need exact seen12 and development24')
    jobs=[dict(case=c['name'],rank=c['evaluation_rank']) for c in cases]
    if Counter(j['rank'] for j in jobs)!={0:18,1:18}:raise ValueError('Expected18 calls/card')
    adapter=Path(m['parent_adapter']);fp=m['parent_fingerprints'];check_fingerprint(adapter,fp)
    return dict(experiment='buyer_compact_check_v1',run=str(run),adapter=str(adapter),fingerprints=fp,
        base_model=cfg['base_model'],revision=cfg['revision'],model_recipe={k:cfg[k] for k in ('rank','alpha','dropout','target_modules')},
        original_personas=originals,compact_personas=compact,persona_chars={k:len(v) for k,v in compact.items()},
        cases=cases,jobs=jobs,baseline=baseline,prompt_lengths=lengths,new_gpu_calls=36,training_updates=0,max_new_tokens=1024,
        sources={k:sha(ROOT/k) for k in SOURCES},evidence_sha256={str(p):sha(p) for p in (source,baseline_path,ROOT/cfg['data_dir']/'manifest.json')},
        limits='Fixed official step84; diagnostic compact personas only. Same seen12 and reused development24, not a sealed test or market score. No output prefill or answers; current observations/tools unchanged. Persona content/density changes cannot isolate length alone.')


def stats(rows):
    return dict(calls=len(rows),valid=sum(r['valid'] for r in rows),positive_total=sum(r['positive'] is True for r in rows),
        captured=sum(r['captured'] for r in rows),negative_total=sum(r['positive'] is False for r in rows),
        negative_safe=sum(r['positive'] is False and r['valid'] for r in rows),cash_errors=sum(r['cash_error'] for r in rows),
        owned_errors=sum('already_owned_trade' in r['errors'] for r in rows),
        not_current_board=sum('not_current_board' in r['errors'] for r in rows),no_tool=sum(r['tool'] is None for r in rows),
        truncated=sum(r['truncated'] for r in rows),state_consistent=sum(not r['state_errors'] for r in rows))


def summary(rows):
    result={}
    for suite in ('self_check_seen','self_check_validation'):
        selected=[r for r in rows if r['suite']==suite];s=stats(selected)
        s['personas']={str(i):stats([r for r in selected if r['persona_index']==i]) for i in (1,2,3)}
        s['channels']={c:stats([r for r in selected if r['channel']==c]) for c in ('board','offer')};result[suite]=s
    return result


def operation_gates(parent,current):
    checks={}
    for suite in parent:
        a,b=parent[suite],current[suite]
        checks[suite+'_capture_retained']=b['captured']>=a['captured']
        checks[suite+'_each_persona_retained']=all(b['personas'][str(i)]['captured']>=a['personas'][str(i)]['captured'] for i in (1,2,3))
        checks[suite+'_all_actions_valid']=b['valid']==b['calls']
        checks[suite+'_no_cash_owned_board_errors']=all(b[k]==0 for k in ('cash_errors','owned_errors','not_current_board','no_tool','truncated'))
    return checks


def review_evidence(c,r):
    """Evidence packet for semantic review, NOT a regex semantic-accuracy claim."""
    from training.buyer_self_checks import truth,target_check
    from training.buyer_dollar_lines import dollar
    facts=truth(c);row=target_check(c,facts)
    reason=r['text'].split('</think>',1)[0].split('<tool_call>',1)[0].removeprefix('<think>').strip()
    lines=reason.splitlines();patterns={'cash':r'\b(cash|wallet|budget|C)\b',
        'total':r'\b(total|price|cost|offer|P)\b|\$\d',
        'held':r'\b(held|hold|owned|own|bought|duplicate|unowned)\b',
        'current_value':r'\b(utility|value|valuation|next|V)\b'}
    expected=dict(cash=dollar(facts['cash']),total=dollar(row['total']),held=row['held'],current_value=dollar(row['next_value']))
    return dict(condition=r['condition'],case=c['name'],persona=c['persona_index'],suite=c['suite'],candidate=row,
        expected=expected,reason=reason,fields={k:dict(status='pending_review',expected=expected[k],
            candidate_quotes=[line for line in lines if re.search(patterns[k],line,re.IGNORECASE)]) for k in expected},
        note='Quotes may concern a different product or use initial value; inspect meaning and identity. No credit inferred from private_state or merely matching a number.')


def write_report(folder,p):
    folder=Path(folder);rows=[json.loads(l) for path in sorted(folder.glob('decisions_rank*.jsonl')) for l in path.read_text().splitlines() if l]
    expected={j['case']:j['rank'] for j in p['jobs']}
    if len(rows)!=36 or len({r['case'] for r in rows})!=36:raise ValueError('Incomplete/repeated outputs')
    if {r['case'] for r in rows}!=set(expected) or any(r['rank']!=expected[r['case']] for r in rows):raise ValueError('Case/GPU route changed')
    parent,current=summary(p['baseline']),summary(rows);checks=operation_gates(parent,current);old={r['case']:r for r in p['baseline']}
    changes=dict(fixed=[r['case'] for r in rows if r['valid'] and not old[r['case']]['valid']],
        regressed=[r['case'] for r in rows if not r['valid'] and old[r['case']]['valid']])
    result=dict(summary={'original_cached':parent,'compact_check':current},operation_gates=checks,changes=changes,
        semantic_review='pending',automatic_market_or_promotion=False,limits=p['limits'])
    (folder/'compact_review.json').write_text(json.dumps(result,indent=2)+'\n')
    cases={c['name']:c for c in p['cases']}
    packet=[review_evidence(cases[r['case']],r) for r in p['baseline']+rows]
    (folder/'semantic_review.json').write_text(json.dumps(packet,indent=2)+'\n')
    lines=['# Compact buyer personas: fixed step84','',p['limits'],'',
        'Tool/economic constraints are automatic. Four-fact semantic checking has an evidence packet for review; there is no required CASH prefix and no invented zero for an unrecognized sentence.',
        'private_state is checked separately under the retained compact JSON convention; these errors are not engine legality errors.','',
        '| Condition | Suite | Valid | Positive capture | Negative safe | Cash errors | Owned errors | Unlisted pair | No tool | Truncated | State consistent |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,suites in result['summary'].items():
        for suite,s in suites.items():
            lines.append(f'| {label} | {suite} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | {s["negative_safe"]}/{s["negative_total"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["not_current_board"]} | {s["no_tool"]} | {s["truncated"]} | {s["state_consistent"]} |')
    lines+=['','Operation gates:','```json',json.dumps(checks,indent=2),'```','',
        f'Fixed validity: {len(changes["fixed"])}; regressed: {len(changes["regressed"])}. Full lists and per-persona/channel counts are in compact_review.json.',
        'Next: review semantic_review.json against raw reasons for cash/total/held/current-value correctness, errors, missing evidence or ambiguity.',
        'Operation gates alone do not complete the semantic gate or establish market surplus. No training, live persona edits, automatic promotion or further GPU work.']
    (folder/'compact_review.md').write_text('\n'.join(lines)+'\n');return result


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from training.buyer_operations_train import adapter_model
    from training.reasoning_checkpoint import load_weights
    from training.official_buyer_train import generation_stop_ids
    from training.decision_ablation import check_fingerprint
    folder=Path(folder);p=json.loads((folder/'plan.json').read_text())
    if p['training_updates']!=0 or p['new_gpu_calls']!=36:raise ValueError('Bounded inference plan changed')
    for k,v in {**{str(ROOT/k):v for k,v in p['sources'].items()},**p['evidence_sha256']}.items():
        if sha(k)!=v:raise ValueError(f'Source/evidence changed: {k}')
    check_fingerprint(p['adapter'],p['fingerprints'])
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(p['devices'][rank]['fraction'],rank)
    print(f'[rank{rank}] Loading fixed step84, no training; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(p['base_model'],revision=p['revision'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base')
    model=adapter_model(base,p['model_recipe']);load_weights(model,p['adapter']);model.eval()
    for param in model.parameters():param.requires_grad_(False)
    t=AutoTokenizer.from_pretrained(p['base_model'],revision=p['revision'],local_files_only=True);eos=generation_stop_ids(model,t)
    cases={c['name']:c for c in p['cases']};jobs=[j for j in p['jobs'] if j['rank']==rank];began=time.monotonic()
    (folder/f'hardware_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=p['devices'][rank]['index'],name=torch.cuda.get_device_name(rank))))
    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream,torch.inference_mode():
        for i,job in enumerate(jobs,1):
            c=cases[job['case']];text=prompt(t,c)
            if hashlib.sha256(text.encode()).hexdigest()!=c['prompt_sha256']:raise ValueError('Prompt changed')
            ids=t(text,add_special_tokens=False,return_tensors='pt').to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=1024,do_sample=False,use_cache=True,
                eos_token_id=eos,pad_token_id=t.pad_token_id,return_dict_in_generate=False)
            tail=out[0,ids.input_ids.shape[1]:];raw=t.decode(tail,skip_special_tokens=False)
            r=assess_case(c,raw,t,len(tail)>=1024 and int(tail[-1]) not in eos,rank);r['new_generated_tokens']=len(tail)
            stream.write(json.dumps(r)+'\n');stream.flush();elapsed=time.monotonic()-began;speed=i/elapsed
            print(f'[rank{rank}] {i}/{len(jobs)} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {(len(jobs)-i)/speed:.0f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--gpus',default='0,1')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    if a.worker:return worker(a.worker)
    print('CPU: verify fixed step84, compact persona copies, unchanged observations and cached36 controls...',flush=True)
    plan=prepare(a.run)
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=2.)
    if note:print(note,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_compact_buyer_',dir=parent));(folder/'personas').mkdir()
    for label,items in (('original',plan['original_personas']),('compact_check',plan['compact_personas'])):
        (folder/'personas'/label).mkdir()
        for i,text in items.items():(folder/'personas'/label/f'persona_{i}.md').write_text(text)
    if a.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');print(f'PASS: CPU only; persona chars {plan["persona_chars"]};36 planned GPU calls, zero updates. Prepared {folder}');return folder
    cards,skipped,selection=training_gpu_plan(a.gpus,2);plan.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');cache,cache_note=runtime_cache(ROOT)
    if cache_note:print(cache_note,flush=True)
    print(f'GPUs {[c["index"] for c in cards]};36 new calls (18/card), zero training; output/log {folder}; load ETA estimating.',flush=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker',str(folder)]
    began=time.monotonic();last=-1
    with (folder/'runner.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=proc.stdout.read1(4096):
                text=chunk.decode(errors='replace');sys.stdout.write(text);sys.stdout.flush();log.write(text);log.flush()
                done=sum(len(f.read_text().splitlines()) for f in folder.glob('decisions_rank*.jsonl'))
                if done!=last:
                    elapsed=time.monotonic()-began;speed=done/elapsed;eta=f'{(36-done)/speed:.0f}s' if speed else 'estimating'
                    line=f'[Overall] {done}/36 | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n';print(line,end='',flush=True);log.write(line);log.flush();last=done
            code=proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=code,elapsed_seconds=time.monotonic()-began,training_updates=0)))
    if code:raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    write_report(folder,plan);print(f'Completed. Read {folder}/compact_review.md; semantic evidence ready for review. No automatic next GPU work.',flush=True);return folder


if __name__=='__main__':main()
