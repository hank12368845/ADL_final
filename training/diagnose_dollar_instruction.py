#!/usr/bin/env python3
"""Compare step84/step6 with an explicit dollar format instruction, zero training."""
from collections import Counter
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'training/runs/20261120-141718_buyer_dollar_lines_sgkzzv1q'
INSTRUCTION='''Start thinking with this dollar check:
CASH=<cash> | ROW=<board:seller:product OR offer:id:product> total=<price*qty> held=<0/1> val=<current next utility> adj=<expected value> afford=<0/1> eligible=<0/1>
afford=1 iff cash>=total. eligible=1 only if unheld, live, affordable and adj>total. Then finish thinking and call one tool.'''
SOURCES=('training/diagnose_dollar_instruction.py','training/buyer_dollar_lines.py','training/buyer_check_lines.py',
    'training/buyer_self_checks.py','training/buyer_operations_train.py','training/reasoning_checkpoint.py',
    'training/official_buyer_train.py','training/step84_repair_plan.py','training/buyer_hf_check.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_instruction(case,original):
    """Change only a diagnostic persona copy. Never append facts or a think prefix."""
    c=copy.deepcopy(case);system=c['messages'][0]['content']
    before,body=system.split('<<<PERSONA_SECTION>>>',1);played,after=body.split('<<<END_PERSONA_SECTION>>>',1)
    if played.strip()!=original.strip():raise ValueError('Original latest persona differs from cached case')
    variant=original+'\n\n'+INSTRUCTION
    if len(variant)>8000:raise ValueError('Instruction persona exceeds8000 chars')
    c['messages'][0]['content']=before+'<<<PERSONA_SECTION>>>\n'+variant+'\n<<<END_PERSONA_SECTION>>>'+after
    return c,variant


def prompt(tokenizer,case):
    text=tokenizer.apply_chat_template(case['messages'],tools=case['tools'],tokenize=False,
        add_generation_prompt=True,enable_thinking=True)
    if not text.endswith('<think>\n'):raise ValueError('Expected empty native think, no supplied answer prefix')
    return text


def prepare(run):
    from transformers import AutoTokenizer
    from training.buyer_dollar_data import check_data
    from training.decision_ablation import check_fingerprint
    run=Path(run).resolve();cfg=json.loads((run/'train_config.json').read_text())
    for name,digest in cfg['training_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Original training source changed: {name}')
    _,_,_,m=check_data(cfg)
    if sha(ROOT/cfg['data_dir']/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Data manifest changed')
    status=json.loads((run/'finished.json').read_text())
    if status['step']<6 or status['status'] not in ('stopped_step6_gate','complete'):raise ValueError('Need completed step6 pilot')
    originals={str(i):(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text() for i in (1,2,3)}
    allcases=json.loads((ROOT/cfg['data_dir']/'evaluation_cases.json').read_text())
    selected=[c for c in allcases if c['suite']=='self_check_seen']
    if len(selected)!=12 or Counter(c['positive'] for c in selected)!={True:6,False:6}:raise ValueError('Expected same12 seen cases')
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    cases=[];variants={}
    for c in selected:
        new,variant=add_instruction(c,originals[str(c['persona_index'])]);variants[str(c['persona_index'])]=variant
        new['original_prompt_sha256']=hashlib.sha256(prompt(t,c).encode()).hexdigest()
        new['prompt_sha256']=hashlib.sha256(prompt(t,new).encode()).hexdigest()
        cases.append(new)
    adapters={'parent84':cfg['parent_adapter'],'dollar6':str(run/'checkpoints/step_0006')}
    fingerprints={};free={};evidence={}
    for label,step in (('parent84',0),('dollar6',6)):
        path=run/f'native_step_{step:04d}.json';evidence[str(path)]=sha(path)
        lookup={r['case']:r for r in json.loads(path.read_text())['decisions']}
        free[label]={c['name']:lookup[c['name']] for c in cases}
        cp=Path(adapters[label]);fingerprints[label]=json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints']
        check_fingerprint(cp,fingerprints[label])
    jobs=[dict(adapter=label,case=c['name'],rank=c['evaluation_rank']) for label in adapters for c in cases]
    if Counter(j['rank'] for j in jobs)!={0:12,1:12}:raise ValueError('Both cards must receive12 jobs')
    return dict(experiment='dollar_instruction_v1',run=str(run),training_updates=0,new_gpu_calls=24,max_new_tokens=1024,
        instruction=INSTRUCTION,original_personas=originals,diagnostic_personas=variants,
        persona_chars={k:len(v) for k,v in variants.items()},cases=cases,jobs=jobs,
        adapters=adapters,fingerprints=fingerprints,free_cached=free,
        base_model=cfg['base_model'],revision=cfg['revision'],model_recipe={k:cfg[k] for k in ('rank','alpha','dropout','target_modules')},
        sources={name:sha(ROOT/name) for name in SOURCES},evidence_sha256=evidence,
        limits='Diagnostic persona copies with an explicit format/rule instruction. Empty native think, no actual numbers, IDs or verdicts supplied. Same12 seen cases; no market/generalization guarantee. Original personas/adapters remain unchanged.')


def write_report(folder,plan):
    from training.buyer_dollar_lines import summarize,step6_gate
    folder=Path(folder);rows=[json.loads(l) for p in sorted(folder.glob('decisions_rank*.jsonl')) for l in p.read_text().splitlines() if l]
    expected={(j['adapter'],j['case']):j['rank'] for j in plan['jobs']}
    if len(rows)!=24 or len({(r['adapter'],r['case']) for r in rows})!=24:raise ValueError('Incomplete/repeated inference outputs')
    if {(r['adapter'],r['case']) for r in rows}!=set(expected) or any(r['rank']!=expected[r['adapter'],r['case']] for r in rows):raise ValueError('Case/GPU route changed')
    table={};gates={};personas={}
    for label in plan['adapters']:
        table[label]={};personas[label]={}
        for condition in ('original_cached','format_instruction'):
            chosen=list(plan['free_cached'][label].values()) if condition=='original_cached' else [r for r in rows if r['adapter']==label]
            table[label][condition]=summarize(chosen)
            personas[label][condition]={str(i):summarize([r for r in chosen if r['persona_index']==i]) for i in (1,2,3)}
        gates[label]=step6_gate(table[label]['format_instruction'])
    result=dict(summary=table,gates=gates,personas=personas,training_updates=0,limits=plan['limits'])
    (folder/'instruction_review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Dollar format instruction: same-case inference comparison','',plan['limits'],'',
        'The model generates its entire scratchpad and native tool response. No CASH prefill, gold arithmetic, or forced think closure.',
        'The changed persona condition is explicit; it is not a claim of original-latest improvement. Fields are compared in dollars, independently of engine tool legality.','',
        '| Adapter | Condition | Starts CASH /12 | All fields /12 | Positive /6 | Negative safe /6 | Cent traps safe /3 | Owned safe /3 | Cash errors | Owned errors | Truncated | State consistent /12 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,conditions in table.items():
        for condition,s in conditions.items():
            lines.append(f'| {label} | {condition} | {s["self_started"]} | {s["all_fields_exact"]} | {s["captured"]} | {s["negative_safe"]} | {s["cash_negative_safe"]} | {s["owned_negative_safe"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["truncated"]} | {s["state_consistent"]} |')
    lines+=['','| Adapter | Cash | Total | Held | Current value | Adjusted | Afford | Eligible |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in table:
        f=table[label]['format_instruction']['target_fields']
        lines.append('| '+label+' | '+' | '.join(str(f[k])+'/12' for k in ('cash','total','held','next_value','adjusted_value','affordable','eligible'))+' |')
    lines+=['','Unchanged predeclared seen12 gates:','```json',json.dumps(gates,indent=2),'```',
        'If parent84 passes checking and actions, prefer testing that persona change more broadly before more SFT. If only dollar6 passes, validate the combination more broadly.',
        'If format improves but afford/held/verdict remains wrong, formatting did not fix the decision problem. If format is absent, this intervention does not isolate conditional reasoning.',
        'Only12 seen cases; even a perfect result is not a market score or proof all roles/generalization are solved. No automatic persona edit, promotion, retraining or extra GPU work.']
    (folder/'instruction_review.md').write_text('\n'.join(lines)+'\n')
    return result


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from training.buyer_operations_train import adapter_model
    from training.reasoning_checkpoint import load_weights
    from training.official_buyer_train import generation_stop_ids
    from training.decision_ablation import check_fingerprint
    from training.buyer_dollar_lines import assess
    folder=Path(folder);p=json.loads((folder/'plan.json').read_text())
    if p['instruction']!=INSTRUCTION or p['training_updates']!=0:raise ValueError('Diagnostic instruction changed')
    for name,digest in {**{str(ROOT/k):v for k,v in p['sources'].items()},**p['evidence_sha256']}.items():
        if sha(name)!=digest:raise ValueError(f'Diagnostic source/evidence changed: {name}')
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(p['devices'][rank]['fraction'],rank)
    print(f'[rank{rank}] Loading fixed base/adapters, zero updates; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(p['base_model'],revision=p['revision'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base')
    model=adapter_model(base,p['model_recipe']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    t=AutoTokenizer.from_pretrained(p['base_model'],revision=p['revision'],local_files_only=True);eos=generation_stop_ids(model,t)
    jobs=[j for j in p['jobs'] if j['rank']==rank];cases={c['name']:c for c in p['cases']};current=None;began=time.monotonic()
    (folder/f'hardware_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=p['devices'][rank]['index'],name=torch.cuda.get_device_name(rank))))
    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream,torch.inference_mode():
        for i,job in enumerate(jobs,1):
            if current!=job['adapter']:
                current=job['adapter'];check_fingerprint(p['adapters'][current],p['fingerprints'][current]);load_weights(model,p['adapters'][current])
            c=cases[job['case']];text=prompt(t,c)
            if hashlib.sha256(text.encode()).hexdigest()!=c['prompt_sha256']:raise ValueError('Diagnostic prompt changed')
            ids=t(text,add_special_tokens=False,return_tensors='pt').to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.generate(**ids,max_new_tokens=1024,do_sample=False,use_cache=True,eos_token_id=eos,pad_token_id=t.pad_token_id,return_dict_in_generate=False)
            tail=out[0,ids.input_ids.shape[1]:];raw=t.decode(tail,skip_special_tokens=False)
            truncated=len(tail)>=1024 and int(tail[-1]) not in eos
            r=assess(c,raw,t,truncated,rank);r.update(adapter=current,condition='format_instruction',new_generated_tokens=len(tail))
            stream.write(json.dumps(r)+'\n');stream.flush();elapsed=time.monotonic()-began;speed=i/elapsed
            print(f'[rank{rank}] {i}/{len(jobs)} | {current} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {(len(jobs)-i)/speed:.0f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--gpus',default='0,1')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    if a.worker:return worker(a.worker)
    print('CPU: verify adapters/same12 cases; append generic format instruction to diagnostic persona copies...',flush=True)
    plan=prepare(a.run)
    from training.local_resources import training_gpu_plan,output_parent,runtime_cache
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=2.)
    if note:print(note,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_dollar_instruction_',dir=parent))
    (folder/'personas').mkdir()
    for condition,personas in (('original',plan['original_personas']),('format_instruction',plan['diagnostic_personas'])):
        (folder/'personas'/condition).mkdir()
        for i,text in personas.items():(folder/'personas'/condition/f'persona_{i}.md').write_text(text)
    if a.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');print(f'PASS: CPU only; persona chars {plan["persona_chars"]};24 planned calls, zero training. Prepared {folder}');return folder
    cards,skipped,selection=training_gpu_plan(a.gpus,2);plan.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');cache,cache_note=runtime_cache(ROOT)
    if cache_note:print(cache_note,flush=True)
    print(f'GPUs {[c["index"] for c in cards]};24 new calls, zero updates; output/log {folder}; loading ETA estimating.',flush=True)
    env={**os.environ,**cache,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker',str(folder)]
    began=time.monotonic();last=-1
    with (folder/'runner.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=process.stdout.read1(4096):
                text=chunk.decode(errors='replace');sys.stdout.write(text);sys.stdout.flush();log.write(text);log.flush()
                done=sum(len(p.read_text().splitlines()) for p in folder.glob('decisions_rank*.jsonl'))
                if done!=last:
                    elapsed=time.monotonic()-began;speed=done/elapsed;eta=f'{(24-done)/speed:.0f}s' if speed else 'estimating'
                    line=f'[Overall] {done}/24 | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n';print(line,end='',flush=True);log.write(line);log.flush();last=done
            code=process.wait()
        except BaseException:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=code,elapsed_seconds=time.monotonic()-began,training_updates=0)))
    if code:raise SystemExit(f'FAIL: worker exit={code}; inspect {folder}/runner.log')
    write_report(folder,plan);print(f'Completed. Read {folder}/instruction_review.md; no automatic next GPU work.',flush=True);return folder


if __name__=='__main__':main()
