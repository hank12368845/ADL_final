#!/usr/bin/env python3
"""User-launched fixed-step84 check of current user personas; no training or added rules."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify(plan):
    from training.decision_ablation import check_fingerprint
    for name,digest in plan['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
    check_fingerprint(plan['checkpoint'],plan['checkpoint_fingerprints'])
    source=Path(plan['reference'])/'native_step_0084.json'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['reference_native_sha256']:raise ValueError('Cached baseline changed')


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from training.buyer_operations_train import adapter_model
    from training.reasoning_checkpoint import load_weights
    from training.official_buyer_train import generation_stop_ids
    from training.step84_repair_plan import assess,intervention
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text());verify(plan)
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(plan['devices'][rank]['fraction'],rank)
    print(f'GPU worker{rank}: loading fixed step84; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(plan['base_model'],revision=plan['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base')
    model=adapter_model(base,plan['model_recipe']);load_weights(model,plan['checkpoint']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(plan['base_model'],revision=plan['revision'],local_files_only=True)
    stop=generation_stop_ids(model,tokenizer);cases={c['name']:c for c in plan['cases']}
    jobs=[j for j in plan['jobs'] if j['cached'] is None and j['rank']==rank];began=time.monotonic()
    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream,torch.inference_mode():
        for number,job in enumerate(jobs,1):
            case=intervention(cases[job['case']],job['arm'])
            prompt=tokenizer.apply_chat_template(case['messages'],tools=case['tools'],tokenize=False,
                add_generation_prompt=True,enable_thinking=True)
            ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.generate(**ids,max_new_tokens=plan['max_new_tokens'],do_sample=False,use_cache=True,
                    eos_token_id=stop,pad_token_id=tokenizer.pad_token_id,return_dict_in_generate=False)
            tail=out[0,ids.input_ids.shape[1]:];raw=tokenizer.decode(tail,skip_special_tokens=False)
            truncated=len(tail)>=plan['max_new_tokens'] and int(tail[-1]) not in stop
            row=assess(case,job['arm'],raw,tokenizer,truncated,rank)
            row.update(generated_tokens=len(tail),prompt_tokens=ids.input_ids.shape[1])
            stream.write(json.dumps(row)+'\n');stream.flush()
            elapsed=time.monotonic()-began;speed=number/elapsed;eta=(len(jobs)-number)/speed
            print(f'[worker{rank}] {number}/{len(jobs)} calls | {job["arm"]}/{case["suite"]} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta:.0f}s',flush=True)
    (folder/f'finished_rank{rank}.json').write_text(json.dumps({'rank':rank,'calls':len(jobs),
        'physical_gpu':plan['devices'][rank]['index'],'peak_gpu_gib':torch.cuda.max_memory_allocated(rank)/2**30,'training_updates':0}))


def collect(folder,plan):
    from transformers import AutoTokenizer
    from training.step84_repair_plan import assess,write_report
    tokenizer=AutoTokenizer.from_pretrained(plan['base_model'],revision=plan['revision'],local_files_only=True)
    cases={c['name']:c for c in plan['cases']};rows=[]
    for job in plan['jobs']:
        if job['cached'] is not None:
            d=job['cached'];rows.append(assess(cases[job['case']],job['arm'],d['text'],tokenizer,d['truncated'],job['rank'],True))
    for rank in (0,1):
        path=Path(folder)/f'decisions_rank{rank}.jsonl'
        if path.exists():rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected={(j['arm'],j['case']) for j in plan['jobs']}
    if len(rows)!=len(expected) or {(r['arm'],r['case']) for r in rows}!=expected:raise ValueError('Missing or duplicate ablation decisions')
    if any(r['rank']!=next(j['rank'] for j in plan['jobs'] if (j['arm'],j['case'])==(r['arm'],r['case'])) for r in rows):raise ValueError('GPU route changed')
    (Path(folder)/'decisions.json').write_text(json.dumps(rows,indent=2)+'\n')
    return write_report(folder,rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='0,1')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/basic_checks');a=p.parse_args()
    if a.worker:return worker(a.worker)
    from training.step84_repair_plan import prepare
    from training.local_resources import output_parent,runtime_cache,training_gpu_plan
    print('CPU: fixed checkpoint, current agents/example_student/buyer personas, cached short control and new engine fixtures...',flush=True)
    plan=prepare();parent,note=output_parent(ROOT,a.output_dir,min_free_gib=2.)
    if note:print(note,flush=True)
    folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_step84_repair_',dir=parent))
    for arm,texts in plan['personas'].items():
        sub=folder/'personas'/arm;sub.mkdir(parents=True)
        for i,text in enumerate(texts,1):(sub/f'persona_{i}.md').write_text(text+'\n')
    print(f'Plan: {plan["gpu_calls"]} NEW GPU calls + {plan["cached_calls"]} labelled cached baseline calls; zero training updates. Output: {folder}',flush=True)
    if a.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n');print('PASS: CPU preparation; no GPU query, model execution or services.');return folder
    cards,skipped,selection=training_gpu_plan(a.gpus,2)
    if len(cards)!=2:raise ValueError('Two usable GPUs required')
    plan.update(devices=cards,skipped_gpus=skipped,gpu_selection=selection)
    (folder/'plan.json').write_text(json.dumps(plan,indent=1)+'\n')
    cache_env,cache_note=runtime_cache(ROOT)
    if cache_note:print(cache_note,flush=True)
    env={**os.environ,**cache_env,'CUDA_VISIBLE_DEVICES':','.join(c['index'] for c in cards),'PYTHONUNBUFFERED':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
        'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker',str(folder)]
    print(f'GPU devices: {[c["index"] for c in cards]}; independent identical step84 replicas, matched case route across arms. Loading ETA estimating.',flush=True)
    began=time.monotonic();last=-1
    with (folder/'runner.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while chunk:=proc.stdout.read1(4096):
                text=chunk.decode(errors='replace');sys.stdout.write(text);sys.stdout.flush();log.write(text);log.flush()
                done=sum(len(f.read_text().splitlines()) for f in folder.glob('decisions_rank*.jsonl'))
                if done!=last:
                    elapsed=time.monotonic()-began;speed=done/elapsed
                    eta=f'{(plan["gpu_calls"]-done)/speed:.0f}s' if speed else 'estimating'
                    line=f'[Overall] {done}/{plan["gpu_calls"]} new calls | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n'
                    print(line,end='',flush=True);log.write(line);log.flush();last=done
            code=proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps({'exit':code,'elapsed_seconds':time.monotonic()-began,'training_updates':0}))
    if code:raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    result=collect(folder,plan)
    print(f'Completed: {folder}/repair_review.md; small-market gate={result["eligible_for_small_market"]}. No automatic next GPU work.',flush=True)
    return folder


if __name__=='__main__':main()
