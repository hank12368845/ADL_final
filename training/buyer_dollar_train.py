"""Two-GPU dollar-line continuation with mandatory step6 early stop."""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]


def summarize(rows):
    from training.buyer_dollar_lines import summarize as line_summary
    from training.buyer_latest_brakes_train import summary as tool_summary
    result={}
    for suite in ('self_check_seen','self_check_validation'):
        chosen=[r for r in rows if r['suite']==suite]
        if not chosen: result[suite]=None; continue
        result[suite]=line_summary(chosen)
        result[suite]['personas']={str(p):line_summary([r for r in chosen if r['persona_index']==p]) for p in (1,2,3)}
        result[suite]['channels']={c:line_summary([r for r in chosen if r['channel']==c]) for c in ('board','offer')}
    original=[r for r in rows if r['suite'] in ('known','fresh','official')]
    result['regression']={k:v for k,v in tool_summary(original).items() if k in ('current','official')} if original else None
    return result


def final_gate(parent,current):
    from training.buyer_dollar_lines import step6_gate
    checks=step6_gate(current['self_check_seen']);v=current['self_check_validation'];old=current['regression'];prior=parent['regression']
    checks.update(validation_fields=v['all_fields_exact']==24, validation_positive_retained=v['captured']>=parent['self_check_validation']['captured'],
        validation_negative_safe=v['negative_safe']==v['negative_total'], validation_state_consistent=v['state_consistent']==24,
        validation_action_consistent=v['line_action_consistent']==24,
        retain_known_fresh=all(old['current'][k]>=prior['current'][k] for k in ('known_captured','fresh_captured')),
        retain_official=old['official']['valid']>=prior['official']['valid'],
        no_cash_owned_regression=all(old[k][f]<=prior[k][f] for k in old for f in ('cash_errors','owned_errors')),
        no_tool_truncation=all(s['no_tool']==0 and s['truncated']==0 for s in (v,*old.values())))
    return checks


def write_review(folder):
    folder=Path(folder);records=[json.loads(p.read_text()) for p in sorted(folder.glob('native_step_*.json'))]
    lines=['# Dollar one-line buyer pilot','',
        'Unchanged latest persona/observation, official step84, thinking ON. Dollar outputs, one current candidate, no supplied prefix or answers.',
        'Step0 is CPU-rescored cached parent data. Step6 tests seen12 ONLY; failed gates stop immediately. Only a passed gate permits training through step12 and all73 cases.',
        'Exact checks are a development convention; native tool legality is scored independently. Seen12 has cent and owned traps, not zero-cash tests.', '',
        '| Step | Suite | Calls | Starts CASH | All fields | Cash | Total | Held | Current value | Adjusted | Afford | Eligible | Positive capture | Negative safe | State consistent |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for suite in ('self_check_seen','self_check_validation'):
            s=r['summary'][suite]
            if s is None: continue
            f=s['target_fields'];values=' | '.join(str(f[k]) for k in ('cash','total','held','next_value','adjusted_value','affordable','eligible'))
            lines.append(f'| {r["step"]} | {suite} | {s["calls"]} | {s["self_started"]} | {s["all_fields_exact"]} | {values} | {s["captured"]}/{s["positive_total"]} | {s["negative_safe"]}/{s["negative_total"]} | {s["state_consistent"]} |')
    for r in records:
        if r['summary']['regression']:
            s=r['summary']['regression'];lines+=['',f'Step{r["step"]} original current: capture {s["current"]["captured"]}/14 (known {s["current"]["known_captured"]}/8; fresh {s["current"]["fresh_captured"]}/6), cash {s["current"]["cash_errors"]}, owned {s["current"]["owned_errors"]}; official valid {s["official"]["valid"]}/9.']
    if (folder/'checkpoint_scores.json').exists(): lines+=['','Gates:','```json',(folder/'checkpoint_scores.json').read_text().strip(),'```']
    if (folder/'finished.json').exists(): lines+=['','Finished:','```json',(folder/'finished.json').read_text().strip(),'```']
    lines+=['','Raw native_step_*.json contains each field, expected values, state errors, and tool consistency; per-persona/channel breakdown is in its summary.',
        'No automatic promotion, data expansion, extra epoch or market evaluation. Stopping at6 is a valid completed experiment, not a launcher failure.']
    (folder/'dollar_review.md').write_text('\n'.join(lines)+'\n')


def worker(folder):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tqdm import tqdm
    from training.buyer_dollar_data import check_data
    from training.buyer_dollar_lines import FIELDS,losses,assess,step6_gate
    from training.buyer_latest_brakes_data import sha
    from training.buyer_operations_train import adapter_model,save_adapter
    from training.reasoning_checkpoint import load_weights,save_state
    from training.official_buyer_train import preflight_training_mode,generation_stop_ids
    from training.decision_ablation import fingerprint
    folder=Path(folder);cfg=json.loads((folder/'train_config.json').read_text())
    for name,digest in cfg['training_sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Launch source changed: {name}')
    if sha(ROOT/cfg['data_dir']/'manifest.json')!=cfg['data_manifest_sha256']: raise ValueError('Manifest changed')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local)
    if int(os.environ['WORLD_SIZE'])!=2: raise ValueError('Two ranks required')
    torch.cuda.set_device(local);torch.cuda.set_per_process_memory_fraction(cfg['devices'][local]['fraction'],local)
    dist.init_process_group('nccl',device_id=device);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    train,val,batches,m=check_data(cfg)
    print(f'[rank{rank}] Loading base + official step84; fresh optimizer; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'): raise ValueError('Incomplete base')
    base.config.use_cache=False;model=adapter_model(base,cfg);load_weights(model,m['parent_adapter'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True);eos=generation_stop_ids(model,t)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/2 if s<2 else .5*(1+math.cos(math.pi*min(1.,(s-2)/10))))
    cases=json.loads((ROOT/cfg['data_dir']/'evaluation_cases.json').read_text())
    baseline=json.loads((ROOT/cfg['data_dir']/'baseline_decisions.json').read_text());parent=summarize(baseline)
    (folder/f'hardware_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=cfg['devices'][local]['index'],name=torch.cuda.get_device_name(local))))
    if rank==0:
        (folder/'native_step_0000.json').write_text(json.dumps(dict(step=0,summary=parent,decisions=baseline,cached_parent=True),indent=2)+'\n');write_review(folder)

    def evaluate(step):
        model.eval();rows=[];chosen=[c for c in cases if step!=6 or c['suite']=='self_check_seen']
        with torch.inference_mode():
            for c in tqdm([c for c in chosen if c['evaluation_rank']==rank],desc=f'rank{rank} native step{step}',position=rank,unit='cases'):
                prompt=t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=t(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model.generate(**ids,max_new_tokens=1024,do_sample=False,use_cache=True,eos_token_id=eos,pad_token_id=t.pad_token_id,return_dict_in_generate=False)
                tail=out[0,ids.input_ids.shape[1]:];raw=t.decode(tail,skip_special_tokens=False)
                truncated=len(tail)>=1024 and int(tail[-1]) not in eos
                rows.append(assess(c,raw,t,truncated,rank))
        parts=[None,None];dist.all_gather_object(parts,rows);rows=[r for part in parts for r in part]
        if len(rows)!=len(chosen) or len({r['case'] for r in rows})!=len(chosen): raise ValueError('Missing/repeated native cases')
        result=dict(step=step,summary=summarize(rows),decisions=rows)
        if rank==0:
            (folder/f'native_step_{step:04d}.json').write_text(json.dumps(result,indent=2)+'\n');write_review(folder)
        model.train();return result

    layers=preflight_training_mode(model);print(f'[rank{rank}] Train-mode memory preflight ({layers} checkpointed layers); ETA estimating.',flush=True)
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device))
    for i in {max(range(len(train)),key=lambda i:len(train[i]['input_ids'])),max(range(len(train)),key=lambda i:train[i]['target_length'])}:
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=losses(model,train[i],device)['total']
        loss.backward();model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device);dist.barrier()
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False);began=time.monotonic();status='complete';scores=[]
    keys=('total','line','fields','response',*('field_'+f for f in FIELDS))
    with (folder/f'metrics_rank{rank}.jsonl').open('w') as metrics,(folder/f'samples_rank{rank}.jsonl').open('w') as samples:
        bar=tqdm(range(1,13),total=12,desc='Dollar checks (gate at6)',unit='steps',disable=rank!=0)
        for step in bar:
            optimizer.zero_grad(set_to_none=True);sums=torch.zeros(len(keys),device=device)
            for micro,i in enumerate(batches[step-1][rank::2]):
                row=train[i];samples.write(json.dumps(dict(step=step,rank=rank,micro=micro,id=row['id'],label_action=row['label_action']))+'\n');samples.flush()
                if rank==0:bar.set_description(f'Dollar step{step}/12 micro{micro+1}/4; gate6')
                with wrapped.no_sync() if micro<3 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=losses(wrapped,row,device)
                    if not torch.isfinite(loss['total']):raise ValueError('Nonfinite loss')
                    (loss['total']/4).backward()
                sums+=torch.stack([loss[k].detach() for k in keys])/4
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();dist.all_reduce(sums)
            record=dict(step=step,loss=dict(zip(keys,(sums/2).tolist())),lr=lr,elapsed_seconds=time.monotonic()-began,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            metrics.write(json.dumps(record)+'\n');metrics.flush()
            if rank==0:bar.set_postfix(line=f'{record["loss"]["line"]:.3f}',fields=f'{record["loss"]["fields"]:.3f}',exposures=f'{step*8}/96')
            if step in (6,12):
                cp=folder/'checkpoints'/f'step_{step:04d}'
                if rank==0:save_adapter(model,cp,cfg,step)
                dist.barrier();result=evaluate(step)
                checks=step6_gate(result['summary']['self_check_seen']) if step==6 else final_gate(parent,result['summary'])
                save_state(cp,model,optimizer,scheduler,step,rank,device);dist.barrier()
                scores.append(dict(step=step,gates=checks,passed=all(checks.values())))
                if rank==0:
                    (cp/'checkpoint_complete.json').write_text(json.dumps(dict(step=step,adapter_fingerprints=fingerprint(cp))))
                    (folder/'checkpoint_scores.json').write_text(json.dumps(scores,indent=2)+'\n');write_review(folder)
                    print(f'Gate step{step}: {json.dumps(checks)}',flush=True)
                if step==6 and not all(checks.values()):status='stopped_step6_gate';break
    if rank==0:
        (folder/'finished.json').write_text(json.dumps(dict(status=status,step=step,development_candidate=None,
            gates=scores,training_and_eval_seconds=time.monotonic()-began,
            note='No automatic promotion, expansion or market run. A step6 stop is deliberate.'),indent=2)+'\n');write_review(folder)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    import argparse,sys
    sys.path.insert(0,str(ROOT));p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);worker(p.parse_args().session)
