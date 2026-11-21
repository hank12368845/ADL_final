"""Small step84 continuation with latest personas and native safety/recall checks."""
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]


def check_data(cfg):
    from training.buyer_latest_brakes_data import sha,VERSION,REFERENCE
    from training.buyer_operations_train import TokenRows
    from training.decision_ablation import check_fingerprint
    folder=ROOT/cfg['data_dir'];manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['version']!=VERSION or not manifest['thinking']:raise ValueError('Wrong latest safety dataset')
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Data/persona source changed: {name}')
    for name,digest in manifest['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Dataset checksum changed')
    check_fingerprint(manifest['parent_adapter'],manifest['parent_fingerprints'])
    if sha(REFERENCE/'decisions.json')!=manifest['reference_decisions_sha256']:raise ValueError('Cached latest baseline source changed')
    for split in ('train','validation'):
        if sha(ROOT/'training/data/buyer_official_v1'/f'{split}.jsonl')!=manifest[f'official_{split}_sha256']:raise ValueError('Official source changed')
    if set(manifest['groups']['train'])&set(manifest['groups']['validation']):raise ValueError('Repair group overlap')
    train=TokenRows(folder/'train.tokens.jsonl',12288);val=TokenRows(folder/'validation.tokens.jsonl',12288)
    batches=json.loads((folder/'batch_plan.json').read_text())
    if len(batches)!=36 or Counter(i for b in batches for i in b)!=Counter(range(len(train))):raise ValueError('Full repair coverage changed')
    for b in batches:
        if len(b)!=8 or Counter(train[i]['label_action'] for i in b)!={'buy_listed':3,'accept_offer':3,'wait':2}:raise ValueError('Unbalanced repair batch')
    return train,val,batches,manifest


def summary(rows):
    result={}
    for label,suites in (('current',('known','fresh')),('new_validation',('repair_validation',)),('official',('official',)),('seen',('seen_train',))):
        selected=[r for r in rows if r['suite'] in suites]
        result[label]={'calls':len(selected),'valid':sum(r['valid'] for r in selected),'positive_total':sum(r['positive'] is True for r in selected),
            'captured':sum(r['captured'] for r in selected),'cash_errors':sum(r['cash_error'] for r in selected),
            'owned_errors':sum('already_owned_trade' in r['errors'] for r in selected),'no_tool':sum(r['tool'] is None for r in selected),
            'truncated':sum(r['truncated'] for r in selected)}
        if label=='current':
            result[label]['known_captured']=sum(r['captured'] for r in selected if r['suite']=='known')
            result[label]['fresh_captured']=sum(r['captured'] for r in selected if r['suite']=='fresh')
    return result


def gates(parent,current):
    return {'retain_current_positive_capture':current['current']['captured']>=parent['current']['captured'],
        'retain_known_and_fresh_capture':all(current['current'][key]>=parent['current'][key] for key in ('known_captured','fresh_captured')),
        'new_positive_capture':current['new_validation']['captured']>=max(10,parent['new_validation']['captured']-1),
        'zero_cash_errors':all(s['cash_errors']==0 for s in current.values()),
        'zero_owned_errors':all(s['owned_errors']==0 for s in current.values()),
        'zero_no_tool_or_truncation':all(s['no_tool']==0 and s['truncated']==0 for s in current.values()),
        'preserve_official_snapshot_validity':current['official']['valid']>=parent['official']['valid']}


def write_review(folder):
    folder=Path(folder);cfg=json.loads((folder/'train_config.json').read_text());results=[]
    reference=Path(cfg.get('reference_run',folder))/'native_step_0000.json'
    if reference.exists():results.append(json.loads(reference.read_text()))
    if cfg.get('reference_run'):
        for path in sorted(Path(cfg['reference_run']).glob('native_step_*.json')):
            r=json.loads(path.read_text())
            if r['step']<=cfg.get('start_step',0) and not any(x['step']==r['step'] for x in results):results.append(r)
    for path in sorted(folder.glob('native_step_*.json')):
        r=json.loads(path.read_text())
        if not any(x['step']==r['step'] for x in results):results.append(r)
    results.sort(key=lambda r:r['step'])
    lines=['# Latest-persona brakes: training review','',
        'Continuation from official step84; original latest personas, ordinary completion CE, no automatic promotion.',
        'Step0 reuses 37 explicitly labelled parent decisions and newly generates 30 validation/seen decisions.',
        'Current/new/official/seen snapshot checks are not market surplus or course grades.','',
        '| Local step | Validation NLL | Current capture /14 | Current cash / owned | New capture /12 | New cash / owned | Official valid /9 | No-tool / truncated (all) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        s=r['summary'];c=s['current'];v=s['new_validation'];o=s['official']
        lines.append(f'| {r["step"]} | {r["validation_loss"]:.4f} | {c["captured"]} | {c["cash_errors"]} / {c["owned_errors"]} | {v["captured"]} | {v["cash_errors"]} / {v["owned_errors"]} | {o["valid"]} | {sum(x["no_tool"] for x in s.values())} / {sum(x["truncated"] for x in s.values())} |')
    finished=folder/'finished.json'
    if finished.exists():lines+=['',finished.read_text()]
    lines+=['','Raw outputs: native_step_*.json. The seen subset is guaranteed trained by local step12; its score cannot establish generalization.']
    (folder/'training_review.md').write_text('\n'.join(lines)+'\n')
    return folder/'training_review.md'


def worker(folder):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tqdm import tqdm
    from training.buyer_operations_train import adapter_model,completion_losses,save_adapter
    from training.reasoning_checkpoint import load_weights,load_optimizer,restore_rng,save_state
    from training.official_buyer_train import preflight_training_mode,generation_stop_ids
    from training.step84_repair_plan import assess
    from training.decision_ablation import fingerprint
    folder=Path(folder);cfg=json.loads((folder/'train_config.json').read_text())
    from training.buyer_latest_brakes_data import sha
    for name,digest in cfg['training_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Training source changed since launch: {name}')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('This experiment requires exactly two ranks')
    torch.cuda.set_device(local);torch.cuda.set_per_process_memory_fraction(cfg['devices'][local]['fraction'],local)
    dist.init_process_group('nccl',device_id=device);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    train,val,batches,manifest=check_data(cfg)
    if rank==0:print('Loading bf16 base + official step84; NEW low-LR optimizer, latest personas; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base')
    base.config.use_cache=False;model=adapter_model(base,cfg);load_weights(model,cfg.get('resume_checkpoint',manifest['parent_adapter']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True);eos=generation_stop_ids(model,tokenizer)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=.01)
    def rate(step):
        if step<3:return (step+1)/3
        return .5*(1+math.cos(math.pi*min(1,(step-3)/33)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate);start=cfg.get('start_step',0)
    if start:load_optimizer(optimizer,scheduler,cfg['resume_checkpoint'],device,start)
    cases=json.loads((ROOT/cfg['data_dir']/'evaluation_cases.json').read_text())
    metrics=(folder/f'metrics_rank{rank}.jsonl').open('w');traces=(folder/f'samples_rank{rank}.jsonl').open('w')
    (folder/f'hardware_rank{rank}.json').write_text(json.dumps({'rank':rank,'physical_gpu':cfg['devices'][local]['index'],
        'name':torch.cuda.get_device_name(local),'total_gib':torch.cuda.get_device_properties(local).total_memory/2**30}))

    def evaluate(step):
        model.eval();rows=[];losses=[]
        with torch.inference_mode():
            for i in tqdm(list(range(len(val)))[rank::2],desc=f'Brake NLL step{step}',disable=rank!=0,unit='examples'):
                with torch.autocast('cuda',dtype=torch.bfloat16):losses.append(completion_losses(model,val[i],device)['completion'].item())
            jobs=[c for c in cases if c['evaluation_rank']==rank]
            for c in tqdm(jobs,desc=f'Native decisions step{step}',disable=rank!=0,unit='cases'):
                cached=c.get('cached_parent') if step==0 else None
                if cached:
                    raw=cached['text'];truncated=cached['truncated']
                else:
                    prompt=tokenizer.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                    ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=1024,do_sample=False,use_cache=True,
                        eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,return_dict_in_generate=False)
                    tail=out[0,ids.input_ids.shape[1]:];raw=tokenizer.decode(tail,skip_special_tokens=False)
                    truncated=len(tail)>=1024 and int(tail[-1]) not in eos
                rows.append(assess(c,'latest',raw,tokenizer,truncated,rank,bool(cached)))
        gathered=[None,None];dist.all_gather_object(gathered,{'rows':rows,'losses':losses})
        merged=[r for part in gathered for r in part['rows']];all_losses=[x for part in gathered for x in part['losses']]
        result={'step':step,'validation_loss':sum(all_losses)/len(all_losses),'summary':summary(merged),'decisions':merged}
        if rank==0:
            (folder/f'native_step_{step:04d}.json').write_text(json.dumps(result,indent=2)+'\n');write_review(folder)
            print(f'Local step{step}: {json.dumps(result["summary"])}',flush=True)
        metrics.write(json.dumps({k:v for k,v in result.items() if k!='decisions'})+'\n');metrics.flush()
        model.train();return result

    n=preflight_training_mode(model)
    if rank==0:print(f'Memory preflight: training mode, {n} checkpointed layers, longest example; ETA estimating.',flush=True)
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device))
    indices={max(range(len(train)),key=lambda i:len(train[i]['input_ids'])),max(range(len(train)),key=lambda i:train[i]['target_length'])}
    for i in sorted(indices):
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,train[i],device)['completion']
        loss.backward();model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device);dist.barrier()
    baseline=cfg.get('baseline_summary');records=cfg.get('previous_scores',[])
    if start:restore_rng(cfg['resume_checkpoint'],rank,device,start)
    else:baseline=evaluate(0)['summary']
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False);model.train();began=time.monotonic();status='complete'
    bar=tqdm(range(start+1,37),initial=start,total=36,disable=rank!=0,desc='Latest brakes',unit='steps')
    for step in bar:
        optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device)
        for micro,i in enumerate(batches[step-1][rank::2]):
            row=train[i];traces.write(json.dumps({'step':step,'rank':rank,'micro':micro,'id':row['id'],'source':row['source'],'label_action':row['label_action']})+'\n');traces.flush()
            if rank==0:bar.set_description(f'Latest brakes {step}/36, micro {micro+1}/4')
            with wrapped.no_sync() if micro<3 else nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(wrapped,row,device)['completion']
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss/4).backward();loss_sum+=loss.detach()/4
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();dist.all_reduce(loss_sum)
        record={'stage':'train','step':step,'loss':loss_sum.item()/2,'lr':lr,'elapsed_seconds':time.monotonic()-began,
            'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30}
        metrics.write(json.dumps(record)+'\n');metrics.flush()
        if rank==0:bar.set_postfix(loss=f'{record["loss"]:.3f}',samples=f'{step*8}/288')
        if step in (12,24,36):
            cp=folder/'checkpoints'/f'step_{step:04d}'
            if rank==0:save_adapter(model,cp,cfg,step)
            dist.barrier();result=evaluate(step);result['gates']=gates(baseline,result['summary']);result['checkpoint_path']=str(cp)
            save_state(cp,model,optimizer,scheduler,step,rank,device);dist.barrier()
            small={k:v for k,v in result.items() if k!='decisions'};records.append(small)
            if rank==0:
                (cp/'checkpoint_complete.json').write_text(json.dumps({'step':step,'baseline_summary':baseline,'adapter_fingerprints':fingerprint(cp)}))
                (folder/'checkpoint_scores.json').write_text(json.dumps(records,indent=2)+'\n')
            if result['summary']['current']['captured']<=baseline['current']['captured']-3 or result['summary']['new_validation']['captured']<=baseline['new_validation']['captured']-4:
                status='stopped_recall_regression';break
    if rank==0:
        eligible=[r for r in records if all(r['gates'].values())]
        chosen=min(eligible,key=lambda r:(-r['summary']['new_validation']['captured'],r['step'])) if eligible else None
        (folder/'finished.json').write_text(json.dumps({'status':status,'step':step,'development_candidate':chosen['checkpoint_path'] if chosen else None,
            'training_seconds':time.monotonic()-began,'baseline_summary':baseline,
            'note':'No automatic promotion or market evaluation; original latest personas unchanged.'},indent=2)+'\n');write_review(folder)
    metrics.close();traces.close();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    import sys,argparse
    sys.path.insert(0,str(ROOT));p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True)
    worker(p.parse_args().session)
