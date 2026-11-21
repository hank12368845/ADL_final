"""Bounded Seller continuation; original native CE, fixed persona, explicit behavior gate."""
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

from training.seller_consistency_data import DATA,audit,engine_for,horizon_findings,own_texts,stock_assertions
from training.official_seller_data import snapshot_review


def score(case,parsed,text,truncated):
    """Engine acceptance and strategy guards reported separately; no profit claim."""
    tool,args=parsed.get('tool'),parsed.get('arguments') or {}
    if not isinstance(args,dict):args={}
    public_args={k:v for k,v in args.items() if k!='private_state'}
    try:
        # The formal policy consumes/truncates private_state before Market.apply.
        result=snapshot_review(case['situation'],case['observation'],tool,public_args,case['tools'])
        engine=engine_for(case).apply(case['situation']['your_id'],tool,public_args).ok
    except (ValueError,KeyError,TypeError,ArithmeticError):
        result=dict(valid=False,profitable_quote=False,errors=['malformed_action']);engine=False
    # raw text includes reasoning and the tool. Public messages are deliberately
    # excluded from truth checking; inspect only own thinking/private memory.
    reason=text.split('</think>',1)[0]
    thinking=reason.rsplit('<think>',1)[-1]
    private_note=str(args.get('private_state','') or '')
    own=[thinking,private_note]
    horizon=[f for t in own for f in horizon_findings(t,case['observation']['round']) if f['kind']!='allowed_offer_expiry']
    assertions=stock_assertions(private_note)
    wrong_stock=[(p,n) for p,n in assertions if case['observation']['you']['stock'].get(p)!=n]
    if truncated:result['errors'].append('truncated');result['valid']=False;result['profitable_quote']=False
    close=text.find('</think>');call=text.find('<tool_call>')
    if close<0 or (call>=0 and call<close):
        result['errors'].append('thinking_not_closed_before_tool');result['valid']=False;result['profitable_quote']=False
    final=case['observation']['round']==5 and case['observation']['turns_left_this_round']==1
    # A conservative language screen also flags explicit discussion rejecting an
    # old R6 plan. Report it for manual reading; never call that proof of failure.
    safe=result['valid'] and not wrong_stock and (not final or tool=='wait')
    return dict(**result,engine_accepted=engine,safe=safe,horizon_risks=horizon,target_stock_mismatches=wrong_stock,
                expected_quote=bool(case.get('expected_quote')),truncated=truncated,
                control_correct=safe and (result['profitable_quote'] if case.get('expected_quote') else tool=='wait') if case['kind']=='heldout_controls' else None)


def summarize(rows):
    out={}
    for kind in sorted({r['kind'] for r in rows}):
        part=[r for r in rows if r['kind']==kind]
        out[kind]=dict(calls=len(part),valid=sum(r['valid'] for r in part),safe=sum(r['safe'] for r in part),
            quote_opportunities=sum(r['expected_quote'] for r in part),
            captured=sum(r['expected_quote'] and r['profitable_quote'] for r in part),
            correct=sum(bool(r['control_correct']) for r in part),truncated=sum(r['truncated'] for r in part),
            missing_tool=sum(not r['tool'] for r in part),horizon_risk=sum(bool(r['horizon_risks']) for r in part),
            stock_mismatch=sum(bool(r['target_stock_mismatches']) for r in part),
            actions=dict(Counter(r['tool'] or 'NO_TOOL' for r in part)),
            by_persona={str(p):dict(valid=sum(r['valid'] for r in part if r['persona_index']==p),
                safe=sum(r['safe'] for r in part if r['persona_index']==p),
                captured=sum(r['expected_quote'] and r['profitable_quote'] for r in part if r['persona_index']==p),
                correct=sum(bool(r['control_correct']) for r in part if r['persona_index']==p)) for p in {r['persona_index'] for r in part}})
    return out


def gate(parent,candidate):
    """Preserve action capture, require visible repair; never select by scalar NLL."""
    p,c=parent,candidate
    checks=dict(
        positive_capture=c['positive_regression']['captured']>=max(1,p['positive_regression']['captured']-1),
        positive_valid=c['positive_regression']['valid']>=p['positive_regression']['valid'],
        official_valid=c['official_heldout']['valid']>=p['official_heldout']['valid'],
        controls_preserved=c['heldout_controls']['correct']>=p['heldout_controls']['correct'],
        control_quotes=c['heldout_controls']['captured']>=p['heldout_controls']['captured'],
        real_safe=c['real_market_development']['safe']>=p['real_market_development']['safe'],
        no_extra_truncation=sum(x['truncated'] for x in c.values())<=sum(x['truncated'] for x in p.values()),
        visible_repair=c['heldout_controls']['correct']>p['heldout_controls']['correct'] or c['real_market_development']['safe']>p['real_market_development']['safe'])
    checks['personas_preserved']=all(
        c[kind]['by_persona'][persona][metric]>=value[metric]
        for kind,metric in [('heldout_controls','correct'),('real_market_development','safe'),('official_heldout','valid')]
        for persona,value in p[kind]['by_persona'].items())
    return dict(passed=all(checks.values()),checks=checks,
        note='Small development gate; not proof of realized profit. No automatic deployment or market.')


def assignments(cases,tokenizer,world):
    """Balance long prompt/output cases across independent HF inference replicas."""
    costs=[]
    for i,c in enumerate(cases):
        ids=tokenizer.apply_chat_template(c['messages'],tools=c['tools'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if len(ids)+c.get('max_new_tokens',1024)>64000:raise ValueError('Probe exceeds formal context limit')
        costs.append((len(ids)+4*c.get('max_new_tokens',1024),i))
    result=[[] for _ in range(world)];loads=[0]*world
    for cost,i in sorted(costs,reverse=True):
        r=min(range(world),key=lambda r:(loads[r],r));result[r].append(i);loads[r]+=cost
    return result


def worker(folder):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tqdm import tqdm
    from training.train_seller_consistency import check_run
    from training.buyer_operations_train import TokenRows,adapter_model,completion_losses,save_adapter
    from training.buyer_surgical_train import install_chunked_mlp,activation_context
    from training.official_buyer_train import generation_stop_ids,preflight_training_mode
    from training.official_seller_train import kernel_dispatch
    from training.reasoning_checkpoint import load_weights,save_state,load_optimizer,restore_rng
    from training.decision_ablation import fingerprint
    folder=Path(folder);cfg=check_run(folder);audit(cfg['data_dir'])
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=cfg['world_size']:raise ValueError('GPU world changed')
    device=torch.device('cuda',local);torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(cfg['devices'][local]['fraction'],local)
    if world>1:dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    def barrier():
        if world>1:dist.barrier()
    def gather(value):
        if world==1:return [value]
        out=[None]*world;dist.all_gather_object(out,value);return out
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);dispatch=kernel_dispatch()
    print(f'[rank{rank}] Loading pinned bf16 base + fixed Seller step12; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base')
    base.config.use_cache=False;model=adapter_model(base,cfg);start=cfg.get('start_step',0)
    load_weights(model,folder/'checkpoints'/f'step_{start:04d}' if start else cfg['parent_adapter'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    layers=preflight_training_mode(model);mlps=install_chunked_mlp(model,cfg['mlp_chunk_tokens'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True);stops=generation_stop_ids(model,tokenizer)
    data=Path(cfg['data_dir']);train=TokenRows(data/'train.tokens.jsonl',cfg['max_length']);val=TokenRows(data/'validation.tokens.jsonl',cfg['max_length'])
    batches=json.loads((data/'batch_plan.json').read_text());cases=json.loads((folder/'native_cases.json').read_text());routing=assignments(cases,tokenizer,world)
    if rank==0:(folder/'inference_routing.json').write_text(json.dumps(dict(indices=routing,names=[[cases[i]['name'] for i in part] for part in routing]),indent=2)+'\n')
    print(f'[rank{rank}] Longest full-context backward preflight; activation offload ≥{cfg["activation_offload_from_tokens"]}; ETA estimating.',flush=True)
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device));began=time.monotonic()
    for index in sorted({max(range(len(train)),key=lambda i:len(train[i]['input_ids'])),max(range(len(train)),key=lambda i:train[i]['target_length'])}):
        row=train[index]
        with activation_context(row,cfg):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,row,device)['completion']
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):raise ValueError('Missing, nonfinite or zero preflight gradients')
        model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device);barrier()
    (folder/f'preflight_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=cfg['devices'][local]['index'],
        seconds=time.monotonic()-began,checkpointed_layers=layers,chunked_mlps=mlps,kernel_dispatch=dispatch,
        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30),indent=2)+'\n')
    def evaluate(step):
        model.eval();rows=[];losses=[]
        with torch.inference_mode():
            for i in tqdm(range(rank,len(val),world),desc=f'Validation step{step} rank{rank}',unit='samples'):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,val[i],device)['completion']
                losses.append(loss.item())
            for i in tqdm(routing[rank],desc=f'Seller decisions step{step} rank{rank}',unit='calls'):
                c=cases[i];maximum=c.get('max_new_tokens',1024)
                prompt=tokenizer.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=maximum,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=stops)
                tail=out[0,ids.input_ids.shape[1]:];text=tokenizer.decode(tail,skip_special_tokens=False)
                from training.buyer_hf_check import parse_raw
                parsed=parse_raw(tokenizer,text,c['tools']);truncated=len(tail)>=maximum and int(tail[-1]) not in stops
                review=score(c,parsed,text,truncated)
                rows.append(dict(name=c['name'],kind=c['kind'],persona_index=c['persona_index'],family=c.get('family'),rank=rank,
                    tool=parsed['tool'],arguments=parsed['arguments'],text=text,prompt_tokens=ids.input_ids.shape[1],generated_tokens=len(tail),**review))
                with (folder/f'decisions_step{step:04d}_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(rows[-1],ensure_ascii=False)+'\n')
        parts=gather(dict(rows=rows,losses=losses));all_rows=[r for p in parts for r in p['rows']];all_loss=[x for p in parts for x in p['losses']]
        behavior=summarize(all_rows)
        if rank==0:
            stats=dict(step=step,parent_checkpoint='step12',completion_loss=sum(all_loss)/len(all_loss),behavior=behavior)
            (folder/f'native_step_{step:04d}.json').write_text(json.dumps(dict(summary=stats,decisions=all_rows),ensure_ascii=False,indent=2)+'\n')
            print(f'Seller repair step{step}: {json.dumps(behavior)}. Diagnostics only; no realized profit.',flush=True)
        model.train();barrier();return behavior
    if not start:baseline=evaluate(0)
    else:baseline=json.loads((folder/'native_step_0000.json').read_text())['summary']['behavior']
    if not start and baseline['heldout_controls']['correct']==12 and baseline['real_market_development']['safe']==3:
        if rank==0:(folder/'training_finished.json').write_text(json.dumps(dict(status='stopped_no_reproducible_repair_target',step=0,gate=None),indent=2)+'\n')
        barrier()
        if world>1:dist.destroy_process_group()
        return
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def rate(s):
        if s<cfg['warmup_steps']:return (s+1)/cfg['warmup_steps']
        return .5*(1+math.cos(math.pi*min(1,(s-cfg['warmup_steps'])/(cfg['max_steps']-cfg['warmup_steps']))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    if start:
        cp=folder/'checkpoints'/f'step_{start:04d}';load_optimizer(optimizer,scheduler,cp,device,start);restore_rng(cp,rank,device,start)
        resumed=evaluate(start);decision=gate(baseline,resumed)
        if not decision['passed']:
            if rank==0:
                (folder/'training_finished.json').write_text(json.dumps(dict(status='stopped_step18_gate',step=start,gate=decision),indent=2)+'\n')
            barrier()
            if world>1:dist.destroy_process_group()
            return
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    accumulation=cfg['effective_batch_size']//world;status='complete';step=start;decision=None;began=time.monotonic()
    with (folder/f'metrics_rank{rank}.jsonl').open('a') as metrics,(folder/f'samples_rank{rank}.jsonl').open('a') as trace:
        bar=tqdm(range(start+1,cfg['max_steps']+1),initial=start,total=cfg['max_steps'],disable=rank!=0,desc='Seller bounded repair',unit='steps')
        for step in bar:
            optimizer.zero_grad(set_to_none=True);mean=torch.zeros((),device=device)
            for micro,index in enumerate(batches[step-1][rank::world]):
                row=train[index];trace.write(json.dumps(dict(step=step,rank=rank,id=row['id'],source_kind=row['source_kind'],family=row['family'],label_action=row['label_action'],group=row['group']))+'\n');trace.flush()
                if rank==0:bar.set_description(f'Seller {step}/{cfg["max_steps"]} micro{micro+1}/{accumulation}')
                with wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext():
                    with activation_context(row,cfg):
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(wrapped,row,device)['completion']
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        (loss/accumulation).backward();mean+=loss.detach()/accumulation
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            if world>1:dist.all_reduce(mean)
            elapsed=time.monotonic()-began;speed=(step-start)/elapsed
            record=dict(step=step,completion_loss=mean.item()/world,lr=lr,gradient_norm=norm.item(),elapsed_seconds=elapsed,
                eta_seconds=(cfg['max_steps']-step)/speed,examples=step*8,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            metrics.write(json.dumps(record)+'\n');metrics.flush()
            if rank==0:bar.set_postfix(loss=f'{record["completion_loss"]:.3f}',lr=f'{lr:.2g}',samples=f'{step*8}/288')
            if step in cfg['checkpoint_steps']:
                cp=folder/'checkpoints'/f'step_{step:04d}'
                if rank==0:save_adapter(model,cp,cfg,step)
                barrier();save_state(cp,model,optimizer,scheduler,step,rank,device);barrier()
                if rank==0:(cp/'checkpoint_complete.json').write_text(json.dumps(dict(step=step,adapter_fingerprints=fingerprint(cp))))
                now=evaluate(step);decision=gate(baseline,now)
                if rank==0:(folder/f'gate_step{step:04d}.json').write_text(json.dumps(decision,indent=2)+'\n')
                if not decision['passed']:
                    status=f'stopped_step{step}_gate';print(f'[rank{rank}] STOP: {decision}; no extension or promotion.',flush=True);break
    if rank==0:(folder/'training_finished.json').write_text(json.dumps(dict(status=status,step=step,gate=decision,training_seconds=time.monotonic()-began),indent=2)+'\n')
    barrier()
    if world>1:dist.destroy_process_group()


def write_review(folder):
    folder=Path(folder);finished=json.loads((folder/'training_finished.json').read_text());lines=['# Seller consistency repair review','',
        f"Status: {finished['status']}; additional optimizer steps: {finished['step']} (parent is official step12).",'',
        '| additional step | positive quotes | held-out control correct | official valid | real-context safe | truncations |',
        '|---|---:|---:|---:|---:|---:|']
    for path in sorted(folder.glob('native_step_*.json')):
        r=json.loads(path.read_text())['summary'];b=r['behavior']
        lines.append(f"| {r['step']} | {b['positive_regression']['captured']}/9 | {b['heldout_controls']['correct']}/12 | {b['official_heldout']['valid']}/6 | {b['real_market_development']['safe']}/3 | {sum(p['truncated'] for p in b.values())} |")
    lines+=['',f"Last gate: {json.dumps(finished.get('gate'))}",
        'HF greedy development diagnostics, not official vLLM market scores. Safe means recorded guards passed; horizon detection is conservative, not a full semantic proof.',
        'Real-context cases are known development failures, not unseen markets. Price margins are not realized profit.',
        'Source/target/prompt are fixed; no automatic deployment, additional epoch, checkpoint sweep, or market follows.']
    (folder/'seller_consistency_review.md').write_text('\n'.join(lines)+'\n');return finished
