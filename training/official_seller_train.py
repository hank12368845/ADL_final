"""Native CE Seller baseline, full fixed subset epoch and autonomous checkpoints."""
from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
from datetime import timedelta
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

from training.official_seller_data import ROOT, PERSONAS, audit_data, current_system, snapshot_review


def batch_indices(size, batch, seed):
    order=list(range(size));random.Random(seed).shuffle(order)
    extra=(-size)%batch
    order.extend((order*((extra//max(1,size))+1))[:extra])
    return [order[i:i+batch] for i in range(0,len(order),batch)]


def frozen_cases(cfg, batches):
    """Official held-out/seen rows plus engine-generated, zero-answer dev situations."""
    from adlmarket.config import AgentSpec
    from adlmarket.market import Market, load_catalog
    from adlmarket.schema import build_items_schema
    from adlmarket.prompts import observation_message
    from training.official_seller_data import current_spec
    data=ROOT/cfg['data_dir'];cases=[]
    for split, count in [('validation',2),('train',1)]:
        rows=[json.loads(l) for l in (data/f'{split}.jsonl').read_text().splitlines()]
        if split=='train':
            allowed={rows[i]['id'] for batch in batches[:cfg['checkpoint_steps'][0]] for i in batch}
            rows=[r for r in rows if r['id'] in allowed]
        pools=defaultdict(list)
        for r in rows:pools[r['label_action']].append(r)
        for action,pool in sorted(pools.items()):
            random.Random(cfg['seed']+len(action)).shuffle(pool)
            for r in pool[:count]:
                cases.append(dict(name=split+':'+r['id'],kind='official_heldout' if split=='validation' else 'seen',
                                  persona_index=r['persona_index'],messages=r['messages'],tools=r['tools'],situation=r['situation'],
                                  observation=r['observation'],expected_action=action))
    spec=current_spec();catalog=load_catalog(ROOT/spec.catalog)
    schema=build_items_schema('seller_native_dev',spec.schema_,spec.schema_.items_file(0),catalog,ROOT,5)
    sellers=[AgentSpec(id=f'probe_s{i}',owner='probe',role='seller',model='base') for i in range(5)]
    buyers=[AgentSpec(id=f'probe_b{i}',owner='ta',role='buyer',model='ta') for i in range(10)]
    original=Market.generate('seller_native_dev',spec,schema,{p.id:p for p in catalog},sellers,buyers,random.Random(20261122))
    for persona in (1,2,3):
        for family in ('stocked','buyer_interest','empty_stock','offer_limit'):
            market=copy.deepcopy(original);market.round=1;market.tick=1;aid=sellers[persona-1].id
            book=market.sellers[aid];pid=max((p for p,n in book.stock.items() if n),key=lambda p:market.products[p].ref_price-book.cost[p])
            if family=='empty_stock':book.stock={p:0 for p in book.stock}
            if family=='offer_limit':
                for i in range(8):
                    result=market.apply(aid,'make_offer',dict(buyer_id=buyers[i].id,product_id=pid,price=market.products[pid].ref_price/100,qty=1))
                    if not result.ok:raise ValueError('Cannot construct engine offer-limit probe')
            sit=market.situation(aid);obs=market.observe(aid,turns_left=6,mark_seen=False);obs['output_tokens_left_this_round']=8192
            if family=='buyer_interest':
                obs['new_messages'].append(dict(channel='dm',**{'from':buyers[0].id},text=f'Please send a binding offer for one {pid} at ${market.products[pid].ref_price/100:.2f}.'))
            text=(PERSONAS/f'persona_{persona}.md').read_text();messages=[dict(role='system',content=current_system(sit,text)),dict(role='user',content=observation_message(obs,'',None))]
            case=dict(name=f'current_p{persona}_{family}',kind='current_rule',family=family,persona_index=persona,
                      situation=sit,observation=obs,messages=messages,tools=market.tool_schemas('seller'),quote_opportunity=family!='empty_stock')
            cases.append(case)
            if persona==1 and family in ('stocked','buyer_interest'):
                control=copy.deepcopy(case);control['name']='p1_v1_control_'+family;control['kind']='p1_v1_control'
                control['messages'][0]['content']=current_system(sit,(ROOT/'training/persona_versions/seller_checkpoint1_v1/persona_1.md').read_text())
                cases.append(control)
    return cases


def summary(rows):
    output={}
    for kind in sorted({r['kind'] for r in rows}):
        part=[r for r in rows if r['kind']==kind]
        output[kind]=dict(calls=len(part),valid=sum(r['valid'] for r in part),quotes=sum(r['profitable_quote'] for r in part),
                          quote_opportunities=sum(bool(r.get('quote_opportunity')) for r in part),
                          captured=sum(bool(r.get('quote_opportunity')) and r['profitable_quote'] for r in part),
                          empty_correct_wait=sum(r.get('family')=='empty_stock' and r['valid'] and r['tool']=='wait' for r in part),
                          truncated=sum(r['truncated'] for r in part),no_tool=sum(not r['tool'] for r in part),
                          actions=dict(Counter(r['tool'] or 'NO_TOOL' for r in part)),errors=dict(Counter(e for r in part for e in r['errors'])))
    return output


def catastrophic(base, now):
    a,b=base['current_rule'],now['current_rule']
    return b['captured'] <= a['captured']-4 or b['valid'] <= a['valid']-4 or (a['captured']>0 and b['captured']==0)


def kernel_dispatch():
    """Inspect HF's chosen original-package implementation, not just import success."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    result={}
    for name in ('torch_chunk_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update'):
        function=getattr(module,name);visited=set();evidence=[]
        def walk(fn):
            if not inspect.isfunction(fn) or id(fn) in visited:return
            visited.add(id(fn));values=inspect.getclosurevars(fn).nonlocals
            impl=values.get('implementation')
            if impl is not None:evidence.append(dict(module=getattr(impl,'__module__',''),name=getattr(impl,'__qualname__',''),external=values.get('is_new_implementation',False)))
            for value in values.values():
                if inspect.isfunction(value):walk(value)
            if hasattr(fn,'__wrapped__'):walk(fn.__wrapped__)
        walk(function);result[name]=evidence
    if not all(any(e['external'] and e['module'].startswith(('fla','causal_conv1d')) for e in evidence) for evidence in result.values()):
        raise RuntimeError(f'Fast-kernel dispatch unavailable; no silent slow training. Evidence:{result}')
    return result


def worker(folder):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    from training.buyer_operations_train import adapter_model,completion_losses,save_adapter
    from training.buyer_surgical_train import install_chunked_mlp
    from training.official_buyer_train import generation_stop_ids,preflight_training_mode,validation_selection
    from training.reasoning_checkpoint import save_state,load_optimizer,restore_rng,load_weights
    from training.decision_ablation import fingerprint,check_fingerprint
    from training.buyer_hf_check import parse_raw
    from training.train_official_seller import check_run
    folder=Path(folder);cfg=check_run(folder);data,manifest=audit_data(cfg);train=data['train'];val=data['validation']
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=cfg['world_size']:raise ValueError('Prepared GPU world changed')
    device=torch.device('cuda',local);torch.cuda.set_device(local);torch.cuda.set_per_process_memory_fraction(cfg['devices'][local]['fraction'],local)
    if world>1:dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=45))
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    dispatch=kernel_dispatch()
    print(f'[rank{rank}] Fast kernels selected; loading pinned bf16 base + fresh Seller rank16 LoRA; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,
                                               device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base weights')
    base.config.use_cache=False;model=adapter_model(base,cfg);start=cfg.get('start_step',0)
    if start:load_weights(model,folder/'checkpoints'/f'step_{start:04d}')
    elif any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n):raise ValueError('Baseline LoRA must have zero initial effect')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    layers=preflight_training_mode(model);mlps=install_chunked_mlp(model,cfg['mlp_chunk_tokens'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    stops=generation_stop_ids(model,tokenizer);batches=json.loads((folder/'batch_plan.json').read_text())['batches'];cases=json.loads((folder/'native_cases.json').read_text())
    validation=validation_selection(val.metadata,cfg['validation_max'],cfg['seed'])
    def barrier():
        if world>1:dist.barrier()
    def gather(payload):
        if world==1:return [payload]
        all_parts=[None]*world;dist.all_gather_object(all_parts,payload);return all_parts
    print(f'[rank{rank}] Longest-example forward/backward preflight, active checkpoint layers={layers}, MLP chunks={mlps}; ETA estimating.',flush=True)
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device));began=time.monotonic()
    for index in sorted({max(range(len(train)),key=lambda i:len(train[i]['input_ids'])),max(range(len(train)),key=lambda i:train[i]['target_length'])}):
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,train[index],device)['completion']
        if not torch.isfinite(loss):raise ValueError('Nonfinite preflight')
        loss.backward();model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device)
    (folder/f'preflight_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=cfg['devices'][local]['index'],
        gpu=torch.cuda.get_device_name(local),kernel_dispatch=dispatch,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
        seconds=time.monotonic()-began,checkpointed_layers=layers,chunked_mlps=mlps),indent=2)+'\n');barrier()
    def evaluate(step):
        model.eval();losses=[];decisions=[]
        with torch.inference_mode():
            for i in tqdm(validation[rank::world],desc=f'Seller validation step{step}',disable=rank!=0,unit='samples'):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,val[i],device)['completion']
                losses.append(dict(action=val.metadata[i]['label_action'],loss=loss.item()))
            for case in tqdm(cases[rank::world],desc=f'Seller free generation step{step}',disable=rank!=0,unit='calls'):
                prompt=tokenizer.apply_chat_template(case['messages'],tools=case['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=cfg['probe_max_tokens'],do_sample=False,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=stops)
                tail=out[0,ids.input_ids.shape[1]:];text=tokenizer.decode(tail,skip_special_tokens=False);parsed=parse_raw(tokenizer,text,case['tools'])
                try:review=snapshot_review(case['situation'],case['observation'],parsed['tool'],parsed['arguments'],case['tools'])
                except (ValueError,TypeError,KeyError,ArithmeticError):
                    review=dict(valid=False,errors=['malformed_action_values'],profitable_quote=False)
                truncated=len(tail)>=cfg['probe_max_tokens'] and int(tail[-1]) not in stops
                if truncated:review['valid']=False;review['profitable_quote']=False;review['errors'].append('truncated')
                decisions.append(dict(name=case['name'],kind=case['kind'],persona_index=case['persona_index'],family=case.get('family'),
                    quote_opportunity=case.get('quote_opportunity'),tool=parsed['tool'],arguments=parsed['arguments'],generated_tokens=len(tail),
                    truncated=truncated,text=text,reference_action_match=parsed['tool']==case.get('expected_action'),**review))
        parts=gather(dict(losses=losses,decisions=decisions));all_loss=[r for p in parts for r in p['losses']];rows=[r for p in parts for r in p['decisions']]
        stats=dict(step=step,completion_loss=sum(r['loss'] for r in all_loss)/len(all_loss),behavior=summary(rows),
                   by_action={a:sum(r['loss'] for r in all_loss if r['action']==a)/sum(r['action']==a for r in all_loss) for a in {r['action'] for r in all_loss}})
        if rank==0:
            (folder/f'native_step_{step:04d}.json').write_text(json.dumps(dict(summary=stats,decisions=rows),ensure_ascii=False,indent=2)+'\n')
            c=stats['behavior']['current_rule'];print(f'Seller step{step}: NLL{stats["completion_loss"]:.4f}; current legal{c["valid"]}/{c["calls"]}, profitable quotes{c["captured"]}/{c["quote_opportunities"]}, truncated{c["truncated"]}. Snapshot diagnostics, not market profit.',flush=True)
        model.train();barrier();return stats
    if not start:baseline=evaluate(0)
    else:
        baseline=json.loads((folder/'native_step_0000.json').read_text())['summary']
        if not (folder/f'native_step_{start:04d}.json').is_file():evaluate(start)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def rate(step):
        if step<cfg['warmup_steps']:return (step+1)/cfg['warmup_steps']
        return .5*(1+math.cos(math.pi*min(1,(step-cfg['warmup_steps'])/max(1,cfg['max_steps']-cfg['warmup_steps']))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    if start:
        cp=folder/'checkpoints'/f'step_{start:04d}';load_optimizer(optimizer,scheduler,cp,device,start);restore_rng(cp,rank,device,start)
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    accumulation=cfg['effective_batch_size']//world;status='complete';began=time.monotonic();step=start
    with (folder/f'metrics_rank{rank}.jsonl').open('a') as metrics,(folder/f'samples_rank{rank}.jsonl').open('a') as trace:
        bar=tqdm(range(start+1,cfg['max_steps']+1),initial=start,total=cfg['max_steps'],disable=rank!=0,unit='steps',desc='Seller official epoch1')
        for step in bar:
            optimizer.zero_grad(set_to_none=True);mean=torch.zeros((),device=device)
            for micro,i in enumerate(batches[step-1][rank::world]):
                row=train[i];trace.write(json.dumps(dict(step=step,rank=rank,id=row['id'],label_action=row['label_action'],group=row['group']))+'\n');trace.flush()
                if rank==0:bar.set_description(f'Seller {step}/{cfg["max_steps"]} micro{micro+1}/{accumulation}')
                with wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(wrapped,row,device)['completion']
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    (loss/accumulation).backward();mean+=loss.detach()/accumulation
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            if world>1:dist.all_reduce(mean)
            elapsed=time.monotonic()-began;speed=(step-start)/elapsed
            record=dict(step=step,completion_loss=mean.item()/world,lr=lr,elapsed_seconds=elapsed,eta_seconds=(cfg['max_steps']-step)/speed,examples=step*cfg['effective_batch_size'],peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            metrics.write(json.dumps(record)+'\n');metrics.flush()
            if rank==0:bar.set_postfix(loss=f'{record["completion_loss"]:.3f}',lr=f'{lr:.2g}',samples=f'{record["examples"]}/{len(batches)*cfg["effective_batch_size"]}')
            if step in cfg['checkpoint_steps']:
                cp=folder/'checkpoints'/f'step_{step:04d}'
                if rank==0:save_adapter(model,cp,cfg,step)
                barrier();save_state(cp,model,optimizer,scheduler,step,rank,device);barrier()
                if rank==0:(cp/'checkpoint_complete.json').write_text(json.dumps(dict(step=step,adapter_fingerprints=fingerprint(cp))))
                score=evaluate(step)
                if catastrophic(baseline['behavior'],score['behavior']):
                    status='stopped_behavior_regression';print(f'[rank{rank}] STOP: autonomous quote/valid-action regression. Checkpoint retained; no extension.',flush=True);break
    if rank==0:
        (folder/'training_finished.json').write_text(json.dumps(dict(status=status,step=step,target_steps=cfg['max_steps'],training_seconds=time.monotonic()-began,
            last_checkpoint=str(folder/'checkpoints'/f'step_{step:04d}'),automatic_promotion=False),indent=2)+'\n')
    barrier()
    if world>1:dist.destroy_process_group()


def write_review(folder):
    folder=Path(folder);cfg=json.loads((folder/'train_config.json').read_text());finished=json.loads((folder/'training_finished.json').read_text())
    scores=[json.loads(p.read_text())['summary'] for p in sorted(folder.glob('native_step_*.json'))]
    lines=['# Official Seller baseline review','',f'Status: {finished["status"]}; {finished["step"]}/{cfg["max_steps"]} optimizer steps.',
           f'Official curated train={cfg["train_examples"]}, one fixed subset epoch; Thinking ON; rank16. Original base at step0.',
           '', '| step | held-out NLL | current-rule valid | profitable quotes | empty-stock wait | held-out valid | seen valid | truncations (all cases) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    base=scores[0]['behavior'];eligible=[]
    for score in scores:
        b=score['behavior'];c=b['current_rule'];o=b['official_heldout'];seen=b['seen']
        lines.append(f'| {score["step"]} | {score["completion_loss"]:.4f} | {c["valid"]}/{c["calls"]} | {c["captured"]}/{c["quote_opportunities"]} | {c["empty_correct_wait"]}/3 | {o["valid"]}/{o["calls"]} | {seen["valid"]}/{seen["calls"]} | {sum(r["truncated"] for r in b.values())} |')
        if score['step'] and c['captured']>=max(1,base['current_rule']['captured']-1) and c['valid']>=base['current_rule']['valid'] and o['valid']>=base['official_heldout']['valid']:
            eligible.append(score)
    best=max(eligible,key=lambda s:(s['behavior']['current_rule']['valid'],s['behavior']['current_rule']['captured'],s['behavior']['official_heldout']['valid'],-s['completion_loss'])) if eligible else None
    candidate=str(folder/'checkpoints'/f'step_{best["step"]:04d}') if best else None
    lines+=['',f'Development candidate: {candidate}',
            'These are autonomous greedy HF snapshot checks, not realized profit, vLLM serving verification, or course grades. Teacher action agreement is not an optimality label.',
            'Current-rule probes are engine-generated development states; official holdouts are whole-market splits. Seen cases are guaranteed in the first checkpoint training prefix.',
            'No deployment, automatic extra training, or market run. A fixed-persona full market versus starter is required before promotion. Raw generations: native_step_*.json.']
    (folder/'seller_training_review.md').write_text('\n'.join(lines)+'\n')
    result=dict(**finished,development_candidate=candidate,complete=True,automatic_market=False)
    (folder/'finished.json').write_text(json.dumps(result,indent=2)+'\n');return result
