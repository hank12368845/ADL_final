"""Official-data SFT worker: full fixed epoch, standard CE, native free-generation checks."""
import argparse
from collections import Counter,defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]


def check_data(cfg):
    from training.buyer_operations_train import TokenRows
    from training.official_buyer_data import sha
    folder=ROOT/cfg['data_dir'];manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['version']!='buyer_official_sft_v1' or not manifest['thinking'] or manifest['revision']!=cfg['revision']:raise ValueError('Wrong official dataset/mode/revision')
    for name,expected in manifest['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Official preprocessing source changed')
    splits={}
    for split in ('train','validation','test'):
        for suffix,key in (('jsonl','raw_sha256'),('tokens.jsonl','tokens_sha256')):
            if sha(folder/f'{split}.{suffix}')!=manifest['splits'][split][key]:raise ValueError('Official split checksum mismatch')
        splits[split]=TokenRows(folder/f'{split}.tokens.jsonl',cfg['max_length'])
        allowed=set(manifest['groups'][split])
        if any(m['group'] not in allowed for m in splits[split].metadata):raise ValueError('Market group leakage')
    groups=manifest['groups']
    if any(set(groups[a])&set(groups[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))):raise ValueError('Overlapping official market groups')
    return splits['train'],splits['validation'],manifest


def batch_indices(size,batch,seed):
    order=list(range(size));random.Random(seed).shuffle(order)
    padding=(-len(order))%batch
    order.extend((order*((padding//max(1,len(order)))+1))[:padding])
    return [order[i:i+batch] for i in range(0,len(order),batch)]


def validation_selection(metadata,maximum,seed):
    pools=defaultdict(list);rng=random.Random(seed)
    for i,row in enumerate(metadata):pools[row['label_action']].append(i)
    for rows in pools.values():rng.shuffle(rows)
    selected=[]
    while len(selected)<min(maximum,len(metadata)):
        for action in sorted(pools):
            if pools[action] and len(selected)<maximum:selected.append(pools[action].pop())
    return selected


def freegen_cases():
    """Frozen current-rule development cases; never included in official training data."""
    import copy
    from training.buyer_diagnostics import replace_persona
    from training.buyer_operations_v2_1_data import cash_pair,contexts
    from training.buyer_operations_v2_data import diagnostic_case
    group=20265155
    cases=[]
    for vi,variant in enumerate(('equal','above_one','wide','zero')):
        persona=(vi%3)+1;text=(ROOT/f'training/persona_versions/buyer_official_v1/persona_{persona}.md').read_text().strip()
        for channel in ('board','offer'):
            selected=[]
            for row in contexts(cash_pair(group,0,channel,variant)):
                if row['history_mode']!='fresh':continue
                case=diagnostic_case(row)
                case['origin'].update({k:row[k] for k in ('cash_variant','history_mode','economic_id')})
                selected.append(case)
            if len(selected)!=2:raise ValueError('Missing frozen current-rule pair')
            for c in selected:
                c=copy.deepcopy(c);c['messages']=replace_persona(c['messages'],text,'recorded');c['persona_index']=persona;c['evaluation_rank']=(vi+(0 if channel=='board' else 1))%2;cases.append(c)
    return cases


def official_cases(cfg,split='validation',allowed_ids=None):
    """A small true held-out official subset, independent of the synthetic safety probes."""
    rows=[json.loads(line) for line in (ROOT/cfg['data_dir']/f'{split}.jsonl').read_text().splitlines()]
    pools=defaultdict(list)
    for row in rows:
        if allowed_ids is None or row['id'] in allowed_ids:pools[row['label_action']].append(row)
    rng=random.Random(cfg['seed']);selected=[];used_groups=Counter()
    for action,pool in sorted(pools.items()):
        rng.shuffle(pool)
        for _ in range(min(2 if split=='validation' else 1,len(pool))):
            row=min(pool,key=lambda r:used_groups[r['group']]);pool.remove(row);used_groups[row['group']]+=1
            selected.append({'name':('official_' if split=='validation' else 'seen_')+row['id'],
                'kind':'official_heldout' if split=='validation' else 'official_train_seen','source_sample_id':row['id'],
                'group':row['group'],'messages':row['messages'],'tools':row['tools'],
                'observation':row['observation'],'situation':row['situation'],
                'expected_tool':row['label_action']})
    return selected


def official_snapshot_review(case,tool,args):
    from training.official_buyer_data import schema_error,cents,ALLOWED
    errors=[];obs=case['observation'];sit=case['situation'];trade=None
    if not tool or tool not in ALLOWED:return {'valid_snapshot':False,'errors':['no_supported_tool'],'positive_trade':False}
    error=schema_error(tool,args,case['tools'])
    if error:return {'valid_snapshot':False,'errors':[error],'positive_trade':False}
    if tool=='buy_listed':
        trade=next((b for b in obs['board'] if (b['seller'],b['product_id'])==(args.get('seller_id'),args.get('product_id')) and b['in_stock']>=1),None)
        qty=args.get('qty',1)
        if not trade:errors.append('not_current_board')
    elif tool=='accept_offer':
        trade=next((o for o in obs.get('offers_to_you',[]) if o['offer_id']==args.get('offer_id') and o['open_until_round']>=obs['round']),None)
        qty=trade['qty'] if trade else 0
        if not trade:errors.append('not_current_offer')
    if trade:
        pid=trade['product_id'];price=cents(trade['price'])*qty;value=cents(obs['you']['utility_of_next_unit'].get(pid,0))
        if qty!=1:errors.append('multiunit_trade')
        if obs['you']['units_bought'].get(pid,0):errors.append('already_owned_trade')
        if price>cents(obs['you']['cash']):errors.append('unaffordable')
        if value<=price:errors.append('nonpositive_visible_trade')
    if tool=='send_message' and args.get('to') not in sit['buyers']+sit['sellers']:errors.append('unknown_recipient')
    if tool=='post_review' and not any(t.get('trade_id',t.get('id'))==args.get('trade_id') for t in obs.get('your_unreviewed_trades',[])):
        errors.append('review_not_in_unreviewed_list')
    return {'valid_snapshot':not errors,'errors':errors,'positive_trade':bool(trade) and not errors}


def probe_summary(rows):
    pairs=defaultdict(dict);valid=positive=waits=0
    for r in rows:
        valid+=r['valid_action'];positive+=r['positive'] and r['captured'];waits+=not r['positive'] and r['correct_negative_wait']
        pairs[r['pair_id']][r['positive']]=r
    if len(rows)!=16 or len(pairs)!=8 or any(set(p)!={True,False} for p in pairs.values()):raise ValueError('Incomplete native behavior check')
    both=sum(p[True]['captured'] and p[False]['correct_negative_wait'] for p in pairs.values())
    return {'calls':16,'valid_actions':valid,'positive_captured':positive,'positive_cases':8,'negative_correct_waits':waits,'both_correct':both,
            'unaffordable':sum('unaffordable' in r['errors'] for r in rows),'no_tool':sum(not r['tool'] for r in rows),
            'truncated':sum(r['truncated'] for r in rows),'generated_tokens':sum(r['generated_tokens'] for r in rows)}


def catastrophic(base,current):
    return current['positive_captured']<=base['positive_captured']-4 or current['valid_actions']<=base['valid_actions']-4


def preflight_training_mode(model):
    """HF loads in eval mode; checkpointing only runs on training decoder layers."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    model.train()
    layers=[m for m in model.modules() if isinstance(m,GradientCheckpointingLayer)]
    if not layers or not all(m.training and m.gradient_checkpointing
                            and hasattr(m,'_gradient_checkpointing_func') for m in layers):
        raise RuntimeError('Memory preflight requires active gradient checkpointing in training mode')
    return len(layers)


def generation_stop_ids(model,tokenizer):
    """Include chat end-of-turn; Qwen's model EOS alone is end-of-text."""
    values=getattr(model.generation_config,'eos_token_id',None)
    values=[] if values is None else ([values] if isinstance(values,int) else list(values))
    if tokenizer.eos_token_id is None:raise ValueError('Missing tokenizer end-of-turn EOS')
    return sorted(set(values+[tokenizer.eos_token_id]))


def worker(session):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from tqdm import tqdm
    from training.buyer_operations_train import adapter_model,completion_losses,save_adapter
    from training.reasoning_checkpoint import save_state,load_weights,load_optimizer,restore_rng
    session=Path(session);cfg=json.loads((session/'train_config.json').read_text())
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',local);torch.cuda.set_device(local)
    if world!=2:raise ValueError('Official baseline requires the prepared two-rank plan')
    torch.cuda.set_per_process_memory_fraction(cfg['devices'][local]['fraction'],local)
    (session/f'hardware_rank{rank}.json').write_text(json.dumps({'rank':rank,'physical_gpu':cfg['devices'][local]['index'],
        'name':torch.cuda.get_device_name(local),'total_gib':torch.cuda.get_device_properties(local).total_memory/2**30}))
    dist.init_process_group('nccl',device_id=device)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    train,val,manifest=check_data(cfg);batches=batch_indices(len(train),cfg['effective_batch_size'],cfg['seed'])
    if len(batches)!=cfg['max_steps']:raise ValueError('Full-epoch schedule changed')
    if rank==0:print('Loading pinned bf16 base and NEW LoRA; startup ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Base weights incomplete')
    base.config.use_cache=False;model=adapter_model(base,cfg)
    if cfg.get('resume_checkpoint'):load_weights(model,cfg['resume_checkpoint'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    def rate(step):
        if step<cfg['warmup_steps']:return (step+1)/cfg['warmup_steps']
        return .5*(1+math.cos(math.pi*min(1,(step-cfg['warmup_steps'])/max(1,cfg['max_steps']-cfg['warmup_steps']))))
    start=cfg.get('start_step',0)
    if not cfg.get('evaluation_only'):
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
        if cfg.get('resume_checkpoint'):load_optimizer(optimizer,scheduler,cfg['resume_checkpoint'],device,start)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    stop_ids=generation_stop_ids(model,tokenizer)
    if rank==0:print(f'Native generation stop IDs: {stop_ids}; includes chat end-of-turn {tokenizer.eos_token_id}.',flush=True)
    frozen=json.loads((session/'native_cases.json').read_text());official_frozen=json.loads((session/'official_native_cases.json').read_text());seen_frozen=json.loads((session/'train_native_cases.json').read_text());validation=validation_selection(val.metadata,cfg['validation_max'],cfg['seed'])
    action_names=sorted({r['label_action'] for r in train.metadata});metrics=(session/f'metrics_rank{rank}.jsonl').open('a')
    traces=(session/f'samples_rank{rank}.jsonl').open('w');checkpoint_steps=set(cfg['checkpoint_steps'])
    best_records=[]
    if cfg.get('resume_checkpoint'):
        old_scores=Path(cfg['resume_checkpoint']).parent.parent/'checkpoint_scores.json'
        if old_scores.exists():best_records=[r for r in json.loads(old_scores.read_text()) if r['step']<=start]
    baseline=cfg.get('baseline_behavior')
    def evaluate(step):
        from training.buyer_hf_check import parse_raw
        from training.buyer_diagnostics import evaluate as review
        from training.diagnostic_scores import ACTION_ERRORS,POLICY_ERRORS
        model.eval();losses=[];by_action=defaultdict(list)
        with torch.inference_mode():
            for i in tqdm(validation[rank::world],desc=f'Official validation step{step}',disable=rank!=0,unit='examples'):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,val[i],device)['completion'].item()
                losses.append(loss);by_action[val.metadata[i]['label_action']].append(loss)
            generated=[]
            jobs=[c for i,c in enumerate(frozen+official_frozen+seen_frozen) if c.get('evaluation_rank',i%world)==rank]
            for c in tqdm(jobs,desc=f'Native HF decisions step{step}',disable=rank!=0,unit='calls'):
                prompt=tokenizer.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(**ids,max_new_tokens=cfg['probe_max_tokens'],do_sample=False,use_cache=True,return_dict_in_generate=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=stop_ids)
                tail=out[0,ids.input_ids.shape[1]:];text=tokenizer.decode(tail,skip_special_tokens=False)
                parsed=parse_raw(tokenizer,text,c['tools'])
                if c.get('kind') in ('official_heldout','official_train_seen'):
                    assessed=official_snapshot_review(c,parsed['tool'],parsed['arguments'])
                    generated.append({'gpu_rank':rank,'kind':c['kind'],'case':c['name'],'group':c['group'],
                        'tool':parsed['tool'],'arguments':parsed['arguments'],'reference_tool':c['expected_tool'],
                        'reference_action_match':parsed['tool']==c['expected_tool'],'generated_tokens':len(tail),
                        'truncated':len(tail)>=cfg['probe_max_tokens'] and int(tail[-1]) not in stop_ids,
                        'text':text,**assessed})
                    continue
                checked=review(c,parsed['tool'],parsed['arguments']);errors=checked['errors']
                positive=c['expected_tool']!='wait';truncated=len(tail)>=cfg['probe_max_tokens'] and int(tail[-1]) not in stop_ids
                valid=not truncated and bool(parsed['tool']) and not bool(set(errors)&(ACTION_ERRORS|POLICY_ERRORS))
                generated.append({'gpu_rank':rank,'case':c['name'],'pair_id':c['origin']['pair_id'],'persona_index':c['persona_index'],'positive':positive,
                    'tool':parsed['tool'],'arguments':parsed['arguments'],'errors':errors,'valid_action':valid,
                    'captured':positive and valid and parsed['tool'] in ('buy_listed','accept_offer'),
                    'correct_negative_wait':not positive and valid and parsed['tool']=='wait','truncated':truncated,'generated_tokens':len(tail),'text':text})
        payload={'losses':losses,'by_action':dict(by_action),'generated':generated};all_parts=[None]*world;dist.all_gather_object(all_parts,payload)
        merged=[r for p in all_parts for r in p['generated']];summary=probe_summary([r for r in merged if not r.get('kind','').startswith('official_')])
        official=[r for r in merged if r.get('kind')=='official_heldout']
        official_summary={'calls':len(official),'snapshot_constraints_passed':sum(r['valid_snapshot'] and not r['truncated'] for r in official),
            'positive_visible_trades':sum(r['positive_trade'] and not r['truncated'] for r in official),
            'reference_action_match':sum(r['reference_action_match'] for r in official),
            'note':'Official held-out free-generation. Snapshot constraints and teacher action agreement are not engine replay or market surplus; alternative valid actions are allowed.'}
        seen_rows=[r for r in merged if r.get('kind')=='official_train_seen']
        seen_summary={'calls':len(seen_rows),'snapshot_constraints_passed':sum(r['valid_snapshot'] and not r['truncated'] for r in seen_rows),
            'reference_action_match':sum(r['reference_action_match'] for r in seen_rows),
            'note':'Official training subset selected from the first checkpoint batch prefix. Unseen by weights at step0; all have been trained by the first checkpoint.'}
        nll=[x for p in all_parts for x in p['losses']];parts=defaultdict(list)
        for p in all_parts:
            for action,values in p['by_action'].items():parts[action].extend(values)
        stats={'stage':'evaluation','step':step,'completion_loss':sum(nll)/len(nll),'validation_examples':len(nll),
            'by_action':{a:sum(v)/len(v) for a,v in parts.items()},'native_behavior':summary,'official_free_generation':official_summary,'train_free_generation':seen_summary,
            'note':'Native greedy ON decisions on frozen current-rule synthetic DEVELOPMENT probes, not official held-out accuracy or market score.'}
        if rank==0:
            (session/f'native_step_{step:04d}.json').write_text(json.dumps({'summary':stats,'decisions':merged},indent=2)+'\n')
            print(f'Validation step{step}: official NLL={stats["completion_loss"]:.4f}; native positives={summary["positive_captured"]}/8, both={summary["both_correct"]}/8, cash_errors={summary["unaffordable"]}; official heldout={official_summary["snapshot_constraints_passed"]}/{len(official)}, train={seen_summary["snapshot_constraints_passed"]}/{len(seen_rows)}.',flush=True)
        metrics.write(json.dumps(stats)+'\n');metrics.flush();model.train();return stats
    if cfg.get('evaluation_only'):
        plan=cfg['evaluation_plan']
        if not plan or plan[0]!={'step':0,'checkpoint':None}:raise ValueError('Recheck must start from the unchanged original base')
        results=[];began=time.monotonic()
        for number,entry in enumerate(plan,1):
            if entry['checkpoint']:load_weights(model,entry['checkpoint'])
            if rank==0:print(f'Recheck version {number}/{len(plan)}: step{entry["step"]}; no optimizer or training updates.',flush=True)
            results.append(evaluate(entry['step']))
        if rank==0:
            (session/'checkpoint_scores.json').write_text(json.dumps(results,indent=2)+'\n')
            (session/'finished.json').write_text(json.dumps({'status':'evaluation_only_complete','step':plan[-1]['step'],
                'development_candidate':None,'evaluation_seconds':time.monotonic()-began,'training_updates':0,
                'note':'Correct end-of-turn stopping. No automatic selection, deployment or market evaluation.'},indent=2)+'\n')
        traces.close();metrics.close();dist.barrier();dist.destroy_process_group();return
    # Memory preflight includes backward on the longest example, never an optimizer update.
    longest=max(range(len(train)),key=lambda i:len(train[i]['input_ids']));largest_target=max(range(len(train)),key=lambda i:train[i]['target_length'])
    active_layers=preflight_training_mode(model)
    if rank==0:print(f'GPU memory preflight: training mode, {active_layers} checkpointed layers; longest source example forward/backward; ETA estimating.',flush=True)
    saved_rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device))
    for index in sorted({longest,largest_target}):
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,train[index],device)['completion']
        loss.backward();model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(saved_rng[0]);torch.cuda.set_rng_state(saved_rng[1],device);dist.barrier()
    initial=evaluate(start)
    if baseline is None:baseline=initial['native_behavior']
    if cfg.get('resume_checkpoint'):restore_rng(cfg['resume_checkpoint'],rank,device,start)
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False);model.train();started=time.monotonic();status='complete'
    accumulation=cfg['effective_batch_size']//world
    bar=tqdm(range(start+1,cfg['max_steps']+1),initial=start,total=cfg['max_steps'],disable=rank!=0,unit='steps',desc='Official Buyer epoch1')
    for step in bar:
        optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device);counts=Counter()
        for micro,index in enumerate(batches[step-1][rank::world]):
            row=train[index];counts[row['label_action']]+=1
            traces.write(json.dumps({'step':step,'rank':rank,'micro':micro,'id':row['id'],'group':row['group'],'label_action':row['label_action']})+'\n');traces.flush()
            if rank==0:bar.set_description(f'Official Buyer {step}/{cfg["max_steps"]}, micro {micro+1}/{accumulation}')
            with wrapped.no_sync() if micro<accumulation-1 else nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(wrapped,row,device)['completion']
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss/accumulation).backward();loss_sum+=loss.detach()/accumulation
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();dist.all_reduce(loss_sum)
        stats={'stage':'train','step':step,'completion_loss':loss_sum.item()/world,'lr':lr,'elapsed_seconds':time.monotonic()-started,
            'examples':step*cfg['effective_batch_size'],'rank_actions':dict(counts),'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30}
        metrics.write(json.dumps(stats)+'\n');metrics.flush()
        if rank==0:bar.set_postfix(loss=f'{stats["completion_loss"]:.3f}',lr=f'{lr:.2g}',samples=f'{stats["examples"]}/{cfg["max_steps"]*cfg["effective_batch_size"]}')
        if step in checkpoint_steps:
            cp=session/'checkpoints'/f'step_{step:04d}'
            if rank==0:print(f'Saving checkpoint step{step}...',flush=True);save_adapter(model,cp,cfg,step)
            dist.barrier();score=evaluate(step);save_state(cp,model,optimizer,scheduler,step,rank,device);dist.barrier()
            if rank==0:
                from training.decision_ablation import fingerprint
                (cp/'checkpoint_complete.json').write_text(json.dumps({'step':step,'baseline_behavior':baseline,'native_behavior':score['native_behavior'],'adapter_fingerprints':fingerprint(cp)}))
                score['checkpoint_path']=str(cp);best_records.append(score)
                (session/'checkpoint_scores.json').write_text(json.dumps(best_records,indent=2))
            if catastrophic(baseline,score['native_behavior']):
                status='completed_behavior_regression' if step==cfg['max_steps'] else 'early_stopped_behavior_regression'
                if rank==0:print('STOP: native positive capture or valid actions fell by >=4 versus step0; checkpoint kept. No automatic extension.',flush=True)
                break
    traces.close();metrics.close()
    (session/f'finished_rank{rank}.json').write_text(json.dumps({'step':step,'status':status,'rank':rank}))
    if rank==0:
        latest=session/'checkpoints'/f'step_{step:04d}'
        eligible=[r for r in best_records if r['native_behavior']['positive_captured']>=max(1,baseline['positive_captured']-1) and r['native_behavior']['both_correct']>=baseline['both_correct']
                  and r['native_behavior']['unaffordable']<=baseline['unaffordable'] and r['native_behavior']['no_tool']<=baseline['no_tool']]
        best=max(eligible,key=lambda r:(r['native_behavior']['both_correct'],r['native_behavior']['positive_captured'],r['official_free_generation']['snapshot_constraints_passed'],-r['completion_loss'])) if eligible else None
        (session/'finished.json').write_text(json.dumps({'status':status,'step':step,'target_steps':cfg['max_steps'],'training_seconds':time.monotonic()-started,
            'last_checkpoint':str(latest),'development_candidate':best.get('checkpoint_path',str(session/'checkpoints'/f'step_{best["step"]:04d}')) if best else None,
            'baseline_behavior':baseline,'note':'Development selection only; held-out free-generation and later market surplus validation required. No automatic promotion.'},indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);worker(p.parse_args().session)
