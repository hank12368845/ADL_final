"""Completion-only LoRA SFT worker. GPU execution is launched by the user wrapper."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class TokenRows(Dataset):
    """Index records without keeping millions of Python token integers per worker."""
    def __init__(self, path, max_length, require_tool_spans=False):
        self.path = Path(path)
        self.offsets = []
        self.metadata = []
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                n, p = len(row['input_ids']), row['prompt_length']
                if not 0 < p < n <= max_length or row['target_length'] != n-p:
                    raise ValueError(f'Invalid loss span or length: {row["id"]}')
                mask = row.get('tool_name_positions')
                if require_tool_spans or mask is not None:
                    if not isinstance(mask,list) or not mask or any(type(i) is not int or not 0<=i<n-p for i in mask) or mask!=sorted(set(mask)):
                        raise ValueError(f'Invalid tool-name target positions: {row["id"]}')
                self.offsets.append(offset)
                self.metadata.append({k:row.get(k) for k in ('id','pair_id','group','task','family','positive','label_action','history_mode','persona_index',
                                                         'cash_variant','economic_id','curriculum_version')})
        if not self.offsets:
            raise ValueError(f'Empty token data: {path}')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[i])
            return json.loads(stream.readline())


def decision_mode(cfg):
    mode = cfg.get('decision_supervision', 'name_mean')
    if mode not in ('name_mean', 'first_token'):
        raise ValueError(f'Unknown decision supervision: {mode}')
    return mode


def completion_losses(model, row, device, decision_weight=0., decision_supervision='name_mean'):
    """Only materialize logits predicting assistant targets, not prompt vocabulary logits."""
    ids = torch.tensor([row['input_ids']], device=device, dtype=torch.long)
    p = row['prompt_length']
    positions = torch.arange(p-1, ids.shape[1]-1, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=positions, use_cache=False)
    targets = ids[:, p:]
    if output.logits.shape[:2] != targets.shape:
        raise ValueError('Completion loss positions do not align with targets')
    per_token = torch.nn.functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                                  targets.reshape(-1), reduction='none')
    mask = row.get('tool_name_positions')
    if decision_weight<0 or not math.isfinite(decision_weight): raise ValueError('Invalid decision loss weight')
    if decision_weight and not mask: raise ValueError('Decision supervision requires tool-name positions')
    decision_mode({'decision_supervision': decision_supervision})
    completion = per_token.mean()
    tool = per_token[torch.tensor(mask,device=device,dtype=torch.long)].mean() if mask else completion.new_zeros(())
    first = per_token[mask[0]] if mask else completion.new_zeros(())
    decision = first if decision_supervision == 'first_token' else tool
    return {'completion':completion,'tool_name':tool,'first_token':first,'decision':decision,
            'total':completion+decision_weight*decision}


def completion_loss(model, row, device, decision_weight=0., decision_supervision='name_mean'):
    return completion_losses(model,row,device,decision_weight,decision_supervision)['total']


def adapter_model(base, cfg):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'],
                          lora_dropout=cfg['dropout'], target_modules=cfg['target_modules'],
                          task_type='CAUSAL_LM', bias='none', use_dora=False))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or '.layers.' not in n for n in trainable):
        raise ValueError('Only language-layer LoRA parameters may be trainable')
    return model


def validation_indices(data, maximum, seed):
    """Cover task types even for a tiny validation budget; avoid stride aliasing."""
    if data.metadata[0].get('curriculum_version') in ('buyer_operations_v2_1','buyer_operations_reasoning_v1','buyer_board_repair_v1'):
        from training.buyer_operations_v2_1_sampling import validation_selection
        return validation_selection(data.metadata,maximum,seed)
    if data.metadata[0].get('pair_id'):
        if maximum<2 or maximum%2: raise ValueError('Paired validation budget must be positive and even')
        pairs=defaultdict(dict); families=defaultdict(list)
        for i,r in enumerate(data.metadata): pairs[r['pair_id']][r['positive']]=i
        for sides in pairs.values():
            if set(sides)!={True,False}: raise ValueError('Incomplete validation pair')
            families[data.metadata[sides[True]]['family']].append([sides[True],sides[False]])
        rng=random.Random(seed)
        for pool in families.values(): rng.shuffle(pool)
        selected=[]
        while len(selected)<min(maximum,len(data)):
            for family in sorted(families):
                if families[family] and len(selected)<min(maximum,len(data)):
                    selected.extend(families[family].pop())
        return selected
    pools = defaultdict(list)
    for i in range(len(data)):
        pools[data[i]['task']].append(i)
    rng = random.Random(seed)
    for pool in pools.values(): rng.shuffle(pool)
    selected = []
    while len(selected) < min(maximum, len(data)):
        for task in sorted(pools):
            if pools[task] and len(selected) < min(maximum, len(data)):
                selected.append(pools[task].pop())
    return selected


def save_adapter(model, folder, cfg, step):
    from adlmarket.config import check_adapter
    folder = Path(folder)
    model.peft_config['default'].base_model_name_or_path = cfg['base_model']
    model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
    check_adapter('buyer_operations', folder, cfg['base_model'],
                  {'model.layers.': 'model.language_model.layers.'})
    (folder/'training_step.json').write_text(json.dumps({'step': step, 'thinking': cfg.get('thinking',False)}))


def check_data(cfg, root):
    decision_mode(cfg)
    folder = root / cfg['data_dir']
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['revision'] != cfg['revision'] or manifest['thinking'] != cfg.get('thinking',False):
        raise ValueError('Tokenizer/base revision or thinking mode mismatch')
    v2=manifest.get('version') in ('buyer_operations_v2','buyer_operations_v2_1','buyer_operations_reasoning_v1','buyer_board_repair_v1','buyer_id_repair_v1','buyer_symmetric_repair_v1')
    if cfg.get('data_version') and cfg['data_version']!=manifest['version']: raise ValueError('Configured data version mismatch')
    sources=manifest['sources'] if v2 else {'training/buyer_operations_data.py':manifest.get('generator_sha256')}
    for path,expected in sources.items():
        if expected != hashlib.sha256((root/path).read_bytes()).hexdigest():
            raise ValueError('Data generator/supervision changed since preparation; regenerate into a NEW data directory')
    groups = manifest['groups']
    if any(set(groups[a]) & set(groups[b]) for a, b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Train/validation/test market leakage')
    for split in ('train', 'validation', 'test'):
        path = folder/f'{split}.tokens.jsonl'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['splits'][split]['tokens_sha256']:
            raise ValueError(f'Data checksum mismatch: {split}')
    train = TokenRows(folder/'train.tokens.jsonl', cfg['max_length'], v2 or cfg.get('decision_weight',0)>0)
    validation = TokenRows(folder/'validation.tokens.jsonl', cfg['max_length'], v2 or cfg.get('decision_weight',0)>0)
    if v2:
        for split,data in [('train',train),('validation',validation)]:
            expected={manifest['seed']+1009*g for g in groups[split]}
            if any(r['group'] not in expected for r in data.metadata): raise ValueError('Record market leaks across split')
        expected_sampling={'buyer_operations_v2':'paired_4_2_2','buyer_operations_v2_1':'cash_offer_4_2_2',
                           'buyer_operations_reasoning_v1':'reasoning_epoch_pairs',
                           'buyer_board_repair_v1':'board_repair_4_1_3',
                           'buyer_id_repair_v1':'board_repair_4_1_3',
                           'buyer_symmetric_repair_v1':'symmetric_repair_2_2_4'}[manifest['version']]
        if cfg.get('sampling')!=expected_sampling or cfg['effective_batch_size']!=8:
            raise ValueError('V2 requires the paired sampler and global batch 8')
    return train, validation


def sampling_components(cfg):
    if cfg.get('sampling')=='symmetric_repair_2_2_4':
        from training.symmetric_repair_sampling import PairedActionSampler,batch_manifest
    elif cfg.get('sampling')=='board_repair_4_1_3':
        from training.board_repair_sampling import PairedActionSampler,batch_manifest
    elif cfg.get('sampling')=='reasoning_epoch_pairs':
        from training.buyer_reasoning_sampling import PairedActionSampler,batch_manifest
    elif cfg.get('sampling')=='cash_offer_4_2_2':
        from training.buyer_operations_v2_1_sampling import PairedActionSampler,batch_manifest
    else:
        from training.buyer_operations_sampling import PairedActionSampler,batch_manifest
    return PairedActionSampler,batch_manifest


def worker(session):
    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    root = Path(__file__).resolve().parents[1]
    session = Path(session)
    cfg = json.loads((session/'train_config.json').read_text())
    rank, local_rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.cuda.set_per_process_memory_fraction(cfg['devices'][local_rank]['fraction'], local_rank)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0:
        print('Loading token data, then bf16 Qwen3.5 text model; startup ETA estimating.', flush=True)
    train, validation = check_data(cfg, root)
    (session/f'hardware_rank{rank}.json').write_text(json.dumps({
        'rank': rank, 'physical_index': cfg['devices'][local_rank]['index'],
        'name': torch.cuda.get_device_name(device),
        'total_gib': torch.cuda.get_device_properties(device).total_memory/1024**3}))
    base, info = AutoModelForCausalLM.from_pretrained(
        cfg['base_model'], revision=cfg['revision'], local_files_only=True,
        dtype=torch.bfloat16, device_map={'': device}, attn_implementation='sdpa',
        output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError(f'Base weights not loaded completely: {info}')
    base.config.use_cache = False
    model = adapter_model(base, cfg)
    start_step = cfg.get('start_step',0)
    if cfg.get('resume_checkpoint'):
        from training.reasoning_checkpoint import load_weights
        load_weights(model, cfg['resume_checkpoint'])
    elif cfg.get('initial_adapter'):
        from training.reasoning_checkpoint import load_weights
        from training.decision_ablation import check_fingerprint
        check_fingerprint(cfg['initial_adapter'],cfg['initial_adapter_fingerprints'])
        load_weights(model,cfg['initial_adapter'])
        if rank==0:print('Loaded parent LoRA; optimizer and scheduler start fresh at local step0.',flush=True)
    model.train()
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
        (session/'loading_info.json').write_text(json.dumps(info, indent=2,
                                                          default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    def schedule(step):
        if step < cfg['warmup_steps']:
            return (step+1)/max(1, cfg['warmup_steps'])
        x = (step-cfg['warmup_steps'])/max(1, cfg.get('schedule_steps',cfg['max_steps'])-cfg['warmup_steps'])
        return .5*(1+math.cos(math.pi*min(1, x)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if cfg.get('resume_checkpoint'):
        from training.reasoning_checkpoint import load_optimizer
        load_optimizer(optimizer,scheduler,cfg['resume_checkpoint'],device,start_step)
    accumulation = cfg['effective_batch_size']//world
    if cfg.get('sampling') in ('paired_4_2_2','cash_offer_4_2_2','reasoning_epoch_pairs','board_repair_4_1_3','symmetric_repair_2_2_4'):
        PairedActionSampler,_=sampling_components(cfg)
        sampler=PairedActionSampler(train.metadata,cfg['max_steps'],cfg['seed'],rank,world)
        if cfg.get('sampling') in ('reasoning_epoch_pairs','board_repair_4_1_3','symmetric_repair_2_2_4'): sampler.start_step=start_step
    else:
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed'], drop_last=True)
    loader = DataLoader(train, batch_size=None, sampler=sampler, num_workers=0)
    epoch, iterator = 0, iter(loader)
    metrics = (session/'metrics.jsonl').open('a') if rank == 0 else None
    trace = (session/f'samples_rank{rank}.jsonl').open('w')
    decision_weight=cfg.get('decision_weight',0.)
    supervision=decision_mode(cfg)

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        indices = validation_indices(validation, cfg['validation_max'], cfg['seed'])[rank::world]
        total = torch.zeros(6, device=device, dtype=torch.float64)
        by_action = torch.zeros(3,5,device=device,dtype=torch.float64)
        for i in tqdm(indices, desc=f'Validation step {step}', disable=rank != 0, unit='examples'):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                parts = completion_losses(model, validation[i], device, decision_weight, supervision)
            if not torch.isfinite(parts['total']):
                raise ValueError('Nonfinite validation loss')
            total += torch.tensor([float(parts[k]) for k in ('completion','tool_name','total','first_token','decision')]+[1], device=device)
            action=validation[i].get('label_action')
            if action:
                j=('buy_listed','accept_offer','wait').index(action)
                by_action[j] += torch.tensor([float(parts[k]) for k in ('completion','tool_name','first_token','decision')]+[1],device=device)
        if world > 1:
            dist.all_reduce(total)
            dist.all_reduce(by_action)
        values = (total[:5]/total[5]).tolist()
        if rank == 0:
            row = {'stage': 'validation', 'step': step, 'completion_loss': values[0],
                   'tool_name_loss':values[1], 'objective_loss':values[2], 'first_token_loss':values[3],
                   'decision_loss':values[4], 'decision_supervision':supervision, 'examples': int(total[5])}
            row['by_action']={action:{**{key:by_action[i,j].item()/by_action[i,4].item()
                 for j,key in enumerate(('completion_loss','tool_name_loss','first_token_loss','decision_loss'))},
                 'examples':int(by_action[i,4])}
                 for i,action in enumerate(('buy_listed','accept_offer','wait')) if by_action[i,4]>0}
            metrics.write(json.dumps(row)+'\n'); metrics.flush()
            print(f'Validation step={step}: completion={values[0]:.5f}, tool_name={values[1]:.5f}, '
                  f'first_token={values[3]:.5f}, decision={values[4]:.5f} ({supervision}), '
                  f'objective={values[2]:.5f}; behavior evaluation required.', flush=True)
        model.train()

    # Exercise the longest training record before committing to a long pilot.
    if rank == 0:
        print('GPU memory / backward preflight: longest training sequence; ETA estimating.', flush=True)
    longest = max(range(len(train)), key=lambda i: len(train[i]['input_ids']))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        preflight_loss = completion_loss(model, train[longest], device, decision_weight, supervision)
    preflight_loss.backward()
    model.zero_grad(set_to_none=True)
    del preflight_loss
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng, device)
    if world > 1: dist.barrier()
    if rank == 0: print('GPU memory / backward preflight passed (optimizer state allocates at step 1).', flush=True)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    evaluate(start_step)
    if cfg.get('resume_checkpoint'):
        from training.reasoning_checkpoint import restore_rng
        restore_rng(cfg['resume_checkpoint'],rank,device,start_step)
    progress = tqdm(range(start_step+1, cfg['max_steps']+1), total=cfg['max_steps'], initial=start_step,
                    desc='Buyer LoRA optimizer steps', disable=rank != 0, unit='steps')
    started = time.monotonic()
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros(5, device=device)
        action_counts = torch.zeros(3,device=device,dtype=torch.long)
        for micro in range(accumulation):
            try:
                row = next(iterator)
            except StopIteration:
                epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); row = next(iterator)
            trace.write(json.dumps({'step':step,'micro':micro,'rank':rank,
                                    **{k:row.get(k) for k in ('id','pair_id','label_action','history_mode','group','family','cash_variant','persona_index')}})+'\n');trace.flush()
            if row.get('label_action'):
                action_counts[('buy_listed','accept_offer','wait').index(row['label_action'])]+=1
            if rank == 0:
                progress.set_description(f'Buyer LoRA step {step}, micro {micro+1}/{accumulation}')
            context = wrapped.no_sync() if world > 1 and micro < accumulation-1 else nullcontext()
            with context, torch.autocast('cuda', dtype=torch.bfloat16):
                parts = completion_losses(wrapped, row, device, decision_weight, supervision)
                loss = parts['total']
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite training loss: {row["id"]}')
                (loss/accumulation).backward()
                step_loss += torch.stack([parts[k].detach() for k in
                                          ('completion','tool_name','total','first_token','decision')])/accumulation
        if world>1: dist.all_reduce(action_counts)
        quotas={'reasoning_epoch_pairs':[2,2,4],'board_repair_4_1_3':[4,1,3],'symmetric_repair_2_2_4':[2,2,4]}.get(cfg.get('sampling'),[4,2,2])
        if cfg.get('sampling') in ('paired_4_2_2','cash_offer_4_2_2','reasoning_epoch_pairs','board_repair_4_1_3','symmetric_repair_2_2_4') and action_counts.tolist()!=quotas:
            raise ValueError(f'Actual global batch quota failed: {action_counts.tolist()}')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        applied_lr=optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        if world > 1:
            dist.all_reduce(step_loss); step_loss /= world
        if rank == 0:
            record = {'stage': 'train', 'step': step, 'loss': step_loss[2].item(),
                      'completion_loss':step_loss[0].item(),'tool_name_loss':step_loss[1].item(),
                      'first_token_loss':step_loss[3].item(),'decision_loss':step_loss[4].item(),
                      'decision_supervision':supervision,
                      'actions':dict(zip(('buy_listed','accept_offer','wait'),action_counts.tolist())),
                      'lr': applied_lr, 'next_lr':scheduler.get_last_lr()[0], 'elapsed_seconds': time.monotonic()-started,
                      'examples': step*cfg['effective_batch_size'],
                      'coverage_epochs':step*cfg['effective_batch_size']/len(train),
                      'peak_gpu_gib': torch.cuda.max_memory_allocated(device)/1024**3}
            metrics.write(json.dumps(record)+'\n'); metrics.flush()
            progress.set_postfix(loss=f'{record["loss"]:.3f}', lr=f'{record["lr"]:.2g}')
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            checkpoint=session/'checkpoints'/f'step_{step:04d}'
            if rank == 0:
                print(f'Saving step {step}: adapter'+(' + optimizer/scheduler/RNG' if cfg.get('thinking') else '')+'...',flush=True)
                save_adapter(model, checkpoint, cfg, step)
            if world > 1:
                dist.barrier()
            if cfg.get('thinking'):
                from training.reasoning_checkpoint import save_state
                save_state(checkpoint,model,optimizer,scheduler,step,rank,device)
                if world>1: dist.barrier()
            if rank==0: print(f'Checkpoint complete: {checkpoint}',flush=True)
            evaluate(step)
    trace.close()
    (session/f'finished_rank{rank}.json').write_text(json.dumps({'rank':rank,'steps':cfg['max_steps'],
        'samples':(cfg['max_steps']-start_step)*accumulation,'start_step':start_step,
        'physical_gpu':cfg['devices'][local_rank]['index'],
        'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/1024**3}))
    if rank == 0:
        save_adapter(model, session/'final_adapter', cfg, cfg['max_steps'])
        metrics.close()
        (session/'finished.json').write_text(json.dumps({'steps': cfg['max_steps'], 'world_size': world,
                                                       'start_step':start_step,'thinking':cfg.get('thinking',False),
                                                       'training_seconds': time.monotonic()-started}))
        print(f'Completed pilot; candidate adapter: {session}/final_adapter. Behavior evaluation required.', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    worker(parser.parse_args().session)
