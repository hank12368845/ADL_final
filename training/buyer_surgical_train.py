"""Twenty-step faithful replay/repair continuation, unchanged native thinking style."""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

from training.prepare_buyer_surgical import ROOT,audit_folder,sha


def load_data(folder):
    from training.buyer_operations_train import TokenRows
    folder=Path(folder);m=audit_folder(folder)
    train=TokenRows(folder/'train.tokens.jsonl',m['max_input_target_tokens'])
    return train,json.loads((folder/'batch_plan.json').read_text()),m


def activation_context(row,cfg):
    import torch
    if len(row['input_ids'])>=cfg['activation_offload_from_tokens']:
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return nullcontext()


def install_chunked_mlp(model,chunk_tokens):
    """Checkpoint tokenwise MLP chunks; attention still receives the full sequence."""
    import types
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    if type(chunk_tokens)!=int or chunk_tokens<1:raise ValueError('Invalid MLP chunk size')
    count=0
    for module in model.modules():
        if not isinstance(module,Qwen3_5MLP):continue
        if hasattr(module,'_surgical_original_forward'):raise ValueError('MLP already patched')
        if any(isinstance(m,torch.nn.Dropout) and m.p for m in module.modules()):
            raise ValueError('Chunked MLP requires the frozen zero-dropout recipe')
        module._surgical_original_forward=module.forward
        module._surgical_chunk_tokens=chunk_tokens
        def forward(self,x):
            original=self._surgical_original_forward
            if not self.training or not torch.is_grad_enabled() or x.shape[-2]<=self._surgical_chunk_tokens:
                return original(x)
            return torch.cat([checkpoint(original,part,use_reentrant=False,preserve_rng_state=True)
                              for part in x.split(self._surgical_chunk_tokens,dim=-2)],dim=-2)
        module.forward=types.MethodType(forward,module);count+=1
    if not count:raise ValueError('No Qwen3.5 MLP found; refuse silent memory fallback')
    return count


def check_run(folder):
    cfg=json.loads((Path(folder)/'train_config.json').read_text())
    for path,h in cfg['sources'].items():
        if sha(path)!=h:raise ValueError(f'Run source changed:{path}')
    if sha(Path(cfg['data_dir'])/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Data manifest changed')
    from importlib.metadata import version
    for name,v in cfg['versions'].items():
        if version(name)!=v:raise ValueError(f'Package changed:{name}')
    audit_folder(cfg['data_dir'])
    if (cfg['max_steps'],cfg['effective_batch_size'],cfg['world_size'],cfg['learning_rate'],cfg['warmup_steps'])!=(20,4,2,5e-6,2):
        raise ValueError('Surgical recipe changed')
    return cfg


def rate(step):
    if step<2:return (step+1)/2
    return .5*(1+math.cos(math.pi*min(1.,(step-2)/18)))


def worker(folder,preflight=False):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM
    from training.buyer_operations_train import adapter_model,completion_losses,save_adapter
    from training.reasoning_checkpoint import load_weights,save_state,load_optimizer,restore_rng
    from training.official_buyer_train import preflight_training_mode
    from training.decision_ablation import fingerprint
    folder=Path(folder);cfg=check_run(folder);train,batches,m=load_data(cfg['data_dir'])
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Exactly two DDP workers required')
    torch.cuda.set_device(rank);torch.cuda.set_per_process_memory_fraction(cfg['devices'][rank]['fraction'],rank)
    dist.init_process_group('nccl',device_id=device);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    print(f'[rank{rank}] Loading fixed base+step84; full history; saved activations offload from{cfg["activation_offload_from_tokens"]}tokens; ETA estimating.',flush=True)
    base,info=AutoModelForCausalLM.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):raise ValueError('Incomplete base load')
    base.config.use_cache=False;model=adapter_model(base,cfg)
    resume=cfg.get('start_step',0)
    load_weights(model,str(folder/'checkpoints'/f'step_{resume:04d}') if resume else cfg['parent_adapter'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    layers=preflight_training_mode(model)
    mlps=install_chunked_mlp(model,cfg['mlp_chunk_tokens'])
    print(f'[rank{rank}] Backward preflight:{layers} checkpointed layers,{mlps} MLPs chunked at{cfg["mlp_chunk_tokens"]}tokens; longest{m["max_input_target_tokens"]}; ETA estimating.',flush=True)
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device))
    for index in {max(range(len(train)),key=lambda i:len(train[i]['input_ids'])),max(range(len(train)),key=lambda i:train[i]['target_length'])}:
        with activation_context(train[index],cfg):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(model,train[index],device)['completion']
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
            loss.backward()
        model.zero_grad(set_to_none=True);del loss
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device);dist.barrier()
    (folder/f'preflight_rank{rank}.json').write_text(json.dumps(dict(rank=rank,physical_gpu=cfg['devices'][rank]['index'],
        name=torch.cuda.get_device_name(rank),checkpointed_layers=layers,chunked_mlps=mlps,mlp_chunk_tokens=cfg['mlp_chunk_tokens'],
        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,activation_offload_from_tokens=cfg['activation_offload_from_tokens'])))
    if preflight:
        dist.destroy_process_group();return
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    if resume:
        cp=folder/'checkpoints'/f'step_{resume:04d}'
        load_optimizer(optimizer,scheduler,cp,device,resume);restore_rng(cp,rank,device,resume)
    wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False);began=time.monotonic()
    with (folder/f'metrics_rank{rank}.jsonl').open('a') as metrics,(folder/f'samples_rank{rank}.jsonl').open('a') as trace:
        for step in range(resume+1,21):
            optimizer.zero_grad(set_to_none=True);mean_loss=torch.zeros((),device=device)
            for micro,i in enumerate(batches[step-1][rank::2]):
                row=train[i];trace.write(json.dumps(dict(step=step,rank=rank,id=row['id'],family=row['family'],label_action=row['label_action']))+'\n');trace.flush()
                if rank==0:print(f'Train{step}/20 micro{micro+1}/2 | {step*4}/80 scheduled exposures | elapsed{time.monotonic()-began:.0f}s',flush=True)
                with (wrapped.no_sync() if micro<1 else nullcontext()),activation_context(row,cfg):
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=completion_losses(wrapped,row,device)['completion']
                    if not torch.isfinite(loss):raise ValueError('Nonfinite SFT loss')
                    (loss/2).backward();mean_loss+=loss.detach()/2
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();dist.all_reduce(mean_loss)
            elapsed=time.monotonic()-began;speed=(step-resume)/elapsed
            record=dict(step=step,loss=mean_loss.item()/2,lr=lr,elapsed_seconds=elapsed,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            metrics.write(json.dumps(record)+'\n');metrics.flush()
            if rank==0:
                bar='#'*step+'-'*(20-step)
                print(f'[{bar}] Train{step}/20 | loss{record["loss"]:.4f} | LR{lr:.2g} | elapsed{elapsed:.0f}s | {speed:.3f}steps/s | ETA{(20-step)/speed:.0f}s',flush=True)
            if step in (10,20):
                cp=folder/'checkpoints'/f'step_{step:04d}'
                if rank==0:save_adapter(model,cp,cfg,step)
                dist.barrier();save_state(cp,model,optimizer,scheduler,step,rank,device);dist.barrier()
                if rank==0:
                    fp=fingerprint(cp)
                    if fp==cfg['parent_fingerprints']:raise ValueError('Adapter did not update')
                    (cp/'checkpoint_complete.json').write_text(json.dumps(dict(step=step,adapter_fingerprints=fp)))
                    print(f'Saved complete checkpoint:{cp}; ordinary CE only,behavior evaluation follows after training.',flush=True)
    if rank==0:(folder/'training_finished.json').write_text(json.dumps(dict(step=20,examples=80,replay_exposures=56,repair_exposures=24,resumed_from=resume,training_seconds=time.monotonic()-began)))
    dist.barrier();dist.destroy_process_group()
