"""Exact optimizer/scheduler and per-rank RNG continuation for reasoning pilots."""
import json
from pathlib import Path
import random
import hashlib

FIXED = ('base_model','revision','data_version','data_dir','sampling','thinking','decision_weight',
         'decision_supervision','rank','alpha','dropout','learning_rate','weight_decay','warmup_steps',
         'schedule_steps','effective_batch_size','seed','max_length','gradient_checkpointing',
         'target_modules','world_size','data_manifest_sha256')


def validate_resume(cfg, checkpoint, plan):
    checkpoint = Path(checkpoint).resolve(); parent = checkpoint.parent.parent
    old = json.loads((parent/'train_config.json').read_text())
    root=Path(__file__).resolve().parents[1]
    for name,expected in old.get('training_sources',{}).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Resume training source changed: {name}')
    changed = [k for k in FIXED if cfg.get(k) != old.get(k)]
    if changed: raise ValueError(f'Resume recipe differs: {changed}')
    step = json.loads((checkpoint/'training_step.json').read_text())['step']
    if not 0 < step < cfg['max_steps']: raise ValueError('Resume checkpoint must precede target step')
    old_plan = json.loads((parent/'batch_plan.json').read_text())
    if old_plan['steps'][:step] != plan['steps'][:step]: raise ValueError('Resume data prefix changed')
    required = ['optimizer.pt']+[f'rng_rank{r}.pt' for r in range(cfg['world_size'])]
    if any(not (checkpoint/p).is_file() for p in required): raise ValueError('No complete optimizer/RNG resume state')
    return step


def save_state(folder, model, optimizer, scheduler, step, rank, device):
    import torch
    folder = Path(folder)
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, folder/'optimizer.pt')
    # The caller synchronizes folder/adapter creation before all ranks write.
    torch.save({'step': step, 'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device),
                'python': random.getstate()}, folder/f'rng_rank{rank}.pt')


def load_weights(model, folder):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    result = set_peft_model_state_dict(model, load_file(str(Path(folder)/'adapter_model.safetensors')))
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
        raise ValueError('Resume adapter keys do not match')


def load_optimizer(optimizer, scheduler, folder, device, expected_step):
    import torch
    state = torch.load(Path(folder)/'optimizer.pt', map_location='cpu', weights_only=True)
    if state['step'] != expected_step: raise ValueError('Optimizer checkpoint step mismatch')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])


def restore_rng(folder, rank, device, expected_step):
    import torch
    state = torch.load(Path(folder)/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
    if state['step'] != expected_step: raise ValueError('RNG checkpoint step mismatch')
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'], device)
    random.setstate(state['python'])
