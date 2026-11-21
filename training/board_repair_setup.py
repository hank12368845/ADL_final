"""CPU checks for an ON adapter warm start with a NEW optimizer and scheduler."""
import json
from pathlib import Path
from training.decision_ablation import check_fingerprint,fingerprint


def prepare_warm_start(cfg,root):
    root=Path(root);parent=(root/cfg['initial_adapter']).resolve()
    manifest=json.loads((root/cfg['data_dir']/'manifest.json').read_text())
    old=json.loads((parent.parent.parent/'train_config.json').read_text())
    step=json.loads((parent/'training_step.json').read_text())
    if not step['thinking'] or step['step']!=32:raise ValueError('Repair requires the thinking-ON step32 parent')
    for key in ('base_model','revision','rank','alpha','target_modules','thinking'):
        if cfg.get(key)!=old.get(key):raise ValueError(f'Warm-start adapter recipe mismatch: {key}')
    check_fingerprint(parent,manifest['parent_fingerprints'])
    cfg['initial_adapter']=str(parent)
    cfg['initial_adapter_fingerprints']=fingerprint(parent)
    cfg['warm_start']={'parent_step':32,'local_start_step':0,'optimizer':'new AdamW','scheduler':'new warmup/cosine',
                       'note':'Adapter warm start; not optimizer resume. Local step16/32 are additional updates.'}
    return cfg['warm_start']
