"""CPU provenance checks for the single-objective v2.2 ablation; no model loading."""
import hashlib
import json
from pathlib import Path

EXPERIMENT = 'buyer_operations_v2_2_first_token'
RECIPE_KEYS = ('base_model', 'revision', 'data_version', 'data_dir', 'sampling',
               'decision_weight', 'comparison_adapter', 'rank', 'alpha', 'dropout',
               'learning_rate', 'weight_decay', 'max_steps', 'effective_batch_size',
               'warmup_steps', 'eval_every', 'validation_max', 'seed', 'max_length',
               'gradient_checkpointing', 'target_modules')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(adapter):
    return {name: digest(Path(adapter)/name)
            for name in ('adapter_config.json', 'adapter_model.safetensors')}


def check_fingerprint(adapter, expected):
    if set(expected) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Incomplete adapter fingerprints')
    if fingerprint(adapter) != expected:
        raise ValueError(f'Comparison adapter changed: {adapter}')


def audit_training(cfg, root, train):
    """Require the archived recipe, exact data and exact two-rank batch sequence."""
    from training.buyer_operations_v2_1_sampling import batch_manifest
    root = Path(root)
    if cfg.get('experiment') != EXPERIMENT or cfg.get('decision_supervision') != 'first_token':
        raise ValueError('Expected the first-token-only decision ablation')
    control = (root/cfg['control_run']).resolve()
    baseline = json.loads((control/'train_config.json').read_text())
    changed = [k for k in RECIPE_KEYS if cfg.get(k) != baseline.get(k)]
    if changed:
        raise ValueError(f'Fixed recipe changed: {changed}')
    if baseline.get('decision_supervision', 'name_mean') != 'name_mean':
        raise ValueError('Control must use mean tool-name supervision')
    if baseline.get('world_size') != 2 or json.loads((control/'wrapper_result.json').read_text())['exit'] != 0:
        raise ValueError('Expected a completed two-GPU control')
    manifest = root/cfg['data_dir']/'manifest.json'
    if digest(manifest) != baseline['data_manifest_sha256']:
        raise ValueError('Data manifest differs from the archived control')
    plan = batch_manifest(train.metadata, cfg['max_steps'], cfg['seed'], 2)
    if plan != json.loads((control/'batch_plan.json').read_text()):
        raise ValueError('Sample order or rank assignment differs from the control')
    first_ids = {}
    name_lengths = {}
    for row in train:
        positions = row['tool_name_positions']
        first_ids.setdefault(row['label_action'], set()).add(row['input_ids'][row['prompt_length']+positions[0]])
        name_lengths.setdefault(row['label_action'], set()).add(len(positions))
    if set(first_ids) != {'buy_listed', 'accept_offer', 'wait'} or any(len(v) != 1 for v in first_ids.values()):
        raise ValueError('Tool first-token mapping is not stable')
    if len(set(next(iter(v)) for v in first_ids.values())) != 3:
        raise ValueError('First tokens do not distinguish all three target tools')
    actual = {}
    for rank in (0, 1):
        trace = [json.loads(line) for line in (control/f'samples_rank{rank}.jsonl').read_text().splitlines()]
        expected = [(s['step'], m, row['id']) for s in plan['steps']
                    for m, row in enumerate(s['ranks'][rank])]
        if [(r['step'], r['micro'], r['id']) for r in trace] != expected:
            raise ValueError('Archived control trace does not match the fixed sampler')
        actual[str(rank)] = len(trace)
    check_fingerprint(root/cfg['comparison_adapter'], baseline['comparison_fingerprints'])
    checkpoints = {name: fingerprint(control/path) for name, path in (
        ('step_0008', 'checkpoints/step_0008'), ('step_0012', 'checkpoints/step_0012'),
        ('final', 'final_adapter'))}
    return {'experiment': EXPERIMENT, 'control_run': str(control),
            'control_config_sha256': digest(control/'train_config.json'),
            'data_manifest_sha256': digest(manifest),
            'cash_cases_sha256': digest(root/cfg['data_dir']/'cash_probe_cases.json'),
            'batch_plan_sha256': digest(control/'batch_plan.json'),
            'control_checkpoints': checkpoints, 'control_samples_by_rank': actual,
            'first_tool_token_ids': {k: next(iter(v)) for k, v in first_ids.items()},
            'tool_name_token_lengths': {k: sorted(v) for k, v in name_lengths.items()},
            'changed_training_factor': 'auxiliary mean tool-name CE -> first tool-name token CE',
            'note': 'Historical single-run control, not a replicated causal or market-score claim.'}
