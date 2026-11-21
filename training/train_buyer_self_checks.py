#!/usr/bin/env python3
"""User-launched, bounded scratchpad learnability pilot; includes all evaluations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ('training/train_buyer_self_checks.py', 'training/buyer_self_checks_train.py',
    'training/buyer_self_checks_data.py', 'training/buyer_self_checks.py',
    'training/buyer_operations_train.py', 'training/official_buyer_train.py',
    'training/official_buyer_data.py', 'training/step84_repair_plan.py',
    'training/buyer_hf_check.py', 'training/reasoning_checkpoint.py', 'training/buyer_latest_brakes_train.py')


def prepare(cfg):
    from importlib.metadata import version
    from training.buyer_self_checks_data import check_data, SEED
    from training.buyer_latest_brakes_data import sha
    cfg = dict(cfg); _, _, batches, m = check_data(cfg)
    expected = dict(max_steps=12, warmup_steps=2, checkpoint_steps=[6, 12], effective_batch_size=8,
        learning_rate=1e-5, max_length=12288, seed=SEED, thinking=True)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('Bounded self-check recipe changed')
    old = json.loads((Path(m['parent_adapter']).parent.parent/'train_config.json').read_text())
    for key in ('base_model', 'revision', 'rank', 'alpha', 'dropout', 'target_modules'):
        cfg[key] = old[key]
    if cfg['rank'] > 32:
        raise ValueError('Adapter rank exceeds submission limit')
    cfg.update(parent_adapter=m['parent_adapter'], parent_fingerprints=m['parent_fingerprints'], world_size=2,
        objective='equal semantic-field CE (target/distractor balanced) + 0.25 native assistant CE; no tool-name multiplier',
        data_manifest_sha256=sha(ROOT/cfg['data_dir']/'manifest.json'),
        training_sources={name: sha(ROOT/name) for name in SOURCES},
        versions={name: version(name) for name in ('torch', 'transformers', 'peft', 'safetensors', 'vllm')})
    return cfg, m, batches


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--gpus', default='0,1')
    p.add_argument('--cpu-check', action='store_true'); p.add_argument('--output-dir', type=Path, default=ROOT/'training/runs')
    p.add_argument('--config', type=Path, default=ROOT/'training/configs/buyer_self_checks_v1.yaml')
    a = p.parse_args()
    print('CPU: verify frozen latest personas, official-train provenance, per-field labels and complete pair coverage...', flush=True)
    cfg, m, batches = prepare(yaml.safe_load(a.config.read_text()))
    print(f'48 unique cases × 2 passes = 96 exposures, 12 optimizer steps, warmup2, LR1e-5, save/evaluate6/12. Mean {m["mean_tokens"]:.0f} input+target tokens.', flush=True)
    print('73 native cases per stage: seen12 / held-out24 / original regression37. Step0 caches37 original tool outputs; 182 NEW GPU calls total. No gold scratchpad at evaluation.', flush=True)
    if a.cpu_check:
        print('PASS: CPU only; no GPU query, service, inference or training.'); return cfg
    from training.local_resources import training_gpu_plan, output_parent, runtime_cache
    cards, skipped, selection = training_gpu_plan(a.gpus, 2)
    if len(cards) != 2 or any(c['budget_gib'] < 18 for c in cards):
        raise ValueError('Need two cards with >=18 GiB startup budget each after headroom')
    parent, note = output_parent(ROOT, a.output_dir, min_free_gib=3.); cache, cache_note = runtime_cache(ROOT)
    for msg in (note, cache_note):
        if msg: print(msg, flush=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_self_checks_', dir=parent))
    cfg.update(devices=cards, skipped_gpus=skipped, gpu_selection=selection)
    (folder/'train_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (folder/'batch_plan.json').write_text(json.dumps(batches)+'\n')
    (folder/'personas').mkdir()
    for i in (1, 2, 3):
        shutil.copy2(ROOT/cfg['data_dir']/f'persona_{i}.md', folder/'personas'/f'persona_{i}.md')
    (folder/'agent.yaml').write_text('reasoning: true\n')
    print(f'GPUs {[c["index"] for c in cards]}; output and runner.log: {folder}; loading/preflight ETA estimating.', flush=True)
    env = {**os.environ, **cache, 'CUDA_VISIBLE_DEVICES': ','.join(c['index'] for c in cards),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
        'PYTHONPATH': str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', '')}
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(ROOT/'training/buyer_self_checks_train.py'), '--session', str(folder)]
    began = time.monotonic()
    with (folder/'runner.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while chunk := proc.stdout.read1(4096):
                message = chunk.decode(errors='replace'); sys.stdout.write(message); sys.stdout.flush(); log.write(message); log.flush()
            code = proc.wait()
        except BaseException:
            os.killpg(proc.pid, signal.SIGINT)
            try: proc.wait(timeout=45)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=code, elapsed_seconds=time.monotonic()-began)))
    if code:
        raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    print(f'Completed. Read {folder}/self_check_review.md; no automatic next GPU work.', flush=True)
    return folder


if __name__ == '__main__':
    main()
