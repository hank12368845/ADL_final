#!/usr/bin/env python3
"""User entry point: CPU check or one/two-GPU DDP Buyer operation LoRA pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import yaml
    from training.buyer_operations_train import check_data,sampling_components,decision_mode
    from training.local_resources import output_parent, training_gpu_plan, runtime_cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'training/configs/buyer_operations_probe.yaml')
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--require-gpus', type=int, choices=[1,2],
                        help='Require this many usable GPUs; explicit --gpus 0,1 already requires two')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--epochs', type=int, choices=[1,2], help='Reasoning full-coverage continuation')
    parser.add_argument('--resume', type=Path, help='Reasoning checkpoint with optimizer and per-rank RNG')
    parser.add_argument('--gate-report', type=Path, help='Passed reasoning pilot gate before epoch continuation')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'training/runs')
    parser.add_argument('--cpu-check', action='store_true', help='Check data/model cache; never query or start GPUs')
    parser.add_argument('--plan-only', action='store_true', help='Also query GPUs, without starting training')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs and args.max_steps is not None: raise ValueError('Use epochs OR max-steps')
    if args.max_steps is not None:
        cfg['max_steps'] = args.max_steps
    if not 1 <= cfg['rank'] <= 32 or cfg['max_steps'] < 1:
        raise ValueError('Rank must be 1..32; max-steps must be positive')
    print('CPU: validating data checksums, splits and loss spans...', flush=True)
    train, validation = check_data(cfg, ROOT)
    repair=cfg.get('sampling') in ('board_repair_4_1_3','symmetric_repair_2_2_4')
    reasoning = cfg.get('sampling') in ('reasoning_epoch_pairs','board_repair_4_1_3','symmetric_repair_2_2_4')
    if repair:
        if args.epochs or not 1<=cfg['max_steps']<=32:
            raise ValueError('Board repair is a maximum 32-step pilot; evaluate before extending')
        from training.board_repair_setup import prepare_warm_start
        warm=prepare_warm_start(cfg,ROOT)
        print(f'CPU: parent step{warm["parent_step"]} adapter verified; NEW optimizer/scheduler, local step0.',flush=True)
    if (args.epochs or args.resume or args.gate_report) and not reasoning:
        raise ValueError('Epoch/resume/gate options require the reasoning curriculum')
    if args.epochs:
        cfg['max_steps']=args.epochs*len(train)//cfg['effective_batch_size']
        cfg['eval_every']=64
    cfg['data_manifest_sha256']=hashlib.sha256((ROOT/cfg['data_dir']/'manifest.json').read_bytes()).hexdigest()
    if reasoning and (not cfg.get('thinking') or not 1<=cfg['max_steps']<=cfg['schedule_steps']):
        raise ValueError('Reasoning requires thinking ON and steps within the fixed schedule horizon')
    if cfg.get('experiment') == 'buyer_operations_v2_2_first_token':
        from training.decision_ablation import audit_training
        cfg['decision_ablation'] = audit_training(cfg, ROOT, train)
        print('CPU: archived control recipe, data, both-rank sample order and adapters verified.', flush=True)
    batch_plan=None
    if cfg.get('sampling') in ('paired_4_2_2','cash_offer_4_2_2','reasoning_epoch_pairs','board_repair_4_1_3','symmetric_repair_2_2_4'):
        _,batch_manifest=sampling_components(cfg)
        batch_plan=batch_manifest(train.metadata,cfg['max_steps'],cfg['seed'],1)
        print(f'CPU: {len(batch_plan["steps"])} deterministic global batches; quotas={batch_plan["quotas"]}; '
              f'objective=completion + {cfg.get("decision_weight",0):g} * decision NLL ({decision_mode(cfg)}).',flush=True)
    if reasoning:
        cfg['world_size']=2
        if args.resume:
            from training.reasoning_checkpoint import validate_resume
            cfg['resume_checkpoint']=str(args.resume.resolve())
            cfg['start_step']=validate_resume(cfg,args.resume,batch_manifest(train.metadata,cfg['max_steps'],cfg['seed'],2))
        if cfg['max_steps']>32:
            if not args.resume or not args.gate_report:
                raise ValueError('More than 32 reasoning steps requires a passed pilot gate and --resume checkpoint')
            from training.buyer_reasoning_gate import validate_continuation
            cfg['pilot_gate']=validate_continuation(args.gate_report,cfg,args.resume)
        print(f'CPU: thinking ON; {len(train)//8} steps/epoch; start={cfg.get("start_step",0)}; '
              f'target={cfg["max_steps"]}; fixed LR horizon={cfg["schedule_steps"]}.',flush=True)
    from huggingface_hub import try_to_load_from_cache
    index_path = try_to_load_from_cache(cfg['base_model'], 'model.safetensors.index.json', revision=cfg['revision'])
    config_path = try_to_load_from_cache(cfg['base_model'], 'config.json', revision=cfg['revision'])
    if not isinstance(index_path,str) or not isinstance(config_path,str):
        raise ValueError('Missing cached model index/config; run training/download.sh')
    snapshot = Path(index_path).parent
    index = json.loads(Path(index_path).read_text())
    if not all((snapshot/name).exists() for name in set(index['weight_map'].values())):
        raise ValueError('Base model cache has missing weight shards; run training/download.sh')
    print(f'PASS: train={len(train)}, validation={len(validation)}; base revision cached.', flush=True)
    if args.cpu_check:
        return
    devices, skipped, selection_plan = training_gpu_plan(args.gpus, args.require_gpus)
    world = len(devices)
    if (cfg.get('decision_ablation') or reasoning) and world != 2:
        raise ValueError('This ablation requires two GPUs to preserve the archived rank/sample order')
    if cfg['effective_batch_size'] % world:
        raise ValueError('Effective batch must be divisible by GPU count')
    cfg['devices'], cfg['world_size'] = devices, world
    cfg['gpu_selection'] = selection_plan
    cfg['skipped_gpus'] = skipped
    cfg['versions'] = {}
    cfg['training_sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ('training/buyer_operations_train.py','training/buyer_operations_sampling.py','training/tool_supervision.py',
         'training/train_buyer_operations.py')}
    if cfg.get('sampling')=='cash_offer_4_2_2':
        source='training/buyer_operations_v2_1_sampling.py'
        cfg['training_sources'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    if cfg.get('decision_ablation'):
        source='training/decision_ablation.py'
        cfg['training_sources'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    if reasoning:
        for source in ('training/buyer_reasoning_sampling.py','training/reasoning_checkpoint.py',
                       'training/buyer_reasoning_gate.py'):
            cfg['training_sources'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    if repair:
        for source in ('training/board_repair_sampling.py','training/board_repair_setup.py',
                       *(['training/symmetric_repair_sampling.py'] if cfg.get('sampling')=='symmetric_repair_2_2_4' else [])):
            cfg['training_sources'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    if cfg.get('comparison_adapter'):
        comparison=ROOT/cfg['comparison_adapter']
        if comparison.is_dir():
            cfg['comparison_fingerprints']={}
            for name in ('adapter_config.json','adapter_model.safetensors'):
                with (comparison/name).open('rb') as stream:
                    cfg['comparison_fingerprints'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
    cfg['data_manifest_sha256']=hashlib.sha256((ROOT/cfg['data_dir']/'manifest.json').read_bytes()).hexdigest()
    from importlib.metadata import version
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors'):
        cfg['versions'][name] = version(name)
    parent, note = output_parent(ROOT, args.output_dir, min_free_gib=5.)
    cache_env, cache_note = runtime_cache(ROOT)
    for message in (note, cache_note):
        if message: print(message, flush=True)
    session = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_operations_', dir=parent))
    (session/'train_config.json').write_text(json.dumps(cfg, indent=2))
    if reasoning:
        import shutil
        shutil.copytree(ROOT/cfg['data_dir']/'personas',session/'personas')
        (session/'agent.yaml').write_text('reasoning: true\n')
    if batch_plan is not None:
        (session/'batch_plan.json').write_text(json.dumps(batch_manifest(train.metadata,cfg['max_steps'],cfg['seed'],world),indent=1))
    print(f'GPU plan: {devices}; skipped={skipped}', flush=True)
    print(f'DDP workers={world}; microbatch=1/GPU; accumulation={cfg["effective_batch_size"]//world}; '
          f'effective batch={cfg["effective_batch_size"]}; steps={cfg["max_steps"]}', flush=True)
    print(f'Output/checkpoints/log: {session}', flush=True)
    if args.plan_only:
        print('PASS: plan only; no GPU training started.'); return
    env = {**os.environ, **cache_env, 'CUDA_VISIBLE_DEVICES': ','.join(d['index'] for d in devices),
           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4'}
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={world}', str(ROOT/'training/buyer_operations_train.py'), '--session', str(session)]
    # Worker import must work even when launched directly from its training/ path.
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
    started = time.monotonic()
    print('Starting workers / loading / memory preflight; startup ETA estimating.', flush=True)
    with (session/'runner.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # Forward tqdm carriage returns live, preserving progress/ETA in the log too.
            assert process.stdout is not None
            while chunk := process.stdout.read1(4096):
                text = chunk.decode(errors='replace'); sys.stdout.write(text); sys.stdout.flush()
                log.write(text); log.flush()
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    (session/'wrapper_result.json').write_text(json.dumps({'exit': code, 'elapsed_seconds': time.monotonic()-started}))
    if code:
        raise SystemExit(f'FAIL: training exit={code}; inspect {session}/runner.log')
    print(f'PASS: training finished; evaluate candidate at {session}/final_adapter before promotion.')


if __name__ == '__main__':
    main()
