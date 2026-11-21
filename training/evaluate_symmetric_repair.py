#!/usr/bin/env python3
"""Evaluate a symmetric-repair run; --latest avoids literal NEW_RUN placeholders."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_run(run=None, latest=False, runs_dir=ROOT/'training/runs'):
    if (run is None) == (not latest):
        raise ValueError('Provide an actual run path OR --latest')
    if latest:
        matches = []
        for cfg_path in Path(runs_dir).glob('*/train_config.json'):
            cfg = json.loads(cfg_path.read_text())
            if cfg.get('data_version') == 'buyer_symmetric_repair_v1':
                matches.append(cfg_path.parent)
        if not matches:
            raise ValueError(f'No symmetric-repair run found in {runs_dir}; train it first')
        run = max(matches, key=lambda p: p.name)
    run = Path(run).resolve()
    cfg = json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version') != 'buyer_symmetric_repair_v1':
        raise ValueError('This wrapper accepts symmetric-repair runs only')
    status = run/'wrapper_result.json'
    if not status.exists() or json.loads(status.read_text()).get('exit') != 0:
        raise ValueError(f'Training has not completed successfully: {run}')
    for step in (16, 32):
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'training_step.json'):
            if not (run/'checkpoints'/f'step_{step:04d}'/name).is_file():
                raise ValueError(f'Missing step{step} checkpoint: {run}')
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, nargs='?')
    p.add_argument('--latest', action='store_true')
    p.add_argument('--runs-dir', type=Path, default=ROOT/'training/runs')
    p.add_argument('--gpus', default='0,1'); p.add_argument('--port', type=int, default=8000)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    run = resolve_run(args.run, args.latest, args.runs_dir)
    print(f'Symmetric repair evaluation run: {run}', flush=True)
    from training.evaluate_board_repair import main as evaluate
    argv = ['evaluate_board_repair.py', str(run), '--gpus', args.gpus, '--port', str(args.port)]
    if args.prepare_only:
        argv.append('--prepare-only')
    previous = sys.argv
    try:
        sys.argv = argv
        return evaluate()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
