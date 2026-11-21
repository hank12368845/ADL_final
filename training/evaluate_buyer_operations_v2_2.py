#!/usr/bin/env python3
"""Same-cash counterfactuals: old v2 step8, matched v2.1 control and new v2.2."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def comparison_spec(run, checkpoint):
    from training.decision_ablation import EXPERIMENT, check_fingerprint, digest
    cfg = json.loads((run/'train_config.json').read_text())
    if cfg.get('experiment') != EXPERIMENT or cfg.get('decision_supervision') != 'first_token':
        raise ValueError('Not a v2.2 first-token ablation run')
    audit = cfg['decision_ablation']
    control = Path(audit['control_run'])
    if digest(control/'train_config.json') != audit['control_config_sha256']:
        raise ValueError('Archived control config changed')
    if digest(control/'batch_plan.json') != audit['batch_plan_sha256']:
        raise ValueError('Archived control batch plan changed')
    if digest(ROOT/cfg['data_dir']/'manifest.json') != audit['data_manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    if checkpoint not in ('step_0008', 'step_0012', 'final'):
        raise ValueError('Use step_0008, step_0012 or final')
    suffix = Path('final_adapter') if checkpoint == 'final' else Path('checkpoints')/checkpoint
    old = (ROOT/cfg['comparison_adapter']).resolve()
    matched = control/suffix
    candidate = run/suffix
    check_fingerprint(old, cfg['comparison_fingerprints'])
    check_fingerprint(matched, audit['control_checkpoints'][checkpoint])
    if json.loads((candidate/'training_step.json').read_text())['step'] != (16 if checkpoint == 'final' else int(checkpoint[-4:])):
        raise ValueError('Candidate checkpoint step mismatch')
    data = ROOT/cfg['data_dir']
    cases = data/'cash_probe_cases.json'
    saved = audit.get('cash_cases_sha256')
    if saved is None or digest(cases) != saved:
        raise ValueError('Cash probe changed since training preparation')
    return {'v2_step8': str(old), 'v21_control': str(matched), 'candidate': str(candidate)}, cases, {
        'experiment': cfg['experiment'], 'checkpoint': checkpoint, 'control_run': str(control),
        'candidate_run': str(run), 'data_manifest_sha256': digest(data/'manifest.json'),
        'cash_cases_sha256': digest(cases), 'changed_training_factor': audit['changed_training_factor'],
        'reference_labels': {'v2_step8': 'Old v2 high-recall reference',
                             'v21_control': f'Archived v2.1 at matched {checkpoint}',
                             'candidate': f'First-token v2.2 at {checkpoint}'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', choices=['step_0008', 'step_0012', 'final'], default='step_0008')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    adapters, cases, provenance = comparison_spec(args.run.resolve(), args.checkpoint)
    from training.decision_ablation import fingerprint
    provenance['adapter_fingerprints'] = {label: fingerprint(path) for label, path in adapters.items()}
    from training.buyer_diagnostics import main as evaluate
    with tempfile.TemporaryDirectory(prefix='adl_v22_comparison_', dir='/tmp') as temporary:
        folder = Path(temporary)
        (folder/'adapters.json').write_text(json.dumps(adapters))
        (folder/'experiment.json').write_text(json.dumps(provenance))
        argv = ['buyer_diagnostics.py', '--versions', 'operations', '--adapters-file', str(folder/'adapters.json'),
                '--experiment-manifest', str(folder/'experiment.json'), '--cases-file', str(cases),
                '--analysis', 'cash_decision', '--seeds', '0,1', '--gpus', args.gpus, '--port', str(args.port)]
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
