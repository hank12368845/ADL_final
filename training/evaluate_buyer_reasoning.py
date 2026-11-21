#!/usr/bin/env python3
"""Native thinking ON: base versus trained adapter on two validation cash markets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from training.decision_ablation import digest, fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--checkpoint',default='step_0032')
    parser.add_argument('--gpus',default='0,1'); parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); run=args.run.resolve()
    cfg=json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version')!='buyer_operations_reasoning_v1' or not cfg.get('thinking'):
        raise ValueError('Expected a thinking-ON training run')
    candidate=run/'final_adapter' if args.checkpoint=='final' else run/'checkpoints'/args.checkpoint
    if candidate.resolve().parent not in ((run/'checkpoints').resolve(),run):
        raise ValueError('Checkpoint must belong to this run')
    step=json.loads((candidate/'training_step.json').read_text())
    if not step['thinking']: raise ValueError('Checkpoint was not trained with thinking ON')
    data=ROOT/cfg['data_dir']; manifest=json.loads((data/'manifest.json').read_text())
    if digest(data/'manifest.json')!=cfg['data_manifest_sha256'] or not manifest['thinking']:
        raise ValueError('Training data changed')
    for source,expected in manifest['sources'].items():
        if digest(ROOT/source)!=expected: raise ValueError('Reasoning prompt/data source changed')
    cases=data/'cash_probe_cases.json'
    if digest(cases)!=manifest['cash_probe_sha256']:raise ValueError('Cash probe changed')
    provenance={'experiment':'buyer_operations_reasoning_v1','thinking':True,'step':step['step'],
                'candidate_checkpoint':str(candidate),'candidate_fingerprint':fingerprint(candidate),
                'training_manifest_sha256':cfg['data_manifest_sha256'],'base_revision':cfg['revision'],
                'comparison':'base ON versus trained ON; same original observations and short personas'}
    from training.buyer_diagnostics import main as evaluate
    with tempfile.TemporaryDirectory(prefix='adl_reasoning_eval_',dir='/tmp') as temporary:
        folder=Path(temporary)
        (folder/'adapters.json').write_text(json.dumps({'base_on':None,'candidate':str(candidate)}))
        (folder/'manifest.json').write_text(json.dumps(provenance))
        argv=['buyer_diagnostics.py','--versions','operations_reasoning','--thinking',
              '--adapters-file',str(folder/'adapters.json'),'--experiment-manifest',str(folder/'manifest.json'),
              '--cases-file',str(cases),'--analysis','cash_decision','--seeds','0,1',
              '--gpus',args.gpus,'--port',str(args.port)]
        if args.prepare_only:argv.append('--prepare-only')
        previous=sys.argv
        try:sys.argv=argv;return evaluate()
        finally:sys.argv=previous


if __name__=='__main__':main()
