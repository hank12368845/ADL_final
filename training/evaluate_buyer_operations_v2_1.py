#!/usr/bin/env python3
"""Paired v2 step8 / v2.1 comparison, using immutable legacy or new cash controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--checkpoint',default='final')
    parser.add_argument('--suite',choices=['legacy','cash'],default='legacy')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve();cfg=json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version')!='buyer_operations_v2_1':raise ValueError('Not a v2.1 training run')
    if args.checkpoint!='final' and args.checkpoint not in {f'step_{s:04d}' for s in (4,8,12,16)}:
        raise ValueError('Use final or a saved step_0004/0008/0012/0016')
    candidate=run/'final_adapter' if args.checkpoint=='final' else run/'checkpoints'/args.checkpoint
    reference=(ROOT/cfg['comparison_adapter']).resolve()
    for name,expected in cfg.get('comparison_fingerprints',{}).items():
        with (reference/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Reference step8 changed since training preparation')
    data=ROOT/cfg['data_dir'];cases=data/('legacy_probe_cases.json' if args.suite=='legacy' else 'cash_probe_cases.json')
    from training.buyer_diagnostics import main as evaluate
    with tempfile.TemporaryDirectory(prefix='adl_v21_comparison_',dir='/tmp') as temporary:
        adapters=Path(temporary)/'adapters.json'
        adapters.write_text(json.dumps({'step8':str(reference),'candidate':str(candidate)}))
        sys.argv=['buyer_diagnostics.py','--versions','operations','--adapters-file',str(adapters),
                  '--cases-file',str(cases),'--seeds','0,1','--gpus',args.gpus,'--port',str(args.port)]
        if args.prepare_only:sys.argv.append('--prepare-only')
        evaluate()


if __name__=='__main__':main()
