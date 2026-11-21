#!/usr/bin/env python3
"""User-started paired base/LoRA v2 validation probe; 192 calls by default."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--checkpoint',default='final',help='final or step_0004 / step_0008 / step_0012 / step_0016')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve();cfg=json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version')!='buyer_operations_v2':raise ValueError('Not a v2 training run')
    checkpoint=run/'final_adapter' if args.checkpoint=='final' else run/'checkpoints'/args.checkpoint
    if args.checkpoint!='final' and args.checkpoint not in {f'step_{i:04d}' for i in (4,8,12,16)}:
        raise ValueError('Use final or a saved v2 probe checkpoint')
    cases=ROOT/cfg['data_dir']/'probe_cases.json'
    from training.buyer_diagnostics import main as evaluate
    sys.argv=['buyer_diagnostics.py','--versions','operations','--compare-base','--adapter',str(checkpoint),
              '--cases-file',str(cases),'--seeds','0,1','--gpus',args.gpus,'--port',str(args.port)]
    if args.prepare_only:sys.argv.append('--prepare-only')
    evaluate()


if __name__=='__main__':main()
