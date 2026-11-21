#!/usr/bin/env python3
"""Experimental low-memory TP2 serving; user-launched only, same full market rules."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.compare_seller_personas import prepare_pair,validate,initial_state,play,report,write_json
from training.diagnose_step84_market import check_sources,sha
from adlmarket.setups import market_run


def tp_plan(rows,selection='0,1',reserve_gib=2.,minimum_gib=8.):
    from training.local_resources import select_gpus
    ids=[s.strip() for s in selection.split(',')]
    if len(ids)!=2 or len(set(ids))!=2:raise ValueError('TP2 requires exactly two distinct physical GPUs')
    selected,skipped=select_gpus(rows,selection,reserve_gib=reserve_gib,min_budget_gib=minimum_gib)
    if len(selected)!=2:raise RuntimeError(f'TP2 needs two cards with >=8GiB budget AFTER reserve:{skipped}')
    # vLLM uses the same utilization fraction on all TP ranks; budget by the smaller card.
    common_fraction=min(d['fraction'] for d in selected)
    return dict(devices=selected,skipped=skipped,fraction=common_fraction,
        per_rank_budget_gib={d['index']:common_fraction*d['total_gib'] for d in selected},
        reserve_gib=reserve_gib,minimum_gib=minimum_gib,
        note='Eligibility only, NOT evidence that Qwen+LoRA+64K cache actually fits; vLLM must profile it.')


def configure_tp2(folder,m,rows,selection,port):
    from training.market_load_balance import validate_equivalent
    plan=tp_plan(rows,selection)
    cfg=yaml.safe_load(Path(m['config']).read_text());before=market_run(m['config'],root=ROOT)
    models=yaml.safe_load((ROOT/'configs/models.yaml').read_text())
    for model in models['models'].values():
        model.pop('replicas',None);model['base_url']=f'http://localhost:{port}/v1'
    server=copy.deepcopy(models['serving']['base'])
    server.update(gpus=[','.join(d['index'] for d in plan['devices'])],host='localhost',port=port,
                  gpu_memory_utilization=plan['fraction'],max_loras=4,max_model_len=64000)
    extra=server['extra_args']
    # Remove the old64-sequence override before adding the memory-limited setting.
    extra=[x for i,x in enumerate(extra) if x!='--max-num-seqs' and not (i and extra[i-1]=='--max-num-seqs')]
    server['extra_args']=extra+['--max-num-seqs','8','--max-num-batched-tokens','4096','--enforce-eager',
        '--revision',m['revision'],'--tokenizer-revision',m['revision']]
    models['serving']={'base':server}
    model_path=folder/'models.yaml';model_path.write_text(yaml.safe_dump(models,sort_keys=False))
    cfg.update(models=str(model_path),call_timeout=600.)
    effective=folder/'effective_market.yaml';effective.write_text(yaml.safe_dump(cfg,sort_keys=False))
    after=market_run(effective,root=ROOT);validate(after,m);validate_equivalent(before,after)
    if initial_state(after)!=m['initial_signature']:raise ValueError('TP2 configuration changed initial market facts')
    if len({s.lora.path for s in after.models.values() if s.lora})>4:raise ValueError('Too many adapters for4 LoRA slots')
    from adlmarket.serving import vllm_command
    # Build the actual official command without starting it; this catches accidental replica syntax.
    command,environment=vllm_command(after.models['base'],after.serving['base'],0)
    if '--tensor-parallel-size' not in command or command[command.index('--tensor-parallel-size')+1]!='2':
        raise ValueError('Official harness did not build a TP2 command')
    write_json(folder/'resource_plan.json',dict(serving_mode='tensor_parallel_2',plan=plan,
        routes={a.id:after.models[a.model].url_for(after.models[a.model].model) for a in after.everyone()},
        official_serve_command=command,gpu_environment=environment,
        note='One base model across two cards, not two full replicas. No quantization or market-rule changes.'))
    m['runtime_inputs']={str(p):sha(p) for p in [model_path,effective]}
    write_json(folder/'suite.json',m)
    return effective


def main():
    from training.local_resources import output_parent,query_gpus,runtime_cache
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8100)
    p.add_argument('--seed',type=int,default=4);p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'compare_plan.json').read_text())
        if plan.get('serving_mode')!='tensor_parallel_2':raise ValueError('Resume only a TP2 experiment; do not mix serving modes')
    else:
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_seller_tp2_',dir=parent))
        print('CPU: freeze the samev1/v2 experiment; experimental TP2 serving, startup fit still unverified.',flush=True)
        plan=prepare_pair(folder,a.seed);plan['serving_mode']='tensor_parallel_2';write_json(folder/'compare_plan.json',plan)
        for path in plan['arms'].values():
            arm=Path(path);m=json.loads((arm/'suite.json').read_text());m['sources'][str(Path(__file__).resolve())]=sha(__file__)
            write_json(arm/'suite.json',m)
    print(f'Prepared:{folder}; TP2 one model across two cards;900actions; no training.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query or service.');return folder
    for index,(label,path) in enumerate(plan['arms'].items()):
        arm=Path(path);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        old=json.loads((arm/'wrapper_result.json').read_text()) if (arm/'wrapper_result.json').exists() else {}
        if old.get('exit')==0 and old.get('actions')==450:
            print(f'Skip completed {label}',flush=True);continue
        for stale in arm.glob('run_market_*'):stale.rename(stale.with_name(f'archived_{time.time_ns()}_{stale.name}'))
        rows=query_gpus();effective=configure_tp2(arm,m,rows,a.gpus,a.port)
        print(f'{label}: TP2 budget={json.loads((arm/"resource_plan.json").read_text())["plan"]["per_rank_budget_gib"]}; '
              'vLLM loading/cache profiling must pass before market actions; ETA estimating.',flush=True)
        cache,note=runtime_cache(ROOT)
        if note:print(note,flush=True)
        play(arm,m,effective,cache,False,index*450)
    result=report(folder)
    print(f'Completed.Read {folder}/persona_comparison.md. TP2 affects serving, not persona/training/market limits.',flush=True)
    return result


if __name__=='__main__':main()
