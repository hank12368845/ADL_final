#!/usr/bin/env python3
"""One full public 5x10 market: fixed official step84 and unchanged current personas."""
from collections import Counter
import argparse
import copy
import hashlib
import importlib.metadata
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
REFERENCE = ROOT/'results/basic_checks/20261120-153410_base_step84__fme0dw_/plan.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_sources(manifest):
    from training.decision_ablation import check_fingerprint
    for path, digest in {**manifest['sources'], **manifest['inputs']}.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen market input/source changed: {path}')
    for path, fingerprint in manifest['adapters'].items():
        check_fingerprint(path, fingerprint)
    for name, version in manifest['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Market package changed: {name}')


def validate_run(loaded, manifest):
    from adlmarket.casting import plan_markets
    from training.step84_market_report import IDS
    if len(loaded.sellers) != 5 or len(loaded.buyers) != 10:
        raise ValueError('Expected exactly5 sellers and10 buyers')
    seats = plan_markets(loaded.config.casting, loaded.sellers, loaded.buyers, loaded.config.seed, pads=loaded.pads)
    if len(seats) != 1 or loaded.config.market.model_dump(mode='json') != manifest['market_rules']:
        raise ValueError('Rules changed or extra market rotations requested')
    targets = [a for a in loaded.buyers if a.owner != 'ta']
    if {a.id for a in targets} != set(IDS):
        raise ValueError('Expected all three original buyer personas')
    for a in targets:
        i = a.id[-1]
        spec = loaded.models[a.model]
        if not a.reasoning or a.persona_body() != manifest['personas'][i].strip():
            raise ValueError('Target persona or thinking mode changed')
        if not spec.lora or Path(spec.lora.path).resolve() != Path(manifest['adapter']).resolve():
            raise ValueError('Target is not fixed official step84')
        if (spec.temperature, spec.top_p, spec.max_tokens) != (.7, .95, 8192):
            raise ValueError('Target sampling differs from official settings')
        if spec.lora_rename != {'model.layers.': 'model.language_model.layers.'}:
            raise ValueError('Missing official HF-to-vLLM adapter name mapping')
    return seats


def prepare(directory, seed=0):
    import yaml
    from adlmarket.setups import market_run, deep_merge
    from training.decision_ablation import check_fingerprint, fingerprint
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REFERENCE.read_text())
    adapter = Path(ref['cached_adapter'])
    check_fingerprint(adapter, ref['cached_adapter_fingerprints'])
    if json.loads((adapter/'training_step.json').read_text()) != {'step':84, 'thinking':True}:
        raise ValueError('Not official thinking-ON step84')
    personas = {}
    target = directory/'buyer'
    target.mkdir()
    for i in ('1','2','3'):
        text = (ROOT/f'agents/example_student/buyer/persona_{i}.md').read_text()
        if text != ref['original_personas'][i] or not 0 < len(text) <= 8000:
            raise ValueError('Original persona changed since the fixed-model diagnostic')
        personas[i] = text
        (target/f'persona_{i}.md').write_text(text)
    (target/'agent.yaml').write_text(yaml.safe_dump(dict(id='example_student_buyer', owner='example_student',
        role='buyer', controller='llm', model='base', reasoning=True, lora=str(adapter)), sort_keys=False))
    economy = yaml.safe_load((ROOT/'configs/economy.yaml').read_text())
    config = yaml.safe_load((ROOT/'configs/buyer_scored_short.yaml').read_text())
    config.update(name='step84_full_market_diagnostic', seed=seed, rotate=False, repeats=1,
        models=str(ROOT/'configs/models.yaml'), save_trajectories=True, save_prompts=True,
        max_workers=15, parallel_markets=1,
        market=deep_merge(economy['market'], economy['phase2']['market']), output_dir=str(directory))
    config['buyers'][0] = str(target)
    config_path = directory/'market.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = market_run(config_path, root=ROOT)
    rules = loaded.config.market
    if (rules.rounds.total,rules.rounds.turns_per_round,rules.rounds.output_tokens_per_round,rules.history_turns) != (5,6,8192,6):
        raise ValueError('Incorrect formal horizon/token/history settings')
    if rules.rounds.free_messages or rules.buyer_swaps:
        raise ValueError('Loss accounting requires original one-action marketplace without swaps')
    from adlmarket.market import load_catalog
    from adlmarket.schema import build_items_schema
    schema = build_items_schema('step84_preflight', rules.schema_, rules.schema_.items_file(0),
                                load_catalog(ROOT/rules.catalog), ROOT, 5)
    if len(schema.plans) != 24:
        raise ValueError('Expected public24-item economy')
    sources = {str(p.resolve()):sha(p) for p in (ROOT/'adlmarket').glob('*.py')}
    paths = [Path(__file__), ROOT/'training/step84_market_report.py', ROOT/'training/basic_checks.py',
             ROOT/'training/market_load_balance.py', ROOT/'training/local_resources.py',
             ROOT/'configs/models.yaml', ROOT/'configs/economy.yaml', ROOT/'configs/buyer_scored_short.yaml',
             ROOT/'scenarios/public/catalog.yaml', ROOT/'scenarios/public/items_example.json', REFERENCE]
    paths += [ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in ('1','2','3')]
    for a in loaded.everyone():
        if a.source:
            paths.append(Path(a.source))
    for role in ('buyer','seller'):
        paths += list((ROOT/f'agents/_ta/{role}').glob('persona_*.md'))
    sources.update({str(p.resolve()):sha(p) for p in paths})
    adapters = {str(Path(loaded.models[a.model].lora.path).resolve()):fingerprint(loaded.models[a.model].lora.path)
                for a in loaded.everyone() if loaded.models[a.model].lora}
    manifest = dict(experiment='step84_one_full_market_v1', adapter=str(adapter), adapter_fingerprints=ref['cached_adapter_fingerprints'],
        base_model=ref['base_model'], revision=ref['revision'], versions=ref['versions'], personas=personas,
        persona_sha256={i:hashlib.sha256(text.encode()).hexdigest() for i,text in personas.items()},
        seed=seed, total_markets=1, total_actions=450, target_actions=90, training_updates=0,
        market_rules=rules.model_dump(mode='json'), config=str(config_path), sources=sources, adapters=adapters,
        inputs={str(p):sha(p) for p in directory.rglob('*') if p.is_file()},
        status='prepared_no_gpu_work', description='One public market; fixed toy opponents, original3 personas, official step84; no base arm, no extra rotations.')
    seats = validate_run(loaded, manifest)
    manifest['seats'] = [dict(market=s.market_id, rotation=s.rotation, sellers=[a.id for a in s.sellers],buyers=[a.id for a in s.buyers]) for s in seats]
    check_sources(manifest)
    (directory/'suite.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def configure_resources(directory, manifest, selection='0,1', port=8000, mock=False):
    import yaml
    from adlmarket.setups import market_run
    from training.local_resources import gpu_models, query_gpus, select_gpus, runtime_cache
    from training.market_load_balance import balance_ta_routes, validate_equivalent
    config = yaml.safe_load(Path(manifest['config']).read_text())
    before = market_run(Path(manifest['config']), root=ROOT)
    models = yaml.safe_load((ROOT/'configs/models.yaml').read_text())
    devices, skipped, cache, balance = [], [], {}, None
    if not mock:
        devices, skipped = select_gpus(query_gpus(), selection, reserve_gib=2., min_budget_gib=14.)
        for d in devices:
            print(f'GPU{d["index"]}: free {d["free_gib"]:.2f}GiB; startup budget {d["budget_gib"]:.2f}GiB after reserve.', flush=True)
        if skipped:
            print(f'Skipped GPUs: {skipped}. No other jobs stopped.', flush=True)
        if len(devices) == 1:
            print('SINGLE-GPU run: only one selected card fits; no automatic live expansion.', flush=True)
        models = gpu_models(models, devices, port)
        if len(devices) == 2:
            balance = balance_ta_routes(config, before, models, directory, ROOT, devices, port)
        for serving in models['serving'].values():
            serving['extra_args'] += ['--revision', manifest['revision'], '--tokenizer-revision', manifest['revision']]
        cache, note = runtime_cache(ROOT)
        if note:
            print(f'CACHE: {note}', flush=True)
    model_path = directory/'models.yaml'
    model_path.write_text(yaml.safe_dump(models, sort_keys=False))
    config['models'] = str(model_path)
    effective = directory/'effective_market.yaml'
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    resolved = market_run(effective, root=ROOT)
    validate_equivalent(before, resolved)
    validate_run(resolved, manifest)
    routes = {a.id:resolved.models[a.model].url_for(resolved.models[a.model].model) for a in resolved.everyone()}
    if devices and len(devices) == 2 and len(set(routes.values())) != 2:
        raise ValueError('Both GPUs configured but agent routes do not use both')
    if balance:
        for aid,index in balance['assignments'].items():
            if routes[aid] != f'http://localhost:{port+index}/v1':
                raise ValueError('Actual agent routing differs from load-balancing plan')
    resource = dict(devices=devices, skipped=skipped, routes=routes, cache_environment=cache,load_balance=balance,mock=mock,
        note='Student personas share the same step84 adapter on server0; TA roles distributed by generation-cost priors. Actual call/token loads reported after play.')
    (directory/'resource_plan.json').write_text(json.dumps(resource,indent=2)+'\n')
    manifest['runtime_inputs'] = {str(p):sha(p) for p in directory.rglob('*') if p.is_file() and
                                  (p.name in ('models.yaml','effective_market.yaml') or 'routing_agents' in p.parts)}
    (directory/'suite.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Agent endpoint counts: {dict(Counter(routes.values()))}; output/log {directory}',flush=True)
    return effective, cache


def play(directory, manifest, effective, cache, mock=False):
    from training.basic_checks import read_new, duration
    command = [sys.executable,'-u',str(ROOT/'run_market.py'),'--config',str(effective), '--mock-llm' if mock else '--launch-vllm']
    env = {**os.environ,**cache,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
           'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    check_sources(manifest)
    if any(sha(path)!=value for path,value in manifest['runtime_inputs'].items()):
        raise ValueError('Runtime routing/model inputs changed before launch')
    offsets={};count=0;target_counts=Counter();position='loading / warm-up';began=time.monotonic();play_start=None;last=0;run=None
    print('One market,450 actions; each target persona30. Loading ETA estimating.',flush=True)
    with (directory/'runner.log').open('w') as log, (directory/'progress.log').open('w') as progress:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(directory/'runner.log',offsets):
                    print(line,flush=True)
                runs=list(directory.glob('run_market_*'))
                if runs:
                    run=runs[0]
                    for path in sorted(run.glob('markets/*/events.jsonl')):
                        for line in read_new(path,offsets):
                            row=json.loads(line)
                            if row.get('type')=='market_start':
                                play_start=play_start or time.monotonic()
                            if row.get('type')=='turn':
                                count+=1
                                if row['agent'].startswith('example_student_buyer_p'):
                                    target_counts[row['agent']]+=1
                                position=f'round{row["round"]}/5 turn{(row["tick"]-1)%6+1}/6'
                    for path in run.glob('vllm/*.log'):
                        lines=read_new(path,offsets)
                        if lines and not count:
                            print(f'[vLLM {path.name}] {lines[-1][-220:]}',flush=True)
                now=time.monotonic()
                if done or now-last>=5:
                    measured=now-play_start if play_start else 0
                    speed=count/measured if count and measured>=5 else 0
                    eta=duration((450-count)/speed) if speed else 'estimating'
                    if count==450:eta='00:00:00'
                    filled=min(24,int(24*count/450));bar='#'*filled+'-'*(24-filled)
                    targets=' '.join(f'p{i}:{target_counts[f"example_student_buyer_p{i}"]}/30' for i in (1,2,3))
                    line=f'[{bar}] {count}/450 | {position} | {targets} | elapsed {duration(now-began)} | {speed:.2f} actions/s | ETA {eta}'
                    if count==450:line+=' | finalizing / server shutdown'
                    print(line,flush=True);progress.write(line+'\n');progress.flush();last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    result=dict(exit=proc.returncode,elapsed_seconds=time.monotonic()-began,actions=count,
                target_actions=dict(target_counts),mock=mock,training_updates=0,run_dir=str(run) if run else None)
    (directory/'wrapper_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode or run is None:
        raise RuntimeError(f'Market failed ({proc.returncode}); inspect {directory}/runner.log and vllm logs')
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--mock',action='store_true')
    parser.add_argument('--report',type=Path)
    a=parser.parse_args()
    from training.step84_market_report import write_report
    if a.report:
        info=json.loads((a.report/'wrapper_result.json').read_text())
        manifest=json.loads((a.report/'suite.json').read_text());check_sources(manifest)
        return write_report(a.report,info['run_dir'])
    from training.local_resources import output_parent
    parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
    if note:print(f'STORAGE: {note}',flush=True)
    directory=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_step84_market_',dir=parent))
    print('CPU: verify step84, original personas, public items and phase2 rules; ETA estimating.',flush=True)
    manifest=prepare(directory,a.seed)
    print(f'Prepared: {directory}; one5x10 market,30 turns,450 actions,90 target actions.',flush=True)
    if a.prepare_only:
        print('PASS CPU preparation; no GPU query or service started.',flush=True);return directory
    effective,cache=configure_resources(directory,manifest,a.gpus,a.port,a.mock)
    run=play(directory,manifest,effective,cache,a.mock)
    check_sources(manifest)
    report=write_report(directory,run)
    if report['mock'] != a.mock:raise ValueError('Mock/real mode mismatch')
    print(f'Completed. Read {directory}/market_loss_review.md; no automatic next training or GPU work.',flush=True)
    return directory


if __name__=='__main__':main()
