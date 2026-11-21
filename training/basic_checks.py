#!/usr/bin/env python3
"""CPU preflight, market progress wrapper, and operation-result audit.

Run from any directory. GPU work requires an explicit --mode gpu invocation by the user.
The wrapper uses the original run_market.py in a subprocess, without engine changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.metadata as metadata
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


def json_rows(path: Path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def preflight() -> int:
    """Read package metadata/import CPU libraries; never initialize CUDA or call a model."""
    from adlmarket.casting import plan_markets
    from adlmarket.market import load_catalog
    from adlmarket.schema import build_items_schema
    from adlmarket.setups import PHASE_MODE, economy_run, market_run

    print('[1/4] Python / package versions (CPU only)', flush=True)
    print(sys.executable, sys.version.split()[0])
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise ValueError('Use the project Python 3.11/3.12 environment.')
    for name in ('openai', 'pydantic', 'pyyaml', 'vllm', 'torch', 'transformers',
                 'peft', 'trl', 'datasets', 'accelerate', 'safetensors'):
        print(f'  {name}: {metadata.version(name)}')
    for name in ('openai', 'pydantic', 'yaml', 'safetensors'):
        importlib.import_module(name)
    # Importing torch and doing CPU arithmetic checks its shared libraries, not CUDA.
    import torch
    if torch.tensor([1, 2], device='cpu').sum().item() != 3:
        raise RuntimeError('torch CPU operation failed')
    print(f'  torch CPU operation: PASS; compiled CUDA: {torch.version.cuda}')

    print('[2/4] Config / personas / LoRA headers / item schema', flush=True)
    catalog = load_catalog(ROOT / 'scenarios/public/catalog.yaml')
    for role in ('buyer', 'seller'):
        run = market_run(ROOT / f'configs/basic_{role}.yaml', root=ROOT)
        plans = plan_markets(run.config.casting, run.sellers, run.buyers,
                             run.config.seed, pads=run.pads)
        if len(plans) != 1 or (len(plans[0].sellers), len(plans[0].buyers)) != (3, 3):
            raise ValueError(f'{role}: expected one 3x3 market')
        schema = build_items_schema('preflight', run.config.market.schema_,
                                    run.config.market.schema_.items_file(0), catalog, ROOT, 3)
        if len(schema.plans) != 24:
            raise ValueError('Expected 24 played items')
        for agent in run.sellers + run.buyers:
            print(f'  {agent.id}: persona={len(agent.persona_body())} chars, '
                  f'thinking={agent.reasoning}, model={agent.model}')
        print(f'  basic_{role}: 1 market, 3x3, 30 turns, 180 agent actions, '
              f'{len(schema.plans)} items, full prompts saved')

    print('[3/4] Official economy seating (no markets played)', flush=True)
    for phase in (1, 2, 3):
        run = economy_run(ROOT / 'configs/economy.yaml', PHASE_MODE[phase], root=ROOT)[0]
        plans = plan_markets(run.config.casting, run.sellers, run.buyers,
                             run.config.seed, pads=run.pads)
        sizes = sorted({(len(p.sellers), len(p.buyers)) for p in plans})
        print(f'  phase {phase}: {len(plans)} markets, sizes={sizes}')

    print('[4/4] Results directory write access', flush=True)
    output = ROOT / 'results/basic_checks'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output) as stream:
        stream.write(b'check')
    print('PASS: CPU preflight. GPU kernels / vLLM / real inference are NOT tested.')
    return 0


def audit(run_dir: Path) -> int:
    """Separate successful process completion from model/action quality."""
    meta = json.loads((run_dir / 'metadata.json').read_text())
    plans = json.loads((run_dir / 'plan.json').read_text())
    problems = []
    errors, tools = Counter(), Counter()
    trades = actions = native = repaired = llm_turns = 0
    for plan in plans:
        folder = run_dir / 'markets' / plan['market']
        final = json.loads((folder / 'final.json').read_text())
        scores = {r['agent']: r for r in json.loads((folder / 'scores.json').read_text())}
        trades += len(final['trades'])
        for seller in final['sellers']:
            expected = round((seller['revenue'] - seller['cost_of_sold']) / 100, 2)
            if scores[seller['id']]['raw'] != expected:
                problems.append(f"{seller['id']}: seller score mismatch")
            if any(q < 0 for q in seller['stock'].values()):
                problems.append(f"{seller['id']}: negative stock")
        for buyer in final['buyers']:
            expected = round((buyer['gained_utility'] + buyer['cash'] - buyer['budget']) / 100, 2)
            if scores[buyer['id']]['raw'] != expected or buyer['cash'] < 0:
                problems.append(f"{buyer['id']}: buyer score/cash mismatch")
        for row in json_rows(folder / 'transcript.jsonl'):
            actions += 1
            tools[row.get('tool') or 'NO_TOOL'] += 1
            if not row.get('result', {}).get('ok'):
                errors[row.get('result', {}).get('error') or 'unknown'] += 1
        # events retain parse-fallback flags absent from compact transcript rows.
        for row in json_rows(folder / 'events.jsonl'):
            if row.get('type') == 'turn' and (llm := row.get('llm')):
                if llm.get('strategy'):  # mock/scripted decisions
                    continue
                llm_turns += 1
                fallback = llm.get('parsed_from_text') or llm.get('arguments_repaired')
                repaired += bool(fallback)
                native += bool(row.get('tool')) and not fallback
    config_path = run_dir / 'config.yaml'
    import yaml
    config = yaml.safe_load(config_path.read_text())
    rounds = config.get('market', {}).get('rounds', {})
    expected_actions = sum(len(p['sellers']) + len(p['buyers']) for p in plans)
    expected_actions *= rounds.get('total', 5) * rounds.get('turns_per_round', 6)
    if actions != expected_actions:
        problems.append(f'action count {actions} != {expected_actions}')
    if meta.get('status') != 'finished':
        problems.append('metadata status is not finished')
    if not trades:
        problems.append('no trades: inspect transcript before calling operation test successful')
    calls = list(json_rows(run_dir / 'inference.jsonl'))
    outcomes = Counter(r.get('outcome') for r in calls)
    print(f"\nRun: {run_dir}\nmode={'MOCK' if meta.get('mock_llm') else 'REAL LLM'}; "
          f'actions={actions}/{expected_actions}; trades={trades}')
    print(f'Tools: {dict(tools)}\nRejected actions: {dict(errors)}')
    print(f'Call outcomes: {dict(outcomes)}; '
          f'no-repair tool decisions={native}/{llm_turns}; policy repairs={repaired}')
    infra = sum(r.get('model_failures', 0) for r in json_rows(run_dir / 'scores.jsonl'))
    no_call = sum(r.get('no_call_turns', 0) for r in json_rows(run_dir / 'scores.jsonl'))
    if infra:
        problems.append(f'{infra} infrastructure failures')
    print(f'Model infrastructure failures={infra}; no-tool turns={no_call}')
    if not meta.get('mock_llm') and (no_call or repaired or errors):
        print('REVIEW: inspect invalid actions / parsing fallbacks; completion does not prove reliability.')
    print((run_dir / 'standings.md').read_text())
    summary = {'mock': meta.get('mock_llm'), 'actions': actions, 'trades': trades,
               'rejected_actions': dict(errors), 'model_outcomes': dict(outcomes),
               'no_call_turns': no_call, 'policy_repairs': repaired, 'problems': problems}
    (run_dir / 'basic_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not meta.get('mock_llm'):
        from training.persona_audit import analyze
        compliance = {p['market']: analyze(run_dir / 'markets' / p['market']) for p in plans}
        (run_dir / 'persona_audit.json').write_text(json.dumps(compliance, indent=2))
        for market, data in compliance.items():
            for aid, checks in data['agents'].items():
                print(f'Persona checks {market}/{aid}: {dict(checks["violations"])}; '
                      f'low-grade loss=${checks["low_loss"]:.2f}')
    print('PASS: operation / score audit (not a strategic grade).' if not problems
          else f'FAIL: {problems}')
    from training.market_scorecard import write_scorecard
    write_scorecard(run_dir)
    if not meta.get('mock_llm'):
        from training.buyer_failure_audit import audit_failures
        audit_failures(run_dir)
    print('The no-repair count does not prove the server/client used no fallback.')
    return int(bool(problems))


def demo() -> int:
    """Exercise real market tools with chosen actions, without any LLM."""
    import random
    from adlmarket.market import Market, load_catalog
    from adlmarket.schema import build_items_schema
    from adlmarket.scoring import score_market
    from adlmarket.setups import market_run

    run = market_run(ROOT / 'configs/basic_seller.yaml', root=ROOT)
    spec = run.config.market
    catalog = load_catalog(ROOT / spec.catalog)
    schema = build_items_schema('demo', spec.schema_, spec.schema_.items_file(0), catalog, ROOT, 3)
    market = Market.generate('cpu_demo', spec, schema, {p.id: p for p in catalog},
                             run.sellers, run.buyers, random.Random(0))
    market.round, market.tick = 1, 1
    pid = 'P72'  # monopoly, high grade only; three units in this 3-buyer market
    seller = next(s for s in market.sellers.values() if s.stock.get(pid, 0) >= 3)
    buyer = next(iter(market.buyers.values()))
    price = round((seller.cost[pid] + buyer.utility[pid][0]) / 200, 2)

    def act(agent, tool, args, expected_ok, expected_error=None):
        result = market.apply(agent, tool, args).as_dict()
        print(f'  {tool} {args} -> {result}', flush=True)
        if result['ok'] != expected_ok or (expected_error and result.get('error') != expected_error):
            raise RuntimeError(f'Unexpected tool result: {result}')

    def show_scores():
        scores = {r['agent']: r['raw'] for r in score_market(market)}
        print(f"  seller profit=${scores[seller.id]:.2f}; buyer surplus=${scores[buyer.id]:.2f}")

    print('[1/7] A message is talk, not a listing or trade.')
    act(seller.id, 'send_message', {'to': buyer.id, 'message': 'P72 special price!'}, True)
    if market.trades or market.listings:
        raise RuntimeError('Message unexpectedly changed trades/listings')
    print('[2/7] A board price becomes binding after end_turn().')
    act(seller.id, 'set_price', {'product_id': pid, 'price': price}, True)
    purchase = {'seller_id': seller.id, 'product_id': pid, 'qty': 1}
    act(buyer.id, 'buy_listed', purchase, False, 'not_listed')
    market.end_turn()
    market.tick += 1
    print('[3/7] Buy the first unit: check cash, stock and both scores.')
    act(buyer.id, 'buy_listed', purchase, True)
    show_scores()
    print('[4/7] Buy it again: allowed by default, but second-unit utility is zero.')
    before = buyer.gained_utility
    market.tick += 1
    act(buyer.id, 'buy_listed', purchase, True)
    if buyer.gained_utility != before:
        raise RuntimeError('Duplicate unit unexpectedly gained utility')
    show_scores()
    print('[5/7] An unaffordable purchase is rejected without transferring cash/stock.')
    act(seller.id, 'set_price', {'product_id': pid, 'price': buyer.cash / 100 + 1}, True)
    market.end_turn()
    market.tick += 1
    snapshot = (buyer.cash, seller.stock[pid], len(market.trades))
    act(buyer.id, 'buy_listed', purchase, False, 'insufficient_cash')
    if snapshot != (buyer.cash, seller.stock[pid], len(market.trades)):
        raise RuntimeError('Rejected trade changed money/stock/trades')
    print('[6/7] Private offer: grade is hidden; delivery uses the actual low-grade utility.')
    quality_seller = next(s for s in market.sellers.values() if s.stock.get('P22', 0))
    act(quality_seller.id, 'make_offer', {'buyer_id': buyer.id, 'product_id': 'P22',
        'price': 3, 'qty': 1, 'grade': 'low', 'message': 'Top quality!'}, True)
    offer = list(market.offers.values())[-1]
    visible = market.observe(buyer.id, turns_left=2)['offers_to_you']
    if any('grade' in o for o in visible):
        raise RuntimeError('Offer leaked hidden grade')
    before = buyer.gained_utility
    act(buyer.id, 'accept_offer', {'offer_id': offer.id}, True)
    gain = buyer.gained_utility - before
    expected = round(buyer.utility['P22'][0] * schema.low_utility)
    if gain != expected or market.trades[-1].grade != 'low':
        raise RuntimeError('Low-grade delivery/utility mismatch')
    print(f'  P22 delivered LOW despite message; actual utility=${gain/100:.2f}, paid=$3.00')
    print('[7/7] A private offer expires at the end of the next round.')
    act(seller.id, 'make_offer', {'buyer_id': buyer.id, 'product_id': pid,
                                 'price': price, 'qty': 1}, True)
    offer = list(market.offers.values())[-1]
    market.end_round()
    if offer.status != 'open':
        raise RuntimeError('Offer expired too early')
    market.round = 2
    market.end_round()
    act(buyer.id, 'accept_offer', {'offer_id': offer.id}, False, 'offer_closed')
    print('PASS: controlled CPU tool demo. This is a chosen-action walkthrough, not an LLM run.')
    return 0


def read_new(path: Path, offsets: dict):
    """Consume complete appended lines only; leave partial JSON for the next poll."""
    if not path.exists():
        return []
    lines = []
    with path.open(encoding='utf-8', errors='replace') as stream:
        stream.seek(offsets.get(path, 0))
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith('\n'):
                stream.seek(start)
                break
            lines.append(line.rstrip('\n'))
        offsets[path] = stream.tell()
    return lines


def duration(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(max(0, int(seconds))))


def run_test(args) -> int:
    import yaml
    from adlmarket.casting import plan_markets
    from adlmarket.setups import market_run
    from training.local_resources import gpu_models, output_parent, query_gpus, runtime_cache, select_gpus

    source = args.config.resolve() if args.config else ROOT / f'configs/basic_{args.role}.yaml'
    config = yaml.safe_load(source.read_text())
    loaded = market_run(source, root=ROOT)
    plans = plan_markets(loaded.config.casting, loaded.sellers, loaded.buyers,
                         loaded.config.seed, pads=loaded.pads)
    rounds = loaded.config.market.rounds
    total = sum(len(p.sellers) + len(p.buyers) for p in plans) * rounds.total * rounds.turns_per_round
    devices = []
    if args.mode == 'gpu':
        devices, skipped = select_gpus(query_gpus(), args.gpu, args.gpu_mem,
                                       args.reserve_gib, args.min_gpu_gib)
        for row in skipped:
            print(f'Skip GPU {row["index"]}: {row["reason"]}', flush=True)
        for i, row in enumerate(devices):
            print(f'GPU {row["index"]}: free {row["free_gib"]:.2f}/{row["total_gib"]:.2f} GiB; '
                  f'vLLM budget {row["budget_gib"]:.2f} GiB ({row["fraction"]:.3f}); '
                  f'server {i} -> port {args.port+i}', flush=True)
        print('Memory planned from current free VRAM; no live expansion or reclamation. '
              'Other jobs can still change available memory before allocation.', flush=True)
    parent, storage_note = output_parent(ROOT, args.output_dir)
    if storage_note:
        print(f'STORAGE: {storage_note}', flush=True)
    cache_env = {}
    if args.mode == 'gpu':
        cache_env, cache_note = runtime_cache(ROOT, args.cache_dir)
        if cache_note:
            print(f'CACHE STORAGE: {cache_note}', flush=True)
        print(f'Compilation cache: {Path(cache_env["VLLM_CACHE_ROOT"]).parent}; '
              f'temporary files: {cache_env["TMPDIR"]}', flush=True)
    session = Path(tempfile.mkdtemp(prefix=f'{time.strftime("%Y%m%d-%H%M%S")}_{args.mode}_{args.role}_',
                                   dir=parent))
    config['output_dir'] = str(session)
    models = yaml.safe_load((ROOT / 'configs/models.yaml').read_text())
    # Isolated derived models config: no edits to course configs.
    if args.mode == 'gpu':
        models = gpu_models(models, devices, args.port)
    load_balance = None
    if args.mode == 'gpu' and len(devices) == 2:
        from training.market_load_balance import balance_ta_routes
        load_balance = balance_ta_routes(config, loaded, models, session, ROOT,
                                        devices, args.port, args.load_profile)
    # A scripted-only opponent needs no model server. Do not reserve an idle GPU.
    if args.mode == 'gpu':
        used = {str(a.model) for a in loaded.everyone() if a.controller == 'llm'}
        used_ta = any('ta_' in name for name in used)
        if not used_ta and 'ta' in models['serving']:
            if len(used) > 1:
                import copy
                urls = [f'http://localhost:{args.port+i}/v1' for i in range(2)]
                models['serving'] = {'serve0': models['serving']['base'],
                                     'serve1': models['serving']['ta']}
                for i in range(2):
                    models['models'][f'serve{i}'] = copy.deepcopy(models['models']['base'])
                    models['models'][f'serve{i}']['base_url'] = urls[i]
                models['models']['base']['replicas'] = urls
            else:
                del models['serving']['ta']
    model_path = session / 'models.yaml'
    model_path.write_text(yaml.safe_dump(models, sort_keys=False))
    config['models'] = str(model_path)
    effective = session / 'input.yaml'
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    # Validate actual adapter-to-endpoint routes, not just the raw server configuration.
    resolved = market_run(effective, root=ROOT)
    from training.market_load_balance import validate_equivalent
    validate_equivalent(loaded, resolved)
    routes = {agent.id: resolved.models[agent.model].url_for(resolved.models[agent.model].model)
              for agent in resolved.sellers + resolved.buyers if agent.controller == 'llm'}
    (session / 'resource_plan.json').write_text(json.dumps(
        {'devices': devices, 'routes': routes, 'result_storage': str(session),
         'plan_only': args.plan_only, 'runtime_cache_env': cache_env,
         'load_balance': load_balance}, indent=2))
    if devices:
        print(f'Agent routes: {dict(Counter(routes.values()))}', flush=True)
        if load_balance and 'estimated_output_tokens_per_tick' in load_balance:
            print('Estimated output tokens/tick by server: '
                  f'{[round(n, 1) for n in load_balance["estimated_output_tokens_per_tick"]]}; '
                  f'profile={load_balance["profile"] or "role/thinking priors"}', flush=True)
            for aid, index in load_balance['assignments'].items():
                if routes[aid] != f'http://localhost:{args.port+index}/v1':
                    raise ValueError(f'Actual adapter route differs from load plan: {aid}')
    command = [sys.executable, '-u', str(ROOT / 'run_market.py'), '--config', str(effective)]
    command += ['--mock-llm'] if args.mode == 'mock' else ['--launch-vllm']
    print(f'Mode={args.mode}; role={args.role}; {len(plans)} market; '
          f'{len(loaded.sellers)}x{len(loaded.buyers)}; '
          f'{rounds.total * rounds.turns_per_round} turns; {total} actions', flush=True)
    print(f'Logs/output: {session}', flush=True)
    if args.plan_only:
        print('PASS: plan/config/adapter routes checked; no vLLM or model inference started.')
        return 0
    print('Startup ETA: estimating; market ETA starts after observed actions.', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.update(cache_env)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    offsets = {}
    started = time.monotonic()
    play_start = None
    count = 0
    position = 'initializing / loading / warm-up'
    last_draw = -10
    run_dir = None
    with (session / 'runner.log').open('w') as log, (session / 'progress.log').open('w') as progress:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                done = process.poll() is not None
                for line in read_new(session / 'runner.log', offsets):
                    print(line, flush=True)
                dirs = list(session.glob('run_market_*'))
                if dirs:
                    run_dir = dirs[0]
                    for path in sorted(run_dir.glob('markets/*/events.jsonl')):
                        for line in read_new(path, offsets):
                            row = json.loads(line)
                            if row.get('type') == 'market_start':
                                play_start = play_start or time.monotonic()
                            if row.get('type') == 'turn':
                                count += 1
                                position = (f'{path.parent.name} round {row["round"]}/{rounds.total} '
                                            f'turn {(row["tick"] - 1) % rounds.turns_per_round + 1}'
                                            f'/{rounds.turns_per_round}')
                    # Show the latest server status during startup, without flooding terminal.
                    for path in run_dir.glob('vllm/*.log'):
                        lines = read_new(path, offsets)
                        if lines and not count:
                            print(f'[vLLM] {lines[-1][-220:]}', flush=True)
                now = time.monotonic()
                if done or now - last_draw >= 5:
                    elapsed = now - started
                    measured = now - play_start if play_start else 0
                    speed = count / measured if count and measured >= 5 else 0
                    speed_text = f'{speed:.2f}' if speed else 'estimating'
                    eta = '00:00:00' if count == total else (
                        duration((total - count) / speed) if speed else 'estimating')
                    filled = min(24, int(24 * count / total))
                    bar = '#' * filled + '-' * (24 - filled)
                    text = (f'[{bar}] {count}/{total} {count/total:.1%} | {position} | '
                            f'elapsed {duration(elapsed)} | {speed_text} actions/s | '
                            f'market ETA {eta}')
                    if count == total:
                        text += ' | finalizing / server shutdown'
                    print(text, flush=True)
                    progress.write(text + '\n')
                    progress.flush()
                    last_draw = now
                if done:
                    break
                time.sleep(1)
        except BaseException:
            print('\nStopping test and its process group; allowing up to 45s for cleanup.', flush=True)
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if process.returncode:
        print(f'FAIL: runner exit={process.returncode}; see {session / "runner.log"}')
        # The engine's traceback tail can hide the actionable startup error.
        sources = [session / 'runner.log', *run_dir.glob('vllm/*.log')] if run_dir else [session / 'runner.log']
        for path in sources:
            for line in path.read_text(errors='replace').splitlines():
                if any(marker in line for marker in ('No space left on device',
                                                     'Free memory on device',
                                                     'CUDA out of memory',
                                                     'maximum number of tokens that can be stored')):
                    print(f'ROOT CAUSE [{path.name}]: {line[-1000:]}')
                    break
        return process.returncode
    if run_dir is None:
        raise RuntimeError('No run directory produced')
    print(f'Completed. Run folder: {run_dir}', flush=True)
    return audit(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight', help='CPU environment/config/asset checks only')
    sub.add_parser('demo', help='CPU walkthrough of messages/listings/trades and scoring')
    report = sub.add_parser('report', help='Audit an already completed basic test')
    report.add_argument('run_dir', type=Path)
    run = sub.add_parser('run', help='Run one basic market with live progress')
    run.add_argument('--mode', choices=['mock', 'gpu'], required=True)
    run.add_argument('--role', choices=['buyer', 'seller'], required=True)
    run.add_argument('--config', type=Path, help='Alternative market config, e.g. quality stress/self-play')
    run.add_argument('--gpu', '--gpus', default='auto',
                     help='Physical index, 0,1 for two replicas, or auto (up to two usable GPUs)')
    run.add_argument('--port', type=int, default=8000)
    run.add_argument('--gpu-mem', default='auto',
                     help='auto or maximum fraction of total VRAM; always bounded by free VRAM minus reserve')
    run.add_argument('--reserve-gib', type=float, default=2.0)
    run.add_argument('--min-gpu-gib', type=float, default=14.0,
                     help='Skip GPUs below this vLLM budget; empirical floor, not a fit guarantee')
    run.add_argument('--output-dir', type=Path,
                     help='Result parent; default project results, falling back to /tmp if disk is full')
    run.add_argument('--cache-dir', type=Path,
                     help='Compilation/temp parent; default project .cache/runtime, falling back to /tmp')
    run.add_argument('--load-profile', type=Path,
                     help='Finished real run directory; use observed output tokens to balance TA traffic')
    run.add_argument('--plan-only', action='store_true',
                     help='Query resources and validate configs/routes, without starting vLLM')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'preflight':
        return preflight()
    if args.command == 'demo':
        return demo()
    if args.command == 'report':
        return audit(args.run_dir.resolve())
    return run_test(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
