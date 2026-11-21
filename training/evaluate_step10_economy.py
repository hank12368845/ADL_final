#!/usr/bin/env python3
"""User-launched official Phase2 economy; fixed surgical step10, no training."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adlmarket.casting import plan_markets
from adlmarket.setups import economy_run
from training.basic_checks import read_new, duration
from training.compare_buyer_markets import CANDIDATE
from training.diagnose_step84_market import check_sources, sha
from training.decision_ablation import fingerprint
from training.step84_market_report import IDS, cents, unit_ledger, turn_stats, stock_error_kind

REFERENCE = ROOT/'results/basic_checks/20261120-225505_buyer_pair_6xfah1ox/step10/suite.json'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def seating(loaded):
    return plan_markets(loaded.config.casting, loaded.sellers, loaded.buyers,
                        loaded.config.seed, pads=loaded.pads)


def plan_signature(loaded):
    """Include persona grouping, which affects official partitioning and padding."""
    rows = [dict(market=p.market_id, economy=p.repeat, rotation=p.rotation, seed_id=p.seed_id,
                 sellers=[a.id for a in p.sellers], buyers=[a.id for a in p.buyers]) for p in seating(loaded)]
    facts = dict(seed=loaded.config.seed, market=loaded.config.market.model_dump(mode='json'), plan=rows,
                 agents=[dict(id=a.id, owner=a.owner, role=a.role, variant_of=a.variant_of,
                              reasoning=a.reasoning, persona=a.persona_body()) for a in loaded.everyone()])
    return hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), rows


def validate(loaded, manifest):
    plans = seating(loaded)
    official, _ = economy_run(ROOT/'configs/economy.yaml', 'buyer', root=ROOT)
    if loaded.config.market != official.config.market or loaded.config.grading != official.config.grading:
        raise ValueError('Official phase2 rules or anchors changed')
    if len(plans) != 15 or loaded.config.casting.repeats != 3 or not loaded.config.casting.rotate:
        raise ValueError('Expected3 economies x5 rotations')
    for p in plans:
        if (len(p.sellers), len(p.buyers)) != (5, 10):
            raise ValueError('Expected5x10 seats in every market')
        if {a.id for a in p.buyers if a.owner != 'ta'} != set(IDS):
            raise ValueError('Missing or unexpected student personas')
    for a in loaded.buyers:
        if a.owner == 'ta':
            continue
        spec = loaded.models[a.model]
        if a.id not in IDS or not a.reasoning or a.persona_body() != manifest['personas'][a.id[-1]].strip():
            raise ValueError('Student persona or thinking setting changed')
        if not spec.lora or Path(spec.lora.path).resolve() != CANDIDATE.resolve():
            raise ValueError('Student is not surgical step10')
        if (spec.temperature, spec.top_p, spec.max_tokens) != (.7, .95, 8192):
            raise ValueError('Sampling settings changed')
    return plans


def prepare(folder, seed=3):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REFERENCE.read_text()); check_sources(ref)
    snapshot = ROOT/'.cache/economy_eval'/folder.name
    snapshot.mkdir(parents=True, exist_ok=False)
    buyer = snapshot/'buyer'; buyer.mkdir()
    for i, text in ref['personas'].items():
        (buyer/f'persona_{i}.md').write_text(text)
    (buyer/'agent.yaml').write_text(yaml.safe_dump(dict(id='example_student_buyer', owner='example_student',
        role='buyer', controller='llm', model='base', reasoning=True, lora=str(CANDIDATE)), sort_keys=False))
    cfg = yaml.safe_load((ROOT/'configs/economy.yaml').read_text())
    cfg.update(name='surgical_step10_official_phase2', seed=seed, output_dir=str(folder),
               models=str(ROOT/'configs/models.yaml'), save_trajectories=True, save_prompts=True)
    cfg['agents']['student'] = dict(sellers=[], buyers=[str(buyer.relative_to(ROOT))])
    config = folder/'economy.yaml'; config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaded, _ = economy_run(config, 'buyer', root=ROOT)
    manifest = dict(experiment='step10_official_phase2_economy_v1', adapter=str(CANDIDATE),
        adapter_fingerprints=fingerprint(CANDIDATE), seed=seed, phase=2, snapshot=str(snapshot),
        config=str(config), personas=ref['personas'], versions=ref['versions'], revision=ref['revision'],
        sources={**ref['sources'], **ref['inputs'], str(Path(__file__).resolve()):sha(__file__),
                 str(ROOT/'run_economy.py'):sha(ROOT/'run_economy.py')},
        inputs={str(p):sha(p) for p in [config, *buyer.iterdir()]},
        adapters={str(Path(loaded.models[a.model].lora.path).resolve()):fingerprint(loaded.models[a.model].lora.path)
                  for a in loaded.everyone() if loaded.models[a.model].lora},
        training_updates=0, automatic_promotion=False, total_markets=15,
        total_actions=6750, target_actions_per_persona=450)
    validate(loaded, manifest)
    manifest['plan_signature'], manifest['markets'] = plan_signature(loaded)
    write_json(folder/'suite.json', manifest); check_sources(manifest)
    return manifest


def route_tas(cfg, loaded, models, snapshot, devices, port):
    """Inline TA model replicas preserve variant_of and official seat assignments.

    Agent-level LoRA path deduplication would otherwise pin both replicas to one GPU.
    Inline ModelSpecs have unique serving names but keep the exact original weights.
    """
    from training.market_load_balance import profile_costs
    plans = seating(loaded)
    profile_run = Path(json.loads((REFERENCE.parent/'wrapper_result.json').read_text())['run_dir'])
    measured, _ = profile_costs(profile_run)
    agents = {a.id:a for a in loaded.everyone()}
    appearances = Counter()
    for p in plans:
        for a in p.sellers+p.buyers:
            template=next((aid for aid in agents if a.id==aid or
                           a.id.startswith(aid+'_') and a.id[len(aid)+1:].isdigit()), None)
            if template is None:raise ValueError(f'Unknown official padding template:{a.id}')
            appearances[template]+=1
    costs = {aid:appearances.get(aid, 15)*measured.get(aid, 500 if a.role=='buyer' else 100)
             for aid,a in agents.items()}
    assignment = {aid:0 for aid,a in agents.items() if a.owner!='ta'}
    loads = [sum(costs[aid] for aid in assignment)] + ([0] if len(devices)==2 else [])
    for aid in sorted((i for i,a in agents.items() if a.owner=='ta'), key=lambda i:(-costs[i],i)):
        index = min(range(len(loads)), key=lambda i:(loads[i],i))
        assignment[aid] = index; loads[index] += costs[aid]
    paths = {'seller':[], 'buyer':[]}
    routing = Path(snapshot)/'routing_agents'; routing.mkdir(exist_ok=True)
    for aid,a in agents.items():
        if a.owner!='ta':continue
        index = assignment[aid]
        data = a.model_dump(mode='json', exclude_none=True)
        for key in ['source','persona','personas','lora']:data.pop(key, None)
        data['persona_text'] = a.persona_body()
        spec = loaded.models[a.model].model_copy(deep=True)
        spec.model = f'{spec.model}-economy-replica-{index}'
        spec.base_url = f'http://localhost:{port+index}/v1'; spec.replicas = []
        data['model'] = spec.model_dump(mode='json', exclude_none=True)
        path = routing/f'{aid}.yaml'; path.write_text(yaml.safe_dump(data, sort_keys=False))
        paths[a.role].append(str(path.relative_to(ROOT)))
    cfg['agents']['ta'] = dict(sellers=paths['seller'], buyers=paths['buyer'])
    return dict(assignments=assignment, estimated_output_token_load=loads, profile=str(profile_run),
                method='measured_generation_cost_with_official_seat_frequency',
                note='Inline TA model replicas retain original variant_of, IDs, personas and LoRA tensors.')


def configure(folder, manifest, selection, port, mock=False):
    from training.local_resources import query_gpus, select_gpus, gpu_models, runtime_cache
    from training.market_load_balance import validate_equivalent
    free=shutil.disk_usage(folder).free/1024**3
    print(f'Result storage free:{free:.1f}GiB; output:{folder}',flush=True)
    if free<2:raise RuntimeError('Less than2GiB free at the result directory; no GPU service started')
    cfg = yaml.safe_load(Path(manifest['config']).read_text()); before, _ = economy_run(manifest['config'], 'buyer', root=ROOT)
    devices, skipped, cache = [], [], {}
    if not mock:
        devices, skipped = select_gpus(query_gpus(), selection, reserve_gib=2., min_budget_gib=14.)
        print(f'GPU startup plan:{devices}; skipped:{skipped}', flush=True)
        if len(devices)==1:print('SINGLE-GPU FALLBACK; no other jobs stopped.', flush=True)
        cache, note = runtime_cache(ROOT)
        if note:print(note, flush=True)
    source = yaml.safe_load((ROOT/'configs/models.yaml').read_text())
    # Mock still exercises two-endpoint routing, without querying/starting any GPU.
    planning = devices or [dict(index='0',fraction=.8),dict(index='1',fraction=.8)]
    models = gpu_models(source, planning, port)
    for server in models['serving'].values():
        server['extra_args'] += ['--revision',manifest['revision'],'--tokenizer-revision',manifest['revision']]
    balance = route_tas(cfg,before,models,manifest['snapshot'],planning,port)
    model_path = folder/'models.yaml'; model_path.write_text(yaml.safe_dump(models, sort_keys=False))
    cfg['models'] = str(model_path)
    effective = folder/'effective_economy.yaml'; effective.write_text(yaml.safe_dump(cfg, sort_keys=False))
    after, _ = economy_run(effective,'buyer',root=ROOT)
    validate(after,manifest); validate_equivalent(before,after)
    if plan_signature(after)[0] != manifest['plan_signature']:
        raise ValueError('GPU routing changed official seating/grouping')
    # validate_equivalent does not include padding; explicitly verify its behavior too.
    left={a.id:a for a in before.everyone()}; right={a.id:a for a in after.everyone()}
    for aid,a in left.items():
        b=right[aid]; ma,mb=before.models[a.model],after.models[b.model]
        if a.variant_of!=b.variant_of or a.persona_body()!=b.persona_body() or ma.lora!=mb.lora:
            raise ValueError(f'Routing changed grouping/persona/weights:{aid}')
    routes={a.id:after.models[a.model].url_for(after.models[a.model].model) for a in after.everyone()}
    if len(planning)==2 and len(set(routes.values()))!=2:raise ValueError('Second GPU has no routes')
    resource=dict(devices=devices,skipped=skipped,routes=routes,balance=balance,mock=mock)
    write_json(folder/'resource_plan.json',resource)
    manifest['runtime_inputs']={str(p):sha(p) for p in [effective,model_path,*Path(manifest['snapshot']).joinpath('routing_agents').glob('*.yaml')]}
    write_json(folder/'suite.json',manifest)
    print(f'Official15-market seating verified; endpoint pool counts:{dict(Counter(routes.values()))}',flush=True)
    return effective,cache


class Progress:
    def __init__(self):self.offsets={};self.states={};self.files={}
    def scan(self, run):
        for path in sorted(Path(run).glob('markets/*/events.jsonl')):
            stat=path.stat(); old=self.files.get(path)
            if old and (old[0]!=stat.st_ino or stat.st_size<self.offsets.get(path,0)):
                self.offsets.pop(path,None);self.states.pop(path,None)
            self.files[path]=(stat.st_ino,stat.st_size)
            state=self.states.setdefault(path,dict(count=0,targets=Counter(),round=0,tick=0))
            for line in read_new(path,self.offsets):
                r=json.loads(line)
                if r.get('type')!='turn':continue
                state['count']+=1;state['round']=r['round'];state['tick']=r['tick']
                if r['agent'] in IDS:state['targets'][r['agent']]+=1
        return sum(s['count'] for s in self.states.values())


def play(folder,manifest,effective,cache,mock=False):
    runs=list(folder.glob('run_economy_*'))
    if len(runs)>1:raise ValueError('Multiple economy outputs')
    run=runs[0] if runs else None
    if run and json.loads((run/'metadata.json').read_text()).get('mock_llm')!=mock:
        raise ValueError('Cannot mix mock and real on resume')
    command=[sys.executable,'-u',str(ROOT/'run_economy.py')]
    command += ['--resume',str(run)] if run else ['--phase','2','--config',str(effective)]
    command += ['--mock-llm' if mock else '--launch-vllm']
    check_sources(manifest)
    for path,digest in manifest['runtime_inputs'].items():
        if sha(path)!=digest:raise ValueError(f'Runtime input changed:{path}')
    if run and json.loads((run/'metadata.json').read_text()).get('status')=='finished':
        print('Already finished; report only.',flush=True);return run
    # Do not activate the HF training-kernel overlay in the official vLLM environment.
    env={**os.environ,**cache,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    overlay=str(ROOT/'.cache/fast_kernels/site-packages')
    env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and p!=overlay)
    start=time.monotonic();last=0;tracker=Progress();base=None;first=None;log_offsets={}
    logfile=folder/'runner.log'
    if logfile.exists():log_offsets[logfile]=logfile.stat().st_size
    with logfile.open('a') as log,(folder/'progress.log').open('a') as progress:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(logfile,log_offsets):print(line,flush=True)
                if run is None:
                    found=list(folder.glob('run_economy_*'));run=found[0] if found else None
                if run:
                    total=tracker.scan(run)
                    if base is None:base=total
                    if total>base and first is None:first=time.monotonic()
                    finished=len(list(run.glob('markets/*/scores.json')))
                else:total=finished=0
                now=time.monotonic()
                if done or now-last>=5:
                    speed=(total-(base or 0))/(now-first) if first and now-first>=5 else 0
                    eta=duration((6750-total)/speed) if speed else 'estimating'
                    active=[(p,s) for p,s in tracker.states.items() if not (p.parent/'scores.json').exists()]
                    position='loading / warm-up'
                    if active:
                        p,s=active[-1];position=f'{p.parent.name} {s["count"]}/450 r{s["round"]}/5 t{(s["tick"]-1)%6+1}/6'
                    targets=Counter()
                    for s in tracker.states.values():targets.update(s['targets'])
                    filled=min(24,total*24//6750)
                    line=(f'[{"#"*filled}{"-"*(24-filled)}] {total}/6750 | markets {finished}/15 | {position} | '
                          + ' '.join(f'p{i}:{targets[IDS[i-1]]}/450' for i in (1,2,3))
                          +f' | elapsed {duration(now-start)} | {speed:.2f} actions/s | ETA {eta}')
                    print(line,flush=True);progress.write(line+'\n');progress.flush();last=now
                    if run and total==0:
                        for p in run.glob('vllm/*.log'):
                            lines=read_new(p,log_offsets)
                            if lines:print(f'[vLLM {p.name}] {lines[-1][-200:]}',flush=True)
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    write_json(folder/'wrapper_result.json',dict(exit=proc.returncode,elapsed_seconds=time.monotonic()-start,
        run_dir=str(run) if run else None,mock=mock,actions=total,training_updates=0))
    if proc.returncode or run is None or total!=6750:raise RuntimeError(f'Economy incomplete; inspect {logfile}; use --resume {folder}')
    return run


def report(folder,run):
    manifest=json.loads((folder/'suite.json').read_text());check_sources(manifest)
    meta=json.loads((run/'metadata.json').read_text())
    if meta.get('status')!='finished' or meta.get('script')!='run_economy.py' or meta.get('phase')!=2:
        raise ValueError('Not a finished official phase2 economy')
    loaded,_=economy_run(run/'config.yaml','buyer',root=ROOT);plans=validate(loaded,manifest)
    if plan_signature(loaded)[0]!=manifest['plan_signature']:raise ValueError('Actual run plan changed')
    records={aid:[] for aid in IDS};load=defaultdict(Counter); total=0;infrastructure_failures=Counter()
    for p in plans:
        directory=run/'markets'/p.market_id;final=json.loads((directory/'final.json').read_text())
        scores={s['agent']:s for s in json.loads((directory/'scores.json').read_text())}
        infrastructure_failures.update({aid:s.get('model_failures',0) for aid,s in scores.items()})
        if {b['id'] for b in final['buyers']}!={a.id for a in p.buyers} or {s['id'] for s in final['sellers']}!={a.id for a in p.sellers}:
            raise ValueError('Final market actors differ from official plan')
        actions=defaultdict(list);seen=Counter()
        with (directory/'events.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r.get('type')!='turn':continue
                total+=1;seen[(r['agent'],r['tick'])]+=1;log=r.get('llm') or {}
                if log.get('base_url'):
                    c=load[log['base_url']];c['calls']+=1;c['output_tokens']+=log.get('output_tokens_spent',0)
                if r['agent'] in IDS:actions[r['agent']].append(r)
        # Full actor/tick coverage is checked from the event file independently of scores.
        if len(seen)!=450 or set(seen.values())!={1}:raise ValueError(f'Incomplete/duplicate actions:{p.market_id}')
        if {aid for aid,t in seen}!={a.id for a in p.sellers+p.buyers} or {t for aid,t in seen}!=set(range(1,31)):
            raise ValueError('Wrong actor/tick coverage')
        for b in final['buyers']:
            aid=b['id']
            if aid not in IDS:continue
            rows=sorted(actions[aid],key=lambda r:r['tick']);units=unit_ledger([t for t in final['trades'] if t['buyer']==aid],b['utility'],final['schema']['low_utility'])
            spent=sum(u['price_cents'] for u in units);gained=sum(u['utility_cents'] for u in units);net=gained-spent
            if spent!=b['spent'] or gained!=b['gained_utility'] or b['cash']!=b['budget']-spent or cents(scores[aid]['raw'])!=net:
                raise ValueError(f'Ledger mismatch:{p.market_id}/{aid}')
            counts=Counter();longest=current=0;previous=None;failures=[]
            for r in rows:
                args=r.get('args') or {};obs=r['observation'];pair=(args.get('seller_id'),args.get('product_id'))
                board=next((o for o in obs['board'] if (o['seller'],o['product_id'])==pair and o['in_stock']>0),None)
                absent=r.get('tool')=='buy_listed' and board is None
                counts['absent_board_pairs']+=absent
                if r['result'].get('error')=='no_stock':counts[stock_error_kind(r.get('tool'),args,board,final['trades'],r['tick'])]+=1
                current=current+1 if absent and pair==previous else 1 if absent else 0
                longest=max(longest,current);previous=pair if absent else None
                if absent or (r.get('llm') or {}).get('stop_reason')=='length':
                    failures.append(dict(tick=r['tick'],tool=r.get('tool'),args=args,error=r['result'].get('error'),reasoning=(r.get('llm') or {}).get('reasoning')))
            for round_ in range(1,6):
                if sum((r.get('llm') or {}).get('output_tokens_spent',0) for r in rows if r['round']==round_)>8192:
                    raise ValueError('Round token budget exceeded')
            records[aid].append(dict(market=p.market_id,economy=p.repeat,rotation=p.rotation,surplus_cents=net,
                trades=len({u['trade_id'] for u in units}),unique_products=len({u['product'] for u in units}),
                zero_marginal_spend_cents=sum(u['price_cents'] for u in units if u['utility_cents']==0),
                negative_unit_loss_cents=sum(max(-u['net_cents'],0) for u in units),cash_left_cents=b['cash'],
                stats=turn_stats(rows),counts=dict(counts),longest_absent_pair_retry=longest,
                model_failures=scores[aid].get('model_failures',0),failures=failures))
    if total!=6750:raise ValueError('Wrong economy action count')
    summary={}
    for aid,rs in records.items():
        if len(rs)!=15:raise ValueError(f'Missing student market results:{aid}')
        values=[r['surplus_cents']/100 for r in rs]
        summary[aid]=dict(markets=len(rs),mean_surplus=statistics.mean(values),minimum=min(values),maximum=max(values),
            by_economy={str(e):statistics.mean(r['surplus_cents']/100 for r in rs if r['economy']==e) for e in range(3)},
            zero_marginal_spend=sum(r['zero_marginal_spend_cents'] for r in rs)/100,
            absent_board_pairs=sum(r['counts']['absent_board_pairs'] for r in rs),
            truncated_turns=sum(r['stats']['truncated_turns'] for r in rs),budget_waits=sum(r['stats']['budget_waits'] for r in rs),
            cash_errors=sum(r['stats']['cash_errors'] for r in rs),model_failures=sum(r['model_failures'] for r in rs),
            longest_absent_pair_retry=max(r['longest_absent_pair_retry'] for r in rs))
    real=not meta.get('mock_llm',False)
    infrastructure_clean=not any(infrastructure_failures.values())
    result=dict(complete=True,valid_real_evaluation=real and infrastructure_clean,
        infrastructure_clean=infrastructure_clean,infrastructure_failures=dict(infrastructure_failures),
        phase=2,seed=manifest['seed'],run_dir=str(run),
        total_markets=15,total_actions=total,summary=summary,markets=records,
        actual_server_load={k:dict(v) for k,v in load.items()},automatic_promotion=False,training_updates=0)
    write_json(folder/'economy_review.json',result)
    lines=['# Step10官方Phase2 economy評估','',f'真模型={real}；基礎設施無失敗={infrastructure_clean}；官方run_economy.py，3economies×5rotations，共15場／6750行動。',
           '原三persona、surgical step10、Thinking ON；公共商品、公開TA pool。',
           '','| Persona | 平均surplus $/場 | 最低/最高 $ | econ0/1/2均值 $ | 零效用總支出 $ | 錯掛牌 | 截斷 | 額度wait | 最長失效連試 |',
           '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for aid,s in summary.items():
        econ='/'.join(f'{s["by_economy"][str(e)]:.2f}' for e in range(3))
        lines.append(f'| {aid} | {s["mean_surplus"]:.2f} | {s["minimum"]:.2f}/{s["maximum"]:.2f} | {econ} | {s["zero_marginal_spend"]:.2f} | {s["absent_board_pairs"]} | {s["truncated_turns"]} | {s["budget_waits"]} | {s["longest_absent_pair_retry"]} |')
    lines+=['',f'官方standings原樣保留：{run}/standings.md；anchors與grade見standings.json。',
            '本地舊版student列彙整三persona；本報告逐persona列出，不冒充FAQ最佳persona正式評分或最終分數。',
            '三economies使用同一public商品檔，TA僅公開toy pool，無private商品／正式TA pool／同學對手；不能保證正式排名。',
            '所有rotation與economy均保留。跨rotation結果相關，單純15場不能當15個獨立樣本。',
            '零效用支出已含在surplus；失敗回合的機會成本不捏造金額。沒有自動訓練／部署／下一場。']
    (folder/'economy_review.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--seed',type=int,default=3);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--mock',action='store_true');parser.add_argument('--resume',type=Path)
    parser.add_argument('--report',type=Path);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.report:
        folder=args.report.resolve();runs=list(folder.glob('run_economy_*'))
        if len(runs)!=1:raise ValueError('Expected one economy run')
        return report(folder,runs[0])
    if args.resume:
        folder=args.resume.resolve();manifest=json.loads((folder/'suite.json').read_text());check_sources(manifest)
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,args.output_dir,min_free_gib=10.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_step10_economy_',dir=parent))
        print('Preparing frozen step10, original personas and official3x5 casting; ETA estimating.',flush=True)
        manifest=prepare(folder,args.seed)
    print(f'Prepared:{folder}; official Phase2,15 markets,6750 actions; no training.',flush=True)
    if args.prepare_only:print('PASS CPU only; no GPU query or service.');return folder
    runs=list(folder.glob('run_economy_*'))
    if runs and json.loads((runs[0]/'metadata.json').read_text()).get('mock_llm')!=args.mock:
        raise ValueError('Do not mix mock and real resume')
    if runs and json.loads((runs[0]/'metadata.json').read_text()).get('status')=='finished':
        result=report(folder,runs[0])
    else:
        effective,cache=configure(folder,manifest,args.gpus,args.port,args.mock)
        run=play(folder,manifest,effective,cache,args.mock);result=report(folder,run)
    print(f'Completed.Read {folder}/economy_review.md; official standings in {result["run_dir"]}; no automatic next GPU work.',flush=True)
    return result


if __name__=='__main__':main()
