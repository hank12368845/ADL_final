#!/usr/bin/env python3
"""New-seed paired5x10 markets: official step84 versus surgical step10. No training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.diagnose_step84_market import prepare as prepare_baseline,configure_resources,check_sources,validate_run,sha
from training.prepare_buyer_surgical import PARENT
from training.step84_market_report import write_report,IDS

CANDIDATE=ROOT/'training/runs/20261120-203130_buyer_surgical_3hb_nseg/checkpoints/step_0010'


def initial_signature(loaded):
    from adlmarket.casting import plan_markets
    from adlmarket.market import Market,load_catalog
    from adlmarket.schema import build_items_schema
    seats=plan_markets(loaded.config.casting,loaded.sellers,loaded.buyers,loaded.config.seed,pads=loaded.pads)[0]
    rules=loaded.config.market;catalog=load_catalog(ROOT/rules.catalog)
    schema=build_items_schema('paired_initial',rules.schema_,rules.schema_.items_file(0),catalog,ROOT,5)
    market=Market.generate(seats.market_id,rules,schema,{p.id:p for p in catalog},seats.sellers,seats.buyers,
        random.Random(f'market:{loaded.config.seed}:{seats.seed_id}'),market_index=seats.index,rotation=seats.rotation,seed_id=seats.seed_id)
    facts=dict(sellers=[a.id for a in seats.sellers],buyers=[a.id for a in seats.buyers],
        situations={a.id:market.situation(a.id) for a in seats.sellers+seats.buyers},
        personas={a.id:dict(persona=a.persona_body(),reasoning=a.reasoning,role=a.role,owner=a.owner,controller=a.controller) for a in seats.sellers+seats.buyers})
    return hashlib.sha256(json.dumps(facts,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def prepare_arm(directory,label,seed):
    import yaml
    from adlmarket.setups import market_run
    from training.decision_ablation import fingerprint,check_fingerprint
    directory=Path(directory);m=prepare_baseline(directory,seed)
    adapter=PARENT if label=='step84' else CANDIDATE
    if label=='step10':
        done=json.loads((adapter/'checkpoint_complete.json').read_text())
        if done['step']!=10:raise ValueError('Candidate is not surgical step10')
        check_fingerprint(adapter,done['adapter_fingerprints'])
    spec=directory/'buyer/agent.yaml';agent=yaml.safe_load(spec.read_text());agent['lora']=str(adapter)
    spec.write_text(yaml.safe_dump(agent,sort_keys=False))
    loaded=market_run(Path(m['config']),root=ROOT)
    m.update(experiment='paired_buyer_market_v1',arm=label,adapter=str(adapter),adapter_fingerprints=fingerprint(adapter),
        description='One new-seed public market; identical opponents/personas/initial state across two separate arms; only target buyer adapter differs.')
    m['adapters']={str(Path(loaded.models[a.model].lora.path).resolve()):fingerprint(loaded.models[a.model].lora.path) for a in loaded.everyone() if loaded.models[a.model].lora}
    m['sources'][str(Path(__file__).resolve())]=sha(__file__)
    m['inputs']={str(p):sha(p) for p in directory.rglob('*') if p.is_file() and p.name!='suite.json'}
    validate_run(loaded,m);check_sources(m);m['initial_signature']=initial_signature(loaded)
    (directory/'suite.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    return m


def prepare_pair(folder,seed=1):
    if seed==0:raise ValueError('Seed0 was used for training; choose a new seed such as1')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arms={label:prepare_arm(folder/label,label,seed) for label in ('step84','step10')}
    if len({a['initial_signature'] for a in arms.values()})!=1:raise ValueError('Initial markets or personas differ')
    if arms['step84']['market_rules']!=arms['step10']['market_rules']:raise ValueError('Market rules differ')
    plan=dict(experiment='paired_buyer_market_v1',seed=seed,total_actions=900,total_target_actions=180,
        initial_signature=arms['step84']['initial_signature'],arms={label:str(folder/label) for label in arms},
        training_updates=0,automatic_promotion=False,source=str(Path(__file__).resolve()),source_sha256=sha(__file__))
    (folder/'compare_plan.json').write_text(json.dumps(plan,indent=2)+'\n');return plan


def play_arm(directory,manifest,effective,cache,mock=False,offset=0):
    """Reuse the tested formal runner; add overall900-action progress to its progress stream."""
    import os,signal,subprocess
    from collections import Counter
    from training.basic_checks import read_new,duration
    directory=Path(directory);check_sources(manifest)
    if any(sha(path)!=value for path,value in manifest['runtime_inputs'].items()):raise ValueError('Runtime inputs changed')
    cmd=[sys.executable,'-u',str(ROOT/'run_market.py'),'--config',str(effective),'--mock-llm' if mock else '--launch-vllm']
    env={**os.environ,**cache,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
    began=time.monotonic();first=None;last=0;offsets={};count=0;targets=Counter();run=None;position='loading / warm-up'
    with (directory/'runner.log').open('w') as log,(directory/'progress.log').open('w') as progress:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                done=proc.poll() is not None
                for line in read_new(directory/'runner.log',offsets):print(line,flush=True)
                runs=list(directory.glob('run_market_*'))
                if runs:
                    if len(runs)!=1:raise ValueError('More than one active market output')
                    run=runs[0]
                    for path in run.glob('markets/*/events.jsonl'):
                        for line in read_new(path,offsets):
                            r=json.loads(line)
                            if r.get('type')=='market_start':first=first or time.monotonic()
                            if r.get('type')=='turn':
                                count+=1
                                if r['agent'] in IDS:targets[r['agent']]+=1
                                position=f'round{r["round"]}/5 turn{(r["tick"]-1)%6+1}/6'
                    for path in run.glob('vllm/*.log'):
                        lines=read_new(path,offsets)
                        if lines and not count:print(f'[vLLM {path.name}] {lines[-1][-200:]}',flush=True)
                now=time.monotonic()
                if done or now-last>=5:
                    elapsed=now-began;speed=count/(now-first) if count and first and now-first>=5 else 0
                    eta=duration((450-count)/speed) if speed else 'estimating'
                    total_eta=duration((900-offset-count)/speed) if speed else 'estimating'
                    filled=min(24,count*24//450);bar='#'*filled+'-'*(24-filled)
                    text=(f'[{bar}] {manifest["arm"]} {count}/450 | overall {offset+count}/900 | {position} | '
                        f'p1:{targets[IDS[0]]}/30 p2:{targets[IDS[1]]}/30 p3:{targets[IDS[2]]}/30 | '
                        f'elapsed {duration(elapsed)} | {speed:.2f}actions/s | market ETA {eta} | overall ETA~{total_eta}')
                    print(text,flush=True);progress.write(text+'\n');progress.flush();last=now
                if done:break
                time.sleep(1)
        except BaseException:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    result=dict(exit=proc.returncode,elapsed_seconds=time.monotonic()-began,actions=count,target_actions=dict(targets),mock=mock,training_updates=0,run_dir=str(run) if run else None)
    (directory/'wrapper_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode or run is None or count!=450:raise RuntimeError(f'{manifest["arm"]} failed; read {directory}/runner.log')
    return run


def report_pair(folder):
    import yaml
    folder=Path(folder);plan=json.loads((folder/'compare_plan.json').read_text());reports={}
    for label,arm in plan['arms'].items():
        arm=Path(arm);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        info=json.loads((arm/'wrapper_result.json').read_text());r=write_report(arm,info['run_dir'])
        played=yaml.safe_load((Path(info['run_dir'])/'config.yaml').read_text())
        if played['seed']!=plan['seed']:raise ValueError('Actual market seed differs from matched plan')
        report_path=arm/'market_loss_review.md'
        report_path.write_text(report_path.read_text().replace('Step84完整市場戰損帳單',f'{label}完整市場戰損帳單').replace('原persona＋官方step84',f'原persona＋{label}'))
        reports[label]=r
    if any(not r['complete'] for r in reports.values()):raise ValueError('Incomplete market')
    real=all(r['valid_real_diagnostic'] for r in reports.values());by_persona={}
    lines=['# Step84 vs step10：新市場收益對照','',f'Seed={plan["seed"]};兩場5seller×10buyer，每場30turn。有效真模型對照={real}。',
        '商品、成本／效用、persona、對手、初始座位與seed相同；只有三個目標buyer的LoRA不同。後續市場依行為自然分歧。','',
        '| Persona | step84 surplus $ | step10 surplus $ | 差額 $ | 零效用支出84/10 $ | 錯掛牌84/10 | 截斷84/10 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for aid in IDS:
        a=reports['step84']['buyers'][aid];b=reports['step10']['buyers'][aid];delta=b['surplus_cents']-a['surplus_cents']
        by_persona[aid]=dict(step84=a,step10=b,delta_surplus_cents=delta)
        lines.append(f'| {aid} | {a["surplus_cents"]/100:.2f} | {b["surplus_cents"]/100:.2f} | {delta/100:+.2f} | {a["zero_marginal_spend_cents"]/100:.2f}/{b["zero_marginal_spend_cents"]/100:.2f} | {a["counts"].get("absent_board_pair_attempts",0)}/{b["counts"].get("absent_board_pair_attempts",0)} | {a["action_stats"]["truncated_turns"]}/{b["action_stats"]["truncated_turns"]} |')
    total={label:sum(r['buyers'][aid]['surplus_cents'] for aid in IDS) for label,r in reports.items()}
    lines+=['',f'三persona本場帳目合計：step84=${total["step84"]/100:.2f}，step10=${total["step10"]/100:.2f}；差額=${(total["step10"]-total["step84"])/100:+.2f}。',
        '合計不是課程persona評分彙整。單seed／單rotation／公開商品及toy對手，不能據此保證正式高分或全域優勝。',
        '各arm的market_loss_review.md/json保留逐筆成本、cash／held、競爭no_stock、token耗盡、雙卡實際負載及完整原文。',
        '零效用支出已包含於surplus，不能重複扣除；失敗回合的機會成本不能直接捏造成金額。',
        '沒有自動部署、訓練或額外市場。Mock結果不作模型成績。']
    result=dict(complete=True,valid_real_comparison=real,seed=plan['seed'],by_persona=by_persona,total_surplus_cents=total,
        delta_total_surplus_cents=total['step10']-total['step84'],automatic_promotion=False,automatic_training=False,
        actual_server_load={label:r['actual_server_load'] for label,r in reports.items()})
    (folder/'market_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (folder/'market_comparison.md').write_text('\n'.join(lines)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--output-dir',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--mock',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.report:return report_pair(a.report)
    if a.resume:
        folder=a.resume.resolve();plan=json.loads((folder/'compare_plan.json').read_text())
        if sha(plan['source'])!=plan['source_sha256']:raise ValueError('Comparison source changed')
    else:
        from training.local_resources import output_parent
        parent,note=output_parent(ROOT,a.output_dir,min_free_gib=5.)
        if note:print(note,flush=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_pair_',dir=parent));plan=prepare_pair(folder,a.seed)
    print(f'Prepared:{folder}; two matched full markets,900actions; no training.',flush=True)
    if a.prepare_only:print('PASS CPU only; no GPU query or service.');return folder
    if 'mock' in plan and plan['mock']!=a.mock:raise ValueError('Do not mix mock and real markets on resume')
    plan['mock']=a.mock;(folder/'compare_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for index,(label,arm) in enumerate(plan['arms'].items()):
        arm=Path(arm);m=json.loads((arm/'suite.json').read_text());check_sources(m)
        prior=json.loads((arm/'wrapper_result.json').read_text()) if (arm/'wrapper_result.json').exists() else {}
        if prior.get('exit')==0 and prior.get('actions')==450:
            print(f'Skip completed {label}:450/450; overall{(index+1)*450}/900',flush=True);continue
        for stale in list(arm.glob('run_market_*'))+([arm/'routing_agents'] if (arm/'routing_agents').exists() else []):
            stale.rename(stale.with_name(f'archived_{time.time_ns()}_{stale.name}'))
        effective,cache=configure_resources(arm,m,a.gpus,a.port,a.mock)
        print(f'Arm{index+1}/2:{label}; overall{index*450}/900',flush=True)
        play_arm(arm,m,effective,cache,a.mock,index*450)
    report_pair(folder)
    print(f'Completed.Read {folder}/market_comparison.md; no automatic training or deployment.',flush=True)
    return folder


if __name__=='__main__':main()
