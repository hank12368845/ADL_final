"""CPU-only independent audit of a completed official Seller baseline.

Keep the original frozen recipe/reports intact. Recheck actual engine acceptance
separately from our stricter useful-quote development criteria.
"""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def probe_market(case):
    from adlmarket.config import AgentSpec
    from adlmarket.market import Market,load_catalog
    from adlmarket.schema import build_items_schema
    from training.official_seller_data import current_spec
    spec=current_spec();catalog=load_catalog(ROOT/spec.catalog)
    schema=build_items_schema('seller_native_dev',spec.schema_,spec.schema_.items_file(0),catalog,ROOT,5)
    sellers=[AgentSpec(id=f'probe_s{i}',owner='probe',role='seller',model='base') for i in range(5)]
    buyers=[AgentSpec(id=f'probe_b{i}',owner='ta',role='buyer',model='ta') for i in range(10)]
    market=Market.generate('seller_native_dev',spec,schema,{p.id:p for p in catalog},sellers,buyers,random.Random(20261122))
    market.round=1;market.tick=1;aid=case['situation']['your_id'];book=market.sellers[aid]
    if case['family']=='empty_stock':book.stock={p:0 for p in book.stock}
    if case['family']=='offer_limit':
        pid=max((p for p,n in book.stock.items() if n),key=lambda p:market.products[p].ref_price-book.cost[p])
        for i in range(8):
            result=market.apply(aid,'make_offer',dict(buyer_id=buyers[i].id,product_id=pid,price=market.products[pid].ref_price/100,qty=1))
            if not result.ok:raise ValueError('Probe setup failed')
    if market.situation(aid)!=case['situation']:raise ValueError('Reconstructed probe facts differ')
    return market,aid


def engine_review(case,decision):
    market,aid=probe_market(case);args=copy.deepcopy(decision['arguments']);state=args.pop('private_state','')
    if decision['truncated'] or not decision['tool']:
        return dict(engine_accepted=False,engine_error='truncated_or_no_single_tool',private_state_chars=len(state),memory_truncated=len(state)>1000,
                    quote_with_no_stock=False,executable_positive_quote=False)
    tool=decision['tool'];pid=args.get('product_id');quote=tool in ('set_price','make_offer')
    available=market.sellers[aid].stock.get(pid,0);quantity=args.get('qty',1)
    no_stock=quote and available<quantity
    demand={}
    if quote and pid in market.products:
        from adlmarket.tools import to_cents
        price=to_cents(args['price']);factor=market.schema.low_utility if args.get('grade')=='low' else 1.
        gains=[round(market._next_high_unit(b,pid)*factor)-price for b in market.buyers.values()]
        demand=dict(positive_surplus_buyers_if_delivered=sum(g>0 for g in gains),zero_surplus_buyers_if_delivered=sum(g==0 for g in gains),
                    note='Private engine facts for CPU analysis only, never inserted into model input; not actual purchase predictions.')
    result=market.apply(aid,tool,args)
    return dict(engine_accepted=result.ok,engine_error=result.error,private_state_chars=len(state),memory_truncated=len(state)>1000,
                quote_with_no_stock=no_stock,executable_positive_quote=quote and not no_stock and result.ok and decision['profitable_quote'],**demand)


def audit(run):
    import torch
    from safetensors.torch import load_file
    from adlmarket.config import check_adapter
    from training.train_official_seller import check_run
    from training.official_seller_data import audit_data,sha
    from training.official_seller_train import frozen_cases
    from training.decision_ablation import check_fingerprint
    run=Path(run).resolve();cfg=check_run(run);data,manifest=audit_data(cfg)
    wrapper=json.loads((run/'wrapper_result.json').read_text());finished=json.loads((run/'training_finished.json').read_text())
    if wrapper['exit'] or finished['step']!=cfg['max_steps']:raise ValueError('Training not complete')
    batches=json.loads((run/'batch_plan.json').read_text())['batches'];cases=json.loads((run/'native_cases.json').read_text())
    if cases!=frozen_cases(cfg,batches):raise ValueError('Evaluation situations or persona inputs changed')
    expected={c['name']:c for c in cases};seen=Counter();metrics=[]
    for rank in range(cfg['world_size']):
        trace=[json.loads(l) for l in (run/f'samples_rank{rank}.jsonl').read_text().splitlines()]
        want=[(step+1,data['train'][i]['id']) for step,batch in enumerate(batches) for i in batch[rank::cfg['world_size']]]
        if [(r['step'],r['id']) for r in trace]!=want:raise ValueError('Actual per-rank sample exposures differ from plan')
        seen.update(r['id'] for r in trace)
        rows=[json.loads(l) for l in (run/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows]!=list(range(1,cfg['max_steps']+1)):raise ValueError('Missing optimizer-step metrics')
        if any(not math.isfinite(r['completion_loss']) or not math.isfinite(r['lr']) for r in rows):raise ValueError('Nonfinite training')
        metrics.extend(rows)
    if len(seen)!=cfg['train_examples']:raise ValueError('Not all unique train samples were seen')
    checkpoints={};previous=None;weights_changed={}
    for step in cfg['checkpoint_steps']:
        cp=run/'checkpoints'/f'step_{step:04d}';complete=json.loads((cp/'checkpoint_complete.json').read_text())
        check_fingerprint(cp,complete['adapter_fingerprints'])
        check_adapter('independent_seller_audit',cp,cfg['base_model'],{'model.layers.':'model.language_model.layers.'})
        weights=load_file(str(cp/'adapter_model.safetensors'),device='cpu')
        if any(not torch.isfinite(v).all().item() for v in weights.values()):raise ValueError('Nonfinite adapter weights')
        state=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer_steps={int(v['step'].item()) if torch.is_tensor(v['step']) else int(v['step']) for v in state['optimizer']['state'].values()}
        if optimizer_steps!={step}:raise ValueError('Saved optimizer was not updated at each expected step')
        checkpoints[str(step)]=dict(tensors=len(weights),nonzero_B=sum(bool(torch.count_nonzero(v).item()) for k,v in weights.items() if 'lora_B' in k),
                                   optimizer_steps=sorted(optimizer_steps),finite=True,adapter_checks_pass=True)
        if previous is not None:weights_changed[str(step)]=sum(not torch.equal(v,previous[k]) for k,v in weights.items())
        previous=weights;del state
    results={}
    for step in [0,*cfg['checkpoint_steps']]:
        native=json.loads((run/f'native_step_{step:04d}.json').read_text());rows=native['decisions']
        if Counter(r['name'] for r in rows)!=Counter(expected.keys()):raise ValueError('Missing/duplicate autonomous decisions')
        reviewed=[]
        for r in rows:
            if r['kind'] not in ('current_rule','p1_v1_control'):continue
            result=engine_review(expected[r['name']],r)
            reviewed.append(dict(name=r['name'],persona=r['persona_index'],family=r['family'],kind=r['kind'],tool=r['tool'],arguments=r['arguments'],
                                 original_errors=r['errors'],text=r['text'],generated_tokens=r['generated_tokens'],**result))
        current=[r for r in reviewed if r['kind']=='current_rule']
        results[str(step)]=dict(original_summary=native['summary'],current_engine_accepts=sum(r['engine_accepted'] for r in current),
                               current_empty_stock_quotes=sum(r['quote_with_no_stock'] for r in current),
                               memory_truncations=sum(r['memory_truncated'] for r in reviewed),
                               total_generated_tokens=sum(r['generated_tokens'] for r in rows),
                               current_by_persona={str(i):dict(engine_accepts=sum(r['engine_accepted'] for r in current if r['persona']==i),
                                   useful_quotes=sum(r['executable_positive_quote'] for r in current if r['persona']==i),
                                   empty_stock_quotes=sum(r['quote_with_no_stock'] for r in current if r['persona']==i)) for i in (1,2,3)},
                               engine_reviewed_rows=reviewed)
    perflight=[json.loads((run/f'preflight_rank{r}.json').read_text()) for r in range(cfg['world_size'])]
    train_raw=[json.loads(l) for l in (ROOT/cfg['data_dir']/'train.jsonl').read_text().splitlines()]
    coverage=dict(actions=dict(Counter(r['label_action'] for r in train_raw)),all_stock_zero=sum(not any(r['observation']['you']['stock'].values()) for r in train_raw),
                  product_counts=dict(Counter(len(r['situation']['products']) for r in train_raw)))
    return dict(run=str(run),source_and_split_guards=True,frozen_native_inputs_match=True,world_size=cfg['world_size'],devices=cfg['devices'],gpu_selection=cfg['gpu_selection'],
                wrapper_seconds=wrapper['elapsed_seconds'],unique_examples=len(seen),total_exposures=sum(seen.values()),steps=finished['step'],warmup=cfg['warmup_steps'],
                checkpoints=checkpoints,changed_tensors_vs_previous_checkpoint=weights_changed,preflight=perflight,coverage=coverage,results=results,
                notes=['Original valid is a development guard score, not pure engine legality.',
                       'Engine accepts zero-stock set_price; with no inventory it cannot produce a sale. make_offer still rejects no_stock.',
                       'Formal LLMPolicy truncates private_state to1000 chars; an oversized note alone does not reject the market action.',
                       'Empty-stock probes are artificial round1 states with no history, not an observed natural market failure rate.',
                       'HF greedy1024 diagnostics differ from vLLM0.7/top_p0.95/8192 per round; no realized profit measured.'],
                automatic_training=False,automatic_promotion=False,automatic_gpu_work=False)


def write(run):
    run=Path(run).resolve();a=audit(run)
    (run/'independent_seller_training_audit.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    lines=['# Seller官方基線獨立審查','',f'完成{a["steps"]}步／{a["unique_examples"]}不同樣本／{a["total_exposures"]}曝光，總耗時{a["wrapper_seconds"]/60:.1f}分鐘；實際{a["world_size"]}張卡。',
           '來源、整市場切分、固定persona與評估題、逐rank曝光、optimizer步數、adapter格式／指紋／有限權重核對均通過。',
           '', '| step | held-out NLL | 有貨正毛利報價／9 | 官方未見守則／6 | 全部25題截斷 | 引擎接受／12 | 空庫存仍報價／3 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for step,v in a['results'].items():
        s=v['original_summary'];b=s['behavior']
        lines.append(f'| {step} | {s["completion_loss"]:.4f} | {b["current_rule"]["captured"]} | {b["official_heldout"]["valid"]} | {sum(r["truncated"] for r in b.values())} | {v["current_engine_accepts"]} | {v["current_empty_stock_quotes"]} |')
    lines+=['', 'Step46是收益待驗候選，不是已驗高分或操作完美。沒有全wait塌縮；有貨9題均能報正毛利價格，截斷下降。但全部庫存0的3題仍掛價，缺少「沒有可賣庫存就停止報價」的條件。',
            '原報表no_stock_or_excess_qty對set_price屬開發守則：引擎實際允許無貨掛牌，不能稱3題工具被拒絕。p1筆記1138chars會被LLMPolicy截為1000，並非整個工具因此失敗。原報表保留不改；獨立JSON逐題列引擎實際結果。',
            '這些全空庫存是人工round1無歷史壓力題，不能推成實戰25%錯誤；真實市場要量混合庫存、售罄後重掛、重複定價、高毛利曝光與成交。',
            'Train僅4個wait，且沒有全空庫存樣本，屬已確認覆蓋缺口；不能未做對照就斷言是唯一因果。24商品train只有1筆，官方未見6題也很小。',
            '正毛利掛價不保證成交：step46 p1 stocked題把P03掛2.00，這個控制世界的買家高品質效用也正好2.00，買家surplus為0。這僅為CPU讀引擎私有事實的診斷，未提供給模型，不硬編碼到persona或訓練。',
            '下一步保留step46＋starter對照，固定v2 persona在新seed完整5×10／30turn量profit；不因這些小題續epoch、換草稿格式或部署。',
            'GPU預檢已實際用到FLA／causal並完成最長反傳；無未加速對照，不宣稱快了幾倍。HF小題為greedy1024上限，不等於正式market設定。',
            '本次僅CPU核帳，未啟GPU或訓練。']
    (run/'independent_seller_training_review.md').write_text('\n'.join(lines)+'\n')
    return a


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=write(p.parse_args().run)
    print(json.dumps({k:a[k] for k in ['run','source_and_split_guards','unique_examples','total_exposures','steps','world_size','wrapper_seconds','coverage']},ensure_ascii=False,indent=2))
    print('Read independent_seller_training_review.md; CPU-only audit complete.')
