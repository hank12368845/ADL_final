"""Market surplus and observed action losses; never changes engine scores."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean

OWNER='example_student'
IDS=[f'example_student_buyer_p{i}' for i in (1,2,3)]


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def cents(value):return int(Decimal(str(value))*100)


def visible_opportunities(obs,situation):
    """Visible, unowned, affordable, positive LOW-quality bound; not guaranteed fills."""
    varies={x['product_id'] for x in situation['market_overview'] if x['quality_varies']}
    found=0
    for channel in ('board','offers_to_you'):
        for row in obs.get(channel,[]):
            pid=row['product_id'];qty=row.get('qty',1)
            if qty!=1 or obs['you']['units_bought'].get(pid,0)>0:continue
            if channel=='board' and row['in_stock']<1:continue
            if channel=='offers_to_you' and row['open_until_round']<obs['round']:continue
            value=cents(obs['you']['utility_of_next_unit'].get(pid,0));price=cents(row['price'])
            if pid in varies and not row.get('certified_high'):value=value*4//10
            found+=price<=cents(obs['you']['cash']) and value>price
    return found


def analyze_arm(run_dir,expected_markets):
    run_dir=Path(run_dir);meta=json.loads((run_dir/'metadata.json').read_text())
    if meta.get('status')!='finished':raise ValueError(f'Unfinished market run: {run_dir}')
    folders=sorted(p for p in (run_dir/'markets').iterdir() if (p/'final.json').exists())
    if len(folders)!=expected_markets:raise ValueError('Incomplete stock rotations')
    entries=[];load=defaultdict(Counter)
    for folder in folders:
        final=json.loads((folder/'final.json').read_text());scores={s['agent']:s for s in json.loads((folder/'scores.json').read_text())}
        agents={a['agent']:a for a in records(folder/'agents.jsonl')}
        events=[r for r in records(folder/'events.jsonl') if r.get('type')=='turn']
        if len(events)!=450 or len(final['sellers'])!=5 or len(final['buyers'])!=10:raise ValueError('Expected full 5x10, 30-turn market')
        buyer_scores=[scores[b['id']]['raw'] for b in final['buyers']]
        for row in events:
            llm=row.get('llm') or {}
            if llm.get('base_url'):
                load[llm['base_url']]['calls']+=1
                load[llm['base_url']]['output_tokens']+=llm.get('usage',{}).get('output_tokens',0)
        for buyer in final['buyers']:
            aid=buyer['id']
            if aid not in IDS:continue
            actions=[r for r in events if r['agent']==aid];trades=[t for t in final['trades'] if t['buyer']==aid]
            if len(actions)!=30:raise ValueError('Missing persona actions')
            expected=round((buyer['gained_utility']+buyer['cash']-buyer['budget'])/100,2)
            if scores[aid]['raw']!=expected:raise ValueError('Engine surplus mismatch')
            counts=Counter();errors=Counter();tools=Counter();examples=[]
            for row in actions:
                tool=row.get('tool');tools[tool or 'NO_TOOL']+=1;obs=row['observation'];args=row.get('args') or {};llm=row.get('llm') or {};error=row['result'].get('error')
                counts['rejected']+=not row['result'].get('ok')
                if error:errors[error]+=1
                counts['round_budget_waits']+=llm.get('outcome')=='round_budget_used'
                counts['length_truncations']+=llm.get('stop_reason')=='length'
                counts['no_tool_turns']+=not bool(tool)
                counts['output_tokens']+=llm.get('usage',{}).get('output_tokens',0)
                counts['wait_with_visible_positive_bound']+=tool=='wait' and bool(visible_opportunities(obs,agents[aid]['situation']))
                pair=(args.get('seller_id'),args.get('product_id'))
                current=next((b for b in obs.get('board',[]) if (b['seller'],b['product_id'])==pair and b['in_stock']>=1),None)
                if tool=='buy_listed':
                    counts['buy_pair_absent_from_current_board']+=current is None
                    counts['already_owned_buy_attempts']+=obs['you']['units_bought'].get(pair[1],0)>0
                if tool=='accept_offer':
                    counts['offer_id_absent_from_current_offers']+=not any(o['offer_id']==args.get('offer_id') for o in obs.get('offers_to_you',[]))
                if error=='no_stock':
                    won=sum(t['qty'] for t in final['trades'] if t['tick']==row['tick'] and (t['seller'],t['product'])==pair)
                    kind='no_stock_missing_current_pair' if current is None else 'no_stock_same_tick_exhausted' if won>=current['in_stock'] else 'no_stock_visible_unresolved'
                    counts[kind]+=1
                if error or (tool=='wait' and visible_opportunities(obs,agents[aid]['situation'])):
                    examples.append({'tick':row['tick'],'round':row['round'],'tool':tool,'args':args,'error':error,
                        'visible_positive_bound':visible_opportunities(obs,agents[aid]['situation']),
                        'cash':obs['you']['cash'],'board':obs['board'],'offers':obs.get('offers_to_you',[]),'reasoning':llm.get('reasoning')})
            entries.append({'market':folder.name,'agent':aid,'surplus':expected,
                'buyer_rank':1+sum(s>expected for s in buyer_scores),'spent':buyer['spent']/100,'cash':buyer['cash']/100,
                'trades':len(trades),'unique_products':len({t['product'] for t in trades}),
                'low_units':sum(t['qty'] for t in trades if t['grade']=='low'),
                'zero_utility_spend':sum(t['price']*t['qty'] for t in trades if t['utility']==0)/100,
                'low_grade_negative_surplus':sum(max(0,t['price']*t['qty']-t['utility']) for t in trades if t['grade']=='low')/100,
                'counts':dict(counts),'errors':dict(errors),'tools':dict(tools),'examples':examples,
                'model_failures':scores[aid].get('model_failures',0)})
    if len(entries)!=expected_markets*3:raise ValueError('Missing persona/rotation scores')
    personas={}
    for aid in IDS:
        part=[r for r in entries if r['agent']==aid];totals=Counter();errors=Counter();tools=Counter()
        for r in part:totals.update(r['counts']);errors.update(r['errors']);tools.update(r['tools'])
        personas[aid]={'markets':len(part),'mean_surplus':round(mean(r['surplus'] for r in part),2),
            'min_surplus':min(r['surplus'] for r in part),'max_surplus':max(r['surplus'] for r in part),
            'mean_buyer_rank':round(mean(r['buyer_rank'] for r in part),2),
            'mean_cash_left':round(mean(r['cash'] for r in part),2),'total_trades':sum(r['trades'] for r in part),
            'zero_utility_spend':round(sum(r['zero_utility_spend'] for r in part),2),
            'low_grade_negative_surplus':round(sum(r['low_grade_negative_surplus'] for r in part),2),
            'counts':dict(totals),'errors':dict(errors),'tools':dict(tools),
            'model_failures':sum(r['model_failures'] for r in part)}
    return {'run_dir':str(run_dir),'mock':meta.get('mock_llm',False),'personas':personas,'markets':entries,
            'actual_server_load':{k:dict(v) for k,v in load.items()}}


def write_report(suite_dir):
    suite_dir=Path(suite_dir);manifest=json.loads((suite_dir/'suite.json').read_text());state=json.loads((suite_dir/'state.json').read_text())
    arms={label:analyze_arm(info['run_dir'],manifest['markets_per_arm']) for label,info in state['completed'].items()}
    complete=set(arms)=={'candidate','base_on'}
    paired=[]
    if complete:
        old={(r['market'],r['agent']):r for r in arms['base_on']['markets']}
        for row in arms['candidate']['markets']:
            base=old[row['market'],row['agent']]
            paired.append({'market':row['market'],'agent':row['agent'],'candidate':row['surplus'],'base_on':base['surplus'],'delta':round(row['surplus']-base['surplus'],2)})
    valid=complete and not any(a['mock'] for a in arms.values()) and not any(p['model_failures'] for a in arms.values() for p in a['personas'].values())
    report={'complete':complete,'valid_real_evaluation':valid,'arms':arms,'paired_surplus':paired,
            'scope':f'One public economy, fixed toy opponents/seed, {manifest["markets_per_arm"]} stock rotation(s) per arm. Does not establish final grade or all-checkpoint superiority.',
            'notes':['Surplus is original engine score. Loss columns overlap and are already included; never subtract them twice.',
                     'Visible positive-bound waits are opportunities at observation time, not guaranteed fills or measured regret.',
                     'Stock rotations share opponents/seed and are not independent economies.',
                     'The candidate is the strongest operational reference selected so far, not a previously proven market winner.']}
    if valid:
        report['best_persona_per_arm']={a:max(arms[a]['personas'],key=lambda p:arms[a]['personas'][p]['mean_surplus']) for a in arms}
    (suite_dir/'market_review.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Buyer完整市場對照','',f'完成：{complete}；有效真模型評估：{valid}。',manifest['description'],'',
        '| 版本 | Persona | 場次 | 平均surplus $ | 最低～最高 $ | 平均buyer名次 /10 | 被拒 | 同回合搶光 | 當前掛牌不存在 |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for label,a in arms.items():
        for aid,p in a['personas'].items():
            c=p['counts'];lines.append(f'| {label} | {aid} | {p["markets"]} | {p["mean_surplus"]:.2f} | {p["min_surplus"]:.2f}～{p["max_surplus"]:.2f} | {p["mean_buyer_rank"]} | {c.get("rejected",0)} | {c.get("no_stock_same_tick_exhausted",0)} | {c.get("buy_pair_absent_from_current_board",0)} |')
    if complete:
        lines+=['','| Persona | 配對平均surplus差：candidate − base $ | 勝／同／負（輪換） |','|---|---:|---|']
        for aid in IDS:
            values=[r['delta'] for r in paired if r['agent']==aid]
            lines.append(f'| {aid} | {mean(values):+.2f} | {sum(v>0 for v in values)}／{sum(v==0 for v in values)}／{sum(v<0 for v in values)} |')
    lines+=['','## 要加強的行動與實際損失','',
        '| 版本 | Persona | 買不起 | 無工具 | Token耗盡等待 | 截斷 | 有可見正收益卻wait | 零效用支出 $ | 低品質負收益 $ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,a in arms.items():
        for aid,p in a['personas'].items():
            c=p['counts'];lines.append(f'| {label} | {aid} | {p["errors"].get("insufficient_cash",0)} | {c.get("no_tool_turns",0)} | {c.get("round_budget_waits",0)} | {c.get("length_truncations",0)} | {c.get("wait_with_visible_positive_bound",0)} | {p["zero_utility_spend"]:.2f} | {p["low_grade_negative_surplus"]:.2f} |')
    lines+=['',report['scope'],'',*report['notes'],'','逐場分數、完整錯誤次數、動作分布、現金、候選機會與失敗tick見market_review.json；完整對話在各run的transcript.md／events.jsonl。']
    (suite_dir/'market_review.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return report
