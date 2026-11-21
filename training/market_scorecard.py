"""Per-student-buyer raw scores and observed losses, using original engine records."""
import json
from pathlib import Path
from collections import Counter


def write_scorecard(run_dir):
    run_dir=Path(run_dir);meta=json.loads((run_dir/'metadata.json').read_text())
    entries=[]
    for folder in sorted((run_dir/'markets').iterdir()):
        if not (folder/'final.json').exists():continue
        final=json.loads((folder/'final.json').read_text())
        agents={a['agent']:a for a in (json.loads(x) for x in (folder/'agents.jsonl').read_text().splitlines())}
        turns=[json.loads(x) for x in (folder/'transcript.jsonl').read_text().splitlines()]
        scores={s['agent']:s for s in json.loads((folder/'scores.json').read_text())}
        for b in final['buyers']:
            if agents[b['id']]['owner']=='ta':continue
            trades=[t for t in final['trades'] if t['buyer']==b['id']]
            actions=[r for r in turns if r['agent']==b['id']]
            seen=set();rebuy_gain=0;rebuy_count=0
            for t in trades:
                if t['product'] in seen and t['utility']>0:
                    rebuy_count+=1;rebuy_gain+=(t['utility']-t['price']*t['qty'])/100
                seen.add(t['product'])
            entries.append({'market':folder.name,'agent':b['id'],'surplus':scores[b['id']]['raw'],
               'utility':b['gained_utility']/100,'spent':b['spent']/100,'cash':b['cash']/100,
               'unique_products':len(seen),'units':sum(t['qty'] for t in trades),
               'zero_gain_trade_spend':sum(t['price']*t['qty'] for t in trades if t['utility']==0)/100,
               'low_grade_negative_surplus':sum(max(0,t['price']*t['qty']-t['utility']) for t in trades if t['grade']=='low')/100,
               'positive_utility_rebuy_count':rebuy_count,'positive_utility_rebuy_net':round(rebuy_gain,2),
               'rejected':dict(Counter(r['result'].get('error') for r in actions if not r['result'].get('ok')))})
    output={'mock':meta['mock_llm'],'buyers':entries,
            'note':'Surplus is original engine raw score. Loss columns overlap and are already in raw score; never subtract them again. '
                   'A second high after low can add utility. This single market does not estimate final grade.'}
    (run_dir/'buyer_scorecard.json').write_text(json.dumps(output,indent=2))
    lines=['# Buyer market scorecard','',f'Mock: {output["mock"]}. {output["note"]}','',
           '| Market | Buyer | Raw surplus $ | Utility $ | Spent $ | Zero-gain spend $ | Low-grade negative $ |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in entries:lines.append('| '+' | '.join(str(r[k]) for k in ['market','agent','surplus','utility','spent','zero_gain_trade_spend','low_grade_negative_surplus'])+' |')
    (run_dir/'buyer_scorecard.md').write_text('\n'.join(lines)+'\n')
    print(f'Buyer raw-score details: {run_dir}/buyer_scorecard.md')
    return output
