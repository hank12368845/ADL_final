"""CPU report correction: seller event stock can alias a later engine book.

Do not change the TA engine or original frozen reports. Use the actual serialized
model input, independently checked against pre-tick inventory reconstructed from
initial stock and earlier trades. This is post-run accounting, not a formatter.
"""
import copy
import json
from pathlib import Path

from training import compare_seller_models as paired


def decision_stock(row,book,trades):
    start=dict(book['initial_stock'])
    for t in trades:
        if t['seller']==row['agent'] and t['tick']<row['tick']:
            start[t['product']]-=t['qty']
    if any(n<0 for n in start.values()):raise ValueError('Negative reconstructed starting inventory')
    messages=(row.get('llm') or {}).get('messages') or []
    current=next((m['content'] for m in reversed(messages) if m.get('role')=='user'),None)
    source='initial_stock_minus_prior_tick_trades'
    if current is not None:
        obs=json.JSONDecoder().raw_decode(current.lstrip())[0]
        actual=obs['you']['stock']
        if actual!=start:raise ValueError('Actual model input and reconstructed pre-tick stock differ')
        source='actual_model_input_and_trade_ledger_agree'
    else:actual=start
    return actual,source


def analyze(run):
    run=Path(run);result=paired.analyze(run);market=next((run/'markets').iterdir())
    final=json.loads((market/'final.json').read_text());books={s['id']:s for s in final['sellers']}
    events=[json.loads(l) for l in (market/'events.jsonl').read_text().splitlines()]
    for aid,seller in result['sellers'].items():
        corrected=[];mismatches=[];verified=0
        for row in events:
            if row.get('type')!='turn' or row.get('agent')!=aid:continue
            stock,source=decision_stock(row,books[aid],final['trades'])
            if source=='actual_model_input_and_trade_ledger_agree':verified+=1
            if stock!=row['observation']['you']['stock']:
                mismatches.append(dict(tick=row['tick'],tool=row.get('tool'),product=(row.get('args') or {}).get('product_id'),
                                       actual_input_stock=stock,event_stock=row['observation']['you']['stock']))
            r=dict(row);r['observation']=copy.deepcopy(row['observation']);r['observation']['you']['stock']=stock;corrected.append(r)
        seller['event_based_quote_diagnostics']=seller['quote_diagnostics']
        seller['quote_diagnostics']=paired.quote_diagnostics(corrected,aid)
        seller['stock_audit']=dict(model_inputs_verified=verified,event_stock_mismatches=mismatches,
                                  source='Actual serialized model input or exact initial-stock-minus-earlier-tick-trades ledger')
    result['observation_audit_note']='Seller event observation.you.stock aliases mutable engine state. Quote diagnostics use pre-tick actual input/ledger; original price/profit/grades unchanged.'
    return result
