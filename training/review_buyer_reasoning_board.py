#!/usr/bin/env python3
"""CPU-only export of missed positive board decisions, with original reasoning and targets."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re
import csv


def money(value):
    return format(Decimal(str(value)), '.2f')


def review(folder):
    from training.cash_decision_report import case_pairs
    from training.buyer_operations_v2_data import choose
    folder=Path(folder).resolve()
    status=json.loads((folder/'evaluation_status.json').read_text())
    if status['status']!='complete':raise ValueError('Evaluation is not complete')
    case_list=json.loads((folder/'cases.json').read_text());case_pairs(case_list)
    cases={c['name']:c for c in case_list}
    rows=[json.loads(s) for s in (folder/'scored_decisions.jsonl').read_text().splitlines()]
    plan=json.loads((folder/'plan.json').read_text())
    if len(rows)!=plan['calls'] or any(not r['service_ok'] for r in rows):raise ValueError('Invalid/incomplete results')
    paired=defaultdict(dict)
    for line,row in enumerate(rows,1):
        label,persona=row['persona'].split('/',1)
        key=(persona,row['case'],row['context'],row['seed'])
        if label in paired[key]:raise ValueError('Duplicate model decision')
        paired[key][label]=(line,row)
    for group in paired.values():
        if set(group)!={'base_on','candidate'}:raise ValueError('Missing matched base ON / candidate')
        if len({(r['input_sha256'],r['gpu']) for _,r in group.values()})!=1:
            raise ValueError('Matched model inputs or GPUs differ')
    indexed=sorted([(line,row) for group in paired.values() for line,row in [group['candidate']]
                    if cases[row['case']]['expected_tool']=='buy_listed' and not row['operation_capture']],
                   key=lambda x:(x[1]['case'],x[1]['persona'],x[1]['seed']))
    misses=[]
    for number,(line,row) in enumerate(indexed,1):
        case=cases[row['case']];obs=case['observation']
        if len(obs['board'])!=1:raise ValueError('This detailed audit expects one current board row')
        board=obs['board'][0];pid=board['product_id']
        current=json.loads(case['messages'][-1]['content'])
        state=current.get('your_private_state','')
        try:note=json.loads(state) if isinstance(state,str) else state
        except ValueError:note={}
        target_tool,target_args=choose(case['situation'],obs)
        if target_tool!='buy_listed':raise ValueError('Exported positive does not have a visible legal board target')
        cash=Decimal(str(obs['you']['cash']));price=Decimal(str(board['price']))
        value=Decimal(str(obs['you']['utility_of_next_unit'][pid]))
        if not cash>=price or not value>price or obs['you']['units_bought'].get(pid,0) or board['in_stock']<1:
            raise ValueError('Positive board preconditions do not hold')
        reasoning=row.get('reasoning') or ''
        noncurrent=sorted(set(re.findall(r'\bP[0-9a-f]{7}\b',reasoning))-{pid})
        last_buy=note.get('last_buy') if isinstance(note,dict) else None
        remembered=last_buy[1] if isinstance(last_buy,list) and len(last_buy)==2 else None
        literal=re.search(r'C\s*-\s*P\s*=\s*(-?\d+(?:\.\d+)?)',reasoning)
        literal_value=Decimal(literal.group(1)) if literal else None
        hints=re.search(r'all checks pass|checks pass|(?:^|[;,])\s*eligible(?:[;,.]|$)|executable and best|execute buy_listed|'
                        r'wait is not required|no ownership/avail/val issues',reasoning,re.I)
        key=(row['persona'].split('/',1)[1],row['case'],row['context'],row['seed'])
        before=paired[key]['base_on'][1]
        category={'wait':'wait_instead_of_positive_board',
                  'buy_listed':'buy_not_on_current_board',
                  'accept_offer':'accept_nonexistent_offer'}.get(row['tool'],'other')
        misses.append({'id':number,'scored_decisions_line':line,'case':row['case'],
                       'persona':key[0],'seed':row['seed'],'gpu':row['gpu'],
                       'origin':case['origin'],'category':category,
                       'current_facts':{'seller_id':board['seller'],'product_id':pid,
                           'cash':money(cash),'price':money(price),'cash_minus_price':money(cash-price),
                           'next_unit_value':money(value),'surplus':money(value-price),
                           'in_stock':board['in_stock'],'owned_units':obs['you']['units_bought'].get(pid,0)},
                       'expected_tool_call':{'name':target_tool,'arguments':target_args},
                       'actual_tool_call':{'name':row['tool'],'arguments':row['arguments']},
                       'reasoning':reasoning,'errors':row['reviewed_evaluation']['errors'],
                       'base_on_captured_same_input':before['operation_capture'],
                       'text_indicators':{'noncurrent_products_mentioned':noncurrent,
                           'last_buy_product_in_note':remembered,
                           'noncurrent_owned_products':[p for p in noncurrent if obs['you']['units_bought'].get(p,0)>0],
                           'mentions_last_buy_instead_of_current':remembered in noncurrent if remembered else False,
                           'explicit_positive_phrase_but_wait':bool(hints and row['tool']=='wait'),
                           'literal_C_minus_P':str(literal_value) if literal_value is not None else None,
                           'literal_matches_current_remainder':literal_value==cash-price if literal_value is not None else None},
                       'original_scored_row':row,'original_case':case})
    board_stats={};flips=Counter()
    for label in ('base_on','candidate'):
        positives=[r for group in paired.values() for _,r in [group[label]] if cases[r['case']]['expected_tool']=='buy_listed']
        board_stats[label]={'positive_n':len(positives),'captured':sum(r['operation_capture'] for r in positives),
                            'by_history':{mode:{'n':sum(cases[r['case']]['origin']['history_mode']==mode for r in positives),
                                'captured':sum(r['operation_capture'] for r in positives if cases[r['case']]['origin']['history_mode']==mode)}
                                          for mode in ('fresh','recorded')}}
    for group in paired.values():
        a,b=group['base_on'][1],group['candidate'][1]
        if cases[b['case']]['expected_tool']=='buy_listed':
            before,after=a['operation_capture'],b['operation_capture']
            flips['fixed' if after and not before else 'regressed' if before and not after else 'tie']+=1
    summary={'missed_calls':len(misses),'distinct_case_names':len({r['case'] for r in misses}),
             'distinct_input_hashes':len({r['original_scored_row']['input_sha256'] for r in misses}),
             'categories':dict(Counter(r['category'] for r in misses)),
             'by_history':dict(Counter(r['origin']['history_mode'] for r in misses)),
             'by_variant':dict(Counter(r['origin']['cash_variant'] for r in misses)),
             'noncurrent_product_mentioned':sum(bool(r['text_indicators']['noncurrent_products_mentioned']) for r in misses),
             'noncurrent_owned_product_mentioned':sum(bool(r['text_indicators']['noncurrent_owned_products']) for r in misses),
             'mentions_last_buy_instead_of_current':sum(r['text_indicators']['mentions_last_buy_instead_of_current'] for r in misses),
             'explicit_positive_phrase_but_wait':sum(r['text_indicators']['explicit_positive_phrase_but_wait'] for r in misses),
             'literal_remainder_matches_current_cash_and_price':sum(r['text_indicators']['literal_matches_current_remainder'] is True for r in misses),
             'wait_with_matching_current_remainder':sum(r['actual_tool_call']['name']=='wait' and
                 r['text_indicators']['literal_matches_current_remainder'] is True for r in misses),
             'board_positive_capture':board_stats,'base_to_candidate_board_flips':dict(flips),
             'note':'Observable output audit, not proof of internal reasoning causes. Calls repeat personas/seeds and some identical variant inputs. '
                    'The probe has TWO validation markets despite the generic cash report note saying one. '
                    'These validation failures are diagnostic exports, not new training data.'}
    output={'summary':summary,'misses':misses}
    (folder/'board_missed_48.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    columns=['id','case','persona','seed','category','product_id','cash','price','cash_minus_price','next_unit_value',
             'surplus','actual_tool','actual_arguments','reasoning','scored_decisions_line']
    with (folder/'board_missed_48.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in misses:
            writer.writerow({**{k:r[k] for k in ('id','case','persona','seed','category','reasoning','scored_decisions_line')},
                             **{k:r['current_facts'][k] for k in ('product_id','cash','price','cash_minus_price','next_unit_value','surplus')},
                             'actual_tool':r['actual_tool_call']['name'],
                             'actual_arguments':json.dumps(r['actual_tool_call']['arguments'],ensure_ascii=False)})
    lines=['# Missed positive board calls','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
           'Each entry includes current facts, the correct tool, the actual tool and unabridged model reasoning. '
           'JSON also includes the complete source case and scored response.','']
    for r in misses:
        f=r['current_facts']
        lines += [f'## {r["id"]}. {r["case"]} / {r["persona"]} / seed {r["seed"]}','',
                  f'Current: seller={f["seller_id"]}, item={f["product_id"]}, cash={f["cash"]}, price={f["price"]}, '
                  f'remainder={f["cash_minus_price"]}, surplus={f["surplus"]}, stock={f["in_stock"]}, owned={f["owned_units"]}.','',
                  f'Expected: `{json.dumps(r["expected_tool_call"],ensure_ascii=False)}`','',
                  f'Actual: `{json.dumps(r["actual_tool_call"],ensure_ascii=False)}`','',
                  '```text',r['reasoning'].rstrip(),'```','']
    (folder/'board_missed_48.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    review(parser.parse_args().run)
