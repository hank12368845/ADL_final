"""Versioned, transparent diagnostics; these scores are not market rewards."""
from pathlib import Path
import json
from collections import defaultdict

ACTION_ERRORS = {'no_native_tool_call','unknown_tool','tool_argument_keys','tool_argument_types','not_on_current_board',
                 'not_current_offer','offer_expired','unaffordable','no_marginal_high_utility',
                 'nonpositive_best_grade_surplus'}
STATE_ERRORS = {'missing_state_string','state_over_800','state_not_json_object','held_snapshot',
                'cash_snapshot','missing_check_fields','incomplete_trade_checklist',
                'trade_check_flags_mismatch','fixed_quality_gain_mismatch','operations_memory'}
POLICY_ERRORS = {'qty_not_one','offer_multiunit_policy','persona_forbids_rebuy',
                 'ignores_known_low_risk','nonpositive_quality_bound','expected_wait'}
SCORING_VERSION = 'diagnostic_v1'


def rate(rows,key):
    return round(100*sum(r[key] for r in rows)/len(rows),2) if rows else None


def write_scores(folder):
    from training.buyer_diagnostics import evaluate, read_rows
    folder=Path(folder);cases={c['name']:c for c in json.loads((folder/'cases.json').read_text())}
    modes={c.get('state_mode','legacy') for c in cases.values()}
    if len(modes)>1:raise ValueError('Do not mix diagnostic memory rubrics in one score table')
    scoring_version='diagnostic_operations_v1' if modes=={'operations'} else SCORING_VERSION
    common={name for name,c in cases.items() if 'recorded' in c['contexts'] and 'fresh' in c['contexts']}
    groups=defaultdict(list);scored=[]
    for r in read_rows(folder/'decisions.jsonl'):
        evaluation=evaluate(cases[r['case']],r['tool'],r['arguments'])
        errors=set(evaluation['errors']);available=r['outcome']=='accepted'
        score={**r,'scoring_version':scoring_version,'reviewed_evaluation':evaluation,
               'service_ok':available,
               'action_ok':available and not bool(ACTION_ERRORS & errors),
               'state_ok':available and bool(r['tool']) and not bool(STATE_ERRORS & errors),
               'policy_ok':available and bool(r['tool']) and not bool(POLICY_ERRORS & errors)}
        score['operation_capture']=available and r['tool'] in ('buy_listed','accept_offer') and not bool((ACTION_ERRORS|POLICY_ERRORS)&errors)
        groups[r['persona']].append(score);scored.append(score)
    table={}
    for persona,rows in sorted(groups.items()):
        matched={context:[r for r in rows if r['context']==context and r['case'] in common] for context in ['recorded','fresh']}
        offers=[r for r in rows if r['case']=='binding_counteroffer']
        opportunities=[r for r in rows if r['reviewed_evaluation'].get('positive_reference')]
        table[persona]={'calls':len(rows),'service_score':rate(rows,'service_ok'),
                        'action_score':rate(rows,'action_ok'),'state_score':rate(rows,'state_ok'),
                        'policy_score':rate(rows,'policy_ok'),
                        'operation_opportunities':len(opportunities),
                        'operation_capture':rate(opportunities,'operation_capture'),
                        'counteroffer_capture':round(100*sum(r['tool']=='accept_offer' for r in offers)/len(offers),2) if offers else None,
                        'matched_recorded_action_score':rate(matched['recorded'],'action_ok'),
                        'matched_fresh_action_score':rate(matched['fresh'],'action_ok')}
    plan_path=folder/'plan.json'
    planned=json.loads(plan_path.read_text()).get('calls') if plan_path.exists() else None
    complete=planned is None or len(scored)==planned
    service_ok=all(r['service_ok'] for r in scored)
    valid=bool(scored) and complete and service_ok
    output={'scoring_version':scoring_version,'scores':table,
            'valid_evaluation':valid,'expected_calls':planned,'recorded_calls':len(scored),
            'service_failures':sum(not r['service_ok'] for r in scored),
            'rubric':{'action_errors':sorted(ACTION_ERRORS),'state_errors':sorted(STATE_ERRORS),'policy_errors':sorted(POLICY_ERRORS)},
            'meaning':'Each score is 100 times passed calls / total calls. Service failures count as not passed and are shown separately. '
                      'Compare identical case mixes and this scoring version. Action score checks selected constraints, not full engine execution or optimality. '
                      'State and policy scores are separate; no weighted aggregate or course-grade claim.'}
    (folder/'scoreboard.json').write_text(json.dumps(output,indent=2))
    (folder/'scored_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    lines=['# Buyer diagnostic scores','',f'Scoring: {scoring_version}. These are decision-check percentages, not market surplus.','',
           '| Persona | Calls | Service | Action /100 | State /100 | Policy /100 | Offer capture % | Matched history action | Matched fresh action | Operation capture % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if not valid:
        lines[2:2]=['**INVALID EVALUATION: missing calls or model service failures. Zero values below are failure accounting, not model performance.**','']
    for name,v in table.items():
        lines.append('| '+name+' | '+' | '.join(str(v[k]) for k in ['calls','service_score','action_score','state_score','policy_score','counteroffer_capture','matched_recorded_action_score','matched_fresh_action_score','operation_capture'])+' |')
    lines += ['',output['meaning'],'','Profitable multi-unit offers may fail the single-unit policy while passing action checks. Inspect economic opportunity separately.']
    (folder/'scoreboard.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return output
