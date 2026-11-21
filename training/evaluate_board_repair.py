#!/usr/bin/env python3
"""Matched ON comparison: original parent step32 and repair-local step16 / step32."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write_review(folder):
    folder=Path(folder)
    report=json.loads((folder/'cash_decision_pairs.json').read_text())
    cases={c['name']:c for c in json.loads((folder/'cases.json').read_text())}
    rows=[json.loads(s) for s in (folder/'scored_decisions.jsonl').read_text().splitlines()]
    plan=json.loads((folder/'plan.json').read_text())
    if set(report['models'])!={'parent_on','repair16','repair32'}:raise ValueError('Missing repair comparison models')
    parent=report['models']['parent_on'];result={}
    channel_persona={}
    for label in report['models']:
        parts={}
        for row in rows:
            if row['adapter']!=label:continue
            channel=cases[row['case']]['origin']['pair_id'].split('_')[1]
            persona=row['persona'].split('/',1)[1]
            key=channel+'/'+persona
            part=parts.setdefault(key,{'positive_cases':0,'positive_captured':0,'unaffordable':0,'wrong_ids_or_fake_offers':0})
            part['positive_cases']+=cases[row['case']]['expected_tool']!='wait'
            part['positive_captured']+=row['operation_capture']
            errors=row['reviewed_evaluation']['errors']
            part['unaffordable']+='unaffordable' in errors
            part['wrong_ids_or_fake_offers']+=any(e in errors for e in ('not_on_current_board','not_current_offer'))
        channel_persona[label]=parts
    for label,metrics in report['models'].items():
        sample=[r for r in rows if r['adapter']==label]
        tokens=sorted(r.get('usage',{}).get('output_tokens',0)+r.get('usage',{}).get('reasoning_tokens',0) for r in sample)
        board,offer=metrics['channel']['board'],metrics['channel']['offer']
        board_cases=[r for r in sample if cases[r['case']]['origin']['pair_id'].split('_')[1]=='board']
        bad_ids=sum(any(e in r['reviewed_evaluation']['errors'] for e in ('not_on_current_board','not_current_offer')) for r in board_cases)
        stats={'mean_generated_tokens':sum(tokens)/len(tokens),'p95_generated_tokens':tokens[math.ceil(.95*len(tokens))-1],
               'truncated':sum(r['stop_reason']=='length' for r in sample),
               'missing_usage':sum(not r.get('usage') for r in sample)}
        checks={'board_positive_at_least_72_of_96':board['positive_captured']>=72,
                'offer_positive_at_least_88_of_96':offer['positive_captured']>=88,
                'offer_both_correct_at_least_78_of_96':offer['both_correct']>=78,
                'board_cash_errors_remain_zero':board['unaffordable']==0,
                'board_wrong_ids_or_fake_offers_zero':bad_ids==0,
                'overall_pair_correctness_not_worse_than_parent':metrics['overall']['both_correct']>=parent['overall']['both_correct'],
                'no_truncation_and_p95_at_most_1024':stats['truncated']==0 and stats['p95_generated_tokens']<=1024 and stats['missing_usage']==0}
        if plan['experiment_manifest'].get('experiment')=='buyer_symmetric_repair_v1':
            checks['overall_cash_errors_not_worse_than_parent']=metrics['overall']['unaffordable']<=parent['overall']['unaffordable']
            checks['offer_cash_errors_not_worse_than_parent']=offer['unaffordable']<=parent['channel']['offer']['unaffordable']
            for key,part in channel_persona[label].items():
                checks[f'{key}_positive_capture_not_worse_than_parent']=part['positive_captured']>=channel_persona['parent_on'][key]['positive_captured']
        result[label]={'channel_persona':channel_persona[label],'board':board,'offer':offer,'overall':metrics['overall'],'token_stats':stats,
                       'board_wrong_ids_or_fake_offers':bad_ids,'checks':checks,
                       'meets_repair_targets':label!='parent_on' and all(checks.values())}
    eligible=[label for label in ('repair16','repair32') if result[label]['meets_repair_targets']]
    selected=max(eligible,key=lambda label:(result[label]['board']['positive_captured'],result[label]['offer']['both_correct'])) if eligible else None
    output={'models':result,'candidate_meeting_targets':selected,'experiment_manifest':plan['experiment_manifest'],
            'note':'Development targets, not market scores or automatic promotion. Same two validation markets/personas/seeds; full market surplus still untested.'}
    (folder/'board_repair_review.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Board-first repair comparison','',
           '| Model | Board positives | Offer positives | Offer both correct | Cash errors board / offer | All both correct | Board bad IDs | Output p95 | Meets targets |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for label,m in result.items():
        lines.append(f'| {label} | {m["board"]["positive_captured"]}/96 | {m["offer"]["positive_captured"]}/96 | '
                     f'{m["offer"]["both_correct"]}/96 | {m["board"]["unaffordable"]} / {m["offer"]["unaffordable"]} | '
                     f'{m["overall"]["both_correct"]}/192 | {m["board_wrong_ids_or_fake_offers"]} | '
                     f'{m["token_stats"]["p95_generated_tokens"]} | {m["meets_repair_targets"]} |')
    lines+=['',f'Candidate meeting targets: {selected}', '',output['note']]
    (folder/'board_repair_review.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return output


def main():
    from training.decision_ablation import digest,check_fingerprint,fingerprint
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();run=a.run.resolve()
    cfg=json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version') not in ('buyer_board_repair_v1','buyer_id_repair_v1','buyer_symmetric_repair_v1') or not cfg.get('thinking'):
        raise ValueError('Expected ON board/ID/symmetric repair run')
    parent=Path(cfg['initial_adapter']);check_fingerprint(parent,cfg['initial_adapter_fingerprints'])
    adapters={'parent_on':str(parent),'repair16':str(run/'checkpoints/step_0016'),'repair32':str(run/'checkpoints/step_0032')}
    for label,path in adapters.items():
        info=json.loads((Path(path)/'training_step.json').read_text())
        expected=16 if label=='repair16' else 32
        if not info['thinking'] or info['step']!=expected:raise ValueError('Comparison checkpoint step/mode mismatch')
    data=ROOT/cfg['data_dir'];manifest=json.loads((data/'manifest.json').read_text());cases=data/'cash_probe_cases.json'
    if digest(data/'manifest.json')!=cfg['data_manifest_sha256'] or digest(cases)!=manifest['cash_probe_sha256']:
        raise ValueError('Frozen repair evaluation inputs changed')
    for name,expected in manifest['sources'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Repair prompt/data source changed')
    provenance={'experiment':cfg.get('experiment','buyer_board_repair_v1'),'thinking':True,'parent':str(parent),
                'checkpoints':{label:fingerprint(path) for label,path in adapters.items()},
                'data_manifest_sha256':cfg['data_manifest_sha256'],'training_stage':'engine-backed demonstration SFT, not preference training'}
    from training.buyer_diagnostics import main as evaluate
    with tempfile.TemporaryDirectory(prefix='adl_board_repair_eval_',dir='/tmp') as temp:
        directory=Path(temp);(directory/'adapters.json').write_text(json.dumps(adapters));(directory/'experiment.json').write_text(json.dumps(provenance))
        argv=['buyer_diagnostics.py','--versions','operations_reasoning','--thinking','--analysis','cash_decision',
              '--cases-file',str(cases),'--adapters-file',str(directory/'adapters.json'),
              '--experiment-manifest',str(directory/'experiment.json'),'--seeds','0,1','--gpus',a.gpus,'--port',str(a.port)]
        if a.prepare_only:argv.append('--prepare-only')
        previous=sys.argv
        try:sys.argv=argv;folder=evaluate()
        finally:sys.argv=previous
    if not a.prepare_only:
        write_review(folder);print(f'Read first: {folder}/board_repair_review.md',flush=True)
    return folder


if __name__=='__main__':main()
