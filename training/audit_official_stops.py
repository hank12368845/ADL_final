"""CPU-only review of saved generations at the tokenizer's first end-of-turn token."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(run):
    from transformers import AutoTokenizer
    from training.buyer_hf_check import parse_raw
    from training.buyer_diagnostics import evaluate
    from training.diagnostic_scores import ACTION_ERRORS,POLICY_ERRORS
    from training.official_buyer_train import probe_summary,official_snapshot_review
    run=Path(run);cfg=json.loads((run/'train_config.json').read_text())
    t=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    cases={c['name']:c for n in ('native_cases.json','official_native_cases.json','train_native_cases.json')
           for c in json.loads((run/n).read_text())}
    report={'method':'CPU-only audit of saved greedy output, ending at first tokenizer EOS. Not a new GPU evaluation or market score. Original outputs unchanged.',
            'tokenizer_eos_id':t.eos_token_id,'steps':[],'source_sha256':{}}
    for f in sorted(run.glob('native_step_*.json')):
        report['source_sha256'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        original=json.loads(f.read_text());rows=[]
        for d in original['decisions']:
            c=cases[d['case']];raw=d['text'];end=raw.find(t.eos_token)
            prefix=raw[:end+len(t.eos_token)] if end>=0 else raw
            parsed=parse_raw(t,prefix,c['tools']);truncated=end<0 and d['truncated']
            token_count=len(t.encode(prefix,add_special_tokens=False))
            common={'case':c['name'],'tool':parsed['tool'],'arguments':parsed['arguments'],
                    'truncated':truncated,'ended_first_turn':end>=0,'generated_tokens':token_count,
                    'original_generated_tokens':d['generated_tokens']}
            if d.get('kind'):
                rows.append({**common,'kind':d['kind'],**official_snapshot_review(c,parsed['tool'],parsed['arguments'])})
                continue
            checked=evaluate(c,parsed['tool'],parsed['arguments']);errors=checked['errors']
            valid=not truncated and bool(parsed['tool']) and not bool(set(errors)&(ACTION_ERRORS|POLICY_ERRORS))
            positive=d['positive']
            rows.append({**common,'pair_id':d['pair_id'],'positive':positive,'errors':errors,'valid_action':valid,
                'captured':positive and valid and parsed['tool'] in ('buy_listed','accept_offer'),
                'correct_negative_wait':not positive and valid and parsed['tool']=='wait'})
        summary={'step':original['summary']['step'],'completion_loss':original['summary']['completion_loss'],
                 'current_first_turn':probe_summary([r for r in rows if not r.get('kind')])}
        for kind in ('official_heldout','official_train_seen'):
            selected=[r for r in rows if r.get('kind')==kind]
            summary[kind]={'passed':sum(r['valid_snapshot'] and not r['truncated'] for r in selected),
                'calls':len(selected),'positive_trades':sum(r['positive_trade'] and not r['truncated'] for r in selected),
                'truncated':sum(r['truncated'] for r in selected)}
        report['steps'].append({'summary':summary,'decisions':rows})
    (run/'stop_boundary_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# First-turn CPU audit','',report['method'],'',
        '| Step | NLL | Legal /16 | Positive /8 | Both /8 | Cash errors | No tool | Truncated | Official valid /9 | Seen valid /6 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['steps']:
        s=row['summary'];b=s['current_first_turn']
        lines.append(f'| {s["step"]} | {s["completion_loss"]:.4f} | {b["valid_actions"]} | {b["positive_captured"]} | {b["both_correct"]} | {b["unaffordable"]} | {b["no_tool"]} | {b["truncated"]} | {s["official_heldout"]["passed"]} | {s["official_train_seen"]["passed"]} |')
    lines+=['','Snapshot legality permits alternative actions; it is not optimal trading or surplus. The original training_review.md used incorrect generation stopping and must not select a checkpoint.']
    (run/'stop_boundary_audit.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    report=audit(p.parse_args().run)
    for entry in report['steps']:print(json.dumps(entry['summary']))
