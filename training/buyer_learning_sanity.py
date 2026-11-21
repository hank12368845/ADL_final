#!/usr/bin/env python3
"""Small train-set free-generation check using EXISTING weights; never trains a model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = ROOT/'training/runs/20261119-183011_buyer_operations_n9qzhy75'


def prepare(run):
    from training.decision_ablation import digest, check_fingerprint, fingerprint
    from training.buyer_operations_v2_data import diagnostic_case
    from training.buyer_reasoning_data import personas
    from training.buyer_diagnostics import replace_persona
    from transformers import AutoTokenizer
    run = Path(run).resolve()
    cfg = json.loads((run/'train_config.json').read_text())
    if cfg.get('data_version') != 'buyer_symmetric_repair_v1' or not cfg['thinking']:
        raise ValueError('Expected completed symmetric ON run')
    if json.loads((run/'wrapper_result.json').read_text())['exit'] != 0:
        raise ValueError('Training did not complete')
    data = ROOT/cfg['data_dir']
    manifest = json.loads((data/'manifest.json').read_text())
    if digest(data/'manifest.json') != cfg['data_manifest_sha256'] or digest(data/'train.jsonl') != manifest['splits']['train']['raw_sha256']:
        raise ValueError('Training input provenance changed')
    if digest(data/'train.tokens.jsonl') != manifest['splits']['train']['tokens_sha256']:
        raise ValueError('Training tokens changed')
    seen = {json.loads(line)['id'] for rank in (0,1)
            for line in (run/f'samples_rank{rank}.jsonl').read_text().splitlines()}
    tokens = {r['id']:r for r in map(json.loads,(data/'train.tokens.jsonl').read_text().splitlines())}
    rows = list(map(json.loads,(data/'train.jsonl').read_text().splitlines()))
    pools = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row['id'] not in seen or row['persona_index'] != 3 or row['cash_variant'] not in ('equal','above_one','wide','zero'):
            continue
        key = (row['family'].split('_')[0],row['cash_variant'],row['history_mode'])
        pools[key][row['pair_id']][row['positive']] = row
    selected = []
    for channel in ('board','offer'):
        for variant in ('equal','above_one','wide','zero'):
            for history in ('fresh','recorded'):
                candidates = pools[channel,variant,history]
                pairs = [pair for _,pair in sorted(candidates.items()) if set(pair)=={True,False}]
                if not pairs:
                    raise ValueError('Missing actually seen decision pair')
                selected.append(pairs[0])
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['revision'],local_files_only=True)
    cases = []
    # ci modulo 2 routes a whole cash pair to one GPU in the existing standard worker.
    # Arrange: positive A, positive B, negative A, negative B.
    for start in range(0,len(selected),2):
        for sign in (True,False):
            for pair in selected[start:start+2]:
                row = pair[sign]
                rendered = replace_persona(row['messages'],personas()[2],'recorded')
                if rendered != row['messages']:
                    raise ValueError('Diagnostic persona replacement changes training input')
                prompt = tokenizer.apply_chat_template(rendered,tools=row['tools'],tokenize=False,
                    add_generation_prompt=True,enable_thinking=True)
                ids = tokenizer.encode(prompt,add_special_tokens=False)
                encoded = tokens[row['id']]
                if ids != encoded['input_ids'][:encoded['prompt_length']]:
                    raise ValueError('Replay prompt differs from the actual training token prefix')
                case = diagnostic_case(row)
                case['origin'].update(split='train_seen',cash_variant=row['cash_variant'],
                    history_mode=row['history_mode'],training_sample_id=row['id'])
                cases.append(case)
    parent = Path(cfg['initial_adapter']); check_fingerprint(parent,cfg['initial_adapter_fingerprints'])
    candidate = run/'checkpoints/step_0032'
    step = json.loads((candidate/'training_step.json').read_text())
    if step != {'step':32,'thinking':True}:
        raise ValueError('Wrong candidate checkpoint')
    adapters = {'parent_on':str(parent),'repair32':str(candidate)}
    provenance = {'experiment':'buyer_train_seen_sanity_v1','training_run':str(run),
        'scope':'P3 only, 32 actually seen cash examples; NOT held-out accuracy or market score.',
        'prompt_prefix_identical_to_training_tokens':True,'actual_seen_sample_ids':[c['name'] for c in cases],
        'checkpoints':{name:fingerprint(path) for name,path in adapters.items()},
        'data_manifest_sha256':cfg['data_manifest_sha256'],'calls':64,'seeds':[0]}
    return cases,adapters,provenance


def write_review(folder):
    from training.cash_decision_report import _counts, case_pairs
    folder = Path(folder); plan = json.loads((folder/'plan.json').read_text())
    cases = json.loads((folder/'cases.json').read_text()); definitions = case_pairs(cases)
    rows = list(map(json.loads,(folder/'scored_decisions.jsonl').read_text().splitlines()))
    if len(rows) != 64 or plan['calls'] != 64:
        raise ValueError('Incomplete 64-call sanity check')
    grouped = defaultdict(dict); matched = defaultdict(dict); seen=set()
    byname={c['name']:c for c in cases}
    for row in rows:
        case=byname[row['case']]; positive=case['expected_tool']!='wait'
        key=(row['adapter'],case['origin']['pair_id'],positive)
        if key in seen or not row['service_ok'] or row['persona'].split('/',1)[1]!='operations_reasoning_p3' or row['seed']!=0:
            raise ValueError('Unexpected sanity replay row')
        seen.add(key); grouped[row['adapter'],case['origin']['pair_id']][positive]=row
        matched[row['case']][row['adapter']]=row
    if set(k[0] for k in grouped)!={'parent_on','repair32'}:
        raise ValueError('Missing expected weights')
    for items in matched.values():
        if len(items)!=2 or len({(r['input_sha256'],r['gpu']) for r in items.values()})!=1:
            raise ValueError('Replay models use different inputs or GPU')
    output={}
    for label in ('parent_on','repair32'):
        parts=defaultdict(list)
        for (model,pair_id),sides in grouped.items():
            if model!=label:continue
            if set(sides)!={True,False} or sides[True]['gpu']!=sides[False]['gpu']:
                raise ValueError('Missing pair side or inconsistent GPU')
            channel=definitions[pair_id][True]['origin']['pair_id'].split('_')[1]
            parts[channel].append((sides[True],sides[False]))
        output[label]={channel:_counts(sample) for channel,sample in parts.items()}
        output[label]['overall']=_counts([pair for sample in parts.values() for pair in sample])
    report={'scope':plan['experiment_manifest']['scope'],'models':output,
        'interpretation':'Low seen-set accuracy means this recipe has not demonstrated learning even its selected training examples. '
                         'High seen-set and low held-out accuracy instead directs attention to generalization. '
                         'One persona/seed and 16 cash pairs cannot establish a unique cause or market performance.'}
    (folder/'learning_sanity_review.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Train-seen free-generation sanity','',report['scope'],'',
           '| Model | Positive capture /16 | Both correct /16 | Unaffordable |',
           '|---|---:|---:|---:|']
    for label,m in output.items():
        o=m['overall']; lines.append(f'| {label} | {o["positive_captured"]} | {o["both_correct"]} | {o["unaffordable"]} |')
    lines+=['',report['interpretation']]
    (folder/'learning_sanity_review.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--gpus',default='0,1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    print('CPU: selecting actually seen p3 pairs and verifying exact training prompt tokens...',flush=True)
    cases,adapters,provenance=prepare(args.run)
    print('32 seen examples, 2 existing adapters, 64 calls. No new training. Loading ETA estimating.',flush=True)
    from training.buyer_diagnostics import main as evaluate
    with tempfile.TemporaryDirectory(prefix='adl_seen_sanity_',dir='/tmp') as tmp:
        tmp=Path(tmp)
        for name,obj in (('cases.json',cases),('adapters.json',adapters),('experiment.json',provenance)):
            (tmp/name).write_text(json.dumps(obj))
        argv=['buyer_diagnostics.py','--versions','operations_reasoning','--thinking','--persona-indices','3',
            '--seeds','0','--cases-file',str(tmp/'cases.json'),'--adapters-file',str(tmp/'adapters.json'),
            '--experiment-manifest',str(tmp/'experiment.json'),'--gpus',args.gpus,'--port',str(args.port)]
        if args.prepare_only:argv.append('--prepare-only')
        previous=sys.argv
        try:sys.argv=argv;folder=evaluate()
        finally:sys.argv=previous
    if not args.prepare_only:
        write_review(folder);print(f'Read: {folder}/learning_sanity_review.md',flush=True)
    return folder


if __name__=='__main__':main()
