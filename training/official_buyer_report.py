"""Readable baseline-versus-checkpoint review from saved official SFT evidence."""
import json
from pathlib import Path


def write_review(run):
    run=Path(run);cfg=json.loads((run/'train_config.json').read_text());finished=json.loads((run/'finished.json').read_text())
    evidence={}
    reference=Path(cfg.get('reference_run',run))/'native_step_0000.json'
    if reference.exists():
        row=json.loads(reference.read_text())['summary'];evidence[row['step']]=row
    for path in sorted(run.glob('native_step_*.json')):
        row=json.loads(path.read_text())['summary'];evidence[row['step']]=row
    if finished['status']!='CPU_mock' and not evidence:raise ValueError('Missing native generation evidence')
    lines=['# Official Buyer baseline review','',f'Status: {finished["status"]}',
        ('Evaluation only: existing base and checkpoints, zero training updates; corrected end-of-turn stopping.' if cfg.get('evaluation_only') else
         f'Target: {cfg["train_examples"]} official training examples, {cfg["max_steps"]} optimizer steps, one epoch.'),
        '', '| Step | Official validation NLL | Current-rule positives /8 | Current-rule both /8 | Cash errors | Official held-out snapshot valid | Training-subset snapshot valid |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for step,row in sorted(evidence.items()):
        c=row['native_behavior'];o=row['official_free_generation'];t=row['train_free_generation']
        lines.append(f'| {step} | {row["completion_loss"]:.4f} | {c["positive_captured"]} | {c["both_correct"]} | {c["unaffordable"]} | {o["snapshot_constraints_passed"]}/{o["calls"]} | {t["snapshot_constraints_passed"]}/{t["calls"]} |')
    lines+=['',f'Development candidate: {finished["development_candidate"]}',
        '', 'Step0 is the original base with an initially zero-effect LoRA; native generations receive no gold reasoning.',
        'Official held-out rows preserve old-world contexts. Current-rule safety probes are synthetic DEVELOPMENT checks, never training examples.',
        'The training subset is unseen by the model at step0 and is guaranteed covered by the first saved checkpoint.',
        'Snapshot validity allows alternative profitable actions and is not full engine execution, market surplus or course grade.',
        'No automatic adapter promotion or full-market run. See native_step_*.json for raw outputs and individual errors.']
    (run/'training_review.md').write_text('\n'.join(lines)+'\n')
    return run/'training_review.md'
