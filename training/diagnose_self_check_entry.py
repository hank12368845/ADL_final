#!/usr/bin/env python3
"""Format-only reasoning entry intervention; zero training, no supplied answers."""
from collections import Counter
from decimal import Decimal
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT = ROOT/'training/runs/20261120-130432_buyer_self_checks_rueclguj'
PREFIX = '{"cash":'
SOURCES = ('training/diagnose_self_check_entry.py', 'training/buyer_self_checks.py',
    'training/buyer_operations_train.py', 'training/reasoning_checkpoint.py',
    'training/official_buyer_train.py', 'training/step84_repair_plan.py', 'training/buyer_hf_check.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cash_probe(case, text):
    """Read the first CASH number even if later JSON punctuation fails.

    Do not scan private_state, backfill from observation, or silently convert dollars.
    Dollar-only matches are a separate unit-confusion diagnostic, never cents credit.
    """
    from training.official_buyer_data import cents
    body = text.split('</think>', 1)[0].removeprefix('<think>').strip()
    match = re.match(r'^\{\s*"cash"\s*:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s|[,}\]]|$)', body)
    got = json.loads(match[1]) if match else None
    expected = cents(case['observation']['you']['cash'])
    dollars = match is not None and Decimal(match[1]) == Decimal(str(case['observation']['you']['cash']))
    correct = type(got) is int and got == expected
    return dict(number_written=match is not None, value=got, cents_correct=correct,
        matches_visible_dollars=dollars, dollars_only=dollars and not correct,
        zero_unit_ambiguous=expected == 0 and correct)


def make_prompt(tokenizer, case):
    prompt = tokenizer.apply_chat_template(case['messages'], tools=case['tools'], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    if not prompt.endswith('<think>\n'):
        raise ValueError('Unexpected native reasoning boundary')
    return prompt, prompt+PREFIX


def prepare(run):
    from training.buyer_self_checks_data import check_data
    from training.decision_ablation import check_fingerprint
    from transformers import AutoTokenizer
    run = Path(run).resolve(); cfg = json.loads((run/'train_config.json').read_text())
    for name, digest in cfg['training_sources'].items():
        if sha(ROOT/name) != digest: raise ValueError(f'Training source changed: {name}')
    _, _, _, manifest = check_data(cfg)
    if sha(ROOT/cfg['data_dir']/'manifest.json') != cfg['data_manifest_sha256']:
        raise ValueError('Original data manifest changed')
    if json.loads((run/'finished.json').read_text())['status'] != 'complete':
        raise ValueError('Need completed self-check pilot')
    cases = [copy.deepcopy(c) for c in json.loads((ROOT/cfg['data_dir']/'evaluation_cases.json').read_text()) if c['suite'] == 'self_check_seen']
    if len(cases) != 12 or Counter(c['positive'] for c in cases) != {True: 6, False: 6}:
        raise ValueError('Expected the same 12 seen cases, six positive/negative pairs')
    adapters = {'parent84': cfg['parent_adapter'], 'self_check12': str(run/'checkpoints/step_0012')}
    fingerprints = {}; free = {}; evidence = {}
    for label, step in (('parent84', 0), ('self_check12', 12)):
        path = run/f'native_step_{step:04d}.json'; evidence[str(path)] = sha(path)
        lookup = {r['case']: r for r in json.loads(path.read_text())['decisions']}
        free[label] = {c['name']: lookup[c['name']] for c in cases}
        cp = Path(adapters[label]); fingerprints[label] = json.loads((cp/'checkpoint_complete.json').read_text())['adapter_fingerprints']
        check_fingerprint(cp, fingerprints[label])
    t = AutoTokenizer.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
    for c in cases:
        persona = (ROOT/f'agents/example_student/buyer/persona_{c["persona_index"]}.md').read_text().strip()
        actual = c['messages'][0]['content'].split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0].strip()
        if actual != persona: raise ValueError('Current latest persona differs from cached inputs')
        prompt, supplied = make_prompt(t, c)
        c['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
        c['prefill_sha256'] = hashlib.sha256(supplied.encode()).hexdigest()
        ids = t.encode(prompt, add_special_tokens=False); added = t.encode(supplied, add_special_tokens=False)
        if added[:len(ids)] != ids: raise ValueError('Native input token prefix changed')
    jobs = [dict(adapter=label, case=c['name'], rank=c['evaluation_rank']) for label in adapters for c in cases]
    if Counter(j['rank'] for j in jobs) != {0: 12, 1: 12}: raise ValueError('Job route must balance both GPUs')
    return dict(experiment='self_check_entry_v1', run=str(run), prefix=PREFIX, training_updates=0,
        base_model=cfg['base_model'], revision=cfg['revision'],
        model_recipe={k: cfg[k] for k in ('rank', 'alpha', 'dropout', 'target_modules')},
        adapters=adapters, fingerprints=fingerprints, cases=cases, jobs=jobs, free_cached=free,
        sources={name: sha(ROOT/name) for name in SOURCES}, evidence_sha256=evidence,
        data_manifest_sha256=cfg['data_manifest_sha256'], new_gpu_calls=24, max_new_tokens=1024,
        limits='Format-only intervention, not autonomous format learning or grading inference. Prefix contains no numbers, IDs, holdings or eligibility. Twelve seen development cases do not establish market/generalization performance.')


def score_output(case, continuation, tokenizer, truncated, rank):
    from training.step84_repair_plan import assess
    from training.buyer_self_checks import score
    text = PREFIX+continuation
    r = assess(case, 'latest', text, tokenizer, truncated, rank)
    r.update(self_check=score(case, text), cash_probe=cash_probe(case, text), persona_index=case['persona_index'],
        supplied_prefix=PREFIX, continuation_text=continuation)
    return r


def summarize(rows, cases):
    from training.buyer_self_checks import summarize as field_summary
    enriched = []
    for original in rows:
        r = copy.deepcopy(original)
        if 'cash_probe' not in r: r['cash_probe'] = cash_probe(cases[r['case']], r['text'])
        enriched.append(r)
    s = field_summary(enriched)
    s['cash_probe'] = {f: sum(r['cash_probe'][f] for r in enriched) for f in
        ('number_written', 'cents_correct', 'matches_visible_dollars', 'dollars_only', 'zero_unit_ambiguous')}
    return s


def write_report(folder, plan):
    folder = Path(folder); rows = []
    for path in sorted(folder.glob('decisions_rank*.jsonl')):
        rows.extend(json.loads(l) for l in path.read_text().splitlines() if l)
    expected = {(j['adapter'], j['case']): j['rank'] for j in plan['jobs']}
    if len(rows) != 24 or len({(r['adapter'], r['case']) for r in rows}) != 24:
        raise ValueError('Incomplete/repeated diagnostic output')
    if {(r['adapter'], r['case']) for r in rows} != set(expected) or any(r['rank'] != expected[r['adapter'], r['case']] for r in rows):
        raise ValueError('Output case/GPU route changed')
    cases = {c['name']: c for c in plan['cases']}; table = {}; by_persona = {}
    for label in plan['adapters']:
        table[label] = {}; by_persona[label] = {}
        for condition in ('free_cached', 'json_open'):
            chosen = list(plan['free_cached'][label].values()) if condition == 'free_cached' else [r for r in rows if r['adapter'] == label]
            table[label][condition] = summarize(chosen, cases)
            by_persona[label][condition] = {str(p): summarize([r for r in chosen if cases[r['case']]['persona_index'] == p], cases) for p in (1, 2, 3)}
    result = dict(summary=table, personas=by_persona, training_updates=0, limits=plan['limits'])
    (folder/'entry_review.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Format-only entry diagnostic', '', plan['limits'], '',
        'Only `{"cash":` was supplied inside the native open think. All numbers, row identities and decisions were generated by the model. Think was not forcibly closed.',
        'The json_open format score measures completion AFTER a supplied opening; it is not autonomous-format accuracy.', '',
        '| Adapter | Condition | JSON /12 | Cash number /12 | Cash cents correct /12 | Cash dollar-only /12 | All fields exact /12 | Positive /6 | Negative safe /6 | Cash errors | Owned errors | Truncated |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, conditions in table.items():
        for condition, s in conditions.items():
            c = s['cash_probe']
            lines.append(f'| {label} | {condition} | {s["format_valid"]} | {c["number_written"]} | {c["cents_correct"]} | {c["dollars_only"]} | {s["all_fields_exact"]} | {s["captured"]} | {s["negative_safe"]} | {s["cash_errors"]} | {s["owned_errors"]} | {s["truncated"]} |')
    lines += ['', '| Adapter | Condition | Cash | Total | Held | Current value | Adjusted value | Affordable | Eligible |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, conditions in table.items():
        for condition, s in conditions.items():
            values = ' | '.join(str(s['target_fields'][f])+'/12' for f in ('cash', 'total', 'held', 'next_value', 'adjusted_value', 'affordable', 'eligible'))
            lines.append(f'| {label} | {condition} | {values} |')
    lines += ['', 'Strict JSON field metrics keep missing/malformed outputs in the denominator. The separate cash probe can credit a correct first number even when later punctuation fails. Dollar-only matches are not silently converted into cents correctness.',
        'If format and field accuracy recover together, entry is a supported bottleneck. Valid JSON with wrong numbers/verdicts means entry is insufficient. Tool recovery without verified fields does not establish correct checking.',
        'If the parent also recovers, the intervention may help the original model too; do not attribute all recovery to the 12-step training.',
        'No automatic training, checkpoint promotion or further GPU work. Proposed plain-text line format has CPU parser tests only and has not been trained/evaluated here.']
    (folder/'entry_review.md').write_text('\n'.join(lines)+'\n')
    return result


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.buyer_operations_train import adapter_model
    from training.reasoning_checkpoint import load_weights
    from training.official_buyer_train import generation_stop_ids
    from training.decision_ablation import check_fingerprint
    folder = Path(folder); plan = json.loads((folder/'plan.json').read_text())
    if plan['prefix'] != PREFIX or plan['training_updates'] != 0: raise ValueError('Intervention changed')
    for name, digest in {**{str(ROOT/k): v for k, v in plan['sources'].items()}, **plan['evidence_sha256']}.items():
        if sha(name) != digest: raise ValueError(f'Source/evidence changed: {name}')
    rank = int(os.environ['LOCAL_RANK']); device = torch.device('cuda', rank); torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(plan['devices'][rank]['fraction'], rank)
    print(f'[rank{rank}] Loading base and fixed adapters; ETA estimating.', flush=True)
    base, info = AutoModelForCausalLM.from_pretrained(plan['base_model'], revision=plan['revision'], local_files_only=True,
        dtype=torch.bfloat16, device_map={'': device}, attn_implementation='sdpa', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'): raise ValueError('Incomplete base weights')
    model = adapter_model(base, plan['model_recipe']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    t = AutoTokenizer.from_pretrained(plan['base_model'], revision=plan['revision'], local_files_only=True)
    eos = generation_stop_ids(model, t); jobs = [j for j in plan['jobs'] if j['rank'] == rank]
    cases = {c['name']: c for c in plan['cases']}; current = None; began = time.monotonic()
    (folder/f'hardware_rank{rank}.json').write_text(json.dumps(dict(rank=rank, physical_gpu=plan['devices'][rank]['index'], name=torch.cuda.get_device_name(rank))))
    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream, torch.inference_mode():
        for i, job in enumerate(jobs, 1):
            if job['adapter'] != current:
                current = job['adapter']; check_fingerprint(plan['adapters'][current], plan['fingerprints'][current]); load_weights(model, plan['adapters'][current])
            case = cases[job['case']]; prompt, supplied = make_prompt(t, case)
            if hashlib.sha256(prompt.encode()).hexdigest() != case['prompt_sha256'] or hashlib.sha256(supplied.encode()).hexdigest() != case['prefill_sha256']:
                raise ValueError('Original prompt or format-only prefix changed')
            ids = t(supplied, add_special_tokens=False, return_tensors='pt').to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.generate(**ids, max_new_tokens=1024, do_sample=False, use_cache=True,
                    eos_token_id=eos, pad_token_id=t.pad_token_id, return_dict_in_generate=False)
            tail = out[0, ids.input_ids.shape[1]:]; continuation = t.decode(tail, skip_special_tokens=False)
            truncated = len(tail) >= 1024 and int(tail[-1]) not in eos
            r = score_output(case, continuation, t, truncated, rank)
            r.update(adapter=current, new_generated_tokens=len(tail))
            stream.write(json.dumps(r)+'\n'); stream.flush()
            elapsed = time.monotonic()-began; speed = i/elapsed
            print(f'[rank{rank}] {i}/{len(jobs)} | {current} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {(len(jobs)-i)/speed:.0f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, default=DEFAULT)
    p.add_argument('--gpus', default='0,1'); p.add_argument('--prepare-only', action='store_true'); p.add_argument('--worker', type=Path)
    p.add_argument('--output-dir', type=Path); a = p.parse_args()
    if a.worker: return worker(a.worker)
    print('CPU: verify fixed adapters/latest inputs and same12 cases; prefix contains no answers...', flush=True)
    plan = prepare(a.run)
    from training.local_resources import training_gpu_plan, output_parent, runtime_cache
    parent, note = output_parent(ROOT, a.output_dir, min_free_gib=2.)
    if note: print(note, flush=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_self_check_entry_', dir=parent))
    if a.prepare_only:
        (folder/'plan.json').write_text(json.dumps(plan, indent=1)+'\n')
        print(f'PASS: CPU only; 24 planned calls, no training. Prepared {folder}'); return folder
    cards, skipped, selection = training_gpu_plan(a.gpus, 2)
    plan.update(devices=cards, skipped_gpus=skipped, gpu_selection=selection)
    (folder/'plan.json').write_text(json.dumps(plan, indent=1)+'\n'); cache, cache_note = runtime_cache(ROOT)
    if cache_note: print(cache_note, flush=True)
    print(f'GPUs {[c["index"] for c in cards]}; output/log {folder}; 24 new calls, zero updates; load ETA estimating.', flush=True)
    env = {**os.environ, **cache, 'CUDA_VISIBLE_DEVICES': ','.join(c['index'] for c in cards),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'PYTHONPATH': str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', '')}
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(Path(__file__).resolve()), '--worker', str(folder)]
    began = time.monotonic(); last = -1
    with (folder/'runner.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while chunk := process.stdout.read1(4096):
                message = chunk.decode(errors='replace'); sys.stdout.write(message); sys.stdout.flush(); log.write(message); log.flush()
                done = sum(len(path.read_text().splitlines()) for path in folder.glob('decisions_rank*.jsonl'))
                if done != last:
                    elapsed = time.monotonic()-began; speed = done/elapsed
                    eta = f'{(24-done)/speed:.0f}s' if speed else 'estimating'
                    message = f'[Overall] {done}/24 | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n'
                    print(message, end='', flush=True); log.write(message); log.flush(); last = done
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    (folder/'wrapper_result.json').write_text(json.dumps(dict(exit=code, elapsed_seconds=time.monotonic()-began, training_updates=0)))
    if code: raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    write_report(folder, plan); print(f'Completed. Read {folder}/entry_review.md; no automatic next GPU work.', flush=True)
    return folder


if __name__ == '__main__': main()
