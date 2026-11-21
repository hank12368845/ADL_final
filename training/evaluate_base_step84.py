#!/usr/bin/env python3
"""Same original buyer personas and 36 prompts: bare base vs cached official step84."""
from collections import Counter, defaultdict
import argparse
import codecs
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_REFERENCE = ROOT / 'results/basic_checks/20261120-150124_compact_buyer_g5p78gcy'

from training.evaluate_compact_buyer import assess_case, prompt, review_evidence, sha, stats, summary


def verify_files(plan):
    for path, digest in {**{str(ROOT / k): v for k, v in plan['sources'].items()},
                         **plan['evidence_sha256']}.items():
        if sha(path) != digest:
            raise ValueError(f'Source/evidence changed: {path}')
    from training.decision_ablation import check_fingerprint
    check_fingerprint(plan['cached_adapter'], plan['cached_adapter_fingerprints'])
    for name, expected in plan['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Package version differs from cached control: {name}')


def prepare(reference=DEFAULT_REFERENCE):
    """CPU only. Reconstruct original prompts, not the rejected compact prompts."""
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    reference = Path(reference).resolve()
    ref_path = reference / 'plan.json'
    old = json.loads(ref_path.read_text())
    if old['experiment'] != 'buyer_compact_check_v1' or old['max_new_tokens'] != 1024:
        raise ValueError('Unexpected cached control experiment or generation bound')
    cfg_path = Path(old['run']) / 'train_config.json'
    cfg = json.loads(cfg_path.read_text())
    # The reference pins the exact data, raw cached outputs, code and adapter.
    evidence = {**old['evidence_sha256'], str(ref_path): sha(ref_path), str(cfg_path): sha(cfg_path)}
    sources = {**old['sources'], 'training/evaluate_base_step84.py': sha(__file__)}
    for path, digest in {**{str(ROOT / k): v for k, v in sources.items()}, **evidence}.items():
        if sha(path) != digest:
            raise ValueError(f'Cached evidence changed: {path}')
    if cfg['base_model'] != old['base_model'] or cfg['revision'] != old['revision']:
        raise ValueError('Base revision differs from cached step84')
    t = AutoTokenizer.from_pretrained(old['base_model'], revision=old['revision'], local_files_only=True)
    try:
        generation = GenerationConfig.from_pretrained(old['base_model'], revision=old['revision'], local_files_only=True)
    except OSError:
        # Match from_pretrained's fallback when this revision has no separate generation config.
        config = AutoConfig.from_pretrained(old['base_model'], revision=old['revision'], local_files_only=True)
        generation = GenerationConfig.from_model_config(config)
    from training.official_buyer_train import generation_stop_ids
    from types import SimpleNamespace
    eos = generation_stop_ids(SimpleNamespace(generation_config=generation), t)
    originals = {}
    for i in (1, 2, 3):
        path = ROOT / f'agents/example_student/buyer/persona_{i}.md'
        text = path.read_text()
        if text != old['original_personas'][str(i)] or not 0 < len(text) <= 8000:
            raise ValueError(f'Original persona changed: {path}')
        originals[str(i)] = text
        evidence[str(path)] = sha(path)
    source = ROOT / cfg['data_dir'] / 'evaluation_cases.json'
    cached_path = Path(old['run']) / 'native_step_0000.json'
    raw = {r['case']: r for r in json.loads(cached_path.read_text())['decisions']}
    reference_cases = {c['name']: c for c in old['cases']}
    reference_rows = {r['case']: r for r in old['baseline']}
    cases, baseline, lengths = [], [], []
    for c in json.loads(source.read_text()):
        if c['suite'] not in ('self_check_seen', 'self_check_validation'):
            continue
        ref = reference_cases[c['name']]
        # Exclude the teacher response from the inference plan entirely.
        c = {k: v for k, v in c.items() if k not in ('answer', 'label_action')}
        persona = c['messages'][0]['content'].split('<<<PERSONA_SECTION>>>', 1)[1].split('<<<END_PERSONA_SECTION>>>', 1)[0]
        if persona.strip() != originals[str(c['persona_index'])].strip():
            raise ValueError('Case does not use original latest persona')
        text = prompt(t, c)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != ref['original_prompt_sha256'] or c['evaluation_rank'] != ref['evaluation_rank']:
            raise ValueError('Original prompt or cached case route changed')
        c['prompt_sha256'] = digest
        saved, previous = raw[c['name']], reference_rows[c['name']]
        if saved['text'] != previous['text'] or saved['truncated'] != previous['truncated']:
            raise ValueError('Cached step84 raw output differs from reference')
        r = assess_case(c, saved['text'], t, saved['truncated'], c['evaluation_rank'], True, 'step84_cached')
        for field in ('valid', 'captured', 'cash_error', 'errors', 'tool', 'arguments'):
            if r[field] != previous[field]:
                raise ValueError(f'Cached scoring changed: {c["name"]}, {field}')
        cases.append(c)
        baseline.append(r)
        lengths.append(dict(case=c['name'],tokens=len(t.encode(text, add_special_tokens=False))))
    if Counter(c['suite'] for c in cases) != {'self_check_seen': 12, 'self_check_validation': 24}:
        raise ValueError('Need exactly seen12 + development24')
    if Counter(c['evaluation_rank'] for c in cases) != {0: 18, 1: 18}:
        raise ValueError('Cached two-GPU routing changed')
    plan = dict(experiment='buyer_base_vs_step84_v1', reference=str(reference),
                base_model=old['base_model'], revision=old['revision'],
                cached_adapter=old['adapter'], cached_adapter_fingerprints=old['fingerprints'],
                loaded_adapter=None, original_personas=originals, cases=cases, baseline=baseline,
                jobs=[dict(case=c['name'], rank=c['evaluation_rank']) for c in cases],
                prompt_lengths=lengths, versions=cfg['versions'], sources=sources, evidence_sha256=evidence,
                generation=dict(thinking=True, do_sample=False, max_new_tokens=1024,
                                eos_token_id=eos, attention='sdpa', dtype='bfloat16'),
                new_gpu_calls=36, cached_calls=36, training_updates=0,
                limits='Same original personas, observations, histories, tools, pinned base revision and greedy thinking-ON decoding. '
                       'Only new generation uses the bare base, without any adapter. Seen12 and reused development24 are not a sealed test or market score. '
                       'Cached step84 is rescored, not regenerated. No automatic winner, deployment, training or market run.')
    verify_files(plan)
    return plan


def load_bare_base(plan, device, loader):
    """Separate loader allows a CPU test to prove that no adapter is attached."""
    import torch
    model, info = loader(plan['base_model'], revision=plan['revision'], local_files_only=True,
                         dtype=torch.bfloat16, device_map={'': device}, attn_implementation='sdpa',
                         output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError('Incomplete base model')
    if getattr(model, 'peft_config', None) or any('lora_' in name for name, _ in model.named_parameters()):
        raise ValueError('Bare-base control unexpectedly contains an adapter')
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def paired_stats(rows):
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['pair_id']][r['positive']] = r
    if any(set(p) != {True, False} for p in pairs.values()):
        raise ValueError('Incomplete positive/negative pair')
    return dict(pairs=len(pairs), both_correct=sum(p[True]['captured'] and p[False]['valid'] for p in pairs.values()))


def write_report(folder, plan):
    from transformers import AutoTokenizer
    folder = Path(folder)
    verify_files(plan)
    raw = [json.loads(line) for path in sorted(folder.glob('decisions_rank*.jsonl'))
           for line in path.read_text().splitlines() if line]
    expected = {j['case']: j['rank'] for j in plan['jobs']}
    if len(raw) != 36 or len({r['case'] for r in raw}) != 36 or {r['case'] for r in raw} != set(expected):
        raise ValueError('Incomplete/repeated outputs; no comparison published')
    t = AutoTokenizer.from_pretrained(plan['base_model'], revision=plan['revision'], local_files_only=True)
    cases = {c['name']: c for c in plan['cases']}
    rows = []
    for saved in raw:
        if saved['rank'] != expected[saved['case']] or saved.get('loaded_adapter', 'missing') is not None:
            raise ValueError('GPU route changed or new output was not bare base')
        c = cases[saved['case']]
        if saved['prompt_sha256'] != c['prompt_sha256']:
            raise ValueError('Generated prompt SHA differs from cached step84 input')
        r = assess_case(c, saved['text'], t, saved['truncated'], saved['rank'], False, 'base')
        r.update(new_generated_tokens=saved['new_generated_tokens'], prompt_sha256=saved['prompt_sha256'], loaded_adapter=None)
        rows.append(r)
    old = {r['case']: r for r in plan['baseline']}
    results = {}
    for label, items in (('step84_cached', plan['baseline']), ('base', rows)):
        results[label] = dict(total=stats(items), suites=summary(items), paired=paired_stats(items),
                              personas={str(i): stats([r for r in items if r['persona_index'] == i]) for i in (1, 2, 3)})
    delta = {key: results['base']['total'][key] - results['step84_cached']['total'][key]
             for key in results['base']['total']}
    changes = dict(base_fixes=[r['case'] for r in rows if r['valid'] and not old[r['case']]['valid']],
                   base_regressions=[r['case'] for r in rows if not r['valid'] and old[r['case']]['valid']],
                   base_extra_captures=[r['case'] for r in rows if r['captured'] and not old[r['case']]['captured']],
                   base_lost_captures=[r['case'] for r in rows if not r['captured'] and old[r['case']]['captured']])
    result = dict(summary=results, delta_base_minus_step84=delta, changes=changes,
                  semantic_review='pending', automatic_next_gpu_work=False, selected_market_model=None,
                  training_updates=0, new_gpu_calls=36, cached_calls=36, limits=plan['limits'])
    (folder / 'base_step84_review.json').write_text(json.dumps(result, indent=2) + '\n')
    (folder / 'rescored_decisions.json').write_text(json.dumps(rows, indent=2) + '\n')
    packet = [review_evidence(cases[r['case']], r) for r in plan['baseline'] + rows]
    (folder / 'semantic_review.json').write_text(json.dumps(packet, indent=2) + '\n')
    lines = ['# 原persona：裸底座 vs 官方step84', '', plan['limits'], '',
             '固定36個相同prompt；step84為cached原始輸出、裸底座36次新推論。沒有訓練。',
             'Valid含工具與經濟約束，並非僅engine拒收率。State格式另列，不當引擎違規。',
             '語意證據尚待逐題閱讀，不用關鍵字推定草稿全對；不因任一指標勝出自動選模型。', '',
             '| Model | Suite | Valid | Positive capture | Negative safe | Cash errors | Owned errors | Unlisted | No tool | Truncated |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, data in results.items():
        for suite, s in [('total', data['total']), *data['suites'].items()]:
            lines.append(f'| {label} | {suite} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | '
                         f'{s["negative_safe"]}/{s["negative_total"]} | {s["cash_errors"]} | {s["owned_errors"]} | '
                         f'{s["not_current_board"]} | {s["no_tool"]} | {s["truncated"]} |')
    lines += ['', '| Persona | Model | Valid | Capture | Negative safe | Cash | Owned |',
              '|---|---|---:|---:|---:|---:|---:|']
    for i in ('1', '2', '3'):
        for label, data in results.items():
            s = data['personas'][i]
            lines.append(f'| p{i} | {label} | {s["valid"]}/{s["calls"]} | {s["captured"]}/{s["positive_total"]} | '
                         f'{s["negative_safe"]}/{s["negative_total"]} | {s["cash_errors"]} | {s["owned_errors"]} |')
    lines += ['', '同一pair的「正例買到＋負例安全」：']
    for label, data in results.items():
        lines.append(f'- {label}: {data["paired"]["both_correct"]}/{data["paired"]["pairs"]}')
    lines += ['', f'裸底座修好 {len(changes["base_fixes"])} 題、退步 {len(changes["base_regressions"])} 題。',
              '完整JSON保留每suite／persona／board與offer分項、變動題目與差值；原始草稿見decisions_rank*.jsonl。',
              '下一步先讀原始失敗與收益取捨，再選固定基線做一場5seller×10buyer、5round×6turn市場診斷。',
              '不續訓、不改persona、不部署、不自動啟動市場。']
    (folder / 'base_step84_review.md').write_text('\n'.join(lines) + '\n')
    return result


def worker(folder):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.official_buyer_train import generation_stop_ids
    folder = Path(folder)
    p = json.loads((folder / 'plan.json').read_text())
    verify_files(p)
    if p['loaded_adapter'] is not None or p['new_gpu_calls'] != 36 or p['training_updates'] != 0:
        raise ValueError('Inference-only bare-base plan changed')
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(p['devices'][rank]['fraction'], rank)
    print(f'[rank{rank}] Loading bare base WITHOUT LoRA; first-call warm-up ETA estimating.', flush=True)
    model = load_bare_base(p, device, AutoModelForCausalLM.from_pretrained)
    t = AutoTokenizer.from_pretrained(p['base_model'], revision=p['revision'], local_files_only=True)
    eos = generation_stop_ids(model, t)
    if eos != p['generation']['eos_token_id']:
        raise ValueError('Generation stop IDs changed')
    cases = {c['name']: c for c in p['cases']}
    jobs = [j for j in p['jobs'] if j['rank'] == rank]
    (folder / f'hardware_rank{rank}.json').write_text(json.dumps(dict(
        rank=rank, physical_gpu=p['devices'][rank]['index'], name=torch.cuda.get_device_name(rank),
        loaded_adapter=None, parameter_count=sum(x.numel() for x in model.parameters()), eos_token_id=eos)))
    began = time.monotonic()
    with (folder / f'decisions_rank{rank}.jsonl').open('w') as stream, torch.inference_mode():
        for i, job in enumerate(jobs, 1):
            c = cases[job['case']]
            text = prompt(t, c)
            if hashlib.sha256(text.encode()).hexdigest() != c['prompt_sha256']:
                raise ValueError('Prompt changed')
            ids = t(text, add_special_tokens=False, return_tensors='pt').to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.generate(**ids, max_new_tokens=1024, do_sample=False, use_cache=True,
                                     eos_token_id=eos, pad_token_id=t.pad_token_id, return_dict_in_generate=False)
            tail = out[0, ids.input_ids.shape[1]:]
            raw = t.decode(tail, skip_special_tokens=False)
            r = assess_case(c, raw, t, len(tail) >= 1024 and int(tail[-1]) not in eos, rank, False, 'base')
            r.update(new_generated_tokens=len(tail), loaded_adapter=None, prompt_sha256=c['prompt_sha256'])
            stream.write(json.dumps(r) + '\n')
            stream.flush()
            elapsed = time.monotonic() - began
            speed = i / elapsed
            print(f'[rank{rank}] {i}/{len(jobs)} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | '
                  f'ETA {(len(jobs)-i)/speed:.0f}s', flush=True)


def stream_progress(proc, folder, log):
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    began = time.monotonic()
    last_print = 0.
    last_done = -1
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            ended = False
            for key, _ in selector.select(timeout=5):
                chunk = os.read(key.fd, 4096)
                if not chunk:
                    ended = True
                    break
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text)
                log.flush()
            done = sum(len(f.read_text().splitlines()) for f in folder.glob('decisions_rank*.jsonl'))
            now = time.monotonic()
            if done != last_done or now-last_print >= 10 or ended:
                elapsed = now-began
                speed = done/elapsed if elapsed else 0.
                eta = f'{(36-done)/speed:.0f}s' if speed else 'estimating'
                phase = 'decisions' if done else 'loading / warm-up / first decision'
                filled = min(24, int(done * 24 / 36))
                bar = '#' * filled + '-' * (24 - filled)
                line = f'[Overall] [{bar}] {done}/36 | {phase} | elapsed {elapsed:.0f}s | {speed:.3f} calls/s | ETA {eta}\n'
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                last_done, last_print = done, now
            if ended:
                break
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--report', type=Path)
    a = parser.parse_args()
    if a.worker:
        return worker(a.worker)
    if a.report:
        p = json.loads((a.report / 'plan.json').read_text())
        return write_report(a.report, p)
    print('CPU: verify original personas, identical36 prompts, cached step84, versions and checksums; ETA estimating.', flush=True)
    plan = prepare(a.reference)
    from training.local_resources import output_parent, runtime_cache, training_gpu_plan
    parent, note = output_parent(ROOT, a.output_dir, min_free_gib=2.)
    if note:
        print(note, flush=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_base_step84_', dir=parent))
    (folder / 'personas').mkdir()
    for i, text in plan['original_personas'].items():
        (folder / 'personas' / f'persona_{i}.md').write_text(text)
    if a.prepare_only:
        (folder / 'plan.json').write_text(json.dumps(plan, indent=1)+'\n')
        print(f'PASS CPU only:36 base calls planned,36 cached step84,zero updates; {folder}', flush=True)
        return folder
    cards, skipped, selection = training_gpu_plan(a.gpus, required=2)
    cache, cache_note = runtime_cache(ROOT)
    if cache_note:
        print(cache_note, flush=True)
    plan.update(devices=cards, skipped_gpus=skipped, gpu_selection=selection, cache_environment=cache)
    (folder / 'plan.json').write_text(json.dumps(plan, indent=1)+'\n')
    print(f'Bare base only,NO LoRA,NO training;36 new calls,18/card; GPUs {[c["index"] for c in cards]}; '
          f'output/log {folder}; loading ETA estimating.', flush=True)
    env = {**os.environ, **cache, 'CUDA_VISIBLE_DEVICES': ','.join(c['index'] for c in cards),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
           'PYTHONPATH': str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', '')}
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
           str(Path(__file__).resolve()), '--worker', str(folder)]
    began = time.monotonic()
    with (folder / 'runner.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = stream_progress(proc, folder, log)
        except BaseException:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    (folder / 'wrapper_result.json').write_text(json.dumps(dict(
        exit=code, elapsed_seconds=time.monotonic()-began, training_updates=0)))
    if code:
        raise SystemExit(f'FAIL: worker exit={code}; see {folder}/runner.log')
    write_report(folder, plan)
    print(f'Completed. Read {folder}/base_step84_review.md; no automatic next GPU work.', flush=True)
    return folder


if __name__ == '__main__':
    main()
