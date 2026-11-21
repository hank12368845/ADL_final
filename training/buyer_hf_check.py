#!/usr/bin/env python3
"""User-started HF/PEFT step8 behavior check; no training and no vLLM server."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
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
DEFAULT_SOURCE = ROOT/'results/basic_checks/20261118-210625_buyer_diagnostic_8svaz4mq'


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1,
                               default=lambda x: sorted(x) if isinstance(x, set) else str(x))+'\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse_raw(tokenizer, text, tools):
    """Use the installed vLLM qwen3_coder parser; never repair a model action."""
    from vllm.tool_parsers import ToolParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    request = ChatCompletionRequest(model='Qwen/Qwen3.5-4B', messages=[], tools=tools)
    parser = ToolParserManager.get_tool_parser('qwen3_coder')(tokenizer, tools=request.tools)
    parsed = parser.extract_tool_calls(text, request)
    calls = parsed.tool_calls
    result = {'tool': None, 'arguments': {}, 'parsed_calls': len(calls), 'parse_error': None}
    if len(calls) != 1:
        result['parse_error'] = f'Expected exactly one tool call, got {len(calls)}'
        return result
    function = calls[0].function
    try:
        args = json.loads(function.arguments)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments are not an object')
        result.update(tool=function.name, arguments=args)
    except (ValueError, TypeError) as exc:
        result['parse_error'] = str(exc)
    return result


def target_encoding(tokenizer, messages, tools, tool, args):
    """Official completion + disjoint token regions; tokens crossing a span count in that span."""
    from training.buyer_operations_data import assistant
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages+[assistant(tool, args)], tools=tools,
                                         tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError('Assistant completion is not a continuation of the prompt')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded['input_ids'][:len(prefix)] != prefix:
        raise ValueError('Target token boundary differs from training mask')
    text = full[len(prompt):]
    name = re.search(r'<function=([^>]+)>', text)
    if not name or name.group(1) != tool:
        raise ValueError('Unexpected official tool syntax')
    spans = [('tool_name', name.start(1), name.end(1))]
    for match in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', text, re.DOTALL):
        field = match.group(1)
        category = 'private_state' if field == 'private_state' else 'trade_arguments'
        spans.append((category, match.start(2), match.end(2)))
    regions = []
    for start, end in encoded['offset_mapping'][len(prefix):]:
        a, b = start-len(prompt), end-len(prompt)
        regions.append(next((label for label, lo, hi in spans if a < hi and b > lo), 'format'))
    if 'tool_name' not in regions or 'private_state' not in regions:
        raise ValueError('Missing decision or memory token span')
    return {'input_ids': encoded['input_ids'], 'prompt_length': len(prefix),
            'regions': regions, 'completion': text}


def split_loss(token_nll, regions):
    if len(token_nll) != len(regions) or not token_nll:
        raise ValueError('Loss/token-region alignment failed')
    import math
    if not all(math.isfinite(v) for v in token_nll):
        raise ValueError('Nonfinite teacher-forced loss')
    return {label: {'tokens': regions.count(label),
                    'sum_nll': sum(v for v, r in zip(token_nll, regions) if r == label),
                    'mean_nll': sum(v for v, r in zip(token_nll, regions) if r == label)/regions.count(label)}
            for label in sorted(set(regions))}


def score_target(model, encoded, device):
    import torch
    ids = torch.tensor([encoded['input_ids']], dtype=torch.long, device=device)
    p = encoded['prompt_length']
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                logits_to_keep=torch.arange(p-1, ids.shape[1]-1, device=device), use_cache=False)
    target = ids[:, p:]
    if out.logits.shape[:2] != target.shape:
        raise ValueError('Teacher-forced logits do not align with completion targets')
    values = torch.nn.functional.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                               target.reshape(-1), reduction='none').tolist()
    return {'mean_nll': sum(values)/len(values), 'regions': split_loss(values, encoded['regions']),
            'token_nll': values}


def verify_loaded_adapter(model, path):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open
    actual = get_peft_model_state_dict(model)
    with safe_open(Path(path)/'adapter_model.safetensors', framework='pt', device='cpu') as source:
        if set(actual)!=set(source.keys()):
            raise ValueError('HF adapter keys differ from the checkpoint')
        for name,value in actual.items():
            if not torch.equal(value.detach().cpu(), source.get_tensor(name)):
                raise ValueError(f'HF loaded adapter tensor differs: {name}')
    return {'tensors_checked':len(actual), 'dtypes':dict(Counter(str(v.dtype) for v in actual.values()))}


def loaded_adapter_metadata(model, path):
    """Attention.scaling is a float; only actual PEFT LoRA layers own adapter scales."""
    from peft.tuners.lora.layer import LoraLayer
    config = model.peft_config['default']
    scales = {name: float(module.scaling['default']) for name, module in model.named_modules()
              if isinstance(module, LoraLayer) and 'default' in module.scaling}
    if not scales:
        raise ValueError('No loaded LoRA layer scaling found')
    return {'r': config.r, 'alpha': config.lora_alpha,
            'tensor_check': verify_loaded_adapter(model, path), 'scaling': scales}


def prepare(source):
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    from huggingface_hub import try_to_load_from_cache
    from adlmarket.config import check_adapter
    from training.buyer_diagnostics import replace_persona
    from training.buyer_operations_data import normalized
    from training.chat_payload import api_messages
    from vllm.entrypoints.chat_utils import _postprocess_messages
    from vllm.renderers.hf import safe_apply_chat_template
    source = Path(source).resolve()
    plan = json.loads((source/'plan.json').read_text())
    status = json.loads((source/'evaluation_status.json').read_text())
    if status['status'] != 'complete' or plan.get('analysis') != 'history_ablation':
        raise ValueError('Need a completed three-checkpoint history-ablation run')
    if len(plan['personas']) != 1 or plan['seeds'] != [0, 1]:
        raise ValueError('Expected one fixed persona and seeds 0,1')
    adapter = Path(plan['adapters']['step8']).resolve()
    cfg = json.loads((adapter.parents[1]/'train_config.json').read_text())
    check_adapter('step8', adapter, cfg['base_model'], {'model.layers.':'model.language_model.layers.'})
    fingerprints = {kind: sha(adapter/name) for kind, name in
                    [('weights_sha256', 'adapter_model.safetensors'), ('config_sha256', 'adapter_config.json')]}
    if fingerprints != plan['experiment_manifest']['adapters']['step8']:
        raise ValueError('Step8 weights/config differ from the completed vLLM experiment')
    config_path = try_to_load_from_cache(cfg['base_model'], 'config.json', revision=cfg['revision'])
    index_path = try_to_load_from_cache(cfg['base_model'], 'model.safetensors.index.json', revision=cfg['revision'])
    if not isinstance(config_path, str) or not isinstance(index_path, str):
        raise ValueError('Missing pinned base config/weights in local cache')
    index = json.loads(Path(index_path).read_text())
    if not all((Path(index_path).parent/name).is_file() for name in set(index['weight_map'].values())):
        raise ValueError('Base weight shard missing from local cache')
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
    config = AutoConfig.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
    try:
        generation = GenerationConfig.from_pretrained(cfg['base_model'], revision=cfg['revision'], local_files_only=True)
        generation_origin = 'cached generation_config.json'
    except OSError:
        generation = GenerationConfig.from_model_config(config)
        generation_origin = 'from model config; no cached generation_config.json'
    # vLLM uses neutral defaults unless an option appears in the model's diff config.
    diff = generation.to_diff_dict()
    eos = generation.eos_token_id or []
    eos = [eos] if isinstance(eos, int) else list(eos)
    eos = sorted(set(eos+[tokenizer.eos_token_id]))
    sampling = {'do_sample': True, 'temperature': .7, 'top_p': .95,
                'top_k': diff.get('top_k', 0), 'repetition_penalty': diff.get('repetition_penalty', 1.),
                'min_p': diff.get('min_p', 0.), 'eos_token_id': eos,
                'pad_token_id': tokenizer.pad_token_id, 'max_new_tokens': 8192}
    persona = next(iter(plan['personas'].values()))
    cases = json.loads((source/'cases.json').read_text())
    decisions = [json.loads(s) for s in (source/'scored_decisions.jsonl').read_text().splitlines() if s]
    jobs = []
    for case in cases:
        if case['expected_tool'] == 'wait':
            continue
        messages = replace_persona(case['messages'], persona, 'clean_fresh')
        text = tokenizer.apply_chat_template(normalized(messages), tools=case['tools'], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        wire = api_messages(messages)
        body_sha = hashlib.sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest()
        conversation = copy.deepcopy(wire)
        _postprocess_messages(conversation)
        other = safe_apply_chat_template(None, tokenizer, conversation, tools=case['tools'],
                 chat_template=tokenizer.get_chat_template(tools=case['tools']), tokenize=False,
                 add_generation_prompt=True, enable_thinking=False)
        if other != text:
            raise ValueError('HF prompt differs from installed vLLM chat-template renderer')
        reference = target_encoding(tokenizer, messages, case['tools'], case['expected_tool'], case['reference_arguments'])
        wait = target_encoding(tokenizer, messages, case['tools'], 'wait',
                    {'private_state': json.dumps({'bad': [], 'pending': None, 'last_buy': None}, separators=(',', ':'))})
        # Exercise exactly the production parser on an official teacher completion on CPU.
        if parse_raw(tokenizer, reference['completion'], case['tools'])['tool'] != case['expected_tool']:
            raise ValueError('Official target fails the installed qwen3_coder parser')
        for seed in plan['seeds']:
            matches = [r for r in decisions if r['adapter']=='step8' and r['context']=='clean_fresh'
                       and r['case']==case['name'] and r['seed']==seed]
            if len(matches)!=1 or matches[0]['outcome']!='accepted' or matches[0]['input_sha256']!=body_sha:
                raise ValueError('Missing or mismatched vLLM reference input')
            jobs.append({'id': f'{case["name"]}_seed{seed}', 'case': case, 'seed': seed,
                         'messages': messages, 'prompt': text,
                         'prompt_ids': tokenizer.encode(text, add_special_tokens=False),
                         'input_sha256': body_sha, 'reference': reference, 'wait': wait,
                         'vllm': {k: matches[0][k] for k in ('tool','arguments','gpu','operation_capture')}})
    if len(jobs)!=8 or len({j['case']['name'] for j in jobs})!=4:
        raise ValueError('Expected four distinct positive cases x two seeds = eight generations')
    provenance = {'source': str(source), 'adapter': str(adapter), 'adapter_sha256': fingerprints,
                  'base_model': cfg['base_model'], 'revision': cfg['revision'], 'base_snapshot': str(Path(config_path).parent),
                  'template_sha256': hashlib.sha256(tokenizer.get_chat_template(tools=jobs[0]['case']['tools']).encode()).hexdigest(),
                  'sampling': sampling, 'generation_config_origin': generation_origin,
                  'hf_dtype': 'bfloat16', 'hf_autocast': 'bfloat16', 'hf_attention': 'sdpa',
                  'calls': 8, 'unique_cases': 4, 'training_updates': 0,
                  'note': 'Behavior comparison, not numerical backend parity. Same seed does not imply identical random draws across libraries.'}
    return jobs, provenance


def worker(folder, rank):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    from training.buyer_diagnostics import evaluate
    folder = Path(folder)
    plan = json.loads((folder/'plan.json').read_text())
    jobs = [j for j in json.loads((folder/'jobs.json').read_text()) if j['worker']==rank]
    status_path = folder/f'progress_{rank}.json'
    state = {'stage': 'loading base / adapter; ETA estimating', 'units': 0, 'generations': 0,
             'total_units': len(jobs)*5, 'gpu': plan['devices'][rank]['index'], 'generated_tokens': 0}
    def update(**values):
        state.update(values, time=time.time()); dump(status_path, state)
    update()
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(plan['devices'][rank]['fraction'], 0)
        provenance = plan['provenance']; device = torch.device('cuda', 0)
        tokenizer = AutoTokenizer.from_pretrained(provenance['base_model'], revision=provenance['revision'], local_files_only=True)
        base, loading = AutoModelForCausalLM.from_pretrained(provenance['base_model'], revision=provenance['revision'],
                      local_files_only=True, dtype=torch.bfloat16, device_map={'': device},
                      attn_implementation='sdpa', output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise ValueError(f'Incomplete training-side base load: {loading}')
        dump(folder/f'loading_{rank}.json', loading)
        model = PeftModel.from_pretrained(base, provenance['adapter'], is_trainable=False)
        model.eval()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Inference model unexpectedly has trainable parameters')
        dump(folder/f'loaded_adapter_{rank}.json', {**loaded_adapter_metadata(model, provenance['adapter']),
            'device': torch.cuda.get_device_name(0), 'physical_gpu': state['gpu']})
        started = time.time(); update(stage='ready', work_started=started)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), (folder/f'decisions_{rank}.jsonl').open('w') as stream:
            for job in jobs:
                scores = {}
                for label in ('base', 'step8'):
                    scores[label] = {}
                    for candidate in ('reference', 'wait'):
                        update(stage=f'{job["id"]}: scoring {label}/{candidate}')
                        with model.disable_adapter() if label=='base' else nullcontext():
                            scores[label][candidate] = score_target(model, job[candidate], device)
                        update(units=state['units']+1)
                    a, b = job['reference'], job['wait']
                    aa, bb = a['input_ids'][a['prompt_length']:], b['input_ids'][b['prompt_length']:]
                    first = next(i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y)
                    scores[label]['branch_logprob_reference_minus_wait'] = (
                        scores[label]['wait']['token_nll'][first]-scores[label]['reference']['token_nll'][first])
                    scores[label]['branch_token_ids'] = [aa[first], bb[first]]
                class TokenProgress:
                    first = True
                    last = 0.
                    def put(self, value):
                        if self.first:
                            self.first = False; return
                        state['generated_tokens'] += value.numel()
                        if time.monotonic()-self.last > 1:
                            self.last = time.monotonic(); update()
                    def end(self): update()
                torch.manual_seed(job['seed']); torch.cuda.manual_seed_all(job['seed'])
                ids = torch.tensor([job['prompt_ids']], device=device, dtype=torch.long)
                update(stage=f'{job["id"]}: generate', generated_tokens=0)
                # Start from a neutral GenerationConfig, overriding HF's default top_k=50.
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    generation_config=GenerationConfig(**provenance['sampling']),
                    use_cache=True, logits_to_keep=1, streamer=TokenProgress())
                new_ids = generated[0, ids.shape[1]:].tolist()
                raw = tokenizer.decode(new_ids, skip_special_tokens=False)
                parsed = parse_raw(tokenizer, raw, job['case']['tools'])
                truncated = len(new_ids)>=provenance['sampling']['max_new_tokens'] and new_ids[-1] not in provenance['sampling']['eos_token_id']
                tool = None if truncated else parsed['tool']
                args = parsed['arguments'] if tool else {}
                evaluation = evaluate(job['case'], tool, args)
                from training.diagnostic_scores import ACTION_ERRORS, POLICY_ERRORS
                capture = tool in ('buy_listed', 'accept_offer') and not set(evaluation['errors']) & (ACTION_ERRORS|POLICY_ERRORS)
                row = {'id':job['id'], 'case':job['case']['name'], 'seed':job['seed'], 'worker':rank,
                       'gpu':state['gpu'], 'input_sha256':job['input_sha256'], 'generated_token_ids':new_ids,
                       'raw_text':raw, 'parsed':parsed, 'truncated':truncated, 'operation_capture':capture,
                       'evaluation':evaluation, 'vllm':job['vllm'], 'scores':scores}
                stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                update(units=state['units']+1, generations=state['generations']+1, stage='generation complete')
        update(stage='complete')
    except BaseException as exc:
        update(stage='error', error=repr(exc)); raise


def report(folder):
    folder = Path(folder)
    jobs = json.loads((folder/'jobs.json').read_text())
    rows = [json.loads(line) for path in folder.glob('decisions_*.jsonl') for line in path.read_text().splitlines() if line]
    expected = {j['id']:j for j in jobs}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):
        raise ValueError('Incomplete/duplicate HF generation matrix')
    for row in rows:
        job = expected[row['id']]
        if row['input_sha256']!=job['input_sha256'] or row['gpu']!=job['gpu'] or row['vllm']!=job['vllm']:
            raise ValueError('HF/vLLM reference input or worker route changed')
    tools = Counter(r['parsed']['tool'] for r in rows)
    counts = {'valid_evaluation':True, 'generations':len(rows), 'unique_cases':len({r['case'] for r in rows}),
              'gpu_calls':dict(Counter(r['gpu'] for r in rows)), 'tools':dict(tools),
              'positive_captured':sum(r['operation_capture'] for r in rows),
              'parse_failures':sum(bool(r['parsed']['parse_error']) for r in rows),
              'truncated':sum(r['truncated'] for r in rows),
              'vllm_waits':sum(r['vllm']['tool']=='wait' for r in rows)}
    if tools.get('wait',0)==len(rows) and not counts['truncated']:
        finding = 'HF also waits on all selected positive probes; vLLM is not required for the observed waiting. This does not identify a unique SFT cause or prove numerical backend parity.'
    elif counts['positive_captured']:
        finding = 'HF captures positive trades missed by vLLM; investigate backend/config/template differences, then compare controlled logits or greedy outputs. Stochastic generation alone does not identify LoRA scaling as the cause.'
    else:
        finding = 'Mixed, malformed or truncated outputs; inspect raw text and token losses before attributing a cause.'
    counts['finding'] = finding
    loss_summary = {}
    for label in ('base','step8'):
        loss_summary[label] = {}
        for candidate in ('reference','wait'):
            regions = {region for r in rows for region in r['scores'][label][candidate].get('regions', {})}
            loss_summary[label][candidate] = {}
            for region in regions:
                cells = [r['scores'][label][candidate]['regions'][region] for r in rows
                         if region in r['scores'][label][candidate].get('regions', {})]
                n = sum(c['tokens'] for c in cells); total = sum(c['sum_nll'] for c in cells)
                loss_summary[label][candidate][region] = {'tokens':n,'sum_nll':total,'mean_nll':total/n}
    counts['loss_by_region'] = loss_summary
    dump(folder/'hf_comparison.json', counts)
    lines = ['# HF/PEFT step8 vs recorded vLLM', '', finding, '',
             '| Case | Seed | GPU | HF tool | HF capture | vLLM tool | Base ref-wait branch logp | Step8 ref-wait branch logp |',
             '|---|---:|---:|---|---|---|---:|---:|']
    for r in sorted(rows, key=lambda r:r['id']):
        lines.append(f'| {r["case"]} | {r["seed"]} | {r["gpu"]} | {r["parsed"]["tool"]} | {r["operation_capture"]} | '
                     f'{r["vllm"]["tool"]} | {r["scores"]["base"]["branch_logprob_reference_minus_wait"]:.4f} | '
                     f'{r["scores"]["step8"]["branch_logprob_reference_minus_wait"]:.4f} |')
    lines += ['', '| HF model | Teacher-forced candidate | Tool-name NLL | Trade-argument NLL | Memory NLL | Format NLL |',
              '|---|---|---:|---:|---:|---:|']
    for label, candidates in loss_summary.items():
        for candidate, regions in candidates.items():
            values = [f'{regions[k]["mean_nll"]:.4f}' if k in regions else 'N/A'
                      for k in ('tool_name','trade_arguments','private_state','format')]
            lines.append(f'| {label} | {candidate} | '+' | '.join(values)+' |')
    lines += ['', f'Counts: {json.dumps({k:v for k,v in counts.items() if k!="loss_by_region"})}', '',
        'Positive branch logp favors the reference over wait at their FIRST differing token, under their shared teacher-forced prefix. '
        'This is not a whole-tool probability or vLLM logit measurement. Full completion / tool_name / trade_arguments / private_state '
        'NLL and raw per-token NLL are saved in decisions_*.jsonl. Repeated seeds/hardware are not independent markets. '
        'The HF run uses the pinned training revision; the historical vLLM server did not record a pinned revision, '
        'so this is a behavior/configuration check, not proof of exact numerical parity.']
    (folder/'hf_comparison.md').write_text('\n'.join(lines)+'\n')
    dump(folder/'evaluation_status.json', {'status':'complete', 'completed_calls':len(rows)})
    print('\n'.join(lines),flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--rank', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.rank); return
    print('CPU preflight: checkpoint fingerprints, official template, tool parser and 8 paired inputs...',flush=True)
    jobs, provenance = prepare(args.source)
    from training.local_resources import output_parent, query_gpus, select_gpus, runtime_cache
    parent, note = output_parent(ROOT,args.output_dir)
    if note: print(note,flush=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S')+'_buyer_hf_',dir=parent))
    plan = {'provenance':provenance,'devices':[]}
    dump(folder/'plan.json',plan); dump(folder/'jobs.json',jobs)
    print(f'4 positive cases x 2 seeds = 8 HF generations; 4 teacher-forced scores/generation. Output: {folder}',flush=True)
    if args.prepare_only:
        print('PASS: CPU inputs prepared; no GPU query, model load or generation.'); return
    snapshot = query_gpus(); devices, skipped = select_gpus(snapshot,args.gpus)
    plan.update(devices=devices,gpu_snapshot=snapshot,skipped=skipped,gpu_selection=args.gpus)
    dump(folder/'plan.json',plan)
    requested = len(args.gpus.split(',')) if args.gpus!='auto' else len(devices)
    if len(devices)!=requested:
        raise SystemExit(f'Requested {requested} GPUs but only {len(devices)} fit; no GPU workers started. {skipped}')
    for job in jobs:
        # Preserve historical hardware pairing for explicit selections when available.
        rank = next((i for i,d in enumerate(devices) if d['index']==job['vllm']['gpu']), job['seed']%len(devices))
        job.update(worker=rank,gpu=devices[rank]['index'])
    dump(folder/'jobs.json',jobs)
    cache, cache_note = runtime_cache(ROOT)
    if cache_note: print(cache_note,flush=True)
    plan['cache_env']=cache; dump(folder/'plan.json',plan)
    print('GPU work split:',dict(Counter(j['gpu'] for j in jobs)),flush=True)
    dump(folder/'evaluation_status.json', {'status':'running','expected_calls':8})
    processes=[]; logs=[]; started=time.time()
    with (folder/'runner.log').open('w') as parentlog:
        def emit(message):
            print(message,flush=True);parentlog.write(message+'\n');parentlog.flush()
        try:
            for rank,d in enumerate(devices):
                log = (folder/f'worker_{rank}.log').open('w');logs.append(log)
                env = {**os.environ,**cache,'CUDA_VISIBLE_DEVICES':d['index'], 'PYTHONUNBUFFERED':'1',
                       'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                       'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
                       'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
                processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(folder),'--rank',str(rank)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while True:
                states=[]
                for rank in range(len(devices)):
                    path=folder/f'progress_{rank}.json'
                    states.append(json.loads(path.read_text()) if path.exists() else {'stage':'loading / warm-up; ETA estimating'})
                units=sum(s.get('units',0) for s in states); done=sum(s.get('generations',0) for s in states)
                times=[s['work_started'] for s in states if 'work_started' in s]
                seconds=time.time()-min(times) if times else 0
                speed=units/seconds if units and seconds>0 else 0
                eta=f'{(40-units)/speed:.0f}s' if speed else 'estimating'
                emit(f'[{"#"*int(24*units/40):24s}] work {units}/40 | generations {done}/8 | elapsed {time.time()-started:.0f}s | {speed:.2f} units/s | rough ETA {eta}')
                for rank,state in enumerate(states):
                    emit(f'GPU{devices[rank]["index"]}: {state["stage"]}; current generated tokens={state.get("generated_tokens",0)}')
                failures=[p.returncode for p in processes if p.poll() is not None and p.returncode]
                if failures:
                    for rank, process in enumerate(processes):
                        if process.returncode not in (None, 0):
                            path = folder/f'worker_{rank}.log'
                            emit(f'ROOT CAUSE GPU{devices[rank]["index"]}: {states[rank].get("error", "see traceback below")}')
                            emit('\n'.join(path.read_text(errors='replace').splitlines()[-22:]))
                    raise RuntimeError(f'HF worker failed: {failures}; inspect worker_*.log at {folder}')
                if all(p.poll() is not None for p in processes): break
                time.sleep(5)
        except BaseException as exc:
            dump(folder/'evaluation_status.json',{'status':'invalid','error':repr(exc)})
            for process in processes:
                if process.poll() is None: os.killpg(process.pid,signal.SIGINT)
            for process in processes:
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            for log in logs:log.close()
    try:
        report(folder)
    except BaseException as exc:
        dump(folder/'evaluation_status.json',{'status':'invalid','error':repr(exc)}); raise
    print(f'Completed. Read: {folder}/hf_comparison.md',flush=True)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
