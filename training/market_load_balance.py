"""Plan local TA traffic by measured generation cost; never start a GPU service.

Student entries retain their original persona expansion. TA persona snapshots only
change their serving endpoint. Separate adapter directories prevent the course
loader's path deduplication from routing both copies back to the same server.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from collections import defaultdict

import yaml


def profile_costs(run_dir):
    """Mean output tokens per turn, including zero-output turns, from a finished run."""
    if run_dir is None:
        return {}, None
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / 'metadata.json').read_text())
    if meta.get('mock_llm'):
        raise ValueError('A mock run cannot provide model generation costs')
    totals, counts = defaultdict(int), defaultdict(int)
    for folder in sorted((run_dir / 'markets').iterdir()):
        if not (folder / 'final.json').exists():
            raise ValueError(f'Load profile contains an unfinished market: {folder}')
        with (folder / 'events.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('type') != 'turn' or not row.get('llm'):
                    continue
                aid = row['agent']
                counts[aid] += 1
                totals[aid] += row['llm'].get('usage', {}).get('output_tokens', 0)
    if not counts:
        raise ValueError(f'No model calls in load profile: {run_dir}')
    return {aid: max(1, totals[aid] / n) for aid, n in counts.items()}, str(run_dir)


def balance_ta_routes(config, loaded, models, session, root, devices, port, profile=None):
    """Greedy cost balancing of TA personas, with students pinned to server 0.

    This supports the local base/ta registry, at most two independent replicas.
    It leaves actions, prompts, owners, market rules and sampling settings intact.
    Costs predict aggregate generation work, not wall time or per-turn balance.
    """
    from adlmarket.config import expand_agents

    if len(devices) != 2:
        return {'method': 'single_server', 'profile': None}
    agents = [a for a in loaded.everyone() if a.controller == 'llm']
    if not any(a.owner == 'ta' for a in agents):
        return {'method': 'existing_non_ta_routing', 'profile': None}
    raw = {a.id: a for side in ('sellers', 'buyers')
           for a in expand_agents(config[side], root, config.get('personas_per_student', 1))}
    if any(raw[a.id].model not in ('base', 'ta') for a in agents):
        raise ValueError('Cost balancing currently requires local base/ta model entries')
    measured, profile_path = profile_costs(profile)
    costs = {a.id: measured.get(a.id, 700 if a.reasoning and a.role == 'buyer'
                               else 140 if a.role == 'buyer' else 80) for a in agents}
    assignment = {}
    loads = [0., 0.]
    # The same student adapter remains shared by all its personas, as in grading.
    for a in agents:
        if a.owner != 'ta':
            assignment[a.id] = 0
            loads[0] += costs[a.id]
    for a in sorted((a for a in agents if a.owner == 'ta'),
                    key=lambda a: (-costs[a.id], a.id)):
        i = min(range(2), key=lambda i: (loads[i], i))
        assignment[a.id] = i
        loads[i] += costs[a.id]
    # Avoid reserving a second GPU when students dominate an entirely tiny TA load.
    if len(set(assignment.values())) < 2:
        models['serving'].pop('ta', None)
    snapshots = Path(session) / 'routing_agents'
    snapshots.mkdir()
    adapters = {}
    copies = []
    for a in agents:
        if a.owner != 'ta':
            continue
        i = assignment[a.id]
        original = raw[a.id]
        data = original.model_dump(mode='json', exclude_none=True)
        for key in ('source', 'persona', 'personas', 'lora'):
            data.pop(key, None)
        data['persona_text'] = original.persona_body()
        data['model'] = 'base' if i == 0 else 'ta'
        # A registration name must be unique even if one TA LoRA lives on both GPUs.
        data['variant_of'] = f'local_{a.id}_server{i}'
        spec = loaded.models[a.model]
        if spec.lora:
            src = Path(spec.lora.path)
            key = (str(src), i)
            if key not in adapters:
                token = hashlib.sha256(str(src).encode()).hexdigest()[:12]
                dst = snapshots / f'adapter_{token}_server{i}'
                dst.mkdir()
                shutil.copyfile(src / 'adapter_config.json', dst / 'adapter_config.json')
                weights = 'adapter_model.safetensors'
                try:
                    os.link(src / weights, dst / weights)
                    method = 'hardlink'
                except OSError:
                    shutil.copyfile(src / weights, dst / weights)
                    method = 'copy'
                    # Cross-device copies must retain exactly the original weights.
                    def digest(path):
                        with path.open('rb') as stream:
                            return hashlib.file_digest(stream, 'sha256').hexdigest()
                    if digest(src / weights) != digest(dst / weights):
                        raise ValueError('Adapter copy checksum mismatch')
                adapters[key] = dst
                copies.append({'source': str(src), 'destination': str(dst),
                               'server': i, 'weights_storage': method})
            data['lora'] = str(adapters[key])
        path = snapshots / f'{hashlib.sha256(a.id.encode()).hexdigest()[:16]}.yaml'
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        # Avoid mutating the Pydantic agent; keep paths in a separate mapping.
        adapters[('agent', a.id)] = path
    for side in ('sellers', 'buyers'):
        entries = []
        for entry in config[side]:
            expanded = expand_agents([entry], root, config.get('personas_per_student', 1))
            if all(a.owner == 'ta' for a in expanded):
                entries.extend(str(adapters[('agent', a.id)]) if a.controller == 'llm'
                               else _scripted_snapshot(a, snapshots) for a in expanded)
            elif any(a.owner == 'ta' for a in expanded):
                raise ValueError('Use separate TA and student entries for local balancing')
            else:
                entries.append(copy.deepcopy(entry))
        config[side] = entries
    return {'method': 'ta_persona_cost_greedy_students_pinned', 'profile': profile_path,
            'estimated_output_tokens_per_tick': loads, 'assignments': assignment,
            'agent_cost_per_turn': costs, 'profile_matched_agents': sorted(set(costs) & set(measured)),
            'adapter_copies': copies,
            'note': 'Static aggregate estimate, not a speedup guarantee. Unprofiled agents use role/thinking priors.'}


def _scripted_snapshot(agent, folder):
    data = agent.model_dump(mode='json', exclude_none=True)
    for key in ('source', 'persona', 'personas'):
        data.pop(key, None)
    data['persona_text'] = agent.persona_body()
    path = folder / f'scripted_{hashlib.sha256(agent.id.encode()).hexdigest()[:16]}.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def validate_equivalent(before, after):
    """Reject a routing plan that changes agent identities, behavior or market rules."""
    if before.config.market != after.config.market:
        raise ValueError('Routing changed market rules')
    for side in ('sellers', 'buyers'):
        left, right = getattr(before, side), getattr(after, side)
        if [a.id for a in left] != [a.id for a in right]:
            raise ValueError(f'Routing changed {side} identity/order')
        for a, b in zip(left, right):
            excluded = {'model', 'source', 'lora', 'persona', 'personas', 'persona_text', 'variant_of'}
            if a.model_dump(exclude=excluded) != b.model_dump(exclude=excluded) or a.persona_body() != b.persona_body():
                raise ValueError(f'Routing changed behavior for {a.id}')
            if a.controller != 'llm':
                continue
            ma, mb = before.models[a.model], after.models[b.model]
            if ma.model_dump(exclude={'model', 'base_url', 'replicas', 'lora'}) != mb.model_dump(exclude={'model', 'base_url', 'replicas', 'lora'}):
                raise ValueError(f'Routing changed model settings for {a.id}')
            if bool(ma.lora) != bool(mb.lora) or (ma.lora and ma.lora.base_model != mb.lora.base_model):
                raise ValueError(f'Routing changed adapter/base for {a.id}')
