"""Local test resource planning. Query only; never start CUDA or stop other jobs."""
from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import shutil
import subprocess
import time

GIB = 1024 ** 3


def query_gpus() -> list[dict]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'Cannot query GPU memory: {result.stderr.strip()}')
    return parse_gpus(result.stdout)


def parse_gpus(output: str) -> list[dict]:
    rows = []
    for fields in csv.reader(output.splitlines(), skipinitialspace=True):
        if not fields:
            continue
        index, uuid, total, free = (field.strip() for field in fields)
        rows.append({'index': index, 'uuid': uuid,
                     'total_gib': float(total) / 1024, 'free_gib': float(free) / 1024})
    return rows


def select_gpus(rows: list[dict], selection: str = 'auto', memory: str = 'auto',
                reserve_gib: float = 2.0, min_budget_gib: float = 14.0) -> tuple[list, list]:
    """Use at most two independent replicas, each with its own startup memory budget."""
    if reserve_gib < 0 or not math.isfinite(reserve_gib):
        raise ValueError('reserve-gib must be finite and >= 0')
    if min_budget_gib <= 0 or not math.isfinite(min_budget_gib):
        raise ValueError('min-gpu-gib must be finite and > 0')
    cap = 0.95 if memory == 'auto' else float(memory)
    if not 0 < cap < 1:
        raise ValueError('gpu-mem must be auto or a fraction strictly between 0 and 1')
    available = {row['index']: row for row in rows}
    if selection == 'auto':
        candidates = sorted(rows, key=lambda row: -row['free_gib'])
    else:
        ids = [part.strip() for part in selection.split(',')]
        if not ids or len(ids) > 2 or len(set(ids)) != len(ids):
            raise ValueError('Use auto, one GPU index, or two distinct indices such as 0,1')
        missing = [index for index in ids if index not in available]
        if missing:
            raise ValueError(f'Unknown physical GPU indices: {missing}')
        candidates = [available[index] for index in ids]
    selected, skipped = [], []
    for row in candidates:
        # Fraction is of TOTAL VRAM, not of currently free VRAM. Round down for margin.
        fraction = math.floor(min(cap, (row['free_gib'] - reserve_gib) /
                                  row['total_gib']) * 1000) / 1000
        budget = fraction * row['total_gib']
        if budget < min_budget_gib:
            skipped.append({**row, 'reason': f'budget {budget:.2f} GiB < {min_budget_gib:g} GiB'})
            continue
        selected.append({**row, 'fraction': fraction, 'budget_gib': budget})
        if len(selected) == 2:
            break
    if not selected:
        raise RuntimeError(f'No GPU has enough free memory after reserve; details: {skipped}')
    return selected, skipped


def training_gpu_plan(selection='auto', required=None, query=None, pause=None):
    """Recheck a single-card startup once; retain why each card was skipped.

    An explicit two-index request requires both cards. Auto still supports a busy
    shared host, but announces any fallback and records the complete snapshots.
    """
    query = query or query_gpus
    pause = pause or time.sleep
    ids = None if selection == 'auto' else [x.strip() for x in selection.split(',')]
    minimum = required if required is not None else len(ids) if ids else 1
    if minimum not in (1, 2) or ids is not None and minimum > len(ids):
        raise ValueError('Required GPU count must be 1 or 2 and fit the explicit selection')
    snapshots = []
    for attempt in range(2):
        rows = query()
        devices, skipped = select_gpus(rows, selection, reserve_gib=2., min_budget_gib=18.)
        snapshots.append({'time': time.time(), 'gpus': rows, 'selected': devices, 'skipped': skipped})
        wants_two = len(ids) == 2 if ids is not None else len(rows) >= 2
        if len(devices) == 1 and wants_two and attempt == 0:
            print('Only one GPU fits the 18 GiB training budget now; rechecking in 5s before launch. '
                  f'Skipped: {skipped}', flush=True)
            pause(5)
            continue
        break
    if len(devices) < minimum:
        raise RuntimeError(f'Requested {minimum} usable GPUs, found {len(devices)}; '
                           f'no training started. Skipped: {skipped}')
    if len(devices) == 1 and wants_two:
        print('SINGLE-GPU FALLBACK: only one GPU fits after recheck. '
              'This DDP run cannot add GPUs later. Use --require-gpus 2 to reject fallback. '
              f'Skipped: {skipped}', flush=True)
    return devices, skipped, {'selection': selection, 'required': minimum,
                              'snapshots': snapshots, 'fallback_to_single': len(devices)==1 and wants_two}


def gpu_models(source: dict, devices: list[dict], port: int) -> dict:
    """Create endpoint templates; the wrapper may redistribute TA personas by cost."""
    if not devices or len(devices) > 2:
        raise ValueError('Expected one or two GPU plans')
    if not 1 <= port <= 65535 - (len(devices) - 1):
        raise ValueError('Invalid server port range')
    models = copy.deepcopy(source)
    template = copy.deepcopy(models['serving']['base'])
    models['serving'] = {}
    for spec in models['models'].values():
        spec.pop('replicas', None)
        spec['base_url'] = f'http://localhost:{port}/v1'
    for i, device in enumerate(devices):
        name = 'base' if i == 0 else 'ta'
        models['models'][name]['base_url'] = f'http://localhost:{port+i}/v1'
        models['serving'][name] = {
            **copy.deepcopy(template), 'gpus': [device['index']],
            'host': 'localhost', 'port': port+i,
            'gpu_memory_utilization': device['fraction']}
    return models


def output_parent(root: Path, requested: Path | None = None,
                  min_free_gib: float = 2.0) -> tuple[Path, str | None]:
    """Fail before model loading, or use /tmp when the project disk is full."""
    candidates = [requested] if requested is not None else [
        root / 'results/basic_checks', Path('/tmp/adl_final_basic_checks')]
    reasons = []
    for candidate in candidates:
        candidate = candidate.expanduser().absolute()
        ancestor = candidate
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free / GIB
        if free < min_free_gib:
            reasons.append(f'{candidate}: only {free:.2f} GiB free')
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            reasons.append(f'{candidate}: {exc}')
            continue
        note = ('; '.join(reasons) + f'; using {candidate} (temporary storage)') if reasons else None
        return candidate, note
    raise RuntimeError('Not enough writable result storage (need >= '
                       f'{min_free_gib:g} GiB): ' + '; '.join(reasons))


def runtime_cache(root: Path, requested: Path | None = None) -> tuple[dict[str, str], str | None]:
    """Keep compilation writes out of inherited HW3 paths; keep model downloads unchanged."""
    candidates = [requested] if requested is not None else [
        root / '.cache/runtime', Path('/tmp/adl_final_runtime_cache')]
    errors = []
    for candidate in candidates:
        try:
            folder, _ = output_parent(root, candidate, min_free_gib=5.0)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        paths = {
            'VLLM_CACHE_ROOT': folder / 'vllm',
            'VLLM_ASSETS_CACHE': folder / 'vllm/assets',
            'TORCHINDUCTOR_CACHE_DIR': folder / 'torchinductor',
            'TRITON_CACHE_DIR': folder / 'triton',
            'TORCH_EXTENSIONS_DIR': folder / 'torch_extensions',
            'FLASHINFER_WORKSPACE_BASE': folder / 'flashinfer',
            'CUDA_CACHE_PATH': folder / 'cuda',
            'TMPDIR': folder / 'tmp',
        }
        try:
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f'{folder}: {exc}')
            continue
        note = '; '.join(errors) + f'; using {folder}' if errors else None
        return {key: str(path) for key, path in paths.items()}, note
    raise RuntimeError('No writable compilation cache with >=5 GiB free: ' + '; '.join(errors))
