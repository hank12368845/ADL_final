"""Recognize a narrowly defined vLLM startup VRAM race, not arbitrary CUDA failures."""
import json
from pathlib import Path


def startup_memory_races(folder,since):
    """Only fresh failed wrappers with zero market actions may be safely requeued.

    Inspect the exact run_dir from wrapper_result, not archived logs from an older
    failure. A genuine execution OOM or a source/config error must still stop.
    """
    root=Path(folder).resolve();found=[]
    wrappers=[root/'wrapper_result.json',*root.glob('*/wrapper_result.json')]
    for wrapper in wrappers:
        if not wrapper.is_file() or wrapper.stat().st_mtime<since:continue
        try:result=json.loads(wrapper.read_text())
        except (ValueError,OSError):continue
        if result.get('exit') in (None,0) or result.get('actions')!=0 or not result.get('run_dir'):continue
        run=Path(result['run_dir']).resolve()
        if not run.is_relative_to(root) or not run.exists():continue
        for log in run.glob('vllm/*.log'):
            if log.stat().st_mtime<since:continue
            for line in log.read_text(errors='replace').splitlines():
                if 'Free memory on device' in line and 'less than desired GPU memory utilization' in line:
                    found.append(dict(wrapper=str(wrapper),log=str(log),detail=line.strip()))
                    break
    return found
