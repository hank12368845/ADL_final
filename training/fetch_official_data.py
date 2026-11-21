#!/usr/bin/env python3
"""Download the public course dataset at a pinned revision; CPU/network only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO='miulab/ADL26W-final-train'
REVISION='fa0f94cff3eeed5bc54da597ad5c2a45ef491e45'
FILES=('README.md','data/train-00000-of-00001.parquet')
EXPECTED={'README.md':'90a463f93aac2e467f0e39bd7a22b5ee891d97d5aec1786451013e1fee107965',
          'data/train-00000-of-00001.parquet':'85e4824b750cb42d77513c16759b0860ee90b1d0e55b1c1cb87e184a29dfa815'}


def fetch(output):
    from huggingface_hub import hf_hub_download
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    print(f'Official data {REPO}@{REVISION}; CPU download, no model inference.',flush=True)
    print(f'Output: {output}; downloading two files (parquet about 88 MB); progress/ETA from HF download.',flush=True)
    for i,name in enumerate(FILES,1):
        print(f'Download [{i}/{len(FILES)}]: {name}',flush=True)
        hf_hub_download(REPO,name,repo_type='dataset',revision=REVISION,local_dir=output,token=False)
    manifest={'repo':REPO,'revision':REVISION,'files':{name:{'bytes':(output/name).stat().st_size,
        'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()} for name in FILES}}
    if any(manifest['files'][name]['sha256']!=EXPECTED[name] for name in FILES):raise ValueError('Official download checksum differs from pinned source')
    (output/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS: official dataset downloaded and checksummed. No GPU work started.',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'training/data/official_raw')
    a=p.parse_args();fetch(a.output)
