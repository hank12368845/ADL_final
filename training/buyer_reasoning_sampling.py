"""Full-coverage paired epochs: every example once, fixed prefix for exact resume."""
from collections import Counter, defaultdict
import random

QUOTAS = {'buy_listed': 2, 'accept_offer': 2, 'wait': 4}


def paired_batches(metadata, steps, seed):
    pairs = defaultdict(dict)
    for i, row in enumerate(metadata):
        if row['positive'] in pairs[row['pair_id']]: raise ValueError('Duplicate pair side')
        pairs[row['pair_id']][row['positive']] = i
    families = defaultdict(list)
    for sides in pairs.values():
        if set(sides) != {True, False}: raise ValueError('Incomplete pair')
        a, b = (metadata[sides[k]] for k in (True, False))
        if any(a[k] != b[k] for k in ('family','group','history_mode','persona_index')):
            raise ValueError('Pair metadata mismatch')
        families[a['family']].append([sides[True], sides[False]])
    board = sorted(f for f in families if f.startswith('board_'))
    offers = sorted(f for f in families if f.startswith('offer_'))
    if len(board) != 4 or len(offers) != 4 or len({len(v) for v in families.values()}) != 1:
        raise ValueError('Expected equally sized eight-family curriculum')
    if len(metadata) % 8: raise ValueError('Global batch must divide full data')
    result = []; epoch = 0; per_epoch = len(metadata)//8
    while len(result) < steps:
        rng = random.Random(seed+100000*epoch)
        pools = {k: rng.sample(v, len(v)) for k, v in families.items()}
        for step in range(per_epoch):
            chosen = [board[(2*step+j)%4] for j in range(2)]+[offers[(2*step+j)%4] for j in range(2)]
            batch = [i for family in chosen for i in pools[family].pop()]
            rng.shuffle(batch)
            if Counter(metadata[i]['label_action'] for i in batch) != Counter(QUOTAS):
                raise ValueError('Full-coverage action quota mismatch')
            result.append(batch)
            if len(result) == steps: return result
        if any(pools.values()): raise ValueError('Epoch did not cover every pair')
        epoch += 1
    return result


class PairedActionSampler:
    def __init__(self, metadata, steps, seed, rank=0, world=1):
        if world not in (1, 2) or not 0 <= rank < world: raise ValueError('Invalid rank/world')
        self.metadata, self.steps, self.seed, self.rank, self.world = metadata, steps, seed, rank, world
        self.start_step = 0; self.set_epoch(0)
    def set_epoch(self, epoch):
        self.batches = paired_batches(self.metadata, self.steps, self.seed+100000*epoch)
    def __iter__(self):
        return iter([i for b in self.batches[self.start_step:] for i in b[self.rank::self.world]])
    def __len__(self): return (self.steps-self.start_step)*8//self.world


def batch_manifest(metadata, steps, seed, world):
    keys = ('id','pair_id','group','label_action','history_mode','persona_index','family','cash_variant')
    return {'version': 'reasoning_full_pair_epochs_v1', 'world_size': world, 'global_batch': 8,
            'steps_per_epoch': len(metadata)//8, 'quotas': QUOTAS,
            'steps': [{'step': s+1, 'actions': dict(Counter(metadata[i]['label_action'] for i in batch)),
                       'ranks': [[{k: metadata[i][k] for k in keys} for i in batch[r::world]] for r in range(world)]}
                      for s, batch in enumerate(paired_batches(metadata, steps, seed))]}
