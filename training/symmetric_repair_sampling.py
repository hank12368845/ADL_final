"""One coverage pass with equal channels, intact decision pairs and interleaved personas."""
from collections import Counter, defaultdict
import random

QUOTAS = {'buy_listed': 2, 'accept_offer': 2, 'wait': 4}


def paired_batches(metadata, steps, seed):
    if len(metadata) != 256 or not 1 <= steps <= 32:
        raise ValueError('Symmetric repair requires 256 examples and at most 32 steps')
    pairs = defaultdict(dict)
    for i, row in enumerate(metadata):
        if row['positive'] in pairs[row['pair_id']]:
            raise ValueError('Duplicate pair side')
        pairs[row['pair_id']][row['positive']] = i
    pools = defaultdict(lambda: defaultdict(list))
    for sides in pairs.values():
        if set(sides) != {True, False}:
            raise ValueError('Incomplete symmetric pair')
        a, b = (metadata[sides[s]] for s in (True, False))
        if any(a[k] != b[k] for k in ('persona_index', 'history_mode', 'group')):
            raise ValueError('Mismatched decision pair')
        channel = a['family'].split('_')[0]
        if b['family'].split('_')[0] != channel:
            raise ValueError('Pair crosses channels')
        if a['label_action'] != {'board': 'buy_listed', 'offer': 'accept_offer'}.get(channel) or b['label_action'] != 'wait':
            raise ValueError('Pair action does not match current source')
        pools[channel][a['persona_index'], a['history_mode']].append([sides[True], sides[False]])
    rng = random.Random(seed)
    orders = {}
    for channel in ('board', 'offer'):
        if sum(map(len, pools[channel].values())) != 64:
            raise ValueError('Each channel needs 64 pairs')
        for pool in pools[channel].values():
            rng.shuffle(pool)
        keys = sorted(pools[channel])
        orders[channel] = []
        while any(pools[channel].values()):
            for key in keys:
                if pools[channel][key]:
                    orders[channel].append(pools[channel][key].pop())
    batches = []
    for step in range(32):
        rows = [i for channel in ('board', 'offer') for pair in orders[channel][2*step:2*step+2] for i in pair]
        rng.shuffle(rows)
        if Counter(metadata[i]['label_action'] for i in rows) != Counter(QUOTAS):
            raise ValueError('Symmetric batch quota mismatch')
        batches.append(rows)
    if Counter(i for batch in batches for i in batch) != Counter(range(256)):
        raise ValueError('Expected exactly one full coverage pass')
    return batches[:steps]


class PairedActionSampler:
    def __init__(self, metadata, steps, seed, rank=0, world=1):
        if world not in (1, 2) or not 0 <= rank < world:
            raise ValueError('Invalid rank/world')
        self.metadata, self.steps, self.seed, self.rank, self.world = metadata, steps, seed, rank, world
        self.start_step = 0
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.batches = paired_batches(self.metadata, self.steps, self.seed+100000*epoch)

    def __iter__(self):
        return iter([i for b in self.batches[self.start_step:] for i in b[self.rank::self.world]])

    def __len__(self):
        return (self.steps-self.start_step)*8//self.world


def batch_manifest(metadata, steps, seed, world):
    keys = ('id', 'pair_id', 'group', 'label_action', 'history_mode', 'persona_index', 'family', 'cash_variant')
    return {'version': 'symmetric_repair_batches_v1', 'world_size': world, 'global_batch': 8, 'quotas': QUOTAS,
            'steps': [{'step': s+1, 'actions': dict(Counter(metadata[i]['label_action'] for i in batch)),
                       'ranks': [[{k: metadata[i][k] for k in keys} for i in batch[r::world]] for r in range(world)]}
                      for s, batch in enumerate(paired_batches(metadata, steps, seed))]}
