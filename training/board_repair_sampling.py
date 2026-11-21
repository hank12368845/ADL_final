"""One 32-step coverage pass: four board positives, one offer, three valid waits."""
from collections import Counter,defaultdict
import random

QUOTAS={'buy_listed':4,'accept_offer':1,'wait':3}


def paired_batches(metadata,steps,seed):
    if not 1<=steps<=32 or len(metadata)!=256:raise ValueError('Repair supports a 256-example, at most 32-step pilot')
    rng=random.Random(seed);positive=defaultdict(list);negative=defaultdict(list);offers=defaultdict(dict)
    for i,r in enumerate(metadata):
        if r['family'].startswith('board'):
            (positive if r['positive'] else negative)[r['history_mode']].append(i)
        else:offers[r['pair_id']][r['positive']]=i
    offer_modes=defaultdict(list)
    for pair in offers.values():
        if set(pair)!={True,False}:raise ValueError('Incomplete offer replay pair')
        offer_modes[metadata[pair[True]]['history_mode']].append(pair)
    for pools in (positive,negative,offer_modes):
        for pool in pools.values():rng.shuffle(pool)
    batches=[]
    for step in range(32):
        rows=[]
        for mode in ('fresh','recorded'):
            rows.extend([positive[mode].pop(),positive[mode].pop(),negative[mode].pop()])
        pair=offer_modes['fresh' if step%2==0 else 'recorded'].pop()
        rows.extend([pair[True],pair[False]]);rng.shuffle(rows)
        if Counter(metadata[i]['label_action'] for i in rows)!=Counter(QUOTAS):raise ValueError('Repair batch quota mismatch')
        batches.append(rows)
    if len({i for b in batches for i in b})!=256:raise ValueError('Repair coverage is not unique')
    return batches[:steps]


class PairedActionSampler:
    def __init__(self,metadata,steps,seed,rank=0,world=1):
        if world not in (1,2) or not 0<=rank<world:raise ValueError('Invalid rank/world')
        self.metadata,self.steps,self.seed,self.rank,self.world=metadata,steps,seed,rank,world
        self.start_step=0;self.set_epoch(0)
    def set_epoch(self,epoch):self.batches=paired_batches(self.metadata,self.steps,self.seed+100000*epoch)
    def __iter__(self):return iter([i for b in self.batches[self.start_step:] for i in b[self.rank::self.world]])
    def __len__(self):return (self.steps-self.start_step)*8//self.world


def batch_manifest(metadata,steps,seed,world):
    keys=('id','pair_id','group','label_action','history_mode','persona_index','family','cash_variant')
    return {'version':'board_first_repair_batches_v1','world_size':world,'global_batch':8,'quotas':QUOTAS,
            'steps':[{'step':s+1,'actions':dict(Counter(metadata[i]['label_action'] for i in batch)),
                      'ranks':[[{k:metadata[i][k] for k in keys} for i in batch[r::world]] for r in range(world)]}
                     for s,batch in enumerate(paired_batches(metadata,steps,seed))]}
