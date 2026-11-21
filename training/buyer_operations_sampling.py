"""Deterministic 4 buy / 2 accept / 2 wait batches, including two full counterfactual pairs."""
from __future__ import annotations
from collections import Counter, defaultdict
import random

QUOTAS={'buy_listed':4,'accept_offer':2,'wait':2}


def paired_batches(metadata,steps,seed):
    pairs=defaultdict(dict); positive=defaultdict(list)
    for i,row in enumerate(metadata):
        if row['positive'] in pairs[row['pair_id']]: raise ValueError('Duplicate side of counterfactual pair')
        pairs[row['pair_id']][row['positive']]=i
        if row['positive']: positive[row['label_action']].append(i)
        elif row['label_action']!='wait': raise ValueError('Counterfactual negative is not wait')
    families=defaultdict(list)
    for sides in pairs.values():
        if set(sides)!={True,False}: raise ValueError('Incomplete counterfactual pair')
        a,b=metadata[sides[True]],metadata[sides[False]]
        if (a['family'],a['group'],a['history_mode'])!=(b['family'],b['group'],b['history_mode']):
            raise ValueError('Pair group/family/context mismatch')
        families[a['family']].append(sides)
    board=sorted(f for f in families if f.startswith('board'))
    offers=sorted(f for f in families if f.startswith('offer'))
    if len(board)!=4 or len(offers)!=4: raise ValueError('Need all eight decision families')
    rng=random.Random(seed); batches=[]
    for step in range(steps):
        rows=[]
        a,b,c=(step%3+1,(step+1)%3+1,(step+2)%3+1)
        for family,persona in [(board[step%len(board)],a),(offers[(step+1)%len(offers)],b)]:
            pool=[s for s in families[family] if metadata[s[True]]['persona_index']==persona]
            if not pool: raise ValueError('Missing family/persona counterfactual pair')
            sides=rng.choice(pool);rows.extend([sides[True],sides[False]])
        for action,persona in [('buy_listed',a),('buy_listed',b),('buy_listed',c),('accept_offer',c)]:
            pool=[i for i in positive[action] if i not in rows and metadata[i]['persona_index']==persona]
            if not pool: raise ValueError('Insufficient distinct positive records for persona-balanced batch')
            rows.append(rng.choice(pool))
        rng.shuffle(rows)
        if Counter(metadata[i]['label_action'] for i in rows)!=Counter(QUOTAS):
            raise ValueError('Global action quota violated')
        batches.append(rows)
    return batches


class PairedActionSampler:
    def __init__(self,metadata,steps,seed,rank=0,world=1):
        if world not in (1,2) or not 0<=rank<world: raise ValueError('Invalid sampler rank/world')
        self.metadata,self.steps,self.seed,self.rank,self.world=metadata,steps,seed,rank,world
        self.set_epoch(0)

    def set_epoch(self,epoch):
        self.batches=paired_batches(self.metadata,self.steps,self.seed+100000*epoch)

    def __iter__(self):
        return iter([i for batch in self.batches for i in batch[self.rank::self.world]])

    def __len__(self): return self.steps*8//self.world


def batch_manifest(metadata,steps,seed,world):
    batches=paired_batches(metadata,steps,seed)
    return {'version':'paired_action_batches_v2','world_size':world,'global_batch':8,'quotas':QUOTAS,
        'steps':[{'step':s+1,'actions':dict(Counter(metadata[i]['label_action'] for i in batch)),
                  'personas':dict(Counter(metadata[i]['persona_index'] for i in batch)),
                  'ranks':[[{k:metadata[i][k] for k in ('id','pair_id','group','label_action','history_mode','persona_index')}
                            for i in batch[r::world]] for r in range(world)]}
                 for s,batch in enumerate(batches)]}
