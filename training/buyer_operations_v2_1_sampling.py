"""4/2/2 actions, 4/4 histories, 3/3/2 personas; replay profitable offers."""
from collections import Counter,defaultdict
import random
from training.buyer_operations_sampling import PairedActionSampler as OriginalSampler,QUOTAS

SCHEDULE=('cash','value','cash','stock','cash','owned','cash','value')
VARIANTS=('equal','above_one','wide','zero')


def paired_batches(metadata,steps,seed):
    pairs=defaultdict(dict);positive=defaultdict(list)
    for i,r in enumerate(metadata):
        if r['positive'] in pairs[r['pair_id']]:raise ValueError('Duplicate pair side')
        pairs[r['pair_id']][r['positive']]=i
        if r['positive']:positive[r['label_action']].append(i)
        elif r['label_action']!='wait':raise ValueError('Negative target must wait')
    pools=defaultdict(list)
    for sides in pairs.values():
        if set(sides)!={True,False}:raise ValueError('Incomplete counterfactual pair')
        a,b=(metadata[sides[k]] for k in (True,False))
        keys=('family','persona_index','history_mode','cash_variant','group')
        if any(a[k]!=b[k] for k in keys):raise ValueError('Pair metadata mismatch')
        pools[a['family'],a['persona_index'],a['history_mode'],a['cash_variant']].append(sides)
    rng=random.Random(seed);batches=[];cash_index=0
    for step in range(steps):
        a,b,c=step%3+1,(step+1)%3+1,(step+2)%3+1
        suffix=SCHEDULE[step%8];cash=suffix=='cash'
        variant=VARIANTS[cash_index%4] if cash else 'standard'
        flip=(cash_index//4+cash_index)%2 if cash else (step//2)%2
        offer_mode='recorded' if flip else 'fresh'
        board_mode='fresh' if flip else 'recorded'
        rows=[];pair_group=None
        for family,persona,mode in [(f'board_{suffix}',a,board_mode),
                                    (f'offer_{"qty" if suffix=="stock" else suffix}',b,offer_mode)]:
            choices=pools[family,persona,mode,variant]
            if cash and pair_group is not None:
                choices=[s for s in choices if metadata[s[True]]['group']==pair_group]
            if not choices:raise ValueError(f'Missing v2.1 pair: {family}/{persona}/{mode}/{variant}')
            chosen=rng.choice(choices);rows.extend([chosen[True],chosen[False]])
            pair_group=metadata[chosen[True]]['group']
        extra_board_mode='recorded' if board_mode=='fresh' else 'fresh'
        extras=[('buy_listed',a,extra_board_mode),('buy_listed',b,extra_board_mode),
                ('buy_listed',c,board_mode),('accept_offer',c,'fresh' if offer_mode=='recorded' else 'recorded')]
        for action,persona,mode in extras:
            choices=[i for i in positive[action] if i not in rows and metadata[i]['persona_index']==persona
                     and metadata[i]['history_mode']==mode]
            if not choices:raise ValueError('Missing context/persona positive replay')
            rows.append(rng.choice(choices))
        rng.shuffle(rows)
        counts=Counter(metadata[i]['label_action'] for i in rows)
        if counts!=Counter(QUOTAS):raise ValueError('Action quota mismatch')
        contexts=Counter((metadata[i]['label_action'],metadata[i]['history_mode']) for i in rows)
        if any(contexts[action,mode]!=n//2 for action,n in QUOTAS.items() for mode in ('fresh','recorded')):
            raise ValueError('Action/history quota mismatch')
        batches.append(rows)
        if cash:cash_index+=1
    return batches


class PairedActionSampler(OriginalSampler):
    def set_epoch(self,epoch):self.batches=paired_batches(self.metadata,self.steps,self.seed+100000*epoch)


def batch_manifest(metadata,steps,seed,world):
    batches=paired_batches(metadata,steps,seed)
    keys=('id','pair_id','group','label_action','history_mode','persona_index','family','cash_variant','economic_id')
    return {'version':'cash_offer_context_batches_v2_1','world_size':world,'global_batch':8,'quotas':QUOTAS,
        'steps':[{'step':s+1,'actions':dict(Counter(metadata[i]['label_action'] for i in batch)),
                  'contexts':dict(Counter(metadata[i]['history_mode'] for i in batch)),
                  'ranks':[[{k:metadata[i][k] for k in keys} for i in batch[r::world]] for r in range(world)]}
                 for s,batch in enumerate(batches)]}


def validation_selection(metadata,maximum,seed):
    if maximum<16 or maximum%16:raise ValueError('V2.1 validation budget must be a multiple of 16')
    pairs=defaultdict(dict);pools=defaultdict(list)
    for i,r in enumerate(metadata):pairs[r['pair_id']][r['positive']]=i
    for sides in pairs.values():
        if set(sides)!={True,False}:raise ValueError('Incomplete validation pair')
        row=metadata[sides[True]];pools[row['family'],row['history_mode']].append(sides)
    rng=random.Random(seed);result=[]
    families=sorted({r['family'] for r in metadata})
    for repetition in range(maximum//16):
        for j,family in enumerate(families):
            mode='fresh' if (j+repetition)%2==0 else 'recorded'
            choice=rng.choice(pools[family,mode]);result.extend([choice[True],choice[False]])
    return result[:maximum]
