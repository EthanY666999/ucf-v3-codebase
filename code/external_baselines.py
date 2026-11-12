"""Framework-local StarPO-S components and GiGPO; see docs/baselines.md."""
import copy,math
from collections import defaultdict
import numpy as np
from patr_core import training_weights

def starpo_select(branches,n_tasks,keep_fraction):
    assert 0<keep_fraction<=1
    groups={i:[b for b in branches if b.task_id==i] for i in range(n_tasks)}
    assert all(len(bs)==4 for bs in groups.values())
    scores={i:float(np.std([b.reward for b in bs],ddof=1)) for i,bs in groups.items()}
    keep=sorted(scores,key=lambda i:(-scores[i],i))[:max(1,int(n_tasks*keep_fraction))]
    remap={i:j for j,i in enumerate(keep)};selected=[]
    for b in branches:
        if b.task_id in remap:
            b=copy.copy(b);b.task_id=remap[b.task_id];selected.append(b)
    return selected,keep,scores

def normalized(values):
    if len(values)<2:return [0.]*len(values)
    x=np.asarray(values,dtype=np.float64)
    return ((x-x.mean())/(x.std(ddof=1)+1e-6)).tolist()

def turn_records(b):
    starts=[i for i,k in enumerate(b.mask) if k and (i==0 or not b.mask[i-1])]
    assert len(starts)==len(b.trace),(len(starts),len(b.trace))
    if getattr(b.env, "environment", None) == "webshop":
        for t,(boundary,event) in enumerate(zip(starts,b.trace)):
            yield boundary, (b.task_id,event["anchor_before"]), t, len(starts)
        return
    board=tuple(tuple(x.decode() if isinstance(x,bytes) else str(x) for x in row) for row in b.env.desc)
    initial=[i for i,x in enumerate(sum((list(row) for row in board),[])) if x=='S']
    assert len(initial)==1
    pos=initial[0]
    for t,(boundary,event) in enumerate(zip(starts,b.trace)):
        anchor=tuple(event['anchor_before']) if 'anchor_before' in event else pos
        yield boundary,(b.task_id,board,anchor),t,len(starts)
        if event.get('action') is not None:pos=int(event['state'])

def gigpo_weights(branches,n_tasks,gamma=.95,omega=1.):
    base=training_weights(branches,n_tasks,False)
    if omega==0:return base
    groups=defaultdict(list);entries=[]
    for b,ae,w in base:
        for boundary,anchor,t,T in turn_records(b):
            idx=len(entries);entries.append((b,boundary,ae,w,anchor,gamma**(T-1-t)*b.reward))
            groups[anchor].append(idx)
    step_adv={}
    for anchor,idxs in groups.items():
        for i,v in zip(idxs,normalized([entries[i][-1] for i in idxs])):step_adv[i]=v
    result=[]
    for i,(b,boundary,ae,w,anchor,ret) in enumerate(entries):
        end = boundary
        while end < len(b.ids) and b.mask[end]:
            end += 1
        row = copy.copy(b)
        row.ids = b.ids[:end]
        row.old_logprobs = b.old_logprobs[:end]
        row.mask = [0] * boundary + b.mask[boundary:end]
        row.lineage = b.lineage + '.gigpo'
        row.gigpo=dict(episode_advantage=ae,step_advantage=step_adv[i],discounted_return=ret,anchor=anchor,group_size=len(groups[anchor]))
        result.append((row,ae+omega*step_adv[i],w))
    return result

def configure(train,cfg,method):
    """Install only the chosen method; capture original update once per process."""
    original=train.update
    if method=='gigpo':
        gamma=cfg.get('gigpo_gamma',.95);omega=cfg.get('gigpo_omega',1.)
        train.training_weights=lambda bs,n,dapo=False:gigpo_weights(bs,n,gamma,omega)
    elif method=='starpo_s':
        if 'starpo_keep_fraction' not in cfg:raise ValueError('StarPO-S retention protocol pending user decision')
        cfg.update(kl_coefficient=0.,clip_low=.2,clip_high=.28)
        def update(actor,ref,optimizer,branches,cfg,method,tokenizer,rng,n_tasks):
            selected,keep,scores=starpo_select(branches,n_tasks,cfg['starpo_keep_fraction'])
            result=original(actor,ref,optimizer,selected,cfg,method,tokenizer,rng,len(keep))
            result.update(starpo_kept_tasks=keep,starpo_reward_std=scores,starpo_keep_fraction=len(keep)/n_tasks)
            return result
        train.update=update
    else:raise ValueError(method)
