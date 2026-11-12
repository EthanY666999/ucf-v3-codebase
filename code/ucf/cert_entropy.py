"""CPU-only first-token approximation and deterministic consequence entropy."""
import math
from collections import defaultdict
ACTIONS={'Left':1,'Down':2,'Right':3,'Up':4}

def entropy(probabilities):
    p=list(probabilities)
    assert all(math.isfinite(x) and x>=0 for x in p)
    assert abs(sum(p)-1)<1e-6
    return max(0.,min(1.,-sum(x*math.log(x) for x in p if x)/math.log(5)))

def token_map(tokenizer):
    mapping={};examples={}
    for word,action in ACTIONS.items():
        for prefix in ('',' '):
            text='```'+prefix+word+'```'
            ids=tokenizer.encode(text,add_special_tokens=False)
            # Locate the action-bearing token after the opening fence. Reject
            # tokenizer merges crossing the fence rather than guessing a position.
            pieces=[tokenizer.decode([i]) for i in ids]
            cumulative='';position=None
            for j,piece in enumerate(pieces):
                before=cumulative;cumulative+=piece
                if before.startswith('```') and not before[3:].strip() and piece.strip().startswith(word):
                    position=j;break
            assert position is not None,(text,ids,pieces)
            first=ids[position]
            assert first not in mapping or mapping[first]==action
            mapping[first]=action
            examples[text]=dict(ids=ids,pieces=pieces,position=position,first=first)
    return mapping,examples

def distribution(out,tokenizer,mapping):
    text='';position=None
    for i,t in enumerate(out.token_ids):
        piece=tokenizer.decode([int(t)])
        if text.startswith('```') and not text[3:].strip() and piece.strip():
            position=i;break
        text+=piece
        if not '```'.startswith(text) and not text.startswith('```'):break
    if position is None:return {0:1.,1:0.,2:0.,3:0.,4:0.},dict(format_missing=True,action_first=False)
    p={a:0. for a in range(5)}
    for token,lp in out.logprobs[position].items():
        if int(token) in mapping:p[mapping[int(token)]]+=math.exp(float(lp.logprob))
    total=sum(p.values());assert total<=1+1e-5,(total,p)
    if total>1:
        p={a:v/total for a,v in p.items()}
    p[0]=max(0.,1-sum(p.values()))
    return p,dict(format_missing=False,action_first=int(out.token_ids[position]) in mapping,position=position,covered_action_mass=1-p[0])

def consequence_classes(env):
    if hasattr(env,"consequence_classes"):return env.consequence_classes()
    r,c=divmod(int(env.s),env.ncol);result={0:'STAY'}
    for action,(dr,dc) in {1:(0,-1),2:(1,0),3:(0,1),4:(-1,0)}.items():
        rr,cc=r+dr,c+dc
        if not(0<=rr<env.nrow and 0<=cc<env.ncol):result[action]='STAY';continue
        tile=env.desc[rr,cc]
        if tile in (b'H','H'):result[action]='FAIL'
        elif tile in (b'G','G'):result[action]='GOAL'
        else:result[action]='CELL:'+str(rr*env.ncol+cc)
    return result

def entropies(env,p):
    merged=defaultdict(float)
    for a,c in consequence_classes(env).items():merged[c]+=p[a]
    return entropy(p.values()),entropy(merged.values()),dict(merged)
