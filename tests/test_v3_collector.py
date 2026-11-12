"""Exercise the V3 prefix and rollout adapter with deterministic fake inference."""
import copy,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'code'
sys.path[:0]=[str(ROOT),str(ROOT/'ucf'),str(Path(__file__).parent)]
import ucf_v3 as v3
from grid_fixture import fixture,Tok,FENCE


def test_candidate_prefix_does_not_look_at_successor_or_reward():
 pre,b,n=fixture();cfg=dict(max_response_tokens=100,max_context_tokens=100)
 results=[]
 for suffix,reward in [([30,FENCE,2],0),([20,FENCE,2],1),([2],0)]:
  o=copy.deepcopy(b);o.ids=o.ids[:4]+suffix;o.mask=[0]+[1]*(len(o.ids)-1);o.old_logprobs=[0.]+[-.2]*(len(o.ids)-1);o.reward=reward
  with patch.object(v3,'initial',return_value=([o],[copy.deepcopy(n)],{},{})):
   _,cs,_,_=v3.candidates_from_initial(None,[dict(uid='x')],None,Tok(),cfg,42)
  x=cs[0][0];results.append((x['branch'].ids,x['boundary'],x['turn_start']))
  assert 'S' not in x
 assert results[0]==results[1]==results[2]


def test_selected_counterfactual_preserves_ids_and_unrestricted_lp(monkeypatch):
 pre,o,n=fixture();n['branch']=copy.deepcopy(pre)
 for attr in ('ids','mask','old_logprobs'):setattr(n['branch'],attr,list(getattr(o,attr)[:4]))
 n.update(boundary=4,turn_start=1,rho=.2,baseline=1/3)
 cfg=dict(max_turns=10,max_response_tokens=100,max_context_tokens=100,temperature=1.,repeat_limit=11,ucf_beta=.25)
 key=(0,0,1);p=np.ones(1000)*.01/997;p[20]=.5;p[30]=.4;p[0]=.09
 p/=p.sum()
 calls=[]
 def requests(engine,jobs):
  outputs={}
  for k,ids,params in jobs:
   calls.append((ids,params))
   if params.max_tokens==1:
    flat=NS(start_indices=[0],end_indices=[1000],token_ids=list(range(1000)),logprobs=np.log(p).tolist())
    outputs[k]=NS(outputs=[NS(token_ids=[30],logprobs=flat)])
   else:
    ids2=[FENCE,2]
    outputs[k]=NS(outputs=[NS(token_ids=ids2,logprobs=[{x:NS(logprob=-.1)} for x in ids2],finish_reason='stop')])
  return outputs,dict(input_tokens=sum(len(j[1]) for j in jobs),generated_tokens=len(jobs),request_seconds=0.,seconds=0.)
 def apply(b,out,tok,c):
  b.ids.extend(out.token_ids);b.mask.extend([1]*len(out.token_ids));b.old_logprobs.extend(lp[t].logprob for t,lp in zip(out.token_ids,out.logprobs))
  b.actions.append(2);b.turns=1;b.reward=0.;b.terminal='hole';b.trace.append(dict(action=2,response=out.text))
 fake=NS(make_env=lambda t,c:copy.deepcopy(pre.env),apply_response=apply)
 engine=NS(llm_engine=NS(model_config=NS(get_vocab_size=lambda:1000)))
 monkeypatch.setitem(sys.modules,'vllm',NS(SamplingParams=lambda **kw:NS(**kw)))
 with patch.object(v3,'requests',side_effect=requests),patch.object(v3,'stream',return_value=([],[],dict(input_tokens=0,generated_tokens=0),{})):
  rows,children,cost=v3.counterfactual(fake,{0:[n]},[o],[dict(uid='x')],engine,Tok(),cfg,42,1)
 child=children[key];x=n['forced_token']
 assert x not in (30,31)
 assert child.ids[:4]==o.ids[:4] and child.old_logprobs[:4]==o.old_logprobs[:4]
 assert abs(child.old_logprobs[4]-np.log(p[x]))<1e-12
 assert rows[1].mask[:4]==[0]*4 and all(rows[1].mask[4:])
 assert len(calls)==2 and calls[0][1].logprobs==-1
 assert 'allowed_token_ids' not in vars(calls[0][1])


def test_probe_batches_preserve_request_identity_and_charge_all_work(monkeypatch):
 nodes={0:[dict(j=i,turn=1,branch=NS(ids=[1,i])) for i in range(19)]}
 calls=[]
 monkeypatch.setitem(sys.modules,'vllm',NS(SamplingParams=lambda **kw:NS(**kw)))
 def requests(engine,jobs):
  calls.append(jobs)
  # Deliberately return reverse completion order.
  return {k:NS(outputs=[NS(identity=k)]) for k,ids,p in reversed(jobs)},dict(input_tokens=2*len(jobs),generated_tokens=len(jobs),request_seconds=1.,seconds=1.)
 cost=dict(input_tokens=0,generated_tokens=0,request_seconds=0.,seconds=0.)
 with patch.object(v3,'requests',side_effect=requests):
  rows=list(v3.selected_probes(nodes,None,42,7,cost))
 assert [len(c) for c in calls]==[8,8,3]
 assert [p.identity for _,_,p in rows]==[(0,i,1) for i in range(19)]
 assert cost==dict(input_tokens=38,generated_tokens=19,request_seconds=3.,seconds=3.)
 for call in calls:
  for key,ids,p in call:
   assert p.seed==v3.stable_seed(42,*key,7,'v3-probe')
   assert p.logprobs==-1 and p.temperature==1. and p.top_p==1. and p.top_k==-1
