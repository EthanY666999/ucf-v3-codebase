"""UCF rows keep the frozen Part A function and minibatch task grouping."""
import copy

def generation_end(branch,boundary):
 assert boundary<len(branch.ids) and branch.mask[boundary]==1
 end=boundary
 while end<len(branch.ids) and branch.mask[end]:end+=1
 return end

def pseudo(branch,boundary,kind,adv,rho,original_tokens,beta,variant):
 end=generation_end(branch,boundary);b=copy.copy(branch)
 b.ids=list(branch.ids[:end]);b.old_logprobs=list(branch.old_logprobs[:end])
 b.mask=[0]*boundary+[1]*(end-boundary)
 b.ucf=dict(kind=kind,adv=float(adv),rho=float(rho),original_tokens=int(original_tokens),beta=float(beta),variant=variant)
 b.lineage=branch.lineage+'.'+kind
 assert len(b.ids)==len(b.mask)==len(b.old_logprobs) and b.ntokens==end-boundary
 return b

def training_weights_ucf(branches,n_tasks,dapo=False):
 assert not dapo
 from patr_core import training_weights
 initial=[b for b in branches if not hasattr(b,'ucf')]
 assert all(sum(b.task_id==ti for b in initial)==4 for ti in range(n_tasks))
 rows=training_weights(initial,n_tasks,False)
 for b in branches:
  if not hasattr(b,'ucf'):continue
  m=b.ucf
  if m['beta'] == 0:continue
  inv=1 if m['variant']=='noIS' else 1/m['rho']
  if m.get('ht_weight_cap') is not None:inv=min(inv,m['ht_weight_cap'])
  weight=m['beta']*inv/(n_tasks*4*m['original_tokens'])
  rows.append((b,m['adv'],weight))
 return rows

def fork_rows(original,cf,node,beta,variant):
 s=node['stratum'];base=node['baseline'];rho=node['rho'];boundary=node['boundary']
 po=1 if s['cf_fallback'] or variant=='noIS' else s['pi_o']
 rows=[pseudo(original,boundary,'orig_fork',2*po*(original.reward-base),rho,original.ntokens,beta,variant)]
 if not s['cf_fallback']:
  assert cf is not None
  pc=1 if variant=='noIS' else s['pi_cf']
  rows.append(pseudo(cf,boundary,'cf_fork',2*pc*(cf.reward-base),rho,original.ntokens,beta,variant))
 return rows
