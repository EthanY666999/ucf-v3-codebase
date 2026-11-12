"""CERT adapter. Sample IDs and prefix logprobs are never retokenized/recomputed."""
import copy,json,math,os,time
from collections import Counter
from cert_entropy import token_map,distribution,entropies
from cert_select import stable_seed


def replay(core,node,task,cfg):
    parent=node['branch'];env=core.make_env(task,cfg)
    for a in parent.actions:
        _,_,done,_=env.step(a)
        assert not done,'Recorded pre-action prefix is terminal'
    assert int(env.s)==node['state'], 'Replay state mismatch'
    if hasattr(env, 'desc'):
        assert (env.desc==parent.env.desc).all(), 'Replay map mismatch'
    if hasattr(env,'anchor'):assert env.anchor()==parent.env.anchor(), 'Replay box/player mismatch'
    child=copy.deepcopy(parent);child.env=env
    child.lineage=f"{node['j']}@{node['turn']}.0"
    child.order=(4+node['j']*cfg['max_turns']+node['turn'],)
    for key in ('ids','mask','old_logprobs'):
        assert getattr(child,key)==getattr(parent,key)
    assert len(child.ids)==node['boundary'] and child.turns==node['turn']-1
    return child


def stream(core,branches,tasks,engine,tokenizer,cfg,base_seed,topk=20,capture=False,mapping=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    backend=engine.llm_engine;serial=getattr(engine,'_patr_request_serial',0)
    pending={};done=[];nodes=[];costs={};started=time.monotonic();last_progress=0
    def enqueue(b):
        nonlocal serial
        rem=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
        if rem<=0:
            b.terminal='context_limit' if len(b.ids)>=cfg['max_context_tokens'] else 'token_limit';done.append(b);return
        n=None
        if capture:
            snapshot=copy.deepcopy(b)
            d=core.distance(b.env)
            n=dict(branch=snapshot,j=int(b.lineage),turn=b.turns+1,state=int(b.env.s),boundary=len(b.ids),
                   S=0. if d is None else max(0.,min(1.,1-d/max(1,b.initial_distance if getattr(b.env,"environment",None)=="sokoban" else b.env.nrow+b.env.ncol-2))))
        params=SamplingParams(temperature=cfg['temperature'],top_p=1.,top_k=-1,max_tokens=rem,logprobs=topk,
             seed=stable_seed(base_seed,tasks[b.task_id]['uid'],b.lineage,b.turns),
             stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY)
        rid=f'cert-stream-{serial}';serial+=1
        key=(b.task_id,b.lineage);costs.setdefault(key,dict(generated_tokens=0,input_tokens=0,executed_actions=0,invalid_actions=0,request_seconds=0.))
        costs[key]['input_tokens']+=len(b.ids)
        backend.add_request(rid,{'prompt_token_ids':list(b.ids)},params)
        pending[rid]=(b,n,time.monotonic())
    for b in branches:enqueue(b)
    while pending:
        now=time.monotonic()
        deadline=cfg.get('_deadline_unix',float('inf'))
        if time.time()>=deadline:
            for rid in pending:backend.abort_request(rid)
            raise TimeoutError('Compute deadline during CERT sampling; no partial cell accepted')
        if now-last_progress>=5 and os.environ.get('PATR_PROGRESS_PATH'):
            core.atomic_json(os.environ['PATR_PROGRESS_PATH'],dict(phase='cert_initial' if capture else 'cert_continuations',active=len(pending),completed=len(done),updated_unix=time.time()))
            last_progress=now
        for result in backend.step():
            if not result.finished:continue
            b,n,ts=pending.pop(result.request_id);out=result.outputs[0];c=costs[(b.task_id,b.lineage)]
            c['request_seconds']+=time.monotonic()-ts;c['generated_tokens']+=len(out.token_ids)
            before=b.turns
            if n is not None:
                p,diagnostics=distribution(out,tokenizer,mapping)
                ha,hc,classes=entropies(n['branch'].env,p)
                n.update(pi=p,H_act=ha,H_cons=hc,classes=classes,diagnostics=diagnostics)
            # Frozen V1 operation: sampled IDs untouched, environment text only encoded.
            core.apply_response(b,out,tokenizer,cfg)
            if b.turns>before:
                c['executed_actions']+=1;c['invalid_actions']+=int(b.actions[-1]==0)
            if n is not None:
                n['parent_action']=b.actions[-1] if b.turns>before else None
                n['diagnostics']['action_first_parse_failed']=n['diagnostics']['action_first'] and n['parent_action'] not in (1,2,3,4)
                nodes.append(n)
            if not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:b.terminal='pruned_repeat'
            if b.terminal:done.append(b)
            else:enqueue(b)
    engine._patr_request_serial=serial
    done.sort(key=lambda b:(b.task_id,b.order))
    total={k:sum(c[k] for c in costs.values()) for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions','request_seconds')}
    total['seconds']=time.monotonic()-started
    return done,nodes,total,costs


def initial(core,tasks,engine,tokenizer,cfg,base_seed,topk=1,with_probes=False):
    assert not with_probes, 'V3 probes only after selection'
    assert topk==1,'Production logprobs must remain 1'
    # Execute frozen collect / SamplingParams / apply_response, with routing beyond H.
    # The observer copies pre-response state only; it consumes no RNG.
    nodes=[];original=core.apply_response
    def observe(b,out,tok,c):
        snapshot=copy.deepcopy(b);d=None if cfg.get('ucf_v3',False) else core.distance(b.env)
        n=dict(branch=snapshot,j=int(b.lineage),turn=b.turns+1,state=int(b.env.s),boundary=len(b.ids),
               S=0. if d is None else max(0.,min(1.,1-d/max(1,b.initial_distance if getattr(b.env,"environment",None)=="sokoban" else b.env.nrow+b.env.ncol-2))))
        before=b.turns;original(b,out,tok,c)
        n['parent_action']=b.actions[-1] if b.turns>before else None
        nodes.append(n)
    class FixedSeed:
        def integers(self,*args,**kwargs):return base_seed
    c=dict(cfg,branch_interval=cfg['max_turns']+1)
    ts=time.monotonic();core.apply_response=observe
    # V3 Part A must be exactly the paired GRPO4 sampler, including termination.
    # Historical tree collectors retain their existing repeat-pruning behavior.
    try:first,stats,_=core.collect(tasks,engine,tokenizer,c,'grpo' if cfg.get('ucf_v3',False) else 'patr',FixedSeed())
    finally:core.apply_response=original
    cost={k:stats[k] for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions')}
    cost.update(seconds=time.monotonic()-ts,request_seconds=0.,request_seconds_unavailable_for_frozen_initial=True)
    return first,nodes,cost,{}
