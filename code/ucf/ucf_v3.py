"""UCF V3 diagnostic: no task SFT, no process score, select before probing."""
import copy
import gzip
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cert_collect import initial, replay, stream
from cert_entropy import token_map, consequence_classes
from cert_select import stable_seed
from first_fence import locate_token_prefix
from ucf_probe import requests
from ucf_weights import generation_end, fork_rows
from v3_sampling import question_probabilities, full_vector_from_output, complement_sample


def candidates_from_initial(core, tasks, engine, tokenizer, cfg, seed):
    first, nodes, cost, _ = initial(core, tasks, engine, tokenizer, cfg, seed,
                                    topk=1, with_probes=False)
    originals = {(b.task_id, int(b.lineage)): b for b in first}
    candidates = {i: [] for i in range(len(tasks))}
    coverage = dict(turns=len(nodes), turns_with_fence=0, candidates=0,
                    prefix_budget_exhausted=0, prefix_already_closed=0)
    for n in nodes:
        o = originals[(n['branch'].task_id, n['j'])]
        start = n['boundary']
        end = generation_end(o, start)
        bd = locate_token_prefix(o.ids, start, end, tokenizer)
        if bd is None:
            continue
        coverage['turns_with_fence'] += 1
        from response_protocol import first_block
        if first_block(tokenizer.decode(o.ids[start:bd], skip_special_tokens=True), cfg.get('environment')):
            coverage['prefix_already_closed'] += 1
            continue
        parent = copy.deepcopy(n['branch'])
        for attr in ('ids', 'mask', 'old_logprobs'):
            setattr(parent, attr, list(getattr(o, attr)[:bd]))
        rem = min(cfg['max_response_tokens']-parent.ntokens, cfg['max_context_tokens']-bd)
        if rem <= 0:
            coverage['prefix_budget_exhausted'] += 1
            continue
        assert bd < end, 'Prefix has budget but sampled successor is absent'
        n.pop('S', None)
        n.update(branch=parent, boundary=bd, turn_start=start, fork_point='first_fence')
        candidates[parent.task_id].append(n)
        coverage['candidates'] += 1
    return first, candidates, cost, coverage


def select(candidates, first, tasks, seed, step, target):
    counts = {i: len(ns) for i, ns in candidates.items()}
    probabilities = question_probabilities(counts, target)
    picked = {i: [] for i in candidates}
    records = []
    for ti, ns in candidates.items():
        for n in ns:
            others = [b for b in first if b.task_id==ti and int(b.lineage)!=n['j']]
            assert len(others)==3
            rho = probabilities[ti]
            u = random.Random(stable_seed(seed, tasks[ti]['uid'], n['j'], n['turn'], step, 'select')).random()
            fields = dict(task_id=ti, j=n['j'], turn=n['turn'], step=step,
                          rho=rho, u=u, selected=u<rho, w=1.,
                          p_hat_minus_j=sum(b.reward>0 for b in others)/3,
                          baseline=sum(b.reward for b in others)/3,
                          calibration='uniform_leave_one_question_out')
            n.update({k:v for k,v in fields.items() if k!='task_id'})
            records.append(dict(fields, uid=tasks[ti]['uid']))
            if u < rho:
                picked[ti].append(n)
    budget = dict(target=target, candidate_counts=counts, rho_by_task=probabilities,
                  expected_forks=sum(counts[i]*probabilities[i] for i in counts)/len(tasks),
                  realized_forks=sum(map(len,picked.values()))/len(tasks),
                  rho_floor=None, manual_score=False, exact_batch_budget_guaranteed=False)
    return picked, records, budget


def selected_probes(picked, engine, seed, step, pcost):
    """Bound full-vocabulary output memory while avoiding one-row GEMV probes.

    Eight is an execution batch size, not a selection or probability parameter.
    Selection and every request seed are fixed before any probe is evaluated.
    """
    from vllm import SamplingParams
    nodes = [(ti, n) for ti, ns in picked.items() for n in ns]
    for start in range(0, len(nodes), 8):
        chunk = nodes[start:start+8]
        jobs = []
        for ti, n in chunk:
            key = (ti, n['j'], n['turn'])
            params = SamplingParams(temperature=1., top_p=1., top_k=-1,
                max_tokens=1, logprobs=-1, flat_logprobs=True,
                seed=stable_seed(seed, *key, step, 'v3-probe'))
            jobs.append((key, list(n['branch'].ids), params))
        outputs, cost = requests(engine, jobs)
        for k in pcost:
            pcost[k] += cost[k]
        for ti, n in chunk:
            key = (ti, n['j'], n['turn'])
            yield ti, n, outputs.pop(key).outputs[0]
        assert not outputs


def selected_vectors(picked, engine, seed, step, pcost, vocab_size, originals, capture):
    if capture is None:
        for ti, n, output in selected_probes(picked, engine, seed, step, pcost):
            lp, log_z = full_vector_from_output(output, vocab_size)
            yield ti, n, lp, log_z
    else:
        for ti, ns in picked.items():
            for n in ns:
                lp, log_z = capture.take(ti, n, originals[(ti,n['j'])])
                yield ti, n, lp, log_z
        assert not capture.records


def counterfactual(core, picked, first, tasks, engine, tokenizer, cfg, seed, step, capture=None):
    from vllm import SamplingParams
    originals = {(b.task_id, int(b.lineage)):b for b in first}
    if cfg.get('environment') == 'webshop':
        from webshop import CLASSES, action_family, family_tokens
        mapping = family_tokens(tokenizer)
    else:
        mapping, _ = token_map(tokenizer)
        action_family = int
    vocab_size = engine.llm_engine.model_config.get_vocab_size()
    children, nodes, audit, jobs = {}, {}, [], []
    pcost = dict(input_tokens=0, generated_tokens=0, request_seconds=0., seconds=0.)
    # At most eight full vectors are live; none are retained in evidence files.
    for ti, n, lp, log_z in selected_vectors(picked, engine, seed, step, pcost,
                                             vocab_size, originals, capture):
        key = (ti, n['j'], n['turn'])
        nodes[key] = n
        parent = n['branch']
        original_token = originals[(ti,n['j'])].ids[n['boundary']]
        classes = CLASSES if cfg.get('environment') == 'webshop' else consequence_classes(parent.env)
        if cfg.get('ucf_partition') == 'action_tokens':
            classes = {action: str(action) for action in {0, *mapping.values()}}
        x, s = complement_sample(lp, original_token, mapping, classes,
                                 stable_seed(seed, tasks[ti]['uid'], *key, step, 'v3-cf'))
        n['stratum'] = s
        observed = originals[(ti,n['j'])].old_logprobs[n['boundary']]
        n['probe_original_logprob_difference'] = float(lp[original_token]-observed)
        n['full_vocabulary_log_normalizer'] = log_z
        # Partition labels only; residual token is not called actual STAY.
        n['original_parse_mismatch'] = mapping.get(original_token,0) != action_family(originals[(ti,n['j'])].trace[n['turn']-1].get('action') or 0)
        audit.append(dict(key=key, **s, log_normalizer=log_z,
                          original_logprob_difference=n['probe_original_logprob_difference']))
        n['probability_source'] = ('original_sampling_logits' if capture is not None
                                   else 'prefix_replay_probe')
        del lp
        if x is None:
            continue
        b = replay(core, n, tasks[ti], cfg)
        b.lineage = f"{n['j']}@{n['turn']}.cf"
        children[key] = b
        n['forced_token'] = x
        n['forced_raw_lp'] = s['forced_old_logprob_unrestricted']
        ids = b.ids+[x]
        rem = min(cfg['max_response_tokens']-b.ntokens-1, cfg['max_context_tokens']-len(ids))
        assert rem >= 0
        if rem and x != tokenizer.eos_token_id:
            jobs.append((key, ids, SamplingParams(temperature=1., top_p=1., top_k=-1,
                max_tokens=rem, logprobs=1, seed=stable_seed(seed,tasks[ti]['uid'],b.lineage,b.turns),
                stop_token_ids=[tokenizer.eos_token_id])))
    tails, tcost = requests(engine, jobs)
    active, done = [], {}
    for key,b in children.items():
        n = nodes[key]
        x, lp = n['forced_token'], n['forced_raw_lp']
        tail = tails[key].outputs[0] if key in tails else SimpleNamespace(
            token_ids=[],logprobs=[],finish_reason='stop' if x==tokenizer.eos_token_id else 'length')
        ids = [x]+list(tail.token_ids)
        out = SimpleNamespace(token_ids=ids,
            logprobs=[{x:SimpleNamespace(logprob=lp)}]+list(tail.logprobs),
            text=tokenizer.decode(b.ids[n['turn_start']:]+ids,skip_special_tokens=True),
            finish_reason=tail.finish_reason)
        core.apply_response(b,out,tokenizer,cfg)
        for attr in ('ids','mask','old_logprobs'):
            assert getattr(b,attr)[:n['boundary']] == getattr(n['branch'],attr)
        assert all(math.isfinite(v) for v in b.old_logprobs)
        n['cf_parse_mismatch'] = action_family(b.trace[-1].get('action') or 0) != mapping.get(x,0)
        if not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:
            b.terminal='pruned_repeat'; b.reward=0.
        if b.terminal:
            done[key]=b
        else:
            active.append(b)
    leaves, _, ccost, _ = stream(core,active,tasks,engine,tokenizer,cfg,seed,topk=1,capture=False)
    reverse={(b.task_id,b.lineage):key for key,b in children.items()}
    for b in leaves:
        done[reverse[(b.task_id,b.lineage)]]=b
    rows=[]
    for key,n in nodes.items():
        o=originals[(key[0],key[1])]; cf=done.get(key)
        rows.extend(fork_rows(o,cf,n,cfg['ucf_beta'],'uniform'))
        n['R_o']=o.reward; n['R_cf']=None if cf is None else cf.reward
    return rows,done,dict(probe=pcost,tail=tcost,continuation=ccost,
        forced_tokens=len(children),sample_audit=audit)


def make_collect(core, original):
    def collect(tasks,engine,tokenizer,cfg,method,rng,evaluation=False):
        if evaluation:
            return original(tasks,engine,tokenizer,cfg,method,rng,evaluation=True)
        assert cfg['temperature']==1. and cfg['initial_branches']==4
        assert cfg['response_protocol']=='first_complete_action_block'
        out=Path(cfg['ucf_log_dir']); out.mkdir(parents=True,exist_ok=True)
        progress=Path(os.environ['PATR_PROGRESS_PATH'])
        step=json.loads((progress.parent/'status.json').read_text())['current_step']
        state=out/'controller.json'
        if state.exists():
            assert step>json.loads(state.read_text())['last_step'],'Reconcile checkpoint before resume'
        seed=int(rng.integers(0,2**31-1))
        capture = None
        if cfg.get('ucf_probability_source') == 'original_sampling_logits':
            from original_logits import CaptureSession
            capture = CaptureSession(tokenizer, len(tasks)*4*cfg['max_turns'],
                                     engine.llm_engine.model_config.get_vocab_size(), cfg.get('environment'))
            with capture:
                first,candidates,icost,coverage=candidates_from_initial(
                    core,tasks,engine,tokenizer,
                    dict(cfg, _ucf_capture_session=capture.serial),seed)
        else:
            first,candidates,icost,coverage=candidates_from_initial(core,tasks,engine,tokenizer,cfg,seed)
        picked,selection,budget=select(candidates,first,tasks,seed,step,cfg.get('ucf_target_forks',2.))
        if capture is not None:
            capture.retain(picked)
        core.atomic_json(out/f'budget_{step:04d}.json',budget)
        with (out/f'select_{step:04d}.jsonl').open('x') as f:
            for row in selection: f.write(json.dumps(row)+'\n')
        pseudo,children,ccost=counterfactual(core,picked,first,tasks,engine,tokenizer,cfg,seed,step,capture)
        with gzip.open(out/f'raw_leaves_{step:04d}.jsonl.gz','xt') as f:
            for kind,bs in [('initial',first),('counterfactual',list(children.values()))]:
                for b in bs:
                    f.write(json.dumps(dict(kind=kind,task=tasks[b.task_id],task_id=b.task_id,
                        lineage=b.lineage,ids=b.ids,mask=b.mask,old_logprobs=b.old_logprobs,
                        actions=b.actions,trace=b.trace,reward=b.reward,terminal=b.terminal))+'\n')
        core.atomic_json(out/f'controller_{step:04d}.json',dict(last_step=step,version='v3',budget=budget))
        core.atomic_json(state,dict(last_step=step,version='v3',budget=budget))
        ns=[n for rows in picked.values() for n in rows]
        cfns=[n for n in ns if not n['stratum']['cf_fallback']]
        types={i:('all_failure' if (n:=sum(b.reward>0 for b in first if b.task_id==i))==0
                  else 'all_success' if n==4 else 'mixed') for i in range(len(tasks))}
        counts=Counter(types.values()); failures=[i for i,t in types.items() if t=='all_failure']
        rescued=sum(any(b.task_id==i and b.reward>0 for b in children.values()) for i in failures)
        leaves=first+list(children.values())
        stats=dict(ucf_version='V3-uniform-full-complement',episodes=len(leaves),
            success_rate=sum(b.reward>0 for b in first)/len(first),
            mean_response_tokens=sum(b.ntokens for b in first)/len(first),
            mean_turns=sum(b.turns for b in first)/len(first),
            terminal_counts=dict(Counter(b.terminal for b in leaves)),branch_events=len(ns),
            mean_forks=len(ns)/len(tasks),fork_budget=budget,first_fence_coverage=coverage,
            candidate_coverage=coverage['candidates']/max(1,coverage['turns']),
            response_protocol=cfg['response_protocol'],
            initial_group_counts={t:counts[t] for t in ['all_failure','mixed','all_success']},
            initial_group_fractions={t:counts[t]/len(tasks) for t in ['all_failure','mixed','all_success']},
            mixed_group_fraction=counts['mixed']/len(tasks),
            rescue_rate=rescued/len(failures) if failures else None,
            counterfactual_flip_rate=sum(n['R_o']!=n['R_cf'] for n in cfns)/len(cfns) if cfns else None,
            cf_fallback_fraction=(len(ns)-len(cfns))/max(1,len(ns)),
            full_vocabulary_log_normalizers=[n['full_vocabulary_log_normalizer'] for n in ns],
            probe_original_logprob_differences=[n['probe_original_logprob_difference'] for n in ns],
            original_token_partition_parser_disagreement=sum(n['original_parse_mismatch'] for n in ns)/max(1,len(ns)),
            counterfactual_token_partition_parser_disagreement=sum(n.get('cf_parse_mismatch',False) for n in cfns)/max(1,len(cfns)),
            selected_turn_histogram=dict(Counter(n['turn'] for n in ns)),
            rho_values=[r['rho'] for r in selection],probe_cost=ccost['probe'])
        if capture is not None:
            stats['original_logits_capture'] = capture.summary()
        for k in ('input_tokens','generated_tokens'):
            stats[k]=icost[k]+sum(ccost[a][k] for a in ('probe','tail','continuation'))
        stats['generated_tokens']+=ccost['forced_tokens']
        stats['executed_actions']=icost['executed_actions']
        stats['executed_actions']+=sum(b.turns-next(n['branch'].turns for n in picked[key[0]] if (n['j'],n['turn'])==key[1:]) for key,b in children.items())
        stats['invalid_actions']=icost['invalid_actions']+sum(sum(a==0 for a in b.actions[key[2]-1:]) for key,b in children.items())
        stats['invalid_action_fraction']=stats['invalid_actions']/max(1,stats['executed_actions'])
        from scripts.hard_metrics import leaf_group_counts
        stats.update(leaf_group_counts(leaves,len(tasks),initial_not_applicable=False))
        stats['leaf_group_source']='initial four plus actual counterfactual leaves'
        events=[dict(kind='ucf_v3_fork',task_id=ti,**{k:v for k,v in n.items() if k!='branch'})
                for ti,rows in picked.items() for n in rows]
        return first+pseudo,stats,events
    return collect
