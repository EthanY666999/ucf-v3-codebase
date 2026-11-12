"""Paper-based FrozenLake rollout policies. Original rLLM environment/agent are vendored.

No oracle selects an action: BFS is used only by PaTR's process scorer.
All actions, including those on failures and pruned branches, come from the policy.
"""
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/rllm_source"))
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.agents.frozenlake_agent import FrozenLakeAgent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def make_datasets(cfg):
    """Load the frozen task splits selected by the experiment configuration."""
    raw = Path(cfg["dataset_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["dataset_sha256"]:
        raise ValueError("Dataset changed")
    return json.loads(raw)


_FROZEN_MAP_CACHE={}
def frozen_maps(cfg):
    path=Path(cfg['frozen_maps_path']);key=(str(path),cfg['frozen_maps_sha256'])
    if key not in _FROZEN_MAP_CACHE:
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==key[1], 'Frozen maps file changed'
        _FROZEN_MAP_CACHE[key]=json.loads(raw)
    return _FROZEN_MAP_CACHE[key]


def make_env(task, cfg):
    if task.get("environment") == "webshop":
        from webshop import WebShopEnv
        return WebShopEnv(task, cfg)
    if task.get("environment")=="sokoban":
        from task_envs import SokobanEnv
        if cfg.get("frozen_maps_path"):
            stored=frozen_maps(cfg)[task["uid"]]
            assert all(stored[k]==task[k] for k in ["walls","player","box","goal"]), "Frozen Sokoban map mismatch"
        return SokobanEnv(task,cfg["max_turns"])
    if task.get("environment")=="flh":
        from task_envs import map_hash
        rows=frozen_maps(cfg)[task['uid']]
        assert map_hash(rows)==task['map_sha256'], 'Frozen FLH map content mismatch'
        e=FrozenLakeEnv(desc=rows,seed=task["seed"],is_slippery=False,max_steps=cfg["max_turns"])
        e.reset();return e
    raise ValueError(f"Unknown task environment: {task.get('environment')!r}")


def distance(env):
    if getattr(env,"environment",None)=="sokoban":
        return env.manhattan(env.box,env.goal)
    start = int(env.s)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pos, d = queue.popleft()
        r, c = divmod(pos, env.ncol)
        if env.desc[r, c] == b"G":
            return d
        for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rr, cc = r + dr, c + dc
            nxt = rr * env.ncol + cc
            if 0 <= rr < env.nrow and 0 <= cc < env.ncol and nxt not in seen and env.desc[rr, cc] != b"H":
                seen.add(nxt)
                queue.append((nxt, d+1))
    return None


@dataclass
class Branch:
    task_id: int
    env: object
    agent: object
    ids: list
    mask: list
    old_logprobs: list
    actions: list = field(default_factory=list)
    turns: int = 0
    reward: float = 0.0
    terminal: str = ""
    initial_distance: int = 1
    entropy: float = 0.0
    initial_entropy: object = None
    trace: list = field(default_factory=list)
    lineage: str = ""
    order: tuple = ()

    @property
    def ntokens(self):
        return sum(self.mask)

    def clone(self, suffix, reset_entropy=False):
        child = copy.deepcopy(self)
        child.lineage += suffix
        if reset_entropy:
            child.initial_entropy = None
        return child


_TURN_SUFFIX = {}
def chat_ids(tokenizer, messages):
    """Prompt token ids with thinking disabled (no-op kwarg for templates without it)."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(ids['input_ids']) if hasattr(ids, 'keys') else list(ids)

def turn_suffix(tokenizer, user_text):
    """Text appended after a sampled assistant turn (which already ends with eos) to open the next turn.

    Derived from the chat template once, so the generation prompt (e.g. Qwen3.5's empty
    <think></think> block when thinking is disabled) is reproduced exactly instead of hard-coded.
    """
    key = id(tokenizer)
    if key not in _TURN_SUFFIX:
        A, U = '@@ASSISTANT@@', '@@USER@@'
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': A}, {'role': 'user', 'content': U}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tail = text.split(A, 1)[1]
        eos = tokenizer.eos_token
        assert tail.startswith(eos), (tail[:40], eos)
        head, rest = tail[len(eos):].split(U, 1)
        _TURN_SUFFIX[key] = (head, rest)
    head, rest = _TURN_SUFFIX[key]
    return head + user_text + rest


def start_branch(task, task_id, tokenizer, cfg, lineage):
    env = make_env(task, cfg)
    if task.get('environment') == 'webshop':
        from webshop import WebShopAgent
        agent = WebShopAgent()
    elif task.get('environment')=='sokoban':
        from task_envs import sokoban_agent
        agent=sokoban_agent(cfg['max_turns'])
    else:agent = FrozenLakeAgent(max_steps=cfg['max_turns'])
    if cfg.get('response_protocol') == 'first_complete_action_block':
        from response_protocol import configure_agent
        configure_agent(agent, cfg)
    agent.update_from_env(env.render(), 0., False, {})
    ids = chat_ids(tokenizer, agent.chat_completions)
    return Branch(task_id, env, agent, ids, [0]*len(ids), [0.]*len(ids),
                  initial_distance=1 if cfg.get('ucf_v3') or task.get('environment')=='webshop' else max(1, distance(env) or 1), lineage=lineage)


def score_branch(b, cfg):
    """Explicit reconstruction of the unspecified FrozenLake scoring blend.

    Accumulated reward and clipped progress relative to starting BFS distance,
    combined using frozen, predeclared weights. Never optimized on the test split.
    """
    d = distance(b.env)
    progress = 0. if d is None else max(0., min(1., 1.-d/max(1,b.initial_distance if getattr(b.env,"environment",None)=="sokoban" else b.env.nrow+b.env.ncol-2)))
    return float(np.clip(cfg['scorer_reward_weight']*b.reward +
                         cfg['scorer_progress_weight']*progress, 0., 1.))


def route_patr(active, cfg, rng, random_selection=False):
    if not active:
        return [], [], []
    scores = np.array([score_branch(b,cfg) for b in active])
    # Uniform expansion control retains the same conservative pruning rule.
    # In particular, ties in zero process scores must not defeat random selection.
    tie_order = rng.permutation(len(active))
    order = tie_order if random_selection else tie_order[np.argsort(-scores[tie_order], kind='stable')]
    expand = set(order[:min(cfg['top_k_branches'],len(active))].tolist())
    median = float(np.median(scores))
    keep, done, events = [], [], [dict(kind="fork_candidates", turn=active[0].turns, candidates=[dict(lineage=b.lineage, score=float(scores[i]), selected=i in expand) for i,b in enumerate(active)])]
    for i,b in enumerate(active):
        # Match Algorithm 1: expansion selection precedes pruning on A \ E.
        if i in expand:
            for j in range(cfg['branch_factor']):
                child=b.clone(f".{j}");child.order=b.order+(j,)
                keep.append(child)
            events.append(dict(kind="expand", lineage=b.lineage, score=float(scores[i]), turn=b.turns))
        elif ((not random_selection and scores[i] < median-cfg['prune_margin']) or
              len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1):
            b.terminal = "pruned"
            done.append(b)
            events.append(dict(kind="prune", lineage=b.lineage, score=float(scores[i]), turn=b.turns))
        else:
            keep.append(b)
    return keep, done, events


def arpo_entropy(logprob_dicts, vocab_size):
    # Faithful to ARPO author's implementation: first 20 token positions,
    # returned top-10 (+sampled token) probabilities, normalized by log(|V|).
    # This is a truncated entropy statistic, not full-vocabulary Shannon entropy.
    values = [float(v.logprob) for d in logprob_dicts[:20] for v in d.values()]
    return -sum(math.exp(v)*v for v in values)/math.log(vocab_size)


def apply_response(b, out, tokenizer, cfg):
    protocol_meta = None
    if cfg.get('response_protocol') == 'first_complete_action_block':
        from response_protocol import truncate_response
        out, protocol_meta = truncate_response(b, out, tokenizer)
    tokens = list(out.token_ids)
    assert tokens and len(tokens)==len(out.logprobs)
    b.ids.extend(tokens)
    b.mask.extend([1]*len(tokens))
    b.old_logprobs.extend(float(lp[t].logprob) for t,lp in zip(tokens,out.logprobs))
    b.entropy = arpo_entropy(out.logprobs, len(tokenizer))
    if b.initial_entropy is None:
        b.initial_entropy = b.entropy
    if protocol_meta is not None:
        b.ids.append(tokenizer.eos_token_id)
        b.mask.append(0)
        b.old_logprobs.append(0.)
    text = out.text
    if out.finish_reason == "length":
        b.terminal = "token_limit"
        b.trace.append(dict(turn=b.turns, response=text, action=None, reward=0., terminal=b.terminal,
                            **({'anchor_before': b.env.anchor()} if hasattr(b.env,'anchor') else {})))
        return
    action = b.agent.update_from_model(text).action
    if getattr(b.env, 'environment', None) == 'webshop':
        action = b.env.resolve(action)
    else:
        action = int(action)
    b.actions.append(action)
    before_anchor=b.env.anchor() if hasattr(b.env,"anchor") else None
    observation, reward, done, info = b.env.step(action)
    b.turns += 1
    b.reward += float(reward)
    b.trace.append(dict(turn=b.turns, response=text, action=action, reward=float(reward),
                        state=int(b.env.s), observation=observation,
                        **({'webshop_score': b.env.raw_score} if hasattr(b.env,'raw_score') else {}), **(dict(anchor_before=before_anchor,anchor_after=b.env.anchor()) if before_anchor is not None else {})))
    if protocol_meta is not None:
        b.trace[-1]['protocol_cutoff'] = protocol_meta
    if done:
        b.terminal = "success" if b.env.success() else ("purchase_incomplete" if getattr(b.env,'environment',None)=='webshop' else
                   "turn_limit" if getattr(b.env,'environment',None)=='sokoban' else "hole")
    elif b.turns >= cfg['max_turns']:
        b.terminal = "turn_limit"
    elif b.ntokens >= cfg['max_response_tokens']:
        b.terminal = "token_limit"
    if not b.terminal:
        b.agent.update_from_env(observation,reward,done,info)
        user_text = b.agent.chat_completions[-1]['content']
        # Preserve exact sampled token IDs. Only newly observed environment text is encoded.
        suffix = turn_suffix(tokenizer, user_text)
        more = tokenizer.encode(suffix,add_special_tokens=False)
        b.ids.extend(more)
        b.mask.extend([0]*len(more))
        b.old_logprobs.extend([0.]*len(more))


def collect_batched(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
    from vllm import SamplingParams
    initial = cfg['eval_samples'] if evaluation else (cfg['flat_group_size'] if method in ('grpo','dapo') else cfg['initial_branches'])
    active, completed, counts, events = [], [], Counter(), []
    for ti,task in enumerate(tasks):
        for j in range(initial):
            active.append(start_branch(task,ti,tokenizer,cfg,str(j)))
        counts[ti] = initial
    generated = input_tokens = 0
    while active:
        viable = []
        for b in active:
            if len(b.ids)>=cfg['max_context_tokens'] or b.ntokens>=cfg['max_response_tokens']:
                b.terminal="context_limit" if len(b.ids)>=cfg['max_context_tokens'] else "token_limit"
                completed.append(b)
            else:
                viable.append(b)
        active = viable
        if not active:
            if method=='arpo' and not evaluation:
                for ti,task in enumerate(tasks):
                    if counts[ti]<cfg['flat_group_size']:
                        active.append(start_branch(task,ti,tokenizer,cfg,f"r{counts[ti]}"))
                        counts[ti]+=1
            if not active:
                break
        params=[]
        for b in active:
            params.append(SamplingParams(
                temperature=cfg['eval_temperature'] if evaluation else cfg['temperature'],
                top_p=cfg['eval_top_p'] if evaluation else 1.,
                top_k=cfg['eval_top_k'] if evaluation else -1,
                max_tokens=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids)),
                logprobs=10 if method=='arpo' and not evaluation else 1,
                seed=int(rng.integers(0,2**31-1)),
                stop_token_ids=[tokenizer.eos_token_id],
            ))
        input_tokens += sum(len(b.ids) for b in active)
        progress_path=os.environ.get('PATR_PROGRESS_PATH')
        if progress_path:
            atomic_json(progress_path,dict(phase='evaluation' if evaluation else 'rollout',method=method,
                 active=len(active),completed=len(completed),generated_tokens=generated,
                 min_turn=min(b.turns for b in active),max_turn=max(b.turns for b in active),updated_unix=time.time()))
        outputs=engine.generate([{'prompt_token_ids':b.ids} for b in active],params,use_tqdm=False)
        assert len(outputs)==len(active)
        survivors=[]
        for b,result in zip(active,outputs):
            out=result.outputs[0]
            generated += len(out.token_ids)
            apply_response(b,out,tokenizer,cfg)
            if not evaluation and method in ('patr','tree_random') and not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:
                b.terminal='pruned_repeat'
                events.append(dict(kind='prune_repeat',task_id=b.task_id,lineage=b.lineage,turn=b.turns))
            if b.terminal:
                completed.append(b)
            else:
                survivors.append(b)
        active=survivors
        if evaluation:
            continue
        if method in ('patr','tree_random'):
            next_active=[]
            for ti in range(len(tasks)):
                bs=[b for b in active if b.task_id==ti]
                if bs and bs[0].turns % cfg['branch_interval']==0:
                    keep,done,ev=route_patr(bs,cfg,rng,method=='tree_random')
                    next_active.extend(keep);completed.extend(done)
                    events.extend(dict(task_id=ti,**x) for x in ev)
                else:
                    next_active.extend(bs)
            active=next_active
        elif method=='arpo':
            extra=[]
            for b in active:
                if counts[b.task_id]>=cfg['flat_group_size']:
                    continue
                delta=b.entropy-b.initial_entropy
                decision=float(np.clip(rng.random()-cfg['arpo_entropy_weight']*delta,0.,1.))
                if decision<=cfg['arpo_branch_probability']:
                    extra.append(b.clone(f".a{counts[b.task_id]}",reset_entropy=True))
                    counts[b.task_id]+=1
                    events.append(dict(kind='arpo_expand',task_id=b.task_id,turn=b.turns,entropy_delta=delta))
            active.extend(extra)
            # Fill unused branch budget by fresh trajectories when all live paths terminate.
            living={b.task_id for b in active}
            for ti,task in enumerate(tasks):
                if ti not in living and counts[ti]<cfg['flat_group_size']:
                    active.append(start_branch(task,ti,tokenizer,cfg,f"r{counts[ti]}"))
                    counts[ti]+=1
    assert all(b.ntokens>0 for b in completed)
    stats=dict(generated_tokens=generated,input_tokens=input_tokens,episodes=len(completed),
               success_rate=float(np.mean([b.reward>0 for b in completed])),
               mean_response_tokens=float(np.mean([b.ntokens for b in completed])),
               mean_turns=float(np.mean([b.turns for b in completed])),
               terminal_counts=dict(Counter(b.terminal for b in completed)),
               branch_events=len(events))
    events.extend(dict(kind='leaf',task_id=b.task_id,lineage=b.lineage,parent=b.lineage.rsplit('.',1)[0] if '.' in b.lineage else None,reward=b.reward,terminal=b.terminal) for b in completed)
    return completed,stats,events


def evaluation_metrics(branches, tasks):
    group={i:[b for b in branches if b.task_id==i] for i in range(len(tasks))}
    assert all(len(bs)==4 for bs in group.values())
    success=np.array([[float(b.reward>0) for b in bs] for bs in group.values()])
    actions=[a for b in branches for a in b.actions]
    extra = ({'webshop_mean_score': float(np.mean([b.env.raw_score for b in branches]))}
             if getattr(branches[0].env, 'environment', None)=='webshop' else {})
    return dict(**extra,success_rate_pct=100*float(success.mean()),pass_at_4_pct=100*float(success.max(1).mean()),
                response_length=float(np.mean([b.ntokens for b in branches])),
                turns=float(np.mean([b.turns for b in branches])),n_tasks=len(tasks),n_trajectories=len(branches),
                invalid_action_fraction=sum(a==0 for a in actions)/len(actions) if actions else None,
                terminal_counts=dict(Counter(b.terminal for b in branches)))


def training_weights(branches, n_tasks, dapo=False):
    """Normalize by question, then by tree size and response length (Eq 2/9).

    DAPO instead uses token-normalized PG loss. Zero-variance groups yield zero PG.
    """
    group={i:[b for b in branches if b.task_id==i] for i in range(n_tasks)}
    adv={}
    for bs in group.values():
        assert bs
        rewards=np.array([b.reward for b in bs],dtype=np.float64)
        std=rewards.std(ddof=1) if len(bs)>1 else 0.
        for b,r in zip(bs,rewards):adv[id(b)]=float((r-rewards.mean())/(std+1e-6))
    ntotal=sum(b.ntokens for b in branches)
    return [(b,adv[id(b)],1./ntotal if dapo else 1./(n_tasks*len(group[b.task_id])*b.ntokens)) for b in branches]


def collect(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
    """Stream completed turns back to vLLM without a global all-task barrier.

    PaTR retains a per-task barrier at every K turns. ARPO retains its per-task
    per-turn allocation barrier. Seeds and branch ordering do not depend on
    cross-task completion order. The optimizer still waits for the whole batch.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    initial=cfg['eval_samples'] if evaluation else (cfg['flat_group_size'] if method in ('grpo','dapo') else cfg['initial_branches'])
    base_seed=int(rng.integers(0,2**31-1))
    def seed_for(*parts):
        raw=json.dumps([base_seed,*parts],sort_keys=True).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**31-1)
    routing_rng={i:np.random.default_rng(seed_for(t['uid'],'routing')) for i,t in enumerate(tasks)}
    backend=engine.llm_engine
    serial=getattr(engine,'_patr_request_serial',0)
    pending={};inflight=Counter();waiting={i:[] for i in range(len(tasks))}
    counts=Counter();completed=[];events=[]
    generated=input_tokens=0
    executed_actions=invalid_actions=0

    def enqueue(b):
        nonlocal serial,input_tokens
        remaining=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
        if remaining<=0:
            b.terminal='context_limit' if len(b.ids)>=cfg['max_context_tokens'] else 'token_limit'
            completed.append(b)
            return
        params=SamplingParams(temperature=cfg['eval_temperature'] if evaluation else cfg['temperature'],
            top_p=cfg['eval_top_p'] if evaluation else 1.,top_k=cfg['eval_top_k'] if evaluation else -1,
            max_tokens=remaining,logprobs=10 if method=='arpo' and not evaluation else 1,
            seed=seed_for(tasks[b.task_id]['uid'],b.lineage,b.turns),
            stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY)
        if cfg.get('_ucf_capture_session') is not None:
            assert not evaluation and method == 'grpo'
            from original_logits import request_metadata
            params.extra_args = request_metadata(cfg['_ucf_capture_session'],
                                                 b.task_id, b.lineage, b.turns)
        rid=f'patr-stream-{serial}';serial+=1
        backend.add_request(rid,{'prompt_token_ids':list(b.ids)},params)
        pending[rid]=b;inflight[b.task_id]+=1;input_tokens+=len(b.ids)

    def route_ready(ti):
        if inflight[ti] or evaluation or method in ('grpo','dapo'):
            return
        ready=sorted(waiting[ti],key=lambda b:b.order);waiting[ti]=[]
        if ready and method in ('patr','tree_random'):
            ready,done,ev=route_patr(ready,cfg,routing_rng[ti],method=='tree_random')
            completed.extend(done);events.extend(dict(task_id=ti,**x) for x in ev)
        elif ready and method=='arpo':
            extra=[]
            for b in ready:
                if counts[ti]>=cfg['flat_group_size']:break
                delta=b.entropy-b.initial_entropy
                decision=float(np.clip(routing_rng[ti].random()-cfg['arpo_entropy_weight']*delta,0.,1.))
                if decision<=cfg['arpo_branch_probability']:
                    child=b.clone(f'.a{counts[ti]}',reset_entropy=True);child.order=(counts[ti],)
                    extra.append(child);counts[ti]+=1
                    events.append(dict(kind='arpo_expand',task_id=ti,turn=b.turns,entropy_delta=delta))
            ready.extend(extra)
        if not ready and method=='arpo' and counts[ti]<cfg['flat_group_size']:
            b=start_branch(tasks[ti],ti,tokenizer,cfg,f'r{counts[ti]}');b.order=(counts[ti],)
            ready.append(b);counts[ti]+=1
        for b in ready:enqueue(b)

    for ti,task in enumerate(tasks):
        counts[ti]=initial
        for j in range(initial):
            b=start_branch(task,ti,tokenizer,cfg,str(j));b.order=(j,)
            enqueue(b)
    last_progress=0.
    while pending:
        now=time.time()
        progress_path=os.environ.get('PATR_PROGRESS_PATH')
        if progress_path and now-last_progress>=5:
            atomic_json(progress_path,dict(phase='evaluation' if evaluation else 'rollout',method=method,
                active=len(pending),waiting=sum(map(len,waiting.values())),completed=len(completed),
                generated_tokens=generated,updated_unix=now,scheduler='per_task_async'))
            last_progress=now
        results=backend.step()
        affected=set()
        for result in results:
            if not result.finished:continue
            b=pending.pop(result.request_id)
            ti=b.task_id;inflight[ti]-=1;affected.add(ti)
            out=result.outputs[0];generated+=len(out.token_ids)
            previous_turn=b.turns
            apply_response(b,out,tokenizer,cfg)
            if not evaluation and method in ('patr','tree_random') and not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:
                b.terminal='pruned_repeat'
                events.append(dict(kind='prune_repeat',task_id=b.task_id,lineage=b.lineage,turn=b.turns))
            if b.turns>previous_turn:
                executed_actions+=1;invalid_actions+=int(b.actions[-1]==0)
            if b.terminal:
                completed.append(b)
            elif not evaluation and (method=='arpo' or method in ('patr','tree_random') and b.turns%cfg['branch_interval']==0):
                waiting[ti].append(b)
            else:
                enqueue(b)
        for ti in sorted(affected):route_ready(ti)
    engine._patr_request_serial=serial
    assert not any(waiting.values()),'Unrouted task barrier'
    assert all(b.ntokens>0 for b in completed)
    completed.sort(key=lambda b:(b.task_id,b.order))
    stats=dict(generated_tokens=generated,input_tokens=input_tokens,episodes=len(completed),
               executed_actions=executed_actions,invalid_actions=invalid_actions,
               invalid_action_fraction=invalid_actions/executed_actions if executed_actions else None,
               success_rate=float(np.mean([b.reward>0 for b in completed])),
               mean_response_tokens=float(np.mean([b.ntokens for b in completed])),
               mean_turns=float(np.mean([b.turns for b in completed])),
               terminal_counts=dict(Counter(b.terminal for b in completed)),branch_events=len(events))
    events.extend(dict(kind='leaf',task_id=b.task_id,lineage=b.lineage,parent=b.lineage.rsplit('.',1)[0] if '.' in b.lineage else None,reward=b.reward,terminal=b.terminal) for b in completed)
    return completed,stats,events
