"""Shared rollout and GRPO objective for the released UCF and baseline methods."""
import hashlib
import json
import os
import sys
import time
from collections import Counter
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
    trace: list = field(default_factory=list)
    lineage: str = ""
    order: tuple = ()

    @property
    def ntokens(self):
        return sum(self.mask)

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
        agent = WebShopAgent(require_argument=cfg.get('method') == 'ucf')
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
                  lineage=lineage)


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


def stable_seed(*parts):
    """The published SHA256 protocol for per-task, per-lineage rollout seeds."""
    raw = json.dumps(list(parts), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**31 - 1)


def collect(tasks, engine, tokenizer, cfg, method, rng, evaluation=False, *,
            branches=None, single_turn=False, on_turn=None):
    """Stream flat trajectories, or resume supplied branches for one turn or to termination."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    if method != 'grpo':
        raise ValueError(f'Unsupported flat collection method {method!r}')
    initial=cfg['eval_samples'] if evaluation else cfg['flat_group_size']
    base_seed=int(rng.integers(0,2**31-1))
    backend=engine.llm_engine
    serial=getattr(engine,'_patr_request_serial',0)
    pending={};completed=[];events=[]
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
            max_tokens=remaining,logprobs=1,
            seed=stable_seed(base_seed, tasks[b.task_id]['uid'],b.lineage,b.turns),
            stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY)
        rid=f'patr-stream-{serial}';serial+=1
        backend.add_request(rid,{'prompt_token_ids':list(b.ids)},params)
        pending[rid]=b;input_tokens+=len(b.ids)

    if branches is None:
        for ti,task in enumerate(tasks):
            for j in range(initial):
                b=start_branch(task,ti,tokenizer,cfg,str(j));b.order=(j,)
                enqueue(b)
    else:
        for branch in branches:
            enqueue(branch)
    respond = on_turn or apply_response
    last_progress=0.
    while pending:
        now=time.time()
        progress_path=os.environ.get('PATR_PROGRESS_PATH')
        if progress_path and now-last_progress>=5:
            atomic_json(progress_path,dict(phase='evaluation' if evaluation else 'rollout',method=method,
                active=len(pending),waiting=0,completed=len(completed),
                generated_tokens=generated,updated_unix=now,scheduler='per_task_async'))
            last_progress=now
        results=backend.step()
        for result in results:
            if not result.finished:continue
            b=pending.pop(result.request_id)
            out=result.outputs[0];generated+=len(out.token_ids)
            previous_turn=b.turns
            respond(b,out,tokenizer,cfg)
            if b.turns>previous_turn:
                executed_actions+=1;invalid_actions+=int(b.actions[-1]==0)
            if b.terminal or single_turn:
                completed.append(b)
            else:
                enqueue(b)
    engine._patr_request_serial=serial
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
