"""Turn-boundary resampling shared by all UCF V3.1 estimators."""
import copy
import gzip
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

from patr_core import stable_seed
from scripts.hard_metrics import group_counts, leaf_group_counts
from task_envs import observed_state, same_outcome, turn_outcome
from ucf.ucf_weights import fork_records


def turn_weight(turn, prior):
    if prior == 'uniform':
        return 1.
    if prior == 'inverse_turn':
        return max(1. / turn, .25)
    raise ValueError(f'Unknown UCF turn prior {prior!r}')


def calibrate(weights_by_task, target):
    """Solve sum(min(1,c*w)) = target * (# training questions) by weight tier."""
    if len(weights_by_task) < 2 or not math.isfinite(target) or target <= 0:
        raise ValueError('Leave-one-question-out calibration needs two questions and a positive target')
    histograms = [Counter(ws) for ws in weights_by_task]
    totals = sum(histograms, Counter())
    desired = target * (len(histograms) - 1)
    fits = []
    for own in histograms:
        other = totals - own
        available = sum(other.values())
        if desired > available:
            fits.append(dict(c=None, saturated_below_target=True, calibration_nodes=available))
            continue
        # On each interval of c, the left-hand side is saturated + c*slope.
        slope = sum(w * count for w, count in other.items())
        saturated = 0
        for w, count in sorted(other.items(), reverse=True):
            c = (desired - saturated) / slope
            if c <= 1 / w or saturated + count == available:
                break
            slope -= w * count
            saturated += count
        fits.append(dict(c=c, saturated_below_target=False, calibration_nodes=available))
    return fits


def write_jsonl(path, rows):
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def select(candidates, tasks, originals, cfg, seed, step, out):
    prior = cfg['ucf_turn_prior']
    weights = [[] for _ in tasks]
    for node in candidates:
        node['w'] = turn_weight(node['turn'], prior)
        weights[node['task_id']].append(node['w'])
    fits = calibrate(weights, cfg['ucf_target_forks'])
    rows = []
    for n in candidates:
        ti, j = n['task_id'], n['j']
        fit = fits[ti]
        n['rho'] = 1. if fit['saturated_below_target'] else min(1., fit['c'] * n['w'])
        rows.append(dict(task_id=ti, uid=tasks[ti]['uid'], j=j, turn=n['turn'],
                         turn_start=n['turn_start'], w=n['w'], rho=n['rho'], M=cfg['ucf_resamples'],
                         R_o=originals[(ti, j)].reward, **fit))
    write_jsonl(out / f'candidates_{step:04d}.jsonl', rows)
    picked = []
    for n, row in zip(candidates, rows):
        u = random.Random(stable_seed(seed, row['uid'], row['j'], row['turn'], step, 'select')).random()
        row.update(u=u, selected=u < row['rho'])
        if row['selected']:
            picked.append(n)
        else:
            row.update(different=None, continued=False, R_cf=None, resample_tokens=0,
                       continuation_tokens=0, leaf_terminal=None)
    write_jsonl(out / f'select_{step:04d}.jsonl', rows)
    return picked, dict(target=cfg['ucf_target_forks'], prior=prior, fits=fits,
                        candidates=len(rows), expected_forks=sum(r['rho'] for r in rows) / len(tasks),
                        realized_forks=len(picked) / len(tasks))


def assert_prefix(child, original, turn_start):
    for attr in ('ids', 'mask', 'old_logprobs'):
        assert getattr(child, attr)[:turn_start] == getattr(original, attr)[:turn_start], attr


def restore_turn(core, task, original, node, tokenizer, cfg):
    """Replay only selected prefixes. Copy their actual sampled IDs and logprobs."""
    branch = core.start_branch(task, original.task_id, tokenizer, cfg, original.lineage)
    for trace in original.trace[:node['turn'] - 1]:
        branch.agent.update_from_model(trace['response'])
        action = trace['action']
        observation, reward, done, info = branch.env.step(action)
        assert not done and observation == trace['observation']
        branch.agent.update_from_env(observation, reward, done, info)
        branch.reward += float(reward)
        branch.actions.append(action)
        branch.turns += 1
        branch.trace.append(trace)
    assert observed_state(branch.env) == node['before']
    assert branch.turns == node['turn'] - 1 < cfg['max_turns']
    boundary = node['turn_start']
    for attr in ('ids', 'mask', 'old_logprobs'):
        setattr(branch, attr, getattr(original, attr)[:boundary])
    assert_prefix(branch, original, boundary)
    return branch


class RolloutSeed:
    """Supply the original rollout batch seed to each invocation of the flat sampler."""
    def __init__(self, value):
        self.value = value

    def integers(self, *args):
        return self.value


def make_collect(core, original):
    def collect(tasks, engine, tokenizer, cfg, method, rng, evaluation=False):
        cfg = dict(cfg, method='ucf')
        if evaluation:
            return original(tasks, engine, tokenizer, cfg, 'grpo', rng, evaluation=True)
        assert cfg['initial_branches'] == cfg['flat_group_size'] == 4
        assert cfg['response_protocol'] == 'first_complete_action_block'
        M = cfg['ucf_resamples']
        if type(M) is not int or M < 1:
            raise ValueError('ucf_resamples must be a positive integer')
        seed = int(rng.integers(0, 2**31 - 1))
        out = Path(cfg['ucf_log_dir'])
        out.mkdir(parents=True, exist_ok=True)
        status = Path(os.environ['PATR_PROGRESS_PATH']).parent / 'status.json'
        step = json.loads(status.read_text())['current_step']
        candidates = []

        def initial_turn(branch, response, tok, config):
            n = dict(task_id=branch.task_id, j=int(branch.lineage), turn=branch.turns + 1,
                     turn_start=len(branch.ids), before=observed_state(branch.env))
            core.apply_response(branch, response, tok, config)
            n['outcome'] = turn_outcome(branch)
            candidates.append(n)

        def sample(**kwargs):
            return original(tasks, engine, tokenizer, cfg, 'grpo', RolloutSeed(seed), **kwargs)

        first, stats, _ = sample(on_turn=initial_turn)
        originals = {(b.task_id, int(b.lineage)): b for b in first}
        candidates.sort(key=lambda n: (n['task_id'], n['j'], n['turn']))
        picked, budget = select(candidates, tasks, originals, cfg, seed, step, out)
        core.atomic_json(out / f'budget_{step:04d}.json', budget)

        children = []
        for node in picked:
            ti, j, t = node['task_id'], node['j'], node['turn']
            parent = restore_turn(core, tasks[ti], originals[ti, j], node, tokenizer, cfg)
            family = []
            for m in range(M):
                b = copy.deepcopy(parent) if m < M - 1 else parent
                b.lineage = f'{j}@{t}.rs{m}'
                b.order = (j, t, m)
                family.append(b)
            children.extend(family)
            node['children'] = family

        resample_tokens = Counter()
        continuation_tokens = Counter()

        def run_children(branches, costs, single_turn):
            if not branches:
                return
            def apply(branch, response, tok, config):
                costs[branch.task_id, branch.lineage] += len(response.token_ids)
                core.apply_response(branch, response, tok, config)
            _, cost, _ = sample(branches=branches, single_turn=single_turn, on_turn=apply)
            for key in ('input_tokens', 'generated_tokens', 'executed_actions', 'invalid_actions'):
                stats[key] += cost[key]

        run_children(children, resample_tokens, single_turn=True)
        leaves = []
        active = []
        for n in picked:
            ti, j, t = n['task_id'], n['j'], n['turn']
            different = []
            for child in n.pop('children'):
                assert_prefix(child, originals[ti, j], n['turn_start'])
                if not same_outcome(n['outcome'], turn_outcome(child)):
                    different.append(child)
            n['different'] = len(different)
            n['resample_tokens'] = sum(resample_tokens[ti, f'{j}@{t}.rs{m}'] for m in range(M))
            n['leaf'] = None
            if different:
                draw = random.Random(stable_seed(seed, tasks[ti]['uid'], j, t, step, 'continue'))
                leaf = draw.choice(different)
                n['leaf'] = leaf
                leaves.append(leaf)
                if not leaf.terminal:
                    active.append(leaf)
        children.clear()
        run_children(active, continuation_tokens, single_turn=False)

        records, events = [], []
        for n in picked:
            ti, j, t = n['task_id'], n['j'], n['turn']
            o, leaf = originals[ti, j], n['leaf']
            if leaf is not None:
                assert leaf.terminal
                assert_prefix(leaf, o, n['turn_start'])
            records.extend(fork_records(o, leaf, turn_start=n['turn_start'], rho=n['rho'],
                                        resamples=M, different=n['different']))
            continued = continuation_tokens[ti, leaf.lineage] if leaf is not None else 0
            events.append(dict(kind='turn_resample', task_id=ti, uid=tasks[ti]['uid'], j=j, turn=t,
                               w=n['w'], rho=n['rho'], turn_start=n['turn_start'], M=M,
                               different=n['different'], resample_tokens=n['resample_tokens'],
                               continued=continued > 0, continuation_tokens=continued,
                               selected_lineage=leaf.lineage if leaf is not None else None,
                               R_o=o.reward, R_cf=leaf.reward if leaf is not None else None,
                               leaf_terminal=leaf.terminal if leaf is not None else None))

        with gzip.open(out / f'raw_leaves_{step:04d}.jsonl.gz', 'xt') as stream:
            for kind, group in [('initial', first), ('counterfactual', leaves)]:
                for b in group:
                    row = {k: getattr(b, k) for k in ('task_id', 'lineage', 'ids', 'mask',
                           'old_logprobs', 'actions', 'trace', 'reward', 'terminal')}
                    stream.write(json.dumps(dict(kind=kind, task=tasks[b.task_id], **row)) + '\n')

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        changed = sum(e['different'] for e in events)
        flips = sum(e['R_cf'] is not None and e['R_cf'] != e['R_o'] for e in events)
        truncated = sum(b.terminal in ('context_limit', 'turn_limit', 'token_limit') for b in leaves)
        actual = first + leaves
        stats.update(group_counts(first, len(tasks)))
        stats.update(leaf_group_counts(actual, len(tasks), initial_not_applicable=False))
        stats.update(ucf_version='v3.1', episodes=len(actual), branch_events=len(events),
                     terminal_counts=dict(Counter(b.terminal for b in actual)),
                     mean_forks=len(events) / len(tasks), fork_budget=budget,
                     fork_counts=dict(selected_nodes=len(events), resamples=len(events) * M,
                                      changed_resamples=changed, changed_nodes=len(leaves),
                                      reward_changed_leaves=flips, leaves=len(leaves), truncated_leaves=truncated),
                     changed_resample_fraction=ratio(changed, len(events) * M),
                     changed_node_fraction=ratio(len(leaves), len(events)),
                     counterfactual_flip_rate=ratio(flips, len(leaves)),
                     leaf_truncation_fraction=ratio(truncated, len(leaves)),
                     invalid_action_fraction=ratio(stats['invalid_actions'], stats['executed_actions']))
        return first + records, stats, events
    return collect
