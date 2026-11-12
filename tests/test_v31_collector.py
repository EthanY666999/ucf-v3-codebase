import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from transformers import AutoTokenizer

import patr_core as core
from ucf import ucf_v31
from patr_core import stable_seed
from task_envs import same_outcome, turn_outcome

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-2B', local_files_only=True)


def task(index):
    return dict(uid=f'puzzle{index}', environment='sokoban', walls=[], player=[2, 1],
                box=[2, 2], goal=[2, 3])


def config():
    return dict(json.loads((ROOT/'configs/common.json').read_text()), environment='sokoban',
                max_turns=3, max_response_tokens=270, max_context_tokens=2400)


def output(tok, text, finish='stop'):
    ids = tok.encode(text, add_special_tokens=False)
    return NS(token_ids=ids, text=text, finish_reason=finish,
              logprobs=[{k: NS(logprob=-.125 - i/100)} for i, k in enumerate(ids)])


def test_cutoff_then_execution_and_no_action(tokenizer):
    cfg = config()
    for lineage in ('0', '0@1.rs0'):
        for text, finish, action in [('thought ```Right```, discarded ```Down```', 'length', 3),
                                     ('incomplete ```', 'length', None),
                                     ('missing action', 'stop', 0)]:
            b = core.start_branch(task(0), 0, tokenizer, cfg, lineage)
            core.apply_response(b, output(tokenizer, text, finish), tokenizer, cfg)
            assert b.trace[-1]['action'] == action
            assert turn_outcome(b) == (b.env.render() if action else None)
            if action:
                assert b.reward == 1 and b.trace[-1]['protocol_cutoff']['discarded_sampled_tokens'] > 0


def test_weighted_calibration_and_inclusion():
    weights = [[1., .5, .25], [1., 1., .5], [1., .25, .25, .25] * 2]
    target = 3.
    fits = ucf_v31.calibrate(weights, target)
    for i, fit in enumerate(fits):
        other = [ws for j, ws in enumerate(weights) if j != i]
        assert sum(min(1., fit['c'] * w) for ws in other for w in ws) / 2 == pytest.approx(target)
        changed = [ws if j != i else [1.] * 1000 for j, ws in enumerate(weights)]
        assert ucf_v31.calibrate(changed, target)[i] == fit
    assert all(f['saturated_below_target'] for f in ucf_v31.calibrate(weights, 50))
    assert [ucf_v31.turn_weight(t, 'inverse_turn') for t in (1, 2, 4, 7)] == [1., .5, .25, .25]
    assert ucf_v31.turn_weight(6, 'uniform') == 1.
    rho = fits[0]['c'] / 4
    rng = random.Random(123)
    assert sum((rng.random() < rho) / rho for _ in range(50000))/50000 == pytest.approx(1., abs=.025)


def test_selection_logs_probabilities_before_any_draw(tmp_path, monkeypatch):
    candidates = [dict(task_id=i, j=0, turn=1, turn_start=30) for i in range(2)]
    tasks = [task(0), task(1)]
    originals = {(i, 0): NS(reward=0) for i in range(2)}
    actual_random = random.Random
    draws = []
    def checked_draw(seed):
        saved = [json.loads(s) for s in (tmp_path/'candidates_0001.jsonl').read_text().splitlines()]
        assert len(saved) == 2 and all(r['rho'] == 1 and r['w'] == 1 for r in saved)
        assert not (tmp_path/'select_0001.jsonl').exists()
        draws.append(seed)
        return actual_random(seed)
    monkeypatch.setattr(ucf_v31.random, 'Random', checked_draw)
    picked, _ = ucf_v31.select(candidates, tasks, originals, config(), 42, 1, tmp_path)
    selected = [json.loads(s) for s in (tmp_path/'select_0001.jsonl').read_text().splitlines()]
    assert len(draws) == len(picked) == 2
    assert all(r['selected'] and r['saturated_below_target'] for r in selected)


def test_collector_replays_turns_with_remaining_budgets(tmp_path, monkeypatch, tokenizer):
    cfg = dict(config(), ucf_target_forks=100, ucf_log_dir=str(tmp_path/'logs'))
    (tmp_path/'status.json').write_text('{"current_step":1}')
    monkeypatch.setenv('PATR_PROGRESS_PATH', str(tmp_path/'progress.json'))
    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=lambda **k: NS(**k)))
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', NS(RequestOutputKind=NS(FINAL_ONLY=1)))
    tasks = [task(0), task(1)]
    seed = int(np.random.default_rng(42).integers(0, 2**31 - 1))
    # Reconstruct the sampling protocol from task, lineage and completed turn count.
    by_seed = {}
    for i, puzzle in enumerate(tasks):
        for j in range(4):
            for turn in range(3):
                by_seed[stable_seed(seed, puzzle['uid'], str(j), turn)] = (i, j, 0, 0, turn)
                for m in range(2):
                    for past in range(turn, 3):
                        lineage = f'{j}@{turn+1}.rs{m}'
                        by_seed[stable_seed(seed, puzzle['uid'], lineage, past)] = (i, j, turn+1, m, past)
    requests = {}
    class Engine:
        def __init__(self):
            self.pending = []
        def add_request(self, rid, prompt, params):
            i, j, t, m, past = key = by_seed[params.seed]
            requests[key] = (prompt['prompt_token_ids'], params)
            assert 'extra_args' not in vars(params) and 'logits_processors' not in vars(params)
            # Original path Down, Up, Down. At t=1, Right wins immediately.
            # At t=2, Right changes position, then the third turn ends by horizon.
            # At t=3, both draws match, so there is no leaf.
            action = ['Down', 'Up', 'Down'][past]
            if t:
                assert len((tmp_path/'logs/select_0001.jsonl').read_text().splitlines()) == 24
                if (t, m, past) in ((1, 0, 0), (2, 1, 1)):
                    action = 'Right'
            response = output(tokenizer, f'my reasoning ```{action}``` discarded')
            self.pending.append(NS(request_id=rid, finished=True, outputs=[response]))
        def step(self):
            results, self.pending = self.pending, []
            return results[::-1]

    branches, stats, events = ucf_v31.make_collect(core, core.collect)(
        tasks, NS(llm_engine=Engine()), tokenizer, cfg, 'ucf', np.random.default_rng(42))
    originals = {(b.task_id, int(b.lineage)): b for b in branches if not hasattr(b, 'ucf')}
    assert len(originals) == 8 and stats['initial_group_counts']['all_failure'] == 2
    assert stats['fork_counts'] == dict(selected_nodes=24, resamples=48, changed_resamples=16,
        changed_nodes=16, reward_changed_leaves=8, leaves=16, truncated_leaves=8)
    assert stats['leaf_truncation_fraction'] == .5
    for event in events:
        i, j, t = event['task_id'], event['j'], event['turn']
        o, start = originals[i, j], event['turn_start']
        for m in range(2):
            ids, params = requests[i, j, t, m, t-1]
            original_params = requests[i, j, 0, 0, t-1][1]
            assert ids == o.ids[:start]
            assert params.max_tokens == original_params.max_tokens == min(
                cfg['max_response_tokens'] - sum(o.mask[:start]), cfg['max_context_tokens'] - start)
            for field in ('temperature', 'top_p', 'top_k', 'logprobs', 'output_kind', 'stop_token_ids'):
                assert getattr(params, field) == getattr(original_params, field)
        assert event['resample_tokens'] > 0 and event['R_o'] == 0
        if t == 3:
            assert event['different'] == 0 and event['selected_lineage'] is None and event['R_cf'] is None
            continue
        leaf = next(b for b in branches if hasattr(b, 'ucf') and b.task_id == i
                    and b.lineage == event['selected_lineage'])
        for field in ('ids', 'mask', 'old_logprobs'):
            assert getattr(leaf, field)[:start] == getattr(o, field)[:start]
        if t == 1:
            assert leaf.turns == 1 and event['R_cf'] == 1 and event['continuation_tokens'] == 0
            assert not event['continued']
        else:
            assert leaf.turns == 3 and event['leaf_terminal'] == 'turn_limit'
            assert event['continued'] and event['continuation_tokens'] > 0


def test_board_and_webshop_option_comparisons():
    page = dict(observation='item page', selected_options={'size':'M', 'color':'red'})
    branch = NS(env=NS(environment='webshop', page=page), trace=[dict(action='click[m]')])
    a = turn_outcome(branch)
    page['selected_options'] = {'color':'red', 'size':'M'}
    assert same_outcome(a, turn_outcome(branch))
    page['selected_options']['size'] = 'L'
    assert not same_outcome(a, turn_outcome(branch))
