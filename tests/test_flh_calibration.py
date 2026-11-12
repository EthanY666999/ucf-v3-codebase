import json
import sys
from unittest.mock import Mock

import pytest

import train
from patr_core import Branch
from scripts import hardware, q35_thinking_policy


@pytest.mark.parametrize('success,pass4,outcome', [
    (5, 20, 'in_target'), (20, 50, 'in_target'),
    (4.75, 19, 'below_target'), (10, 19, 'below_target'),
    (20.25, 45, 'above_target'), (15, 51, 'above_target'),
])
def test_step_zero_intervals(success, pass4, outcome):
    metrics = dict(success_rate_pct=success, pass_at_4_pct=pass4)
    report = train.flh_base_assessment(metrics, 'hard')
    assert report['outcome'] == outcome
    assert report['difficulty'] == 'hard'
    assert 'next_difficulty' not in report


@pytest.mark.parametrize('standalone,difficulty,success,pass4', [
    (True, 'hard', 0, 0),
    (True, 'medium', 15, 35),
    (False, 'hard', 0, 0),
    (False, 'hard', 15, 35),
    (False, 'medium', 40, 75),
])
def test_validation_scores_preserve_run_schedule(monkeypatch, tmp_path, standalone, difficulty, success, pass4):
    out = tmp_path / 'run'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(environment='flh', flh_difficulty=difficulty, seed=42, steps=400,
                                      learning_rate=1e-6, max_response_tokens=20480,
                                      train_batch_size=1, flat_group_size=4, save_every=10, eval_every=25,
                                      dataset_sha256=f'{difficulty}-data', frozen_maps_sha256=f'{difficulty}-maps')))
    argv = ['train.py', '--method', 'grpo', '--config', str(config), '--output-dir', str(out)]
    if standalone:
        argv.append('--validate-base')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setenv('PATR_PROGRESS_PATH', '')
    monkeypatch.setattr(hardware, 'annotate_hardware', lambda *args: None)
    monkeypatch.setattr(q35_thinking_policy, 'check_prompt_contract', lambda *args: {})
    monkeypatch.setattr(train, 'build_models', lambda *args: (Mock(), None, None, Mock()))
    data = {'validation': [dict(uid=str(i)) for i in range(100)]}
    if not standalone:
        data.update(train=[dict(uid='train')], test=[dict(uid='test')])
    monkeypatch.setattr(train, 'make_datasets', lambda cfg: data)
    monkeypatch.setattr(train.signal, 'signal', Mock())
    monkeypatch.setattr(train.torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(train.torch.cuda, 'max_memory_allocated', lambda: 0)

    def evaluate(engine, tokenizer, cfg, tasks, path, seed, split):
        assert tasks == data[split]
        assert seed == (100042 if split == 'validation' else 200042)
        assert cfg['flh_difficulty'] == difficulty and cfg['frozen_maps_sha256'] == f'{difficulty}-maps'
        metrics = dict(success_rate_pct=success, pass_at_4_pct=pass4, n_tasks=len(tasks),
                       n_trajectories=4 * len(tasks))
        train.atomic_json(path, metrics)
        return metrics

    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr(train, 'evaluate', evaluator)

    def collect(tasks, engine, tokenizer, cfg, method, rng):
        assert tasks == data['train']
        assert cfg['flh_difficulty'] == difficulty and cfg['frozen_maps_sha256'] == f'{difficulty}-maps'
        branches = [Branch(0, None, None, [1], [1], [0.], lineage=str(i)) for i in range(4)]
        return branches, {}, []

    optimizer = Mock()
    monkeypatch.setattr(train.torch.optim, 'AdamW', optimizer)
    monkeypatch.setattr(train, 'collect', collect)
    update = Mock(return_value={})
    monkeypatch.setattr(train, 'update', update)
    monkeypatch.setattr(train, 'sync_weights', Mock())
    monkeypatch.setattr(train, 'save_checkpoint', Mock())

    train.main()
    status = json.loads((out / 'status.json').read_text())
    if standalone:
        optimizer.assert_not_called()
        update.assert_not_called()
        assert (status['status'], status['step']) == ('base_validation_complete', 0)
        assert not (out / 'train_metrics.jsonl').exists()
        expected_evaluations = ['validation_0000.json']
    else:
        assert (status['status'], status['step']) == ('complete', 400)
        assert update.call_count == 400
        logged_steps = [json.loads(line)['step'] for line in (out / 'train_metrics.jsonl').read_text().splitlines()]
        assert logged_steps == list(range(1, 401))
        expected_evaluations = [f'validation_{step:04d}.json' for step in range(0, 401, 25)] + ['final_test.json']
    assert [call.args[4].name for call in evaluator.call_args_list] == expected_evaluations
    report = json.loads((out / 'flh_calibration.json').read_text())
    assert (report['difficulty'], report['pass_at_4_pct']) == (difficulty, pass4)
    assert (report['n_tasks'], report['n_trajectories'], report['max_response_tokens']) == (100, 400, 20480)
    assert report['maps_sha256'] == f'{difficulty}-maps' and 'next_difficulty' not in report
