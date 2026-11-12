import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import train
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
    assert report['next_difficulty'] == ('medium' if outcome == 'below_target' else 'hard')


@pytest.mark.parametrize('standalone,difficulty,success,pass4,expected_status', [
    (True, 'hard', 15, 35, 'base_validation_complete'),
    (True, 'medium', 15, 35, 'base_validation_complete'),
    (False, 'hard', 3, 8, 'evaluate_medium'),
    (False, 'hard', 15, 35, 'reaches_optimizer'),
])
def test_calibration_at_training_entry(monkeypatch, tmp_path, standalone, difficulty, success, pass4, expected_status):
    out = tmp_path / 'run'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(environment='flh', flh_difficulty=difficulty, seed=42, steps=400,
                                      learning_rate=1e-6, max_response_tokens=20480,
                                      dataset_sha256='dataset-fixture', frozen_maps_sha256='maps-fixture')))
    argv = ['train.py', '--method', 'grpo', '--config', str(config), '--output-dir', str(out)]
    if standalone:
        argv.append('--validate-base')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setenv('PATR_PROGRESS_PATH', '')
    monkeypatch.setattr(hardware, 'annotate_hardware', lambda *args: None)
    monkeypatch.setattr(q35_thinking_policy, 'check_prompt_contract', lambda *args: {})
    monkeypatch.setattr(train, 'build_models', lambda *args: (Mock(), None, None, None))
    validation = [dict(uid=str(i)) for i in range(100)]
    # Accessing a training or test split in validation-only mode fails the test.
    monkeypatch.setattr(train, 'make_datasets', lambda cfg: {'validation': validation})

    def evaluate(engine, tokenizer, cfg, tasks, path, seed, split):
        assert tasks == validation and split == 'validation' and seed == 100042
        assert Path(path).name == 'validation_0000.json'
        metrics = dict(success_rate_pct=success, pass_at_4_pct=pass4, n_tasks=len(tasks),
                       n_trajectories=4 * len(tasks))
        train.atomic_json(path, metrics)
        return metrics

    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr(train, 'evaluate', evaluator)

    class ReachedOptimizer(Exception):
        pass

    optimizer = Mock(side_effect=ReachedOptimizer)
    monkeypatch.setattr(train.torch.optim, 'AdamW', optimizer)
    if expected_status == 'reaches_optimizer':
        with pytest.raises(ReachedOptimizer):
            train.main()
        optimizer.assert_called_once()
    else:
        train.main()
        optimizer.assert_not_called()
        status = json.loads((out / 'status.json').read_text())
        assert (status['status'], status['step']) == (expected_status, 0)
    evaluator.assert_called_once()
    report = json.loads((out / 'flh_calibration.json').read_text())
    assert (report['difficulty'], report['pass_at_4_pct']) == (difficulty, pass4)
    assert (report['n_tasks'], report['n_trajectories'], report['max_response_tokens']) == (100, 400, 20480)
    assert not (out / 'final_test.json').exists() and not (out / 'checkpoint.pt').exists()
