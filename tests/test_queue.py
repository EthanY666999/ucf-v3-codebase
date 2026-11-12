import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import collect_results
import run_queue


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def queue(tmp_path, entries):
    path = tmp_path / 'queue.json'
    dump(path, entries)
    return SimpleNamespace(size='2b', env='sokoban', model_dir=tmp_path / 'model',
                           run_root=tmp_path / 'runs', queue=path)


def config(command):
    # The preview takes the real parser/configuration path on CPU.
    return json.loads(subprocess.check_output([*command, '--print-config'], text=True))


@pytest.mark.parametrize('size,env', [
    ('2b', 'sokoban'), ('4b', 'sokoban'), ('2b', 'flh'), ('4b', 'flh'), ('4b', 'webshop'),
])
def test_default_queue_order(size, env, tmp_path, monkeypatch):
    args = queue(tmp_path, [])
    args.size, args.env, args.queue = size, env, run_queue.ROOT / 'configs/queue.json'
    def stop_at_first_launch(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(run_queue, 'run_child', stop_at_first_launch)
    with pytest.raises(KeyboardInterrupt):
        run_queue.run_queue(args)
    items = run_queue.read_json(args.run_root / 'queue_status.json')['items']
    expected = ['ucf', 'grpo4', 'gigpo', 'ucf-merge', 'ucf-success', 'ucf-m1', 'ucf-uniform', 'no-part-b', 'grpo8']
    if (size, env) == ('2b', 'sokoban'):
        expected.insert(1, 'ucf-20480')
        assert all((e['seed'], e['until']) == (42, 400) for e in items[:2])
    assert [e['method'] for e in items] == expected


def test_queue_retries_advances_and_skips_complete(monkeypatch, tmp_path):
    args = queue(tmp_path, [dict(method='ucf', until=25, seed=43), dict(method='ucf-20480', until=25, seed=43),
                            dict(method='grpo4'), dict(method='gigpo')])
    calls = []
    def mock_run(command, log, lock):
        cfg = config(command)
        out = args.run_root / 'train' / cfg['run_label']
        method = command[command.index('--method') + 1]
        resume = '--resume' in command
        calls.append((method, resume))
        dump(out / 'launch.json', cfg)
        (out / 'checkpoint.pt').touch()
        if method == 'grpo4' or (method == 'ucf-20480' and not resume):
            dump(out / 'status.json', dict(step=10, status='training'))
            raise ValueError('mock device failure')
        step = int(command[command.index('--until') + 1])
        dump(out / f'validation_{step:04}.json', dict(success_rate_pct=54))
        if step == 400:
            dump(out / 'final_test.json', dict(success_rate_pct=52))
        dump(out / 'status.json', dict(step=step, status='complete' if step == 400 else 'paused_checkpointed'))

    monkeypatch.setattr(run_queue, 'run_child', mock_run)
    assert run_queue.run_queue(args) == 1
    assert calls == [('ucf', False), ('ucf-20480', False), ('ucf-20480', True), ('grpo4', False), ('grpo4', True),
                     ('gigpo', False)]
    state = run_queue.read_json(args.run_root / 'queue_status.json')
    assert state['current_item'] is None
    first, long_budget, second, third = state['items']
    assert (first['completed_steps'], first['latest_validation']['step'], first['state']) == (25, 25, 'complete')
    assert (long_budget['completed_steps'], long_budget['state']) == (25, 'complete')
    short, long = [run_queue.read_json(args.run_root / 'train' / e['run_label'] / 'launch.json')
                   for e in (first, long_budget)]
    assert (short['method'], short['max_response_tokens']) == ('ucf', 10240)
    assert short['run_label'] != long['run_label']
    assert long == dict(short, max_response_tokens=20480, run_label=long_budget['run_label'])
    labels = {r['run'] for r in collect_results.collect_rows(args.run_root) if r['split'] == 'validation'}
    assert {short['run_label'], long['run_label']} <= labels
    assert second['state'] == 'failed' and 'device failure' in second['attempts'][-1]['error']
    assert third['state'] == 'complete'
    calls.clear()
    assert run_queue.run_queue(args) == 1
    assert calls == [('grpo4', True), ('grpo4', True)]


def test_failure_before_checkpoint_preserved_and_final_test_resumed(monkeypatch, tmp_path):
    args = queue(tmp_path, [dict(method='ucf')])
    def fail_once(command, log, lock):
        cfg = config(command)
        out = args.run_root / 'train' / cfg['run_label']
        dump(out / 'launch.json', cfg)
        if not (args.run_root / 'interrupted').exists():
            (out / 'early-failure').write_text('evidence')
            raise ValueError('failed before first checkpoint')
        assert '--resume' not in command
        (out / 'checkpoint.pt').touch()
        dump(out / 'status.json', dict(step=400, status='paused_checkpointed', final_evaluation_pending=True))
    monkeypatch.setattr(run_queue, 'run_child', fail_once)
    assert run_queue.run_queue(args) == 1
    archived, = (args.run_root / 'interrupted').iterdir()
    assert (archived / 'early-failure').read_text() == 'evidence'
    def finish_final(command, log, lock):
        assert '--resume' in command
        out = args.run_root / 'train' / config(command)['run_label']
        dump(out / 'status.json', dict(step=400, status='complete'))
        dump(out / 'final_test.json', dict(success_rate_pct=97))
    monkeypatch.setattr(run_queue, 'run_child', finish_final)
    assert run_queue.run_queue(args) == 0


def test_legacy_run_reports_v3_tag(monkeypatch, tmp_path):
    args = queue(tmp_path, [dict(method='ucf')])
    out = args.run_root / 'train/2b_sokoban_ucf_seed42'
    dump(out / 'launch.json', dict(ucf_v3=True))
    monkeypatch.setattr(run_queue, 'run_child', lambda *a: pytest.fail('legacy run launched'))
    assert run_queue.run_queue(args) == 1
    item, = run_queue.read_json(args.run_root / 'queue_status.json')['items']
    assert 'v3-final' in item['attempts'][-1]['error'] and (out / 'launch.json').exists()


def test_results_pool_counts_and_read_only_metrics(tmp_path, monkeypatch):
    out = tmp_path / 'train/2b_sokoban_ucf_seed42'
    dump(out / 'status.json', dict(status='complete', step=2))
    for name in ['validation_0000.json', 'validation_0002.json', 'final_test.json']:
        dump(out / name, dict(success_rate_pct=60, pass_at_4_pct=73, webshop_mean_score=.25, generated_tokens=88))
    costs = []
    for step, resamples, changed in [(1, 2, 2), (2, 8, 1)]:
        counts = dict(selected_nodes=resamples // 2, resamples=resamples, changed_resamples=changed,
                      changed_nodes=1, reward_changed_leaves=0, leaves=1, truncated_leaves=0)
        costs.append(json.dumps(dict(step=step, step_seconds=1800, generated_tokens=1000,
                    fork_counts=counts, initial_group_counts=dict(all_success=1, all_failure=2, mixed=3))))
    (out / 'train_metrics.jsonl').write_text('\n'.join(costs) + '\n')
    (out / 'checkpoint.pt').touch()
    (out / 'validation_0002.jsonl').touch()
    original_open = Path.open
    def only_metrics(path, *args, **kwargs):
        assert path.name in {'results.md', 'results.csv', 'train_metrics.jsonl', 'status.json',
                             'final_test.json', 'validation_0000.json', 'validation_0002.json'}
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', only_metrics)
    collect_results.write_results(tmp_path)
    assert 'Fixed validation' in (tmp_path / 'results.md').read_text()
    with (tmp_path / 'results.csv').open() as file:
        summary, base, final_validation, final = csv.DictReader(file)
    assert (summary['training_tokens'], summary['training_hours']) == ('2000', '1.0')
    assert summary['changed_resample_fraction'] == '0.3' and summary['mixed'] == '6'
    assert [(row['split'], row['step']) for row in (base, final_validation, final)] == [
        ('validation', '0'), ('validation', '2'), ('test', '2')]
    assert final['webshop_mean_score'] == '0.25'
