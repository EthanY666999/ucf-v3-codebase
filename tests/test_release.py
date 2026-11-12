import json
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

import patr_core as core
from response_protocol import first_block
from scripts.q35_thinking_policy import check_prompt_contract

ROOT = Path(__file__).resolve().parents[1]
MATRIX = [(size, p.stem) for size in ('2b','4b') for p in (ROOT/'configs'/size).glob('*.json')]


def command(size, env, method, *extra):
    return [sys.executable, str(ROOT/'run.py'), '--size', size, '--env', env, '--method', method, *extra]


def test_published_environments():
    assert set(MATRIX) == {('2b', 'sokoban'), ('2b', 'flh'),
                           ('4b', 'sokoban'), ('4b', 'flh'), ('4b', 'webshop')}
    old = subprocess.run(command('2b', 'frozenlake', 'grpo4', '--print-config'), capture_output=True)
    assert old.returncode != 0 and b'invalid choice' in old.stderr


@pytest.mark.parametrize('size,env', MATRIX)
def test_supported_presets_render_through_cli(size, env):
    for method in json.loads((ROOT/'configs/methods.json').read_text()):
        cfg = json.loads(subprocess.check_output(command(size,env,method,'--print-config'), cwd=ROOT.parent))
        assert cfg['environment'] == env and cfg['flat_group_size'] == (8 if method=='grpo8' else 4)
        assert cfg['ucf_beta'] == (0 if method=='no-part-b' else .25)
        assert cfg['ucf_variant'] == 'turn_resample' and cfg['eval_every'] == 25
        assert cfg['ucf_target_forks'] == 3.0
        assert cfg['ucf_resamples'] == (1 if method == 'ucf-m1' else 2)
        if env == 'flh':
            assert cfg['flh_difficulty'] == 'hard'
            assert (cfg['max_turns'], cfg['max_response_tokens']) == (20, 20480)
    if env != 'webshop':
        data = core.make_datasets(cfg)
        assert [len(data[x]) for x in ('train','validation','test')] == [10000,100,100]
        uid = [{t['uid'] for t in data[x]} for x in ('train','validation','test')]
        assert not (uid[0]&uid[1] or uid[0]&uid[2] or uid[1]&uid[2])
        cmd = command(size,env,'ucf','--model-dir',str(ROOT/'metadata'/f'Qwen3.5-{size.upper()}'),'--check')
        subprocess.run(cmd, cwd=ROOT.parent, check=True)
        for split in ('train','validation','test'):
            grid = core.make_env(data[split][0], cfg)
            assert grid.render() and len(grid.step(1)) == 4


@pytest.mark.parametrize('size', ['2b', '4b'])
def test_medium_flh_uses_distinct_frozen_maps_and_output(size):
    flags = ['--flh-difficulty', 'medium']
    cfg = json.loads(subprocess.check_output(command(size, 'flh', 'grpo4', *flags, '--print-config')))
    hard = json.loads(subprocess.check_output(command(size, 'flh', 'grpo4', '--print-config')))
    assert (cfg['max_turns'], cfg['max_response_tokens'], cfg['flh_difficulty']) == (20, 20480, 'medium')
    assert cfg['run_label'] != hard['run_label']
    assert cfg['actor_device'] == hard['actor_device']
    data = core.make_datasets(cfg)
    maps = core.frozen_maps(cfg)
    hard_maps = {t['map_sha256'] for tasks in core.make_datasets(hard).values() for t in tasks}
    tasks = [t for split in data.values() for t in split]
    assert [len(data[k]) for k in ('train', 'validation', 'test', 'supplemental')] == [10000, 100, 100, 500]
    assert len({t['map_sha256'] for t in tasks} - hard_maps) == len(tasks)
    assert set(maps) == {t['uid'] for t in tasks}
    assert all(5 <= t['shortest_path_length'] <= 9 for t in tasks)
    model = str(ROOT / 'metadata' / f'Qwen3.5-{size.upper()}')
    subprocess.run(command(size, 'flh', 'grpo4', *flags, '--model-dir', model, '--check'), check=True)


@pytest.mark.parametrize('size',['2B','4B'])
def test_production_token_and_dialog_contract(size):
    tok = AutoTokenizer.from_pretrained(ROOT/'metadata'/f'Qwen3.5-{size}',local_files_only=True)
    assert check_prompt_contract(tok)['passed']
    text = 'Working. ```search[waterproof backpack]``` other'
    assert first_block(tok.decode(tok.encode(text)), 'webshop').group(1) == 'search[waterproof backpack]'


def test_archived_launch_is_rejected_before_gpu_or_writes(tmp_path):
    import shutil
    out = tmp_path / 'train/2b_sokoban_ucf_seed42'
    out.mkdir(parents=True)
    old = ROOT/'docs/archive/V3_S_ucf_budget10240_from16_seed42.json'
    shutil.copyfile(old, out/'launch.json')
    invalid = subprocess.run(command('2b','sokoban','ucf','--resume','--output-root',str(tmp_path)),
                             capture_output=True)
    assert invalid.returncode and b'v3-final' in invalid.stderr
    assert list(out.iterdir()) == [out/'launch.json']


def test_failed_steps_archived_before_replaying(tmp_path):
    from scripts.run_recovery import archive_uncommitted
    raw = b'{"step":1}\n{"step":2}\n{"part'
    path = tmp_path/'train_metrics.jsonl'
    path.write_bytes(raw)
    rollout = tmp_path/'rollouts_0002.jsonl.gz'
    rollout.write_bytes(b'old')
    logs = tmp_path/'ucf_logs'
    logs.mkdir()
    for filename in ('select_0001.jsonl', 'candidates_0002.jsonl', 'select_0002.jsonl',
                     'budget_0002.json', 'raw_leaves_0002.jsonl.gz'):
        (logs/filename).write_bytes(b'evidence')
    archive_uncommitted(tmp_path,1)
    assert list(logs.iterdir()) == [logs/'select_0001.jsonl']
    assert path.read_bytes() == b'{"step":1}\n'
    folder, = (tmp_path/'uncommitted').iterdir()
    assert (folder/path.name).read_bytes()==raw
    assert (folder/rollout.name).read_bytes()==b'old'
    archive_uncommitted(tmp_path,1)
    assert not rollout.exists()
