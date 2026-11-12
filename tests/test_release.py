import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

import patr_core as core
from cert_entropy import token_map
from first_fence import locate_token_prefix
from response_protocol import first_block
from scripts.q35_thinking_policy import check_prompt_contract
from webshop import CLASSES, family_tokens
from v3_sampling import complement_sample

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
    for method in ['ucf','gigpo','grpo4','grpo8','starpo-s','no-part-b'] + ([] if env=='webshop' else ['token-partition']):
        cfg = json.loads(subprocess.check_output(command(size,env,method,'--print-config'), cwd=ROOT.parent))
        assert cfg['environment'] == env and cfg['flat_group_size'] == (8 if method=='grpo8' else 4)
        assert cfg['ucf_beta'] == (0 if method=='no-part-b' else .25)
        assert cfg['ucf_variant'] == 'uniform' and cfg['eval_every'] == 25
        assert cfg['ucf_probability_source'] == 'original_sampling_logits'
        assert cfg['fork_point'] == 'first_fence'
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
    grid,_ = token_map(tok)
    assert set(grid.values()) == {1,2,3,4}
    web = family_tokens(tok)
    assert set(web.values()) == {1,2}
    text = 'Working. ```search[waterproof backpack]``` other'
    ids = tok.encode(text, add_special_tokens=False)
    bd = locate_token_prefix(ids, 0, len(ids), tok)
    assert tok.decode(ids[:bd]).endswith('```')
    assert web[ids[bd]] == 1 and first_block(text,'webshop').group(1) == 'search[waterproof backpack]'
    lp = np.log(np.repeat(1/len(tok),len(tok)))
    forced, label = complement_sample(lp, ids[bd], web, CLASSES, 42)
    assert CLASSES[web.get(forced,0)] != 'SEARCH' and label['unsampled_mass'] == 0.


def test_selection_references_and_historical_anchor():
    base = ROOT/'protocols'
    actual = json.loads((base/'V3_S_ucf_budget10240_from16_seed42.json').read_text())
    current = json.loads(subprocess.check_output(command('2b','sokoban','ucf','--print-config')))
    for k in ('train_batch_size','mini_batch_tasks','clip_high','clip_low','max_response_tokens',
              'max_context_tokens','ucf_beta','ucf_target_forks','fork_point','ucf_probability_source','dataset_sha256'):
        assert current[k] == actual[k]
    assert 'anchor16' not in current
    invalid = subprocess.run(command('4b','flh','ucf','--anchor16','cp.pt','--print-config'),capture_output=True)
    assert invalid.returncode


def test_failed_steps_archived_before_replaying(tmp_path):
    from scripts.run_recovery import archive_uncommitted
    raw = b'{"step":1}\n{"step":2}\n{"part'
    path = tmp_path/'train_metrics.jsonl'
    path.write_bytes(raw)
    rollout = tmp_path/'rollouts_0002.jsonl.gz'
    rollout.write_bytes(b'old')
    archive_uncommitted(tmp_path,1)
    assert path.read_bytes() == b'{"step":1}\n'
    folder, = (tmp_path/'uncommitted').iterdir()
    assert (folder/path.name).read_bytes()==raw
    assert (folder/rollout.name).read_bytes()==b'old'
    archive_uncommitted(tmp_path,1)
    assert not rollout.exists()
