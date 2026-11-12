"""Exercise run.py --smoke, its runtime, train.main, collection and all objectives on CPU."""
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
from transformers import AutoTokenizer

import patr_core
import run
import train
from scripts import hardware
from ucf import component_grad


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(run.ROOT / 'metadata/Qwen3.5-2B', local_files_only=True)


class ScriptedEngine:
    """Only the regular sampled-token request API is available."""
    def __init__(self, tok):
        self.tok = tok
        self.pending = []

    def add_request(self, rid, prompt, params):
        assert prompt["prompt_token_ids"] and params.max_tokens > 0 and params.logprobs == 1
        if int(rid.rsplit("-", 1)[-1]) % 2 == 0:
            text, reason = 'Push the box. ```Right```', 'stop'
            ids = self.tok.encode(text, add_special_tokens=False)
        else:
            reason = 'length'
            ids = [self.tok.encode('?', add_special_tokens=False)[0]] * params.max_tokens
            text = self.tok.decode(ids)
        sample = NS(token_ids=ids, logprobs=[{i:NS(logprob=-.2)} for i in ids],
                    text=text, finish_reason=reason)
        self.pending.append(NS(request_id=rid, finished=True, outputs=[sample]))

    def step(self):
        ready, self.pending = self.pending, []
        return ready


@pytest.mark.parametrize("alias,estimator,m,prior,beta,override", [
    ("ucf", "unbiased", 3, "inverse_turn", .25, True),
    ("ucf-merge", "merge", 2, "inverse_turn", .25, False),
    ("ucf-success", "success", 2, "inverse_turn", .25, False),
    ("ucf-m1", "unbiased", 1, "inverse_turn", .25, False),
    ("ucf-uniform", "unbiased", 2, "uniform", .25, False),
    ("no-part-b", "unbiased", 2, "inverse_turn", 0., False),
])
def test_smoke_from_alias_through_collector_and_objective(
        alias, estimator, m, prior, beta, override, tmp_path, tokenizer, monkeypatch):
    tasks = [dict(uid=f'test-{j}', environment='sokoban', walls=[], player=[2,1],
                  box=[2,2], goal=[2,3]) for j in range(8)]
    monkeypatch.setattr(train, 'make_datasets', lambda cfg: dict(train=tasks, validation=tasks))
    monkeypatch.setattr(patr_core, 'frozen_maps', lambda cfg: {t['uid']:t for t in tasks})
    monkeypatch.setattr(hardware, 'annotate_hardware', lambda *args: None)
    monkeypatch.setattr(train.signal, 'signal', lambda *args: None)
    monkeypatch.setenv('PATR_PROGRESS_PATH', '')
    # train.main installs these module globals just once in a CLI process.
    for name in ('collect', 'training_weights'):
        monkeypatch.setattr(train, name, getattr(train, name))

    sampler = ModuleType('vllm')
    sampler.__version__ = '0.19.1'
    def sampling_params(*, temperature, top_p, top_k, max_tokens, logprobs, seed, stop_token_ids, output_kind):
        assert (temperature, top_p, top_k) == (1., 1., -1)
        return NS(max_tokens=max_tokens, logprobs=logprobs)
    sampler.SamplingParams = sampling_params
    monkeypatch.setitem(sys.modules, 'vllm', sampler)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', NS(RequestOutputKind=NS(FINAL_ONLY=1)))
    monkeypatch.setitem(sys.modules, 'scoped_inference',
                        NS(configure_train=lambda *args: NS(transitions=[], training=lambda: None)))
    actor = NS(parameters=lambda: [], save_pretrained=lambda *args, **kwargs: None)
    monkeypatch.setattr(tokenizer, 'save_pretrained', lambda *args: None)
    monkeypatch.setattr(train, 'build_models', lambda cfg, seed:
                        (actor, None, NS(llm_engine=ScriptedEngine(tokenizer)), tokenizer))
    monkeypatch.setattr(train.torch.optim, 'AdamW', lambda *args, **kwargs: None)
    monkeypatch.setattr(train, 'evaluate', lambda *args: {})
    monkeypatch.setattr(train, 'sync_weights', lambda *args: None)
    monkeypatch.setattr(train, 'save_checkpoint', lambda *args: None)
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(train.torch.cuda, 'max_memory_allocated', lambda *args: 0)
    monkeypatch.setattr(run, 'check_weights', lambda *args: None)
    monkeypatch.setattr(run, 'version',
                        {'torch':'2.10.0','vllm':'0.19.1','transformers':'5.22.2'}.__getitem__)
    monkeypatch.setattr(component_grad, 'component_grad',
                        lambda actor, ref, bs, cfg, tok, part: len(component_grad.component_rows(bs, cfg, part)))

    def inspect_update(actor, ref, optimizer, branches, cfg, method, tok, rng, n_tasks):
        assert method == cfg['method'] == 'ucf'
        assert (cfg['ucf_estimator'],cfg['ucf_resamples'],cfg['ucf_turn_prior'],cfg['ucf_beta']) == (
            estimator, m, prior, beta)
        assert (cfg['steps'],cfg['train_batch_size'],cfg['mini_batch_tasks'],cfg['eval_every']) == (1,4,4,1)
        assert cfg['max_response_tokens'] == 48
        rows = train.training_weights(branches, n_tasks)
        leaves = [b for b in branches if getattr(b, 'ucf', {}).get('kind') == 'cf']
        extra = [(b,a,w) for b,a,w in rows if hasattr(b,'ucf')]
        assert leaves and all(b.ucf['resamples'] == m for b in leaves)
        if beta == 0:
            assert not extra
        elif estimator == 'unbiased':
            assert extra and all(a != 0 for _,a,_ in extra)
        elif estimator == 'merge':
            assert len(extra) == len(leaves) and any(a < 0 for _,a,_ in extra)
        else:
            assert 0 < len(extra) < len(leaves) and all(a >= 0 for _,a,_ in extra)
        return {'mock_weighted_rows':len(rows)}
    monkeypatch.setattr(train, 'update', inspect_update)

    def execute(executable, argv, env):
        assert Path(argv[1]) == run.ROOT / 'runtime/train_scoped.py'
        assert argv[2] == '--config' and argv[-1] == '--smoke'
        monkeypatch.setattr(sys, 'argv', argv[1:])
        saved_path = sys.path[:]
        try:
            runpy.run_path(argv[1], run_name='__main__')
        finally:
            sys.path[:] = saved_path
    monkeypatch.setenv('VLLM_BATCH_INVARIANT', '1')
    monkeypatch.setattr(run.os, 'execve', execute)
    argv = ['run.py','--size','2b','--env','sokoban','--method',alias,'--smoke',
            '--model-dir',str(run.ROOT/'metadata/Qwen3.5-2B'),'--output-root',str(tmp_path),
            '--token-budget','48']
    if override:
        argv.extend(['--ucf-resamples',str(m)])
    monkeypatch.setattr(sys, 'argv', argv)
    run.main()
    out = tmp_path/'smoke'/f'2b_sokoban_{alias}_seed42'
    assert json.loads((out/'status.json').read_text())['status'] == 'smoke_complete'
    progress = json.loads((out/'progress.json').read_text())
    assert (progress['phase'], progress['active'], progress['step']) == ('smoke_complete', 0, 1)
    record = json.loads((out/'train_metrics.jsonl').read_text())
    assert record['ucf_version'] == 'v3.1'
    assert sum(record['ucf_preupdate_fullbatch_components'].values()) == record['mock_weighted_rows']
    assert set(record['ucf_preupdate_fullbatch_components']) == (
        {'A','B'} if estimator=='unbiased' else {'originals','leaves'})


def test_engine_uses_only_regular_sampled_logprobs(monkeypatch):
    from unittest.mock import MagicMock
    from scripts import q35_engine_loader

    model = MagicMock()
    tokenizer = MagicMock()
    monkeypatch.setattr(train.AutoModelForCausalLM, 'from_pretrained', model)
    monkeypatch.setattr(train.AutoTokenizer, 'from_pretrained', tokenizer)
    monkeypatch.setitem(sys.modules, 'vllm', NS(LLM=object))
    builder = MagicMock()
    monkeypatch.setattr(q35_engine_loader, 'build_engine', builder)
    cfg = run.read(run.CONFIGS/'common.json')
    train.build_models(dict(cfg, method='ucf', model='synthetic'), 42)
    options = builder.call_args.kwargs
    assert 'logits_processors' not in options and options['max_logprobs'] == 20


def test_archived_checkpoint_rejected_before_model_construction(tmp_path, monkeypatch):
    cfg = dict(run.read(run.CONFIGS/'common.json'), environment='sokoban', method='ucf')
    launch = tmp_path/'launch.json'
    launch.write_text(json.dumps(cfg))
    train.torch.save(dict(config=dict(ucf_v3=True)), tmp_path/'checkpoint.pt')
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(launch), '--method', 'ucf',
                                      '--output-dir', str(tmp_path), '--resume'])
    with pytest.raises(ValueError, match='v3-final'):
        train.main()
    assert not (tmp_path/'status.json').exists()
