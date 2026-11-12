"""Drive the HTTP bridge and RL adapters with a two-product test double."""
import copy
import json
import subprocess
import sys
import threading
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from transformers import AutoTokenizer

import patr_core as core
from cert_collect import replay
from external_baselines import gigpo_weights
from scripts.prepare_webshop import prepare
from scripts.webshop_server import Simulator, handler_for
from webshop import WebShopEnv, bridge_call

ROOT = Path(__file__).resolve().parents[1]


class CatalogFixture:
    def __init__(self):
        self.server = NS(goals=[{'instruction_text': f'Find bag {i}'} for i in range(1600)], user_sessions={})
        self.browser = NS(current_url=None)

    def reset(self, session):
        assert type(session) is int
        self.page = 'home'
        self.server.user_sessions[str(session)] = self.page
        return f'Find bag {session}', None

    def get_available_actions(self):
        choices = {'home': [], 'listing':['blue bag','red bag'], 'blue bag':['buy now'], 'red bag':['buy now'], 'done':[]}
        return {'has_search_bar': self.page == 'home','clickables': choices[self.page]}

    def step(self, action):
        if self.page == 'home' and action == 'search[bag]':
            self.page = 'listing'
        elif self.page == 'listing' and action in ('click[blue bag]','click[red bag]'):
            self.page = action[6:-1]
        elif self.page in ('blue bag','red bag') and action == 'click[buy now]':
            score = 1. if self.page == 'blue bag' else .4
            self.page = 'done'
            return 'Purchased', score, True, {}
        return self.page, 0., False, {}


@pytest.fixture(scope='module')
def server():
    sim = Simulator(CatalogFixture())
    http = HTTPServer(('127.0.0.1', 0), handler_for(sim))
    t = threading.Thread(target=http.serve_forever, daemon=True)
    t.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}', sim
    finally:
        http.shutdown()
        t.join()
        http.server_close()


def sampled(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return NS(token_ids=ids, logprobs=[{i:NS(logprob=-.1)} for i in ids], text=text, finish_reason='stop')


def test_bridge_freezes_disjoint_ids_and_refuses_catalog_drift(server,tmp_path):
    url, sim = server
    manifest = prepare(bridge_call(url,'/health'), tmp_path)
    assert manifest['sizes'] == {'train':100,'validation':100,'test':500}
    prepare(sim.health(), tmp_path)
    with pytest.raises(ValueError):
        prepare(dict(sim.health(),catalog='0'*64), tmp_path)
    subprocess.run([sys.executable,str(ROOT/'run.py'),'--size','4b','--env','webshop','--method','ucf',
                    '--model-dir',str(ROOT/'metadata/Qwen3.5-4B'),'--webshop-data',str(tmp_path),
                    '--webshop-url',url,'--check'],check=True)
    with pytest.raises(RuntimeError):
        WebShopEnv({'goal_id':1500},{'webshop_url':url,'webshop_catalog':'outdated'})


def test_action_trajectory_replay_rewards_and_gigpo(server):
    url, sim = server
    cfg = dict(json.loads((ROOT/'configs/common.json').read_text()), environment='webshop', max_turns=15,
               webshop_url=url,webshop_catalog=sim.catalog)
    task = {'environment':'webshop','goal_id':1500,'uid':'webshop_1500'}
    tok = AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-4B',local_files_only=True)
    branches = []
    for i in range(4):
        b = core.start_branch(task,0,tok,cfg,str(i))
        prompt_len = len(b.ids)
        out = sampled(tok,'Thinking ```search[bag]``` then ```click[buy now]```')
        core.apply_response(b,out,tok,cfg)
        assert b.actions == ['search[bag]'] and 'blue bag' in b.env.render()
        count = b.ntokens
        assert b.ids[prompt_len:prompt_len+count] == out.token_ids[:count]
        assert b.mask[prompt_len+count] == 0
        node = {'branch':copy.deepcopy(b),'boundary':len(b.ids),'j':i,'turn':2,'state':b.env.s}
        child = replay(core,node,task,cfg)
        assert child.env.anchor() == b.env.anchor()
        color = 'blue bag' if i%2 else 'red bag'
        core.apply_response(b,sampled(tok,'```click['+color+']```'),tok,cfg)
        core.apply_response(b,sampled(tok,'```click[buy now]```'),tok,cfg)
        assert b.terminal == ('success' if i%2 else 'purchase_incomplete')
        assert child.env.history == ['search[bag]']
        branches.append(b)
    m = core.evaluation_metrics(branches,[task])
    assert m['success_rate_pct']==50 and m['pass_at_4_pct']==100
    assert m['webshop_mean_score'] == pytest.approx(.7)
    assert len(gigpo_weights(branches,1)) == 12
    assert not sim.env.server.user_sessions


def test_webshop_v3_collector_full_complement_on_cpu(server,tmp_path,monkeypatch):
    """Use scripted generation; run the production capture store, selection, fork, replay and HT rows."""
    import importlib
    import math
    from types import ModuleType
    import numpy as np
    from ucf_v3 import make_collect
    from ucf_weights import training_weights_ucf

    pytest.importorskip('torch')
    for name in ('vllm','vllm.sampling_params','vllm.v1','vllm.v1.sample',
                 'vllm.v1.sample.logits_processor','vllm.v1.sample.logits_processor.interface'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['vllm'].SamplingParams = lambda **options: NS(**options)
    sys.modules['vllm.sampling_params'].RequestOutputKind = NS(FINAL_ONLY=1)
    interface = sys.modules['vllm.v1.sample.logits_processor.interface']
    interface.LogitsProcessor, interface.MoveDirectionality = object, NS(SWAP=1)
    monkeypatch.delitem(sys.modules,'original_logits',raising=False)
    capture_module = importlib.import_module('original_logits')
    from first_fence import locate_token_prefix

    url, sim = server
    tok = AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-4B',local_files_only=True)
    vocab = len(tok)
    lp = -math.log(vocab)
    original_tokens = tok.encode('```search[bag]```',add_special_tokens=False)
    boundary = locate_token_prefix(original_tokens,0,len(original_tokens),tok)

    class Backend:
        model_config = NS(get_vocab_size=lambda: vocab)
        def __init__(self):
            self.queue = []
        def add_request(self, rid, prompt, params):
            tokens = prompt['prompt_token_ids']
            tag = getattr(params,'extra_args',{}).get('ucf_original_logits')
            if tag:
                cap, key = capture_module._active, tuple(tag['key'])
                cap.register(key,tokens,params)
                cap.save(key,tokens,original_tokens[:boundary],np.full(vocab,lp,dtype=np.float32))
                out = sampled(tok,'```search[bag]```')
            else:
                out = sampled(tok,' then ```click[blue bag]```')
            out.logprobs = [{t:NS(logprob=lp)} for t in out.token_ids]
            self.queue.append(NS(request_id=rid,finished=True,outputs=[out]))
        def step(self):
            rows,self.queue = self.queue,[]
            return rows

    cfg = json.loads((ROOT/'configs/common.json').read_text())
    cfg.update(environment='webshop',max_turns=1,webshop_url=url,webshop_catalog=sim.catalog,
               ucf_log_dir=str(tmp_path/'ucf_logs'))
    (tmp_path/'status.json').write_text('{"current_step":1}')
    monkeypatch.setenv('PATR_PROGRESS_PATH',str(tmp_path/'progress.json'))
    tasks = [{'goal_id':i,'uid':f'webshop_{i}','environment':'webshop'} for i in (1500,1501)]
    branches,metrics,events = make_collect(core,core.collect)(
        tasks,NS(llm_engine=Backend()),tok,cfg,'ucf',np.random.default_rng(42))
    assert metrics['original_logits_capture']['captured_nodes']==8
    assert metrics['original_logits_capture']['pending_vectors']==0 and metrics['mean_forks']>0
    assert metrics['initial_group_counts']['all_failure']==2
    assert all(e['stratum']['original_class']=='SEARCH' and e['stratum']['unsampled_mass']==0 for e in events)
    weights = training_weights_ucf(branches,2)
    assert len(weights)>8 and all(np.isfinite(w) and w>0 for _,_,w in weights)
    for b, _, _ in weights:
        if hasattr(b,'ucf'):
            assert b.ucf['rho']==.5 and len(b.ids)==len(b.old_logprobs)==len(b.mask)
