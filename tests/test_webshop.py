"""Drive the HTTP bridge and RL adapters with a two-product test double."""
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
        self.session = str(session)
        self.server.user_sessions[self.session] = {'options': {}}
        return f'Find bag {session}', None

    def get_available_actions(self):
        choices = {'home': [], 'listing':['blue bag','red bag'], 'blue bag':['buy now', 'large'], 'red bag':['buy now', 'large'], 'done':[]}
        return {'has_search_bar': self.page == 'home','clickables': choices[self.page]}

    def step(self, action):
        if self.page == 'home' and action == 'search[bag]':
            self.page = 'listing'
        elif self.page == 'listing' and action in ('click[blue bag]','click[red bag]'):
            self.page = action[6:-1]
        elif self.page in ('blue bag','red bag') and action == 'click[large]':
            self.server.user_sessions[self.session]['options']['size'] = 'large'
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
        color = 'blue bag' if i%2 else 'red bag'
        core.apply_response(b,sampled(tok,'```click['+color+']```'),tok,cfg)
        core.apply_response(b,sampled(tok,'```click[buy now]```'),tok,cfg)
        assert b.terminal == ('success' if i%2 else 'purchase_incomplete')
        branches.append(b)
    m = core.evaluation_metrics(branches,[task])
    assert m['success_rate_pct']==50 and m['pass_at_4_pct']==100
    assert m['webshop_mean_score'] == pytest.approx(.7)
    assert len(gigpo_weights(branches,1)) == 12
    assert not sim.env.server.user_sessions


def test_full_catalog_requires_matching_index_and_paths(tmp_path):
    from scripts.configure_webshop_full import configure
    from scripts.webshop_server import full_catalog_identity
    utils = tmp_path/'web_agent_site/utils.py'
    utils.parent.mkdir()
    utils.write_text("DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2_1000.json')\n"
                     "DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle_1000.json')\n")
    configure(tmp_path)
    assert '1000' not in utils.read_text()
    once = utils.read_bytes()
    configure(tmp_path)
    assert utils.read_bytes() == once
    env = NS(server=NS(all_products=[{'asin':'X'}],search_engine=NS(num_docs=1)))
    with pytest.raises(ValueError):
        full_catalog_identity(env)
    env.server.all_products *= 100001
    with pytest.raises(ValueError):
        full_catalog_identity(env)
    env.server.search_engine.num_docs = 100001
    assert full_catalog_identity(env)['products'] == 100001


def test_only_visible_actions_reach_simulator(server):
    url, sim = server
    env = WebShopEnv({'goal_id':1500},{'webshop_url':url,'webshop_catalog':sim.catalog})
    assert env.resolve('CLICK[buy now]') == 0
    env.step('SEARCH[bag]')
    assert env.resolve('search[hidden]')==0 and env.resolve('CLICK[BLUE BAG]')=='click[blue bag]'


def test_bridge_option_change_on_same_page(server):
    from task_envs import observed_state, same_outcome
    url, sim = server
    env = WebShopEnv({'goal_id':1500}, {'webshop_url':url, 'webshop_catalog':sim.catalog})
    for command in ('search[bag]', 'click[blue bag]'):
        env.step(command)
    before = observed_state(env)
    env.step('click[large]')
    assert env.page['selected_options'] == {'size':'large'} and env.page['observation'] == before[0]
    assert not same_outcome(before, observed_state(env))


def test_ucf_skips_empty_argument_before_valid_action(server):
    from task_envs import turn_outcome
    url, sim = server
    cfg = dict(json.loads((ROOT/'configs/common.json').read_text()), environment='webshop',
               max_turns=15, webshop_url=url, webshop_catalog=sim.catalog)
    task = {'environment':'webshop','goal_id':1500,'uid':'webshop_1500'}
    tok = AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-4B', local_files_only=True)
    text = '```click[ ]``` then ```search[bag]``` tail'
    for method in ('grpo', 'ucf'):
        branch = core.start_branch(task, 0, tok, dict(cfg, method=method), '0')
        core.apply_response(branch, sampled(tok, text), tok, cfg)
        if method == 'ucf':
            assert branch.actions == ['search[bag]'] and turn_outcome(branch)[0] == 'listing'
        else:
            assert branch.actions == [0]


def test_ucf_evaluation_uses_flat_rollouts_and_valid_argument_protocol(server, monkeypatch):
    import numpy as np
    from ucf.ucf_v31 import make_collect
    url, sim = server
    cfg = dict(json.loads((ROOT/'configs/common.json').read_text()), environment='webshop',
               max_turns=1, webshop_url=url, webshop_catalog=sim.catalog)
    task = {'environment':'webshop','goal_id':1500,'uid':'webshop_1500'}
    tok = AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-4B', local_files_only=True)
    monkeypatch.setitem(sys.modules, 'vllm', NS(SamplingParams=lambda **params: NS(**params)))
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', NS(RequestOutputKind=NS(FINAL_ONLY=1)))
    pending = []
    def enqueue(rid, prompt, params):
        assert params.temperature == cfg['eval_temperature'] and params.top_p == cfg['eval_top_p']
        out = sampled(tok, '```click[ ]``` after ```search[bag]```')
        pending.append(NS(request_id=rid, finished=True, outputs=[out]))
    engine = NS(llm_engine=NS(add_request=enqueue, step=lambda: pending))
    branches, _, _ = make_collect(core, core.collect)([task], engine, tok, cfg, 'grpo',
                                                        np.random.default_rng(42), evaluation=True)
    assert len(branches) == cfg['eval_samples'] and all(b.actions == ['search[bag]'] for b in branches)
