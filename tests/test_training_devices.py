"""Exercise optimizer packing and weight transfer using small CPU tensors."""
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from patr_core import Branch
import train


class TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(4, 4)
        self.layers = torch.nn.ModuleList([layer])

    def forward(self, input_ids, **unused):
        state = self.layers[0].mlp.down_proj(self.embedding(input_ids))
        return NS(last_hidden_state=state)


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TextModel()
        self.lm_head = torch.nn.Linear(4, 8)
        self.config = NS(tie_word_embeddings=False)

    def load_weights(self, pairs):
        with torch.no_grad():
            weights = dict(self.named_parameters())
            for name, value in pairs:
                assert value.device == weights[name].device and value.dtype == torch.bfloat16
                weights[name].copy_(value)
        return weights


def test_synchronization_and_optimizer_follow_model_devices(monkeypatch):
    torch.manual_seed(42)
    actor = Actor()
    reference = copy.deepcopy(actor).requires_grad_(False)
    inference = Actor().bfloat16()
    cache_resets = []
    backend = NS(model_executor=NS(driver_worker=NS(model_runner=NS(model=inference))),
                 reset_prefix_cache=lambda:cache_resets.append(True))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    assert train.sync_weights(actor,NS(llm_engine=backend)) == len(dict(actor.named_parameters()))
    assert cache_resets == [True]
    trajectories = [Branch(0,None,None,[1,2+j],[0,1],[0.,-2.],reward=float(j==0)) for j in range(4)]
    cfg = dict(mini_batch_tasks=1,micro_batch_size=2,micro_batch_token_limit=24,clip_low=.28,
               clip_high=.28,kl_coefficient=.001,max_grad_norm=1.)
    optimizer = torch.optim.AdamW(actor.parameters(),lr=.001,foreach=False)
    result = train.update(actor,reference,optimizer,trajectories,cfg,'grpo',NS(pad_token_id=0),np.random.default_rng(42),1)
    assert result['grad_norm'] > 0 and result['max_parameter_change'] > 0
    train.sync_weights(actor,NS(llm_engine=backend))
    assert torch.equal(inference.model.layers[-1].mlp.down_proj.weight,actor.model.layers[-1].mlp.down_proj.weight.bfloat16())
