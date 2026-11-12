"""Read-only V1 observer of probabilities at the actual sampled first-fence node.

The budget-10240 runs use this observer. A session is local to
the in-process, single-GPU vLLM worker. It changes no logits and consumes no RNG.
Only float32 CPU vectors are retained; unselected vectors are released before CF.
"""
import itertools
import math
import threading
from contextlib import AbstractContextManager

import numpy as np
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality

from first_fence import locate_token_prefix
from response_protocol import first_block

_active = None
_serial = itertools.count(1)


def request_metadata(session, task_id, lineage, turns):
    return {'ucf_original_logits': {'session': session,
            'key': [int(task_id), str(lineage), int(turns)]}}


class CaptureSession(AbstractContextManager):
    def __init__(self, tokenizer, max_nodes, vocab_size, environment=None):
        self.environment = environment
        self.tokenizer = tokenizer
        self.max_nodes = int(max_nodes)
        self.vocab_size = int(vocab_size)
        self.serial = next(_serial)
        self.owner = threading.get_ident()
        self.records = {}
        self.registered = {}
        self.captured = 0
        self.peak_vector_bytes = 0
        self.closed_at_fence = 0

    def __enter__(self):
        global _active
        assert _active is None, 'Overlapping capture sessions are unsupported'
        assert threading.get_ident() == self.owner
        _active = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active
        assert _active is self
        _active = None
        if exc_type is not None:
            self.records.clear()
        return False

    def register(self, key, prompt, params):
        assert threading.get_ident() == self.owner
        signature = (tuple(prompt), params.seed)
        if key in self.registered:
            assert self.registered[key] == signature, 'Request key was reused'
        else:
            assert len(self.registered) < self.max_nodes, 'Capture request capacity exceeded'
            self.registered[key] = signature

    def save(self, key, prompt, output, vector):
        assert key not in self.records, 'The same node was captured twice'
        assert vector.dtype == np.float32 and vector.shape == (self.vocab_size,)
        assert np.isfinite(vector).any() and not np.isnan(vector).any()
        assert not np.isposinf(vector).any()
        self.records[key] = (tuple(prompt) + tuple(output), vector)
        self.captured += 1
        self.peak_vector_bytes = max(self.peak_vector_bytes,
                                    len(self.records)*self.vocab_size*4)

    def retain(self, picked):
        wanted = {(ti, str(n['j']), n['turn']-1)
                  for ti, nodes in picked.items() for n in nodes}
        assert wanted <= self.records.keys(), 'Selected candidate lacks its original logits'
        self.records = {key: self.records[key] for key in wanted}

    def take(self, ti, node, original):
        key = (ti, str(node['j']), node['turn']-1)
        prefix, raw = self.records.pop(key)
        assert prefix == tuple(node['branch'].ids), 'Captured prefix differs from sampled IDs'
        assert len(prefix) == node['boundary']
        lp = raw.astype(np.float64)
        peak = float(lp.max())
        log_z = peak + math.log(float(np.exp(lp-peak).sum()))
        assert abs(log_z) <= 1e-4, f'Captured distribution is not normalized: {log_z}'
        lp -= log_z
        token = original.ids[node['boundary']]
        difference = float(lp[token] - original.old_logprobs[node['boundary']])
        assert abs(difference) <= 1e-4, f'Original sampling probability mismatch: {difference}'
        return lp, log_z

    def summary(self):
        return dict(source='original_sampling_logits', captured_nodes=self.captured,
                    closed_at_fence=self.closed_at_fence,
                    peak_vector_bytes=self.peak_vector_bytes,
                    capacity_vector_bytes=self.max_nodes*self.vocab_size*4,
                    pending_vectors=len(self.records), model_forward_probes=0)


class OriginalLogitsObserver(LogitsProcessor):
    """Return the original tensor unchanged; observe only explicitly tagged calls."""
    def __init__(self, vllm_config, device, is_pin_memory):
        self.entries = {}

    @classmethod
    def validate_params(cls, params):
        if not (params.extra_args or {}).get('ucf_original_logits'):
            return
        assert params.temperature == 1. and params.top_p == 1. and params.top_k == -1
        assert params.min_p == 0. and params.min_tokens == 0
        assert params.presence_penalty == params.frequency_penalty == 0.
        assert params.repetition_penalty == 1.
        assert not params.allowed_token_ids and not params.logit_bias and not params.bad_words

    def is_argmax_invariant(self):
        return True

    def update_state(self, batch_update):
        if batch_update is None:
            return
        # V1 interface contract: removals, additions, then moves/swaps.
        for index in batch_update.removed:
            self.entries.pop(index, None)
        for index, params, prompt, output in batch_update.added:
            self.entries.pop(index, None)
            tag = (params.extra_args or {}).get('ucf_original_logits')
            if tag is None:
                continue
            assert _active is not None and tag['session'] == _active.serial
            self.validate_params(params)
            key = tuple(tag['key'])
            assert prompt is not None
            _active.register(key, prompt, params)
            self.entries[index] = dict(session=_active, key=key, prompt=tuple(prompt),
                                       output=output, seen=0,
                                       finished=key in _active.records)
        for source, target, direction in batch_update.moved:
            a = self.entries.pop(source, None)
            b = self.entries.pop(target, None)
            if a is not None:
                self.entries[target] = a
            if direction == MoveDirectionality.SWAP and b is not None:
                self.entries[source] = b

    def apply(self, logits):
        if _active is None:
            return logits
        assert threading.get_ident() == _active.owner
        for index, state in self.entries.items():
            if state['session'] is not _active or state['finished']:
                continue
            output = state['output']
            size = len(output)
            assert size >= state['seen'], 'Output token history shrank'
            # Scan only newly appended tokens, decode the complete prefix only at
            # potential backticks. No successor is read to place the node.
            from token_boundaries import fence_token_positions
            possible = next(fence_token_positions(output, state['seen'], size,
                                                  _active.tokenizer), None)
            bd = (locate_token_prefix(output, 0, size, _active.tokenizer)
                  if possible is not None else None)
            state['seen'] = size
            if bd is None:
                continue
            assert bd == size, 'Missed fence boundary (speculation/preemption unsupported here)'
            state['finished'] = True
            if first_block(_active.tokenizer.decode(output, skip_special_tokens=True), _active.environment):
                _active.closed_at_fence += 1
                continue
            # This vector is copied before sampling the successor. No in-place
            # tensor operation, sample or generator draw is performed here.
            vector = logits[index].log_softmax(-1, dtype=torch.float32).cpu().numpy()
            _active.save(state['key'], state['prompt'], output, vector)
        return logits
