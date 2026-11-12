"""Experimental scope for vLLM0.19.1's process-global inference-only kernels.

Imported by the standalone training entry point. It is pinned to the upstream
implementation and must pass real CUDA backward/rollout transition checks.
"""
import functools
import threading


class InferenceScope:
    def __init__(self, torch, backend):
        self.torch, self.backend = torch, backend
        self.owner = threading.get_ident()
        self.native_bmm = torch.bmm
        self.fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        self.bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        self.blas = torch.backends.cuda.preferred_blas_library()
        self.precision = [x.fp32_precision for x in self.precision_objects()]
        self.transitions = []

    def precision_objects(self):
        t = self.torch
        return [t.backends.cuda.matmul, t.backends.cudnn.conv, t.backends.cudnn.rnn]

    def _check_owner(self):
        assert threading.get_ident() == self.owner, 'Not safe for concurrent inference/training'

    def inference(self):
        self._check_owner()
        b, t = self.backend, self.torch
        if b._batch_invariant_MODE:
            return
        b.enable_batch_invariant_mode()
        for x in self.precision_objects():
            x.fp32_precision = 'ieee'
        assert t.bmm is b.bmm_batch_invariant
        self.transitions.append('inference')

    def training(self):
        self._check_owner()
        b, t = self.backend, self.torch
        if not b._batch_invariant_MODE:
            assert t.bmm is self.native_bmm
            return
        # This is exactly the destruction operation used by torch.library's
        # own _scoped_library context manager. No installed file is modified.
        assert b._batch_invariant_LIB is not None
        b._batch_invariant_LIB._destroy()
        b._batch_invariant_LIB = None
        b._batch_invariant_MODE = False
        t.bmm = self.native_bmm
        t.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.fp16
        t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.bf16
        t.backends.cuda.preferred_blas_library(self.blas)
        for x, value in zip(self.precision_objects(), self.precision):
            x.fp32_precision = value
        self.transitions.append('training')
        assert t.bmm is self.native_bmm


def tiny_native_gradients(torch, autocast=False):
    """Deterministic gradients for the native linear+bmm+softmax+mean path."""
    x = torch.arange(48, device='cuda', dtype=torch.float32).reshape(2, 3, 8) / 48
    w = torch.arange(40, device='cuda', dtype=torch.float32).reshape(5, 8) / 40
    x = x.detach().requires_grad_(); w = w.detach().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):
        z = torch.nn.functional.linear(x, w)
        y = torch.bmm(z, z.transpose(1, 2)).log_softmax(-1).square().mean()
    y.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    return [x.grad.detach().clone(), w.grad.detach().clone()]


def configure_train(train, torch, vllm):
    """Use the same scope wiring in the isolated contract and training entrypoint.

    The entrypoint imports train as a module (rather than executing train.py),
    so component_grad's import of train uses these wrapped functions too.
    """
    from vllm.model_executor.layers import batch_invariant as backend
    assert torch.__version__.split('+')[0] == '2.10.0'
    assert vllm.__version__ == '0.19.1'
    assert not hasattr(train, '_v3_inference_scope'), 'Already configured'
    scope = InferenceScope(torch, backend)
    original_llm = vllm.LLM

    def llm(*args, **kwargs):
        kwargs['attention_config'] = {'backend': 'FLASH_ATTN'}
        return original_llm(*args, **kwargs)

    vllm.LLM = llm
    original_build = train.build_models

    @functools.wraps(original_build)
    def build(cfg, seed):
        scope.training()
        result = original_build(cfg, seed)
        engine = result[2]
        original_step = engine.llm_engine.step

        @functools.wraps(original_step)
        def step(*args, **kwargs):
            scope.inference()
            return original_step(*args, **kwargs)

        engine.llm_engine.step = step
        scope.training()
        return result

    train.build_models = build

    def native_call(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            scope.training()
            return fn(*args, **kwargs)
        return call

    for name in ('selected_logprobs', 'update', 'sync_weights'):
        setattr(train, name, native_call(getattr(train, name)))
    train._v3_inference_scope = scope
    return scope
