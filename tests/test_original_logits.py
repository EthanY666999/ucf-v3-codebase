"""CPU contracts for the experimental original-sampling observer (no GPU use)."""
import copy
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality as D

from original_logits import CaptureSession, OriginalLogitsObserver, request_metadata


class Tok:
    clean_up_tokenization_spaces = False
    def decode(self, ids, **kwargs):
        return ''.join({1:'reason', 2:'`', 3:'``', 4:'```', 5:'Right', 6:'\n',
                        7:'```Right```', 8:' tail'}.get(x,'?') for x in ids)


def params(session, key=(0,'0',0)):
    return SamplingParams(temperature=1., top_p=1., top_k=-1, seed=42,
                          extra_args=request_metadata(session.serial, *key))


def add(index, p, prompt, output):
    return (index, p, prompt, output)


def test_sampling_tensor_and_all_rng_states_are_untouched():
    obs=OriginalLogitsObserver(None,'cpu',False);output=[]
    logits=torch.arange(16,dtype=torch.float32)[None]
    before=logits.clone()
    with CaptureSession(Tok(),4,16) as cap:
        obs.update_state(BatchUpdate(1,[],[add(0,params(cap),[9],output)],[]))
        assert obs.apply(logits) is logits and not cap.records
        output.extend([1,4])
        py=random.getstate();np_state=np.random.get_state();cpu=torch.random.get_rng_state().clone()
        assert obs.apply(logits) is logits
        assert random.getstate()==py
        now=np.random.get_state()
        assert now[0]==np_state[0] and np.array_equal(now[1],np_state[1]) and now[2:]==np_state[2:]
        assert torch.equal(cpu,torch.random.get_rng_state()) and torch.equal(logits,before)
        prefix,lp=cap.records[(0,'0',0)]
        assert prefix==(9,1,4) and lp.dtype==np.float32
        assert np.array_equal(lp,torch.log_softmax(logits[0],-1).numpy())
        # Later suffixes cannot overwrite the captured node or probabilities.
        output.extend([5,4,8]);obs.apply(logits+17)
        assert cap.captured==1 and cap.records[(0,'0',0)][0]==prefix
    assert cap.summary()['pending_vectors']==1


@pytest.mark.parametrize('pieces',[[1,2,3],[1,3,2],[1,2,2,2],[1,4],[4]])
def test_fragmented_fence_is_captured_at_same_causal_prefix(pieces):
    obs=OriginalLogitsObserver(None,'cpu',False);output=[];logits=torch.zeros(1,16)
    with CaptureSession(Tok(),1,16) as cap:
        obs.update_state(BatchUpdate(1,[],[add(0,params(cap),[9],output)],[]))
        for token in pieces:
            obs.apply(logits);output.append(token)
        obs.apply(logits)
        assert cap.records[(0,'0',0)][0]==tuple([9]+pieces)
        assert cap.captured==1


def test_batch_swaps_removal_addition_and_preemption_do_not_mix_requests():
    obs=OriginalLogitsObserver(None,'cpu',False);a=[];b=[]
    logits=torch.stack([torch.arange(16,dtype=torch.float32),-torch.arange(16,dtype=torch.float32)])
    with CaptureSession(Tok(),3,16) as cap:
        pa=params(cap,(0,'0',0));pb=params(cap,(1,'0',0))
        obs.update_state(BatchUpdate(2,[],[add(0,pa,[10],a),add(1,pb,[11],b)],[]))
        obs.update_state(BatchUpdate(2,[],[],[(0,1,D.SWAP)]))
        a.append(4);b.append(4);obs.apply(logits)
        assert np.array_equal(cap.records[(0,'0',0)][1],logits[1].log_softmax(-1).numpy())
        assert np.array_equal(cap.records[(1,'0',0)][1],logits[0].log_softmax(-1).numpy())
        # Same index removed and then replaced in one update.
        c=[4];pc=params(cap,(2,'0',0))
        obs.update_state(BatchUpdate(2,[0],[add(0,pc,[12],c)],[]));obs.apply(logits)
        assert cap.records[(2,'0',0)][0]==(12,4)
        # A preempted already-captured request may rejoin with its existing prefix.
        obs.update_state(BatchUpdate(2,[1],[add(1,pa,[10],a)],[]));obs.apply(logits)
        assert cap.captured==3


def test_unidirectional_move_keeps_only_source_and_unmarked_request_is_noop():
    obs=OriginalLogitsObserver(None,'cpu',False);out=[]
    with CaptureSession(Tok(),1,16) as cap:
        obs.update_state(BatchUpdate(3,[],[add(2,params(cap),[9],out)],[]))
        obs.update_state(BatchUpdate(1,[],[],[(2,0,D.UNIDIRECTIONAL)]))
        assert set(obs.entries)=={0}
        out.append(4);obs.apply(torch.zeros(1,16))
        obs.update_state(BatchUpdate(1,[0],[add(0,SamplingParams(),[8],[])],[]))
        assert not obs.entries
    x=torch.ones(1,16);assert obs.apply(x) is x


def test_retention_is_bounded_and_take_validates_original_tokens_and_logprob():
    obs=OriginalLogitsObserver(None,'cpu',False);logits=torch.zeros(2,16)
    with CaptureSession(Tok(),2,16) as cap:
        obs.update_state(BatchUpdate(2,[],[add(i,params(cap,(i,'0',0)),[9],[4]) for i in range(2)],[]))
        obs.apply(logits)
    n=dict(j=0,turn=1,boundary=2,branch=NS(ids=[9,4]))
    cap.retain({0:[n]});assert len(cap.records)==1
    original=NS(ids=[9,4,5],old_logprobs=[0,0,float(logits[0].log_softmax(-1)[5])])
    lp,z=cap.take(0,n,original)
    assert abs(np.exp(lp).sum()-1)<1e-12 and abs(z)<1e-4
    assert cap.summary()['peak_vector_bytes']==2*16*4 and not cap.records


@pytest.mark.parametrize('change',[{'temperature':.7},{'top_p':.9},{'top_k':20},
                                  {'frequency_penalty':.1},{'min_tokens':1},
                                  {'allowed_token_ids':[1]},{'logit_bias':{1:1.}}])
def test_nonmatching_sampling_distribution_is_rejected(change):
    with CaptureSession(Tok(),1,16) as cap:
        p=params(cap)
        for k,v in change.items():setattr(p,k,v)
        with pytest.raises(AssertionError):OriginalLogitsObserver.validate_params(p)


def test_capacity_key_reuse_and_missed_boundaries_fail_instead_of_silent_fallback():
    obs=OriginalLogitsObserver(None,'cpu',False)
    with CaptureSession(Tok(),1,16) as cap:
        p=params(cap);cap.register((0,'0',0),[9],p)
        with pytest.raises(AssertionError):cap.register((0,'0',0),[10],p)
        with pytest.raises(AssertionError):cap.register((1,'0',0),[9],p)
        obs.update_state(BatchUpdate(1,[],[add(0,p,[9],[4,5])],[]))
        with pytest.raises(AssertionError):obs.apply(torch.zeros(1,16))


def test_fence_token_already_contains_complete_action_is_not_a_candidate():
    obs=OriginalLogitsObserver(None,'cpu',False)
    with CaptureSession(Tok(),1,16) as cap:
        obs.update_state(BatchUpdate(1,[],[add(0,params(cap),[9],[7])],[]))
        obs.apply(torch.zeros(1,16))
        assert not cap.records and cap.closed_at_fence==1


def test_exception_releases_active_session_and_cpu_vectors():
    cap=CaptureSession(Tok(),1,16)
    with pytest.raises(RuntimeError):
        with cap:
            cap.records[(0,'0',0)]=((9,),np.zeros(16,dtype=np.float32))
            raise RuntimeError('synthetic')
    assert not cap.records
    with CaptureSession(Tok(),1,16):pass


def test_take_rejects_mismatched_actual_sample_probability():
    cap=CaptureSession(Tok(),1,16)
    cap.records[(0,'0',0)]=((9,4),np.full(16,-np.log(16),dtype=np.float32))
    n=dict(j=0,turn=1,boundary=2,branch=NS(ids=[9,4]))
    with pytest.raises(AssertionError,match='sampling probability mismatch'):
        cap.take(0,n,NS(ids=[9,4,5],old_logprobs=[0,0,-1.]))
