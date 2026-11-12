import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v3_sampling import question_probabilities, normalized_vector, complement_sample, full_vector_from_output


def test_own_suffix_cannot_change_own_admission():
    counts={0:12,1:16,2:18,3:20}
    before=question_probabilities(counts)
    for replacement in (0,1,80,800):
        after=question_probabilities(dict(counts,**{}) | {0:replacement})
        assert before[0]==after[0]


def test_long_horizon_budget_has_no_floor_overflow():
    rho=question_probabilities({i:80 for i in range(64)})
    assert all(p==.025 and 80*p==2 for p in rho.values())


def test_empty_other_questions_and_saturation():
    assert question_probabilities({0:20,1:0,2:0})[0]==1
    assert question_probabilities({0:1,1:1,2:1})=={0:1.,1:1.,2:1.}
    with pytest.raises(ValueError):question_probabilities({0:1})


def test_complete_vector_and_sampled_token_duplicate():
    p=np.array([.08,.3,.2,.22,.2])
    ids=[0,1,2,3,4,2]; lp=np.log(p)[ids]
    out,z=normalized_vector(ids,lp,5)
    np.testing.assert_allclose(np.exp(out),p,rtol=1e-14)
    assert abs(z)<1e-14
    with pytest.raises(ValueError):normalized_vector([0,1],np.log([.5,.5]),5)
    with pytest.raises(ValueError):normalized_vector(ids,lp+.01,5)
    lp[-1]-=.01
    with pytest.raises(ValueError):normalized_vector(ids,lp,5)


def test_flat_vector_adapter():
    f=SimpleNamespace(start_indices=[0],end_indices=[3],token_ids=[0,1,2],logprobs=np.log([.1,.2,.7]))
    lp,z=full_vector_from_output(SimpleNamespace(logprobs=f),3)
    np.testing.assert_allclose(np.exp(lp),[.1,.2,.7])


@pytest.mark.parametrize('original',[0,1,2,3,4])
def test_full_complement_never_draws_original_layer(original):
    p=np.array([.08,.3,.2,.22,.2]); mapping={1:1,2:2,3:3,4:4}
    cls={0:'STAY_OR_RESIDUAL',1:'STAY_OR_RESIDUAL',2:'FAIL',3:'GOAL',4:'CELL'}
    for seed in range(100):
        token,info=complement_sample(np.log(p),original,mapping,cls,seed)
        assert cls[mapping.get(token,0)]!=cls[mapping.get(original,0)]
        assert info['unsampled_mass']==0 and math.isclose(info['pi_o']+info['pi_cf'],1.)


def test_residual_mass_is_sampleable_not_dropped():
    p=np.array([.08,.3,.2,.22,.2]); mapping={1:1,2:2,3:3,4:4}
    cls={0:'STAY_OR_RESIDUAL',1:'STAY_OR_RESIDUAL',2:'FAIL',3:'GOAL',4:'CELL'}
    n=20000
    observed=sum(complement_sample(np.log(p),2,mapping,cls,seed)[0]==0 for seed in range(n))
    expected=.08/.8
    z=abs(observed-n*expected)/math.sqrt(n*expected*(1-expected))
    assert z<4


def test_conditional_ht_stratified_estimator_exact_enumeration():
    # Target: TWO unrestricted next-token samples for fixed prefix/baseline.
    # Does not test clipped PPO, random length normalization or token parsing.
    p=np.array([.08,.3,.2,.22,.2]); groups=np.array([0,0,1,2,3]); reward=np.array([0,0,0,1,1])
    baseline=.37
    score=np.eye(5)-p[None,:]
    f=(reward-baseline)[:,None]*score
    target=2*(p[:,None]*f).sum(axis=0)
    rho=.025
    expectation=np.zeros(5)
    for x,px in enumerate(p):
        mask=groups!=groups[x]; pc=p[mask].sum(); po=1-pc
        conditional_cf=(p[mask,None]*f[mask]).sum(axis=0)/pc
        selected_value=(2*po*f[x]+2*pc*conditional_cf)/rho
        expectation+=px*rho*selected_value
    np.testing.assert_allclose(expectation,target,atol=1e-14)


def test_tiny_complement_is_sampled_without_rejection_fallback():
    lp=np.array([math.log1p(-1e-12),math.log(1e-12)])
    token,info=complement_sample(lp,0,{1:1},{0:'residual',1:'other'},4)
    assert token==1 and not info['cf_fallback']
    assert math.isclose(info['pi_cf'],1e-12,rel_tol=1e-12)


def test_all_tokens_same_partition_has_explicit_fallback():
    token,info=complement_sample(np.log([.2,.8]),0,{1:1},{0:'same',1:'same'},4)
    assert token is None and info['cf_fallback']
