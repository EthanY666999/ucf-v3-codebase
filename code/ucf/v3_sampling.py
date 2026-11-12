"""CPU primitives for UCF V3: uniform causal admission and complete complements.

The partition is over next tokens. Its residual class is not a guarantee that
the eventual parsed action is STAY. Tests concern this explicitly stated
sampling estimator, not an unbiasedness claim for the full PPO objective.
"""
import math
import numpy as np


def question_probabilities(candidate_counts, target=2.0):
    """The held-out question cannot tune its own probability via its suffix."""
    if len(candidate_counts) < 2 or not math.isfinite(target) or target <= 0:
        raise ValueError('Need at least two questions and a positive finite budget')
    if any(not isinstance(n, int) or n < 0 for n in candidate_counts.values()):
        raise ValueError('Candidate counts must be nonnegative integers')
    total = sum(candidate_counts.values())
    return {k: min(1.0, target / ((total-n)/(len(candidate_counts)-1)))
            if total > n else 1.0 for k, n in candidate_counts.items()}


def normalized_vector(token_ids, logprobs, vocab_size):
    """Accept the FULL vocabulary only, allowing an identical sampled-token copy."""
    ids = np.asarray(token_ids, dtype=np.int64)
    values = np.asarray(logprobs, dtype=np.float64)
    if ids.shape != values.shape or ids.ndim != 1:
        raise ValueError('Malformed vocabulary logprobs')
    if len(ids) < vocab_size or np.any(ids < 0) or np.any(ids >= vocab_size):
        raise ValueError('Truncated or out-of-range vocabulary')
    order = np.argsort(ids, kind='stable')
    si, sv = ids[order], values[order]
    duplicate = si[1:] == si[:-1]
    if np.any(duplicate & (sv[1:] != sv[:-1])):
        raise ValueError('Inconsistent duplicate token probabilities')
    unique = np.r_[True, si[1:] != si[:-1]]
    if not np.array_equal(si[unique], np.arange(vocab_size)):
        raise ValueError('Missing vocabulary support')
    lp = sv[unique]
    if np.any(np.isnan(lp)) or np.any(np.isposinf(lp)) or not np.any(np.isfinite(lp)):
        raise ValueError('Invalid full-vocabulary probabilities')
    peak = float(np.max(lp))
    log_z = peak + math.log(float(np.exp(lp-peak).sum()))
    # FP32 log-softmax has tiny summation error. Larger drift is a backend error,
    # not a license to normalize a partial action vocabulary or hide bad scores.
    if abs(log_z) > 1e-4:
        raise ValueError(f'Full-vocabulary normalization drift: {log_z}')
    return lp-log_z, log_z


def full_vector_from_output(output, vocab_size):
    flat = output.logprobs
    if hasattr(flat, 'start_indices'):
        if len(flat.start_indices) != 1:
            raise ValueError('Probe must produce exactly one position')
        a, b = flat.start_indices[0], flat.end_indices[0]
        return normalized_vector(flat.token_ids[a:b], flat.logprobs[a:b], vocab_size)
    if len(flat) != 1:
        raise ValueError('Probe must produce exactly one position')
    row = flat[0]
    return normalized_vector(list(row), [p.logprob for p in row.values()], vocab_size)


def complement_sample(lp, original_token, mapping, classes, seed):
    """Exact categorical sampling from the full complement, stable in log space."""
    original = classes[mapping.get(original_token, 0)]
    mask = np.full(len(lp), original != classes[0], dtype=bool)
    for token, action in mapping.items():
        mask[token] = classes[action] != original
    pc = float(np.exp(lp[mask]).sum())
    po = float(np.exp(lp[~mask]).sum())
    if not math.isclose(po+pc, 1., rel_tol=0., abs_tol=1e-10):
        raise ValueError('Partition does not cover the vocabulary')
    support = np.flatnonzero(mask & np.isfinite(lp))
    info = dict(original_class=original, first_action=mapping.get(original_token, 0),
                pi_o=po, pi_cf=pc, unsampled_mass=0., parse_mismatch=False,
                sampling_support='full vocabulary complement',
                cf_fallback=len(support)==0)
    if not len(support):
        info['fallback_reason'] = 'empty_positive_probability_complement'
        return None, info
    scaled = np.exp(lp[support]-np.max(lp[support]))
    scaled /= scaled.sum()
    token = int(np.random.default_rng(seed).choice(support, p=scaled))
    assert classes[mapping.get(token, 0)] != original
    residual_mask = mask.copy()
    residual_mask[list(mapping)] = False
    info.update(forced_token=token, forced_old_logprob_unrestricted=float(lp[token]),
                residual_conditional_mass=float(np.exp(lp[residual_mask]).sum()/pc) if pc else 0.,
                sampled_residual=token not in mapping,
                pi_cf_underflow=pc==0.)
    # Even when total complement mass underflows, log-space sampling remains
    # defined. Its finite-precision estimator coefficient is zero and is logged.
    return token, info
