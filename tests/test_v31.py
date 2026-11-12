"""CPU oracles for the three UCF objectives and the published baseline weights."""
import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from ucf.component_grad import component_names, component_rows
from external_baselines import gigpo_weights, starpo_select
from patr_core import Branch, training_weights
from test_external_baselines import trajectories
from ucf.ucf_weights import fork_records, training_weights_ucf


def config(estimator="unbiased", beta=1.):
    return dict(ucf_estimator=estimator, ucf_beta=beta)


def trajectory(reward, length, lineage="0"):
    # One generated token in the tested turn, followed by a second generated turn.
    mask = [0, 1, 0] + [1] * (length - 1)
    return Branch(0, None, None, list(range(len(mask))), mask, [-.1]*len(mask),
                  lineage=lineage, reward=reward)


@dataclass(frozen=True)
class Sample:
    outcome: str
    probability: float
    reward: float
    length: int


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("law", [
    (Sample("A", .5, 1, 2), Sample("B", .5, 0, 4)),
    (Sample("A", .125, 1, 2), Sample("A", .25, .5, 5),
     Sample("B", .375, .75, 3), Sample("C", .25, 0, 4)),
])
def test_unbiased_exhaustive_per_trajectory_normalization(law, m):
    siblings = [trajectory(0., 2, str(j)) for j in (1, 2, 3)]
    expected = sum(s.probability * s.reward / s.length for s in law)
    total = wrong_original_denominator = 0.
    for original_sample, *draws in itertools.product(law, repeat=m + 1):
        original = trajectory(original_sample.reward, original_sample.length)
        different = [s for s in draws if s.outcome != original_sample.outcome]
        probability = math.prod(s.probability for s in [original_sample] + draws)
        # Conditional expectation over the uniform choice in D, with multiplicities.
        for selected in different or [None]:
            leaf = trajectory(selected.reward, selected.length, "leaf") if selected else None
            records = fork_records(original, leaf, turn_start=1, rho=1.,
                                   resamples=m, different=len(different))
            rows = training_weights_ucf([original] + siblings + records, 1, cfg=config())
            for b, advantage, weight in rows:
                if not hasattr(b, "ucf"):
                    continue
                # Undo fixed 1/4 task-group weighting and the factor 2 in the Part B advantage.
                assert b.ntokens == 1
                term = probability * 2 * advantage * weight / max(1, len(different))
                total += term
                wrong_original_denominator += term * b.ucf["trajectory_tokens"] / original_sample.length
    assert total == pytest.approx(expected, abs=2e-13)
    if len(law) == 2:
        assert total == pytest.approx(1/4)
        assert wrong_original_denominator == pytest.approx(3/16)


def test_unbiased_full_turn_mask_and_sibling_baseline():
    originals = [trajectory(r, 5, str(j)) for j, r in enumerate([1., 0., .5, .25])]
    leaf = trajectory(.75, 8, "leaf")
    records = fork_records(originals[0], leaf, turn_start=3, rho=.125, resamples=4, different=1)
    rows = training_weights_ucf(originals + records, 1, cfg=config(beta=.25))
    assert rows[:4] == training_weights(originals, 1)
    for (row, a, w), full, coefficient in zip(rows[4:], [originals[0], leaf], [.75, .25]):
        assert row.mask == [0, 0, 0] + full.mask[3:]
        assert row.old_logprobs == full.old_logprobs
        assert a == 2 * (full.reward - .25)
        assert w == .25 * coefficient / (.125 * 4 * full.ntokens)
    assert training_weights_ucf(originals + records, 1, cfg=config(beta=0.)) == rows[:4]


def test_zero_part_b_row_contributes_neither_policy_gradient_nor_kl():
    originals = [trajectory(1., 3, str(j)) for j in range(4)]
    records = fork_records(originals[0], None, turn_start=1, rho=.5, resamples=2, different=0)
    log_ratio = 1.
    nonzero_kl = math.exp(log_ratio) - log_ratio - 1.
    assert nonzero_kl > 0.
    rows = training_weights_ucf(originals + records, 1, cfg=config())
    assert rows == training_weights(originals, 1)  # No extra row reaches pack or the loss.
    part_b = component_rows(originals + records, config(), "B")
    assert not part_b
    # The real optimizer sees exactly the Part A mask/advantages/weights.
    import torch
    from train import pack
    _, _, _, advantages, weights, old = pack(rows, 0, 'cpu')
    current = old
    reference = current + 1.
    ratio = (current - old).exp()
    log_ratio = reference - current
    kl = log_ratio.exp() - log_ratio - 1.
    assert torch.all(kl > 0)
    assert (weights * kl).sum().item() == pytest.approx(nonzero_kl)
    assert (weights * advantages * ratio).sum().item() == 0.
    assert sum(w * b.ntokens * nonzero_kl for b, _, w in part_b) == 0.
    assert sum(w * b.ntokens * a for b, a, w in part_b) == 0.


@pytest.mark.parametrize("estimator", ["merge", "success"])
def test_merge_statistics_before_masking_and_success_filter(estimator):
    originals = [trajectory(r, 5, str(j)) for j, r in enumerate([1., .5, .5, 0.])]
    leaves = [trajectory(1., 7, "positive"), trajectory(0., 4, "negative")]
    records = [record for original, leaf in zip(originals, leaves)
               for record in fork_records(original, leaf, turn_start=3, rho=.125,
                                           resamples=2, different=1)]
    cfg = config(estimator)
    branches = originals + records
    rows = training_weights_ucf(branches, 1, cfg=cfg)
    expected = originals + (leaves if estimator == "merge" else leaves[:1])
    assert [b.lineage for b, _, _ in rows] == [b.lineage for b in expected]
    rewards = np.array([b.reward for b in originals + leaves])
    for (row, adv, weight), source in zip(rows, expected):
        assert adv == (source.reward - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)
        assert weight == 1 / (6 * source.ntokens)
        assert row.ids == source.ids and row.old_logprobs == source.old_logprobs
        assert row.mask == (source.mask if source in originals else [0, 0, 0] + source.mask[3:])
    names = component_names(cfg)
    assert names == ("originals", "leaves")
    assert [b.lineage for part in names for b, _, _ in component_rows(branches, cfg, part)] == [
        b.lineage for b in expected]
    # Assign a nonzero sampled KL only to the negative leaf.
    nonzero_kl = math.exp(1.) - 2
    leaf_kl = sum(w * b.ntokens * nonzero_kl for b, _, w in rows if b.lineage == "negative")
    assert leaf_kl == pytest.approx(3 * nonzero_kl / 24 if estimator == "merge" else 0.)


def test_success_keeps_leaf_with_zero_advantage():
    originals = [trajectory(.5, 4, str(j)) for j in range(4)]
    records = fork_records(originals[0], trajectory(.5, 5, "zero"), turn_start=3,
                           rho=.25, resamples=2, different=1)
    rows = training_weights_ucf(originals + records, 1, cfg=config("success"))
    assert rows[-1][0].lineage == "zero" and rows[-1][1] == 0.


@pytest.mark.parametrize("method,digest", [
    ("grpo8", "5cdfdc6a41061f280e0fb6f70a604da6659cdb2a6eb0e96ce7bb5f827781f96f"),
    ("grpo4", "3bcb35c9715fcdb38e56cba5a60c576f83741a629945c318d1abac482006ccc8"),
    ("gigpo", "0b41cb1c26de3855834501723e1af899299949fbd86cb210e8b989f3d2e6d74e"),
    ("starpo-s", "6e8b98b13684296d0be3377833202c83b4aa5f5f9488e5c10f410958bde10371"),
])
def test_baseline_rows_bitwise_against_v3_final(method, digest):
    # Fingerprints from the tagged implementation. float.hex preserves every bit.
    branches = []
    for task in range(2):
        group = trajectories(task)
        if method == 'grpo8':
            group += trajectories(task)
            for j, branch in enumerate(group):
                branch.lineage = str(j)
        branches.extend(group)
    branches[-1].reward = 1
    if method == "starpo-s":
        selected, _, _ = starpo_select(branches, 2, .5)
        rows = training_weights(selected, 1)
    elif method == "gigpo":
        rows = gigpo_weights(branches, 2)
    else:
        rows = training_weights(branches, 2)
    encoded = json.dumps([(b.task_id, b.lineage, b.ids, b.mask, b.old_logprobs, a.hex(), w.hex())
                          for b, a, w in rows], separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest
