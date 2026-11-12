"""Turn-resampling objectives. Collection records always retain full trajectories."""
import copy
from collections import defaultdict

from patr_core import training_weights


def fork_records(original, leaf, *, turn_start, rho, resamples, different):
    """Preserve complete lengths, rewards, and the original sibling identity."""
    if not (0 < rho <= 1 and resamples > 0 and 0 <= different <= resamples):
        raise ValueError("Invalid turn-resampling probabilities or counts")
    if (leaf is not None) != (different > 0):
        raise ValueError("A changed outcome must have exactly one completed leaf")
    records = []
    for kind, trajectory in (("orig", original), ("cf", leaf)):
        if trajectory is None:
            continue
        assert trajectory.task_id == original.task_id
        record = copy.copy(trajectory)
        assert 0 <= turn_start < len(record.mask) and record.mask[turn_start]
        record.ucf = dict(kind=kind, original_lineage=original.lineage,
                          turn_start=turn_start, rho=rho, resamples=resamples,
                          different=different, trajectory_tokens=trajectory.ntokens)
        records.append(record)
    return records


def _turn_row(branch):
    """Select sampled tokens of one complete turn, retaining its conditioning IDs."""
    start = branch.ucf["turn_start"]
    end = start
    while end < len(branch.ids) and branch.mask[end]:
        end += 1
    row = copy.copy(branch)
    row.ids = branch.ids[:end]
    row.old_logprobs = branch.old_logprobs[:end]
    row.mask = [0] * start + branch.mask[start:end]
    return row


def _unbiased(originals, records, n_tasks, beta):
    rows = training_weights(originals, n_tasks)
    if beta == 0:
        return rows
    rewards = defaultdict(dict)
    for original in originals:
        rewards[original.task_id][original.lineage] = original.reward
    for branch in records:
        info = branch.ucf
        siblings = rewards[branch.task_id]
        assert len(siblings) == 4 and info["original_lineage"] in siblings
        baseline = sum(reward for lineage, reward in siblings.items()
                       if lineage != info["original_lineage"]) / 3
        advantage = 2 * (branch.reward - baseline)
        count = (info["different"] if info["kind"] == "cf"
                 else info["resamples"] - info["different"])
        if advantage == 0 or count == 0:
            continue
        coefficient = count / info["resamples"]
        weight = beta * coefficient / (info["rho"] * n_tasks * 4 * info["trajectory_tokens"])
        rows.append((_turn_row(branch), advantage, weight))
    return rows


def _merge(originals, records, n_tasks, keep_success):
    leaves = [b for b in records if b.ucf["kind"] == "cf"]
    # Normalize on the complete group and full lengths, before masking or filtering.
    rows = []
    for branch, advantage, weight in training_weights(originals + leaves, n_tasks):
        if hasattr(branch, "ucf"):
            if keep_success and advantage < 0:
                continue
            start = branch.ucf["turn_start"]
            branch = copy.copy(branch)
            branch.mask = [0] * start + branch.mask[start:]
        rows.append((branch, advantage, weight))
    return rows


def training_weights_ucf(branches, n_tasks, *, cfg):
    originals = [b for b in branches if not hasattr(b, "ucf")]
    assert all(sum(b.task_id == i for b in originals) == 4 for i in range(n_tasks))
    records = [b for b in branches if hasattr(b, "ucf")]
    estimator = cfg["ucf_estimator"]
    if estimator == "unbiased":
        return _unbiased(originals, records, n_tasks, cfg["ucf_beta"])
    if estimator in ("merge", "success"):
        return _merge(originals, records, n_tasks, keep_success=estimator == "success")
    raise ValueError(f"Unknown UCF estimator {estimator!r}")
