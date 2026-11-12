"""Reporting-only protocol additions. Never used for selection or stopping."""
import json,gzip
from pathlib import Path
from collections import Counter

def group_counts(branches,n):
    groups={i:[] for i in range(n)}
    for b in branches:groups[b.task_id].append(b)
    # Flat collectors always provide the initial four completed trajectories.
    assert all(len(bs)==4 for bs in groups.values())
    counts=Counter('all_failure' if sum(b.reward>0 for b in bs)==0 else 'all_success' if sum(b.reward>0 for b in bs)==4 else 'mixed' for bs in groups.values())
    return dict(initial_group_counts={k:counts[k] for k in ['all_success','all_failure','mixed']},initial_group_fractions={k:counts[k]/n for k in ['all_success','all_failure','mixed']})

def leaf_group_counts(branches,n,initial_not_applicable=True):
    groups={i:[b for b in branches if b.task_id==i] for i in range(n)}
    assert n>0 and all(groups.values()), 'Each sampled task must have at least one real leaf'
    counts=Counter('all_success' if all(b.reward>0 for b in bs) else 'all_failure' if not any(b.reward>0 for b in bs) else 'mixed' for bs in groups.values())
    result=dict(leaf_group_counts={k:counts[k] for k in ['all_success','all_failure','mixed']},leaf_group_fractions={k:counts[k]/n for k in ['all_success','all_failure','mixed']},leaf_group_source='all actual completed leaves; no pseudo rows or importance weighting',leaf_group_total=len(branches))
    if initial_not_applicable:result.update(initial_group_counts=None,initial_group_source='not defined: initial roots fork before termination; no diagnostic rollouts added')
    return result
