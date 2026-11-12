"""Measure the pre-update gradient of each component of the selected objective."""
import math

from ucf.ucf_weights import training_weights_ucf


def component_names(cfg):
    return ('A', 'B') if cfg['ucf_estimator'] == 'unbiased' else ('originals', 'leaves')


def component_rows(branches, cfg, part):
    names = component_names(cfg)
    if part not in names:
        raise ValueError(f"Unknown component {part!r} for {cfg['ucf_estimator']}")
    n_tasks = len({branch.task_id for branch in branches})
    rows = training_weights_ucf(branches, n_tasks, cfg=cfg)
    return [row for row in rows if hasattr(row[0], 'ucf') == (part == names[1])]


def component_grad(actor, ref, branches, cfg, tok, part):
    import torch
    import train

    assert getattr(actor.config, 'attention_dropout', 0.) == 0., 'Diagnostic requires dropout-free actor'
    rows = component_rows(branches, cfg, part)
    was_training = actor.training
    actor.train()
    actor.zero_grad(set_to_none=True)
    loss_total = 0.
    try:
        for chunk in train.microbatches(rows, cfg):
            ids, attn, mask, adv, weights, old = train.pack(chunk, tok.pad_token_id, next(actor.parameters()).device)
            current = train.selected_logprobs(actor, ids, attn, mask, gradient=True)
            with torch.no_grad():
                reference = train.selected_logprobs(ref, ids, attn, mask)
            ratio = (current - old).exp()
            clipped = ratio.clamp(1 - cfg['clip_low'], 1 + cfg['clip_high'])
            pg = -torch.minimum(ratio * adv, clipped * adv)
            log_ratio = reference - current
            kl = log_ratio.exp() - log_ratio - 1
            loss = ((pg + cfg['kl_coefficient'] * kl) * weights).sum()
            loss.backward()
            loss_total += loss.item()
            del ids, attn, mask, adv, weights, old, current, reference, ratio, clipped, pg, log_ratio, kl, loss
        norm = math.sqrt(sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None))
        assert math.isfinite(norm) and math.isfinite(loss_total)
        return dict(loss=loss_total, grad_norm=norm)
    finally:
        actor.zero_grad(set_to_none=True)
        actor.train(was_training)
