"""Causal first-fence node helpers. No suffix is used to decide eligibility."""
import math

def locate(ids, start, end, fence):
    """Return exclusive prefix end (the action boundary), or None."""
    for pos in range(start, end):
        if ids[pos] == fence:
            return pos + 1
    return None

def locate_token_prefix(ids, start, end, tokenizer):
    """Earliest sampled token completing the first textual fence.

    Handles spaced/fragmented fence tokens without scanning for a later
    canonical fence ID. Never inspect any token beyond the returned prefix.
    """
    from token_boundaries import fence_token_positions
    for pos in fence_token_positions(ids,start,end,tokenizer):
        if '```' in tokenizer.decode(ids[start:pos+1],skip_special_tokens=True):
            return pos+1
    return None

def action_masses(raw, mapping):
    assert set(raw) == set(mapping), 'Every action-token variant must be scored'
    assert all(math.isfinite(v) and v <= 1e-6 for v in raw.values())
    p = {a:sum(math.exp(raw[t]) for t in mapping if mapping[t] == a) for a in range(1,5)}
    assert sum(p.values()) <= 1 + 1e-6
    p[0] = max(0., 1 - sum(p.values()))
    return p

def segments(mask):
    start = None
    for pos, value in enumerate(list(mask) + [0]):
        if value and start is None: start = pos
        if not value and start is not None:
            yield start, pos
            start = None
