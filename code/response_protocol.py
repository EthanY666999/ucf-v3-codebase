"""Opt-in first complete action block protocol for future no-SFT runs.

Only sampled token prefixes are retained. No retokenization or synthetic
training target is permitted. The chat EOS added after a cutoff is context
only, with mask and old_logprob zero.
"""
import hashlib
import re
from types import MethodType, SimpleNamespace

NAME = 'first_complete_action_block'
BLOCK = re.compile(r'```\s*(left|down|right|up|[1-4])\s*```', re.I)
ACTIONS = dict(left=1, down=2, right=3, up=4)

def first_block(text, environment=None):
    if environment == 'webshop':
        from webshop import BLOCK as command_block
        return command_block.search(text)
    return BLOCK.search(text)

def parse_first_block(self, response):
    match = first_block(response)
    if match is None:
        return response, '0'
    word = match.group(1).lower()
    action = ACTIONS[word] if word in ACTIONS else int(word)
    return response[:match.start()].strip(), str(action)

def configure_agent(agent, cfg):
    if cfg.get('response_protocol') == NAME and cfg.get('environment') != 'webshop':
        agent._parse_model_response = MethodType(parse_first_block, agent)

def truncate_response(branch, out, tokenizer):
    # A counterfactual branch may already contain a sampled reasoning/fence
    # prefix for this same turn. Ordinary branches end with mask=0 prompts.
    start = len(branch.ids)
    while start and branch.mask[start-1]:
        start -= 1
    copied = list(branch.ids[start:])
    fresh = list(out.token_ids)
    full = copied + fresh
    text = tokenizer.decode(full, skip_special_tokens=True)
    match = first_block(text, getattr(getattr(branch, 'env', None), 'environment', None))
    if match is None:
        return out, None
    # Stop at the first sampled TOKEN completing the block. A token is atomic:
    # retain and record any characters sharing its closing-fence token.
    endpoint = None
    from token_boundaries import fence_token_positions
    for pos in fence_token_positions(full,len(copied),len(full),tokenizer):
        n=pos+1
        prefix = tokenizer.decode(full[:n], skip_special_tokens=True)
        if len(prefix) >= match.end():
            assert prefix.startswith(text[:match.end()])
            endpoint = n
            break
    assert endpoint is not None
    keep = endpoint - len(copied)
    assert keep > 0
    meta = dict(protocol=NAME, retained_sampled_tokens=endpoint,
                discarded_sampled_tokens=len(full)-endpoint,
                discarded_ids_sha256=hashlib.sha256(repr(full[endpoint:]).encode()).hexdigest(),
                context_eos_injected=True, context_eos_trained=False,
                boundary_unit='sampled_token', action_block_character_end=match.end(),
                boundary_token_overhang=prefix[match.end():])
    result = SimpleNamespace(token_ids=fresh[:keep], logprobs=list(out.logprobs[:keep]),
                             text=prefix, finish_reason='stop')
    return result, meta


def protocol_metrics(branches):
    environment = getattr(getattr(branches[0],'env',None),'environment',None) if branches else None
    traces=[tr for b in branches if not hasattr(b,'ucf') for tr in b.trace]
    blocks=[tr for tr in traces if first_block(tr['response'], environment) is not None]
    overhang=sum(bool(tr.get('protocol_cutoff',{}).get('boundary_token_overhang')) for tr in blocks)
    nonfirst=sum(first_block(tr['response'], environment).start()!=tr['response'].find('```') for tr in blocks)
    return dict(turns=len(traces),complete_action_blocks=len(blocks),boundary_token_extra_characters=overhang,
                boundary_token_extra_fraction=overhang/max(1,len(blocks)),first_fence_not_first_complete_block_start=nonfirst,
                first_fence_not_first_complete_block_fraction=nonfirst/max(1,len(blocks)),
                denominator='turns with a complete action block; pseudo training rows excluded')
