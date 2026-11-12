"""User-authorized prompt contract; base-generated closing markers are recorded."""
import gzip
import hashlib
import json
import re
from pathlib import Path

EMPTY_THINK = re.compile(r'<think>\s*</think>\s*$')


def check_prompt_contract(tokenizer):
    from patr_core import chat_ids, turn_suffix, _TURN_SUFFIX
    calls = []
    class TrackedTokenizer:
        def __getattr__(self, name): return getattr(tokenizer, name)
        def apply_chat_template(self, *args, **kwargs):
            calls.append(kwargs.copy())
            return tokenizer.apply_chat_template(*args, **kwargs)
    tracked = TrackedTokenizer()
    ids = chat_ids(tracked, [{'role':'user', 'content':'Move to the goal.'}])
    first = tokenizer.decode(ids, skip_special_tokens=False)
    tail = turn_suffix(tracked, 'Next observation.')
    _TURN_SUFFIX.pop(id(tracked), None)
    assert calls and all(c.get('enable_thinking') is False for c in calls)
    assert EMPTY_THINK.search(first) and EMPTY_THINK.search(tail), 'Missing empty think block in prompt'
    return dict(passed=True, enable_thinking=False, initial_empty_think=True, subsequent_empty_think=True,
                initial_prompt_sha256=hashlib.sha256(first.encode()).hexdigest(),
                subsequent_prompt_sha256=hashlib.sha256(tail.encode()).hexdigest())
