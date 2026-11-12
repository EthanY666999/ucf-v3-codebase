"""Skip impossible ASCII-fence completion tokens, without decoding a suffix.

For ByteLevel decoding, ASCII backticks are preserved in the vocabulary piece.
We still decode the complete sampled prefix at every possible completion. Other
decoders/cleanup policies retain the original exhaustive scan.
"""
import json


def fence_token_positions(ids,start,end,tokenizer):
    cache='_ucf_bytelevel_backtick_ids'
    if not hasattr(tokenizer,cache):
        eligible=None
        backend=getattr(tokenizer,'backend_tokenizer',None)
        if backend is not None and not getattr(tokenizer,'clean_up_tokenization_spaces',True):
            decoder=json.loads(backend.decoder.__getstate__())
            if decoder.get('type')=='ByteLevel':
                eligible=frozenset(i for piece,i in tokenizer.get_vocab().items() if '`' in piece)
        setattr(tokenizer,cache,eligible)
    eligible=getattr(tokenizer,cache)
    for pos in range(start,end):
        if eligible is None or ids[pos] in eligible:
            yield pos
