def fence_id(tokenizer):
    ids=tokenizer.encode('```',add_special_tokens=False)
    assert len(ids)==1, ('Fence must be one token',ids)
    return ids[0]
