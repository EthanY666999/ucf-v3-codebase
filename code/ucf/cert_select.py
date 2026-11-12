import hashlib,json
def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(list(parts),sort_keys=True).encode()).digest()[:8],'big')%(2**31-1)
