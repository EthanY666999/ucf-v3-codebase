"""Rebuild the medium FLH split using the hard split's sizes and ice probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from task_envs import generate_flh_map


def freeze(output):
    hard = json.loads((ROOT / 'data/flh/data.json').read_text())
    seen = {task['map_sha256'] for tasks in hard.values() for task in tasks}
    splits = {}
    maps = {}
    for split, tasks in hard.items():
        splits[split] = []
        for source in tasks:
            seed = source['seed'] + 3_000_000
            uid = f'flhm_{seed}'
            rows, meta = generate_flh_map(source['size'], source['p'], seed, min_steps=5, max_steps=9)
            if uid in maps or meta['map_sha256'] in seen:
                raise ValueError(f'Duplicate frozen task or geometry at {uid}')
            seen.add(meta['map_sha256'])
            maps[uid] = rows
            splits[split].append(dict(uid=uid, seed=seed, environment='flh', split=split,
                                      size=source['size'], p=source['p'], map_sha256=meta['map_sha256'],
                                      shortest_path_length=meta['shortest_path_length'],
                                      requires_detour=meta['requires_detour']))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('data.json', splits), ('maps.json', maps)]:
        raw = (json.dumps(data, separators=(',', ':')) + '\n').encode()
        (output / name).write_bytes(raw)
        print(name, hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
