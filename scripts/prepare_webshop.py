"""Freeze the ordered upstream human-goal IDs for the matched WebShop arms."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


def prepare(health, output):
    n = health['n_goals']
    catalog = health['catalog']
    spec = json.loads((Path(__file__).resolve().parents[1]/'data/webshop/split.json').read_text())
    if n < spec['train_start'] + 64 or len(catalog) != 64:
        raise ValueError('Use the full human-goal dataset (at least 64 training goals after ID 1499)')
    boundaries = {'train': range(spec['train_start'], n), 'validation': range(*spec['validation']), 'test': range(*spec['test'])}
    data = {split: [{'uid': f'webshop_{i}', 'environment': 'webshop', 'goal_id': i, 'split': split}
                    for i in ids] for split, ids in boundaries.items()}
    raw = json.dumps(data, separators=(',', ':')).encode()
    manifest = {**health, 'catalog': catalog, 'n_goals': n, 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
                'sizes': {k: len(v) for k, v in data.items()}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Immutable files: concurrent identical preparations converge, another catalog is rejected.
    for name, content in [('data.json', raw), ('manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())]:
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as tmp:
            tmp.write(content)
        temp = Path(tmp.name)
        try:
            try:
                os.link(temp, output/name)
            except FileExistsError:
                if (output/name).read_bytes() != content:
                    raise ValueError(f'Different WebShop corpus at {output}; choose a fresh directory')
        finally:
            temp.unlink()
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--out', type=Path, default=Path('artifacts/webshop'))
    args = p.parse_args()
    with urlopen(args.url.rstrip('/')+'/health', timeout=30) as response:
        print(json.dumps(prepare(json.load(response), args.out), indent=2))
