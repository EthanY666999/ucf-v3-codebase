"""Preserve training records ahead of the optimizer state before replaying steps."""
import json
import time
from pathlib import Path


def archive_uncommitted(out, step):
    out = Path(out)
    archive = out/'uncommitted'/str(time.time_ns())
    for name in ('train_metrics.jsonl', 'branch_events.jsonl'):
        path = out/name
        if not path.exists():
            continue
        # Stream both files. Branch-level records can be much larger than train metrics.
        temp = path.with_suffix('.recovery.tmp')
        changed = False
        with path.open('rb') as src, temp.open('wb') as dst:
            for line in src:
                try:
                    row = json.loads(line)
                    keep = row['step'] <= step
                except (ValueError, KeyError):
                    keep = False
                if keep:
                    dst.write(line)
                else:
                    changed = True
        if changed:
            archive.mkdir(parents=True, exist_ok=True)
            import os
            os.link(path, archive/name)
            temp.replace(path)
        else:
            temp.unlink()
    for folder, patterns in (
        (out, ('rollouts_*.jsonl.gz', 'validation_*.json', 'validation_*.jsonl')),
        (out/'ucf_logs', ('candidates_*.jsonl', 'select_*.jsonl', 'budget_*.json', 'raw_leaves_*.jsonl.gz')),
    ):
        for pattern in patterns:
            for path in folder.glob(pattern):
                number = int(path.name.rsplit('_', 1)[1].split('.')[0])
                if number > step:
                    destination = archive / path.relative_to(out)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(destination)
