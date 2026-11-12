"""Complete a checkpoint-step fixed validation before resuming training."""
import hashlib
import json
import time
from pathlib import Path

def validation_complete(path, tasks):
    path=Path(path)
    try:
        summary=json.loads(path.read_text())
        rows=[json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        expected={task['uid']:task for task in tasks}
        groups={}
        for row in rows:
            uid=row['task']['uid']
            if uid not in expected or row['task']!=expected[uid] or row['reward'] not in [0,1]:
                return False
            groups.setdefault(uid,[]).append(row)
        if set(groups)!=set(expected) or not groups:
            return False
        if not all(len(g)==4 and len({r['lineage'] for r in g})==4 for g in groups.values()):
            return False
        sr=100*sum(r['reward'] for r in rows)/len(rows)
        pa=100*sum(any(r['reward'] for r in g) for g in groups.values())/len(groups)
        return (summary['n_tasks']==len(tasks) and summary['n_trajectories']==len(rows)
                and abs(summary['success_rate_pct']-sr)<1e-9
                and abs(summary['pass_at_4_pct']-pa)<1e-9)
    except (OSError,ValueError,KeyError,TypeError):
        return False

def complete_checkpoint_validation(out, step, cfg, tasks, evaluate):
    # Never evaluate an earlier checkpoint with newer weights.
    if step<=0 or (step%cfg['eval_every']!=0 and step!=cfg['steps']):
        return False
    path=Path(out)/f'validation_{step:04d}.json'
    if validation_complete(path,tasks):
        return False
    files=[p for p in [path,path.with_suffix('.jsonl')] if p.exists()]
    if files:
        archive=Path(out)/'validation_recovery_archive'/str(time.time_ns())
        archive.mkdir(parents=True)
        evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (archive/'receipt.json').write_text(json.dumps(dict(step=step,sha256=evidence,reason='incomplete checkpoint-step validation'),indent=2)+'\n')
        for p in files:p.rename(archive/p.name)
    evaluate(path)
    assert validation_complete(path,tasks), 'Recovered fixed validation is incomplete'
    return True
