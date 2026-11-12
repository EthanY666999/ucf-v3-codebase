"""Restore UCF side effects to the optimizer checkpoint, preserving later evidence."""
import hashlib
import json
import re
import time
from pathlib import Path

STEP_FILE = re.compile(r"(?:controller|select|budget|raw_leaves)_(\d+)\.(?:json|jsonl|jsonl\.gz)$")

def reconcile_ucf(log_dir, checkpoint_step):
    log_dir = Path(log_dir)
    assert isinstance(checkpoint_step, int) and checkpoint_step >= 0
    snapshot = log_dir / f"controller_{checkpoint_step:04d}.json"
    if checkpoint_step:
        # Validate before moving anything. Never invent missing controller history.
        assert snapshot.exists(), "Missing checkpoint-aligned UCF controller snapshot"
        target = snapshot.read_bytes()
        assert json.loads(target)["last_step"] == checkpoint_step
    else:
        target = None
    current = log_dir / "controller.json"
    later = [p for p in sorted(log_dir.glob("*")) if p.is_file()
             and (m := STEP_FILE.fullmatch(p.name)) and int(m.group(1)) > checkpoint_step]
    needs_controller = (current.read_bytes() if current.exists() else None) != target
    if not later and not needs_controller:
        return dict(checkpoint_step=checkpoint_step, changed=False)
    archive = log_dir / "recovery_archive" / str(time.time_ns())
    archive.mkdir(parents=True)
    files = later + ([current] if current.exists() and needs_controller else [])
    evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    receipt = dict(checkpoint_step=checkpoint_step, changed=True,
                   archived_sha256=evidence, restored_snapshot=str(snapshot) if target else None)
    (archive / "receipt.json").write_text(json.dumps(dict(receipt, status="planned"), indent=2)+"\n")
    for p in files:
        p.rename(archive / p.name)
    if target is not None and needs_controller:
        temporary = current.with_suffix(".restore.tmp")
        temporary.write_bytes(target)
        temporary.replace(current)
    (archive / "receipt.json").write_text(json.dumps(dict(receipt, status="complete"), indent=2)+"\n")
    return dict(receipt, archive=str(archive))
