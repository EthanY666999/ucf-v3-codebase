"""Run the configured sequence on one device, resuming from durable run artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run import check_release

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def validation_files(out):
    return sorted(out.glob('validation_[0-9][0-9][0-9][0-9].json'))


def progress(out):
    status_path = out / 'status.json'
    status = read_json(status_path) if status_path.exists() else {}
    validations = validation_files(out)
    latest = None
    if validations:
        path = validations[-1]
        latest = {'step': int(path.stem.split('_')[1]), **read_json(path)}
    return status, {'completed_steps': status.get('step', 0), 'latest_validation': latest}


def completed(out, status, target, steps):
    if status.get('step', 0) < target:
        return False
    if status.get('status') == 'complete':
        return (out / 'final_test.json').is_file()
    return (target < steps and status.get('status') == 'paused_checkpointed'
            and (out / 'checkpoint.pt').is_file())


def prepare_run(out, config, archive_root):
    """Check launch identity before skipping or resuming. Preserve first-step failures."""
    if not out.exists():
        return False
    launch = out / 'launch.json'
    if launch.exists():
        previous = read_json(launch)
        check_release(previous, previous.get('method'))
        if previous != config:
            raise ValueError(f'Launch configuration differs at {out}; use its original configuration.')
    if not launch.exists():
        for name in ('config.json', 'launch_config.json'):
            if (out / name).is_file():
                previous = read_json(out / name)
                check_release(previous, previous.get('method'))
    if (out / 'checkpoint.pt').is_file():
        if not launch.exists():
            raise ValueError(f'Missing launch.json for checkpoint at {out}. For UCF V3 use Git tag v3-final.')
        return True
    if out.is_dir() and all(p.name == '.run.lock' for p in out.iterdir()):
        return False
    if not launch.exists():
        raise ValueError(f'Unknown files at {out} without launch.json. Choose a clean run root.')
    # run.py also takes this lock, including for launches outside the queue.
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / f'{out.name}-{time.time_ns()}'
        out.rename(archived)
        print(f'Preserved interrupted run at {archived}', flush=True)
    return False


def run_child(command, log, queue_lock):
    with log.open('wb') as output:
        # Retain the queue lock across run.py's exec if the supervisor is killed.
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                pass_fds=(queue_lock.fileno(),))
    if result.returncode:
        with log.open('rb') as output:
            output.seek(max(0, output.seek(0, os.SEEK_END) - 2048))
            tail = output.read().decode(errors='replace').strip()
        raise ValueError(f'run.py exited {result.returncode}. Log: {log}\n{tail}')


def run_queue(args):
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    methods = read_json(ROOT / 'configs/methods.json')
    for item in read_json(args.queue):
        method = item['method']
        seed = item.get('seed', 42)
        until = item.get('until', 400)
        if method not in methods or type(seed) is not int or type(until) is not int or until < 1:
            raise ValueError(f'Invalid queue entry: {item}')
        entries.append(dict(method=method, seed=seed, until=until, state='pending',
                            completed_steps=0, latest_validation=None, attempts=[]))
    if len({(e['method'], e['seed']) for e in entries}) != len(entries):
        raise ValueError('Queue entries must have distinct (method, seed) pairs')
    state = dict(size=args.size, environment=args.env, current_item=None, items=entries)
    def persist():
        temporary = root / '.queue_status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'queue_status.json')

    with (root / '.queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another queue or its active worker still holds {root / ".queue.lock"}') from error
        logs = root / 'queue_logs'
        logs.mkdir(exist_ok=True)
        for item in entries:
            method = item['method']
            seed = item['seed']
            target = item['until']
            state['current_item'] = dict(method=method, seed=seed)
            item['state'] = 'running'
            persist()
            command = [sys.executable, str(ROOT / 'run.py'), '--size', args.size, '--env', args.env,
                       '--method', method, '--seed', str(seed), '--until', str(target),
                       '--output-root', str(root), '--model-dir', str(args.model_dir.resolve())]
            out = None
            for attempt in range(1, 3):
                log = logs / f'{method}_seed{seed}-{time.time_ns()}.log'
                record = dict(number=attempt)
                try:
                    # Use the actual launcher's label, overrides and frozen data identity.
                    preview = subprocess.run([*command, '--print-config'], text=True, capture_output=True)
                    if preview.returncode:
                        raise ValueError(preview.stderr.strip() or f'Config preview exited {preview.returncode}')
                    config = json.loads(preview.stdout)
                    steps = config['steps']
                    if target > steps:
                        raise ValueError(f'Queue target {target} exceeds configured steps {steps}')
                    out = root / 'train' / config['run_label']
                    item['run_label'] = out.name
                    status, info = progress(out)
                    item.update(info)
                    if (out / 'launch.json').exists() and read_json(out / 'launch.json') == config:
                        if completed(out, status, target, steps):
                            item['state'] = 'complete'
                            break
                    resume = prepare_run(out, config, root / 'interrupted')
                    record.update(log=str(log), resume=resume)
                    item['attempts'].append(record)
                    persist()
                    print(f'{method} seed={seed} until={target}, attempt {attempt}, log {log}', flush=True)
                    run_child(command + (['--resume'] if resume else []), log, lock)
                    status, info = progress(out)
                    item.update(info)
                    if not completed(out, status, target, steps):
                        raise ValueError(f'Child exited before completing target {target}: {status.get("status")}')
                    item['state'] = 'complete'
                    break
                except (ValueError, OSError, KeyError) as exc:
                    if record not in item['attempts']:
                        item['attempts'].append(record)
                    record['error'] = str(exc)
                    if out is not None:
                        try:
                            _, info = progress(out)
                            item.update(info)
                        except (ValueError, OSError):
                            pass
                    if attempt == 2:
                        item['state'] = 'failed'
                    print(f'{method}, attempt {attempt} failed: {exc}', file=sys.stderr, flush=True)
                    persist()
            persist()
        state['current_item'] = None
        persist()
    return int(any(entry['state'] == 'failed' for entry in entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', required=True, choices=['2b', '4b'])
    parser.add_argument('--env', required=True, choices=['sokoban', 'flh', 'webshop'])
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--queue', type=Path, default=ROOT / 'configs/queue.json')
    return run_queue(parser.parse_args())


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyError, ValueError, OSError) as error:
        sys.exit(str(error))
