"""Summarize training and fixed evaluations using only small JSON/JSONL metrics."""
import argparse
import csv
import json
from pathlib import Path

from run_queue import read_json, validation_files

FORK_COUNTS = ('selected_nodes', 'resamples', 'changed_resamples', 'changed_nodes',
               'reward_changed_leaves', 'leaves', 'truncated_leaves')
GROUPS = ('all_success', 'all_failure', 'mixed')
RATIOS = {
    'changed_resample_fraction': ('changed_resamples', 'resamples'),
    'changed_node_fraction': ('changed_nodes', 'selected_nodes'),
    'counterfactual_flip_rate': ('reward_changed_leaves', 'leaves'),
    'leaf_truncation_fraction': ('truncated_leaves', 'leaves'),
}
EVAL_FIELDS = ('success_rate_pct', 'pass_at_4_pct', 'webshop_mean_score', 'n_tasks', 'n_trajectories')
FIELDS = ('run', 'state', 'split', 'step', *EVAL_FIELDS, 'evaluation_tokens', 'training_tokens',
          'training_hours', *FORK_COUNTS, *RATIOS, *GROUPS)


def training_summary(path):
    """Training costs sum step_seconds (sampling + update); evaluation time is separate."""
    stats = {}
    last_step = 0
    if path.exists():
        with path.open('rb') as file:
            for line in file:
                if not line.endswith(b'\n'):
                    break  # A live writer may be partway through its last record.
                rec = json.loads(line)
                step = rec['step']
                if step <= last_step:
                    raise ValueError(f'Non-increasing training step {step} in {path}')
                last_step = step
                for source, dest in [('generated_tokens', 'training_tokens'), ('step_seconds', 'seconds')]:
                    stats[dest] = stats.get(dest, 0) + rec[source]
                for counts, fields in [('fork_counts', FORK_COUNTS), ('initial_group_counts', GROUPS)]:
                    if rec.get(counts) is not None:
                        for field in fields:
                            stats[field] = stats.get(field, 0) + rec[counts][field]
    if stats:
        stats['training_hours'] = stats.pop('seconds') / 3600
    for label, (a, b) in RATIOS.items():
        if stats.get(b):
            stats[label] = stats[a] / stats[b]
    return last_step, stats


def collect_rows(root):
    queue_file = root / 'queue_status.json'
    items = {}
    if queue_file.exists():
        state = read_json(queue_file)
        for item in state['items']:
            label = item.get('run_label') or f"{state['size']}_{state['environment']}_{item['method']}_seed{item['seed']}"
            items[label] = item['state']
    train = root / 'train'
    if train.is_dir():
        for out in sorted(train.iterdir()):
            if out.is_dir():
                items.setdefault(out.name, None)
    rows = []
    for label, queue_state in items.items():
        out = train / label
        status_file = out / 'status.json'
        status = read_json(status_file) if status_file.exists() else {}
        base = dict(run=label, state=queue_state or status.get('status', 'unknown'))
        step, stats = training_summary(out / 'train_metrics.jsonl')
        rows.append(dict(**base, split='training', step=step, **stats))
        for path in [*validation_files(out), out / 'final_test.json']:
            if not path.exists():
                continue
            data = read_json(path)
            final = path.name == 'final_test.json'
            eval_step = status.get('step', step) if final else int(path.stem.split('_')[1])
            row = dict(**base, split='test' if final else 'validation', step=eval_step,
                       evaluation_tokens=data.get('generated_tokens'))
            rows.append(dict(row, **{key: data[key] for key in EVAL_FIELDS if key in data}))
    return rows


def markdown_table(fields, rows):
    result = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                value = f'{value:.6g}'
            values.append('—' if value is None else str(value))
        result.append('| ' + ' | '.join(values) + ' |')
    return result


def write_results(root):
    rows = collect_rows(root)
    training = [row for row in rows if row['split'] == 'training']
    blocks = [
        ('Training cost', ('run', 'state', 'step', 'training_tokens', 'training_hours', *GROUPS), training),
        ('Fork statistics', ('run', *FORK_COUNTS, *RATIOS), [row for row in training if 'selected_nodes' in row]),
        ('Fixed validation and final test', ('run', 'split', 'step', *EVAL_FIELDS, 'evaluation_tokens'),
         [row for row in rows if row['split'] != 'training']),
    ]
    lines = ['# Results', '', 'Training hours sum rollout and optimization time across recorded steps. '
             'Evaluation tokens are reported per split. Fork fractions use pooled counts.']
    for title, fields, data in blocks:
        lines.extend(['', f'## {title}', ''])
        lines.extend(markdown_table(fields, data) if data else ['No records.'])
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    with (root / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    write_results(parser.parse_args().run_root.resolve())
