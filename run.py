"""Compose a complete, relocatable UCF V3 experiment and launch it locally."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANCHOR_SHA = 'a4a4ccaae75a3f9056d17b924d4184facaed7b4ae677709ddb04a0daf1338c0b'
CONFIGS = ROOT / 'configs'


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def resolve(args, preview=False):
    path = CONFIGS / args.size / (args.env + '.json')
    if not path.is_file():
        raise ValueError(f'No {args.size}/{args.env} experiment in this release')
    cfg = {**read(CONFIGS/'common.json'), **read(path), **read(CONFIGS/'methods.json')[args.method]}
    if args.method == 'token-partition' and args.env == 'webshop':
        raise ValueError('WebShop already uses an action-family partition; this ablation is grid-only')
    model = read(CONFIGS/'models.json')[args.size]
    cfg.update(model=str(args.model_dir.resolve()) if args.model_dir else model['hf_id'], model_revision=model['revision'],
               seed=args.seed, run_label=f'{args.size}_{args.env}_{args.method}_seed{args.seed}')
    if args.flh_difficulty and args.env != 'flh':
        raise ValueError('--flh-difficulty applies only to --env flh')
    if args.env == 'flh':
        cfg['flh_difficulty'] = args.flh_difficulty or 'hard'
        if cfg['flh_difficulty'] == 'medium':
            hashes = read(ROOT/'data/SHA256.json')
            dataset = 'data/flh/medium/data.json'
            maps = 'data/flh/medium/maps.json'
            cfg.update(dataset_path=dataset, dataset_sha256=hashes[dataset],
                       frozen_maps_path=maps, frozen_maps_sha256=hashes[maps],
                       run_label=f'{args.size}_flh_medium_{args.method}_seed{args.seed}')
    if args.token_budget is not None:
        if not 0 < args.token_budget < cfg['max_context_tokens']:
            raise ValueError('Whole-trajectory token budget must be positive and smaller than the context window')
        cfg['max_response_tokens'] = args.token_budget
    if args.actor_device:
        cfg['actor_device'] = args.actor_device
    for arg in ('micro_batch_token_limit','micro_batch_size','vllm_max_sequences','vllm_memory_fraction'):
        value = getattr(args,arg)
        if value is not None:
            if value <= 0 or (arg == 'vllm_memory_fraction' and value > 1):
                raise ValueError('Invalid engineering setting: '+arg)
            cfg[arg] = value
    if args.env == 'webshop':
        cfg['webshop_url'] = args.webshop_url
        data_dir = args.webshop_data.resolve()
        manifest = data_dir/'manifest.json'
        if manifest.is_file():
            m = read(manifest)
            cfg.update(dataset_path=str(data_dir/'data.json'), dataset_sha256=m['dataset_sha256'], webshop_catalog=m['catalog'])
        elif preview:
            cfg.update(dataset_path=str(data_dir/'data.json'), dataset_sha256=None, webshop_catalog=None)
        else:
            raise ValueError('Freeze the live WebShop split with scripts/prepare_webshop.py first')
    else:
        for key in ('dataset_path', 'frozen_maps_path'):
            if key in cfg:
                cfg[key] = str(ROOT / cfg[key])
    if args.anchor16:
        if (args.size, args.env, args.method, args.seed, cfg['max_response_tokens']) not in [
            ('2b', 'sokoban', method, 42, 10240) for method in ('grpo4', 'ucf')]:
            raise ValueError('The historical anchor belongs to the 2B Sokoban seed42 matched pair')
        cfg['anchor16'] = str(args.anchor16.resolve())
        cfg['anchor16_sha256'] = ANCHOR_SHA
    return cfg


def check_inputs(cfg, size, model_dir):
    for path_key, hash_key in [('dataset_path','dataset_sha256'),('frozen_maps_path','frozen_maps_sha256')]:
        if path_key in cfg and file_hash(Path(cfg[path_key])) != cfg[hash_key]:
            raise ValueError(f'Frozen input checksum differs: {cfg[path_key]}')
    if model_dir is None:
        raise ValueError('Provide --model-dir or MODEL_DIR pointing to the official local model')
    metadata = read(CONFIGS/'models.json')[size]
    for name, key in [('config.json','config_sha256'),('tokenizer.json','tokenizer_sha256'),('tokenizer_config.json','tokenizer_config_sha256'),('chat_template.jinja','template_sha256')]:
        if file_hash(model_dir/name) != metadata[key]:
            raise ValueError(f'{name} differs from the pinned {size} revision')
    if cfg['environment']=='webshop':
        from urllib.request import urlopen
        with urlopen(cfg['webshop_url'].rstrip('/')+'/health', timeout=30) as r:
            if json.load(r)['catalog'] != cfg['webshop_catalog']:
                raise ValueError('WebShop service and frozen goal IDs belong to different catalogs')


def check_weights(model_dir):
    index = model_dir/'model.safetensors.index.json'
    files = set(read(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not files or any(not (model_dir/f).is_file() or (model_dir/f).stat().st_size == 0 for f in files):
        raise ValueError('Missing model weight shards in --model-dir')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=['2b','4b'], required=True)
    p.add_argument('--env', choices=['sokoban','flh','webshop'], required=True)
    p.add_argument('--flh-difficulty', choices=['hard','medium'], help='FrozenLake map set (default: hard)')
    p.add_argument('--method', choices=read(CONFIGS/'methods.json'), required=True)
    p.add_argument('--model-dir', type=Path, default=os.environ.get('MODEL_DIR'))
    p.add_argument('--anchor16', type=Path, help='Optional original shared step-16 checkpoint; default is a fresh base')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--token-budget', type=int, help='Assistant tokens over the entire trajectory')
    p.add_argument('--output-root', type=Path, default=Path(os.environ.get('RUN_ROOT', ROOT/'runs')))
    p.add_argument('--webshop-data', type=Path, default=ROOT/'artifacts/webshop')
    p.add_argument('--webshop-url', default=os.environ.get('WEBSHOP_URL','http://127.0.0.1:8765'))
    p.add_argument('--actor-device', choices=['cuda:0','cuda:1'])
    p.add_argument('--micro-batch-size', type=int)
    p.add_argument('--micro-batch-token-limit', type=int)
    p.add_argument('--vllm-max-sequences', type=int)
    p.add_argument('--vllm-memory-fraction', type=float)
    p.add_argument('--until', type=int, help='Stop with a checkpoint after this absolute training step')
    p.add_argument('--resume', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--print-config', action='store_true')
    group.add_argument('--check', action='store_true', help='Check data and tokenizer metadata on CPU')
    group.add_argument('--smoke', action='store_true', help='One real GPU update and validation')
    group.add_argument('--validate-base', action='store_true',
                       help='Evaluate the original model on the fixed validation split and exit at step 0')
    args = p.parse_args()
    if args.validate_base and (args.resume or args.anchor16 or args.until is not None):
        p.error('--validate-base evaluates fresh weights without a training step or checkpoint')
    cfg = resolve(args, preview=args.print_config)
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return
    check_inputs(cfg, args.size, args.model_dir)
    if args.check:
        print(f'Checked {args.size}/{args.env}/{args.method} inputs and model metadata')
        return
    if args.until is not None and not 1 <= args.until <= cfg['steps']:
        raise ValueError('--until must be between 1 and configured steps')
    if args.anchor16 and (args.smoke or (args.until is not None and args.until <= 16)):
        raise ValueError('Use --anchor16 only for updates after step 16')
    if args.anchor16 and not args.resume and file_hash(args.anchor16) != ANCHOR_SHA:
        raise ValueError('Historical step-16 checkpoint SHA256 mismatch')
    check_weights(args.model_dir)
    for pkg, target in [('torch','2.10.0'),('vllm','0.19.1'),('transformers','5.22.2')]:
        found = version(pkg).split('+')[0]
        if found != target:
            raise ValueError(f'Requires {pkg} {target}, found {found}')
    import torch
    index = int(cfg['actor_device'].split(':')[-1])
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        raise ValueError(f"Selected layout requires at least {index+1} visible CUDA devices")
    phase = 'base-validation' if args.validate_base else ('smoke' if args.smoke else 'train')
    out = args.output_root.resolve()/phase/cfg['run_label']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = out/'launch.json'
        if args.resume:
            if not (out/'checkpoint.pt').is_file() or not frozen.exists():
                raise ValueError('This run has no resumable checkpoint')
            if read(frozen) != cfg:
                raise ValueError('Resolve the same config used to produce this checkpoint')
        elif any(f.name != '.run.lock' for f in out.iterdir()):
            raise ValueError(f'Existing output at {out}; resume it or choose a new --output-root')
        else:
            frozen.write_text(json.dumps(cfg, indent=2)+'\n')
        command = [sys.executable, str(ROOT/'runtime/train_scoped.py'), '--config', str(frozen),
                   '--method', cfg['method'], '--output-dir', str(out)]
        if args.resume: command.append('--resume')
        if args.smoke: command.append('--smoke')
        if args.validate_base: command.append('--validate-base')
        if args.until: command.extend(['--stop-after-step', str(args.until)])
        env = dict(os.environ, VLLM_BATCH_INVARIANT='1')
        # exec preserves scheduler signals and keeps the run lock open in the trainer.
        os.set_inheritable(lock.fileno(), True)
        os.execve(sys.executable, command, env)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
