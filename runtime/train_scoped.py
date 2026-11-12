"""Apply the release inference kernels and restore native PyTorch for training."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

os.environ['VLLM_BATCH_INVARIANT'] = '1'
root = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(root))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    a, _ = p.parse_known_args()
    if sys.flags.optimize:
        raise RuntimeError('Run without python -O to retain numerical and token-boundary checks')
    import train
    import torch
    import vllm
    from scoped_inference import configure_train
    scope = configure_train(train, torch, vllm)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train.atomic_json(a.output_dir / 'numerical_runtime.json', dict(
        protocol='vllm0191_batch_invariant_inference_native_training',
        torch=torch.__version__, vllm=vllm.__version__,
        attention_backend='FLASH_ATTN', installed_files_modified=False,
        helper_sha256=hashlib.sha256(Path(__file__).with_name('scoped_inference.py').read_bytes()).hexdigest(),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        train.main()
    finally:
        scope.training()
        train.atomic_json(a.output_dir / 'numerical_transitions.json', dict(
            transitions=scope.transitions, native_training_restored=True))


if __name__ == '__main__':
    main()
