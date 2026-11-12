import hashlib
import platform
import subprocess
from pathlib import Path
from importlib.metadata import version


def annotate_hardware(output, cfg):
    import torch
    from patr_core import atomic_json
    root = Path(__file__).resolve().parents[2]
    paths = sorted([root/'run.py'] + [p for folder in ('code','scripts','runtime') for p in (root/folder).rglob('*.py')])
    source = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rev = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    atomic_json(output / 'release.json', {'commit':rev.stdout.strip() or None, 'source_sha256':source})
    atomic_json(output / 'hardware.json', {
        'python': platform.python_version(),
        'packages': {pkg: version(pkg) for pkg in ('torch', 'vllm', 'transformers', 'gymnasium', 'numpy')},
        'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'actor_device': cfg['actor_device'],
        'reference_and_rollout_device': 'cuda:0',
    })
