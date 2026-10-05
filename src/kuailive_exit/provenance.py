"""Content-addressed inputs, executable source and runtime provenance."""
from pathlib import Path
import hashlib
import importlib.metadata
import platform
import json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_manifest(root):
    root = Path(root)
    paths = sorted([*root.glob('src/**/*.py'), *root.glob('scripts/*.py'),
                    *root.glob('configs/*'), root / 'pyproject.toml', root / 'requirements.lock', root / 'Makefile'])
    files = {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}
    return {'files': files, 'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def runtime_manifest():
    names = ['numpy', 'pandas', 'pyarrow', 'polars', 'lightgbm', 'scikit-learn',
             'joblib', 'PyYAML', 'matplotlib', 'huggingface-hub']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {n: importlib.metadata.version(n) for n in names}}
