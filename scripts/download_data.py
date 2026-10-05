"""Download exactly the pinned dataset inputs and verify their content hashes."""
from pathlib import Path
import json
from huggingface_hub import hf_hub_download
from kuailive_exit.config import load_config
from kuailive_exit.provenance import sha256


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'configs/raw_sha256.json').read_text())
    raw = cfg['data']['raw_dir']
    for name, digest in expected['files'].items():
        p = raw / name
        if not p.exists():
            hf_hub_download(repo_id=cfg['data']['repo_id'], repo_type='dataset',
                            revision=cfg['data']['revision'], filename=name, local_dir=raw)
        if sha256(p) != digest:
            raise ValueError(f'Raw input checksum mismatch: {name}')
        print(f'verified {name}', flush=True)


if __name__ == '__main__':
    main()
