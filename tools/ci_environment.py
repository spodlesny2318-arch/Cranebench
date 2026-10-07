"""Check the named compatibility profile and save machine-readable provenance."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('profile', choices=['minimum','revision'])
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    expected={'minimum':((3,10),'1.24.0','1.10.0'),
              'revision':((3,14),'2.3.3','1.18.0')}[args.profile]
    versions={d.metadata['Name']:d.version for d in metadata.distributions()}
    assert sys.version_info[:2]==expected[0],sys.version
    assert metadata.version('numpy')==expected[1]
    assert metadata.version('scipy')==expected[2]
    root=Path(__file__).resolve().parents[1]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'cranebench').rglob('*.py'))}
    result={'profile':args.profile,'python':sys.version,'platform':platform.platform(),
            'executable':sys.executable,'packages':versions,'source_sha256':hashes}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(args.profile,platform.python_version(),metadata.version('numpy'),metadata.version('scipy'))

if __name__=='__main__':main()
