"""Install a built artifact in a clean environment and exercise the README API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    root = Path(__file__).resolve().parents[1]
    expected = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / 'cranebench').rglob('*.py')}
    example = (root / 'README.md').read_text(encoding='utf-8').split('```python\n', 1)[1].split('```', 1)[0]
    smoke = '''import hashlib, importlib.metadata, json, pathlib
import cranebench
package = pathlib.Path(cranebench.__file__).parent
assert 'site-packages' in package.parts, package
expected = json.loads(pathlib.Path('expected.json').read_text())
actual = {'cranebench/' + p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')}
assert actual == expected, 'Installed source differs from candidate'
assert cranebench.__version__ == importlib.metadata.version('cranebench')
exec(pathlib.Path('example.py').read_text())
import numpy as np
assert metrics is not None and np.isfinite(X).all() and np.isfinite(U).all()
assert np.isfinite(metrics['peak_swing']) and np.isfinite(metrics['residual_swing'])
assert abs(float(t[-1]) - 40.0) < .01
print(json.dumps({'installed_version': cranebench.__version__, 'source_files_checked': len(actual), 'readme_example_passed': True}))
'''
    logs = []
    def run(command, cwd):
        result = subprocess.run([str(a) for a in command], cwd=cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf-8', errors='replace')
        logs.append(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout)
        return result.stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='cranebench-package-') as temp:
            work = Path(temp)
            run([sys.executable, '-m', 'venv', work / 'env'], work)
            python = work / 'env' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
            run([python, '-m', 'pip', 'install', '-c', root / 'ci/minimum.txt', artifact], work)
            run([python, '-m', 'pip', 'check'], work)
            (work / 'expected.json').write_text(json.dumps(expected), encoding='utf-8')
            (work / 'example.py').write_text(example, encoding='utf-8')
            stdout = run([python, '-I', '-c', smoke], work)
            report = json.loads(stdout.strip().splitlines()[-1])
            report.update(artifact=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report))
    finally:
        args.output.with_suffix('.log').write_text('\n'.join(logs), encoding='utf-8')


if __name__ == '__main__':
    main()
