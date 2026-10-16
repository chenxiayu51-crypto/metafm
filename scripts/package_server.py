#!/usr/bin/env python3
"""Package committed source only; never include raw data or local outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='external directory for ZIP, tar.gz, hashes')
    parser.add_argument('--name', default='metafmodel-server-20261016')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('release output must be outside source checkout')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('name must be one filename component')
    status = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).strip()
    if status:
        raise ValueError(f'commit tracked source changes before packaging: {status}')
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    contents = {}
    for name in sorted(filter(None, names)):
        parts = Path(name).parts
        if name.startswith('docs/superpowers/'):
            continue
        if any(x in parts for x in ('.git', '.venv', 'reports', '__pycache__')) or Path(name).suffix.lower() in ('.h5', '.hdf5', '.ckpt', '.pt', '.zip', '.gz'):
            raise ValueError(f'non-source tracked file: {name}')
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'unsafe source link: {name}')
        data = path.read_bytes()
        contents[name] = data.replace(b'\r\n', b'\n') if name.endswith('.sh') else data
    release = dict(schema='metafmodel-source-release-v1', name=args.name,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   files={name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
                   note='Committed source snapshot; shell scripts normalized to LF. No datasets, weights, environment, reports or Git history.')
    contents['SOURCE_RELEASE.json'] = (json.dumps(release, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    paths = [output / (args.name + suffix) for suffix in ('.zip', '.tar.gz', '-SHA256SUMS.txt')]
    if any(path.exists() for path in paths):
        raise FileExistsError('release already exists; choose a new name')
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths[0], 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            entry = zipfile.ZipInfo('metafmodel/' + name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            archive.writestr(entry, data)
    with tarfile.open(paths[1], 'w:gz') as archive:
        for name, data in contents.items():
            entry = tarfile.TarInfo('metafmodel/' + name)
            entry.size = len(data)
            entry.mode = 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(entry, io.BytesIO(data))
    paths[2].write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in paths[:2]), encoding='ascii')
    print(json.dumps(dict(archives=[str(p) for p in paths], source_files=len(release['files']), base_commit=release['base_commit']), indent=2))


if __name__ == '__main__':
    main()
