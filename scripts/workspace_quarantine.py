"""Reversible build-directory quarantine. Never deletes files or overwrites.

This tool accepts only legacy_root/build, refuses data/checkpoints and links,
and writes exact original/destination SHA256 mappings before moving anything.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def _verify_ancestors(path):
    path=Path(os.path.abspath(os.fspath(path)))
    for parent in (path,*path.parents):
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024)):
            raise ValueError(f'path redirected by link/reparse point: {parent}')
    if path.resolve()!=path:
        raise ValueError('path redirected from saved absolute location')
    return path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def _plain_tree(source):
    paths=[source,*source.rglob('*')]
    for path in paths:
        if path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024):
            raise ValueError('protected: links/reparse points cannot be quarantined')
        if path.is_file() and path.suffix.lower() not in {'.py','.pyc','.txt','.md','.toml','.cfg','.ini','.json'}:
            raise ValueError(f'protected or unclassified file: {path}')
    return [path for path in paths if path.is_file()]


def quarantine_build(legacy_root,destination,dry_run=False):
    legacy=_verify_ancestors(legacy_root);source=legacy/'build';target=_verify_ancestors(destination)
    if not source.is_dir():raise FileNotFoundError(source)
    source.resolve().relative_to(legacy)
    if target==source or target.is_relative_to(source):raise ValueError('destination lies in source')
    if target.exists():raise FileExistsError(target)
    files=_plain_tree(source)
    mapping=dict(source=str(source),destination=str(target/'build'),state='planned',
                 files=[dict(original=str(path),quarantined=str(target/'build'/path.relative_to(source)),
                             sha256=digest(path),size=path.stat().st_size) for path in files],released_bytes=0)
    if dry_run:return mapping
    target.mkdir(parents=True)
    map_path=target/'path_mapping.json'
    map_path.write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    _verify_ancestors(source);_verify_ancestors(target)
    source.rename(target/'build')
    for row in mapping['files']:
        if digest(row['quarantined'])!=row['sha256']:raise RuntimeError('quarantine hash mismatch; retain mapping and files')
    mapping['state']='quarantined';map_path.write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    return map_path


def restore(mapping_path):
    mapping_path=_verify_ancestors(mapping_path);mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    original=Path(mapping['source']);target=Path(mapping['destination'])
    if original.name!='build' or target!=mapping_path.parent/'build':raise ValueError('invalid mapping roots')
    if not original.is_absolute() or not target.is_absolute():raise ValueError('invalid relative mapping')
    _verify_ancestors(original);_verify_ancestors(target)
    if original.exists():raise FileExistsError(original)
    files=_plain_tree(target)
    expected={Path(row['quarantined']) for row in mapping['files']}
    if set(files)!=expected:raise ValueError('quarantine contents changed; refusing restore')
    for row in mapping['files']:
        a,b=Path(row['original']),Path(row['quarantined'])
        if not a.is_relative_to(original) or not b.is_relative_to(target) or a.relative_to(original)!=b.relative_to(target):
            raise ValueError('mapping escapes root')
        if digest(b)!=row['sha256']:raise ValueError('quarantine hash changed')
    _verify_ancestors(original);_verify_ancestors(target)
    target.rename(original)
    mapping['state']='restored';mapping_path.write_text(json.dumps(mapping,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root');parser.add_argument('--destination')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--restore')
    args=parser.parse_args()
    if args.restore:restore(args.restore);print('Restored verified files without overwriting.')
    elif args.legacy_root and args.destination:
        print(quarantine_build(args.legacy_root,args.destination,args.dry_run))
    else:parser.error('provide --restore or --legacy-root and --destination')
