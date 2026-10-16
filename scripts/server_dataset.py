#!/usr/bin/env python3
"""Inspect official raw releases, resume canonical conversion, launch native E3."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import h5py
import numpy as np

from prepare_data import CONVERTERS, NATURE_DATASETS, _converter_module, build_splits, write_rows
from semg_fm.data.raw_signal import field_names, read_field, validate_time

OFFICIAL = {'train', 'val', 'test'}
TASKS = {'emg2pose': 'pose', 'emg2qwerty': 'typing',
         'gni_discrete_gestures': 'discrete_gesture', 'gni_handwriting': 'handwriting',
         'gni_wrist': 'wrist_angle'}


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _basename(value):
    return str(value).replace('\\', '/').rsplit('/', 1)[-1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _paths(config):
    roots = {key: Path(config['raw_roots'][key]).expanduser().resolve() for key in CONVERTERS}
    canonical = Path(config['canonical_root']).expanduser().resolve()
    runs = Path(config['run_root']).expanduser().resolve()
    pairs = [('canonical', canonical), ('run', runs)]
    for name, output in pairs:
        for raw in [*roots.values(), REPO]:
            if output == raw or output.is_relative_to(raw) or raw.is_relative_to(output):
                raise ValueError(f'{name} output overlaps/inside raw or source tree: {output} / {raw}')
    if canonical == runs or canonical.is_relative_to(runs) or runs.is_relative_to(canonical):
        raise ValueError('canonical and run roots overlap')
    for i, root in enumerate(roots.values()):
        for other in list(roots.values())[i + 1:]:
            if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                raise ValueError(f'raw corpus roots overlap: {root} / {other}')
    return roots, canonical, runs


def _tail(node, field):
    return node[field].shape[1:] if isinstance(node, h5py.Group) else node.dtype.fields[field][0].shape


def inspect_file(dataset, path):
    """Header plus first/last 4096 samples; full signal stays on disk."""
    layout = 'data' if dataset in NATURE_DATASETS else f'{dataset}/timeseries'
    with h5py.File(path, 'r') as handle:
        node = handle[layout]
        fields = field_names(node)
        n = len(node['time']) if isinstance(node, h5py.Group) else len(node)
        if not n or 'time' not in fields:
            raise ValueError('empty recording or missing time')
        signals = ['emg_left', 'emg_right'] if dataset == 'emg2qwerty' else ['emg']
        if dataset in NATURE_DATASETS:
            expected = {'gni_discrete_gestures': 'discrete_gestures',
                        'gni_handwriting': 'handwriting', 'gni_wrist': 'wrist'}[dataset]
            if not isinstance(node, h5py.Dataset) or _text(node.attrs.get('task', '')) != expected:
                raise ValueError(f'expected documented data dataset with task={expected}')
            candidates = [field for field in fields if _tail(node, field) == (16,)]
            signals = ['emg'] if 'emg' in candidates else candidates
            if len(signals) != 1:
                raise ValueError('ambiguous/missing 16 channel EMG')
        for field in signals:
            if field not in fields or _tail(node, field) != (16,):
                raise ValueError(f'{field}: expected (N,16)')
        targets = []
        if dataset == 'emg2pose':
            if 'joint_angles' not in fields or _tail(node, 'joint_angles') != (20,):
                raise ValueError('joint_angles: expected (N,20)')
            targets = ['joint_angles']
        if dataset == 'gni_wrist':
            targets = [x for x in fields if x not in ['time', *signals]
                       and np.issubdtype(node.dtype.fields[x][0].base, np.number)]
            if not targets:
                raise ValueError('missing continuous wrist targets')
        windows = sorted({(0, min(n, 4096)), (max(0, n - 4096), n)})
        for start, end in windows:
            times = np.asarray(read_field(node, 'time', start, end))
            if times.ndim != 1 or not np.isfinite(times).all():
                raise ValueError('invalid time field')
            if len(times) > 1:
                validate_time(times, 2000, packetized=dataset in NATURE_DATASETS)
            for field in [*signals, *targets]:
                if not np.isfinite(read_field(node, field, start, end)).all():
                    raise ValueError(f'nonfinite {field} in sampled windows')
        origin = float(read_field(node, 'time', 0, 1)[0])
        end = float(read_field(node, 'time', n - 1, n)[0]) + 1 / 2000
        if end <= origin:
            raise ValueError('invalid source time bounds')
        declarations = {_text(owner.attrs[key]) for owner in (handle, node.parent, node)
                        for key in ('split', 'official_split') if key in owner.attrs}
        declarations.discard('unknown')
        if len(declarations) > 1 or declarations - OFFICIAL:
            raise ValueError(f'invalid/conflicting source split: {declarations}')
        result = dict(path=str(path), samples=n, fields=fields, source_start=origin,
                      source_end=end, source_split=next(iter(declarations), 'unknown'),
                      uncompressed_bytes=int(n * node.dtype.itemsize) if isinstance(node, h5py.Dataset)
                      else sum(int(x.size * x.dtype.itemsize) for x in node.values()),
                      size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    if TASKS[dataset] in ('typing', 'discrete_gesture', 'handwriting'):
        from semg_fm.data.raw_adapter import _source_labels
        labels, _ = _source_labels(str(path), TASKS[dataset], result['mtime_ns'], result['size'])
        result['label_counts'] = {key: len(value) for key, value in labels.items()}
    return result


def _csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _split_lookup(paths):
    lookup = {}
    for path in paths:
        columns, rows = _csv(path)
        key = next((x for x in ('path', 'source_file', 'file', 'filename', 'session') if x in columns), None)
        if not key or 'split' not in columns:
            raise ValueError(f'official split CSV lacks filename/session and split: {path}')
        for row in rows:
            name = str(row[key]).replace('\\', '/').strip()
            split = row['split'].strip()
            if split not in OFFICIAL:
                raise ValueError(f'invalid split {split!r}: {path}')
            if name in lookup and lookup[name] != split:
                raise ValueError(f'conflicting split for {name}')
            lookup[name] = split
    return lookup


def _qwerty_yaml(path):
    import yaml
    with Path(path).open(encoding='utf-8') as stream:
        content = yaml.safe_load(stream)
    lookup = {}
    for split, rows in content['dataset'].items():
        if split not in OFFICIAL:
            raise ValueError(f'invalid official YAML split {split}')
        for row in rows:
            name = str(row['session'])
            if name in lookup and lookup[name] != split:
                raise ValueError(f'conflicting YAML session split {name}')
            lookup[name] = split
    return lookup


def inspect(config):
    roots, canonical, runs = _paths(config)
    report = dict(schema='metafmodel-server-inspection-v1', datasets={}, protocol_errors=[],
                  format_errors=[], nature_intervals=[], metadata_sources=[],
                  scope='All HDF5 headers, first/last 4096 samples and task labels; full conversion validates records.',
                  config=config)
    corpus_paths = {Path(p).expanduser().resolve() for p in config.get('nature_corpus_csvs', [])}
    all_nature = {}
    coverage = {key: Counter() for key in CONVERTERS}
    for dataset, root in roots.items():
        files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.h5', '.hdf5')) if root.is_dir() else []
        current = report['datasets'][dataset] = dict(root=str(root), hdf5_files=len(files), records_per_file=2 if dataset == 'emg2qwerty' else 1, files=[], excluded_sources=[])
        if not files:
            report['format_errors'].append(f'{dataset}: no HDF5 files under {root}; extract the official archive first')
        csvs = sorted(root.rglob('*.csv')) if root.is_dir() else []
        split_csvs = []
        for path in csvs:
            try:
                columns, _ = _csv(path)
                if dataset in NATURE_DATASETS and {'dataset', 'split', 'start', 'end'} <= set(columns):
                    corpus_paths.add(path.resolve())
                elif dataset not in NATURE_DATASETS and 'split' in columns and set(columns) & {'path', 'source_file', 'file', 'filename', 'session'}:
                    split_csvs.append(path)
            except (OSError, UnicodeError, csv.Error) as exc:
                report['protocol_errors'].append(f'{path}: {exc}')
        explicit = config.get('official_split_csvs', {}).get(dataset)
        if explicit:
            split_csvs = [Path(explicit)] if isinstance(explicit, str) else list(map(Path, explicit))
        lookup = {}
        yaml_lookup = None
        try:
            lookup = _split_lookup(split_csvs)
            for path in split_csvs:
                report['metadata_sources'].append(dict(path=str(path), sha256=sha256(path)))
            yaml_path = config.get('qwerty_generic_yaml') if dataset == 'emg2qwerty' else None
            if yaml_path:
                path = Path(yaml_path)
                if not path.is_absolute():
                    path = REPO / path
                yaml_lookup = _qwerty_yaml(path)
                report['metadata_sources'].append(dict(path=str(path), sha256=sha256(path)))
        except (OSError, ValueError, KeyError) as exc:
            report['protocol_errors'].append(f'{dataset}: {exc}')
        basenames = Counter(path.name for path in files)
        for index, path in enumerate(files):
            if index % 100 == 0:
                print(f'[inspect] {dataset} {index + 1}/{len(files)}', flush=True)
            try:
                entry = inspect_file(dataset, path)
            except Exception as exc:
                report['format_errors'].append(f'{path}: {type(exc).__name__}: {exc}')
                continue
            current['files'].append(entry)
            if dataset in NATURE_DATASETS:
                if path.name in all_nature:
                    report['protocol_errors'].append(f'ambiguous Nature basename: {path.name}')
                all_nature[path.name] = (dataset, entry)
                continue
            keys = [path.relative_to(root).as_posix(), path.name, path.stem]
            splits = {lookup[key] for key in keys if key in lookup}
            if yaml_lookup is not None and path.stem in yaml_lookup:
                splits.add(yaml_lookup[path.stem])
            if entry['source_split'] in OFFICIAL:
                splits.add(entry['source_split'])
            if len(splits) > 1:
                report['protocol_errors'].append(f'{path}: conflicting official split {splits}')
            elif len(splits) == 1:
                if basenames[path.name] > 1:
                    report['protocol_errors'].append(f'ambiguous non-Nature basename: {path.name}')
                entry['official_split'] = next(iter(splits))
                coverage[dataset][entry['official_split']] += 1
            elif yaml_lookup is not None and config.get('exclude_qwerty_outside_generic', False):
                entry['excluded'] = True
                current['excluded_sources'].append(str(path))
            else:
                report['protocol_errors'].append(f'{path}: official split missing')
        if dataset not in NATURE_DATASETS and files and not coverage[dataset]:
            report['protocol_errors'].append(f'{dataset}: zero files assigned to official protocol')

    # Retain exactly the official intervals intersecting available recordings.
    # In the public handwriting CSV, [0, inf) means the complete recording.
    # Bound it to actual source samples for the native finite-interval loader.
    intervals = set()
    for path in sorted(corpus_paths):
        try:
            columns, rows = _csv(path)
            if not {'dataset', 'split', 'start', 'end'} <= set(columns):
                raise ValueError('missing official corpus columns')
            report['metadata_sources'].append(dict(path=str(path), sha256=sha256(path)))
            for row in rows:
                name = _basename(row['dataset'].strip())
                if name not in all_nature:
                    continue
                dataset, entry = all_nature[name]
                split = row['split'].strip()
                a, b = float(row['start']), float(row['end'])
                if split not in OFFICIAL or math.isnan(a) or math.isnan(b) or a >= b:
                    raise ValueError(f'invalid corpus interval: {row}')
                if entry['source_split'] in OFFICIAL and split != entry['source_split']:
                    raise ValueError(f'corpus/source split conflict: {name}')
                a, b = max(a, entry['source_start']), min(b, entry['source_end'])
                if a < b:
                    intervals.add((name, split, a, b))
        except (OSError, ValueError) as exc:
            report['protocol_errors'].append(f'{path}: {exc}')
    grouped = defaultdict(list)
    for name, split, start, end in sorted(intervals):
        grouped[name].append((start, end, split))
        report['nature_intervals'].append(dict(dataset=name, split=split, start=start, end=end))
        coverage[all_nature[name][0]][split] += 1
    for name, (dataset, _) in all_nature.items():
        if not grouped[name]:
            report['protocol_errors'].append(f'{name}: no official corpus interval intersects the recording')
        previous_end = -math.inf
        for start, end, split in sorted(grouped[name]):
            if start < previous_end:
                report['protocol_errors'].append(f'{name}: official corpus overlap at {start}')
            previous_end = max(previous_end, end)
    report['format_status'] = 'FAIL' if report['format_errors'] else 'PASS'
    report['protocol_status'] = 'BLOCKED' if report['protocol_errors'] else 'PASS'
    report['ready_to_prepare'] = report['format_status'] == report['protocol_status'] == 'PASS'
    report['coverage'] = {key: dict(value) for key, value in coverage.items()}
    report['training_coverage_status'] = 'PASS' if all(x['train'] and x['val'] for x in coverage.values()) else 'BLOCKED'
    report['estimated_uncompressed_input_bytes'] = sum(e['uncompressed_bytes'] for d in report['datasets'].values() for e in d['files'])
    report['note'] = 'Coverage counts are metadata only; native preflight must still verify usable windows, batches, CUDA and optimizer steps.'
    return report


def prepare(config, report=None):
    roots, canonical, _ = _paths(config)
    # Never trust a saved report or bypass inspection after source metadata changes.
    report = inspect(config)
    if not report['ready_to_prepare']:
        raise ValueError('inspection/protocol failed: ' + '; '.join([*report['format_errors'], *report['protocol_errors']][:12]))
    canonical.mkdir(parents=True, exist_ok=True)
    from semg_h5.convert_emg2qwerty import _acquire_manifest_lock, _release_manifest_lock
    lock = _acquire_manifest_lock(canonical / 'server_prepare')
    try:
        return _prepare_locked(config, report, roots, canonical)
    finally:
        _release_manifest_lock(lock)


def _prepare_locked(config, report, roots, canonical):
    state = canonical / '.prepare'
    owner_path = state / 'owner.json'
    identity = {key: str(path) for key, path in roots.items()}
    if owner_path.exists():
        owner = json.loads(owner_path.read_text(encoding='utf-8'))
        if owner['raw_roots'] != identity:
            raise ValueError('canonical ownership/source roots changed; use a new canonical root')
        for name in ('manifest.csv', 'splits.csv', 'nature_corpus.csv'):
            target = canonical / name
            expected = {table[name] for table in (owner.get('metadata', {}), owner.get('pending_metadata', {})) if name in table}
            if target.exists() and (not expected or sha256(target) not in expected):
                raise ValueError(f'canonical metadata hash changed/unowned: {target}')
            if not target.exists() and name in owner.get('metadata', {}) and not owner.get('pending_metadata'):
                raise ValueError(f'committed metadata missing: {target}')
    else:
        unexpected = [p for p in canonical.rglob('*') if p.is_file() and p.name != '.server_prepare.lock']
        if unexpected:
            raise ValueError(f'nonempty canonical contains unowned files: {unexpected[0]}; use a new canonical root')
        owner = dict(schema='metafmodel-canonical-owner-v1', raw_roots=identity, metadata={})
        _json(owner_path, owner)
    journals = state / 'journals'
    journals.mkdir(parents=True, exist_ok=True)
    result = dict(converted_sources=[], skipped_sources=[], excluded_sources=[])
    rows = []
    active = set()
    for dataset, detail in report['datasets'].items():
        module = _converter_module(dataset)
        for entry in detail['files']:
            source = Path(entry['path'])
            if entry.get('excluded'):
                result['excluded_sources'].append(str(source))
                continue
            key = hashlib.sha256((dataset + ':' + str(source)).encode()).hexdigest()
            active.add(key + '.json')
            journal = journals / (key + '.json')
            stage = state / 'objects' / key
            fingerprint = sha256(source)
            split = entry.get('official_split', 'unknown')
            if journal.exists():
                saved = json.loads(journal.read_text(encoding='utf-8'))
                if saved['source_sha256'] != fingerprint or saved['official_split'] != split:
                    raise ValueError(f'source hash/split changed: {source}; use a new canonical root')
                resumed = saved['state'] == 'committed'
            else:
                # Existing staging belongs solely to this source and is replaceable
                # until its complete transaction has been journaled.
                need = entry['uncompressed_bytes'] * 2
                if shutil.disk_usage(canonical).free < need:
                    raise ValueError(f'insufficient conversion scratch space for {source}; need approximately {need} bytes')
                print(f'[convert] {dataset}: {source}', flush=True)
                converted = module.convert_file(source, stage, official_split=split, overwrite=True,
                                                source_file=source.relative_to(roots[dataset]).as_posix())
                converted = converted if isinstance(converted, list) else [converted]
                if sha256(source) != fingerprint:
                    raise ValueError(f'source changed during conversion: {source}')
                outputs = {row['path']: sha256(stage / row['path']) for row in converted}
                for relative in outputs:
                    if (canonical / relative).exists():
                        raise ValueError(f'unowned output already exists: {canonical / relative}; use a new canonical root')
                saved = dict(state='planned', source=str(source), source_sha256=fingerprint,
                             official_split=split, rows=converted, outputs=outputs)
                _json(journal, saved)
                resumed = False
            for relative, expected in saved['outputs'].items():
                target = (canonical / relative).resolve()
                if not target.is_relative_to(canonical):
                    raise ValueError(f'journal path escapes canonical: {relative}')
                if target.exists():
                    if sha256(target) != expected:
                        raise ValueError(f'output hash changed: {target}')
                elif saved['state'] == 'committed':
                    raise ValueError(f'committed output missing: {target}')
                else:
                    staged = stage / relative
                    if not staged.is_file() or sha256(staged) != expected:
                        raise ValueError(f'staging hash changed/missing: {staged}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staged.rename(target)
            saved['state'] = 'committed'
            _json(journal, saved)
            rows.extend(saved['rows'])
            result['skipped_sources' if resumed else 'converted_sources'].append(str(source))
    stale = {p.name for p in journals.glob('*.json')} - active
    if stale:
        raise ValueError('source inventory changed: stale conversion journals; use a new canonical root')
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('canonical output path collision')
    columns = _converter_module('emg2qwerty').MANIFEST_COLUMNS
    finalize = state / 'finalize'
    finalize.mkdir(parents=True, exist_ok=True)
    write_rows(finalize / 'manifest.csv', rows, columns)
    write_rows(finalize / 'nature_corpus.csv', report['nature_intervals'], ['dataset', 'split', 'start', 'end'])
    build_splits(canonical, int(config.get('seed', 42)), manifest_path=finalize / 'manifest.csv',
                 output_path=finalize / 'splits.csv')
    owner['pending_metadata'] = {name: sha256(finalize / name) for name in ('manifest.csv', 'splits.csv', 'nature_corpus.csv')}
    _json(owner_path, owner)
    for name in owner['pending_metadata']:
        (finalize / name).replace(canonical / name)
    owner['metadata'] = owner.pop('pending_metadata')
    _json(owner_path, owner)
    _json(canonical / 'preparation_report.json', dict(inspection=report, result=result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'prepare', 'preflight', 'train'])
    parser.add_argument('--config', default=str(REPO / 'configs/server/user_paths.json'))
    parser.add_argument('--report', help='optional inspection JSON path outside raw/source roots')
    parser.add_argument('--batch-size', type=int, help='same per-device batch size for native preflight and training')
    args, extra = parser.parse_known_args(argv)
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    roots, canonical, runs = _paths(config)
    if args.mode in ('inspect', 'prepare'):
        if extra:
            parser.error(f'unrecognized arguments: {extra}')
        report = inspect(config)
        target = Path(args.report).resolve() if args.report else runs / 'data_inspection.json'
        for forbidden in [*roots.values(), REPO, canonical]:
            if target == forbidden or target.is_relative_to(forbidden):
                raise ValueError('inspection report must be outside raw/source/canonical roots')
        _json(target, report)
        print(json.dumps({k: report[k] for k in ('format_status', 'protocol_status', 'training_coverage_status', 'ready_to_prepare', 'format_errors', 'protocol_errors')}, ensure_ascii=False, indent=2))
        print(f'Inspection: {target}')
        if not report['ready_to_prepare']:
            return 1
        if args.mode == 'prepare':
            print(json.dumps(prepare(config, report), ensure_ascii=False, indent=2))
        return 0
    env = dict(os.environ, E3_DATA_ROOT=str(canonical), E3_RUN_ROOT=str(runs), E3_PYTHON=sys.executable)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError('batch-size must be positive')
        env['E3_BATCH_SIZE'] = str(args.batch_size)
    # The configured corpus is the bounded copy produced by prepare, not a
    # stale environment override left over from another project.
    env['E3_NATURE_CORPUS'] = str(canonical / 'nature_corpus.csv')
    if args.mode == 'preflight':
        cmd = [sys.executable, str(REPO / 'scripts/server_preflight.py'), '--data-root', str(canonical),
               '--output', str(runs / 'preflight'), '--batch-size', env.get('E3_BATCH_SIZE', '8'), *extra]
    else:
        if env.get('E3_SKIP_PREFLIGHT') == '1' or '--skip-preflight' in extra:
            raise ValueError('this deployment entry requires native preflight')
        cmd = ['bash', str(REPO / 'scripts/run_e3_server.sh'), str(canonical), *extra]
    return subprocess.run(cmd, cwd=REPO, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
