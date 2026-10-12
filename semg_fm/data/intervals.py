"""Official group protection and Nature half-open corpus indexing."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np

OFFICIAL = {'train', 'val', 'test'}
NATURE_DATASETS = {'gni_wrist', 'gni_handwriting', 'gni_discrete_gestures'}


def validate_group_assignments(frame):
    rows = frame.to_dict('records')
    pairs, subjects = {}, {}
    for row in rows:
        official = str(row['official_split']).strip()
        assigned = str(row.get('split', '')).strip()
        if official not in OFFICIAL | {'', 'unknown'} or assigned not in OFFICIAL | {'', 'unknown'}:
            raise ValueError('invalid official_split or assigned split')
        # An official release split is authoritative.  A stale/generated
        # ``split`` column may disagree and is ignored for that row.
        effective = official if official in OFFICIAL else assigned
        subject = (row['dataset_id'], row['subject_id'])
        subjects.setdefault(subject, []).append((official, effective))
        if row['pair_id'] and effective in OFFICIAL:
            key = (row['dataset_id'], row['pair_id'])
            if key in pairs and pairs[key] != effective:
                raise ValueError('pair crosses split boundaries')
            pairs[key] = effective
    for group in subjects.values():
        known = {split for _, split in group if split in OFFICIAL}
        if any(official not in OFFICIAL for official, _ in group) and len(known) > 1:
            raise ValueError('unknown official subject has ambiguous cross-split assignments')


def anchored_group_splits(frame, seed):
    from .splits import assign_group_splits
    validate_group_assignments(frame)
    if any(value in NATURE_DATASETS for value in frame['dataset_id']):
        raise ValueError('Nature requires official corpus intervals, not whole-file random splits')
    result = assign_group_splits(frame, seed)
    # Reuse upstream grouping itself with a unique seed-independent tag per
    # component, rather than depending on the generated split being unique.
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    rows = frame.to_dict('records')
    for i, row in enumerate(rows):
        keys = [('subject', row['dataset_id'], row['subject_id'])]
        if row['pair_id']: keys.append(('pair', row['dataset_id'], row['pair_id']))
        for key in keys:
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = {}
    for i in range(len(rows)): groups.setdefault(find(i), []).append(i)
    for positions in groups.values():
        known = {rows[i]['official_split'] for i in positions if rows[i]['official_split'] in OFFICIAL}
        unknown = [i for i in positions if rows[i]['official_split'] not in OFFICIAL]
        if unknown and len(known) > 1:
            raise ValueError('unknown official subject has ambiguous official anchors')
        for i in positions:
            official = rows[i]['official_split']
            result.loc[result.index[i], 'split'] = official if official in OFFICIAL else next(iter(known), result.iloc[i]['split'])
    validate_group_assignments(result)
    return result


def corpus_bounds(row, corpus, times=None, origin=None):
    """Return all official intervals as (start_index,end_index,split,start_s,end_s)."""
    with Path(corpus).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'dataset', 'split', 'start', 'end'}.issubset(reader.fieldnames or []):
            raise ValueError('corpus requires dataset,split,start,end')
        rows = [r for r in reader if Path(r['dataset']).name == Path(row['source_file']).name]
    if not rows:
        raise ValueError(f"source {row['source_file']} missing from corpus")
    if times is None:
        with h5py.File(row['path'], 'r') as f:
            source=f.attrs.get('source_file','')
            source=source.decode() if isinstance(source,bytes) else str(source)
            if source!=row['source_file']:
                raise ValueError('source_file does not match manifest')
            times = f['signal/time_s'][:]
            origin = json.loads(f['meta/source_json'].asstr()[()])['source_time_origin_s']
    origin = float(origin)
    if not np.isfinite(origin) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('invalid corpus source clock')
    bounds = []
    for r in rows:
        lo, hi = float(r['start']), float(r['end'])
        if not np.isfinite([lo, hi]).all() or lo >= hi or r['split'] not in OFFICIAL:
            raise ValueError('invalid corpus interval/split')
        if row['official_split'] in OFFICIAL and row['official_split']!=r['split']:
            raise ValueError('corpus split conflicts with source official split')
        a = int(np.searchsorted(times, lo-origin, side='left'))
        b = int(np.searchsorted(times, hi-origin, side='left'))
        if a < b: bounds.append((a, b, r['split'], lo, hi))
    ordered = sorted(bounds)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('corpus overlap across or within splits')
    return ordered


def selected_records(records, split, corpus_path=None):
    """Expand whole recordings into official intervals before window indexing."""
    names = [Path(r['source_file']).name for r in records if r['dataset_id'] in NATURE_DATASETS]
    if len(names) != len(set(names)):
        raise ValueError('ambiguous duplicate Nature source filename')
    selected = []
    for row in records:
        if row['dataset_id'] in NATURE_DATASETS:
            corpus = corpus_path.get(row['task_id']) if isinstance(corpus_path, dict) else corpus_path
            if not corpus:
                raise ValueError('Nature dataset requires official corpus intervals')
            for start, end, official, lo, hi in corpus_bounds(row, corpus):
                if official == split:
                    selected.append(dict(row, sample_start=start, sample_end=end, split=split,
                                         corpus_start_s=lo, corpus_end_s=hi, source_time_origin_s=float(_origin(row)),
                                         corpus_path=str(Path(corpus).resolve())))
        elif row['effective_split'] == split:
            selected.append(dict(row, split=split, sample_start=0, sample_end=row['sample_count']))
    return selected


def _origin(row):
    with h5py.File(row['path'],'r') as f:
        return json.loads(f['meta/source_json'].asstr()[()])['source_time_origin_s']
