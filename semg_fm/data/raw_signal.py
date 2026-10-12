# Raw format helpers migrated from MetaEMGFM; source recorded in UPSTREAM.md.
"""Read original HDF5 lazily and index windows within explicit split intervals."""
import re
from pathlib import Path

import h5py
import numpy as np


LAYOUTS = {
    'pose': ('emg2pose/timeseries', {'unknown': 'emg'}),
    'qwerty': ('emg2qwerty/timeseries', {'left': 'emg_left', 'right': 'emg_right'}),
    'gesture': ('data', {'unknown': 'emg'}),
    'handwriting': ('data', {'unknown': 'emg'}),
    'wrist': ('data', {'unknown': 'emg'}),
}


def field_names(node):
    return list(node.keys()) if isinstance(node, h5py.Group) else list(node.dtype.names or ())


def read_field(node, field, start, end):
    if isinstance(node, h5py.Group):
        return node[field][start:end]
    return node.fields(field)[start:end]


def signal_field(dataset_id, hand_side):
    if dataset_id not in LAYOUTS:
        raise ValueError(f'Unknown dataset_id: {dataset_id}')
    if dataset_id == 'qwerty':
        if hand_side not in ('left', 'right'):
            raise ValueError('QWERTY requires explicit left or right hand_side')
        return f'emg_{hand_side}'
    return 'emg'


def describe_record(record):
    path = Path(record['path']).resolve()
    layout = LAYOUTS[record['dataset_id']][0]
    field = signal_field(record['dataset_id'], record.get('hand_side', 'unknown'))
    with h5py.File(path, 'r') as f:
        if layout not in f:
            raise ValueError(f'{path}: missing HDF5 node {layout}')
        node = f[layout]
        fields = field_names(node)
        if field not in fields:
            raise ValueError(f'{path}: missing {field}; fields={fields}')
        if isinstance(node, h5py.Group):
            shape = node[field].shape
            length = shape[0]
            tail = shape[1:]
        else:
            length = len(node)
            tail = node.dtype.fields[field][0].shape
        if tail != (16,):
            raise ValueError(f'{path}: expected 16 channels (N,16), got {tail}')
        time_field = next((x for x in ('time', 'timestamp', 'timestamps') if x in fields), None)
        subject = 'unknown'
        official_split = None
        match = re.search(r'_user_(\d+)_', path.name)
        if match:
            subject = match.group(1)
        hand = record.get('hand_side', 'unknown')
        for owner in (f, node.parent, node):
            if 'split' in owner.attrs:
                official_split = _text(owner.attrs['split'])
            for attr in ('subject_id', 'user_id', 'subject', 'user'):
                if attr in owner.attrs:
                    subject = _text(owner.attrs[attr])
            for attr in ('hand_side', 'side', 'hand'):
                if attr in owner.attrs and hand == 'unknown':
                    candidate = _text(owner.attrs[attr]).lower()
                    hand = candidate if candidate in ('left', 'right') else 'unknown'
        return dict(path=str(path), length=length, field=field, time_field=time_field,
                    subject_id=record.get('subject_id', subject), hand_side=hand, official_split=official_split)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def validate_time(time, sample_rate, packetized=False):
    if time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) < 0):
        raise ValueError('Invalid timestamps: expected finite nondecreasing seconds')
    dt = np.diff(time)
    expected = 1 / sample_rate
    if packetized:
        # Packet clocks repeat for multiple samples. Cropped packet endpoints
        # do not estimate acquisition rate: use the declared manifest rate.
        if len(time)/sample_rate>0.1 and not np.any(dt>0):
            raise ValueError('Invalid timestamps: constant clock in long packet window')
        if np.max(dt) > 0.1:
            raise ValueError('Timestamp discontinuity: packet gap > 100ms')
        return
    if np.any(dt <= 0):
        raise ValueError('Invalid timestamps: expected strictly increasing seconds')
    if not np.isclose(np.median(dt), expected, rtol=0.02, atol=1e-7):
        raise ValueError(f'Unexpected sampling interval {np.median(dt)}; expected {expected} seconds')
    if np.max(dt) > expected * 1.5 or np.min(dt) < expected * 0.5:
        raise ValueError('Timestamp discontinuity inside window')
