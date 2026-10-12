"""Original HDF5 windows feeding the same Ring/Lightning training interface.

Canonical conversion remains optional. Uses migrated MetaEMGFM signal helpers
and the upstream window geometry. No amplitude transform or resampling.
"""
from pathlib import Path
from functools import lru_cache

import h5py
import numpy as np
import torch

from .manifest import load_manifest
from .intervals import OFFICIAL, NATURE_DATASETS, corpus_bounds, validate_group_assignments
from .raw_signal import describe_record, read_field, field_names, validate_time, LAYOUTS
from .window_dataset import CanonicalWindowDataset

TASK_MAP={'pose':'pose','typing':'qwerty','discrete_gesture':'gesture','handwriting':'handwriting','wrist_angle':'wrist'}


class RawWindowDataset(CanonicalWindowDataset):
    @classmethod
    def from_manifest(cls,manifest_path,split,window_ms,stride_ms,root=None,task_id=None,corpus_path=None):
        if split not in OFFICIAL: raise ValueError('invalid split')
        frame=load_manifest(manifest_path,root or Path(manifest_path).parent)
        validate_group_assignments(frame)
        names=[Path(r['source_file']).name for r in frame.to_dict('records') if r['dataset_id'] in NATURE_DATASETS]
        if len(names)!=len(set(names)): raise ValueError('ambiguous Nature source filename')
        records=[]
        for row in frame.to_dict('records'):
            if task_id is not None and row['task_id']!=task_id: continue
            if row['task_id'] not in TASK_MAP: raise ValueError('unsupported raw task')
            row['path']=Path(row['resolved_path'])
            row['sample_count']=int(row['sample_count']);row['sample_rate_hz']=float(row['sample_rate_hz'])
            info=describe_record(dict(path=row['path'],dataset_id=TASK_MAP[row['task_id']],hand_side=row['side']))
            if info['length']!=row['sample_count']: raise ValueError('sample_count mismatch')
            if info['official_split'] in OFFICIAL and info['official_split']!=row['official_split']:
                raise ValueError('raw official split conflicts with manifest')
            if Path(row['source_file']).name!=row['path'].name: raise ValueError('source_file mismatch')
            row.update(raw_info=info,record_id=row['path'].stem)
            effective=row['official_split'] if row['official_split'] in OFFICIAL else row.get('split','')
            if row['dataset_id'] in NATURE_DATASETS:
                corpus=corpus_path.get(row['task_id']) if isinstance(corpus_path,dict) else corpus_path
                if not corpus: raise ValueError('Nature requires official corpus intervals')
                with h5py.File(row['path']) as f:
                    times=read_field(f[LAYOUTS[TASK_MAP[row['task_id']]][0]],'time',0,row['sample_count'])
                for a,b,official,lo,hi in corpus_bounds(row,corpus,times=times,origin=0):
                    if official==split:
                        records.append(dict(row,sample_start=a,sample_end=b,split=split,corpus_start_s=lo,corpus_end_s=hi))
            elif effective==split:
                records.append(dict(row,sample_start=0,sample_end=row['sample_count'],split=split))
        return cls(records,window_ms,stride_ms)

    def __getitem__(self,index):
        path,a,b=self._index[index]
        position=int(np.searchsorted(self._index.offsets,int(index)%len(self),side='right')-1)
        row=self._record_rows[position]
        task=TASK_MAP[row['task_id']]
        info=describe_record(dict(path=path,dataset_id=task,hand_side=row['side']))
        if info['official_split'] in OFFICIAL and info['official_split']!=row['official_split']:
            raise ValueError('raw official split changed')
        with h5py.File(path) as f:
            node=f[LAYOUTS[task][0]]
            emg=np.asarray(read_field(node,info['field'],a,b),dtype='f4')
            times=np.asarray(read_field(node,info['time_field'],a,b),dtype='f8')
            validate_time(times,row['sample_rate_hz'],packetized=task in ('wrist','gesture','handwriting'))
            if not np.isfinite(emg).all(): raise ValueError('nonfinite raw EMG')
            fields=field_names(node)
            item=dict(emg=torch.from_numpy(emg),sample_mask=torch.ones(b-a,dtype=torch.bool),
                      dataset_id=row['dataset_id'],task_id=row['task_id'],record_id=row['record_id'],
                      subject_id=row['subject_id'],session_id=row['session_id'],side=row['side'],hand_side=row['side'],
                      pair_id=row['pair_id'],source_file=row['source_file'],split=row['split'],sample_start=a,sample_end=b,
                      sampling_rate=row['sample_rate_hz'],sample_rate_hz=row['sample_rate_hz'],start_s=float(times[0]))
            if task=='pose':
                target=np.asarray(read_field(node,'joint_angles',a,b),dtype='f4')
                if not np.isfinite(target).all(): raise ValueError('nonfinite Pose angles')
                mask=np.asarray(read_field(node,'no_ik_failure',a,b),dtype=bool) if 'no_ik_failure' in fields else np.ones(b-a,dtype=bool)
                item.update(target=torch.from_numpy(target),target_mask=torch.from_numpy(mask))
            elif task=='wrist':
                names=[name for name in fields if name not in ('time','emg') and np.issubdtype(node.dtype.fields[name][0].base,np.number)] if isinstance(node,h5py.Dataset) else []
                targets={name:torch.as_tensor(read_field(node,name,a,b),dtype=torch.float32) for name in names}
                if not targets: raise ValueError('raw Wrist lacks continuous targets')
                item.update(target=targets,target_mask={name:torch.isfinite(value) for name,value in targets.items()})
            else:
                # Preserve original task label decoding via upstream readers.
                # They read source metadata but never write a converted file.
                item.update(_source_targets(row,path,float(times[0]),min(float(times[-1]+.5/row['sample_rate_hz']),row.get('corpus_end_s',float('inf')))))
        return item


def _source_targets(row,path,start,end):
    info=Path(path).stat()
    labels,origin=_source_labels(str(path),row['task_id'],info.st_mtime_ns,info.st_size)
    start-=origin;end-=origin
    if row['task_id']=='discrete_gesture':
        return dict(events=[(str(r['name']),float(r['time_s'])-start) for r in labels['events'] if start<=float(r['time_s'])<end])
    prompts=labels.get('prompts',[])
    containing=[r for r in prompts if float(r['start_s'])<=start and end<=float(r['end_s'])]
    intersecting=[r for r in prompts if float(r['start_s'])<end and start<float(r['end_s'])]
    prompt=next(iter(containing or intersecting),None)
    result=dict(target_text=str(prompt['text']) if prompt else '',prompt_start_s=float(prompt['start_s']) if prompt else None,prompt_end_s=float(prompt['end_s']) if prompt else None)
    if row['task_id']=='typing':
        for key in ('prompt_start_s','prompt_end_s'):
            if result[key] is not None:result[key]-=start
        result['keystrokes']=[(str(r['key']),float(r['press_s'])-start,float(r['release_s'])-start) for r in labels['keystrokes'] if start<=float(r['press_s'])<end]
    return result


@lru_cache(maxsize=64)
def _source_labels(path,task,mtime_ns,size):
    """Cache bounded lightweight label metadata, never HDF5 handles or EMG.

    Reuses upstream converter metadata/table decoders. File generation enters
    the cache key so edited sources cannot reuse stale label tables.
    """
    path=Path(path)
    with h5py.File(path) as f:
        node=f['emg2qwerty/timeseries'] if task=='typing' else f['data']
        origin=float(read_field(node,'time',0,1)[0])
        if task=='typing':
            from semg_h5.convert_emg2qwerty import _relative_qwerty_labels,_decode_value
            group=f['emg2qwerty']
            metadata={str(key):_decode_value(value) for key,value in group.attrs.items()}
            labels,_=_relative_qwerty_labels(group,metadata,origin)
            return labels,origin
        expected='discrete_gestures' if task=='discrete_gesture' else 'handwriting'
        actual=node.attrs.get('task','');actual=actual.decode() if isinstance(actual,bytes) else str(actual)
        if actual!=expected:raise ValueError(f'raw source task {actual!r} does not match {expected!r}')
    if task=='discrete_gesture':
        from semg_h5.convert_gni_discrete_gestures import _read_gni_table,_finite_float,_required_value
        rows=_read_gni_table(path,'prompts')
        if not rows:raise ValueError('Discrete gesture records require prompt/event rows')
        return dict(events=[dict(name=str(_required_value(r,('name','gesture'),'gesture')),
                 time_s=_finite_float(_required_value(r,('time','time_s'),'gesture'),'gesture time')-origin) for r in rows]),origin
    from semg_h5.convert_gni_handwriting import _rows,_finite_float,_required
    rows=_rows(path,'prompts');prompts=[]
    if not rows:raise ValueError('Handwriting records require prompt rows')
    for row in rows:
        start=_finite_float(_required(row,('start','start_s'),'handwriting prompt'),'handwriting prompt start')
        end=_finite_float(_required(row,('end','end_s'),'handwriting prompt'),'handwriting prompt end')
        if start>end:raise ValueError('handwriting prompt start must not exceed end')
        prompts.append(dict(text=str(_required(row,('prompt','text'),'handwriting prompt')),start_s=start-origin,end_s=end-origin))
    return dict(prompts=prompts),origin
