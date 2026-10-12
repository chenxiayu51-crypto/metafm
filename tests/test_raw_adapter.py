import h5py
import numpy as np
import pandas as pd
import pytest


def test_raw_pose_time_major_values_labels_and_official_split(tmp_path):
    from semg_fm.data.raw_adapter import RawWindowDataset
    path=tmp_path/'pose.hdf5'
    x=np.zeros(800,dtype=[('emg','f4',(16,)),('time','f8'),('joint_angles','f4',(20,)),('no_ik_failure','?')])
    x['emg']=np.arange(800*16,dtype='f4').reshape(800,16)
    x['joint_angles']=1
    x['time']=1000+np.arange(800)/2000
    x['no_ik_failure']=True;x['no_ik_failure'][10]=False
    with h5py.File(path,'w') as f:
        f['emg2pose/timeseries']=x
        f['emg2pose'].attrs.update(split='val',user='u',side='left')
    row=dict(path=path.name,dataset_id='emg2pose',task_id='pose',subject_id='u',session_id='s',side='left',pair_id='',
             official_split='val',sample_count=800,duration_s=.3995,sample_rate_hz=2000,source_file=path.name,source_sha256='0'*64)
    p=tmp_path/'manifest.csv';pd.DataFrame([row]).to_csv(p,index=False)
    ds=RawWindowDataset.from_manifest(p,'val',200,100)
    item=ds[0]
    np.testing.assert_array_equal(item['emg'],x['emg'][:400])
    assert item['target'].shape==(400,20) and not item['target_mask'][10]
    row['official_split']='train';pd.DataFrame([row]).to_csv(p,index=False)
    with pytest.raises(ValueError,match='official'):
        RawWindowDataset.from_manifest(p,'train',200,100)


def test_packet_timestamps_do_not_redefine_declared_sample_rate():
    from semg_fm.data.raw_signal import validate_time
    # A short crop can start/end inside timestamp packets. Its span is not
    # an estimator of the declared 2kHz acquisition rate.
    times=1000+np.repeat(np.arange(11)*.02,40)[:400]
    validate_time(times,2000,packetized=True)


@pytest.mark.parametrize('task', ['typing','discrete_gesture','handwriting'])
def test_raw_labels_match_original_canonical_converter(tmp_path,qwerty_h5,gni_h5_by_task,task,monkeypatch):
    import importlib
    import torch
    from semg_fm.data.raw_adapter import RawWindowDataset
    from semg_fm.data.window_dataset import CanonicalWindowDataset
    module=importlib.import_module('semg_h5.'+{'typing':'convert_emg2qwerty',
        'discrete_gesture':'convert_gni_discrete_gestures','handwriting':'convert_gni_handwriting'}[task])
    source=qwerty_h5 if task=='typing' else gni_h5_by_task[task]
    output=tmp_path/'canonical';output.mkdir()
    converted=module.convert_file(source,output,official_split='train' if task=='typing' else 'unknown')
    rows=converted if isinstance(converted,list) else [converted]
    manifest=output/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    raw_manifest=tmp_path/'raw.csv'
    raw_rows=[dict(r,path=source.name,source_file=source.name) for r in rows]
    pd.DataFrame(raw_rows).to_csv(raw_manifest,index=False)
    corpus=tmp_path/'corpus.csv'
    pd.DataFrame([dict(dataset=source.name,split='train',start=30,end=30.002)]).to_csv(corpus,index=False)
    kwargs=dict(corpus_path=corpus) if task!='typing' else {}
    canonical=CanonicalWindowDataset.from_manifest(manifest,'train',1,1,root=output,task_id=task,**kwargs)
    raw=RawWindowDataset.from_manifest(raw_manifest,'train',1,1,root=source.parent,task_id=task,**kwargs)
    def forbidden_full_reader(*args,**kwargs):
        raise AssertionError('raw windows must not invoke whole-record EMG converters')
    monkeypatch.setattr(module,'read_emg2qwerty' if task=='typing' else '_read_gni',forbidden_full_reader)
    assert len(raw)==len(canonical)
    for i in range(len(raw)):
        a,b=raw[i],canonical[i]
        torch.testing.assert_close(a['emg'],b['emg'],rtol=0,atol=0)
        for key in ('events','target_text','prompt_start_s','prompt_end_s','keystrokes'):
            if key in b:
                if key.endswith('_s') and b[key] is not None:
                    assert a[key]==pytest.approx(b[key])
                else:assert a[key]==b[key]


def test_long_packet_window_rejects_constant_clock():
    from semg_fm.data.raw_signal import validate_time
    with pytest.raises(ValueError,match='constant'):
        validate_time(np.ones(4000),2000,packetized=True)
