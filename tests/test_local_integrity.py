import json
import h5py
import numpy as np
import pandas as pd
import pytest

from semg_fm.data.window_dataset import CanonicalWindowDataset


def wrist(root):
    path=root/'wrist.h5'
    row=dict(path=path.name,dataset_id='gni_wrist',task_id='wrist_angle',subject_id='001',
             session_id='session',side='unknown',pair_id='',official_split='unknown',
             sample_count=8000,duration_s=3.9995,sample_rate_hz=2000,
             source_file='wrist_source.hdf5',source_sha256='0'*64)
    with h5py.File(path,'w') as f:
        f.attrs.update(schema_version='1.0',**{k:row[k] for k in ('dataset_id','task_id','subject_id','session_id','side','pair_id','official_split','sample_rate_hz','source_file')})
        f['signal/emg']=np.arange(8000*16,dtype='f4').reshape(8000,16)
        f['signal/time_s']=np.arange(8000,dtype='f8')/2000
        f['labels/targets/wrist_angles']=np.zeros((8000,2))
        f['meta/source_json']=json.dumps(dict(source_time_origin_s=1000))
    manifest=root/'manifest.csv'
    pd.DataFrame([row]).to_csv(manifest,index=False)
    return path,manifest,row


def test_nature_no_corpus_rejects_whole_file_assignment(tmp_path):
    path,p,row=wrist(tmp_path)
    row['split']='train'
    pd.DataFrame([row]).to_csv(p,index=False)
    with pytest.raises(ValueError,match='corpus'):
        CanonicalWindowDataset.from_manifest(p,'train',2000,1000)


def test_nature_interval_indexes_and_metadata(tmp_path):
    path,p,row=wrist(tmp_path)
    corpus=tmp_path/'corpus.csv'
    corpus.write_text('dataset,split,start,end\nwrist_source.hdf5,test,1000,1001\nwrist_source.hdf5,train,1001,1004\n')
    ds=CanonicalWindowDataset.from_manifest(p,'train',2000,1000,corpus_path=corpus)
    assert len(ds)==2
    assert ds[0]['sample_start']==2000 and ds[1]['sample_end']==8000
    assert ds[0]['split']=='train' and ds[0]['source_file']=='wrist_source.hdf5'
    assert ds[0]['hand_side']=='unknown' and ds[0]['sampling_rate']==2000
    assert ds[0]['emg'].shape==(4000,16)
    np.testing.assert_array_equal(ds[0]['emg'].numpy(),np.arange(8000*16,dtype='f4').reshape(8000,16)[2000:6000])


def test_corpus_cross_split_overlap_rejected(tmp_path):
    _,p,_=wrist(tmp_path)
    corpus=tmp_path/'corpus.csv'
    corpus.write_text('dataset,split,start,end\nwrist_source.hdf5,train,1000,1004\nwrist_source.hdf5,test,1001,1003\n')
    with pytest.raises(ValueError,match='overlap'):
        CanonicalWindowDataset.from_manifest(p,'train',200,100,corpus_path=corpus)


def test_corpus_cannot_override_whole_record_official_split(tmp_path):
    path,p,row=wrist(tmp_path)
    row['official_split']='test'
    with h5py.File(path,'r+') as f: f.attrs['official_split']='test'
    pd.DataFrame([row]).to_csv(p,index=False)
    corpus=tmp_path/'corpus.csv'
    corpus.write_text('dataset,split,start,end\nwrist_source.hdf5,train,1000,1004\n')
    with pytest.raises(ValueError,match='official'):
        CanonicalWindowDataset.from_manifest(p,'train',200,100,corpus_path=corpus)


def test_event_targets_cannot_cross_corpus_gap_boundary(tmp_path):
    path,p,row=wrist(tmp_path)
    row.update(dataset_id='gni_discrete_gestures',task_id='discrete_gesture')
    times=np.concatenate((np.arange(4000)/2000,2.1+np.arange(4000)/2000))
    with h5py.File(path,'r+') as f:
        f.attrs.update(dataset_id=row['dataset_id'],task_id=row['task_id'])
        f['signal/time_s'][:]=times
        f['labels/events']=np.array([(b'test_event',2.07)],dtype=[('name','S20'),('time_s','f8')])
    pd.DataFrame([row]).to_csv(p,index=False)
    corpus=tmp_path/'corpus.csv'
    corpus.write_text('dataset,split,start,end\nwrist_source.hdf5,train,1000,1002.05\nwrist_source.hdf5,test,1002.05,1006\n')
    ds=CanonicalWindowDataset.from_manifest(p,'train',2000,1000,corpus_path=corpus)
    assert ds[0]['events']==[]


def test_source_file_crosscheck(tmp_path):
    _,p,row=wrist(tmp_path)
    row['source_file']='wrong.hdf5'
    pd.DataFrame([row]).to_csv(p,index=False)
    corpus=tmp_path/'corpus.csv'
    corpus.write_text('dataset,split,start,end\nwrong.hdf5,train,1000,1004\n')
    with pytest.raises(ValueError,match='source_file'):
        CanonicalWindowDataset.from_manifest(p,'train',200,100,corpus_path=corpus)


def test_official_override_is_ignored(canonical_meta_root):
    p=canonical_meta_root/'manifest.csv'
    frame=pd.read_csv(p,dtype=str,keep_default_na=False)
    frame['split']='test'
    frame.to_csv(p,index=False)
    ds=CanonicalWindowDataset.from_manifest(p,'train',200,100)
    assert len(ds)>0 and ds[0]['split']=='train'
    assert len(CanonicalWindowDataset.from_manifest(p,'test',200,100))==0
