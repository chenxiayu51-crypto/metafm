"""Deployment tests use tiny fixtures; no fixture is real server evidence."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_path():
    # Deployment intentionally rejects production output inside the checkout.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='metafmodel-deployment-') as directory:
        yield Path(directory)


def load():
    spec=importlib.util.spec_from_file_location('server_dataset',Path(__file__).parents[1]/'scripts/server_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def deployment(tmp_path,qwerty_h5,gni_h5_by_task):
    pose=tmp_path/'pose/record.h5';pose.parent.mkdir()
    rows=np.zeros(4,dtype=[('emg','f4',(16,)),('time','f8'),('joint_angles','f4',(20,)),('no_ik_failure','?')])
    rows['time']=10+np.arange(4)/2000;rows['no_ik_failure']=True
    with h5py.File(pose,'w') as f:
        f['emg2pose/timeseries']=rows
        f['emg2pose'].attrs.update(split='train',subject_id='p',session_id='s',side='left')
    # Isolate the three GNI releases, as on the actual server.
    nature={}
    for task,path in gni_h5_by_task.items():
        root=tmp_path/task;root.mkdir();target=root/path.name;path.rename(target);nature[task]=root
        pd.DataFrame([dict(dataset=target.name,split='train',start=30,end=30.002)]).to_csv(root/'official_corpus.csv',index=False)
    pd.DataFrame([dict(path=qwerty_h5.name,split='train')]).to_csv(qwerty_h5.parent/'official_split.csv',index=False)
    return dict(raw_roots=dict(emg2pose=str(pose.parent),emg2qwerty=str(qwerty_h5.parent),
        gni_discrete_gestures=str(nature['discrete_gesture']),gni_handwriting=str(nature['handwriting']),
        gni_wrist=str(nature['wrist_angle'])),canonical_root=str(tmp_path/'canonical'),
        run_root=str(tmp_path/'runs'),nature_corpus_csvs=[],official_split_csvs={},seed=42)


def test_inspect_five_formats_official_protocol_without_writing_raw(deployment):
    module=load();report=module.inspect(deployment)
    assert report['format_status']=='PASS' and report['protocol_status']=='PASS'
    assert report['ready_to_prepare']
    assert report['training_coverage_status']=='BLOCKED'  # fixtures train only
    assert report['datasets']['emg2qwerty']['records_per_file']==2
    assert len(report['nature_intervals'])==3
    assert not Path(deployment['canonical_root']).exists()


def test_unknown_qwerty_split_blocks_without_random_fallback(deployment):
    (Path(deployment['raw_roots']['emg2qwerty'])/'official_split.csv').unlink()
    report=load().inspect(deployment)
    assert report['format_status']=='PASS' and not report['ready_to_prepare']
    assert any('split' in error for error in report['protocol_errors'])


def test_nature_overlapping_split_blocks(deployment):
    root=Path(deployment['raw_roots']['gni_wrist']);source=next(root.glob('*.h5'))
    pd.DataFrame([dict(dataset=source.name,split='train',start=30,end=30.0015),
                  dict(dataset=source.name,split='val',start=30.001,end=30.002)]).to_csv(root/'official_corpus.csv',index=False)
    report=load().inspect(deployment)
    assert not report['ready_to_prepare']
    assert any('overlap' in error.lower() for error in report['protocol_errors'])


def test_wrong_pose_dimension_detected_before_conversion(deployment):
    path=next(Path(deployment['raw_roots']['emg2pose']).glob('*.h5'))
    with h5py.File(path,'r+') as f:
        del f['emg2pose/timeseries']
        data=np.zeros(4,dtype=[('emg','f4',(16,)),('time','f8'),('joint_angles','f4',(3,))])
        data['time']=np.arange(4)/2000;f['emg2pose/timeseries']=data
    report=load().inspect(deployment)
    assert report['format_status']=='FAIL' and not report['ready_to_prepare']


def test_prepare_converts_five_tasks_resumes_and_rejects_changed_output(deployment):
    module=load();report=module.inspect(deployment)
    result=module.prepare(deployment,report)
    canonical=Path(deployment['canonical_root'])
    frame=pd.read_csv(canonical/'splits.csv',keep_default_na=False)
    assert len(frame)==6  # QWERTY two wrists
    assert set(frame.dataset_id)==set(deployment['raw_roots'])
    assert (frame[frame.dataset_id.str.startswith('gni_')]['split']=='').all()
    assert (frame[~frame.dataset_id.str.startswith('gni_')].official_split=='train').all()
    assert len(result['converted_sources'])==5
    resumed=module.prepare(deployment,module.inspect(deployment))
    assert len(resumed['skipped_sources'])==5 and not resumed['converted_sources']
    target=canonical/frame.iloc[0]['path']
    with h5py.File(target,'r+') as f:f['signal/emg'][0,0]=99
    with pytest.raises(ValueError,match='changed|hash'):
        module.prepare(deployment,module.inspect(deployment))


def test_canonical_inside_raw_and_source_split_conflicts_rejected(deployment):
    module=load();deployment['canonical_root']=str(Path(deployment['raw_roots']['emg2pose'])/'converted')
    with pytest.raises(ValueError,match='overlap|inside'):module.inspect(deployment)


def test_no_official_corpus_never_creates_one(deployment):
    for root in deployment['raw_roots'].values():
        for csv in Path(root).glob('*corpus.csv'):csv.unlink()
    report=load().inspect(deployment)
    assert not report['ready_to_prepare']
    with pytest.raises(ValueError,match='protocol|inspection'):load().prepare(deployment,report)
    assert not Path(deployment['canonical_root']).exists()


def test_public_handwriting_unbounded_interval_means_entire_source(deployment):
    root=Path(deployment['raw_roots']['gni_handwriting']);source=next(root.glob('*.h5'))
    pd.DataFrame([dict(dataset=source.name,split='train',start=0,end='inf')]).to_csv(root/'official_corpus.csv',index=False)
    report=load().inspect(deployment)
    assert report['ready_to_prepare']
    row=next(r for r in report['nature_intervals'] if r['dataset']==source.name)
    assert row['start']==30 and row['end']==pytest.approx(30.002)


def test_official_qwerty_yaml_excludes_unassigned_sessions_without_relabelling(deployment,tmp_path):
    root=Path(deployment['raw_roots']['emg2qwerty'])
    (root/'official_split.csv').unlink()
    yaml=tmp_path/'generic.yaml';yaml.write_text('dataset:\n  train:\n    - session: typing_1\n  val: []\n  test: []\n')
    deployment['qwerty_generic_yaml']=str(yaml)
    deployment['exclude_qwerty_outside_generic']=True
    import shutil
    shutil.copy2(root/'typing_1.h5',root/'personalization_only.h5')
    report=load().inspect(deployment)
    assert report['ready_to_prepare']
    assert report['datasets']['emg2qwerty']['excluded_sources']==[str(root/'personalization_only.h5')]


def test_csv_conflicting_with_source_split_blocks(deployment):
    root=Path(deployment['raw_roots']['emg2pose'])
    pd.DataFrame([dict(filename='record',split='val')]).to_csv(root/'metadata.csv',index=False)
    report=load().inspect(deployment)
    assert not report['ready_to_prepare']
    assert any('conflict' in e for e in report['protocol_errors'])


def test_resume_half_committed_qwerty_pair_after_interruption(deployment,monkeypatch):
    module=load();original=Path.rename;count=0
    def crash(path,target):
        nonlocal count
        if '.prepare' in str(path) and 'typing' in str(path):
            count+=1
            if count==2:raise OSError('simulated power loss')
        return original(path,target)
    monkeypatch.setattr(Path,'rename',crash)
    with pytest.raises(OSError,match='power loss'):module.prepare(deployment)
    monkeypatch.setattr(Path,'rename',original)
    module.prepare(deployment)
    frame=pd.read_csv(Path(deployment['canonical_root'])/'splits.csv')
    assert len(frame)==6 and len(set(frame.path))==6


def test_resume_rejects_modified_raw_source(deployment):
    module=load();module.prepare(deployment)
    path=next(Path(deployment['raw_roots']['emg2pose']).glob('*.h5'))
    with h5py.File(path,'r+') as f:f['emg2pose'].attrs['new_metadata']='changed'
    with pytest.raises(ValueError,match='hash.*changed'):module.prepare(deployment)


def test_prepare_refuses_existing_unowned_manifest(deployment):
    canonical=Path(deployment['canonical_root']);canonical.mkdir()
    (canonical/'manifest.csv').write_text('IMPORTANT_EXISTING_MANIFEST')
    with pytest.raises(ValueError,match='unowned|nonempty'):load().prepare(deployment)
    assert (canonical/'manifest.csv').read_text()=='IMPORTANT_EXISTING_MANIFEST'


def test_resume_refuses_modified_corpus_metadata(deployment):
    module=load();module.prepare(deployment)
    corpus=Path(deployment['canonical_root'])/'nature_corpus.csv'
    corpus.write_text(corpus.read_text()+'\n')
    with pytest.raises(ValueError,match='metadata.*changed|hash.*changed'):module.prepare(deployment)


def test_cli_training_batch_is_shared_with_preflight(deployment,tmp_path,monkeypatch):
    module=load();config=tmp_path/'config.json';config.write_text(json.dumps(deployment))
    seen={}
    def run(command,**kwargs):
        seen.update(command=command,**kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    assert module.main(['train','--config',str(config),'--batch-size','32'])==0
    assert seen['env']['E3_BATCH_SIZE']=='32'
    assert '--batch-size' not in seen['command']


def test_resume_after_partial_metadata_commit(deployment,monkeypatch):
    module=load();original=Path.replace;count=0
    def crash(path,target):
        nonlocal count
        if 'finalize' in str(path) and path.name.endswith('.csv'):
            count+=1
            if count==2:raise OSError('metadata power loss')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',crash)
    with pytest.raises(OSError,match='metadata power loss'):module.prepare(deployment)
    monkeypatch.setattr(Path,'replace',original)
    result=module.prepare(deployment)
    assert len(result['skipped_sources'])==5
    root=Path(deployment['canonical_root'])
    owner=json.loads((root/'.prepare/owner.json').read_text())
    assert not owner.get('pending_metadata')
    assert all(module.sha256(root/name)==digest for name,digest in owner['metadata'].items())
