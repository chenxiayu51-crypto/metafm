"""Read-only data/model checks and subprocess orchestration of native entries."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .data.manifest import load_manifest
from .data.window_dataset import CanonicalWindowDataset
from .data.raw_adapter import RawWindowDataset
from .data.datamodule import task_homogeneous_collate
from .models import RingCrissCrossEncoder
from .lightning import PretrainModule
from .lightning.downstream_module import DownstreamModule


def _write(run,name,value):
    run.output.mkdir(parents=True,exist_ok=True)
    (run.output/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _command(run,root,args,log):
    with (run.output/log).open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,*args],cwd=root,stdout=f,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        raise RuntimeError(f'Native command failed ({result.returncode}); see {run.output/log}')


def check_environment(run,cfg,root,run_id):
    from .reproducibility import environment_info
    env=environment_info()
    env.update(executable=sys.executable,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               total_vram_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    deps={name:importlib.metadata.version(name) for name in ('torch','lightning','hydra-core','h5py','pandas','numpy','pytest','tables')}
    env['dependencies']=deps
    _write(run,'environment.json',env)
    run.record('ENVIRONMENT','PASS' if env['cuda_available'] else 'BLOCKED',env)
    run.record('DEPENDENCIES','PASS',deps)


def check_data(run,cfg,root,run_id):
    frame=load_manifest(cfg['canonical_manifest'],cfg['canonical_root'])
    records=[];raw_rows={'pose':[],'wrist_angle':[]}
    for row in frame.to_dict('records'):
        task=row['task_id']
        if task not in raw_rows:continue
        raw_root=Path(cfg['pose_raw_root'] if task=='pose' else cfg['wrist_raw_root'])
        raw_path=raw_root/row['source_file']
        if not raw_path.is_file():raise FileNotFoundError(raw_path)
        assert _hash(raw_path)==row['source_sha256'],f'Source hash mismatch {raw_path}'
        raw_row={k:v for k,v in row.items() if k!='resolved_path'}
        raw_row['path']=row['source_file'];raw_rows[task].append(raw_row)
        with h5py.File(raw_path) as raw,h5py.File(row['resolved_path']) as canonical:
            source=raw['emg2pose/timeseries'] if task=='pose' else raw['data']
            origin=float(source['time'][0]);count=len(source)
            assert count==int(row['sample_count'])==len(canonical['signal/emg'])
            for start in range(0,count,65536):
                values=source[start:start+65536];end=start+len(values)
                np.testing.assert_array_equal(values['emg'].astype('f4'),canonical['signal/emg'][start:end])
                np.testing.assert_array_equal(values['time'].astype('f8')-origin,canonical['signal/time_s'][start:end])
                if task=='pose':
                    np.testing.assert_array_equal(values['joint_angles'],canonical['labels/joint_angles'][start:end])
                    if 'no_ik_failure' in source.dtype.names:
                        np.testing.assert_array_equal(values['no_ik_failure'],canonical['labels/no_ik_failure'][start:end])
                else:
                    np.testing.assert_array_equal(values['wrist_angles'],canonical['labels/targets/wrist_angles'][start:end])
            if task=='pose':
                official=raw['emg2pose'].attrs['split'];official=official.decode() if isinstance(official,bytes) else str(official)
                assert official==row['official_split']==canonical.attrs['official_split']
                label_dim=source.dtype.fields['joint_angles'][0].shape[-1]
                assert label_dim==20
            else:official='corpus_intervals'
        records.append(dict(task=task,source=str(raw_path),samples=count,official_split=official,
                            exact_signal=True,exact_relative_time=True,exact_labels=True,source_hash_match=True))
    for task,rows in raw_rows.items():
        if rows:pd.DataFrame(rows).to_csv(run.output/f'raw_{task}_manifest.csv',index=False)
    if not raw_rows['pose'] or not raw_rows['wrist_angle']:raise ValueError('Missing local real Pose/Wrist')
    pose=CanonicalWindowDataset.from_manifest(cfg['canonical_manifest'],'val',cfg['window_ms'],cfg['stride_ms'],root=cfg['canonical_root'],task_id='pose')
    wrist=CanonicalWindowDataset.from_manifest(cfg['canonical_manifest'],'train',cfg['window_ms'],cfg['stride_ms'],root=cfg['canonical_root'],task_id='wrist_angle',corpus_path=cfg['wrist_corpus'])
    raw_wrist=RawWindowDataset.from_manifest(run.output/'raw_wrist_angle_manifest.csv','train',cfg['window_ms'],cfg['stride_ms'],root=cfg['wrist_raw_root'],task_id='wrist_angle',corpus_path=cfg['wrist_corpus'])
    raw_pose=RawWindowDataset.from_manifest(run.output/'raw_pose_manifest.csv','val',cfg['window_ms'],cfg['stride_ms'],root=cfg['pose_raw_root'],task_id='pose')
    assert len(wrist)==len(raw_wrist) and len(pose)==len(raw_pose)
    assert [(r['source_file'],r['sample_start'],r['sample_end']) for r in wrist._record_rows]==[(r['source_file'],r['sample_start'],r['sample_end']) for r in raw_wrist._record_rows]
    # Representative windows from every source/interval; entire signals/labels
    # were checked above. No cross-process HDF5 handles retained.
    for a,b in ((pose,raw_pose),(wrist,raw_wrist)):
        for offset in a._index.offsets[:-1]:
            i=int(offset)
            if i>=len(a):continue
            left,right=a[i],b[i]
            torch.testing.assert_close(left['emg'],right['emg'],rtol=0,atol=0)
            if torch.is_tensor(left['target']):
                torch.testing.assert_close(left['target'],right['target'],rtol=0,atol=0)
                assert torch.equal(left['target_mask'],right['target_mask'])
            else:
                for name,value in left['target'].items():
                    torch.testing.assert_close(value,right['target'][name],rtol=0,atol=0)
                    assert torch.equal(left['target_mask'][name],right['target_mask'][name])
    long_wrist=CanonicalWindowDataset.from_manifest(cfg['canonical_manifest'],'train',2000,1000,root=cfg['canonical_root'],task_id='wrist_angle',corpus_path=cfg['wrist_corpus'])
    split=dict(pose_official_counts=frame[frame.task_id=='pose'].official_split.value_counts().to_dict(),
               wrist_train_intervals=len(wrist._record_rows),wrist_windows=len(wrist),wrist_2s_windows=len(long_wrist),
               exact_raw_canonical_bounds=True,half_open=True,corpus_sha256=_hash(cfg['wrist_corpus']),
               official_pose_train_found=bool((frame[frame.task_id=='pose'].official_split=='train').any()))
    _write(run,'data_report.json',dict(real=records,synthetic_tasks=['typing','discrete_gesture','handwriting'],
                                       note='Additional tasks are tested with upstream fixtures, not accepted as real data.'))
    _write(run,'split_report.json',split)
    run.record('DATA_FORMAT','PASS',dict(real_pose=len(raw_rows['pose']),real_wrist=len(raw_rows['wrist_angle']),input=[int(cfg['window_ms']*2),16]))
    run.record('SPLIT_INTEGRITY','PASS',split)
    run.record('HDF5_CONSISTENCY','PASS',dict(files=len(records),full_signal_and_label_comparison=True))


def check_model(run,cfg,root,run_id):
    if not torch.cuda.is_available():
        for name in ('ENCODER_FORWARD','ENCODER_BACKWARD','RING_EQUIVARIANCE'):run.record(name,'BLOCKED',dict(reason='CUDA unavailable'))
        return
    model=RingCrissCrossEncoder(**cfg['model']).cuda();model.eval()
    shapes=[]
    for samples in (400,4000):
        x=torch.randn(2,samples,16,device='cuda')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        output=model(x);torch.cuda.synchronize();forward=time.perf_counter()-start
        start=time.perf_counter();output['z_eq'].square().mean().backward();torch.cuda.synchronize();backward=time.perf_counter()-start
        assert all(torch.isfinite(value).all() for value in output.values())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
        shapes.append(dict(input=list(x.shape),output={key:list(value.shape) for key,value in output.items()},
                           forward_s=forward,backward_s=backward,peak_memory_bytes=torch.cuda.max_memory_allocated(),
                           finite=True,feature_std=float(output['z_motion'].detach().std())))
        model.zero_grad(set_to_none=True)
    x=torch.randn(2,400,16,device='cuda')
    with torch.no_grad():
        a=model(x);b=model(torch.roll(x,1,-1))
        eq=float((torch.roll(a['z_eq'],1,1)-b['z_eq']).abs().max())
        motion=float((a['z_motion']-b['z_motion']).abs().max())
    assert eq<1e-4 and motion<1e-4
    loss_module=PretrainModule(model_config=cfg['model'],warmup_steps=0,max_steps=20).cuda()
    batch=dict(emg=x,sample_mask=torch.ones(2,400,dtype=torch.bool,device='cuda'),task_id='wrist_angle')
    total,losses=loss_module._losses(batch)
    assert not losses['ring_equivariance_error'].requires_grad
    torch.testing.assert_close(total,loss_module.latent_weight*losses['latent']+loss_module.kinematic_weight*losses['kinematic'])
    report=dict(parameters=sum(p.numel() for p in model.parameters()),architecture=cfg['model'],measurements=shapes,
                ring_equivariance_max_abs=eq,motion_invariance_max_abs=motion,
                loss_audit=dict(ring_diagnostic_no_grad=True,ring_weight_optimized=False,total='latent_weight * latent + kinematic_weight * kinematic'),
                real_electrode_rotation_robustness='NOT_RUN: electrode numbering/hand conventions unverified')
    _write(run,'tensor_shapes.json',shapes);_write(run,'model_report.json',report)
    run.record('ENCODER_FORWARD','PASS',report);run.record('ENCODER_BACKWARD','PASS',dict(nonzero_gradient=True))
    run.record('RING_EQUIVARIANCE','PASS',dict(eq_max_abs=eq,motion_max_abs=motion,scope='Artificial cyclic permutation only'))
    del model,loss_module,x;torch.cuda.empty_cache()


def check_ssl(run,cfg,root,run_id):
    if not torch.cuda.is_available():
        run.record('EMA_TRAINING','BLOCKED',dict(reason='CUDA unavailable'));run.record('CHECKPOINT_RESUME','NOT_RUN',{});return
    first=root/'outputs/pretrain'/f'smoke_wrist_{run_id}'/'first'
    resumed=first.parent/'resumed'
    base=['pretrain.py','data=local','trainer=local_gpu','+pretrain=local_wrist',
          f'data.root={cfg["canonical_root"]}',f'data.manifest_path={cfg["canonical_manifest"]}',f'data.corpus_path={cfg["wrist_corpus"]}',
          f'data.window_ms={cfg["window_ms"]}',f'data.stride_ms={cfg["stride_ms"]}',f'data.batch_size={cfg["batch_size"]}',
          f'seed={cfg["seed"]}',f'optimizer.max_steps={cfg["total_steps"]}']
    for key,value in cfg['model'].items():base.append(f'model.{key}={value}')
    _command(run,root,[*base,f'trainer.max_steps={cfg["first_steps"]}',f'hydra.run.dir={first.as_posix()}'],'ssl_first_process.log')
    checkpoint=first/'checkpoint_last.ckpt'
    _command(run,root,[*base,f'trainer.max_steps={cfg["total_steps"]}',f'resume_from={checkpoint.as_posix()}',f'hydra.run.dir={resumed.as_posix()}'],'ssl_resume_process.log')
    a=json.loads((first/'metrics.json').read_text(encoding='utf-8'));b=json.loads((resumed/'metrics.json').read_text(encoding='utf-8'))
    assert a['global_step']==cfg['first_steps'] and b['global_step']==cfg['total_steps'] and b['ema_updates']==cfg['total_steps']
    for report in (a,b):
        assert report['student_updated'] and report['teacher_updated'] and report['teacher_has_no_grads']
        assert all(math.isfinite(r['loss']) and r['student_gradient_norm']>0 and r['feature_std']>0 for r in report['steps'])
    resume=b['resume']
    assert all(resume[key] for key in ('student_exact','teacher_exact','optimizer_exact','scheduler_exact','rng_saved'))
    ssl=dict(stage='SMOKE TEST',batch_size=cfg['batch_size'],window_ms=cfg['window_ms'],first=a,resumed=b,
             checkpoint=str(resumed/'checkpoint_last.ckpt'),checkpoint_sha256=_hash(resumed/'checkpoint_last.ckpt'))
    _write(run,'ssl_report.json',ssl);_write(run,'checkpoint_report.json',resume)
    run.ssl_checkpoint=resumed/'checkpoint_last.ckpt'
    run.record('EMA_TRAINING','PASS',dict(steps=cfg['total_steps'],batch_size=cfg['batch_size'],last_loss=b['steps'][-1]['loss'],checkpoint=ssl['checkpoint']))
    run.record('CHECKPOINT_RESUME','PASS',dict(**resume,amp='NOT_RUN: precision32',sample_sequence_reproducible=False))


def check_pose(run,cfg,root,run_id):
    from .pose_verification import verify_pose
    verify_pose(run,cfg,root,run_id)
