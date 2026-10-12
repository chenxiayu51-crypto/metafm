"""Pose acceptance using the upstream head, trainer and evaluation entry.

Real official validation data is read only. Synthetic runs test optimizer and
checkpoint mechanisms and never constitute a real supervised experiment.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data.window_dataset import CanonicalWindowDataset
from .data.datamodule import task_homogeneous_collate
from .models import RingCrissCrossEncoder
from .lightning.downstream_module import DownstreamModule
from .verification_checks import _command, _write


def make_synthetic_pose(root):
    """Write three explicit, disjoint SYNTHETIC TEST canonical recordings."""
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42); rows=[]
    for split in ('train','val','test'):
        n=1200; path=root/f'synthetic_{split}.h5'
        row=dict(path=path.name,dataset_id='synthetic_pose',task_id='pose',
                 subject_id=f'synthetic_{split}',session_id=split,side='left',pair_id='',
                 official_split=split,sample_count=n,duration_s=(n-1)/2000,
                 sample_rate_hz=2000,source_file=path.name,source_sha256='0'*64)
        with h5py.File(path,'w') as f:
            f.attrs.update({key:row[key] for key in ('dataset_id','task_id','subject_id','session_id','side','pair_id','official_split','sample_rate_hz','source_file')})
            f.attrs['schema_version']='1.0'; f.attrs['synthetic']=True
            f['signal/emg']=rng.standard_normal((n,16)).astype('f4')
            f['signal/time_s']=np.arange(n,dtype='f8')/2000
            f['labels/joint_angles']=rng.standard_normal((n,20)).astype('f4')
            mask=np.ones(n,dtype=bool);mask[50:75]=False
            f['labels/no_ik_failure']=mask
            f['meta/source_json']=json.dumps(dict(synthetic=True,source_time_origin_s=0))
        rows.append(row)
    manifest=root/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    return manifest


def verify_pose(run,cfg,root,run_id):
    checkpoint=getattr(run,'ssl_checkpoint',None)
    if checkpoint is None:
        run.record('POSE_WEIGHT_LOADING','BLOCKED',dict(reason='Fresh SSL checkpoint unavailable'))
        run.record('POSE_FORWARD','NOT_RUN',{})
        run.record('POSE_TRAINING','BLOCKED',dict(reason='Missing fresh SSL checkpoint; no official Pose train'))
        run.record('POSE_EVALUATION','NOT_RUN',{})
        return
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    module=DownstreamModule(RingCrissCrossEncoder(**cfg['model']),task_id='pose',output_dim=20)
    report=module.load_backbone_checkpoint(str(checkpoint))
    assert report['configuration_checked'] and not report['missing_keys'] and not report['unexpected_keys']
    for key,value in module.backbone.state_dict().items():
        torch.testing.assert_close(value,source['state_dict']['student.'+key],rtol=0,atol=0)
    run.record('POSE_WEIGHT_LOADING','PASS',report)
    ds=CanonicalWindowDataset.from_manifest(cfg['canonical_manifest'],'val',cfg['window_ms'],cfg['stride_ms'],root=cfg['canonical_root'],task_id='pose')
    items=[ds[0],ds[1]];batch=task_homogeneous_collate(items)
    module.eval()
    with torch.no_grad():
        loss,metrics,encoded=module._loss_and_metrics(batch)
        prediction=module.head(encoded['features'])
    assert prediction.shape[-1]==20 and torch.isfinite(loss)
    forward=dict(input=list(batch['emg'].shape),features=list(encoded['features'].shape),
                 head=list(prediction.shape),target=list(batch['target'].shape),loss=float(loss),
                 metrics={k:float(v) for k,v in metrics.items()},split='val',training=False,
                 time_alignment='Same sample indices for EMG/labels/mask; linear interpolation token predictions to target length, align_corners=False',
                 units='Source angles preserved without scaling; radian convention requires confirmation from official metadata',
                 valid_windows=len(ds))
    run.record('POSE_FORWARD','PASS',forward)
    _write(run,'pose_forward.json',forward)
    del module
    synthetic_root=run.output/'synthetic_pose';manifest=make_synthetic_pose(synthetic_root)
    runs={}
    for mode in ('probe','partial','full'):
        # Gradient/freeze and parameter update checks use synthetic labels only.
        module=DownstreamModule(RingCrissCrossEncoder(**cfg['model']),task_id='pose',output_dim=20,finetune_mode=mode,unfrozen_blocks=1)
        module.load_backbone_checkpoint(str(checkpoint))
        synthetic_ds=CanonicalWindowDataset.from_manifest(manifest,'train',200,100,root=synthetic_root,task_id='pose')
        synthetic_batch=task_homogeneous_collate([synthetic_ds[0],synthetic_ds[1]])
        before={name:p.detach().clone() for name,p in module.named_parameters()}
        optimizer=module.configure_optimizers()
        synthetic_loss,_,_=module._loss_and_metrics(synthetic_batch);synthetic_loss.backward()
        frozen=[name for name,p in module.named_parameters() if not p.requires_grad]
        assert all(p.grad is None for name,p in module.named_parameters() if name in frozen)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.head.parameters())
        optimizer.step()
        assert all(torch.equal(before[name],p) for name,p in module.named_parameters() if name in frozen)
        assert any(not torch.equal(before['head.'+name],p) for name,p in module.head.named_parameters())
        if mode!='probe':
            assert any(not torch.equal(before['backbone.'+name],p) for name,p in module.backbone.named_parameters() if p.requires_grad)
        checks=dict(synthetic_loss=float(synthetic_loss.detach()),frozen_parameters=len(frozen),frozen_unchanged=True,head_updated=True)
        del module,optimizer;torch.cuda.empty_cache()
        output=root/'outputs/finetune'/f'synthetic_pose_{run_id}'/mode
        base=['finetune.py','data=local','trainer=local_finetune',f'finetune_mode={mode}',
              f'data.root={synthetic_root.as_posix()}',f'data.manifest_path={manifest.as_posix()}',
              'data.corpus_path=null','data.task_id=pose','data.batch_size=2',
              f'pretrained_checkpoint={checkpoint.as_posix()}',f'hydra.run.dir={output.as_posix()}']
        base.extend(f'model.{k}={v}' for k,v in cfg['model'].items())
        _command(run,root,base,f'pose_synthetic_{mode}.log')
        saved=output/'last.ckpt';state=torch.load(saved,map_location='cpu',weights_only=False)
        assert state['global_step']==2
        restored=DownstreamModule(RingCrissCrossEncoder(**cfg['model']),task_id='pose',output_dim=20,finetune_mode=mode,unfrozen_blocks=1)
        loaded=restored.load_state_dict(state['state_dict'],strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        for key,value in restored.state_dict().items():
            torch.testing.assert_close(value,state['state_dict'][key],rtol=0,atol=0)
        runs[mode]=dict(**checks,checkpoint=str(saved),global_step=2,save_load_exact=True)
    real_train=bool((pd.read_csv(cfg['canonical_manifest'],dtype=str).query("task_id == 'pose'").official_split=='train').any())
    run.record('POSE_TRAINING','NOT_RUN' if real_train else 'BLOCKED',
               dict(reason='Official Pose train found; real supervised training outside smoke budget' if real_train else 'Only official Pose val exists; supervised training forbidden',
                    synthetic_test=dict(status='PASS',protocols=runs),real_training=False))
    # Native evaluator deliberately selects official val without relabeling it.
    output=root/'outputs/evaluation'/f'pose_interface_{run_id}'
    args=['evaluate.py','data=local','trainer=local_finetune','data.task_id=pose',
          f'data.root={cfg["canonical_root"]}',f'data.manifest_path={cfg["canonical_manifest"]}',
          'data.corpus_path=null','+data.evaluation_split=val',f'checkpoint_path={runs["full"]["checkpoint"]}',
          'evaluation_scope=synthetic_checkpoint_real_val_interface_only',f'hydra.run.dir={output.as_posix()}']
    args.extend(f'model.{k}={v}' for k,v in cfg['model'].items())
    _command(run,root,args,'pose_evaluation.log')
    evaluation=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    assert evaluation and all(np.isfinite(float(value)) for value in evaluation.values())
    run.record('POSE_EVALUATION','PASS',dict(scope='Interface only: synthetic-trained checkpoint, real official val, two batches',
               independent_performance=False,metrics=evaluation,output=str(output)))
    _write(run,'pose_report.json',dict(weight_loading=report,forward=forward,synthetic_test=runs,
            real_training=False,independent_performance=False,evaluation_interface=evaluation))
