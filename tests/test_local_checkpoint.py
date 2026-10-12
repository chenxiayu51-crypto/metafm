import random
import numpy as np
import torch
import pytest
from semg_fm.lightning.pretrain_module import PretrainModule
from semg_fm.lightning.downstream_module import DownstreamModule
from semg_fm.models import RingCrissCrossEncoder


def test_pretrain_rng_checkpoint_hooks_restore_all_generators():
    module=PretrainModule(model_config=dict(d_model=8,depth=1,heads=2,patch_samples=40))
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    checkpoint={}
    module.on_save_checkpoint(checkpoint)
    expected=(random.random(),np.random.rand(),torch.rand(3))
    module.on_load_checkpoint(checkpoint)
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)


def test_pose_strict_load_reports_exact_student_state(tmp_path):
    source=PretrainModule(model_config=dict(d_model=8,depth=1,heads=2,patch_samples=40))
    path=tmp_path/'ssl.ckpt';torch.save(dict(state_dict=source.state_dict(),hyper_parameters=dict(model_config=source.model_config)),path)
    pose=DownstreamModule(RingCrissCrossEncoder(d_model=8,depth=1,heads=2,patch_samples=40),task_id='pose',output_dim=20)
    report=pose.load_backbone_checkpoint(str(path))
    assert report['missing_keys']==[] and report['unexpected_keys']==[]
    for key,value in source.student.state_dict().items():
        torch.testing.assert_close(value,pose.backbone.state_dict()[key],rtol=0,atol=0)


def test_shape_compatible_attention_config_mismatch_is_rejected(tmp_path):
    source=PretrainModule(model_config=dict(d_model=8,depth=1,heads=2,patch_samples=40))
    path=tmp_path/'ssl.ckpt'
    torch.save(dict(state_dict=source.state_dict(),hyper_parameters=dict(model_config=source.model_config)),path)
    pose=DownstreamModule(RingCrissCrossEncoder(d_model=8,depth=1,heads=4,patch_samples=40),task_id='pose',output_dim=20)
    with pytest.raises(ValueError,match='configuration'):
        pose.load_backbone_checkpoint(str(path))


@pytest.mark.parametrize('changed', [dict(ema_decay=.5),dict(latent_weight=0),dict(mask_config={'temporal_span':1}),dict(max_steps=40)])
def test_resume_rejects_changed_training_semantics(changed):
    config=dict(d_model=8,depth=1,heads=2,patch_samples=40)
    source=PretrainModule(model_config=config)
    resumed=PretrainModule(model_config=config,**changed)
    with pytest.raises(ValueError,match='resume configuration'):
        resumed.on_load_checkpoint(dict(hyper_parameters=dict(source.hparams)))
