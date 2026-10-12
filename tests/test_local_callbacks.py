import json
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from semg_fm.lightning.pretrain_module import PretrainModule


def test_diagnostics_observes_gradients_and_persists_native_checkpoint(tmp_path):
    from semg_fm.local_callbacks import TrainingDiagnostics
    module=PretrainModule(model_config=dict(d_model=8,depth=1,heads=2,patch_samples=40),warmup_steps=0,max_steps=1)
    batch=dict(emg=torch.randn(2,400,16),sample_mask=torch.ones(2,400,dtype=torch.bool),task_id='wrist_angle')
    trainer=Trainer(accelerator='cpu',max_steps=1,logger=False,enable_checkpointing=False,
                    enable_progress_bar=False,enable_model_summary=False,callbacks=[TrainingDiagnostics(tmp_path)])
    trainer.fit(module,train_dataloaders=DataLoader([batch],batch_size=None))
    report=json.loads((tmp_path/'metrics.json').read_text())
    assert report['steps'][0]['student_gradient_norm']>0
    assert report['steps'][0]['feature_std']>0
    assert report['student_updated'] and report['teacher_updated'] and report['teacher_has_no_grads']
    ckpt=torch.load(tmp_path/'checkpoint_last.ckpt',weights_only=False)
    assert ckpt['global_step']==1 and ckpt['optimizer_states'] and ckpt['lr_schedulers'] and ckpt['rng_state']
