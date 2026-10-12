"""Small auditable callbacks layered on native Lightning checkpointing."""
import json
import math
import time
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback


class TrainingDiagnostics(Callback):
    def __init__(self, output_dir='.'):
        self.output_dir=Path(output_dir);self.steps=[];self.initial=None;self.teacher_initial=None
    def on_fit_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.initial={k:v.detach().cpu().clone() for k,v in pl_module.student.state_dict().items()}
        self.teacher_initial={k:v.detach().cpu().clone() for k,v in pl_module.teacher.state_dict().items()}
        self.feature_std=0.0
        self.feature_hook=pl_module.student.register_forward_hook(lambda _m,_i,o:setattr(self,'feature_std',float(o['z_motion'].detach().float().std())))
        self.started=time.perf_counter()
        self.resume_report={}
        if trainer.ckpt_path:
            checkpoint=torch.load(trainer.ckpt_path,map_location='cpu',weights_only=False)
            self.resume_report=dict(source=str(trainer.ckpt_path),saved_step=checkpoint['global_step'],
                                   student_exact=all(torch.equal(v.detach().cpu(),checkpoint['state_dict']['student.'+k]) for k,v in pl_module.student.state_dict().items()),
                                   teacher_exact=all(torch.equal(v.detach().cpu(),checkpoint['state_dict']['teacher.'+k]) for k,v in pl_module.teacher.state_dict().items()),
                                   rng_saved='rng_state' in checkpoint)
    def on_train_start(self,trainer,pl_module):
        if trainer.ckpt_path:
            checkpoint=torch.load(trainer.ckpt_path,map_location='cpu',weights_only=False)
            self.resume_report.update(global_step=trainer.global_step,
                                      optimizer_exact=_equal_state(trainer.optimizers[0].state_dict(),checkpoint['optimizer_states'][0]),
                                      scheduler_exact=_equal_state(trainer.lr_scheduler_configs[0].scheduler.state_dict(),checkpoint['lr_schedulers'][0]))
    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if torch.cuda.is_available(): torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        self.batch_started=time.perf_counter()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        values=[p.grad.detach().float().norm() for p in pl_module.student.parameters() if p.grad is not None]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
        row=dict(step=int(trainer.global_step),student_gradient_norm=float(torch.stack(values).norm()) if values else 0.0,
                 feature_std=self.feature_std,loss=float(loss.detach()),
                 loss_latent=float(trainer.callback_metrics.get('train/latent_loss',torch.tensor(0.0))),
                 learning_rate=trainer.optimizers[0].param_groups[0]['lr'],
                 GPU_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                 step_time_s=time.perf_counter()-self.batch_started)
        if not all(math.isfinite(row[key]) for key in ('loss','feature_std','student_gradient_norm')): raise RuntimeError('nonfinite training diagnostics')
        self.steps.append(row)
        line=json.dumps(row)
        print(line,flush=True)
        with (self.output_dir/'train.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    def on_fit_end(self, trainer, pl_module):
        student_updated=any(not torch.equal(v,pl_module.student.state_dict()[k].detach().cpu()) for k,v in self.initial.items())
        teacher_updated=any(not torch.equal(v,pl_module.teacher.state_dict()[k].detach().cpu()) for k,v in self.teacher_initial.items())
        teacher_no_grads=all(p.grad is None for p in pl_module.teacher.parameters())
        report=dict(stage='smoke',steps=self.steps,student_updated=student_updated,teacher_updated=teacher_updated,teacher_has_no_grads=teacher_no_grads,
                    ema_updates=pl_module.ema_updates,global_step=trainer.global_step,elapsed_s=time.perf_counter()-self.started,
                    resume=self.resume_report,amp_status='NOT_RUN: precision 32',sample_sequence_reproducible=False)
        (self.output_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        trainer.save_checkpoint(self.output_dir/'checkpoint_last.ckpt')
        self.feature_hook.remove()
        if (self.output_dir/'resolved_config.yaml').exists():
            import shutil
            shutil.copy2(self.output_dir/'resolved_config.yaml',self.output_dir/'config.yaml')
        from .reproducibility import environment_info
        (self.output_dir/'environment.json').write_text(json.dumps(environment_info(),indent=2),encoding='utf-8')
        if (self.output_dir/'manifest.json').exists():
            import shutil
            shutil.copy2(self.output_dir/'manifest.json',self.output_dir/'data_manifest_hash.json')


def _equal_state(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(_equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(_equal_state(x,y) for x,y in zip(a,b))
    return a==b
