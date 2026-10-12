"""Unified engineering acceptance; native Hydra entries own all training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from semg_fm.local_verification import VerificationRun


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/local/local_verify.yaml')
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    cfg=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True)
    for key in ('legacy_root','canonical_root','canonical_manifest','wrist_corpus','pose_raw_root','wrist_raw_root','output_root'):
        cfg[key]=str((ROOT/cfg[key]).resolve())
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=Path(cfg['output_root'])/f'local_{run_id}'
    run=VerificationRun(output)
    from semg_fm.verification_checks import check_environment,check_data,check_model,check_ssl,check_pose
    for check in (check_environment,check_data,check_model,check_ssl,check_pose):
        print(f'Running {check.__name__}',flush=True)
        try: check(run,cfg,ROOT,run_id)
        except Exception as error:
            import traceback
            print(traceback.format_exc(),flush=True)
            run.output.mkdir(parents=True,exist_ok=True)
            (run.output/(check.__name__+'_error.txt')).write_text(traceback.format_exc(),encoding='utf-8')
            modules={'check_environment':['ENVIRONMENT','DEPENDENCIES'],'check_data':['DATA_FORMAT','SPLIT_INTEGRITY','HDF5_CONSISTENCY'],
                     'check_model':['ENCODER_FORWARD','ENCODER_BACKWARD','RING_EQUIVARIANCE'],
                     'check_ssl':['EMA_TRAINING','CHECKPOINT_RESUME'],
                     'check_pose':['POSE_WEIGHT_LOADING','POSE_FORWARD','POSE_TRAINING','POSE_EVALUATION']}[check.__name__]
            for name in modules:
                if name not in run.statuses:run.record(name,'FAIL',dict(error=str(error)))
        run.write_summary()
    latest=Path(cfg['output_root'])/'latest.json'
    latest.write_text(json.dumps(dict(run_id=run_id,path=str(output),summary=str(output/'summary.md')),indent=2),encoding='utf-8')
    print(str(output),flush=True)
    return int(any(value['status']=='FAIL' for value in run.statuses.values()))


if __name__=='__main__':sys.exit(main())
