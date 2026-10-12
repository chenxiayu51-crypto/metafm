"""Status-first report writer for local engineering acceptance."""
import json
from pathlib import Path

STATES={'PASS','FAIL','BLOCKED','NOT_RUN'}
class VerificationRun:
    def __init__(self,output): self.output=Path(output);self.statuses={}
    def record(self,name,status,details):
        if name not in {'ENVIRONMENT','DEPENDENCIES','DATA_FORMAT','SPLIT_INTEGRITY','HDF5_CONSISTENCY','ENCODER_FORWARD','ENCODER_BACKWARD','RING_EQUIVARIANCE','EMA_TRAINING','CHECKPOINT_RESUME','POSE_WEIGHT_LOADING','POSE_FORWARD','POSE_TRAINING','POSE_EVALUATION'}: raise ValueError('invalid verification module')
        if status not in STATES: raise ValueError('status must be PASS, FAIL, BLOCKED, or NOT_RUN')
        self.statuses[name]={'status':status,'details':details}
    def write_summary(self):
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'summary.md').write_text('\n'.join(['# Local verification','',*['- **%s**: %s — %s'%(k,v['status'],json.dumps(v['details'],ensure_ascii=False)) for k,v in self.statuses.items()]]),encoding='utf-8')
        (self.output/'summary.json').write_text(json.dumps(self.statuses,indent=2,ensure_ascii=False),encoding='utf-8')
        for key,value in self.statuses.items(): (self.output/(key.lower()+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
