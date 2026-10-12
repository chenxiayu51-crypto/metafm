"""Compare fresh Windows spawn workers against serial real-data HDF5 slices."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Subset
from semg_fm.data.window_dataset import CanonicalWindowDataset
from semg_fm.data.raw_adapter import RawWindowDataset
from semg_fm.data.datamodule import task_homogeneous_collate


if __name__=='__main__':
    latest=json.loads((ROOT/'outputs/verification/latest.json').read_text())
    output=Path(latest['path']);legacy=ROOT.parent/'MetaEMGFM'
    results=[]
    for kind,cls,manifest,data_root in (
        ('canonical',CanonicalWindowDataset,legacy/'data/canonical/manifest.csv',legacy/'data/canonical'),
        ('raw',RawWindowDataset,output/'raw_wrist_angle_manifest.csv',legacy/'data/real/wrist'),
    ):
        dataset=cls.from_manifest(manifest,'train',200,100,root=data_root,task_id='wrist_angle',corpus_path=legacy/'data/real/wrist/wrist_corpus.csv')
        subset=Subset(dataset,[0,1,2,3])
        serial=list(DataLoader(subset,batch_size=2,num_workers=0,collate_fn=task_homogeneous_collate))
        spawned=list(DataLoader(subset,batch_size=2,num_workers=2,collate_fn=task_homogeneous_collate))
        for a,b in zip(serial,spawned):
            torch.testing.assert_close(a['emg'],b['emg'],rtol=0,atol=0)
            assert [x['sample_start'] for x in a['metadata']]==[x['sample_start'] for x in b['metadata']]
        results.append(dict(format=kind,status='PASS',workers=2,windows=4,exact_emg=True))
    (ROOT/'reports/windows_workers.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))
