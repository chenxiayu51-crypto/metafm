"""Read-only inventory of research directories, including inaccessible paths."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(roots, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, checkpoints, errors = [], [], []
    for root in roots:
        if not root.exists():
            errors.append(dict(path=str(root), error='NOT_FOUND'))
            continue
        direct = {}
        texts = []
        for current, dirs, files in os.walk(root, onerror=lambda e: errors.append(dict(path=e.filename,error=str(e)))):
            current = Path(current)
            rel = current.relative_to(root)
            stat = dict(files=0, bytes=0, latest=0, types=Counter(), h5=0, ckpt=0)
            for name in files:
                path = current/name
                try:
                    info = path.stat()
                    stat['files'] += 1
                    stat['bytes'] += info.st_size
                    stat['latest'] = max(stat['latest'], info.st_mtime)
                    stat['types'][path.suffix.lower() or '<none>'] += 1
                    stat['h5'] += int(path.suffix.lower() in ('.h5','.hdf5'))
                    if path.suffix in ('.py','.yaml','.json','.md','.toml','.ps1') and info.st_size<2_000_000 and not set(rel.parts)&{'.venv','.git','outputs','data','references','build'}:
                        texts.append((str(path),path.read_text(encoding='utf-8',errors='replace')))
                    if path.suffix in ('.pt','.ckpt') and not set(rel.parts)&{'.venv','build'}:
                        stat['ckpt'] += 1
                        item=dict(path=str(path),size=info.st_size,sha256=sha256(path))
                        try:
                            import torch
                            saved=torch.load(path,map_location='cpu',weights_only=False)
                            keys=list(saved) if isinstance(saved,dict) else []
                            item.update(keys=keys,optimizer=any('optimizer' in k for k in keys),
                                        scheduler=any('scheduler' in k for k in keys),
                                        ema=any('teacher' in k for k in keys) or any(k.startswith('teacher.') for k in saved.get('state_dict',{})),
                                        step=saved.get('global_step',saved.get('step')),stage=saved.get('stage'))
                        except Exception as e:
                            item['error']=str(e)
                        checkpoints.append(item)
                except OSError as e:
                    errors.append(dict(path=str(path),error=str(e)))
            direct[current]=stat
        hashes=Counter(r['sha256'] for r in checkpoints)
        for current in sorted(direct,key=lambda p:len(p.parts),reverse=True):
            if current!=root and current.parent in direct:
                a,b=direct[current.parent],direct[current]
                for k in ('files','bytes','h5','ckpt'): a[k]+=b[k]
                a['latest']=max(a['latest'],b['latest'])
                a['types'].update(b['types'])
        for current,stat in direct.items():
            rel=current.relative_to(root)
            if len(rel.parts)>1 and rel.parts[0] in ('.venv','.git'):
                continue
            normalized=rel.as_posix()
            references=[path for path,text in texts if normalized!='.' and normalized in text.replace('\\','/')]
            contained=[item for item in checkpoints if Path(item['path']).is_relative_to(current)]
            denied=any(Path(e['path']).is_relative_to(current) for e in errors if e.get('path'))
            category='UNKNOWN'
            if rel.parts and rel.parts[0] in ('data','.git','configs','semg_fm','semg_h5','meta_emg','tests','scripts','docs','reports'): category='KEEP'
            if rel.parts and rel.parts[0]=='outputs':
                category='ACTIVE' if references else 'ARCHIVE' if contained else 'CLEANABLE' if not stat['h5'] else 'UNKNOWN'
            if denied: category='UNKNOWN'
            rows.append(dict(project=str(root),path=normalized,file_count=stat['files'],total_bytes=stat['bytes'],
                             last_modified=datetime.fromtimestamp(stat['latest'],timezone.utc).isoformat() if stat['latest'] else '',
                             file_types=json.dumps(dict(stat['types']),sort_keys=True),contains_hdf5=bool(stat['h5']),
                             contains_checkpoint=bool(contained),contains_unique_checkpoint=any(hashes[c['sha256']]==1 for c in contained),
                             optimizer_state=any(c.get('optimizer') for c in contained),ema_state=any(c.get('ema') for c in contained),
                             metric_files=any(k in stat['types'] for k in ('.json','.csv')),
                             referenced_by=json.dumps(references),regenerable='candidate' if category=='CLEANABLE' else 'not_established',
                             classification=category,incomplete_scan=denied))
    if rows:
        with (output/'file_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'inventory_details.json').write_text(json.dumps(dict(checkpoints=checkpoints,errors=errors),indent=2),encoding='utf-8')
    roots_rows=[r for r in rows if r['path']=='.']
    print(json.dumps(dict(totals=roots_rows,checkpoint_files=len(checkpoints),scan_errors=len(errors)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    inventory([Path(p).resolve() for p in args.roots],Path(args.output).resolve())
