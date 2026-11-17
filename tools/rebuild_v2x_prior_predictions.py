"""Rebuild the adopted 359-frame pedestrian static-map input; no real data vendored."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from release_paths import REPO,WORK_ROOT
from reproduce_v2x import SUFFIXES,BASE
import v2x_fusion as ev
MEMBERS=['V3','V4','V5','V6','V7','V30','V29']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--cache-root',type=Path,default=WORK_ROOT/'map_members/v2x');ap.add_argument('--out',type=Path);a=ap.parse_args()
    rows=json.loads((REPO/'configs/members.json').read_text());ev.POINTS=WORK_ROOT/'data/v2x_real/points'
    ids=sorted(p.stem for p in ev.POINTS.glob('*.bin'));assert len(ids)==359
    order=[(m,s) for suffixes in [SUFFIXES[:4],SUFFIXES[4:]] for m in MEMBERS for s in suffixes]
    runs=[]
    for m,s in order:
        p=a.cache_root/(m+s+'.json')
        if a.infer:
            p.parent.mkdir(parents=True,exist_ok=True);row=rows['v2x/'+m]
            flags=['--rot',s[2:]] if s.startswith('_r') else (['--flip',{'_fx':'x','_fy':'y','_fxy':'xy'}[s]] if s else [])
            if m in ['V3','V4','V5','V6','V7']:flags+=['--zshift','-.5']
            raw=p.with_suffix('.raw.json')
            subprocess.run([sys.executable,str(REPO/'tools/predict.py'),'--cfg',str(REPO/row['config']),'--ckpt',str(WORK_ROOT/row['checkpoint']),'--frames',str(ev.POINTS),'--out',str(raw),'--score-thresh','.01','--max-per-frame','6000','--nms-post','1000','--nms-pre','4000','--topk','2000']+flags,check=True)
            ds=json.loads(raw.read_text());p.write_text(json.dumps({f:sorted([d for d in v if d['class']=='pedestrian' and d['score']>=.01],key=lambda d:-d['score'])[:400] for f,v in ds.items()}))
        runs.append(json.loads(p.read_text()))
    out={}
    for i,f in enumerate(ids):
        packed=np.concatenate([np.array([d['box']+[d['score'],si,0] for d in run[f]],float).reshape(-1,10) for si,run in enumerate(runs)])
        boxes,_,_=ev.fuse(packed,56,56);_,ds=ev.refine_pp((f,boxes))
        for d in ds:d.pop('audit_id')
        out[f]=ds
        if (i+1)%50==0:print(i+1,len(ids),flush=True)
    p=a.out or WORK_ROOT/'outputs/v2x/realval/ped30/ped_v3029rotR10/T_pp.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out))
if __name__=='__main__':main()
