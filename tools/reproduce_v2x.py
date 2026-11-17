"""Reproduce H3 from cached per-member predictions and local point/prior inputs."""
import argparse, json, math, pickle
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from release_paths import WORK_ROOT, KIT_ROOT, REPO
import v2x_fusion as ev
import v2x_quality as quality
from bg_prior import load, box_shares, frame_keys, share_map
from static_prior import build as static_build
from vert_fit import zext, place
from ped_rules import features
CLASSES=['pedestrian','truck','vehicle']
YAW={'power':1,'blend':.25,'consensus':0.}
SUFFIXES=['','_fx','_fy','_fxy','_r5','_r-5','_r10','_r-10']
VEH=['V25','V21','V26','V27','V28','V30','V33','VR1']
PED=['V29','V30','V4','V5','V6','V7','V26','NEW_V']
TRUCK=['B_cp16_dens','V2','V12','V15','V22','V29','V36','R24_truck','R28_truck']
BASE=WORK_ROOT/'outputs/v2x/final'
POINTS=KIT_ROOT/'v2x/detection_test_frames'
SM={};BG={}
def read(p): return json.loads(Path(p).read_text())
def source(model,suffix,kind):
    if model=='R24_truck': return BASE/'t9/r24/truck/sources'/(model+suffix+'.json')
    if model=='R28_truck': return BASE/'t10/r28/truck/sources'/(model+suffix+'.json')
    if kind=='pedestrian':
        return BASE/('t9/r27a/sources' if model=='NEW_V' else 't4/sources')/(model+suffix+'.json')
    return BASE/'t3/sources'/(model+suffix+'.json')
def load_sources(order,ids):
    out=[]
    for model,suffix,kind in order:
        raw=read(source(model,suffix,kind))
        out.append({fid:raw[fid] for fid in ids})
    return out
def rebuild_maps(out):
    # Same 359 non-forbidden validation points / fixed historical prediction chain.
    pred=WORK_ROOT/'outputs/v2x/realval/ped30/ped_v3029rotR10/T_pp.json'
    sm,n=static_build([pred],{'pedestrian'},.5,.3)
    assert n==359
    paths=sorted((WORK_ROOT/'data/v2x_real/points').glob('*.bin'))
    assert len(paths)==359
    bg=share_map([frame_keys(p,.3) for p in paths])
    with Path(out).open('wb') as stream: pickle.dump({'SM':sm['pedestrian'],'BG':bg},stream,protocol=5)
def fuse_truck(a,iou,denominator):
    """Exact R27b/R27m arithmetic; only truck clustering threshold is variable."""
    out=[]
    for ci,cls in enumerate(CLASSES):
        rows=a[a[:,10]==ci]
        rows=rows[np.argsort(-rows[:,7],kind='stable')]
        if not len(rows):continue
        lab,n=ev.cluster(rows,iou if cls=='truck' else .4)
        order=np.argsort(lab,kind='stable')
        ends=np.cumsum(np.bincount(lab,minlength=n))
        start=0
        for end in ends:
            members=rows[order[start:end]]
            start=end
            scores=members[:,7].copy()
            boxes=members[:,:7].copy()
            angle=float(boxes[0,6])
            if cls=='truck':
                _,first=np.unique(members[:,11].astype(np.int64),return_index=True)
                support=members[np.sort(first)]
                weights=support[:,7]**YAW['power']
                cosine=np.sum(weights*np.cos(2*support[:,6]))/weights.sum()
                sine=np.sum(weights*np.sin(2*support[:,6]))/weights.sum()
                concentration=math.hypot(float(cosine),float(sine))
                if concentration>=YAW['consensus'] and concentration>=1e-9:
                    mean=.5*math.atan2(float(sine),float(cosine))
                    delta=.5*math.atan2(math.sin(2*(mean-angle)),math.cos(2*(mean-angle)))
                    angle+=YAW['blend']*delta
            box=list((boxes[:,:6]*scores[:,None]).sum(0)/scores.sum())+[angle]
            out.append({'box':[round(float(v),4) for v in box],
                        'score':round(float(min(1.,scores.sum()/denominator)),5),'class':cls})
    return sorted(out,key=lambda d:-d['score'])[:1000]
def fuse_general(a,denom,thr,cap):
    fused=[]
    for ci,cls in enumerate(['pedestrian','truck','vehicle']):
        z=a[a[:,10]==ci];z=z[np.argsort(-z[:,7],kind='stable')]
        if not len(z):continue
        lab,n=ev.cluster(z,thr);order=np.argsort(lab,kind='stable');ends=np.cumsum(np.bincount(lab,minlength=n));start=0
        for end in ends:
            m=z[order[start:end]];start=end;s=m[:,7].copy();b=m[:,:7].copy();box=list((b[:,:6]*s[:,None]).sum(0)/s.sum())+[float(b[0,6])]
            fused.append({'box':[round(float(v),4) for v in box],'score':round(float(min(1.,s.sum()/denom)),5),'class':cls})
    return sorted(fused,key=lambda d:-d['score'])[:cap]
def finish_direct(item,mults):
    fid,ds=item;scene=fid.split('_')[0];mk,ms=BG[scene]
    pts=load(POINTS/(fid+'.bin'));tree=cKDTree(pts[:,:2])
    for d in ds:
        cell=tuple(int(np.floor(x/.5)) for x in d['box'][:2])
        if SM[scene].get(cell,0)>=.5:d['score']*=mults[0]
    ds.sort(key=lambda d:-d['score'])
    for d,s in zip(ds,box_shares(pts,tree,ds,{'pedestrian'},mk,ms,.3)):
        if s is not None and s>=.4:d['score']*=mults[1]
    ds.sort(key=lambda d:-d['score'])
    for d in ds:
        ext=zext(pts,tree,d['box'],'pedestrian')
        if ext is not None:d['box']=place(d['box'],ext,'mid',.5)
        nb,top,_=features(pts,tree,d['box'])
        if nb<6:d['score']*=mults[2]
        if top<1.2:d['score']*=mults[3]
    ds.sort(key=lambda d:-d['score'])
    for d in ds:
        rg=math.hypot(*d['box'][:2])
        if rg<10:d['box'][2]+=-.5*(1-rg/10)
    return fid,ds
def fuse_rider(rows):
    if not rows:
        return []
    a = np.array(sorted(rows, key=lambda r: -r[7]))
    refs, members = [], []
    tree, pts = None, []
    for k in range(len(a)):
        j = -1
        if refs:
            d = np.hypot(np.array(pts)[:, 0] - a[k, 0], np.array(pts)[:, 1] - a[k, 1])
            i = int(np.argmin(d))
            if d[i] < 0.6:
                j = i
        if j < 0:
            refs.append(k); pts.append(a[k, :2]); members.append([k])
        else:
            members[j].append(k)
    out = []
    for r, m in zip(refs, members):
        M = a[m]
        s = M[:, 7]
        b = list((M[:, :6] * s[:, None]).sum(0) / s.sum()) + [float(a[r, 6])]
        out.append((b, min(1.0, float(s.sum()) / 8)))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--maps',type=Path);ap.add_argument('--rebuild-maps',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    maps_path=args.maps or args.out/'maps.pkl'
    if args.rebuild_maps or not maps_path.exists(): rebuild_maps(maps_path)
    with maps_path.open('rb') as stream: maps=pickle.load(stream)
    ids=read(KIT_ROOT/'v2x/detection_test_frame_ids.json')
    setting=read(REPO/'configs/v2x_pedestrian.json')
    ev.POINTS=POINTS
    vehicles={};trucks={};peds={}
    for cls,members in [('vehicle',VEH),('truck',TRUCK),('pedestrian',PED)]:
        if cls=='vehicle':order=[(m,s,cls) for m in members for s in SUFFIXES[:4]]
        elif cls=='truck':order=[(m,s,cls) for group in [SUFFIXES[:4],SUFFIXES[4:]] for m in members for s in group]
        else: order=[(m,s,cls) for s in SUFFIXES for m in members]
        runs=load_sources(order,ids)
        for ix,fid in enumerate(ids):
            points=load(POINTS/(fid+'.bin'));tree=cKDTree(points[:,:2])
            if cls=='pedestrian':
                canonical=['V4','V5','V6','V7','V30','V29','V26','NEW_V']
                packed=np.concatenate([np.array([d['box']+[d['score'],si,canonical.index(m)] for d in run[fid]],float).reshape(-1,10) for si,((m,_,_),run) in enumerate(zip(order,runs))])
                flat,ss,_=ev.fuse(packed,64,64);mm=np.zeros((len(flat),8))
                for si,(m,_,_) in enumerate(order):mm[:,canonical.index(m)]=np.maximum(mm[:,canonical.index(m)],ss[:,si])
                mm=mm.mean(1);scene=fid.split('_')[0];SM[scene]=maps['SM'];BG[scene]=maps['BG']
                _,pp=ev.refine_pp((fid,flat));_,post=finish_direct((fid,pp),setting['multipliers']);post=post[:800]
                pts64=points.astype(np.float64);tree64=cKDTree(pts64[:,:2])
                fs=np.array([quality.point_features(pts64,tree64,d['box']) for d in post])
                trans=[min(1.,d['score']*mm[d['audit_id']]/flat[d['audit_id'],7]) if flat[d['audit_id'],7] else 0. for d in post]
                scores=quality.apply_scores({'score':np.array([d['score'] for d in post]),'features':fs,'source_scores':{'modelmax':np.array(trans)}},setting['spec'])
                ds=[{'box':d['box'],'score':float(s),'class':'pedestrian'} for d,s in zip(post,scores)]
                ds.sort(key=lambda d:-d['score']);peds[fid]=ds
            else:
                packed=np.concatenate([np.array([d['box']+[d['score'],0,0,CLASSES.index(d['class']),si] for d in run[fid]],float).reshape(-1,12) for si,run in enumerate(runs)])
                ds=fuse_general(packed,32,.6,1000) if cls=='vehicle' else fuse_truck(packed,.6,72)
                result=[]
                for d in ds:
                    if d['class']!=cls:continue
                    b=d['box'];h=b[5]
                    if cls=='vehicle':
                        b[3:6]=[4.55,2.04,1.70];b[2]+=(1.70-h)/2
                        radius=max(np.hypot(b[0],b[1]),1e-6);fac=max(radius-.10,0)/radius
                        b[0],b[1]=round(b[0]*fac,4),round(b[1]*fac,4)
                    else:
                        b[3:6]=[round(b[3+i]*s,4) for i,s in enumerate([1.15,1.08,.95])];b[2]+=(b[5]-h)/2
                        ex=zext(points,tree,b,'truck')
                        if ex is not None:d['box']=place(b,ex,'fit',.5)
                    result.append(d)
                (vehicles if cls=='vehicle' else trucks)[fid]=result
            if (ix+1)%10==0:print(cls,ix+1,len(ids),flush=True)
        del runs
    riders=[read(BASE/'t7/sources'/('PR1_R14'+s+'.json')) for s in SUFFIXES]
    for fid in ids:
        rows=[d['box']+[d['score']] for src in riders for d in src[fid] if d['class']=='pedestrian' and d['score']>=.05 and d['box'][3]>=.9]
        peds[fid]+=[{'class':'pedestrian','box':[round(float(v),4) for v in b],'score':round(s*.3,5)} for b,s in fuse_rider(rows) if b[3]>=1.]
    # The original packer appends replacement classes in vehicle,pedestrian,truck order.
    result={fid:vehicles[fid]+peds[fid]+trucks[fid] for fid in ids}
    (args.out/'predictions.json').write_text(json.dumps(result))
if __name__=='__main__':main()
