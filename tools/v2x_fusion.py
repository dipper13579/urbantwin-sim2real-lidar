"""Adopted deterministic WBF and pedestrian geometry; no search code."""
import math
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from bg_prior import load
POINTS = None
@njit(cache=True)
def corners(b):
    a=np.empty((4,2)); c=math.cos(b[6]); s=math.sin(b[6])
    for i in range(4):
        x=b[3]*.5*(1 if i==0 or i==3 else -1)
        y=b[4]*.5*(1 if i<2 else -1)
        a[i,0]=b[0]+x*c-y*s; a[i,1]=b[1]+x*s+y*c
    return a

@njit(cache=True)
def area_inter(a,b):
    # Circumscribed-circle rejection followed by float64 polygon clipping.
    rr=.5*(math.hypot(a[3],a[4])+math.hypot(b[3],b[4]))
    if (a[0]-b[0])**2+(a[1]-b[1])**2>=rr*rr: return 0.
    p=np.empty((16,2)); q=np.empty((16,2)); ca=corners(a); cb=corners(b)
    p[:4]=ca; n=4
    for edge in range(4):
        x1,y1=cb[edge]; x2,y2=cb[(edge+1)%4]; dx=x2-x1; dy=y2-y1
        m=0
        for j in range(n):
            sx,sy=p[(j+n-1)%n]; ex,ey=p[j]
            cs=dx*(sy-y1)-dy*(sx-x1); ce=dx*(ey-y1)-dy*(ex-x1)
            ins=cs>=0; ine=ce>=0
            if ins!=ine:
                t=cs/(cs-ce); q[m,0]=sx+t*(ex-sx); q[m,1]=sy+t*(ey-sy); m+=1
            if ine: q[m,0]=ex; q[m,1]=ey; m+=1
        n=m
        if n<3: return 0.
        p[:n]=q[:n]
    ar=0.
    for j in range(n): ar+=p[j,0]*p[(j+1)%n,1]-p[j,1]*p[(j+1)%n,0]
    return abs(ar)*.5

@njit(cache=True)
def bev(a,b):
    inter=area_inter(a,b); union=a[3]*a[4]+b[3]*b[4]-inter
    return inter/union if union>0 else 0.

@njit(cache=True)
def cluster(a,thr):
    refs=np.empty((len(a),7)); radii=np.empty(len(a)); labels=np.empty(len(a),np.int64); nc=0
    for k in range(len(a)):
        r=.5*math.hypot(a[k,3],a[k,4]); found=-1
        for j in range(nc):
            dx=refs[j,0]-a[k,0]; dy=refs[j,1]-a[k,1]; rr=r+radii[j]
            if abs(dx)>=min(6.,rr) or abs(dy)>=min(6.,rr): continue
            d2=dx*dx+dy*dy
            if d2<36 and d2<rr*rr and bev(refs[j],a[k])>=thr:
                found=j; break
        if found<0:
            found=nc; refs[nc]=a[k,:7]; radii[nc]=r; nc+=1
        labels[k]=found
    return labels,nc

@njit(cache=True)
def aggregate(a,labels,nc,denom,nsources):
    sums=np.zeros(nc); bs=np.zeros((nc,7)); maxs=np.zeros((nc,nsources)); counts=np.zeros(nc,np.int64)
    for k in range(len(a)):
        j=labels[k]; s=a[k,7]
        if counts[j]==0: bs[j,6]=a[k,6]
        counts[j]+=1; sums[j]+=s
        for t in range(6): bs[j,t]+=a[k,t]*s
        src=int(a[k,8]); maxs[j,src]=max(maxs[j,src],s)
    out=np.empty((nc,8))
    for j in range(nc):
        for t in range(6): out[j,t]=bs[j,t]/sums[j]
        out[j,6]=bs[j,6]; out[j,7]=min(1.,sums[j]/denom)
    return out,maxs,counts

def fuse(a,denom,nsources=56):
    a=a[np.argsort(-a[:,7],kind='stable')]
    lab,n=cluster(a,.4); out,maxs,counts=aggregate(a,lab,n,denom,nsources)
    # Preserve NumPy's exact summation order from tools/ensemble.py. A sequential
    # compiled sum can differ at a 5-decimal half-round boundary, changing ties.
    ids=np.argsort(lab,kind='stable'); ends=np.cumsum(counts); start=0
    rows=[]
    for end in ends:
        members=a[ids[start:end]]; start=end
        s=members[:,7].copy(); B=members[:,:7].copy()
        box=list((B[:,:6]*s[:,None]).sum(0)/s.sum())+[float(B[0,6])]
        rows.append([round(float(v),4) for v in box]+[round(float(min(1.,s.sum()/denom)),5)])
    out=np.array(rows)
    ix=np.argsort(-out[:,7],kind='stable')[:8000]
    return out[ix],maxs[ix],counts[ix]

def refine_pp(item):
    fid,arr=item; pts=load(POINTS/f'{fid}.bin').astype(np.float64); tree=cKDTree(pts[:,:2]); out=[]
    for idx,row in enumerate(arr):
        b=row[:7].tolist(); c=np.array(b[:2]); zlo=b[2]-b[5]/2-.1; zhi=b[2]+b[5]/2+.1; ok=False
        for _ in range(2):
            p=pts[tree.query_ball_point(c,.5)]; p=p[(p[:,2]>zlo)&(p[:,2]<zhi)]
            if len(p)<5: break
            med=np.median(p[:,:2],0); u=med/max(np.linalg.norm(med),1e-6)
            c=.5*c+.5*(med+.08*u); ok=True
        if ok: b[:2]=[round(float(x),4) for x in c]
        h=b[5]; b[3:6]=[.66,.69,1.72]; b[2]+=(1.72-h)/2; b[2]=round(b[2]+.09,4)
        out.append({'box':b,'score':float(row[7]),'class':'pedestrian','audit_id':idx})
    return fid,sorted(out,key=lambda d:-d['score'])
