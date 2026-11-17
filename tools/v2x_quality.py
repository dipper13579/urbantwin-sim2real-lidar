import math
import numpy as np
NAMES=['height_quality','position_quality','support_quality','ground_quality','combined_quality','n_vertical','n_body_wide','n_body_inside','vertical_span','vertical_midpoint_offset','xy_offset','inside_fraction','occupied_height_bands','ground_valid','ground_gap','range','ground_samples','vertical_raw_quality','position_raw_quality','support_raw_quality','ground_raw_quality']
def confidence(n):
    return float(np.clip((n-4)/16,0,1))

def point_features(pts,tree,b):
    x,y,z,dx,dy,h,yaw=b; bottom=z-h/2; top=z+h/2
    radius=.5*np.hypot(dx,dy)
    q=pts[tree.query_ball_point([x,y],float(radius+1.2))]
    row=np.zeros(len(NAMES)); row[:5]=1.; row[15]=math.hypot(x,y)
    if not len(q): return row
    ex,ey=q[:,0]-x,q[:,1]-y; c,s=math.cos(yaw),math.sin(yaw)
    u,v=ex*c+ey*s,-ex*s+ey*c; dist=np.hypot(ex,ey)
    ring=q[(dist>radius+.3)&(np.abs(q[:,2]-bottom)<=.6),2]
    row[16]=len(ring)
    ground=None; qg=1.; eg=1.
    if len(ring)>=10:
        q20,q25,q75=np.percentile(ring,[20,25,75])
        if q75-q25<=.25:
            ground=float(q20); gap=bottom-ground
            qg=math.exp(-.5*(max(abs(gap)-.15,0)/.3)**2)
            eg=1.-min(1.,len(ring)/30)*(1.-qg)
            row[13:15]=[1,gap]
    vertical=q[(np.abs(u)<=dx/2+.1)&(np.abs(v)<=dy/2+.1)&(q[:,2]>bottom-.6)&(q[:,2]<top+.6),2]
    qz=1.; ez=1.; row[5]=len(vertical)
    if len(vertical)>=5:
        lo,hi=np.percentile(vertical,[2,98])
        inter=max(0.,min(hi,top)-max(lo,bottom)); union=max(hi,top)-min(lo,bottom)
        qz=inter/max(union,1e-6); ez=1-confidence(len(vertical))*(1-qz)
        row[8:10]=[hi-lo,(hi+lo)/2-z]
    body_floor=max(bottom+.25,ground+.25) if ground is not None else bottom+.25
    bm=(np.abs(u)<=dx/2+.25)&(np.abs(v)<=dy/2+.25)&(q[:,2]>body_floor)&(q[:,2]<top+.25)
    body=q[bm]; nb=len(body); row[6]=nb
    qxy=1.; exy=1.; qs=1.; es=1.
    if nb>=5:
        med=np.median(body[:,:2],axis=0)
        ray=np.array([x,y])/max(math.hypot(x,y),1e-6)
        off=float(np.linalg.norm(med+.08*ray-np.array([x,y])))
        qxy=math.exp(-.5*(max(off-.1,0)/.25)**2)
        exy=1-confidence(nb)*(1-qxy)
        inside=(np.abs(u[bm])<=dx/2+.05)&(np.abs(v[bm])<=dy/2+.05)&(body[:,2]<top)
        ni=int(inside.sum()); frac=ni/nb
        zinside=body[inside,2]
        bands=len(np.unique(np.clip(((zinside-(bottom+.2))/(h-.2)*4).astype(int),0,3)))
        qs=math.sqrt(min(1.,frac/.7)*min(1.,bands/3))
        es=1-confidence(nb)*(1-qs)
        row[7]=ni; row[10:13]=[off,frac,bands]
    row[:5]=[ez,exy,es,eg,(ez*exy*es*eg)**.25]
    row[17:21]=[qz,qxy,qs,qg]
    assert np.isfinite(row).all() and ((row[:5]>=0)&(row[:5]<=1)).all()
    return row
def weights(f,s):
    n=f[:,6]; valid=n>=5
    a=np.minimum(1,f[:,11]/s.get('inside_target',.7))
    b=np.minimum(1,f[:,12]/s.get('bands_target',3.))
    raw=a**s.get('inside_power',.5)*b**s.get('bands_power',.5)
    full=s.get('confidence_full',20.); start=s.get('confidence_start',4.)
    mode=s.get('confidence_mode','linear')
    if mode=='linear': c=np.clip((n-start)/(full-start),0,1)
    elif mode=='exponential': c=1-np.exp(-np.maximum(n-start,0)/full)
    elif mode=='rational': c=np.maximum(n-start,0)/(np.maximum(n-start,0)+full)
    else: raise ValueError(mode)
    if 'confidence_range_decay' in s:
        c*=np.exp(-np.maximum(f[:,15]-15,0)/s['confidence_range_decay'])
    q=np.where(valid,1-c*(1-raw),1.)
    ground_raw=np.exp(-.5*(np.maximum(np.abs(f[:,14])-s.get('ground_tolerance',.15),0)/s.get('ground_scale',.30))**2)
    g=np.where(f[:,13]>0,1-np.minimum(1,f[:,16]/s.get('ground_confidence_full',30))*(1-ground_raw),1.)
    g=1-s.get('ground_strength',1)*(1-g)
    value=np.maximum(s.get('floor',.1),q**s.get('power',2))*g**s.get('ground_power',1.)
    if s.get('height_strength',0): value*=1-s['height_strength']*(1-f[:,0])
    if s.get('position_strength',0): value*=1-s['position_strength']*(1-f[:,1])
    if s.get('sparse_strength',0):
        sparse=np.clip((s.get('sparse_points',5.)-n)/s.get('sparse_points',5.),0,1)
        if 'sparse_range_end' in s:
            sparse*=np.clip((s['sparse_range_end']-f[:,15])/(s['sparse_range_end']-s.get('sparse_range_start',0.)),0,1)
        value*=1-s['sparse_strength']*sparse
    assert np.isfinite(value).all() and (value>=0).all() and (value<=1+1e-14).all()
    return value

def apply_scores(v,s):
    if 'base_rule' in s:
        result=apply_scores(v,s['base_rule'])
    elif 'blend_rule' in s:
        blend=s['blend_rule']; alpha=blend['weight']
        assert 0<=alpha<=1
        result=(1-alpha)*apply_scores(v,blend['first'])+alpha*apply_scores(v,blend['second'])
    else:
        original=v['score']
        if s.get('source_blend',0):
            alpha=s['source_blend']; assert 0<=alpha<=1
            original=(1-alpha)*original+alpha*v['source_scores'][s['source_kind']]
        result=original**s.get('score_power',1.)*weights(v['features'],s)
    if 'density' in s:
        d=s['density']; f=v['features']
        expected=d['count10']*np.exp(-(f[:,15]-10)/d['decay'])
        logratio=np.log((f[:,6]+1)/(expected+1))
        residual=np.abs(logratio) if d['side']=='both' else (-logratio if d['side']=='low' else logratio)
        q=np.exp(-.5*(np.maximum(residual-d['tolerance'],0)/d['scale'])**2)
        result*=1-d['strength']*(1-q)
    assert np.isfinite(result).all() and (result>=0).all() and (result<=1+1e-14).all()
    return result
