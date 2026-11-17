"""R14 procedural rider augmentation. Task-local, synthetic training only.

install(log_dir, seed=666) adds riders after the existing procedural walker
generator. The original dataset, walker generator, source points and DB stay
unchanged on disk. No real-data file is loaded by this module.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

from release_paths import WORK_ROOT, PCDET_ROOT
ROOT = WORK_ROOT
sys.path.insert(0, str(PCDET_ROOT))
from pcdet.datasets.urbantwin import synth_person as sp

VERSION = 'R14_procedural_rider_v1'
SPEC = {
    'version': VERSION, 'target_fraction': .085,
    'subtypes': ['BicycleRider', 'ScooterRider', 'MotorcyleRider'],
    'probabilities': [.50, .45, .05],
    'box_medians': [[1.82, .75, 1.72], [1.09, .68, 1.81], [1.87, .88, 1.84]],
    'dimension_jitter': [.9, 1.1], 'surface_samples': 2400,
    'point_noise_sigma': .015, 'placement_attempts_per_object': 40,
    'geometry_source': 'Procedural primitives only; W_a aggregate box medians supplied in prereg/brief.',
    'density_rule': 'Same COUNT_AT_10M, COUNT_DECAY_M, COUNT_SIGMA and visible-normal rule as existing SYNTH_PERSON; no size multiplier.',
    'occlusion_rule': 'Original occlude_people(radius=.22), applied to new riders with prior pedestrian boxes as occluders; prior points unchanged.',
}
DEFAULT_CONFIG = {'ENABLED': True, 'TARGET_FRACTION': .085, **SPEC}


def _torus(center, radius, tube_radius, n, rng):
    # Wheel lies in x/z plane, axle along y. Normals use the torus tube direction.
    u = rng.uniform(0, 2*np.pi, n); v = rng.uniform(0, 2*np.pi, n)
    radial = np.c_[np.cos(u), np.zeros(n), np.sin(u)]
    normals = radial*np.cos(v)[:, None] + np.c_[np.zeros(n), np.sin(v), np.zeros(n)]
    return np.asarray(center) + radius*radial + tube_radius*normals, normals


def _cuboid(center, extent, n, rng):
    face = rng.integers(0, 6, n); axis = face//2; sign = np.where(face % 2, 1., -1.)
    p = rng.uniform(-.5, .5, (n, 3))*np.asarray(extent)
    p[np.arange(n), axis] = sign*np.asarray(extent)[axis]/2
    normal = np.zeros_like(p); normal[np.arange(n), axis] = sign
    return p+np.asarray(center), normal


def rider_surface(subtype, dimensions, rng, n=2400, return_parts=False):
    """Dense wheel/frame/body surfaces + normals, local x forward, z ground=0."""
    l, w, h = map(float, dimensions); parts=[]; labels=[]
    def add(label, pair):
        parts.append(pair); labels.extend([label]*len(pair[0]))
    def tube(label, a, b, radius, count):
        add(label, sp._capsule(np.asarray(a), np.asarray(b), radius, count, rng))
    def ell(label, center, radii, count):
        add(label, sp._ellipsoid(np.asarray(center), np.asarray(radii), count, rng))
    scooter = subtype == 'ScooterRider'; moto = subtype == 'MotorcyleRider'
    assert subtype in SPEC['subtypes']
    rr = (.115 if scooter else .18)*l
    tr = (.035 if scooter else (.055 if moto else .023))*w
    axle = l*.49 - rr - tr
    wheel_z = rr + tr + .006*h
    rear, front = [-axle, 0., wheel_z], [axle, 0., wheel_z]
    for name, center in [('rear_wheel',rear),('front_wheel',front)]:
        add(name, _torus(center, rr, tr, int(n*.12), rng))
        # Small axle + spokes supply interior structure, all analytical geometry.
        tube(name, [center[0],-.04*w,center[2]], [center[0],.04*w,center[2]], .012*w, 18)
        for theta in np.linspace(0, np.pi, 5, endpoint=False):
            d=np.array([np.cos(theta),0.,np.sin(theta)])*.92*rr
            tube(name,np.asarray(center)-d,np.asarray(center)+d,.006*w,8)
    handle=np.array([.26*l, 0., (.65 if scooter else .60)*h])
    if scooter:
        deck=np.array([-.06*l,0.,.085*h])
        add('deck',_cuboid(deck,[.57*l,.24*w,.027*h],int(n*.085),rng))
        tube('frame',rear,deck,.027*w,65)
        tube('frame',front,handle,.027*w,100)
        hip=np.array([-.10*l,0.,.52*h]); shoulder=np.array([-.08*l,0.,.79*h])
        feet=[np.array([-.22*l,.11*w,.115*h]),np.array([-.03*l,-.11*w,.115*h])]
        knees=[np.array([-.10*l,.11*w,.31*h]),np.array([-.04*l,-.11*w,.31*h])]
    else:
        hub=np.array([-.03*l,0.,.25*h]); seat=np.array([-.15*l,0.,.45*h])
        stem=np.array([.24*l,0.,.43*h])
        for a,b in [(rear,hub),(rear,seat),(seat,hub),(hub,stem),(seat,stem),(stem,front),(stem,handle)]:
            tube('frame',a,b,.020*w,48)
        ell('seat',seat,[.115*l,.14*w,.025*h],55)
        if moto:
            ell('tank',[.07*l,0.,.39*h],[.18*l,.26*w,.10*h],150)
            ell('motor',[0.,0.,.25*h],[.21*l,.25*w,.10*h],100)
        hip=np.array([-.13*l,0.,(.51 if moto else .49)*h])
        shoulder=np.array([.015*l,0.,.77*h])
        # Bent legs form a seated pose; pedals / footrests sit below the knees.
        feet=[np.array([-.015*l,.20*w,.14*h]),np.array([-.07*l,-.20*w,.18*h])]
        knees=[np.array([.15*l,.17*w,.31*h]),np.array([.12*l,-.17*w,.34*h])]
        for foot in feet:
            add('pedal' if not moto else 'footrest',_cuboid(foot-[0,0,.025*h],[.10*l,.14*w,.018*h],28,rng))
    tube('handlebar',handle+[0,-.41*w,0],handle+[0,.41*w,0],.020*w,90)
    # A leaning ellipsoid torso, plus articulated upper/lower limbs.
    torso_c=(hip+shoulder)/2; angle=np.arctan2((shoulder-hip)[0],(shoulder-hip)[2])
    cp,cn=sp._ellipsoid(np.zeros(3),np.array([.070*l,.20*w,.16*h]),int(n*.16),rng)
    ca,sa=np.cos(angle),np.sin(angle); rot=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    add('torso',(cp@rot.T+torso_c,cn@rot.T))
    head_c=shoulder+np.array([.015*l,0.,(.12 if scooter else .14)*h])
    tube('neck',shoulder,head_c-np.array([0.,0.,.06*h]),.045*w,int(n*.025))
    ell('head',head_c,[.060*l,.115*w,.075*h],int(n*.08))
    for side,knee,foot in zip([1,-1],knees,feet):
        thigh=hip+np.array([0,side*.12*w,-.015*h])
        tube('thigh',thigh,knee,.065*w,int(n*.045))
        tube('shin',knee,foot,.050*w,int(n*.045))
        ell('shoe',foot,[.065*l,.065*w,.025*h],30)
        sh=shoulder+np.array([0,side*.20*w,0])
        elbow=np.array([.14*l,side*.26*w,.66*h])
        hand=handle+np.array([0,side*.36*w,0])
        tube('upper_arm',sh,elbow,.050*w,int(n*.035))
        tube('forearm',elbow,hand,.040*w,int(n*.035))
    pts=np.concatenate([p for p,_ in parts]); normals=np.concatenate([q for _,q in parts])
    labels=np.asarray(labels)
    # Guard rails catch accidental primitive coordinates outside the prescribed box.
    inside=(np.abs(pts[:,0])<=l/2)&(np.abs(pts[:,1])<=w/2)&(pts[:,2]>=0)&(pts[:,2]<=h)
    assert inside.all(), (subtype,dimensions,pts[~inside].min(0),pts[~inside].max(0))
    assert np.isfinite(pts).all() and np.isfinite(normals).all()
    return (pts,normals,labels) if return_parts else (pts,normals)


def ground_candidates(points,cfg):
    xyz=points[:,:3];tol=cfg.get('GROUND_TOL',.25)
    if not len(xyz):return xyz,xyz
    if cfg.get('LOCAL_GROUND',False):
        cell=np.floor(xyz[:,:2]).astype(np.int64);key=(cell[:,0]+1000)*10000+(cell[:,1]+1000)
        order=np.argsort(key,kind='stable');uniq,start=np.unique(key[order],return_index=True)
        zmin=np.minimum.reduceat(xyz[order,2],start)[np.searchsorted(uniq,key)];rel=xyz[:,2]-zmin
        return xyz[rel<tol],xyz[(rel>.3)&(rel<2.3)]
    gz=np.median(xyz[:,2][xyz[:,2]<=np.percentile(xyz[:,2],30)])
    return xyz[np.abs(xyz[:,2]-gz)<tol],xyz[(xyz[:,2]>gz+.3)&(xyz[:,2]<gz+2.3)]


def footprint_collision(c,heading,l,w,boxes,above,clear):
    """Conservative full-footprint disjointness, not merely centre clearance."""
    # Circumscribed rider disk + original existing-box exclusion radius is
    # deliberately conservative and guarantees rectangular footprints disjoint.
    radius=np.hypot(l,w)/2
    if len(boxes):
        other=np.hypot(boxes[:,3],boxes[:,4])/2+.15
        if np.any(np.linalg.norm(boxes[:,:2]-c,axis=1)<radius+other):return 'box'
    if len(above):
        rel=(above[:,:2]-c)@sp._rotz(heading)[:2,:2]
        if np.any((np.abs(rel[:,0])<l/2+.15)&(np.abs(rel[:,1])<w/2+.15)):return 'obstacle_footprint'
        if np.min(np.linalg.norm(above[:,:2]-c,axis=1))<clear:return 'obstacle_clear'
    return None


def visible_and_thin(surface,normals,c,gz,h,heading,cfg,rng):
    rot=sp._rotz(heading);world=surface@rot.T+np.array([*c,gz]);normals=normals@rot.T
    sensors=cfg.get('SENSORS',None)
    if sensors:
        s=np.asarray(sensors,float);order=np.argsort(np.linalg.norm(s[:,:2]-c,axis=1))
        views=[s[order[0]]]+([s[order[1]]] if len(s)>1 and rng.uniform()<.6 else [])
    else:views=[np.array([0.,0.,gz+2.])]
    visible=np.zeros(len(world),bool);mid=np.array([*c,gz+h/2])
    for sensor in views:
        v=sensor-mid;visible|=(normals@(v/np.linalg.norm(v)))>.1
    world=world[visible]
    if len(world)<3:return None,{'surface_visible':len(world)}
    dist=np.hypot(*c);med=cfg.COUNT_AT_10M*np.exp(-(dist-10.)/cfg.COUNT_DECAY_M)
    k=int(np.clip(rng.lognormal(np.log(max(med,1.)),cfg.COUNT_SIGMA),3,len(world)))
    world=world[rng.choice(len(world),k,replace=False)]+rng.normal(0,.015,(k,3))
    return world,{'surface_visible':int(visible.sum()),'requested_points':k,'expected_median_points':float(med),'range_m':float(dist)}


def insert_riders(points,gt_boxes,gt_names,cfg,rng):
    """Synthetic-only caller; returns augmented arrays and explicit generation audit."""
    base_n=int(np.sum(np.asarray(gt_names)=='pedestrian'))
    desired=base_n*SPEC['target_fraction']/(1-SPEC['target_fraction'])
    requested=int(np.floor(desired)+(rng.random()<desired%1))
    audit={'walkers_before':base_n,'requested_riders':requested,'accepted_riders':0,'objects':[],'rejections':{},'tries':0}
    if requested==0:return points,gt_boxes,gt_names,audit
    ground,above=ground_candidates(points,cfg)
    if len(ground)<100:audit['rejections']['insufficient_ground']=1;return points,gt_boxes,gt_names,audit
    people=[];occupied=gt_boxes.copy();clear=cfg.get('OBSTACLE_CLEAR',.7)
    def reject(why):audit['rejections'][why]=audit['rejections'].get(why,0)+1
    for _ in range(requested*SPEC['placement_attempts_per_object']):
        if len(people)>=requested:break
        audit['tries']+=1
        typ=int(rng.choice(3,p=SPEC['probabilities']));subtype=SPEC['subtypes'][typ]
        dims=np.array(SPEC['box_medians'][typ])*rng.uniform(.9,1.1,3);l,w,h=dims
        heading=float(rng.uniform(-np.pi,np.pi));c=ground[rng.integers(len(ground)),:2]+rng.normal(0,.3,2)
        if np.hypot(*c)>cfg.get('MAX_RANGE',38.):reject('range');continue
        collision=footprint_collision(c,heading,l,w,occupied,above,clear)
        if collision:reject(collision);continue
        near=ground[np.linalg.norm(ground[:,:2]-c,axis=1)<1.5]
        if len(near)<3:reject('no_visible_ground');continue
        gz=float(np.median(near[:,2]))
        if cfg.get('LOCAL_GROUND',False):
            wide=ground[np.linalg.norm(ground[:,:2]-c,axis=1)<5.]
            if gz-np.percentile(wide[:,2],10)>.5:reject('roof');continue
        surface,normals=rider_surface(subtype,dims,rng)
        world,render=visible_and_thin(surface,normals,c,gz,h,heading,cfg,rng)
        if world is None:reject('no_surface');continue
        local=(world-np.array([*c,gz]))@sp._rotz(heading)
        inside=(np.abs(local[:,0])<=l/2)&(np.abs(local[:,1])<=w/2)&(local[:,2]>=0)&(local[:,2]<=h)
        world=world[inside]
        if len(world)<3:reject('noise_outside_box');continue
        box=[*map(float,c),gz+float(h)/2,*map(float,dims),heading]
        people.append({'pts':world,'c':c,'gz':gz,'h':float(h),'box':box,'subtype':subtype,'render':render})
        occupied=np.concatenate([occupied,np.array([box],np.float32)])
    if not people:return points,gt_boxes,gt_names,audit
    if cfg.get('OCCLUSION',False):
        occluders=[{'pts':np.empty((0,3)),'c':b[:2],'gz':float(b[2]-b[5]/2),'h':float(b[5])}
                   for b,n in zip(gt_boxes,gt_names) if n=='pedestrian']
        sp.occlude_people(people+occluders,cfg.get('SENSORS',None),radius=.22)
    audit['accepted_riders']=len(people)
    audit['objects']=[{'subtype':p['subtype'],'box':p['box'],'points_after_occlusion':len(p['pts']),**p['render']} for p in people]
    extra=np.concatenate([p['pts'] for p in people]).astype(points.dtype)
    if points.shape[1]>3:extra=np.c_[extra,np.zeros((len(extra),points.shape[1]-3),dtype=points.dtype)]
    return (np.concatenate([points,extra]),np.concatenate([gt_boxes,np.array([p['box'] for p in people],np.float32)]),
            np.concatenate([gt_names,np.array(['pedestrian']*len(people))]),audit)


_INSTALLED=False;_CONTEXT=[];_FILES={};_COUNTERS={};_ORIGINALS={}
def _flush():
    for f in _FILES.values():
        if not f.closed:f.flush()
atexit.register(_flush)


def install(log_dir,seed=666):
    """Enable only DATA_CONFIG.SYNTH_RIDER.ENABLED in this process, never inference."""
    global _INSTALLED
    assert not _INSTALLED,'R14 hook already installed'
    from pcdet.datasets.urbantwin import urbantwin_dataset as ud
    original_insert=ud.insert_persons;original_getitem=ud.UrbanTwinDataset.__getitem__
    log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    def wrapped_insert(points,boxes,names,cfg,rng):
        points,boxes,names=original_insert(points,boxes,names,cfg,rng)
        if not _CONTEXT or not _CONTEXT[-1]['enabled']:return points,boxes,names
        ctx=_CONTEXT[-1];pid=os.getpid();call=_COUNTERS.get(pid,0);_COUNTERS[pid]=call+1
        import torch
        wi=torch.utils.data.get_worker_info();worker=wi.id if wi else -1
        key=f'{seed}:{worker}:{call}:{ctx["frame_id"]}'
        rider_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        out=insert_riders(points,boxes,names,cfg,np.random.default_rng(rider_seed))
        ctx.update(out[3]);ctx.update(worker_id=worker,worker_call=call,rider_seed=rider_seed)
        return out[:3]
    def wrapped_getitem(self,index):
        rcfg=self.dataset_cfg.get('SYNTH_RIDER',{});enabled=bool(self.training and rcfg.get('ENABLED',False))
        idx=index%len(self.infos);ctx={'frame_id':str(self.infos[idx]['frame_id']),'enabled':enabled,'depth':len(_CONTEXT)}
        if enabled:
            assert Path(self.root_path).resolve()==(ROOT/'data/v2x_v3').resolve(),'R14 training hook permits synthetic v2x_v3 only'
            assert 'bin_path' not in self.infos[idx] and self.dataset_cfg.SYNTH_PERSON.ENABLED
            if 'TARGET_FRACTION' in rcfg:assert rcfg['TARGET_FRACTION']==SPEC['target_fraction']
        _CONTEXT.append(ctx)
        try:out=original_getitem(self,index)
        finally:_CONTEXT.pop()
        if enabled:
            assert 'accepted_riders' in ctx,'R14 hook was not called'
            cls=self.class_names.index('pedestrian')+1
            ctx['post_prepare_pedestrian_count']=int(np.sum(out['gt_boxes'][:,-1]==cls))
            # Existing V30 augmentors apply rigid transforms plus one shared
            # isotropic scale; each rider's independently jittered dimension
            # ratios provide a label-free identity fingerprint for this audit.
            # This diagnostic never changes any array passed to the model.
            final_dims=out['gt_boxes'][out['gt_boxes'][:,-1]==cls,3:6]
            retained=0;ambiguous=0
            for obj in ctx['objects']:
                dims=np.asarray(obj['box'][3:6]);scales=final_dims[:,2]/dims[2]
                matches=(np.max(np.abs(final_dims-dims[None]*scales[:,None]),axis=1)<2e-5)&(scales>.94)&(scales<1.06)
                count=int(matches.sum());obj['post_prepare_dimension_matches']=count
                obj['retained_after_prepare']=bool(count==1)
                if count==1:obj['post_prepare_scale']=float(scales[matches][0]);retained+=1
                elif count>1:ambiguous+=1
            ctx['post_prepare_riders_retained']=retained
            ctx['post_prepare_rider_ambiguities']=ambiguous
            ctx['returned_frame_id']=str(out['frame_id']);ctx['generator_version']=VERSION
            ctx['synthetic_root']=str(Path(self.root_path).resolve());ctx['pid']=os.getpid()
            file=_FILES.get(os.getpid())
            if file is None:
                file=(log_dir/f'rider_worker_{os.getpid()}.jsonl').open('a',buffering=1);_FILES[os.getpid()]=file
            file.write(json.dumps(ctx,separators=(',',':'))+'\n')
        return out
    ud.insert_persons=wrapped_insert;ud.UrbanTwinDataset.__getitem__=wrapped_getitem
    _ORIGINALS.update(insert_persons=original_insert,getitem=original_getitem)
    _INSTALLED=True
    return {'version':VERSION,'spec':SPEC,'log_dir':str(log_dir),'rider_seed_base':seed,
            'original_walker_rng':'Unchanged existing np.random.default_rng() behavior; training is not bitwise reproducible merely from torchseed.',
            'real_data_reads':0,'shared_repo_files_modified':False,'dataset_files_modified':False}
