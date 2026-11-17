"""Generate adopted test member caches with original TTA ordering and export limits."""
import argparse,json,subprocess,sys
from pathlib import Path
from release_paths import REPO,WORK_ROOT,KIT_ROOT
from reproduce_v2x import source,SUFFIXES,VEH,PED,TRUCK
HC=['--score-thresh','.01','--max-per-frame','6000','--nms-post','1000','--nms-pre','4000','--topk','2000']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('track',choices=['lumpi','v2x']);ap.add_argument('--frames',type=Path);ap.add_argument('--member-root',type=Path);ap.add_argument('--list-only',action='store_true');a=ap.parse_args()
    members=json.loads((REPO/'configs/members.json').read_text());frames=a.frames or KIT_ROOT/a.track/'detection_test_frames';jobs=[]
    if a.track=='lumpi':
        root=a.member_root or WORK_ROOT/'outputs/lumpi'
        d47=['F_v'+str(n) for n in [9,11,16,19,22,23,24,25,26,27,30]]
        for m in ['A_cp20','F_v3','F_v4','F_v7','F_v8','F_v9','F_v11','F_v16','F_v34','F_v36']:
            jobs.append((m,root/m/'predictions_u.json',['--max-per-frame','600']))
        for m in ['F_v3','F_v4','F_v8','F_v34','F_v36']:
            for r in [5,-5]:jobs.append((m,root/m/f'predictions_ur{r}.json',['--max-per-frame','600','--rot',str(r)]))
        h=['--max-per-frame','3000','--nms-post','500','--nms-pre','2000','--topk','1000']
        for m in ['A_cp20','F_v3','F_v4']:
            jobs.append((m,root/m/'predictions_h.json',h))
            if m!='A_cp20':
                for r in [5,-5]:jobs.append((m,root/m/f'predictions_hr{r}.json',h+['--rot',str(r)]))
        for m in d47+['F_v35']:
            jobs.append((m,root/m/'predictions_h2.json',HC))
            if m!='F_v35':
                for r in [5,-5]:jobs.append((m,root/(m+f'r{r}')/'predictions_h2.json',HC+['--rot',str(r)]))
    else:
        done=set()
        for cls,ms,suffixes in [('vehicle',VEH,SUFFIXES[:4]),('truck',TRUCK,SUFFIXES),('pedestrian',PED,SUFFIXES),('rider',['PR1_R14'],SUFFIXES)]:
            for m in ms:
                for s in suffixes:
                    dest=source(m,s,cls) if cls!='rider' else WORK_ROOT/'outputs/v2x/final/t7/sources'/(m+s+'.json')
                    if str(dest) in done:continue
                    done.add(str(dest));flags=list(HC) if cls in ['pedestrian','rider'] else []
                    if s.startswith('_r'):flags+=['--rot',s[2:]]
                    elif s:flags+=['--flip',{'_fx':'x','_fy':'y','_fxy':'xy'}[s]]
                    if cls=='pedestrian' and m in ['V4','V5','V6','V7']:flags+=['--zshift','-.5']
                    jobs.append((m,dest,flags))
    for m,dest,flags in jobs:
        row=members[a.track+'/'+m]
        raw=dest.with_suffix('.raw.json') if a.track=='v2x' and flags[:len(HC)]==HC else dest
        cmd=[sys.executable,str(REPO/'tools/predict.py'),'--cfg',str(REPO/row['config']),'--ckpt',str(WORK_ROOT/row['checkpoint']),'--frames',str(frames),'--out',str(raw)]+flags
        if a.list_only:print(json.dumps({'member':m,'destination':str(dest.relative_to(WORK_ROOT)) if dest.is_relative_to(WORK_ROOT) else dest.name,'flags':flags}));continue
        dest.parent.mkdir(parents=True,exist_ok=True);subprocess.run(cmd,check=True)
        if raw!=dest:
            p=json.loads(raw.read_text());p={f:sorted([d for d in ds if d['class']=='pedestrian' and d['score']>=.01],key=lambda d:-d['score'])[:400] for f,ds in p.items()}
            dest.write_text(json.dumps(p))
if __name__=='__main__':main()
