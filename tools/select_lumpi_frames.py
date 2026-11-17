"""Reconstruct adopted frame selections, writing only private local name lists."""
import argparse,subprocess,sys
import numpy as np
from release_paths import REPO,WORK_ROOT
from lumpi_real import is_excluded,forbidden_ids,load_labels
def save(tag,ids):
    p=WORK_ROOT/'selection'/('lumpi_'+tag+'.txt');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(f'lidar/{int(i):06d}.ply\n' for i in sorted(ids)))
    print(tag,len(ids))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['base','rare','window']);a=ap.parse_args()
    if a.phase=='base':
        m4=set(np.linspace(20,6150,150).round().astype(int))|{500,3000}
        cand=np.arange(20,12200);cand=cand[~is_excluded(cand)]
        m6=set(cand[np.linspace(0,len(cand)-1,100).round().astype(int)])|{500,3000}
        for m,ids in [(4,m4),(6,m6)]:
            assert not is_excluded(list(ids)).any();save('M'+str(m),ids)
    elif a.phase=='rare':
        for m,gap,budget in [(4,6,80),(6,10,110)]:
            subprocess.run([sys.executable,str(REPO/'tools/select_rare_frames.py'),'--budget',f'{m}={budget}','--gap',str(gap),'--out-dir',str(WORK_ROOT/'selection')],check=True)
    else:
        fb=np.array(forbidden_ids());df=load_labels(WORK_ROOT/'data/lumpi_real/M4_Label.csv',buffer=2)
        have={int(p.stem.split('_')[1]) for tag in ['M4','M4r'] for p in (WORK_ROOT/'data/lumpi_real'/('points_'+tag)).glob('*.bin')}
        cs=set(int(f) for f in df.frame.unique() if fb[0]<f<fb[-1])-have
        picks=[int((a+b)//2) for a,b in zip(fb[:-1],fb[1:]) if int((a+b)//2) in cs]
        while len(picks)<100:
            rest=np.array(sorted(cs-set(picks)));d=np.minimum(np.abs(rest[:,None]-np.array(picks)[None,:]).min(1),np.abs(rest[:,None]-fb[None,:]).min(1))
            if d.max()<4:break
            picks.append(int(rest[np.argmax(d)]))
        assert len(picks)==100 and not is_excluded(picks,2).any();save('M4w',picks)
if __name__=='__main__':main()
