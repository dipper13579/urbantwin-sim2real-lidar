"""Rebuild the exact synthetic dataset variants. No real points are training inputs."""
import argparse,json,subprocess,sys
from release_paths import REPO,WORK_ROOT
SCALE={'Bicycle':[1.55,1.89,1.5],'Motorcycle':[1.35,1.41,1.45],'Van':[1.06,1.35,1.14],'Bus':[1.5,.93,.98]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['lumpi','lumpi_v2','lumpi_v3b','v2x','v2x_v2','v2x_v3','v2x_dense']);a=ap.parse_args()
    name=a.dataset;track='lumpi' if name.startswith('lumpi') else 'v2x'
    dest=WORK_ROOT/'data'/name
    flags=[]
    if name in ['lumpi_v2','lumpi_v3b']:flags+=['--shift=-1.25,0.25,0.04','--size-scale',json.dumps(SCALE)]
    if name=='lumpi_v3b':flags+=['--pts-scale',json.dumps({'Car':{'pts':[1.17,1.24,1.28],'box':[1.17,1.23,1.27]}}),'--obj-keep','2.3']
    if name=='v2x_v2':flags+=['--class-override','Van=vehicle']
    if name=='v2x_dense':
        dest=WORK_ROOT/'outputs/v2x/final/t9/r24/synthetic_dataset'
        stats=json.loads((REPO/'configs/v2x_dense_statistics.json').read_text())
        flags+=['--pts-scale',json.dumps(stats['scales']),'--obj-keep','2.3']
    subprocess.run([sys.executable,str(REPO/'tools/prepare_ut.py'),'--track',track,'--out',str(dest),'--workers','4']+flags,check=True)
if __name__=='__main__':main()
