"""Train one inventory member, or parse all recipes without training."""
import argparse, json, os, sys
from pathlib import Path
from release_paths import REPO, WORK_ROOT, PCDET_ROOT
from runtime_config import resolve
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--member');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();rows=json.loads((REPO/'configs/members.json').read_text())
    selected={args.member:rows[args.member]} if args.member else rows
    checked=[]
    for name,row in selected.items():
        cfg=resolve(REPO/row['config'])
        assert cfg.DATA_CONFIG.DATASET=='UrbanTwinDataset'
        assert len(cfg.CLASS_NAMES)==len(set(cfg.CLASS_NAMES))
        assert cfg.DATA_CONFIG.DATA_SPLIT.train=='train'
        assert not cfg.DATA_CONFIG.get('INFER_DIR')
        assert cfg.OPTIMIZATION.NUM_EPOCHS>0
        checked.append({'member':name,'config':row['config'],'dataset':str(Path(cfg.DATA_CONFIG.DATA_PATH).relative_to(WORK_ROOT)),'epoch':row['checkpoint_epoch'],'parsed':True})
        if not args.dry_run:
            if not args.member:ap.error('Choose one --member; use the documented dependency order.')
            sys.path.insert(0,str(PCDET_ROOT/'tools'));import train as trainer
            argv=['train.py','--cfg_file',str(REPO/row['config']),'--epochs',str(row['planned_epochs'] or row['checkpoint_epoch']),'--batch_size',str(row.get('batch_size') or 4),'--workers',str(row.get('workers') or 4),'--extra_tag','release_'+row['member'],'--num_epochs_to_eval','1']
            if row['parent']:
                parent=rows[row['track']+'/'+row['parent']]
                argv+=['--pretrained_model',str(WORK_ROOT/parent['checkpoint'])]
            if str(row['fix_random_seed']).lower()=='true':argv+=['--fix_random_seed']
            if cfg.DATA_CONFIG.get('SYNTH_RIDER',{}).get('ENABLED'):
                import synth_rider
                synth_rider.install(WORK_ROOT/'training_audit'/row['member'],seed=666)
            sys.argv=argv;original=trainer.parse_config
            def parse():
                a,c=original();c.update(cfg);return a,c
            trainer.parse_config=parse
            trainer.main()
    print(json.dumps({'training_executed':not args.dry_run,'recipes':checked},indent=2))
if __name__=='__main__':main()
