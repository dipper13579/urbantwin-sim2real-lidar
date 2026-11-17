"""User-owned inputs live outside the public checkout."""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get('WORK_ROOT', str(REPO / 'work'))).resolve()
RAW_ROOT = Path(os.environ.get('RAW_ROOT', str(WORK_ROOT / 'raw'))).resolve()
KIT_ROOT = Path(os.environ.get('KIT_ROOT', str(WORK_ROOT / 'starting_kit'))).resolve()
PCDET_ROOT = Path(os.environ.get('PCDET_ROOT', str(REPO / 'OpenPCDet'))).resolve()
