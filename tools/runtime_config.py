"""Resolve portable YAML with the actual OpenPCDet parser."""
from pathlib import Path
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from release_paths import REPO, WORK_ROOT, KIT_ROOT
import os
def resolve(path):
    os.chdir(REPO)
    cfg=EasyDict();cfg_from_yaml_file(str(path),cfg)
    data=cfg.DATA_CONFIG
    data.DATA_PATH=str(WORK_ROOT/data.DATA_PATH)
    if 'OFFICIAL_EVAL_DIR' in data:
        data.OFFICIAL_EVAL_DIR=str(KIT_ROOT/'lumpi/scoring_program')
    return cfg
