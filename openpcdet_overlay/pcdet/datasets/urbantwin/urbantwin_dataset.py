"""UrbanTwin Sim2Real challenge dataset (xyz-only roadside LiDAR).

Expects the layout written by ~/urbantwin/tools/prepare_ut.py:
    <DATA_PATH>/points/<frame_id>.npy      float32 (N, 3), already cropped + density-matched
    <DATA_PATH>/infos_<split>.pkl          [{'frame_id', 'gt_names', 'gt_boxes', 'num_points_in_gt'}]
    <DATA_PATH>/dbinfos_train.pkl + gt_database/   for gt_sampling

INFER_DIR (set via --set DATA_CONFIG.INFER_DIR <dir>) switches to label-free inference over
<dir>/*.bin files, e.g. the challenge's detection_test_frames.
Evaluation uses the official challenge scorer (scoring_program/detection_eval.py).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from .synth_clutter import insert_clutter
from .synth_person import insert_persons


def stretch_objects(points, boxes, names, cfg, rng):
    """Lengthen some objects of the given classes (points and box, about the box center, along the heading).

    UT-LUMPI trucks are ~6.2 m but real LUMPI M4 trucks are 8.5-10.8 m, so detectors fire the Bus head on them.
    Points above the box bottom (+ GROUND_TOL) are scaled along the heading by s ~ U(SCALE); points that
    would now fall inside the lengthened box but were outside the old one are removed; boxes whose
    lengthened footprint would overlap another box are left alone.
    """
    from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

    classes = list(cfg.get('CLASSES', ['Truck']))
    lo, hi = cfg.get('SCALE', [1.1, 1.8])
    prob, tol = cfg.get('PROB', 0.5), cfg.get('GROUND_TOL', 0.1)
    points, boxes = points.copy(), boxes.copy()
    keep = np.ones(len(points), dtype=bool)
    for i in np.flatnonzero(np.isin(names, classes)):
        if rng.random() > prob:
            continue
        b = boxes[i].copy()
        s = rng.uniform(lo, hi)
        nb = b.copy()
        nb[3] *= s
        others = np.delete(boxes, i, axis=0)
        if len(others) and boxes_bev_iou_cpu(nb[None, :7], others[:, :7]).max() > 0:
            continue
        h, n = np.array([np.cos(b[6]), np.sin(b[6])]), np.array([-np.sin(b[6]), np.cos(b[6])])
        rel = points[:, :2] - b[:2]
        a, c = rel @ h, rel @ n
        bottom, top = b[2] - b[5] / 2, b[2] + b[5] / 2
        zin = (points[:, 2] > bottom + tol) & (points[:, 2] < top + 0.05)
        lat = np.abs(c) <= b[4] / 2 + 0.05
        inside = lat & zin & (np.abs(a) <= b[3] / 2 + 0.05)
        new_region = lat & zin & ~inside & (np.abs(a) <= nb[3] / 2 + 0.05)
        keep &= ~new_region
        points[inside, :2] += np.outer(a[inside] * (s - 1), h).astype(points.dtype)
        boxes[i] = nb
    return points[keep], boxes


def load_bin_xyz(path):
    raw = np.fromfile(str(path), dtype=np.float32)
    pts = raw.reshape(-1, 3) if raw.size % 3 == 0 else raw.reshape(-1, 4)[:, :3]  # same order as the official loader
    return np.ascontiguousarray(pts)


class UrbanTwinDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)
        infer_dir = self.dataset_cfg.get('INFER_DIR', None)
        if infer_dir and not training:
            files = sorted(Path(infer_dir).glob('*.bin'))
            self.infos = [{'frame_id': f.stem, 'bin_path': str(f)} for f in files]
        else:
            split = self.dataset_cfg.DATA_SPLIT[self.mode]
            with open(self.root_path / f'infos_{split}.pkl', 'rb') as f:
                self.infos = pickle.load(f)
            max_frames = self.dataset_cfg.get('MAX_FRAMES', {}).get(self.mode, 0)
            if max_frames and len(self.infos) > max_frames:
                keep = np.linspace(0, len(self.infos) - 1, max_frames).round().astype(int)
                self.infos = [self.infos[i] for i in keep]
        if self.logger is not None:
            self.logger.info(f'UrbanTwin {self.mode}: {len(self.infos)} frames')

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def get_lidar(self, info):
        if 'bin_path' in info:
            return load_bin_xyz(info['bin_path'])
        return np.load(self.root_path / 'points' / f"{info['frame_id']}.npy").astype(np.float32)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = self.infos[index]
        points = self.get_lidar(info)
        keep_ratio = self.dataset_cfg.get('RANDOM_KEEP_RATIO', None)  # density augmentation, e.g. [0.6, 1.0]
        if self.training and keep_ratio and keep_ratio[0] < 1.0:
            n = int(len(points) * np.random.uniform(*keep_ratio))
            points = points[np.sort(np.random.choice(len(points), n, replace=False))]
        input_dict = {'frame_id': info['frame_id'], 'points': points}
        if 'gt_boxes' in info:
            input_dict['gt_names'] = np.array(info['gt_names'])
            input_dict['gt_boxes'] = np.array(info['gt_boxes'], dtype=np.float32).reshape(-1, 7)
            clutter = self.dataset_cfg.get('SYNTH_CLUTTER', None)  # unlabelled negatives, placed before people
            if self.training and clutter and clutter.get('ENABLED', False):
                input_dict['points'] = insert_clutter(input_dict['points'], input_dict['gt_boxes'], clutter,
                                                      np.random.default_rng())
            synth = self.dataset_cfg.get('SYNTH_PERSON', None)
            if self.training and synth and synth.get('ENABLED', False):
                input_dict['points'], input_dict['gt_boxes'], input_dict['gt_names'] = insert_persons(
                    input_dict['points'], input_dict['gt_boxes'], input_dict['gt_names'], synth,
                    np.random.default_rng())
            stretch = self.dataset_cfg.get('LENGTH_STRETCH', None)
            if self.training and stretch and stretch.get('ENABLED', False):
                input_dict['points'], input_dict['gt_boxes'] = stretch_objects(
                    input_dict['points'], input_dict['gt_boxes'], input_dict['gt_names'], stretch,
                    np.random.default_rng())
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'gt_boxes' not in self.infos[0]:
            return 'No ground-truth boxes for evaluation', {}
        sys.path.insert(0, str(Path(self.dataset_cfg.OFFICIAL_EVAL_DIR).expanduser()))
        from detection_eval import evaluate_3d_detection  # official challenge scorer

        gt = {info['frame_id']: [{'box': [float(v) for v in b], 'class': n}
                                 for b, n in zip(info['gt_boxes'], info['gt_names'])]
              for info in self.infos}
        preds = {fid: [] for fid in gt}
        for anno in det_annos:
            preds[anno['frame_id']] = [{'box': [float(v) for v in b[:7]], 'score': float(s), 'class': str(n)}
                                       for b, s, n in zip(anno['boxes_lidar'], anno['score'], anno['name'])]
        eval_classes = self.dataset_cfg.get('EVAL_CLASSES', class_names)
        res = evaluate_3d_detection(preds, gt, list(eval_classes), iou_thresholds=(0.5, 0.7), iou_type='3d')
        bev = evaluate_3d_detection(preds, gt, list(eval_classes), iou_thresholds=(0.5,), iou_type='bev')

        lines = [f"{'class':<12}{'AP3d@.5':>9}{'AP3d@.7':>9}{'n_gt':>8}{'n_pred':>8}"]
        ap_dict = {}
        for cls, info in res['per_class'].items():
            lines.append(f"{cls:<12}{info['AP_0.5_R40']:>9.4f}{info['AP_0.7_R40']:>9.4f}{info['n_gt']:>8}{info['n_pred']:>8}")
            ap_dict[f'{cls}_3d_0.5_R40'] = info['AP_0.5_R40']
        ap_dict.update({'mAP_3d_0.5_R40': res['mAP_0.5_R40'], 'mAP_3d_0.7_R40': res['mAP_0.7_R40'],
                        'mAP_bev_0.5_R40': bev['mAP_0.5_R40']})
        lines.append(f"mAP_3d_0.5_R40={res['mAP_0.5_R40']:.4f}  mAP_3d_0.7_R40={res['mAP_0.7_R40']:.4f}  "
                     f"mAP_bev_0.5_R40={bev['mAP_0.5_R40']:.4f}")
        return '\n' + '\n'.join(lines), ap_dict
