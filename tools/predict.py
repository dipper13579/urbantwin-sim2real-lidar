"""Run a trained OpenPCDet model and write detections in the challenge's predictions.json format.

  # official test frames -> submission predictions
  python tools/predict.py --cfg cfgs/urbantwin/lumpi_centerpoint.yaml --ckpt <ckpt.pth> \
      --frames starting_kit/lumpi/detection_test_frames --out outputs/lumpi/run1/predictions.json
  # synthetic validation split (no --frames) -> also writes <out>_gt.json for tools/eval_detection.py
  python tools/predict.py --cfg ... --ckpt ... --out outputs/lumpi/run1/val_pred.json --max-frames 500

--cfg is relative to OpenPCDet/tools (as for train.py).
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=Path, default=None, help="folder of .bin frames (label-free inference)")
    ap.add_argument("--max-frames", type=int, default=0, help="val split: evenly spaced subset (0 = all)")
    ap.add_argument("--score-thresh", type=float, default=0.03)
    ap.add_argument("--max-per-frame", type=int, default=150)
    ap.add_argument("--nms-post", type=int, default=0,
                    help="override NMS_POST_MAXSIZE (boxes kept per head after NMS; config 100) -- 0 = config")
    ap.add_argument("--nms-pre", type=int, default=0, help="override NMS_PRE_MAXSIZE (config 1000) -- 0 = config")
    ap.add_argument("--topk", type=int, default=0, help="override MAX_OBJ_PER_SAMPLE (heatmap top-K per head) -- 0 = config")
    ap.add_argument("--nms-thresh", type=float, default=0.0, help="override NMS_THRESH (BEV IoU, per head; config 0.2) -- 0 = config")
    ap.add_argument("--ground-align", type=float, default=None,
                    help="label-free test-time height alignment: shift every frame in z so that its ground (the "
                         "--ground-pct percentile of point z in the --ground-ring BEV ring) sits at this height (the "
                         "synthetic training ground), predict, shift the boxes back")
    ap.add_argument("--ground-ring", default="10,30", help="BEV ring (m) used to estimate the frame's ground height")
    ap.add_argument("--ground-pct", type=float, default=5.0)
    ap.add_argument("--ground-scale", type=float, default=1.0, help="apply this fraction of the per-frame shift")
    ap.add_argument("--zshift", type=float, default=None, help="constant z shift for every frame (instead of --ground-align)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--data-path", default=None, help="override DATA_CONFIG.DATA_PATH")
    ap.add_argument("--flip", choices=["none", "x", "y", "xy"], default="none",
                    help="test-time augmentation: mirror the input (x -> -x and/or y -> -y), un-mirror the boxes")
    ap.add_argument("--rot", type=float, default=0.0,
                    help="test-time augmentation: rotate the input by this many degrees about the sensor z axis, rotate the boxes back")
    ap.add_argument("--shift", default=None,
                    help="test-time augmentation: translate the input by 'dx,dy' metres (changes the voxel phase), translate the boxes back")
    args = ap.parse_args()
    if sum([args.rot != 0, args.flip != "none", args.shift is not None]) > 1:
        ap.error("--rot, --flip and --shift are separate augmentations; use one per run")
    out = args.out.resolve()
    frames = args.frames.resolve() if args.frames else None
    ckpt = str(Path(args.ckpt).resolve())

    from release_paths import REPO
    os.chdir(REPO)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    from runtime_config import resolve
    cfg.update(resolve(args.cfg))
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    if args.nms_post:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = args.nms_post
    if args.nms_pre:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE = args.nms_pre
    if args.topk:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE = args.topk
    if args.nms_thresh:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = args.nms_thresh
    if frames:
        cfg.DATA_CONFIG.INFER_DIR = str(frames)
    if args.data_path:
        cfg.DATA_CONFIG.DATA_PATH = args.data_path
    cfg.DATA_CONFIG.MAX_FRAMES = {"train": 0, "test": args.max_frames}
    logger = common_utils.create_logger()
    dataset, loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=args.batch_size,
                                          dist=False, workers=4, training=False, logger=logger)
    sx = -1.0 if "x" in args.flip else 1.0
    sy = -1.0 if "y" in args.flip else 1.0
    if args.flip != "none":
        get_lidar = dataset.get_lidar

        def flipped(info):
            p = get_lidar(info).copy()
            p[:, 0] *= sx
            p[:, 1] *= sy
            return p
        dataset.get_lidar = flipped  # workers are forked, so they inherit the patched method
    rc, rs = np.cos(np.deg2rad(args.rot)), np.sin(np.deg2rad(args.rot))
    if args.rot:
        get_lidar_r = dataset.get_lidar

        def rotated(info):
            p = get_lidar_r(info).copy()
            x, y = p[:, 0].copy(), p[:, 1].copy()
            p[:, 0] = rc * x - rs * y
            p[:, 1] = rs * x + rc * y
            return p
        dataset.get_lidar = rotated
    tx, ty = (float(v) for v in args.shift.split(",")) if args.shift else (0.0, 0.0)
    if args.shift:
        get_lidar_t = dataset.get_lidar

        def translated(info):
            p = get_lidar_t(info).copy()
            p[:, 0] += tx
            p[:, 1] += ty
            return p
        dataset.get_lidar = translated
    zshift = {}
    if args.ground_align is not None or args.zshift is not None:
        r0, r1 = (float(v) for v in args.ground_ring.split(","))
        raw_lidar = dataset.get_lidar
        for info in dataset.infos:
            if args.zshift is not None:
                zshift[str(info["frame_id"])] = args.zshift
                continue
            p = raw_lidar(info)
            r = np.hypot(p[:, 0], p[:, 1])
            z = p[(r > r0) & (r < r1), 2]
            zshift[str(info["frame_id"])] = (args.ground_scale * float(args.ground_align - np.percentile(z, args.ground_pct))
                                             if len(z) else 0.0)

        def aligned(info):
            p = raw_lidar(info).copy()
            p[:, 2] += zshift[str(info["frame_id"])]
            return p
        dataset.get_lidar = aligned
        v = np.array(list(zshift.values()))
        print(f"ground-align to {args.ground_align} (zshift {args.zshift}, scale {args.ground_scale}): per-frame z shift mean {v.mean():+.3f} min {v.min():+.3f} max {v.max():+.3f}")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    preds = {}
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            for anno in dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES):
                if args.flip != "none" and len(anno["boxes_lidar"]):
                    b = anno["boxes_lidar"]
                    b[:, 0] *= sx
                    b[:, 1] *= sy
                    if sx < 0:
                        b[:, 6] = np.pi - b[:, 6]
                    if sy < 0:
                        b[:, 6] = -b[:, 6]
                if args.rot and len(anno["boxes_lidar"]):
                    b = anno["boxes_lidar"]
                    x, y = b[:, 0].copy(), b[:, 1].copy()
                    b[:, 0] = rc * x + rs * y
                    b[:, 1] = -rs * x + rc * y
                    b[:, 6] = b[:, 6] - np.deg2rad(args.rot)
                if args.shift and len(anno["boxes_lidar"]):
                    anno["boxes_lidar"][:, 0] -= tx
                    anno["boxes_lidar"][:, 1] -= ty
                if zshift and len(anno["boxes_lidar"]):
                    anno["boxes_lidar"][:, 2] -= zshift[str(anno["frame_id"])]
                order = np.argsort(-anno["score"])[: args.max_per_frame]
                preds[str(anno["frame_id"])] = [
                    {"box": [round(float(v), 4) for v in anno["boxes_lidar"][i][:7]],
                     "score": round(float(anno["score"][i]), 5), "class": str(anno["name"][i])}
                    for i in order]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds))
    n = sum(len(v) for v in preds.values())
    print(f"wrote {out}: {len(preds)} frames, {n} detections")

    if not frames:  # validation split: dump matching GT in the same format
        gt = {info["frame_id"]: [{"box": [float(v) for v in b], "class": str(c)}
                                 for b, c in zip(info["gt_boxes"], info["gt_names"])] for info in dataset.infos}
        gt_path = out.with_name(out.stem + "_gt.json")
        gt_path.write_text(json.dumps(gt))
        print(f"wrote {gt_path}")


if __name__ == "__main__":
    main()
