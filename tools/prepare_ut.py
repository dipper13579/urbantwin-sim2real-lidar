"""Convert a downloaded UrbanTwin sequence into the training layout used by UrbanTwinDataset.

  python tools/prepare_ut.py --track lumpi
  python tools/prepare_ut.py --track v2x

Per frame: crop to the track's evaluation range, uniformly subsample to --points (the LUMPI
detection-test frames hold exactly 50,000 xyz points), map UT class names to the track's
classes, and drop boxes with fewer than --min-pts points (CARLA labels every actor, including
fully occluded ones). Frames with index >= --val-from form the synthetic validation split.
Incremental: re-run as more frames download; finished frames are cached under meta/.
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TRACKS = {
    "lumpi": {
        "src": str(RAW_ROOT / 'UT-LUMPI/extracted/UCF-DT-LUMPI-Seq-1'),
        "range": [-40, -40, -5, 40, 40, 5],
        "classes": ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"],
        "map": {"Car": "Car", "Van": "Van", "Truck": "Truck", "Bus": "Bus", "Motorcycle": "Motorcycle",
                "Bicycle": "Bicycle", "Unknown": "Unknown", "Person": "Person", "Pedestrian": "Person"},
    },
    "v2x": {
        "src": str(RAW_ROOT / 'UT-V2X-Real-IC/extracted/UCF-DT-V2X-Real-Seq-1'),
        "range": [-40, -40, -8, 40, 40, 2],
        "classes": ["vehicle", "pedestrian", "truck"],
        # V2X-Real SUPER_CLASS_MAP (opencood/data_utils/__init__.py); note Van -> truck
        "map": {**{c: "vehicle" for c in ["Car", "PoliceCar", "LongVehicle"]},
                **{c: "pedestrian" for c in ["Pedestrian", "Child", "RoadWorker", "Scooter", "ScooterRider",
                                             "Motorcycle", "MotorcyleRider", "BicycleRider"]},
                **{c: "truck" for c in ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"]}},
    },
}

ARGS = None


def own_points(pts, b, scale=(1.0, 1.0, 1.0)):
    """Mask of the points of an object: inside the box (x scale, 0.1 m margin) and above the ground returns."""
    c, s = np.cos(b[6]), np.sin(b[6])
    ex, ey, z = pts[:, 0] - b[0], pts[:, 1] - b[1], pts[:, 2] - (b[2] - b[5] / 2)
    return ((np.abs(ex * c + ey * s) <= b[3] * scale[0] / 2 + 0.1) & (np.abs(-ex * s + ey * c) <= b[4] * scale[1] / 2 + 0.1)
            & (z > 0.05) & (z < b[5] * scale[2] + 0.1))


def scale_objects(pts, boxes, names, spec):
    """Enlarge the objects of some classes about the box bottom centre, points and box (the twin's cars are smaller than
    real ones in every dimension). spec = {class: {"pts": [sl, sw, sh], "box": [sl, sw, sh]}}. An object whose enlarged
    footprint would touch another box is left as it is. Returns points, boxes, objects enlarged, objects left."""
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

    keep = np.ones(len(pts), bool)
    done = left = 0
    for i in np.flatnonzero(np.isin(names, list(spec))):
        b = boxes[i].copy()
        sp, sb = np.array(spec[names[i]]["pts"], np.float32), np.array(spec[names[i]]["box"], np.float32)
        nb = b.copy()
        nb[3:6] = b[3:6] * np.maximum(sp, sb)
        others = np.delete(boxes, i, axis=0)
        if len(others) and boxes_bev_iou_cpu(nb[None, :7], others[:, :7]).max() > 0:
            left += 1
            continue
        own = own_points(pts, b)
        keep &= own | ~own_points(pts, b, sp)  # other points the enlarged object would now contain
        h, n = np.array([np.cos(b[6]), np.sin(b[6])], np.float32), np.array([-np.sin(b[6]), np.cos(b[6])], np.float32)
        rel = pts[own, :2] - b[:2]
        pts[own, :2] = b[:2] + np.outer((rel @ h) * sp[0], h) + np.outer((rel @ n) * sp[1], n)
        bottom = b[2] - b[5] / 2
        pts[own, 2] = bottom + (pts[own, 2] - bottom) * sp[2]
        boxes[i, 3:6] = b[3:6] * sb
        boxes[i, 2] = bottom + boxes[i, 5] / 2
        done += 1
    return pts[keep], boxes, done, left


def process(fid: str):
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    cfg, src, out = TRACKS[ARGS.track], Path(ARGS.src), Path(ARGS.out)
    meta = out / "meta" / f"{fid}.pkl"
    if meta.exists():
        return pickle.loads(meta.read_bytes())

    shift = np.array(ARGS.shift, np.float32)
    pts = np.load(src / "pcd" / f"{fid}.npy")[:, :3].astype(np.float32) + shift
    lo, hi = np.array(cfg["range"][:3]), np.array(cfg["range"][3:])
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if ARGS.pts_scale or ARGS.obj_keep != 1.0:
        return process_objects(fid, pts)
    if len(pts) > ARGS.points:
        rng = np.random.default_rng(int(fid))
        pts = pts[np.sort(rng.choice(len(pts), ARGS.points, replace=False))]
    np.save(out / "points" / f"{fid}.npy", pts)

    boxes, names = read_boxes(fid)
    if len(boxes):
        n_pts = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(boxes)).numpy().sum(1)
    else:
        n_pts = np.zeros(0, int)
    keep = n_pts >= ARGS.min_pts
    info = {"frame_id": fid, "gt_boxes": boxes[keep], "gt_names": names[keep],
            "num_points_in_gt": n_pts[keep].astype(np.int32), "num_points": len(pts)}
    meta.write_bytes(pickle.dumps(info))
    return info


def process_objects(fid: str, pts):
    """The frame with --pts-scale / --obj-keep: objects are enlarged and their points kept at --obj-keep times the
    uniform ratio, before the frame is cut down to --points (real objects carry about twice the points of the twin's)."""
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    out = Path(ARGS.out)
    boxes, names = read_boxes(fid)
    pts, boxes, done, left = scale_objects(pts, boxes, names, ARGS.pts_scale)
    if len(pts) > ARGS.points:
        rng = np.random.default_rng(int(fid))
        obj = np.zeros(len(pts), bool)
        for b in boxes:
            obj |= own_points(pts, b)
        sel = np.flatnonzero(obj & (rng.random(len(pts)) < min(1.0, ARGS.obj_keep * ARGS.points / len(pts))))
        rest = rng.choice(np.flatnonzero(~obj), ARGS.points - len(sel), replace=False)
        pts = pts[np.sort(np.concatenate([sel, rest]))]
    np.save(out / "points" / f"{fid}.npy", pts)
    if len(boxes):
        n_pts = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(boxes)).numpy().sum(1)
    else:
        n_pts = np.zeros(0, int)
    keep = n_pts >= ARGS.min_pts
    info = {"frame_id": fid, "gt_boxes": boxes[keep], "gt_names": names[keep],
            "num_points_in_gt": n_pts[keep].astype(np.int32), "num_points": len(pts), "enlarged": (done, left)}
    (out / "meta" / f"{fid}.pkl").write_bytes(pickle.dumps(info))
    return info


def read_boxes(fid: str):
    """Boxes (twin-to-site shift and --size-scale applied) and class names of the labels with their centre in the range."""
    cfg, src = TRACKS[ARGS.track], Path(ARGS.src)
    shift = np.array(ARGS.shift, np.float32)
    lo, hi = np.array(cfg["range"][:3]), np.array(cfg["range"][3:])
    boxes, names = [], []
    for line in open(src / "lbl" / f"{fid}.txt"):
        p = line.split()
        if len(p) >= 8 and p[7] in cfg["map"]:
            boxes.append([float(v) for v in p[:7]])
            names.append(cfg["map"][p[7]])
    boxes = np.array(boxes, np.float32).reshape(-1, 7)
    names = np.array(names)
    boxes[:, :3] += shift
    for cls, scale in ARGS.size_scale.items():  # move UT box conventions toward the real annotations
        sel = names == cls
        new = boxes[sel, 3:6] * np.array(scale, np.float32)
        boxes[sel, 2] += (new[:, 2] - boxes[sel, 5]) / 2  # keep the box bottom on the ground
        boxes[sel, 3:6] = new
    inside = np.all((boxes[:, :3] >= lo) & (boxes[:, :3] <= hi), axis=1)
    return boxes[inside], names[inside]


def build_gt_database(out: Path, infos, classes):
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    db_dir = out / "gt_database"
    db_dir.mkdir(exist_ok=True)
    db = {c: [] for c in classes}
    for info in infos:
        if not len(info["gt_boxes"]):
            continue
        pts = np.load(out / "points" / f"{info['frame_id']}.npy")
        mask = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(info["gt_boxes"])).numpy()
        for i, (box, name) in enumerate(zip(info["gt_boxes"], info["gt_names"])):
            obj = pts[mask[i] > 0].copy()
            obj[:, :3] -= box[:3]
            rel = f"gt_database/{info['frame_id']}_{name}_{i}.bin"
            obj.astype(np.float32).tofile(out / rel)
            db[name].append({"name": name, "path": rel, "gt_idx": i, "box3d_lidar": box,
                             "num_points_in_gt": obj.shape[0], "difficulty": 0})
    with open(out / "dbinfos_train.pkl", "wb") as f:
        pickle.dump(db, f)
    return {k: len(v) for k, v in db.items()}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=TRACKS, required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--points", type=int, default=50_000)
    ap.add_argument("--min-pts", type=int, default=5)
    ap.add_argument("--val-from", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shift", default="0,0,0", help="dx,dy,dz added to points and boxes (twin-to-site alignment)")
    ap.add_argument("--size-scale", default="{}", help='JSON {class: [sl, sw, sh]} applied to box sizes')
    ap.add_argument("--class-override", default="", help="'Van=vehicle,...' overrides the UT -> track class map")
    ap.add_argument("--pts-scale", default="{}",
                    help='JSON {class: {"pts": [sl, sw, sh], "box": [sl, sw, sh]}}: enlarge the objects, points and box')
    ap.add_argument("--obj-keep", type=float, default=1.0,
                    help="object points are kept at this multiple of the uniform keep ratio")
    ARGS = ap.parse_args()
    ARGS.pts_scale = json.loads(ARGS.pts_scale)
    for kv in filter(None, ARGS.class_override.split(",")):
        k, v = kv.split("=")
        TRACKS[ARGS.track]["map"][k] = v
    ARGS.shift = [float(v) for v in ARGS.shift.split(",")]
    ARGS.size_scale = json.loads(ARGS.size_scale)
    cfg = TRACKS[ARGS.track]
    ARGS.src = ARGS.src or cfg["src"]
    ARGS.out = ARGS.out or str(Path.home() / "urbantwin" / "data" / ARGS.track)
    out = Path(ARGS.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)

    fids = sorted(p.stem for p in (Path(ARGS.src) / "pcd").glob("*.npy"))
    print(f"{len(fids)} frames with point clouds in {ARGS.src}", flush=True)
    with Pool(ARGS.workers) as pool:
        infos = pool.map(process, fids, chunksize=16)

    if ARGS.pts_scale:
        print("objects enlarged / left (footprint would touch another box): "
              f"{sum(i['enlarged'][0] for i in infos)} / {sum(i['enlarged'][1] for i in infos)}", flush=True)
    train = [i for i in infos if int(i["frame_id"]) < ARGS.val_from]
    val = [i for i in infos if int(i["frame_id"]) >= ARGS.val_from]
    for split, rows in (("train", train), ("val", val)):
        with open(out / f"infos_{split}.pkl", "wb") as f:
            pickle.dump(rows, f)
        counts = {}
        for r in rows:
            for n in r["gt_names"]:
                counts[n] = counts.get(n, 0) + 1
        print(f"{split}: {len(rows)} frames, boxes per class {counts}", flush=True)
    print("gt_database:", build_gt_database(out, train, cfg["classes"]), flush=True)
    print("=== PREPARE DONE ===", flush=True)


if __name__ == "__main__":
    main()
