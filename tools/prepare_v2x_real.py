"""Real V2X-Real validation / realism-calibration set from val.zip (infrastructure LiDAR -1 only).

Verified: all 50 official detection-test frames come from infrastructure agent -1 (static background
matches val -1 frames point-for-point) but none of them is inside val.zip, so val is outside the
challenge's held-out pool. Frames mirror the test frames: crop to [-40,40]x[-40,40]x[-8,2], uniformly
subsample to 50,000 xyz points. Labels: OPV2V world-frame objects -> LiDAR frame with V2X-Real's own
pose convention (x_to_world / x1_to_x2), V2X-Real SUPER_CLASS_MAP to vehicle / pedestrian / truck.

  python tools/prepare_v2x_real.py
Writes data/v2x_real/points/<scenario>_<frame>.bin, gt.json, realism_calib/, realism_check/.
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(str(RAW_ROOT / 'V2X-Real/val.zip'))
OUT = WORK_ROOT / "data" / "v2x_real"
RANGE = np.array([-40, -40, -8, 40, 40, 2], dtype=np.float64)
N_POINTS = 50_000
SUPER = {**{c: "vehicle" for c in ["Car", "PoliceCar", "LongVehicle"]},
         **{c: "pedestrian" for c in ["Pedestrian", "Child", "RoadWorker", "Scooter", "ScooterRider",
                                      "Motorcycle", "MotorcyleRider", "BicycleRider"]},
         **{c: "truck" for c in ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"]}}


def x_to_world(pose):
    """V2X-Real / OpenCOOD convention, pose = [x, y, z, roll, yaw, pitch] (degrees)."""
    x, y, z, roll, yaw, pitch = pose
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    m = np.identity(4)
    m[:3, 3] = [x, y, z]
    m[0, :3] = [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr]
    m[1, :3] = [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr]
    m[2, :3] = [sp, -cp * sr, cp * cr]
    return m


def objects_to_lidar(objs, lidar_pose):
    w2l = np.linalg.inv(x_to_world(lidar_pose))
    boxes, names = [], []
    for o in objs.values():
        cls = SUPER.get(o.get("obj_type"))
        if cls is None:
            continue
        loc, ctr, ang, ext = o["location"], o["center"], o["angle"], o["extent"]
        m = w2l @ x_to_world([loc[0] + ctr[0], loc[1] + ctr[1], loc[2] + ctr[2], ang[0], ang[1], ang[2]])
        c = m[:3, 3]
        yaw = np.arctan2(m[1, 0], m[0, 0])
        boxes.append([c[0], c[1], c[2], 2 * ext[0], 2 * ext[1], 2 * ext[2], yaw])
        names.append(cls)
    return np.array(boxes, np.float32).reshape(-1, 7), names


def main():
    from guard_v2x import verify
    verify()
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    (OUT / "points").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    bins = sorted(n for n in z.namelist() if n.endswith(".bin") and "/-1/" in n)
    scenarios = sorted({n.split("/")[1] for n in bins})
    gt, kept = {}, []
    for n in bins[::2]:                      # every 2nd frame: 10 Hz neighbours are near-duplicates
        scen, frame = n.split("/")[1], Path(n).stem
        fid = f"s{scenarios.index(scen)}_{frame}"
        pts = np.frombuffer(z.read(n), np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        pts = pts[np.all((pts >= RANGE[:3]) & (pts <= RANGE[3:]), axis=1)]
        rng = np.random.default_rng(scenarios.index(scen) * 100_000 + int(frame))
        if len(pts) > N_POINTS:
            pts = pts[rng.choice(len(pts), N_POINTS, replace=False)]
        pts = pts.astype(np.float32)
        pts.tofile(OUT / "points" / f"{fid}.bin")
        meta = yaml.safe_load(z.read(n[:-4] + ".yaml"))
        boxes, names = objects_to_lidar(meta.get("vehicles", {}), meta["lidar_pose"])
        inside = np.all((boxes[:, :3] >= RANGE[:3]) & (boxes[:, :3] <= RANGE[3:]), axis=1)
        boxes, names = boxes[inside], [nm for nm, k in zip(names, inside) if k]
        n_in = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(boxes)).numpy().sum(1) if len(boxes) else []
        gt[fid] = [{"box": [float(v) for v in b], "class": c, "n_pts": int(k)} for b, c, k in zip(boxes, names, n_in)]
        kept.append(fid)
    (OUT / "gt.json").write_text(json.dumps(gt))
    allb = [g for v in gt.values() for g in v]
    print(f"{len(kept)} frames from {len(scenarios)} scenarios; boxes: "
          + ", ".join(f"{c}={sum(g['class'] == c for g in allb)}" for c in ("vehicle", "pedestrian", "truck")))
    print(f"boxes with >=1 point: {sum(g['n_pts'] > 0 for g in allb)}/{len(allb)}")

    # realism sets: check = 50 frames spread over all scenarios, calib = the rest
    order = np.array(kept)
    check = set(order[np.linspace(0, len(order) - 1, 50).round().astype(int)])
    for name, sel in (("realism_check", [f for f in kept if f in check]), ("realism_calib", [f for f in kept if f not in check])):
        d = OUT / name
        d.mkdir(exist_ok=True)
        for p in d.glob("*.bin"):
            p.unlink()
        for k, f in enumerate(sel, 1):
            (d / f"{k:04d}.bin").write_bytes((OUT / "points" / f"{f}.bin").read_bytes())
        print(f"{name}: {len(sel)} frames")


if __name__ == "__main__":
    main()
