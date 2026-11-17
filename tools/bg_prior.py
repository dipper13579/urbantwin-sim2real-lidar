"""Point-level background prior for a fixed roadside site. Voxel occupancy over unlabeled non-forbidden real frames
(all frames are the same 50k-point subsample, so shares are comparable) gives, per voxel, the share of frames in
which it holds a point. A box whose above-ground points sit mostly in voxels occupied in many frames is a static
structure (pole, bush, sign); its score is multiplied by --w when that mean share is >= --t.

  python tools/bg_prior.py --pred p.json --frames test_frames/ --map-frames real_a/ real_b/ --out o.json \
      --classes Person,Bicycle,Motorcycle --voxel 0.3 --t 0.4 --w 0.3
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load(path):
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(-1, 3) if raw.size % 3 == 0 else raw.reshape(-1, 4)[:, :3]  # official loader order


def voxel_keys(p, v):
    i = np.floor((p + np.array([60.0, 60.0, 20.0])) / v).astype(np.int64)
    return (i[:, 0] * 1000 + i[:, 1]) * 1000 + i[:, 2]


def frame_keys(path, v):
    return np.unique(voxel_keys(load(path), v))


def share_map(key_lists):
    """key_lists: per-frame unique voxel keys -> (sorted keys, share of frames holding each key)."""
    k, c = np.unique(np.concatenate(key_lists), return_counts=True)
    return k, c / len(key_lists)


def lookup(keys, mk, ms):
    i = np.clip(np.searchsorted(mk, keys), 0, len(mk) - 1)
    return np.where(mk[i] == keys, ms[i], 0.0)


def box_shares(pts, tree, dets, classes, mk, ms, v, ground=0.25, min_pts=5):
    """Mean background share of the above-ground points inside each box of the given classes (None if too few)."""
    out = []
    for d in dets:
        if d["class"] not in classes:
            out.append(None); continue
        x, y, z, l, w, h, yaw = d["box"][:7]
        idx = tree.query_ball_point([x, y], 0.5 * np.hypot(l, w) + 0.1)
        if len(idx) < min_pts:
            out.append(None); continue
        q = pts[idx]; c, s = np.cos(-yaw), np.sin(-yaw)
        dx, dy = q[:, 0] - x, q[:, 1] - y
        u, t = c * dx - s * dy, s * dx + c * dy
        ins = (np.abs(u) <= l / 2 + 0.1) & (np.abs(t) <= w / 2 + 0.1) & (q[:, 2] > z - h / 2 + ground) & (q[:, 2] < z + h / 2 + 0.1)
        if ins.sum() < min_pts:
            out.append(None); continue
        out.append(float(lookup(voxel_keys(q[ins], v), mk, ms).mean()))
    return out


def frame_files(d):
    return sorted(Path(d).glob("*.bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--map-frames", nargs="+", required=True, help="dirs of unlabeled non-forbidden real frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="Person,Bicycle,Motorcycle")
    ap.add_argument("--voxel", type=float, default=0.3)
    ap.add_argument("--t", type=float, default=0.4)
    ap.add_argument("--w", type=float, default=0.3)
    args = ap.parse_args()
    classes = set(args.classes.split(","))
    files = [f for d in args.map_frames for f in frame_files(d)]
    mk, ms = share_map([frame_keys(f, args.voxel) for f in files])
    preds = json.load(open(args.pred))
    fdir = Path(args.frames); hit = tot = 0
    for fid, dets in preds.items():
        p = load(fdir / f"{fid}.bin"); tree = cKDTree(p[:, :2])
        for d, s in zip(dets, box_shares(p, tree, dets, classes, mk, ms, args.voxel)):
            if s is None:
                continue
            tot += 1
            if s >= args.t:
                d["score"] *= args.w; hit += 1
        dets.sort(key=lambda d: -d["score"])
    json.dump(preds, open(args.out, "w"))
    print(f"map frames {len(files)}; down-weighted {hit}/{tot} boxes of {sorted(classes)}")


if __name__ == "__main__":
    main()
