"""Turn downloaded real LUMPI frames into a challenge-like validation / realism-calibration set.

Mirrors the official test frames: crop to [-40,40]x[-40,40]x[-5,5], uniformly subsample to 50,000
xyz points, float32 N x 3 .bin (verified: the test frames use LUMPI's raw coordinates). GT comes
from Label.csv of the same frame. Frames near forbidden IDs were never downloaded, and
lumpi_real.load_labels drops them again.

  python tools/prepare_lumpi_real.py
Writes data/lumpi_real/points_M{m}/M{m}_{frame}.bin, gt_M{m}.json (submission format + n_pts),
and realism reference folders realism_M4_a / realism_M4_b (50 frames each, disjoint).
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumpi_real import CLASS_NAMES, is_excluded, load_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(str(RAW_ROOT / 'LUMPI-real'))
OUT = WORK_ROOT / "data" / "lumpi_real"
RANGE = np.array([-40, -40, -5, 40, 40, 5], dtype=np.float64)
N_POINTS = 50_000


def main():
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    for m in (4, 6):
        (OUT / f"points_M{m}").mkdir(parents=True, exist_ok=True)
        files = sorted((RAW / f"M{m}" / "lidar").glob("*.ply"))
        frames = np.array([int(f.stem) for f in files])
        assert not is_excluded(frames).any(), "a frame near the forbidden IDs slipped in"
        labels = load_labels(OUT / f"M{m}_Label.csv")
        by_frame = {k: g for k, g in labels.groupby("frame")}
        gt = {}
        for f, fr in zip(files, frames):
            fid = f"M{m}_{fr:06d}"
            dst = OUT / f"points_M{m}" / f"{fid}.bin"
            if dst.exists():
                pts = np.fromfile(dst, np.float32).reshape(-1, 3)
            else:
                v = PlyData.read(str(f))["vertex"]
                pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
                pts = pts[np.all((pts >= RANGE[:3]) & (pts <= RANGE[3:]), axis=1)]
                rng = np.random.default_rng(m * 1_000_000 + int(fr))
                if len(pts) > N_POINTS:
                    pts = pts[rng.choice(len(pts), N_POINTS, replace=False)]
                pts = pts.astype(np.float32)
                pts.tofile(dst)
            g = by_frame.get(int(fr))
            boxes = []
            if g is not None:
                g = g[(g.x.abs() <= 40) & (g.y.abs() <= 40) & (g.z >= -5) & (g.z <= 5)]
                arr = g[["x", "y", "z", "l", "w", "h", "heading"]].to_numpy(np.float32)
                n_in = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(arr)).numpy().sum(1) if len(arr) else []
                for b, c, n in zip(arr, g["cls"].astype(int), n_in):
                    boxes.append({"box": [float(x) for x in b], "class": CLASS_NAMES.get(int(c), f"cls{c}"), "n_pts": int(n)})
            gt[fid] = boxes
        (OUT / f"gt_M{m}.json").write_text(json.dumps(gt))
        n_box = sum(len(v) for v in gt.values())
        print(f"M{m}: {len(gt)} frames, {n_box} in-range boxes -> {OUT}/gt_M{m}.json")

    # two disjoint 50-frame realism reference sets from M4 (calibrate on a, check on b)
    m4 = sorted((OUT / "points_M4").glob("M4_*.bin"))
    for name, sel in (("realism_M4_a", m4[0::2][:50]), ("realism_M4_b", m4[1::2][:50])):
        d = OUT / name
        d.mkdir(exist_ok=True)
        for p in d.glob("*.bin"):
            p.unlink()
        for k, p in enumerate(sel, 1):
            (d / f"{k:04d}.bin").write_bytes(p.read_bytes())
        print(f"{name}: {len(sel)} frames")


if __name__ == "__main__":
    main()
