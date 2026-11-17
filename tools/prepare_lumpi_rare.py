"""Rare-class (Bus / Motorcycle) real LUMPI validation sets from the frames fetched by setup/18_fetch_rare.sh.

Same processing as prepare_lumpi_real.py (crop [-40,40]x[-40,40]x[-5,5], 50k-point subsample, GT + n_pts from
Label.csv with forbidden-adjacent frames dropped), written to separate sets so the original M4 / M6 val sets
stay comparable: data/lumpi_real/points_M{m}r/M{m}r_{frame}.bin and gt_M{m}r.json.

  python tools/prepare_lumpi_rare.py
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
        tag = f"M{m}r"
        (OUT / f"points_{tag}").mkdir(parents=True, exist_ok=True)
        files = sorted((RAW / tag / "lidar").glob("*.ply"))
        frames = np.array([int(f.stem) for f in files])
        assert not is_excluded(frames).any(), "a frame near the forbidden IDs slipped in"
        labels = load_labels(OUT / f"M{m}_Label.csv")
        by_frame = {k: g for k, g in labels.groupby("frame")}
        gt = {}
        for f, fr in zip(files, frames):
            fid = f"{tag}_{fr:06d}"
            dst = OUT / f"points_{tag}" / f"{fid}.bin"
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
        (OUT / f"gt_{tag}.json").write_text(json.dumps(gt))
        allb = [b for v in gt.values() for b in v if b["n_pts"] >= 1]
        print(f"{tag}: {len(gt)} frames; boxes with >=1 pt: "
              + ", ".join(f"{c}={sum(b['class'] == c for b in allb)}" for c in CLASS_NAMES.values()))


if __name__ == "__main__":
    main()
