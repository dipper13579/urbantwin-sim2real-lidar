"""Density-matched synthetic frames for the realism score.

UT-LUMPI frames put their points in different places than real LUMPI frames (more on the south
side and on the ground, fewer far away and on objects/vegetation), which drives FPD/EMD. This
re-samples UT points (they all stay synthetic) so that their coarse 3D occupancy matches a real
calibration set: points are drawn with weight p_real(bin) / p_syn(bin) over 4 m x 4 m x z-bands.

Calibrate on one real set and check on a disjoint one:
  python tools/realism_match.py --calib data/lumpi_real/realism_M4_a --check data/lumpi_real/realism_M4_b \
      --out outputs/lumpi/synthetic50_matched
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT / "lumpi" / "scoring_program"))
import metrics  # noqa: E402  official
from score import load_bin, normalize  # noqa: E402  official

TRACKS = {
    "lumpi": (np.array([-40, -40, -5, 40, 40, 5], dtype=np.float32),
              Path(str(RAW_ROOT / 'UT-LUMPI/extracted/UCF-DT-LUMPI-Seq-1/pcd'))),
    "v2x": (np.array([-40, -40, -8, 40, 40, 2], dtype=np.float32),
            Path(str(RAW_ROOT / 'UT-V2X-Real-IC/extracted/UCF-DT-V2X-Real-Seq-1/pcd'))),
}
TRACK = "lumpi"
RANGE, UT_RAW = TRACKS[TRACK]


def bin_index(p, xy_cell, z_edges):
    ix = np.clip(((p[:, 0] - RANGE[0]) / xy_cell).astype(int), 0, int(80 / xy_cell) - 1)
    iy = np.clip(((p[:, 1] - RANGE[1]) / xy_cell).astype(int), 0, int(80 / xy_cell) - 1)
    iz = np.clip(np.searchsorted(z_edges, p[:, 2], side="right") - 1, 0, len(z_edges) - 2)
    n_xy = int(80 / xy_cell)
    return (ix * n_xy + iy) * (len(z_edges) - 1) + iz, n_xy * n_xy * (len(z_edges) - 1)


def crop(p):
    return p[np.all((p >= RANGE[:3]) & (p <= RANGE[3:]), axis=1)]


def score(syn, real):
    cfg = json.loads((KIT_ROOT / TRACK / "reference_data_dummy/config.json").read_text())
    r = metrics.compute_realism(syn, real, tuple(cfg["point_cloud_range"]), seed=1234)
    parts = [normalize(r[k], *cfg["normalization"][k], lower_is_better=True) for k in ("CD", "MMD", "EMD", "FPD")]
    return r, float(np.mean(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True, help="real frames used to estimate the target density")
    ap.add_argument("--check", type=Path, default=None, help="disjoint real frames for an honest score")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xy-cell", type=float, default=4.0)
    ap.add_argument("--z-edges", default="-5,-2.5,-2.0,-1.5,-1.0,0,1,2,3,5")
    ap.add_argument("--points", type=int, default=50_000)
    ap.add_argument("--ut-frames", default="9000:9998:20", help="start:stop:step over UT-LUMPI frame ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill", action="store_true", help="generate points in bins UT never covers")
    ap.add_argument("--shift", default="0,0,0", help="dx,dy,dz applied to UT points (twin-to-site alignment)")
    ap.add_argument("--noise", default="0,0,0", help="Gaussian sigma per axis added to UT points")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--exact", action="store_true",
                    help="match every bin's mass exactly: UT points up to the target count, generated points for the rest")
    ap.add_argument("--track", choices=list(TRACKS), default="lumpi")
    args = ap.parse_args()
    global TRACK, RANGE, UT_RAW
    TRACK = args.track
    RANGE, UT_RAW = TRACKS[TRACK]
    z_edges = np.array([float(v) for v in args.z_edges.split(",")])

    real = [crop(load_bin(p)) for p in sorted(args.calib.glob("*.bin"))]
    a, b, s = (int(v) for v in args.ut_frames.split(":"))
    shift = np.array([float(v) for v in args.shift.split(",")], np.float32)
    noise = np.array([float(v) for v in args.noise.split(",")], np.float32)
    nrng = np.random.default_rng(args.seed + 1)
    ut = []
    for i in range(a, b, s):
        p = np.load(UT_RAW / f"{i:06d}.npy")[:, :3].astype(np.float32) + shift   # align twin to the real site
        p += (nrng.normal(size=p.shape) * noise).astype(np.float32)               # range noise / ground roughness
        ut.append(crop(p))
    ut = ut[:50]

    bi_r, n_bins = bin_index(np.concatenate(real), args.xy_cell, z_edges)
    p_real = np.bincount(bi_r, minlength=n_bins) / len(bi_r)
    bi_s, _ = bin_index(np.concatenate(ut), args.xy_cell, z_edges)
    p_syn = np.bincount(bi_s, minlength=n_bins) / len(bi_s)
    covered = p_real[p_syn > 0].sum()
    print(f"{len(ut)} UT frames, {len(real)} real calib frames; real mass in bins UT also covers: {covered:.3f}")
    w_bin = np.where(p_syn > 0, p_real / np.maximum(p_syn, 1e-12), 0.0)

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # bins with real mass that UT never reaches: filled by a coarse generative model
    # (uniform inside the cell and z-band, mass from the real histogram)
    gap = np.where((p_syn == 0) & (p_real > 0))[0]
    n_xy = int(80 / args.xy_cell)
    nz = len(z_edges) - 1
    out = []
    if args.exact:
        # per frame and bin: target = multinomial(N, p_real); take UT points up to the target, fill the rest
        ut_share = []
        for k, p in enumerate(ut, 1):
            bins_p = bin_index(p, args.xy_cell, z_edges)[0]
            target = rng.multinomial(args.points, p_real)
            order = rng.permutation(len(p))
            bins_sorted = bins_p[order]
            srt = np.argsort(bins_sorted, kind="stable")
            ob, op = bins_sorted[srt], order[srt]
            starts = np.searchsorted(ob, np.arange(n_bins))
            ends = np.searchsorted(ob, np.arange(n_bins), side="right")
            avail = ends - starts
            take = np.minimum(target, avail)
            idx = np.concatenate([op[s:s + t] for s, t in zip(starts[take > 0], take[take > 0])]) if take.sum() else np.zeros(0, int)
            deficit = target - take
            cells = np.repeat(np.arange(n_bins), deficit)
            iz, ixy = cells % nz, cells // nz
            ix, iy = ixy // n_xy, ixy % n_xy
            m = len(cells)
            fx = RANGE[0] + (ix + rng.uniform(size=m)) * args.xy_cell
            fy = RANGE[1] + (iy + rng.uniform(size=m)) * args.xy_cell
            fz = z_edges[iz] + rng.uniform(size=m) * (z_edges[iz + 1] - z_edges[iz])
            q = np.ascontiguousarray(crop(np.concatenate([p[idx], np.stack([fx, fy, fz], 1)]).astype(np.float32)))
            ut_share.append(len(idx) / max(len(q), 1))
            if not args.no_write:
                q.tofile(args.out / f"{k:04d}.bin")
            out.append(q)
        print(f"exact matching: UT share of points {np.mean(ut_share):.3f}")
    for k, p in enumerate(ut if not args.exact else [], 1):
        n_fill = int(round(args.points * (1 - covered))) if args.fill else 0
        w = w_bin[bin_index(p, args.xy_cell, z_edges)[0]]
        n_ut = min(args.points - n_fill, int((w > 0).sum()))
        idx = rng.choice(len(p), n_ut, replace=False, p=w / w.sum())
        parts = [p[idx]]
        if n_fill and len(gap):
            cells = rng.choice(gap, n_fill, p=p_real[gap] / p_real[gap].sum())
            iz, ixy = cells % nz, cells // nz
            ix, iy = ixy // n_xy, ixy % n_xy
            fx = RANGE[0] + (ix + rng.uniform(size=n_fill)) * args.xy_cell
            fy = RANGE[1] + (iy + rng.uniform(size=n_fill)) * args.xy_cell
            fz = z_edges[iz] + rng.uniform(size=n_fill) * (z_edges[iz + 1] - z_edges[iz])
            parts.append(np.stack([fx, fy, fz], 1))
        q = np.ascontiguousarray(crop(np.concatenate(parts).astype(np.float32)))
        if not args.no_write:
            q.tofile(args.out / f"{k:04d}.bin")
        out.append(q)

    base = [p[rng.choice(len(p), args.points, replace=False)] for p in ut]
    for name, ref in (("calib", real), ("check", [crop(load_bin(p)) for p in sorted(args.check.glob("*.bin"))] if args.check else None)):
        if ref is None:
            continue
        r0, s0 = score(base, ref)
        r1, s1 = score(out, ref)
        fmt = lambda r: " ".join(f"{k}={r[k]:.4g}" for k in ("CD", "MMD", "EMD", "FPD"))
        print(f"[{name}] uniform UT : realism={s0:.4f}  {fmt(r0)}")
        print(f"[{name}] matched UT : realism={s1:.4f}  {fmt(r1)}")
    print(f"wrote {len(out)} frames to {args.out}")


if __name__ == "__main__":
    main()
