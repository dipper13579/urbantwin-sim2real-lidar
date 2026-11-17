"""Long-vehicle geometry for LUMPI (Truck / Bus): refit the boxes to the points they sit on and re-score the class from the measured
shape. Works on a finished prediction file; every frame is processed on its own, from its own points and boxes only.

 1. measure   every Truck / Bus box with score >= MIN_SCORE: the contiguous run of above-ground points along the box heading that
              passes through the box centre gives the length L and the point top of the object under the box.
 2. refit     boxes on a run of L >= 9 m ("long", both classes) and Truck boxes on a run of 5-9 m with top >= 2.4 m ("mid") are replaced
              by a box fitted to the points: PCA heading, extent along and across, ground to top, plus pads. Refitted boxes of one class
              whose centres fall inside each other are reduced to the best scored one.
 3. re-score  Bus boxes by the measured shape (w_rb). With --share the long objects are scored per object instead: objectness = best
              class score, split between Truck and Bus by a shape prior (bus-like cell: top 2.8-3.45 m and L 10.5-13.5 m), and both
              classes get the object's box.

The detector caps long vehicles at ~11-12 m and its Truck / Bus class scores carry little class information on long objects (real val,
validation analysis, validation analysis). The pads, the width limits and the prior values are box-size statistics of labelled real frames
outside the 300-frame margin around the forbidden ids (validation analysis, validation analysis); nothing else here comes from labels.

Tried and dropped (val): mid refit of Bus boxes (Bus down on all four sets), per-class pads (same as pooled), down-weighting Truck
boxes with a low point top (fails M6r, site-6 trucks are low).

  python tools/lv_geo.py --pred outputs/lumpi/X/predictions.json --frames starting_kit/lumpi/detection_test_frames \
      --out outputs/lumpi/XG/predictions.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bg_prior import load

LV = ("Truck", "Bus")
MIN_SCORE = 0.05
# label - measurement medians over refit objects matched to a Truck / Bus label on M4, M4r, M6, M6r (validation analysis, "cal all")
LONG = dict(pl=0.60, pw=0.13, ph=0.25, pb=-0.07, wlo=2.75, whi=3.67, wdef=3.20)
MID = dict(pl=0.61, pw=0.16, ph=0.05, pb=0.05, wlo=2.20, whi=3.39, wdef=2.55)
PRIOR = None        # share of Bus among labelled long objects inside / outside the bus-like cell; set from validation analysis if --share is adopted
EXTRA = ("fit", "kind", "f")


def measure(pts, tree, x, y, yaw, w):
    """Run of points above ground + 0.4 m along the heading through (x, y): n, L, top (p98), ground g; None with < 20 points nearby."""
    p = pts[tree.query_ball_point([x, y], 14.0)]
    if len(p) < 20:
        return None
    c, s = np.cos(yaw), np.sin(yaw); ex, ey = p[:, 0] - x, p[:, 1] - y
    lx, ly = ex * c + ey * s, -ex * s + ey * c
    near = p[np.hypot(ex, ey) < 4.0]
    g = float(np.percentile(near[:, 2], 3) if len(near) >= 30 else np.percentile(p[:, 2], 3))
    zr = p[:, 2] - g
    hw = min(max(w / 2, 1.0) + 0.2, 1.45)
    m = (np.abs(ly) < hw) & (zr > 0.4) & (zr < 4.8)
    if m.sum() < 5:
        return dict(n=int(m.sum()), L=0.0, top=0.0, g=g)
    order = np.argsort(lx[m]); l = lx[m][order]; z = zr[m][order]
    i0 = int(np.argmin(np.abs(l)))
    if abs(l[i0]) > 1.5:
        return dict(n=0, L=0.0, top=0.0, g=g)
    gap = np.diff(l) > 0.8
    a = b = i0
    while a > 0 and not gap[a - 1]:
        a -= 1
    while b < len(l) - 1 and not gap[b]:
        b += 1
    return dict(n=b - a + 1, L=float(l[b] - l[a]), top=float(np.percentile(z[a:b + 1], 98)), g=g, la=float(l[a]), lb=float(l[b]), loc=(lx, ly, zr))


def fit_raw(r, box):
    """Raw fit of a run, no pads: PCA heading, extent along / across, top, roof flatness. None with < 30 points."""
    lx, ly, zr = r["loc"]; yaw = box[6]
    m = (lx > r["la"] - 0.4) & (lx < r["lb"] + 0.4) & (np.abs(ly) < 2.3) & (zr > 0.5) & (zr < 4.8)
    if m.sum() < 30:
        return None
    qx, qy, qz = lx[m], ly[m], zr[m]
    o = np.argsort(qy); sy = qy[o]
    core = qy[np.abs(qy) < 1.0]
    i0 = int(np.argmin(np.abs(sy - (np.median(core) if len(core) else 0.0))))
    gaps = np.diff(sy) > 0.4
    a = b = i0
    while a > 0 and not gaps[a - 1]:
        a -= 1
    while b < len(sy) - 1 and not gaps[b]:
        b += 1
    k = o[a:b + 1]; qx, qy, qz = qx[k], qy[k], qz[k]
    if len(qx) < 30:
        return None
    ev, evec = np.linalg.eigh(np.cov(np.vstack([qx, qy])))
    d = float(np.arctan2(evec[1, 1], evec[0, 1]))
    d = (d + np.pi / 2) % np.pi - np.pi / 2
    if abs(d) < np.radians(12) and ev[1] > 6 * ev[0]:
        c, s = np.cos(d), np.sin(d); qx, qy = qx * c + qy * s, -qx * s + qy * c; yaw = yaw + d
    x0, x1 = np.percentile(qx, [0.5, 99.5]); y0, y1 = np.percentile(qy, [1, 99])
    top = float(np.percentile(qz, 99))
    nb = max(int(np.ceil(x1 - x0)), 1); idx = np.clip((qx - x0).astype(int), 0, nb - 1)
    tops = np.array([np.percentile(qz[idx == i], 95) for i in range(nb) if (idx == i).sum() >= 5])
    return dict(x=box[0], y=box[1], yaw=float(yaw), x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1), top=top, g=r["g"], bw=box[4],
                flat=float(np.mean(tops >= top - 0.4)) if len(tops) else 0.0)


def make_box(f, cal):
    """Box of a raw fit with pads. A side seen less than 2 m wide is one-sided: keep the lateral centre, width >= the default."""
    wm = f["y1"] - f["y0"]
    if wm >= 2.0:
        cy, w = (f["y0"] + f["y1"]) / 2, float(np.clip(wm + cal["pw"], cal["wlo"], cal["whi"]))
    else:
        cy, w = 0.0, max(f["bw"], cal["wdef"])
    cx = (f["x0"] + f["x1"]) / 2
    bot = f["g"] + cal["pb"]; top = f["g"] + f["top"] + cal["ph"]
    c, s = np.cos(f["yaw"]), np.sin(f["yaw"])
    return [float(f["x"] + cx * c - cy * s), float(f["y"] + cx * s + cy * c), float((bot + top) / 2), float(f["x1"] - f["x0"] + cal["pl"]), w,
            float(top - bot), f["yaw"]]


def zone(r):
    if r is None or r["n"] < 5:
        return "nopts"
    if r["top"] < 1.85:
        return "car"
    if r["top"] < 2.4:
        return "transp"
    if r["top"] < 2.8 and r["L"] < 7.5:
        return "hiroof"
    return "lorry"


def w_rb(r):
    """Bus score weight from the run under the box: a bus is long, and its point top is between 2.5 and 3.5 m."""
    if zone(r) == "nopts":
        return 1.0
    return (0.3 if r["L"] < 7.0 else 1.0) * (0.2 if r["top"] < 2.5 else 0.4 if r["top"] > 3.5 else 1.0)


def bus_cell(f):
    return 2.8 <= f["top"] < 3.45 and 10.5 <= f["x1"] - f["x0"] < 13.5


def inside(box, x, y, m=0.3):
    c, s = np.cos(box[6]), np.sin(box[6]); ex, ey = x - box[0], y - box[1]
    return abs(ex * c + ey * s) < box[3] / 2 + m and abs(-ex * s + ey * c) < box[4] / 2 + m


def dedup(dets):
    """Class-wise: of the refitted boxes whose centres fall inside each other keep the best scored one."""
    out = []
    for c in LV:
        kept = []
        for d in sorted((d for d in dets if d["class"] == c and d.get("fit")), key=lambda d: -d["score"]):
            b = d["box"]
            if not any(inside(k["box"], b[0], b[1], 0.0) or inside(b, k["box"][0], k["box"][1], 0.0) for k in kept):
                kept.append(d)
        out += kept
    return out + [d for d in dets if not (d["class"] in LV and d.get("fit"))]


def objects(dets):
    """Class-agnostic grouping of the long refitted boxes of one frame: one object per run, with the best score of each class."""
    objs = []
    for d in sorted((d for d in dets if d.get("kind") == "long"), key=lambda d: -d["score"]):
        b = d["box"]
        for o in objs:
            if inside(o["box"], b[0], b[1], 0.0) or inside(b, o["box"][0], o["box"][1], 0.0):
                o["s"][d["class"]] = max(o["s"].get(d["class"], 0.0), d["score"]); o["m"].append(d); break
        else:
            objs.append(dict(box=b, f=d["f"], s={d["class"]: d["score"]}, m=[d]))
    return objs


def share_scores(dets, prior, complete=True):
    """Per long object: the likelier class of its shape cell keeps the objectness, the other gets it times the prior odds.
    complete: objectness = best class score and both classes get the object's box; otherwise the existing boxes are only re-weighted."""
    res = [d for d in dets if d.get("kind") != "long"]
    for o in objects(dets):
        r = prior["r_in"] if bus_cell(o["f"]) else prior["r_out"]
        w = {"Bus": r / max(r, 1 - r), "Truck": (1 - r) / max(r, 1 - r)}
        if complete:
            s = max(o["s"].values())
            res += [{"box": o["box"], "score": float(s * w[c]), "class": c, "fit": True, "kind": "long", "f": o["f"]} for c in LV]
        else:
            res += [dict(d, score=float(d["score"] * w[d["class"]])) for d in o["m"]]
    return res


def analyse(preds, fdir):
    """Measure every Truck / Bus box with score >= MIN_SCORE once; the raw fit is kept for runs >= 9 m ("long") and for 5-9 m with top >= 2.4 ("mid")."""
    recs = {}
    for fid in sorted(preds):
        pts = load(Path(fdir) / f"{fid}.bin"); tree = cKDTree(pts[:, :2]); rows = []
        for d in preds[fid]:
            if d["class"] not in LV or d["score"] < MIN_SCORE:
                rows.append(dict(d=d, lv=False)); continue
            b = d["box"]; r = measure(pts, tree, b[0], b[1], b[6], b[4]); z = zone(r); kind = None
            if z != "nopts" and r["L"] >= 9.0:
                kind = "long"
            elif z != "nopts" and r["L"] >= 5.0 and r["top"] >= 2.4:
                kind = "mid"
            f = fit_raw(r, b) if kind else None
            rows.append(dict(d=d, lv=True, r=None if r is None else dict(n=r["n"], L=r["L"], top=r["top"]), f=f, kind=kind if f else None))
        recs[fid] = rows
    return recs


def build(recs, long_cal=LONG, mid_cal=MID, mid_classes=("Truck",), rb=True, share=None, complete=True):
    out = {}
    for fid, rows in recs.items():
        res = []
        for rec in rows:
            d = rec["d"]
            if not rec["lv"]:
                res.append(d); continue
            kind = rec["kind"]
            if (kind == "long" and long_cal is None) or (kind == "mid" and (mid_cal is None or d["class"] not in mid_classes)):
                kind = None
            w = w_rb(rec["r"]) if rb and d["class"] == "Bus" and not (share and kind == "long") else 1.0
            nd = dict(d, score=float(d["score"] * w))
            if kind:
                nd.update(box=make_box(rec["f"], long_cal if kind == "long" else mid_cal), fit=True, kind=kind, f=rec["f"])
            res.append(nd)
        res = dedup(res)
        out[fid] = share_scores(res, share, complete) if share else res
    return out


def strip(out):
    return {fid: [{k: v for k, v in d.items() if k not in EXTRA} for d in ds] for fid, ds in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mid", action="store_true")
    ap.add_argument("--no-rb", action="store_true")
    ap.add_argument("--share", action="store_true", help="score long objects per object with the shape prior")
    args = ap.parse_args()
    if args.share and PRIOR is None:
        ap.error("--share needs PRIOR")
    preds = json.loads(Path(args.pred).read_text())
    out = build(analyse(preds, args.frames), LONG, None if args.no_mid else MID, rb=not args.no_rb, share=PRIOR if args.share else None)
    n = len(out); fit = [d for ds in out.values() for d in ds if d.get("fit")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(strip(out)))
    print(f"{args.out}: {n} frames, refitted boxes per frame long {sum(d['kind'] == 'long' for d in fit) / max(n, 1):.2f} "
          f"mid {sum(d['kind'] == 'mid' for d in fit) / max(n, 1):.2f}")


if __name__ == "__main__":
    main()
