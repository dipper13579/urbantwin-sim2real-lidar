"""Procedural street clutter (unlabelled negatives) for UrbanTwin training frames.

On the real V2X-Real site two thirds of the detector's pedestrian boxes sit on nothing labelled: low, sparse
clusters (median 5 body points, 0.86 m tall) such as bollards, bins, hydrants, sign posts, cones, shrubs and
parked bikes. UT-V2X scenes have none of these, so the detector never saw them as background. Objects are placed
like synthetic people (open ground, clear of obstacles and GT boxes), sampled on their surface facing the nearest
sensor, thinned to the scene's point density, and added WITHOUT a GT box.

Config (DATA_CONFIG.SYNTH_CLUTTER):
    ENABLED, PROB (chance per training frame), NUM [lo, hi] (objects per frame), MAX_RANGE (m from origin),
    SENSORS [[x, y, z], ...], COUNT_AT_10M / COUNT_DECAY_M / COUNT_SIGMA (point-count model of a person, scaled by
    the object's visible size), GROUND_TOL, LOCAL_GROUND, OBSTACLE_CLEAR (m to the nearest obstacle point,
    default 0.7), MIN_SEP (m between clutter objects, default 1.0), BOX_CLEAR (m added to the GT box radius,
    default 0.6), KINDS (subset of post, bollard, bin, bench, shrub, bike, cone, crate; default all).
"""
import numpy as np

from .synth_person import _capsule, _ellipsoid, _rotz

KINDS = ("post", "bollard", "bin", "bench", "shrub", "bike", "cone", "crate")


def _box_surface(l, w, h, n, rng):
    """n points on the four side faces and the top of an l x w x h box standing at the origin, with normals."""
    faces = [  # (area weight, sampler)
        (w * h, lambda m: (np.c_[np.full(m, l / 2), rng.uniform(-w / 2, w / 2, m), rng.uniform(0, h, m)], [1, 0, 0])),
        (w * h, lambda m: (np.c_[np.full(m, -l / 2), rng.uniform(-w / 2, w / 2, m), rng.uniform(0, h, m)], [-1, 0, 0])),
        (l * h, lambda m: (np.c_[rng.uniform(-l / 2, l / 2, m), np.full(m, w / 2), rng.uniform(0, h, m)], [0, 1, 0])),
        (l * h, lambda m: (np.c_[rng.uniform(-l / 2, l / 2, m), np.full(m, -w / 2), rng.uniform(0, h, m)], [0, -1, 0])),
        (l * w, lambda m: (np.c_[rng.uniform(-l / 2, l / 2, m), rng.uniform(-w / 2, w / 2, m), np.full(m, h)], [0, 0, 1])),
    ]
    tot = sum(a for a, _ in faces); pts = []; nrm = []
    for a, f in faces:
        m = max(int(n * a / tot), 2)
        p, v = f(m); pts.append(p); nrm.append(np.tile(np.asarray(v, float), (m, 1)))
    return np.concatenate(pts), np.concatenate(nrm)


def _cylinder(r, h, n, rng, z0=0.0):
    pts, nrm = _capsule(np.array([0, 0, z0]), np.array([0, 0, z0 + h]), r, int(n * 0.85), rng)
    top, _ = _ellipsoid(np.array([0, 0, z0 + h]), np.array([r, r, 0.01]), max(int(n * 0.15), 2), rng)
    return np.concatenate([pts, top]), np.concatenate([nrm, np.tile([0, 0, 1.0], (len(top), 1))])


def make_object(kind, rng, n=1200):
    """Surface points + normals of one clutter object standing at the origin, and its (l, w, h)."""
    if kind == "post":
        r, h = rng.uniform(0.04, 0.12), rng.uniform(0.9, 3.5)
        pts, nrm = _cylinder(r, h, n, rng)
        if rng.uniform() < 0.5:  # sign plate near the top, facing a random direction
            pw, ph = rng.uniform(0.4, 0.8), rng.uniform(0.3, 0.6)
            p, v = _box_surface(0.03, pw, ph, int(n * 0.5), rng)
            p[:, 2] += h - ph - rng.uniform(0.0, 0.3)
            R = _rotz(rng.uniform(-np.pi, np.pi)); p = p @ R.T; v = v @ R.T
            pts, nrm = np.concatenate([pts, p]), np.concatenate([nrm, v])
            return pts, nrm, (max(2 * r, pw), max(2 * r, pw), h)
        return pts, nrm, (2 * r, 2 * r, h)
    if kind == "bollard":
        r, h = rng.uniform(0.08, 0.2), rng.uniform(0.5, 1.1)
        pts, nrm = _cylinder(r, h, n, rng); return pts, nrm, (2 * r, 2 * r, h)
    if kind == "cone":
        r, h = rng.uniform(0.12, 0.2), rng.uniform(0.45, 0.8)
        pts, nrm = _cylinder(r, h, n, rng); pts[:, :2] *= (1.0 - 0.7 * pts[:, 2:3] / h); return pts, nrm, (2 * r, 2 * r, h)
    if kind == "bin":
        l, w, h = rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.8), rng.uniform(0.7, 1.3)
        pts, nrm = _box_surface(l, w, h, n, rng); return pts, nrm, (l, w, h)
    if kind == "crate":
        l, w, h = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.6)
        pts, nrm = _box_surface(l, w, h, n, rng); return pts, nrm, (l, w, h)
    if kind == "bench":
        l, w, h = rng.uniform(1.2, 2.0), rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.9)
        pts, nrm = _box_surface(l, w, h, n, rng); return pts, nrm, (l, w, h)
    if kind == "bike":  # parked bicycle / scooter, no rider
        l, w, h = rng.uniform(1.5, 1.8), rng.uniform(0.2, 0.45), rng.uniform(0.7, 1.05)
        pts, nrm = _box_surface(l, w, h, n, rng)
        keep = rng.uniform(size=len(pts)) < 0.5  # frames are see-through: sparse
        return pts[keep], nrm[keep], (l, w, h)
    if kind == "shrub":
        rx, ry, rz = rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.7)
        pts, nrm = _ellipsoid(np.array([0, 0, rz]), np.array([rx, ry, rz]), n, rng)
        inner = pts + rng.normal(0, 0.08, pts.shape) * np.array([rx, ry, rz])  # leafy: volume noise
        pts = np.concatenate([pts, inner[: n // 2]]); nrm = np.concatenate([nrm, nrm[: n // 2]])
        return pts, nrm, (2 * rx, 2 * ry, 2 * rz)
    raise ValueError(kind)


def insert_clutter(points, gt_boxes, cfg, rng):
    """Return points with unlabelled clutter objects added (gt_boxes untouched; used only to keep clear)."""
    if rng.uniform() > cfg.get('PROB', 0.8):
        return points
    n_obj = rng.integers(cfg.NUM[0], cfg.NUM[1] + 1)
    kinds = list(cfg.get('KINDS', KINDS))
    xyz = points[:, :3]
    tol = cfg.get('GROUND_TOL', 0.25)
    if cfg.get('LOCAL_GROUND', False):
        cell = np.floor(xyz[:, :2]).astype(np.int64)
        key = (cell[:, 0] + 1000) * 10000 + (cell[:, 1] + 1000)
        order = np.argsort(key, kind="stable")
        uniq, start = np.unique(key[order], return_index=True)
        zmin = np.minimum.reduceat(xyz[order, 2], start)[np.searchsorted(uniq, key)]
        rel = xyz[:, 2] - zmin
        ground = xyz[rel < tol]
        above = xyz[(rel > 0.3) & (rel < 2.3)]
    else:
        ground_z = np.median(xyz[:, 2][xyz[:, 2] <= np.percentile(xyz[:, 2], 30)])
        ground = xyz[np.abs(xyz[:, 2] - ground_z) < tol]
        above = xyz[(xyz[:, 2] > ground_z + 0.3) & (xyz[:, 2] < ground_z + 2.3)]
    if len(ground) < 100:
        return points

    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 7)
    occupied = gt_boxes[:, :2]; occ_r = np.maximum(gt_boxes[:, 3], gt_boxes[:, 4]) / 2 + cfg.get('BOX_CLEAR', 0.6)
    min_sep, clear = cfg.get('MIN_SEP', 1.0), cfg.get('OBSTACLE_CLEAR', 0.7)
    sensors = cfg.get('SENSORS', None)
    placed = []; extra = []; tries = 0
    while len(placed) < n_obj and tries < n_obj * 15:
        tries += 1
        c = ground[rng.integers(len(ground)), :2] + rng.normal(0, 0.3, 2)
        dist = np.hypot(*c)
        if dist > cfg.get('MAX_RANGE', 38.0) or dist < 3.0:
            continue
        if len(occupied) and np.any(np.hypot(*(occupied - c).T) < occ_r):
            continue
        if any(np.hypot(*(c - p)) < min_sep for p in placed):
            continue
        if len(above) and np.min(np.hypot(*(above[:, :2] - c).T)) < clear:
            continue
        near = ground[np.hypot(*(ground[:, :2] - c).T) < 1.5]
        if len(near) < 3:
            continue
        gz = np.median(near[:, 2])
        if cfg.get('LOCAL_GROUND', False):
            wide = ground[np.hypot(*(ground[:, :2] - c).T) < 5.0]
            if gz - np.percentile(wide[:, 2], 10) > 0.5:
                continue
        kind = kinds[rng.integers(len(kinds))]
        surf, nrm, (l, w, h) = make_object(kind, rng)
        R = _rotz(rng.uniform(-np.pi, np.pi))
        surf = surf @ R.T; nrm = nrm @ R.T
        world = surf + np.array([c[0], c[1], gz])
        if sensors:
            s = np.asarray(sensors, dtype=float)
            sp = s[np.argmin(np.linalg.norm(s[:, :2] - c, axis=1))]
        else:
            sp = np.array([0.0, 0.0, gz + 2.0])
        v = sp - np.array([c[0], c[1], gz + h / 2])
        vis = (nrm @ (v / np.linalg.norm(v))) > 0.1
        world = world[vis]
        if len(world) < 3:
            continue
        # point count: a person's count model scaled by the visible size relative to a 1.72 m person
        size = (h * max(l, w)) / (1.72 * 0.5)
        med = cfg.COUNT_AT_10M * size * np.exp(-(dist - 10.0) / cfg.COUNT_DECAY_M)
        k = int(np.clip(rng.lognormal(np.log(max(med, 1.0)), cfg.COUNT_SIGMA), 2, len(world)))
        world = world[rng.choice(len(world), k, replace=False)]
        world += rng.normal(0, 0.015, world.shape)
        placed.append(c); extra.append(world)
    if not extra:
        return points
    extra = np.concatenate(extra).astype(points.dtype)
    if points.shape[1] > 3:
        extra = np.concatenate([extra, np.zeros((len(extra), points.shape[1] - 3), points.dtype)], 1)
    return np.concatenate([points, extra])
