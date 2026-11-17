"""Procedural pedestrians for UrbanTwin scenes (UT-LUMPI has no Person labels).

A person is a few primitives (legs, torso, arms, head) sampled on their surface, kept only where
the surface faces one or two virtual viewpoints, then thinned to a point count that follows the
scene's density at that distance. People are placed on open ground (ground points present, no
obstacle points nearby, no overlap with existing boxes) and get a 'Person' GT box.

Config (DATA_CONFIG.SYNTH_PERSON):
    ENABLED, PROB (chance per training frame), NUM [lo, hi], HEIGHT [lo, hi],
    BOX_LW [[lo, hi], [lo, hi]] (box dx, dy), BOX_H [lo, hi] (box height; default = body height), SENSORS [[x, y, z], ...] (LiDAR positions in the
    frame; people are seen from the nearest one or two), COUNT_AT_10M (median points of a person
    10 m from the origin), COUNT_DECAY_M (e-folding distance), COUNT_SIGMA (log-normal spread),
    MAX_RANGE (m from origin), GROUND_TOL (m), CLASS_NAME (default 'Person')
Crowds (all optional, off by default): GROUP_PROB (chance that a placed person starts a group),
    GROUP_SIZE [lo, hi] (people per group), GROUP_RADIUS (m), MIN_SEP (m between synthetic people,
    default 0.8), OBSTACLE_CLEAR (m to the nearest obstacle point, default 0.7), OCCLUSION (drop the
    points of a person hidden behind another synthetic person, seen from the nearest sensor)
"""
import numpy as np


def _capsule(p0, p1, r, n, rng):
    """n points on a vertical-ish capsule surface between p0 and p1 (cylinder body only) + normals."""
    axis = p1 - p0
    length = np.linalg.norm(axis) + 1e-6
    a = axis / length
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    t = rng.uniform(0, 1, n)
    th = rng.uniform(0, 2 * np.pi, n)
    nrm = np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
    return p0 + np.outer(t, axis) + r * nrm, nrm


def _ellipsoid(c, radii, n, rng):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = c + d * radii
    nrm = d / radii
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm


def person_surface(height, rng, n=1500):
    """Surface samples + outward normals of a person standing at the origin, facing +x."""
    h = height
    hip, sh = 0.50 * h, 0.81 * h
    stride = rng.uniform(-0.22, 0.22)          # walking pose: legs apart along +-x
    lat = 0.09
    parts = []
    # legs
    for side, s in ((1, stride), (-1, -stride)):
        parts.append(_capsule(np.array([s, side * lat, 0.05]), np.array([0, side * lat, hip]), 0.075, int(n * 0.22), rng))
    # torso (elliptic: depth 0.11 along x, width 0.18 along y)
    tp, tn = _capsule(np.array([0, 0, hip]), np.array([0, 0, sh]), 1.0, int(n * 0.30), rng)
    tp[:, 0] *= 0.11; tp[:, 1] *= 0.18
    tn[:, 0] /= 0.11; tn[:, 1] /= 0.18
    tn /= np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9
    parts.append((tp, tn))
    # arms, swinging opposite to the legs
    for side, s in ((1, -stride * 0.6), (-1, stride * 0.6)):
        parts.append(_capsule(np.array([0, side * 0.22, sh - 0.03]), np.array([s, side * 0.24, hip - 0.05]), 0.045, int(n * 0.08), rng))
    # head
    parts.append(_ellipsoid(np.array([0, 0, 0.935 * h]), np.array([0.09, 0.08, 0.11]), int(n * 0.10), rng))
    pts = np.concatenate([p for p, _ in parts])
    nrm = np.concatenate([q for _, q in parts])
    return pts, nrm


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def nearest_sensor_dist(xy, sensors):
    """2D distance from each xy (N, 2) to its nearest sensor; distance to the origin without sensors."""
    xy = np.atleast_2d(xy)
    if sensors is None or len(sensors) == 0:
        return np.hypot(xy[:, 0], xy[:, 1])
    s = np.asarray(sensors, dtype=float)[:, :2]
    return np.min(np.linalg.norm(xy[:, None, :] - s[None], axis=2), axis=1)


def occlude_people(people, sensors, radius=0.22):
    """Drop points of each person that lie behind another (closer) person, seen from its nearest sensor.

    people: list of dicts with 'pts' (N, 3), 'c' (2,), 'gz', 'h'. A point is hidden when the 2D ray from the
    sensor to it passes within `radius` of a closer person's axis, at a height inside that person's body.
    """
    if len(people) < 2:
        return people
    s_all = np.asarray(sensors, dtype=float) if sensors else np.array([[0.0, 0.0, 2.0]])
    for a in people:
        s = s_all[np.argmin(np.linalg.norm(s_all[:, :2] - a['c'], axis=1))]
        da = np.hypot(*(a['c'] - s[:2]))
        hidden = np.zeros(len(a['pts']), bool)
        for b in people:
            if b is a or np.hypot(*(b['c'] - s[:2])) >= da:
                continue
            d = a['pts'][:, :2] - s[:2]                                  # rays sensor -> point (2D)
            t = np.clip(((b['c'] - s[:2]) @ d.T) / np.maximum((d * d).sum(1), 1e-9), 0, 1)
            close = np.hypot(*(s[:2] + t[:, None] * d - b['c']).T) < radius
            zr = s[2] + t * (a['pts'][:, 2] - s[2])                     # ray height where it passes b
            hidden |= close & (t < 0.98) & (zr > b['gz']) & (zr < b['gz'] + b['h'])
        if (~hidden).sum() >= 3:                                         # keep at least a few points
            a['pts'] = a['pts'][~hidden]
    return people


def insert_persons(points, gt_boxes, gt_names, cfg, rng):
    """Return (points, gt_boxes, gt_names) with procedural pedestrians added."""
    if rng.uniform() > cfg.get('PROB', 0.8):
        return points, gt_boxes, gt_names
    n_people = rng.integers(cfg.NUM[0], cfg.NUM[1] + 1)
    xyz = points[:, :3]
    tol = cfg.get('GROUND_TOL', 0.25)
    if cfg.get('LOCAL_GROUND', False):
        # sloped / multi-level sites: ground = points near the lowest point of their 1 m cell
        cell = np.floor(xyz[:, :2]).astype(np.int64)
        key = (cell[:, 0] + 1000) * 10000 + (cell[:, 1] + 1000)
        order = np.argsort(key, kind="stable")
        uniq, start = np.unique(key[order], return_index=True)
        zmin = np.minimum.reduceat(xyz[order, 2], start)[np.searchsorted(uniq, key)]
        rel = xyz[:, 2] - zmin
        ground = xyz[rel < tol]
        above = xyz[(rel > 0.3) & (rel < 2.3)]
    else:
        # `<=`: UT ground is perfectly flat, so the 30th percentile often equals the ground height itself
        ground_z = np.median(xyz[:, 2][xyz[:, 2] <= np.percentile(xyz[:, 2], 30)])
        ground = xyz[np.abs(xyz[:, 2] - ground_z) < tol]
        above = xyz[(xyz[:, 2] > ground_z + 0.3) & (xyz[:, 2] < ground_z + 2.3)]
    if len(ground) < 100:
        return points, gt_boxes, gt_names

    people = []
    occupied = [b for b in gt_boxes[:, :2]] if len(gt_boxes) else []
    occ_r = [max(b[3], b[4]) / 2 + 0.6 for b in gt_boxes] if len(gt_boxes) else []
    min_sep, clear = cfg.get('MIN_SEP', 0.8), cfg.get('OBSTACLE_CLEAR', 0.7)
    group_c, group_left = None, 0
    tries = 0
    while len(people) < n_people and tries < n_people * 15:
        tries += 1
        if group_left > 0:                                      # next member of the current group
            group_left -= 1
            ang, rad = rng.uniform(-np.pi, np.pi), cfg.get('GROUP_RADIUS', 1.2) * np.sqrt(rng.uniform())
            c = group_c + rad * np.array([np.cos(ang), np.sin(ang)])
        else:
            c = ground[rng.integers(len(ground)), :2] + rng.normal(0, 0.3, 2)
            if cfg.get('GROUP_PROB', 0.0) > 0 and rng.uniform() < cfg.GROUP_PROB:
                group_c = c
                group_left = rng.integers(cfg.GROUP_SIZE[0], cfg.GROUP_SIZE[1] + 1) - 1
        dist = np.hypot(*c)
        if dist > cfg.get('MAX_RANGE', 38.0):
            continue
        if any(np.hypot(*(c - o)) < r for o, r in zip(occupied, occ_r)):
            continue
        if any(np.hypot(*(c - p['c'])) < min_sep for p in people):
            continue
        if len(above) and np.min(np.hypot(*(above[:, :2] - c).T)) < clear:
            continue                                            # obstacle (pole, wall, vegetation...)
        near = ground[np.hypot(*(ground[:, :2] - c).T) < 1.5]
        if len(near) < 3:
            continue                                            # no line of sight to this ground
        gz = np.median(near[:, 2])
        if cfg.get('LOCAL_GROUND', False):
            wide = ground[np.hypot(*(ground[:, :2] - c).T) < 5.0]
            if gz - np.percentile(wide[:, 2], 10) > 0.5:
                continue                                        # likely a roof / raised object, not walkable ground

        h = rng.uniform(*cfg.HEIGHT)
        surf, nrm = person_surface(h, rng)
        heading = rng.uniform(-np.pi, np.pi)
        R = _rotz(heading)
        surf = surf @ R.T
        nrm = nrm @ R.T
        world = surf + np.array([c[0], c[1], gz])
        # visible surface: faces the nearest sensor, plus the second nearest with some probability
        sensors = cfg.get('SENSORS', None)
        if sensors:
            s = np.asarray(sensors, dtype=float)
            order = np.argsort(np.linalg.norm(s[:, :2] - c, axis=1))
            views = [s[order[0]]] + ([s[order[1]]] if len(s) > 1 and rng.uniform() < 0.6 else [])
        else:
            views = [np.array([0.0, 0.0, gz + 2.0])]
        vis = np.zeros(len(world), bool)
        mid = np.array([c[0], c[1], gz + h / 2])
        for sp in views:
            v = sp - mid
            vis |= (nrm @ (v / np.linalg.norm(v))) > 0.1
        world = world[vis]
        # point count following the scene density (log-normal around an exponential falloff in the
        # distance from the origin: with several LiDARs around the site this fits UT objects better
        # than the distance to the nearest sensor)
        med = cfg.COUNT_AT_10M * np.exp(-(dist - 10.0) / cfg.COUNT_DECAY_M)
        k = int(np.clip(rng.lognormal(np.log(max(med, 1.0)), cfg.COUNT_SIGMA), 3, len(world)))
        world = world[rng.choice(len(world), k, replace=False)]
        world += rng.normal(0, 0.015, world.shape)

        l, w = rng.uniform(*cfg.BOX_LW[0]), rng.uniform(*cfg.BOX_LW[1])
        bh = rng.uniform(*cfg.BOX_H) if cfg.get('BOX_H', None) else h   # annotation height (with margin)
        people.append({'pts': world, 'c': c, 'gz': gz, 'h': h, 'box': [c[0], c[1], gz + bh / 2, l, w, bh, heading]})

    if not people:
        return points, gt_boxes, gt_names
    if cfg.get('OCCLUSION', False):
        people = occlude_people(people, cfg.get('SENSORS', None))
    new_boxes = [p['box'] for p in people]
    extra = np.concatenate([p['pts'] for p in people]).astype(points.dtype)
    if points.shape[1] > 3:
        extra = np.concatenate([extra, np.zeros((len(extra), points.shape[1] - 3), points.dtype)], 1)
    points = np.concatenate([points, extra])
    gt_boxes = np.concatenate([gt_boxes.reshape(-1, 7), np.array(new_boxes, np.float32)])
    gt_names = np.concatenate([gt_names, np.array([cfg.get('CLASS_NAME', 'Person')] * len(new_boxes))])
    return points, gt_boxes, gt_names
