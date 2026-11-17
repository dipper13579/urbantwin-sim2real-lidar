"""Pick extra real LUMPI frames rich in rare classes (Bus, Motorcycle) for a separate validation set.

Only labels of non-forbidden frames are read (lumpi_real.load_labels drops every frame within 300 of a
forbidden ID before returning anything); frames already downloaded are skipped. Greedy set cover over the
distinct rare objects (>= --gap frames between picks), then the rest of the budget goes to the frames with
the most rare objects that are >= 2 * --gap away from every pick (new positions of the same objects).

  python tools/select_rare_frames.py --budget 4=120,6=80 --out-dir work/selection
Writes <out-dir>/rare_names_m<M>.txt (lidar/NNNNNN.ply entries for fetch_ut.py --names-file).
"""
from release_paths import WORK_ROOT, RAW_ROOT
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumpi_real import CLASS_NAMES, is_excluded, load_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", default="4=120,6=80")
    ap.add_argument("--classes", default="3,4", help="LUMPI class ids (3 Motorcycle, 4 Bus)")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rare = [int(c) for c in args.classes.split(",")]
    for item in args.budget.split(","):
        m, budget = (int(v) for v in item.split("="))
        df = load_labels(WORK_ROOT / "data" / "lumpi_real" / f"M{m}_Label.csv")
        df = df[(df.x.abs() <= 40) & (df.y.abs() <= 40) & (df.z >= -5) & (df.z <= 5)]
        have = {int(p.stem.split("_")[1]) for p in (WORK_ROOT / "data" / "lumpi_real" / f"points_M{m}").glob("*.bin")}
        r = df[df.cls.isin(rare) & ~df.frame.isin(have)]
        objs = {f: set(g.id) for f, g in r.groupby("frame")}
        frames = np.array(sorted(objs))
        picked, covered = [], set()
        all_ids = set(r.id)
        cand = set(frames.tolist())
        while len(picked) < budget and covered != all_ids and cand:
            f = max(cand, key=lambda k: (len(objs[k] - covered), len(objs[k])))
            if not objs[f] - covered:
                break
            picked.append(f)
            covered |= objs[f]
            cand -= set(range(f - args.gap, f + args.gap + 1))
        rest = sorted(cand, key=lambda k: -len(objs[k]))
        for f in rest:
            if len(picked) >= budget:
                break
            if min(abs(f - p) for p in picked) >= 2 * args.gap:
                picked.append(f)
        picked = np.array(sorted(picked))
        assert not is_excluded(picked).any(), "a frame near the forbidden IDs slipped in"
        sel = r[r.frame.isin(picked)]
        stats = {CLASS_NAMES[c]: f"{(sel.cls == c).sum()} boxes / {sel[sel.cls == c].id.nunique()} of "
                                 f"{r[r.cls == c].id.nunique()} objects" for c in rare}
        print(f"M{m}: {len(picked)} frames ({picked.min()}..{picked.max()}), {stats}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"rare_names_m{m}.txt").write_text("".join(f"lidar/{f:06d}.ply\n" for f in picked))


if __name__ == "__main__":
    main()
