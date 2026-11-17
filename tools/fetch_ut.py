"""Selective download from split zips over HTTP (UrbanTwin on Harvard Dataverse, or any URL).

A split zip (name.zip.001, .002, ...) is one zip cut into byte chunks. This reads the central
directory from the tail of the last volume, then downloads only the byte ranges of the files you
ask for (HTTP Range) and extracts them (stored or deflated entries).

  python tools/fetch_ut.py list  --doi doi:10.7910/DVN/N6N4UR
  python tools/fetch_ut.py fetch --doi doi:10.7910/DVN/N6N4UR --folders lbl,pcd --stride 2 \
      --out ${RAW_ROOT}/UT-V2X-Real-IC/extracted
  # plain URLs: volumes <url>.001 ... are discovered with HEAD requests
  python tools/fetch_ut.py list  --url https://data.uni-hannover.de/vault/ikg/busch/LUMPI/Measurement5/lidar.zip
  python tools/fetch_ut.py fetch --url ... --include 'lidar/00(1|2)\\d{3}\\.ply$' --out ${RAW_ROOT}/LUMPI-real

--stride/--offset pick every k-th file (frames are 10 Hz, so neighbours are near-duplicates);
--names-file fetches exactly the listed entry names. Resumable: files already on disk are skipped.
"""
from release_paths import WORK_ROOT, RAW_ROOT
import argparse
import json
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://dataverse.harvard.edu/api"
UA = {"User-Agent": "urbantwin-fetch/1.0 (+https://dataverse.harvard.edu/dataverse/ucf-ut)"}  # default Python UA gets 403
MAX_UNIT_BYTES = 128 * 2**20


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_opener = urllib.request.build_opener(NoRedirect)
_url_cache, _url_lock = {}, threading.Lock()


def volumes(doi: str):
    req = urllib.request.Request(f"{API}/datasets/:persistentId/?persistentId={doi}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    vols = sorted(((f["dataFile"]["filename"], f["dataFile"]["id"], f["dataFile"]["filesize"]) for f in files))
    out, start = [], 0
    for name, fid, size in vols:
        out.append({"name": name, "id": fid, "size": size, "start": start})
        start += size
    return out


def url_volumes(base: str):
    """Volumes base.001, base.002, ... (or a single unsplit base) with sizes from HEAD requests."""
    def head(u):
        try:
            req = urllib.request.Request(u, method="HEAD", headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except urllib.error.HTTPError:
            return None
    out, start, i = [], 0, 1
    while (size := head(f"{base}.{i:03d}")) is not None:
        out.append({"name": f"{base.rsplit('/', 1)[-1]}.{i:03d}", "url": f"{base}.{i:03d}", "size": size, "start": start})
        start += size
        i += 1
    if not out:
        size = head(base)
        if size is None:
            raise SystemExit(f"not found: {base}")
        out.append({"name": base.rsplit("/", 1)[-1], "url": base, "size": size, "start": 0})
    return out


def signed_url(fid: int, fresh: bool = False) -> str:
    """S3 URL behind Dataverse's redirect; cached ~45 min (signatures last 1 h)."""
    with _url_lock:
        hit = _url_cache.get(fid)
        if hit and not fresh and time.time() - hit[1] < 45 * 60:
            return hit[0]
    try:
        _opener.open(urllib.request.Request(f"{API}/access/datafile/{fid}", headers=UA), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (302, 303) and e.headers.get("Location"):
            with _url_lock:
                _url_cache[fid] = (e.headers["Location"], time.time())
            return e.headers["Location"]
        raise
    raise RuntimeError(f"no redirect for datafile {fid}")


def get_range(vol, lo: int, hi: int, tries: int = 10) -> bytes:
    """Bytes [lo, hi) of one volume."""
    for t in range(tries):
        try:
            url = vol["url"] if "url" in vol else signed_url(vol["id"], fresh=t > 0)
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={lo}-{hi - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == hi - lo:
                return data
            raise IOError(f"short read {len(data)} != {hi - lo}")
        except Exception as e:  # noqa: BLE001 - network: retry everything
            wait = min(120, 5 * 2**t)
            print(f"  retry {vol['name']} [{lo},{hi}) in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed {vol['name']} [{lo},{hi})")


def read_global(vols, lo: int, hi: int) -> bytes:
    out = bytearray()
    for v in vols:
        a, b = max(lo, v["start"]), min(hi, v["start"] + v["size"])
        if a < b:
            out += get_range(v, a - v["start"], b - v["start"])
    return bytes(out)


def central_directory(vols):
    """Entries sorted by offset, each with 'end' = where the next local header starts."""
    total = vols[-1]["start"] + vols[-1]["size"]
    tail_lo = max(0, total - 16 * 2**20)
    tail = read_global(vols, tail_lo, total)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found")
    n_entries = struct.unpack("<H", tail[i + 10:i + 12])[0]
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x07", 0, i)  # zip64 locator
    if j >= 0:
        z64 = struct.unpack("<Q", tail[j + 8:j + 16])[0] - tail_lo
        n_entries, cd_size, cd_off = struct.unpack("<QQQ", tail[z64 + 32:z64 + 56])
    cd = tail[cd_off - tail_lo:cd_off - tail_lo + cd_size] if cd_off >= tail_lo else read_global(vols, cd_off, cd_off + cd_size)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize, nlen, xlen, clen = struct.unpack("<IIHHH", cd[p + 20:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        k = 0
        while k + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[k:k + 4])
            if hid == 0x0001:  # zip64: only the saturated fields are present, in this order
                vals, q = struct.unpack("<" + "Q" * (hlen // 8), extra[k + 4:k + 4 + hlen]), 0
                if usize == 0xFFFFFFFF:
                    usize, q = vals[q], q + 1
                if csize == 0xFFFFFFFF:
                    csize, q = vals[q], q + 1
                if off == 0xFFFFFFFF:
                    off = vals[q]
            k += 4 + hlen
        entries.append({"name": name, "offset": off, "csize": csize, "usize": usize, "method": method})
        p += 46 + nlen + xlen + clen
    assert len(entries) == n_entries, (len(entries), n_entries)
    entries.sort(key=lambda e: e["offset"])
    for a, b in zip(entries, entries[1:] + [{"offset": cd_off}]):
        a["end"] = b["offset"]
    return entries


def folder_of(name: str) -> str:
    """Name of the directory that directly contains the file ('' at the archive root)."""
    parts = name.strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


def get_volumes(args):
    return volumes(args.doi) if args.doi else url_volumes(args.url)


def cmd_list(args):
    vols = get_volumes(args)
    entries = central_directory(vols)
    print(f"{len(vols)} volumes, {sum(v['size'] for v in vols) / 2**30:.2f} GiB, {len(entries)} entries")
    by = {}
    for e in entries:
        if not e["name"].endswith("/"):
            d = by.setdefault(folder_of(e["name"]), [0, 0, e["name"]])
            d[0] += 1
            d[1] += e["csize"]
    for f, (n, size, first) in by.items():
        print(f"  {f or '.':<10} files={n:>6}  {size / 2**30:6.2f} GiB  first={first}")


def units_for(todo):
    """Group archive-contiguous entries into download units of <= MAX_UNIT_BYTES."""
    units, cur = [], []
    for e in todo:
        if cur and (cur[-1]["end"] != e["offset"] or e["end"] - cur[0]["offset"] > MAX_UNIT_BYTES):
            units.append(cur)
            cur = []
        cur.append(e)
    if cur:
        units.append(cur)
    return units


def fetch_unit(vols, unit, out: Path) -> int:
    lo, hi = unit[0]["offset"], unit[-1]["end"]
    buf = read_global(vols, lo, hi)
    for e in unit:
        rel = e["offset"] - lo
        if buf[rel:rel + 4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {e['name']}")
        nlen, xlen = struct.unpack("<HH", buf[rel + 26:rel + 30])
        d0 = rel + 30 + nlen + xlen
        dst = out / e["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = buf[d0:d0 + e["csize"]]
        data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        tmp.rename(dst)
    return hi - lo


def cmd_fetch(args):
    vols = get_volumes(args)
    entries = [e for e in central_directory(vols) if not e["name"].endswith("/")]
    if args.include:
        entries = [e for e in entries if re.search(args.include, e["name"])]
    if args.names_file:
        wanted = {l.strip() for l in Path(args.names_file).read_text().splitlines() if l.strip()}
        entries = [e for e in entries if e["name"] in wanted]
    out = Path(args.out)
    groups = args.folders.split(",") if args.folders else [None]
    for folder in groups:
        sel = [e for e in entries if folder is None or folder_of(e["name"]) == folder]
        if not sel:
            raise SystemExit(f"no entries for folder {folder!r}")
        if folder != "lbl":  # UT labels are tiny: always fetch all of them
            sel = sel[args.offset::args.stride]
            if args.max_frames:
                sel = sel[: args.max_frames]
        assert all(e["method"] in (0, 8) for e in sel), "only stored or deflated entries are supported"
        todo = [e for e in sel if not (out / e["name"]).exists()]
        units = units_for(todo)
        total = sum(u[-1]["end"] - u[0]["offset"] for u in units)
        print(f"[{folder}] {len(todo)}/{len(sel)} files to fetch, {total / 2**30:.2f} GiB in {len(units)} requests", flush=True)
        t0, done, n_files = time.time(), 0, 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_unit, vols, u, out): u for u in units}
            for k, f in enumerate(as_completed(futs), 1):
                done += f.result()
                n_files += len(futs[f])
                if k % max(1, len(units) // 50) == 0 or k == len(units):
                    rate = done / max(time.time() - t0, 1e-6) / 2**20
                    eta = (total - done) / max(rate, 1e-6) / 2**20 / 60
                    print(f"[{folder}] {n_files}/{len(todo)} files, {done / 2**30:.2f}/{total / 2**30:.2f} GiB, "
                          f"{rate:.2f} MiB/s, ETA {eta:.0f} min", flush=True)
        print(f"[{folder}] DONE", flush=True)
    print("=== FETCH DONE ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    b = sub.add_parser("fetch")
    for p in (a, b):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--doi", help="Harvard Dataverse dataset DOI")
        src.add_argument("--url", help="zip URL without the .001 suffix")
    b.add_argument("--folders", default=None, help="comma-separated parent folders, e.g. lbl,pcd")
    b.add_argument("--include", default=None, help="regex on entry names")
    b.add_argument("--names-file", default=None, help="text file with exact entry names to fetch")
    b.add_argument("--out", required=True)
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--offset", type=int, default=0)
    b.add_argument("--max-frames", type=int, default=0)
    b.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
