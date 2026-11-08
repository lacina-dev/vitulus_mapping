#!/usr/bin/env python3
"""
terrainexport.py -- the persistent terrain map as files for the UI (T31), without rospy.

  export_mosaic(store, out_dir, site)   <site>/terrain_mosaic/: the same files as the
                                        offline mosaic (vitulus_terrain_mosaic terrain_map.py):
                                        teren_<site>.json + _cost_rgba / _reason / _cost /
                                        _elevation16 .png, in the map frame of the store's
                                        datum (contract ~/vitulus/predavka/t31-teren-ui-kontrakt.md)
  nogo_candidates(...)                  clusters of confirmed terrain bans (a list only)
  load_mosaic_dir(dir)                  an offline mosaic as layers for TerrainStore.import_map_grid
  write_png(path, array)                minimal PNG writer (gray8, gray16, RGBA8), numpy + zlib

Only numpy, zlib, json: runs on the robot without PIL / matplotlib.
"""

import glob
import hashlib
import json
import math
import os
import struct
import tempfile
import time
import zlib

import numpy as np

from . import terrainlayers as tl

__all__ = ["export_mosaic", "export_from_disk", "nogo_candidates", "load_mosaic_dir", "write_png", "cost_rgba",
           "REASONS_UI", "ENCODING"]

REASONS_UI = {"0": "neznámé", "1": "sjízdné (cena < 50)", "2": "vyšší cena (50–99)",
              "3": "nesjízdné: schod / hrana", "4": "nesjízdné: sklon", "5": "nesjízdné: překážka",
              "6": "schod / svah dolů nepotvrzený (99)", "7": "zóna dokovací stanice (99)"}
ENCODING = {
    "cost.png": "uint8 gray: cost 0..100 (100 = lethal), 255 = unknown (as /traversability/ui)",
    "reason.png": "uint8 gray: reason codes of /traversability/ui (reasons_ui)",
    "cost_rgba.png": "RGBA colour of the cost 0-99 (green-red), dark red = lethal, alpha 0 = unknown",
    "elevation16.png": "uint16: 0 = unknown, else elevation [mm] above the lowest point + 1",
}
CORNER_NOTE = ("origin = min corner (x, y) in the map frame of the bottom-left pixel; images row 0 = max y "
               "(north-up), column 0 = min x; pixel (row r, col c) centre = (origin_x + (c + 0.5) res, "
               "origin_y + (height - r - 0.5) res)")
NORTH_NOTE = "direction of UTM grid north in the map frame (atan2 angle from +x)"
TENTATIVE_NOTE = ("reason 3 / 4 with cost 99 = a terrain ban not confirmed yet (needs >= 3 lethal visits, "
                  ">= 50 % of the visits, >= 2 sessions, >= 2 headings); obstacles count from the last visit only")
# the mosaic's colour map (matplotlib LinearSegmentedColormap of these, equally spaced)
_CMAP = np.array([[0x1a, 0x98, 0x50], [0x91, 0xcf, 0x60], [0xfe, 0xe0, 0x8b], [0xfc, 0x8d, 0x59],
                  [0xd7, 0x30, 0x27]], np.float64)


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------
def _png_bytes(arr):
    a = np.asarray(arr)
    if a.ndim == 2 and a.dtype == np.uint8:
        color, depth, raw = 0, 8, a
    elif a.ndim == 2 and a.dtype == np.uint16:
        color, depth, raw = 0, 16, a.astype(">u2")
    elif a.ndim == 3 and a.shape[2] == 4 and a.dtype == np.uint8:
        color, depth, raw = 6, 8, a
    else:
        raise ValueError("unsupported image %s %s" % (a.shape, a.dtype))
    h, w = a.shape[:2]
    rows = np.ascontiguousarray(raw).reshape(h, -1).view(np.uint8)
    data = np.hstack([np.zeros((h, 1), np.uint8), rows]).tobytes()

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(data, 6)) + chunk(b"IEND", b""))


def _atomic(path, data):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_png(path, arr):
    _atomic(path, _png_bytes(arr))


def _rgba_lut():
    lut = np.zeros((256, 4), np.uint8)               # 255 and anything else unknown: clear
    t = np.arange(100) / 99.0 * (len(_CMAP) - 1)
    i = np.minimum(np.floor(t).astype(int), len(_CMAP) - 2)
    u = (t - i)[:, None]
    lut[:100, :3] = np.round(_CMAP[i] * (1 - u) + _CMAP[i + 1] * u).astype(np.uint8)
    lut[:100, 3] = 200
    lut[100] = (60, 0, 0, 230)
    return lut


_LUT = _rgba_lut()


def cost_rgba(cost, known):
    """RGBA of the mosaic: cost 0..99 green -> red (alpha 200), 100 dark red (alpha 230), unknown clear
    (a lookup table: little memory on a big site)."""
    c = np.where(known, np.clip(cost, 0, 255), 255).astype(np.uint8)
    return _LUT[c]


# --------------------------------------------------------------------------
# raster of the store in the map frame
# --------------------------------------------------------------------------
def _known_bbox_map(store):
    """Map-frame bbox (x0, y0, x1, y1) of the known cells of every indexed tile."""
    d = store.datum
    yaw = float(d.get("yaw_rad", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    xs, ys = [], []
    for key in list(store.index["tiles"].keys()) + ["%d_%d" % k for k in store.tiles.keys()]:
        tx, ty = (int(v) for v in key.split("_"))
        t = store.tile(tx, ty)
        if t is None:
            continue
        ly, lx = np.nonzero(np.isfinite(t.h))
        if ly.size == 0:
            continue
        e = ((tx << tl.TILE_SHIFT) + lx + 0.5) * tl.RES - d["utm_e"]
        n = ((ty << tl.TILE_SHIFT) + ly + 0.5) * tl.RES - d["utm_n"]
        x = c * e + s * n
        y = -s * e + c * n
        xs += [x.min(), x.max()]
        ys += [y.min(), y.max()]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def raster_map(store, margin=0.5):
    """Effective layers of the store resampled to a 5 cm grid in the map frame.
    Returns dict origin_x, origin_y, width, height and arrays [row, col] with row 0 =
    LOWEST y: h (NaN unknown), cost (uint8, 255 unknown), reason (uint8); or None."""
    bb = _known_bbox_map(store)
    if bb is None:
        return None
    res = tl.RES
    x0 = math.floor((bb[0] - margin) / res) * res
    y0 = math.floor((bb[1] - margin) / res) * res
    w = int(math.ceil((bb[2] + margin - x0) / res))
    h = int(math.ceil((bb[3] + margin - y0) / res))
    d = store.datum
    yaw = float(d.get("yaw_rad", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    hh = np.full((h, w), np.nan, np.float32)
    cost = np.full((h, w), 255, np.uint8)
    reason = np.zeros((h, w), np.uint8)
    eff = {}                                     # tile -> (reason, cost) computed once
    xs = x0 + (np.arange(w) + 0.5) * res
    for r0 in range(0, h, 128):                  # in stripes: bounded memory on a big site
        r1 = min(h, r0 + 128)
        ys = y0 + (np.arange(r0, r1) + 0.5) * res
        x, y = np.meshgrid(xs, ys)
        e = d["utm_e"] + c * x - s * y
        n = d["utm_n"] + s * x + c * y
        keys = tl.pack_keys(np.floor(e / res), np.floor(n / res)).ravel()
        sh = np.full(keys.size, np.nan, np.float32)
        sc = np.full(keys.size, 255, np.uint8)
        sr = np.zeros(keys.size, np.uint8)
        for (tx, ty), sel, ly, lx in store._groups(keys):
            t = store.tile(tx, ty)
            if t is None:
                continue
            if (tx, ty) not in eff:
                rz, cz, _tent = t.effective()
                eff[(tx, ty)] = (rz, cz)
            rz, cz = eff[(tx, ty)]
            sh[sel] = t.h[ly, lx]
            sc[sel] = cz[ly, lx]
            sr[sel] = rz[ly, lx]
        hh[r0:r1] = sh.reshape(r1 - r0, w)
        cost[r0:r1] = sc.reshape(r1 - r0, w)
        reason[r0:r1] = sr.reshape(r1 - r0, w)
    known = np.isfinite(hh)
    cost[~known] = 255
    reason[~known] = 0
    return {"origin_x": x0, "origin_y": y0, "width": w, "height": h, "h": hh, "cost": cost, "reason": reason}


# --------------------------------------------------------------------------
# nogo candidates
# --------------------------------------------------------------------------
def _hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def nogo_candidates(r, min_cells=20, buffer_m=0.15, limit=50):
    """Connected (8-neighbour) clusters of confirmed terrain bans (cost 100, reason 3 / 4)
    of a raster_map; polygon = hull of the cells grown by buffer_m, map frame. A list for
    the UI only: nothing is written anywhere."""
    if r is None:
        return []
    res = tl.RES
    mask = (r["cost"] == 100) & ((r["reason"] == tl.REASON_STEP) | (r["reason"] == tl.REASON_SLOPE))
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rows, cols))}
    parent = list(range(len(idx)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (a, b), i in idx.items():
        for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = idx.get((a + da, b + db))
            if j is not None:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    groups = {}
    for (a, b), i in idx.items():
        groups.setdefault(find(i), []).append((a, b))
    out = []
    g = res / 2.0 + buffer_m
    for cells in groups.values():
        if len(cells) < min_cells:
            continue
        cells = np.array(cells)
        x = r["origin_x"] + (cells[:, 1] + 0.5) * res
        y = r["origin_y"] + (cells[:, 0] + 0.5) * res
        rs = r["reason"][cells[:, 0], cells[:, 1]]
        corners = np.concatenate([np.stack([x + dx, y + dy], 1) for dx in (-g, g) for dy in (-g, g)])
        hull = _hull(np.round(corners, 3).tolist())
        cx, cy = float(x.mean()), float(y.mean())
        cid = hashlib.sha1(("%.1f,%.1f" % (round(cx * 2) / 2, round(cy * 2) / 2)).encode()).hexdigest()[:6]
        out.append({"id": cid, "kind": "step" if np.count_nonzero(rs == tl.REASON_STEP) * 2 >= rs.size else "slope",
                    "area_m2": round(len(cells) * res * res, 3), "cells": int(len(cells)),
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "polygon": [[round(px, 2), round(py, 2)] for px, py in hull]})
    out.sort(key=lambda c: -c["cells"])
    return out[:limit]


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
LAYER_KEYS = ("cost", "reason", "cost_rgba", "elevation16")


def _old_versions(out_dir, prefix, keep):
    """Versioned image files of earlier exports, except the versions in `keep`."""
    out = []
    for p in glob.glob(os.path.join(out_dir, prefix + "_v*_*.png")):
        try:
            v = int(os.path.basename(p)[len(prefix) + 2:].split("_", 1)[0])
        except ValueError:
            continue
        if v not in keep:
            out.append(p)
    # unversioned images of the T31 v1 export
    out += [p for p in (os.path.join(out_dir, "%s_%s.png" % (prefix, k)) for k in LAYER_KEYS) if os.path.exists(p)]
    return out


def _read_counter(path):
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (IOError, OSError, ValueError):
        return 0


def counter_path(out_dir):
    """The export version counter beside terrain_mosaic/ (it survives delete_map, which removes the
    export and moves terrain/ aside: a version number is never used twice)."""
    return os.path.join(os.path.dirname(os.path.abspath(out_dir)), "terrain_export.version")


def export_mosaic(store, out_dir, site, sessions=None, r=None, extra=None, counter=None):
    """Write the mosaic files of the store to out_dir. Images get versioned names
    (teren_<site>_v<N>_<layer>.png, written first), then teren_<site>.json is replaced
    atomically (it names the images of version N), then images older than N - 1 are
    removed: a reader that has the old JSON still finds its images. Returns the JSON
    dict, or None when the store has no known cell (the JSON is removed first)."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = "teren_%s" % site
    jpath = os.path.join(out_dir, prefix + ".json")
    version = 0
    try:
        with open(jpath) as f:
            version = int(json.load(f).get("version", 0))
    except (IOError, OSError, ValueError, TypeError, AttributeError):
        pass
    counter = counter_path(out_dir) if counter is None else counter
    version = max(version, _read_counter(counter))
    r = raster_map(store) if r is None else r
    if r is None:
        if os.path.exists(jpath):
            os.remove(jpath)
        for p in _old_versions(out_dir, prefix, keep=()):
            os.remove(p)
        return None
    v = version + 1
    names = {k: "%s_v%d_%s.png" % (prefix, v, k) for k in LAYER_KEYS}
    known = np.isfinite(r["h"])
    flip = lambda a: np.ascontiguousarray(a[::-1])     # noqa: E731  row 0 = max y in the images
    write_png(os.path.join(out_dir, names["cost_rgba"]), flip(cost_rgba(r["cost"], known)))
    write_png(os.path.join(out_dir, names["reason"]), flip(r["reason"]))
    write_png(os.path.join(out_dir, names["cost"]), flip(r["cost"]))
    lo = float(np.nanmin(r["h"]))
    e16 = np.where(known, np.clip(np.round((r["h"] - lo) * 1000.0) + 1, 1, 65535), 0).astype(np.uint16)
    write_png(os.path.join(out_dir, names["elevation16"]), flip(e16))
    yaw = float(store.datum.get("yaw_rad", 0.0))
    now = time.time()
    meta = {
        "frame": "map", "resolution": tl.RES, "width": r["width"], "height": r["height"], "yaw": 0.0,
        "origin_x": round(r["origin_x"], 4), "origin_y": round(r["origin_y"], 4), "corner_note": CORNER_NOTE,
        "north_angle_rad": math.pi / 2.0 - yaw, "north_note": NORTH_NOTE,
        "files": names, "encoding": ENCODING, "reasons_ui": REASONS_UI, "tentative_note": TENTATIVE_NOTE,
        "elevation_lowest_m": round(lo, 4),
        "source": "terrain_recorder", "site": site, "version": v,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), "generated_unix": round(now, 3),
        "stamp": now, "area_m2": round(int(known.sum()) * tl.RES ** 2, 1), "sessions": sessions,
    }
    if extra:
        meta.update(extra)
    _atomic(jpath, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    _atomic(counter, str(v).encode())
    for p in _old_versions(out_dir, prefix, keep=(v, v - 1)):
        try:
            os.remove(p)
        except OSError:
            pass
    return meta


def export_from_disk(terrain_dir, datum, out_dir, site, sessions=None):
    """The whole export from the stored tiles only (a separate, niced process: no lock
    of the recorder, no shared memory). Returns a summary dict."""
    t0 = time.process_time()
    store = tl.TerrainStore(terrain_dir, datum, params={"max_loaded_tiles": 100000}, readonly=True)
    r = raster_map(store)
    cands = nogo_candidates(r)
    meta = export_mosaic(store, out_dir, site, sessions=sessions, r=r,
                         extra={"nogo_candidates": cands})
    return {"version": meta["version"] if meta else None, "area_m2": meta["area_m2"] if meta else 0.0,
            "pixels": int(r["width"] * r["height"]) if r else 0, "nogo_candidates": cands,
            "cpu_s": round(time.process_time() - t0, 3)}


def main(argv=None):
    """python3 -m vitulus_mapping.terrainexport --terrain DIR --datum JSON --out DIR --site S [--sessions N]
    -> one JSON line on stdout (export_from_disk)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--datum", required=True, help="JSON {utm_e, utm_n, yaw_rad}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--sessions", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        out = export_from_disk(a.terrain, json.loads(a.datum), a.out, a.site, a.sessions)
        out["ok"] = True
    except Exception as e:                       # noqa: BLE001
        out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


# --------------------------------------------------------------------------
# offline mosaic as import layers
# --------------------------------------------------------------------------
def load_mosaic_dir(mdir):
    """(meta, layers, t) of an offline mosaic dir (teren_<tag>.json + _layers.npz, optional
    grid/grid.json + grid/t_last.f32) for TerrainStore.import_map_grid. Raises ValueError."""
    js = [p for p in glob.glob(os.path.join(mdir, "teren_*.json"))]
    if len(js) != 1:
        raise ValueError("expected one teren_*.json in %s, found %d" % (mdir, len(js)))
    with open(js[0]) as f:
        meta = json.load(f)
    if meta.get("frame") != "map" or abs(float(meta.get("yaw", 0.0))) > 1e-9:
        raise ValueError("mosaic not in the map frame without rotation")
    if meta.get("source") == "terrain_recorder":
        raise ValueError("this is a terrain_recorder export, not a mosaic")
    lay = np.load(os.path.join(mdir, meta["files"]["layers"]))
    h, w = int(meta["height"]), int(meta["width"])
    flip = lambda a: np.asarray(a)[::-1]          # noqa: E731  row 0 = max y -> row 0 = min y
    t = float(os.path.getmtime(js[0]))
    gjs = os.path.join(mdir, "grid", "grid.json")
    tl_path = os.path.join(mdir, "grid", "t_last.f32")
    if os.path.exists(gjs) and os.path.exists(tl_path):
        with open(gjs) as f:
            g = json.load(f)
        tlast = np.fromfile(tl_path, np.float32).reshape(int(g["height"]), int(g["width"]))
        t = flip(float(g["t_ref"]) + np.where(np.isfinite(tlast), tlast, np.nanmax(tlast)))
    layers = {
        "elevation": flip(lay["elevation"]), "sigma": flip(lay["elevation_std"]),
        "cost": flip(lay["cost"]), "reason": flip(lay["reason"]),
        "slope": np.radians(flip(lay["slope_deg"])), "step": flip(lay["step"]),
        "roughness": flip(lay["roughness"]),
    }
    for k, a in layers.items():
        if a.shape != (h, w):
            raise ValueError("layer %s: %s, expected %s" % (k, a.shape, (h, w)))
    return meta, layers, t


if __name__ == "__main__":
    raise SystemExit(main())
