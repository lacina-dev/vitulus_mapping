#!/usr/bin/env python3
"""
mowreport.py -- how a finished mow run went, zone by zone. Contract:
docs/MAP_OBJECTS.md section 11.1.

NO rospy. Reads a run of the Mow load trail (mowtrail.read_run), the zones and the
areas nobody mows (obstacle edits, nogo and noblade rules) of the site, and says
per zone of the program:

  progress   paths_first/last/total from the ctx events, visited, complete
  load       blade load 0..1 over the cut rows, percentiles weighted by the path
             length: p50, p90, max, and the share of the working time in overload
             or blocked. Free running reads about 0.10.
  work       cut length, mowing time, time in the zone at all
  coverage   the cut trail buffered by the deck (deck_m wide) against the zone on a
             grid of grid_m, minus what nobody mows. Gaps: uncovered after an
             opening of gap_open_m (a strip thinner than twice that is the edge,
             not a miss), at least min_gap_m2, with their area and a point inside
  heavy      clusters of cut rows with load >= max(heavy_load, zone p90) on cells
             of cluster_m: centre, area, max load, overloads
  loc        p95 |xte| against the plan, lidar and jump p95, RTK share, off_path
             events. Older runs (20, 26 or 34 columns) have less: None.

and gives a machine list `advice`: {kind, zone, where: [x, y] | None, why, value}
with kind unfinished | resow_gaps | heavy_area | gnss_check | path_tracking.

ZONE OF A ROW. The zone of the last `ctx` event before it (the smach's [EXE] START
line) AND inside that zone's polygon grown by deck_m; anything else (out of the
dock, back to it, transfers) is `outside`. A run without ctx uses the zones of its
program by point in polygon. Zones of one name are one multipolygon; holes are the
further rings of a zones.geojson feature, when the bundle has them.

RESUMED RUNS. A run that starts past the first path of its program continues the
previous run of the same program when that one ended at or before the path this one
starts with and at most chain_h before it. Coverage and paths done are then the
union of the chain. Gaps are advice (resow_gaps) only for a zone whose paths are all
done; a run with paths left gives `unfinished` with what is left.

FILES  <run>.report.json next to <run>.npz, atomic. Valid while its `sig` matches:
version, datum, the zones of the program and the rules (nogo, noblade, obstacles),
deck_m, the npz (mtime, size) and the runs it could continue.

    report(site, run=None, cfg=None)     -> dict, computed now
    load(site, run=None, cfg=None)       -> the stored one if valid, else computed
    save(site, run, cfg=None)            -> path written
    backfill(site, cfg=None, pause_s)    -> [runs written], one by one with a pause;
                                            drops reports of gone runs
CLI: python3 -m vitulus_mapping.mowreport [run] [--site S] [--json] [--save] [--all]
The recorder node runs it as a child process at nice 19.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

from . import mowtrail as mt
from . import worldlayers

__all__ = ["REPORT_VERSION", "DEFAULTS", "report", "load", "save", "backfill", "report_path",
           "pick_run", "signature", "main"]

REPORT_VERSION = 2
SUFFIX = ".report.json"
DEFAULTS = {
    "deck_m": mt.DEFAULTS["deck_m"],    # cut width; also how far outside its polygon a
                                        # row still belongs to a zone
    "grid_m": 0.05,             # coverage grid; zones over big_zone_m2 use grid_big_m
    "grid_big_m": 0.10,
    "big_zone_m2": 3000.0,
    "max_seg_m": 0.4,           # a longer step between two rows is a jump, not cut
    "gap_open_m": 0.10,         # strips thinner than 2x this are the edge, not a gap
    "min_gap_m2": 0.5,
    "max_gaps": 10,             # per zone, largest first
    "idle_load": 0.10,          # blade running free
    "heavy_load": 0.30,         # a cut row this heavy joins a heavy cluster
    "cluster_m": 0.5,
    "max_heavy": 5,             # per zone
    "heavy_advice_load": 0.5,   # a cluster this heavy (or with an overload) is advice
    "max_off_path": 20,         # per zone
    "chain_h": 48.0,            # a resumed run continues one that ended this recently
    "chain_max": 4,
    "backfill_pause_s": 1.0,
    # advice limits
    "resow_min_m2": 0.5,
    "rtk_min": 0.95,
    "lidar_p95_cm": 30,
    "jump_p95_cm": 20,
    "xte_p95_cm": 12,
    "off_path_n": 3,
}
_EXCLUDE_TYPES = ("obstacle", "nogo", "noblade")
_PATH_ORDER = 100000            # program position = zone index * this + path


# --------------------------------------------------------------------------
# site
# --------------------------------------------------------------------------
def _zone_holes(site, datum):
    """{zone name: [hole rings in map]} from the further rings of zones.geojson."""
    from . import bundle, geo
    out = {}
    try:
        with open(bundle.zones_path(site)) as f:
            fc = json.load(f)
    except (IOError, OSError, ValueError):
        return out
    for feat in fc.get("features") or []:
        try:
            name = (feat.get("properties") or {}).get("name")
            rings = (feat.get("geometry") or {}).get("coordinates") or []
            for ring in rings[1:]:
                pts = [geo.utm_to_map(float(c[0]), float(c[1]), datum) for c in ring]
                if len(pts) >= 2 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                if name and len(pts) >= 3:
                    out.setdefault(name, []).append([(float(x), float(y)) for x, y in pts])
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return out


def _site_shapes(site, datum=None):
    """({zone name: {"rings": [...], "holes": [...]}}, [rings nobody mows]).
    Zones of one name are one multipolygon. Empty when unknown."""
    if datum is None:
        datum = worldlayers.read_datum(site)
    if datum is None:
        return {}, []
    try:
        from . import objects
        objs = objects.load_all(site) or []
    except Exception:                                       # noqa: BLE001
        return {}, []
    zones = {}
    excl = []
    for o in objs:
        pts = [(float(p[0]), float(p[1])) for p in (o.get("points") or [])]
        if len(pts) < 3 or o.get("active") is False:
            continue
        if o.get("family") == "zone" and o.get("name"):
            zones.setdefault(o["name"], {"rings": [], "holes": []})["rings"].append(pts)
        elif o.get("type") in _EXCLUDE_TYPES and o.get("geometry", "area") == "area":
            excl.append(pts)
    for name, holes in _zone_holes(site, datum).items():
        if name in zones:
            zones[name]["holes"].extend(holes)
    return zones, excl


def _program_zones(site, program):
    try:
        from . import objects
        return list((objects.program_zone_names(site) or {}).get(program) or [])
    except Exception:                                       # noqa: BLE001
        return []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _pct(vals, q, weights=None, digits=3):
    """Percentile; weighted when weights are given and not all zero."""
    if not len(vals):
        return None
    if weights is None or not float(np.sum(weights)) > 0.0:
        return round(float(np.percentile(vals, q)), digits)
    order = np.argsort(vals, kind="stable")
    v = np.asarray(vals, float)[order]
    cw = np.cumsum(np.asarray(weights, float)[order])
    i = int(np.searchsorted(cw, q / 100.0 * cw[-1], side="left"))
    return round(float(v[min(i, len(v) - 1)]), digits)


def _known(col):
    return col[col != mt.NONE]


def _cut_mask(rows):
    """Rows where the blade cut: cutting, or overload with a real load (an ERR of the
    firmware is overload without one and says nothing of cutting)."""
    state = rows[:, mt.STATE] & (mt.BRK - 1)
    return (state == mt.CUTTING) | ((state == mt.OVERLOAD) & (rows[:, mt.LOAD] != mt.NONE))


def _steps(rows):
    """Metres from the row before, 0 at a break or the first row."""
    if len(rows) < 2:
        return np.zeros(len(rows))
    step = np.hypot(np.diff(rows[:, mt.X]), np.diff(rows[:, mt.Y])) / 100.0
    joined = rows[1:, mt.STATE] < mt.BRK
    return np.concatenate(([0.0], np.where(joined, step, 0.0)))


def _ctx(events):
    """[(t, zone, path, paths)] sorted by time, from the ctx events."""
    out = []
    for e in events:
        if e[3] != "ctx":
            continue
        parts = str(e[4]).split("|")
        if len(parts) != 4:
            continue
        try:
            out.append((int(e[0]), parts[0], int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    out.sort(key=lambda c: c[0])
    return out


def _progress(ctx, succeeded):
    """{zone: {"first", "last", "total", "done": set}} of one run. A path is done
    when a later ctx names another path; the last one when the run succeeded."""
    out = {}
    for i, (_t, zone, path, total) in enumerate(ctx):
        p = out.setdefault(zone, {"first": path, "last": path, "total": total, "done": set()})
        p["last"] = path
        p["total"] = max(p["total"], total)
        nxt = ctx[i + 1] if i + 1 < len(ctx) else None
        if (nxt is not None and (nxt[1], nxt[2]) != (zone, path)) or (nxt is None and succeeded):
            p["done"].add(path)
    return out


def _position(order, zone, path):
    i = order.index(zone) if zone in order else len(order)
    return i * _PATH_ORDER + path


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------
class _Grid(object):
    """One zone (a multipolygon less its holes) on a grid, padded by deck_m."""

    def __init__(self, zone, excl, cfg):
        from scipy import ndimage
        rings = zone["rings"]
        allx = [p[0] for r in rings for p in r]
        ally = [p[1] for r in rings for p in r]
        bbox = (max(allx) - min(allx)) * (max(ally) - min(ally))
        self.res = res = cfg["grid_m"] if bbox <= cfg["big_zone_m2"] else cfg["grid_big_m"]
        pad = cfg["deck_m"] + 2 * res
        self.ox, self.oy = min(allx) - pad, min(ally) - pad
        self.w = int(math.ceil((max(allx) + pad - self.ox) / res)) + 1
        self.h = int(math.ceil((max(ally) + pad - self.oy) / res)) + 1
        mask = np.zeros((self.h, self.w), bool)
        for ring in rings:
            mask |= self._poly(ring)
        for hole in zone["holes"]:
            mask &= ~self._poly(hole)
        self.zone = mask
        mow = mask.copy()
        for ring in excl:
            if self._touches(ring):
                mow &= ~self._poly(ring)
        self.mowable = mow
        # where a row still belongs to the zone: the zone grown by deck_m
        self.near = ndimage.distance_transform_edt(~mask) * res <= cfg["deck_m"]

    def _poly(self, ring):
        return worldlayers._poly_mask(self.ox, self.oy, self.res, self.h, self.w, ring)

    def _touches(self, ring):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return not (max(xs) < self.ox or min(xs) > self.ox + self.w * self.res
                    or max(ys) < self.oy or min(ys) > self.oy + self.h * self.res)

    def cells(self, xs, ys):
        c = np.floor((xs - self.ox) / self.res).astype(np.int64)
        r = np.floor((ys - self.oy) / self.res).astype(np.int64)
        ok = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h)
        return r, c, ok

    def inside(self, xs, ys):
        """bool per point (map m): in the zone grown by deck_m."""
        r, c, ok = self.cells(xs, ys)
        out = np.zeros(len(xs), bool)
        out[ok] = self.near[r[ok], c[ok]]
        return out


def _cut_points(rows, step, max_seg):
    """Points every `step` m along the cut segments of the run (map, m). A segment
    longer than max_seg is a jump of the pose, dropped."""
    if len(rows) < 2:
        return np.zeros((0, 2))
    seg = (rows[1:, mt.STATE] < mt.BRK) & _cut_mask(rows)[1:]
    a = rows[:-1][seg][:, (mt.X, mt.Y)] / 100.0
    b = rows[1:][seg][:, (mt.X, mt.Y)] / 100.0
    if not len(a):
        return np.zeros((0, 2))
    d = np.hypot(*(b - a).T)
    keep = d <= max_seg
    a, b, d = a[keep], b[keep], d[keep]
    if not len(a):
        return np.zeros((0, 2))
    n = np.maximum(1, np.ceil(d / step).astype(np.int64))
    cnt = n + 1
    rep = np.repeat(np.arange(len(a)), cnt)
    first = np.cumsum(cnt) - cnt
    k = np.arange(int(cnt.sum())) - np.repeat(first, cnt)
    frac = k / np.repeat(n, cnt)
    return a[rep] + (b - a)[rep] * frac[:, None]


def _coverage(grid, points, cfg):
    """Coverage of one zone grid by the cut points buffered by deck/2."""
    from scipy import ndimage
    res = grid.res
    zone = grid.mowable
    hit = np.zeros(zone.shape, bool)
    if len(points):
        r, c, ok = grid.cells(points[:, 0], points[:, 1])
        hit[r[ok], c[ok]] = True
    if hit.any():
        covered = ndimage.distance_transform_edt(~hit) * res <= cfg["deck_m"] / 2.0
    else:
        covered = hit
    cell = res * res
    todo = int(zone.sum())
    miss = zone & ~covered
    rad = int(round(cfg["gap_open_m"] / res))
    if rad > 0:
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        opened = ndimage.binary_opening(miss, structure=(xx * xx + yy * yy) <= rad * rad)
    else:
        opened = miss
    labels, n = ndimage.label(opened, structure=np.ones((3, 3), bool))
    gaps = []
    if n:
        sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        big = np.nonzero(sizes * cell >= cfg["min_gap_m2"])[0]
        if len(big):
            big = big[np.argsort(-sizes[big], kind="stable")]
            idx = (big + 1).tolist()
            inner = ndimage.distance_transform_edt(opened)
            deep = ndimage.maximum_position(inner, labels, idx)
            mid = ndimage.center_of_mass(opened, labels, idx)
            for i, (rr, cc), (mr_, mc) in zip(big, deep, mid):
                gaps.append({"x": round(grid.ox + (cc + 0.5) * res, 2),
                             "y": round(grid.oy + (rr + 0.5) * res, 2),
                             "cx": round(grid.ox + (mc + 0.5) * res, 2),
                             "cy": round(grid.oy + (mr_ + 0.5) * res, 2),
                             "area_m2": round(float(sizes[i]) * cell, 2),
                             "width_m": round(float(inner[rr, cc]) * 2 * res, 2)})
    area = float(grid.zone.sum()) * cell
    done = float((zone & covered).sum())
    return {"grid_m": res, "zone_m2": round(area, 1),
            "mowable_m2": round(todo * cell, 1),
            "excluded_m2": round(max(0.0, area - todo * cell), 1),
            "covered_m2": round(done * cell, 1),
            "frac": round(done / todo, 4) if todo else None,
            "missed_m2": round(float(miss.sum()) * cell, 2),
            "gap_m2": round(sum(g["area_m2"] for g in gaps), 2),
            "gaps_total": len(gaps),
            "gaps": gaps[:int(cfg["max_gaps"])]}


def _heavy(rows, cfg, start_t, limit):
    """Clusters of cut rows with load >= limit (0..1), 8-neighbour cells of cluster_m."""
    from scipy import ndimage
    load = rows[:, mt.LOAD]
    sel = _cut_mask(rows) & (load != mt.NONE) & (load >= limit * 100.0)
    pick = rows[sel]
    if not len(pick):
        return []
    cs = cfg["cluster_m"]
    cell = np.floor(pick[:, (mt.X, mt.Y)] / 100.0 / cs).astype(np.int64)
    lo = cell.min(axis=0)
    cell -= lo
    span = cell.max(axis=0) + 1
    flat = cell[:, 1] * span[0] + cell[:, 0]
    occ = np.zeros(int(span[0] * span[1]), bool)
    occ[flat] = True
    lab, n = ndimage.label(occ.reshape(span[1], span[0]), structure=np.ones((3, 3), bool))
    lab = lab.ravel()
    cid = lab[flat] - 1
    ncell = np.bincount(lab[occ] - 1, minlength=n)
    w = pick[:, mt.LOAD].astype(float)
    cnt = np.bincount(cid, minlength=n)
    sw = np.bincount(cid, w, minlength=n)
    sx = np.bincount(cid, w * pick[:, mt.X], minlength=n)
    sy = np.bincount(cid, w * pick[:, mt.Y], minlength=n)
    over = np.bincount(cid, (pick[:, mt.STATE] & (mt.BRK - 1)) == mt.OVERLOAD, minlength=n)
    order = np.lexsort((-w, cid))            # per cluster the heaviest row first
    top = np.full(n, -1)
    first = np.ones(len(order), bool)
    first[1:] = cid[order][1:] != cid[order][:-1]
    top[cid[order][first]] = order[first]
    out = []
    for k in range(n):
        if cnt[k] < 2 and not over[k]:
            continue
        t = pick[top[k]]
        out.append({"x": round(float(sx[k] / sw[k]) / 100.0, 2),
                    "y": round(float(sy[k] / sw[k]) / 100.0, 2),
                    "area_m2": round(float(ncell[k]) * cs * cs, 2), "rows": int(cnt[k]),
                    "max_load": round(float(t[mt.LOAD]) / 100.0, 2),
                    "mean_load": round(float(sw[k] / cnt[k]) / 100.0, 2),
                    "overload_rows": int(over[k]),
                    "peak_x": round(t[mt.X] / 100.0, 2), "peak_y": round(t[mt.Y] / 100.0, 2),
                    "t": round(start_t + t[mt.T] / 10.0, 1)})
    out.sort(key=lambda c: (-c["max_load"], -c["area_m2"]))
    return out[:int(cfg["max_heavy"])]


def _dt(rows):
    """Seconds each row ends, as mowtrail.summarize counts them."""
    if len(rows) < 2:
        return np.zeros(len(rows))
    return np.concatenate(([0.0], np.minimum(np.diff(rows[:, mt.T]) / 10.0, 60.0)))


def _work(rows, mask, dt, steps, cutm):
    """Load, length and time of the rows in mask (whole-run arrays)."""
    state = rows[:, mt.STATE] & (mt.BRK - 1)
    cut = mask & cutm
    out = {"time_s": int(dt[mask].sum()), "time_mow_s": int(dt[cut].sum()),
           "length_m": round(float(steps[mask].sum()), 1),
           "cut_m": round(float(steps[cut].sum()), 1)}
    good = cut & (rows[:, mt.LOAD] != mt.NONE)
    load = rows[good, mt.LOAD] / 100.0
    wts = steps[good]
    work = mask & (cutm | (state == mt.OVERLOAD) | (state == mt.BLOCKED))
    bad = work & ((state == mt.OVERLOAD) | (state == mt.BLOCKED))
    wt = float(dt[work].sum())
    out["load"] = {"rows": int(len(load)), "p50": _pct(load, 50, wts),
                   "p90": _pct(load, 90, wts),
                   "max": round(float(load.max()), 3) if len(load) else None,
                   "mean": round(float(load.mean()), 3) if len(load) else None,
                   "overload_blocked_frac": round(float(dt[bad].sum()) / wt, 4) if wt > 0 else None,
                   "overload_rows": int((mask & (state == mt.OVERLOAD)).sum()),
                   "blocked_rows": int((mask & (state == mt.BLOCKED)).sum())}
    return out


def _loc(rows, mask, events, start_t, cfg):
    sub = rows[mask]
    xte = np.abs(_known(sub[:, mt.XTE]))
    lidar = _known(sub[:, mt.LIDAR])
    jump = _known(sub[:, mt.JUMP])
    fix = _known(sub[:, mt.FIX])
    off = [e for e in events if e[3] == "off_path"]
    out = {"xte_p95_cm": _pct(xte, 95, digits=0), "xte_max_cm": int(xte.max()) if len(xte) else None,
           "lidar_p95_cm": _pct(lidar, 95, digits=0), "jump_p95_cm": _pct(jump, 95, digits=0),
           "rtk_frac": round(float((fix == 4).mean()), 3) if len(fix) else None,
           "gps_jumps": sum(1 for e in events if e[3] == "gps_jump"),
           "fix_losses": sum(1 for e in events if e[3] == "fix_down"),
           "off_path_n": len(off),
           "off_path": [{"x": round(e[1] / 100.0, 2), "y": round(e[2] / 100.0, 2),
                         "t": round(start_t + e[0] / 10.0, 1), "text": str(e[4])[:60]}
                        for e in off[:int(cfg["max_off_path"])]]}
    for key in ("xte_p95_cm", "lidar_p95_cm", "jump_p95_cm"):
        if out[key] is not None:
            out[key] = int(out[key])
    out["worst"] = None
    for col in (mt.LIDAR, mt.JUMP):
        vals = sub[:, col]
        if (vals != mt.NONE).any():
            i = int(np.argmax(np.where(vals != mt.NONE, vals, -1)))
            if out["worst"] is None or vals[i] > out["worst"][2]:
                out["worst"] = [round(sub[i, mt.X] / 100.0, 2), round(sub[i, mt.Y] / 100.0, 2),
                                int(vals[i])]
    jumps = [e for e in events if e[3] == "gps_jump"]
    out["gps_jump_at"] = [round(jumps[0][1] / 100.0, 2), round(jumps[0][2] / 100.0, 2)] if jumps else None
    return out


def _advice(name, z, cfg):
    out = []
    cov = z.get("coverage")
    if z.get("complete") and cov and cov["gap_m2"] >= cfg["resow_min_m2"]:
        g = cov["gaps"][0]
        out.append({"kind": "resow_gaps", "zone": name, "where": [g["x"], g["y"]],
                    "value": cov["gap_m2"],
                    "why": "%d gap(s) %.1f m2, largest %.1f m2, coverage %.1f %%"
                           % (cov["gaps_total"], cov["gap_m2"], g["area_m2"],
                              100.0 * (cov["frac"] or 0.0)),
                    "gaps": [[g2["x"], g2["y"], g2["area_m2"]] for g2 in cov["gaps"]]})
    if not z.get("visited"):
        return out
    for c in z["heavy"]:
        if c["max_load"] >= cfg["heavy_advice_load"] or c["overload_rows"]:
            out.append({"kind": "heavy_area", "zone": name, "where": [c["x"], c["y"]],
                        "value": c["max_load"], "peak": [c["peak_x"], c["peak_y"]],
                        "area_m2": c["area_m2"],
                        "why": "max load %.2f, %.1f m2, %d overload row(s)"
                               % (c["max_load"], c["area_m2"], c["overload_rows"])})
            if sum(1 for a in out if a["kind"] == "heavy_area") >= 3:
                break
    loc = z["loc"]
    hits = []                       # (metric, value, text, where)
    worst = loc["worst"][:2] if loc["worst"] else None
    if loc["rtk_frac"] is not None and loc["rtk_frac"] < cfg["rtk_min"]:
        hits.append(("rtk_frac", loc["rtk_frac"], "RTK %.0f %%" % (100.0 * loc["rtk_frac"]), worst))
    if loc["lidar_p95_cm"] is not None and loc["lidar_p95_cm"] > cfg["lidar_p95_cm"]:
        hits.append(("lidar_p95_cm", loc["lidar_p95_cm"], "lidar p95 %d cm" % loc["lidar_p95_cm"],
                     worst))
    if loc["jump_p95_cm"] is not None and loc["jump_p95_cm"] > cfg["jump_p95_cm"]:
        hits.append(("jump_p95_cm", loc["jump_p95_cm"], "jump p95 %d cm" % loc["jump_p95_cm"], worst))
    if loc["gps_jumps"]:
        hits.append(("gps_jumps", loc["gps_jumps"], "%d GPS jump(s)" % loc["gps_jumps"],
                     loc["gps_jump_at"] or worst))
    if hits:
        out.append({"kind": "gnss_check", "zone": name, "where": hits[0][3],
                    "metric": hits[0][0], "value": hits[0][1],
                    "why": ", ".join(h[2] for h in hits)})
    bits = []
    if loc["xte_p95_cm"] is not None and loc["xte_p95_cm"] > cfg["xte_p95_cm"]:
        bits.append("xte p95 %d cm" % loc["xte_p95_cm"])
    if loc["off_path_n"] >= cfg["off_path_n"]:
        bits.append("%d off_path" % loc["off_path_n"])
    if bits:
        first = loc["off_path"][0] if loc["off_path"] else None
        out.append({"kind": "path_tracking", "zone": name,
                    "where": [first["x"], first["y"]] if first else None,
                    "value": loc["xte_p95_cm"], "why": ", ".join(bits)})
    return out


# --------------------------------------------------------------------------
# runs and chains
# --------------------------------------------------------------------------
def pick_run(site, run=None):
    """run as given, or the newest finished run, or the open one when none is."""
    if run not in (None, "", "latest", "last"):
        return run
    runs = mt.list_runs(site)
    if runs:
        return runs[0]["id"]
    d = mt.run_dir(site)
    try:
        open_runs = sorted(n[:-6] for n in os.listdir(d) if n.endswith(".jsonl"))
    except OSError:
        open_runs = []
    return open_runs[-1] if open_runs else None


def report_path(site, run):
    return os.path.join(mt.run_dir(site), str(run) + SUFFIX)


def _cfg(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


def _candidates(site, run, program, start, cfg, runs=None):
    """Finished runs of the same program that ended at most chain_h before `start`,
    newest first: [(id, end)]."""
    out = []
    for r in (mt.list_runs(site) if runs is None else runs):
        if r.get("id") == run or (r.get("program") or "") != program or not program:
            continue
        end = r.get("end")
        if end is None or start is None or end > start + 1.0:
            continue
        if start - end <= cfg["chain_h"] * 3600.0:
            out.append((r["id"], float(end)))
    out.sort(key=lambda c: -c[1])
    return out


def _chain(site, run, program, start, first_pos, order, datum, cfg):
    """The earlier runs this one continues, newest first: [(id, meta, rows, events, ctx)]."""
    out = []
    cands = _candidates(site, run, program, start, cfg)
    pos = first_pos
    while cands and len(out) < int(cfg["chain_max"]):
        prev_id = cands.pop(0)[0]
        got = mt.read_run(site, prev_id, datum)
        if got is None:
            break
        meta, rows, events = got
        ctx = _ctx(events)
        if not ctx or _position(order, ctx[-1][1], ctx[-1][2]) > pos:
            break
        out.append((prev_id, meta, rows, events, ctx))
        p0 = _position(order, ctx[0][1], ctx[0][2])
        if p0 <= _position(order, order[0] if order else ctx[0][1], 1):
            break                   # that one started at the beginning
        pos = p0
        start = float(meta.get("start") or 0.0)
        cands = [c for c in cands
                 if c[1] <= start + 1.0 and start - c[1] <= cfg["chain_h"] * 3600.0]
    return out


def _labels(rows, events, ctx, grids, prog_zones):
    """Zone name per row and per event ('' = outside), and how they were told."""
    names = np.full(len(rows), "", dtype=object)
    ev_names = [""] * len(events)
    xs, ys = rows[:, mt.X] / 100.0, rows[:, mt.Y] / 100.0
    ex = np.array([e[1] for e in events], float) / 100.0
    ey = np.array([e[2] for e in events], float) / 100.0
    et = np.array([e[0] for e in events], np.int64)
    if ctx:
        times = np.array([c[0] for c in ctx])
        cz = np.array([c[1] for c in ctx], dtype=object)
        ri = np.searchsorted(times, rows[:, mt.T], side="right") - 1
        ei = np.searchsorted(times, et, side="right") - 1
        for z in set(cz.tolist()):
            if z not in grids:
                # an unknown zone: no polygon to hold the rows to, the ctx tells
                names[(ri >= 0) & (cz[np.maximum(ri, 0)] == z)] = z
                for j in np.nonzero((ei >= 0) & (cz[np.maximum(ei, 0)] == z))[0]:
                    ev_names[j] = z
                continue
            sel = (ri >= 0) & (cz[np.maximum(ri, 0)] == z)
            sel[sel] = grids[z].inside(xs[sel], ys[sel])
            names[sel] = z
            if len(events):
                es = (ei >= 0) & (cz[np.maximum(ei, 0)] == z)
                es[es] = grids[z].inside(ex[es], ey[es])
                for j in np.nonzero(es)[0]:
                    ev_names[j] = z
        return names, ev_names, "ctx"
    cand = [z for z in prog_zones if z in grids]
    for z in cand:
        free = names == ""
        free[free] = grids[z].inside(xs[free], ys[free])
        names[free] = z
        if len(events):
            for j in np.nonzero(grids[z].inside(ex, ey))[0]:
                if not ev_names[j]:
                    ev_names[j] = z
    return names, ev_names, ("program" if cand else "none")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _npz_stamp(site, run):
    try:
        st = os.stat(os.path.join(mt.run_dir(site), run + ".npz"))
        return [round(st.st_mtime, 3), int(st.st_size)]
    except OSError:
        return None


def signature(site, run, cfg=None, shapes=None, datum=None, runs=None):
    """What a stored report depends on, hashed. None when the run is not stored."""
    cfg = _cfg(cfg)
    stamp = _npz_stamp(site, run)
    if stamp is None:
        return None
    if datum is None:
        datum = worldlayers.read_datum(site)
    zones, excl = shapes if shapes is not None else _site_shapes(site, datum)
    runs = mt.list_runs(site) if runs is None else runs
    me = next((r for r in runs if r.get("id") == run), {})
    program = me.get("program") or ""
    pz = _program_zones(site, program)
    picked = {z: zones[z] for z in sorted(zones) if not pz or z in pz}

    def rnd(rings):
        return [[[round(p[0], 3), round(p[1], 3)] for p in r] for r in rings]
    cands = _candidates(site, run, program, me.get("start"), cfg, runs)
    body = {"v": REPORT_VERSION,
            "datum": [round(v, 4) for v in mt._datum_row(datum)] if datum else None,
            "program": program, "order": pz,
            "zones": {z: [rnd(v["rings"]), rnd(v["holes"])] for z, v in picked.items()},
            "excl": sorted(json.dumps(r) for r in rnd(excl)),
            "deck_m": cfg["deck_m"], "npz": stamp,
            "chain": [[c[0], _npz_stamp(site, c[0])] for c in cands]}
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()


def report(site, run=None, cfg=None, now=None, shapes=None):
    """The report of a run as a dict; {"ok": False, "reason"} when it cannot be made."""
    cfg = _cfg(cfg)
    run = pick_run(site, run)
    datum = worldlayers.read_datum(site)
    if run is None:
        return {"ok": False, "reason": "no run", "site": site, "run": None}
    got = mt.read_run(site, run, datum)
    if got is None:
        return {"ok": False, "reason": "no such run", "site": site, "run": run}
    meta, rows, events = got
    start_t = float(meta.get("start") or 0.0)
    zones, excl = shapes if shapes is not None else _site_shapes(site, datum)
    program = meta.get("program") or ""
    stop = meta.get("stop_reason") or ""
    succeeded = stop == "succeeded"
    prog_zones = _program_zones(site, program)
    ctx = _ctx(events)
    order = list(prog_zones)
    for c in ctx:
        if c[1] not in order:
            order.append(c[1])
    grids = {}
    for z in order:
        if z in zones and z not in grids:
            grids[z] = _Grid(zones[z], excl, cfg)
    names, ev_names, how = _labels(rows, events, ctx, grids, prog_zones)

    # progress of this run and of the runs it continues
    prog = _progress(ctx, succeeded)
    resumed = bool(ctx) and _position(order, ctx[0][1], ctx[0][2]) > _position(order, order[0], 1)
    chain = (_chain(site, run, program, start_t, _position(order, ctx[0][1], ctx[0][2]), order,
                    datum, cfg) if resumed else [])
    union = {z: {"done": set(p["done"]), "total": p["total"]} for z, p in prog.items()}
    points = [_cut_points(rows, cfg["grid_m"] / 2.0, cfg["max_seg_m"])]
    for _id, pmeta, prows, _pev, pctx in chain:
        for z, p in _progress(pctx, (pmeta.get("stop_reason") or "") == "succeeded").items():
            u = union.setdefault(z, {"done": set(), "total": p["total"]})
            u["done"] |= p["done"]
            u["total"] = max(u["total"], p["total"])
        points.append(_cut_points(prows, cfg["grid_m"] / 2.0, cfg["max_seg_m"]))
    points = np.concatenate(points) if len(points) > 1 else points[0]

    def zone_done(z, table):
        if not ctx:
            return succeeded and how != "none"
        p = table.get(z)
        return bool(p) and p["total"] > 0 and set(range(1, p["total"] + 1)) <= p["done"]

    dt = _dt(rows)
    steps = _steps(rows)
    cutm = _cut_mask(rows)
    visited_names = set(n for n in names.tolist() if n) | set(prog)
    out_zones = []
    advice = []
    left = []
    tot_mow = tot_cov = 0.0
    for name in order:
        mask = names == name
        visited = name in visited_names
        if not visited and name not in grids:
            continue
        p = prog.get(name)
        z = {"zone": name, "known": name in grids, "visited": visited,
             "paths_first": p["first"] if p else None, "paths_last": p["last"] if p else None,
             "paths_total": (p or union.get(name) or {}).get("total"),
             "paths_done": sorted(union[name]["done"]) if name in union else []}
        z["complete_run"] = zone_done(name, prog)
        z["complete"] = zone_done(name, union)
        evs = [e for e, n in zip(events, ev_names) if n == name]
        z.update(_work(rows, mask, dt, steps, cutm))
        z["coverage"] = _coverage(grids[name], points, cfg) if name in grids else None
        # the hardest tenth of the zone, never below heavy_load: a lawn heavy all
        # over is told by its p90, the clusters say where it was worst
        z["heavy_limit"] = round(max(cfg["heavy_load"], z["load"]["p90"] or 0.0), 3)
        z["heavy"] = _heavy(rows[mask], cfg, start_t, z["heavy_limit"]) if visited else []
        z["loc"] = _loc(rows, mask, evs, start_t, cfg)
        z["stops"] = sum(1 for e in evs if e[3] in mt.STOP_KINDS)
        cov = z["coverage"]
        if cov and cov["mowable_m2"]:
            tot_mow += cov["mowable_m2"]
            tot_cov += cov["covered_m2"]
        if not z["complete"]:
            tot = z["paths_total"]
            rest = (sorted(set(range(1, tot + 1)) - set(z["paths_done"])) if tot else None)
            left.append({"zone": name, "visited": visited, "paths": rest, "paths_total": tot})
        advice.extend(_advice(name, z, cfg))
        out_zones.append(z)
    complete = all(z["complete"] for z in out_zones) and bool(out_zones)
    if left and (ctx or not succeeded):
        first = next((z for z in out_zones if z["zone"] == left[0]["zone"]), None)
        where = None
        if first and first["zone"] in grids:
            g = grids[first["zone"]]
            rr, cc = np.nonzero(g.zone)
            if len(rr):
                where = [round(g.ox + (cc.mean() + 0.5) * g.res, 2),
                         round(g.oy + (rr.mean() + 0.5) * g.res, 2)]
        npaths = sum(len(x["paths"]) if x["paths"] is not None else 1 for x in left)
        advice.insert(0, {"kind": "unfinished", "zone": left[0]["zone"], "where": where,
                          "value": npaths, "remaining": left,
                          "why": "%s; left: %s" % (
                              stop or "no stop reason",
                              "; ".join("%s %s" % (x["zone"], (
                                  "paths " + ",".join(map(str, x["paths"])) if x["paths"]
                                  else "not visited" if not x["visited"] else "all"))
                                  for x in left))})
    outside = names == ""
    total = mt.summarize(rows, events)
    whole = _work(rows, np.ones(len(rows), bool), dt, steps, cutm)["load"]
    total.update({"coverage_frac": round(tot_cov / tot_mow, 4) if tot_mow else None,
                  "mowable_m2": round(tot_mow, 1), "covered_m2": round(tot_cov, 1),
                  "gap_m2": round(sum(z["coverage"]["gap_m2"] for z in out_zones
                                      if z["coverage"]), 2),
                  "load_p50": whole["p50"], "load_p90": whole["p90"],
                  "overload_blocked_frac": whole["overload_blocked_frac"]})
    paths = {"first": ([ctx[0][1], ctx[0][2]] if ctx else None),
             "last": ([ctx[-1][1], ctx[-1][2]] if ctx else None),
             "total": {z: p["total"] for z, p in prog.items()}}
    return {"ok": True, "v": REPORT_VERSION, "site": site, "run": run, "program": program,
            "start": meta.get("start"), "end": meta.get("end"),
            "stop_reason": stop, "active": "end" not in meta,
            "generated": round(time.time() if now is None else now, 1),
            "datum": mt._datum_row(datum) if datum else None,
            "rows": int(len(rows)), "zones_from": how,
            "paths_first": paths["first"], "paths_last": paths["last"],
            "paths_total": paths["total"],
            "resumed": resumed, "continues": [c[0] for c in chain],
            "complete": succeeded and all(z["complete_run"] for z in out_zones) and bool(out_zones),
            "complete_union": complete,
            "cfg": {k: cfg[k] for k in ("deck_m", "grid_m", "idle_load", "heavy_load",
                                        "gap_open_m", "min_gap_m2", "max_seg_m")},
            "total": total, "zones": out_zones,
            "outside": dict(_work(rows, outside, dt, steps, cutm), rows=int(outside.sum())),
            "advice": advice}


def _atomic_write(path, payload):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save(site, run, cfg=None, shapes=None, sig=None):
    """Compute and write <run>.report.json; the path, None when there is no report."""
    rep = report(site, run, cfg, shapes=shapes)
    if not rep.get("ok") or rep.get("active"):
        return None
    rep["sig"] = sig or signature(site, rep["run"], cfg, shapes)
    path = report_path(site, rep["run"])
    _atomic_write(path, rep)
    return path


def _stored(path):
    try:
        with open(path) as f:
            rep = json.load(f)
        return rep if isinstance(rep, dict) else None
    except (IOError, OSError, ValueError):
        return None


def load(site, run=None, cfg=None):
    """Stored report when still valid, else computed now (not written). Never raises."""
    try:
        run = pick_run(site, run)
        if run is None:
            return {"ok": False, "reason": "no run", "site": site, "run": None}
        path = report_path(site, run)
        if os.path.exists(path):
            rep = _stored(path)
            sig = signature(site, run, cfg)
            if rep and rep.get("ok") and sig and rep.get("sig") == sig:
                rep["path"] = path
                return rep
        rep = report(site, run, cfg)
        rep["path"] = None
        return rep
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:200], "site": site, "run": run}


def backfill(site, cfg=None, log=None, pause_s=None):
    """Write the missing or stale reports of finished runs one by one, pause_s
    between two computed ones; drop those of runs retention removed."""
    cfg = _cfg(cfg)
    pause = cfg["backfill_pause_s"] if pause_s is None else pause_s
    d = mt.run_dir(site)
    try:
        names = os.listdir(d)
    except OSError:
        return []
    runs = set(n[:-4] for n in names if n.endswith(".npz"))
    for n in names:
        if n.endswith(SUFFIX) and n[:-len(SUFFIX)] not in runs:
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass
    datum = worldlayers.read_datum(site)
    shapes = _site_shapes(site, datum)
    index = mt.list_runs(site)
    done = []
    for run in sorted(runs, reverse=True):
        try:
            sig = signature(site, run, cfg, shapes, datum, index)
            rep = _stored(report_path(site, run))
            if rep and rep.get("sig") == sig:
                continue
            if done and pause > 0:
                time.sleep(pause)
            if save(site, run, cfg, shapes, sig):
                done.append(run)
        except Exception as exc:                            # noqa: BLE001
            if log:
                log("mow_report %s: %s" % (run, exc))
    return done


# --------------------------------------------------------------------------
# worker for a node: one thread, one child process at nice 19 at a time
# --------------------------------------------------------------------------
def _child(args, timeout_s):
    """(exit code, stdout) of `python3 -m vitulus_mapping.mowreport args` at nice 19.
    A child keeps the computation off the node's GIL; nice keeps it off the CPU
    when anything else wants it."""
    import subprocess
    try:
        got = subprocess.run([sys.executable, "-m", "vitulus_mapping.mowreport"] + list(args),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             preexec_fn=lambda: os.nice(19), timeout=timeout_s)
        err = got.stderr.decode("utf-8", "replace").strip()
        return got.returncode, got.stdout.decode("utf-8", "replace"), err[-300:]
    except subprocess.TimeoutExpired:
        return -1, "", "timeout after %d s" % timeout_s
    except OSError as exc:
        return -1, "", str(exc)


class Worker(object):
    """Report jobs of a node. backfill(site) and request(site, run, req) queue a job;
    a job already queued for the same run (or the same site's backfill) takes the
    new request along instead of a second computation. At most `limit` jobs wait;
    a request beyond that is answered {"ok": False, "reason": "busy"}.
    answer(report dict with "req" and "reqs") is called from the worker thread."""

    def __init__(self, answer, log=None, cfg=None, limit=8, timeout_s=600, runner=None):
        import collections
        import threading
        self.answer = answer
        self.log = log or (lambda text: None)
        self.cfg = _cfg(cfg)
        self.limit = int(limit)
        self.timeout_s = timeout_s
        self.runner = runner or _child
        self.jobs = collections.OrderedDict()      # key -> [req, ...]
        self.cond = threading.Condition()
        self.closed = False
        self.thread = threading.Thread(target=self._loop, name="mow_report", daemon=True)
        self.thread.start()

    def _put(self, key, req):
        with self.cond:
            if key in self.jobs:
                if req is not None:
                    self.jobs[key].append(req)
                return True
            if len(self.jobs) >= self.limit:
                return False
            self.jobs[key] = [] if req is None else [req]
            self.cond.notify()
            return True

    def backfill(self, site):
        self._put(("all", site, ""), None)

    def request(self, site, run, req=None):
        if not site:
            self.answer({"ok": False, "reason": "no site", "req": req, "reqs": [req]})
            return
        if not self._put(("run", site, run or ""), req):
            self.answer({"ok": False, "reason": "busy", "site": site, "run": run or None,
                         "req": req, "reqs": [req]})

    def pending(self):
        with self.cond:
            return list(self.jobs)

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify()

    def _args(self, site):
        return ["--site", site, "--deck", repr(float(self.cfg["deck_m"]))]

    def _loop(self):
        while True:
            with self.cond:
                while not self.jobs and not self.closed:
                    self.cond.wait()
                if self.closed:
                    return
                key = next(iter(self.jobs))
                reqs = self.jobs.pop(key)
            try:
                self._run(key, reqs)
            except Exception as exc:                        # noqa: BLE001
                self.log("mow_report: %s" % exc)

    def _run(self, key, reqs):
        kind, site, run = key
        if kind == "all":
            code, out, err = self.runner(self._args(site) + [
                "--all", "--pause", repr(float(self.cfg["backfill_pause_s"]))], self.timeout_s)
            if code != 0:
                self.log("mow_report backfill: %s" % (err or code))
            elif out.strip() and out.strip() != "written: -":
                self.log(out.strip())
            return
        code, out, err = self.runner(self._args(site) + ["--json"] + ([run] if run else []),
                                     self.timeout_s)
        try:
            rep = json.loads(out)
            if not isinstance(rep, dict):
                raise ValueError("not a report")
        except ValueError:
            rep = {"ok": False, "reason": (err or "exit %s" % code)[:200], "site": site,
                   "run": run or None}
        rep["req"] = reqs[-1] if reqs else None
        rep["reqs"] = reqs
        self.answer(rep)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _text(rep):
    if not rep.get("ok"):
        return "no report: %s" % rep.get("reason")
    t = rep["total"]
    lines = ["run %s  program %s  %s  rows %d  cut %.0f m  mow %d s  coverage %s  gaps %.1f m2"
             "  paths %s..%s  resumed %s%s  complete %s"
             % (rep["run"], rep["program"] or "-", rep["stop_reason"] or "-", rep["rows"],
                t["cut_m"], t["time_mow_s"],
                "-" if t["coverage_frac"] is None else "%.1f %%" % (100 * t["coverage_frac"]),
                t["gap_m2"], rep["paths_first"], rep["paths_last"], rep["resumed"],
                (" (continues %s)" % ",".join(rep["continues"])) if rep["continues"] else "",
                rep["complete_union"])]
    for z in rep["zones"]:
        c = z["coverage"] or {}
        ld, lc = z["load"], z["loc"]
        lines.append("  %-12s %s cut %6.1f m %5d s  load p50 %s p90 %s max %s ovl/blk %s  "
                     "cov %s gaps %s (%s m2)  heavy %d  xte95 %s lidar95 %s jump95 %s rtk %s off %d"
                     % (z["zone"], "done" if z["complete"] else
                        ("part" if z["visited"] else "none"),
                        z["cut_m"], z["time_mow_s"], ld["p50"], ld["p90"], ld["max"],
                        ld["overload_blocked_frac"],
                        "-" if c.get("frac") is None else "%.1f%%" % (100 * c["frac"]),
                        c.get("gaps_total", "-"), c.get("gap_m2", "-"), len(z["heavy"]),
                        lc["xte_p95_cm"], lc["lidar_p95_cm"], lc["jump_p95_cm"], lc["rtk_frac"],
                        lc["off_path_n"]))
    for a in rep["advice"]:
        lines.append("  -> %s %s at %s: %s" % (a["kind"], a["zone"], a["where"], a["why"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mow_report", description="Zone report of a mow run.")
    ap.add_argument("run", nargs="?", default=None, help="run id, default the newest finished")
    ap.add_argument("--site", default=None, help="default the served site")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--save", action="store_true", help="write <run>.report.json")
    ap.add_argument("--all", action="store_true", help="write reports of all finished runs")
    ap.add_argument("--deck", type=float, default=None, help="cut width m, default %.2f"
                    % DEFAULTS["deck_m"])
    ap.add_argument("--pause", type=float, default=None, help="--all: s between two runs")
    args = ap.parse_args(argv)
    site = args.site or worldlayers.read_served_site()
    if not site:
        print("no site: give --site", file=sys.stderr)
        return 2
    cfg = {"deck_m": args.deck} if args.deck else None
    if args.all:
        done = backfill(site, cfg, lambda s: print(s, file=sys.stderr), args.pause)
        print("written: %s" % (", ".join(done) or "-"))
        return 0
    if args.save:
        path = save(site, pick_run(site, args.run), cfg)
        print("written: %s" % (path or "-"), file=sys.stderr if args.json else sys.stdout)
    rep = load(site, args.run, cfg)
    print(json.dumps(rep, separators=(",", ":")) if args.json else _text(rep))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
