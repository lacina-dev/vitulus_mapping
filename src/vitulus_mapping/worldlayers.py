#!/usr/bin/env python3
"""
worldlayers.py -- recorded world layers of a site: where the blade cut, how good
GNSS is, how steep the ground is. Contract: docs/MAP_OBJECTS.md section 7.

NO rospy. `WorldRecorder` holds all the logic of nodes/map_world_recorder and is
driven with plain values, so all of it runs in tests without a ROS master.

Grids are arrays [row, col]: row 0 = south, col 0 = west, map-frame axes, origin
= (ox, oy) the south-west corner of cell (0, 0), aligned to whole GROW_CELLS
blocks so that growing never shifts cells.

coverage.npz       res, ox, oy, datum, last_mowed (uint32 unix s, 0 = never)
gnss_quality.npz   res, ox, oy, datum, hacc_best_mm, hacc_median_mm, class_best,
                   class_median (0 = no data in the hAcc arrays, class -1), plus
                   hacc_hist, class_hist, hacc_edges_mm from which the medians
                   are derived and accumulate across restarts; rtcm_age_ds worst
                   RTCM age seen, 0.1 s, RTCM_NONE = not known, absent in old files

Each file stores the datum it was written under. Rotating or moving the datum
moves the map frame under the grid, so a loaded grid is resampled into the
current datum (nearest cell) rather than read as if nothing happened.

Fix class: 0 no fix, 1 2D/3D fix, 2 DGNSS, 3 RTK float, 4 RTK fixed.
"""

import base64
import json
import math
import os
import struct
import tempfile
import time
import zlib

import numpy as np
import yaml

from . import bundle
from . import geo

__all__ = ["CoverageGrid", "GnssGrid", "WorldRecorder", "fix_class", "encode_png",
           "slope_from_dem", "coverage_layer", "gnss_layer", "slope_layer",
           "empty_layer", "read_served_site", "read_datum", "layer_query",
           "DEFAULTS"]

FORMAT_VERSION = 1
GROW_CELLS = 64
MAX_CELLS = 6000            # one axis; a pose beyond this is not a garden
MAX_PNG = 1024
COVERAGE_RES = 0.10
GNSS_RES = 0.50

# hAcc histogram: geometric bins in mm, one extra bin above the last edge
HACC_EDGES_MM = np.array([10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320, 640,
                          1280, 2560, 5120], dtype=np.uint32)
N_CLASS = 5
HIST_CAP = 60000            # halve a cell's histogram before uint16 overflows
BEST_NONE = 0xFFFFFFFF
RTCM_NONE = 0xFFFF

DEFAULTS = {
    "deck_diameter_m": 0.29,      # vitulus_planner MapSpec.COVERAGE_DIAMETER
    "deck_offset_x_m": 0.0,       # planner drives the deck centre on base_link
    "max_step_m": 1.0,            # longer pose step = jump, do not sweep across
    "gnss_step_m": 0.25,          # one GNSS sample per this much travel
    "flush_s": 30.0,
    "layer_min_s": 30.0,          # contract: at most every 30 s
    "layer_refresh_s": 3600.0,    # coverage colours age with time
    "site_poll_s": 5.0,
    "slope_res_m": 0.5,
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def fix_class(fix_type, flags):
    """Fix class from NAV-PVT fixType and flags."""
    if not flags & 0x01 or fix_type < 2 or fix_type > 4:      # gnssFixOK
        return 0
    carr = (flags >> 6) & 0x03
    if carr == 2:
        return 4
    if carr == 1:
        return 3
    if flags & 0x02:                                          # diffSoln
        return 2
    return 1


def encode_png(rgba):
    """PNG of an [h, w, 4] uint8 array, first array row = first PNG row."""
    h, w = rgba.shape[:2]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    rows = np.zeros((h, w * 4 + 1), np.uint8)        # filter byte 0 = None
    rows[:, 1:] = rgba.reshape(h, w * 4)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + chunk(b"IEND", b""))


def _hex_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _paint(indices, valid, colors, alpha=200):
    """RGBA [h, w, 4] from a palette index per cell; invalid cells transparent."""
    palette = np.zeros((len(colors) + 1, 4), np.uint8)
    for i, c in enumerate(colors):
        palette[i, :3] = _hex_rgb(c)
        palette[i, 3] = alpha
    idx = np.where(valid, np.clip(indices, 0, len(colors) - 1), len(colors))
    return palette[idx]


def _layer_msg(site, now, ox, oy, res, rgba, valid, legend):
    """Layer JSON payload, cropped to the data. Row 0 of the arrays is south,
    the PNG is a normal image, so it is flipped: first PNG row = highest y."""
    if not valid.any():
        return empty_layer(site, now, res, legend)
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    img = rgba[r0:r1, c0:c1]
    step = int(math.ceil(max(img.shape[0], img.shape[1]) / float(MAX_PNG)))
    if step > 1:
        img = img[::step, ::step]
    return {"site": site, "stamp": now,
            "origin": [round(ox + c0 * res, 3), round(oy + r0 * res, 3)],
            "res": round(res * step, 4), "w": int(img.shape[1]), "h": int(img.shape[0]),
            "png_b64": base64.b64encode(encode_png(img[::-1])).decode("ascii"),
            "legend": legend}


def empty_layer(site, now, res, legend):
    return {"site": site, "stamp": now, "origin": [0.0, 0.0], "res": res,
            "w": 0, "h": 0, "png_b64": "", "legend": legend}


def _atomic_npz(path, arrays):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------
class _Grid(object):
    """Growing grid of named arrays sharing rows and columns."""
    OPTIONAL = ()

    RES = 1.0
    FILENAME = ""
    # name -> (dtype, fill, extra shape)
    LAYERS = {}

    def __init__(self):
        self.res = self.RES
        self.ox = 0.0
        self.oy = 0.0
        self.a = {}
        for name, (dtype, _fill, extra) in self.LAYERS.items():
            self.a[name] = np.zeros((0, 0) + extra, dtype)
        self.dirty = False          # differs from the file
        self.changed = False        # differs from the last published layer

    @property
    def h(self):
        return next(iter(self.a.values())).shape[0]

    @property
    def w(self):
        return next(iter(self.a.values())).shape[1]

    def ensure(self, x0, y0, x1, y1):
        """Grow so the box exists. False when it would exceed MAX_CELLS."""
        block = GROW_CELLS * self.res
        bx0 = math.floor(x0 / block) * GROW_CELLS
        by0 = math.floor(y0 / block) * GROW_CELLS
        bx1 = (math.floor(x1 / block) + 1) * GROW_CELLS
        by1 = (math.floor(y1 / block) + 1) * GROW_CELLS
        if self.h == 0:
            nc0, nr0, nc1, nr1 = bx0, by0, bx1, by1
        else:
            c0 = int(round(self.ox / self.res))
            r0 = int(round(self.oy / self.res))
            if bx0 >= c0 and by0 >= r0 and bx1 <= c0 + self.w and by1 <= r0 + self.h:
                return True
            nc0, nr0 = min(c0, bx0), min(r0, by0)
            nc1, nr1 = max(c0 + self.w, bx1), max(r0 + self.h, by1)
        cap = getattr(self, "MAX_CELLS", MAX_CELLS)      # a subclass may cap lower
        if nc1 - nc0 > cap or nr1 - nr0 > cap:
            return False
        if self.h == 0:
            dx, dy = 0, 0
        else:
            dx, dy = c0 - nc0, r0 - nr0
        for name, (dtype, fill, extra) in self.LAYERS.items():
            grown = np.full((nr1 - nr0, nc1 - nc0) + extra, fill, dtype)
            old = self.a[name]
            grown[dy:dy + old.shape[0], dx:dx + old.shape[1]] = old
            self.a[name] = grown
        self.ox = nc0 * self.res
        self.oy = nr0 * self.res
        return True

    # ---- files --------------------------------------------------------------
    def _extra_arrays(self):
        return {}

    def save(self, site_dir, site, datum):
        arrays = dict(self.a)
        arrays.update(self._extra_arrays())
        arrays.update(version=np.array(FORMAT_VERSION), site=np.array(site),
                      res=np.array(self.res), ox=np.array(self.ox), oy=np.array(self.oy),
                      datum=np.array([datum["utm_e"], datum["utm_n"], datum["yaw_rad"]]))
        _atomic_npz(os.path.join(site_dir, self.FILENAME), arrays)
        self.dirty = False

    @classmethod
    def load(cls, site_dir, datum, readonly=False):
        """Grid from the site's file, in `datum`. Missing = empty grid. A file
        that cannot be read is set aside, never overwritten (`readonly`: only
        left alone, for readers other than the recorder)."""
        grid = cls()
        path = os.path.join(site_dir, cls.FILENAME)
        if not os.path.exists(path):
            return grid
        try:
            with np.load(path, allow_pickle=False) as data:
                if abs(float(data["res"]) - cls.RES) > 1e-9:
                    raise ValueError("resolution %s" % data["res"])
                grid.ox = float(data["ox"])
                grid.oy = float(data["oy"])
                for name in cls.LAYERS:
                    if name in cls.OPTIONAL and name not in data:
                        grid.a.pop(name, None)     # filled below, to the file's shape
                        continue
                    arr = data[name]
                    if arr.shape[2:] != cls.LAYERS[name][2]:
                        raise ValueError("shape of %s" % name)
                    grid.a[name] = arr.astype(cls.LAYERS[name][0])
                shapes = set(arr.shape[:2] for arr in grid.a.values())
                if len(shapes) != 1:
                    raise ValueError("array shapes differ")
                for name in cls.OPTIONAL:
                    if name not in grid.a:
                        kind, fill, tail = cls.LAYERS[name]
                        grid.a[name] = np.full(shapes.copy().pop() + tail, fill, kind)
                stored = [float(v) for v in data["datum"]]
        except Exception:                                  # noqa: BLE001
            if not readonly:
                try:
                    os.replace(path, "%s.bad.%d" % (path, int(time.time())))
                except OSError:
                    pass
            return cls()
        return grid._rebased(stored, datum)

    def _rebased(self, stored, datum):
        """This grid (written under `stored`) expressed in `datum`."""
        de = datum["utm_e"] - stored[0]
        dn = datum["utm_n"] - stored[1]
        dyaw = geo.wrap_pi(datum["yaw_rad"] - stored[2])
        if math.hypot(de, dn) < 0.02 and abs(dyaw) < 0.0005:
            return self
        if self.h == 0:
            return self
        old = {"utm_e": stored[0], "utm_n": stored[1], "yaw_rad": stored[2]}
        out = type(self)()
        xs0 = self.ox + np.array([0, self.w, 0, self.w]) * self.res
        ys0 = self.oy + np.array([0, 0, self.h, self.h]) * self.res
        corners = [geo.utm_to_map(*geo.map_to_utm(float(x), float(y), old), datum)
                   for x, y in zip(xs0, ys0)]
        if not out.ensure(min(c[0] for c in corners), min(c[1] for c in corners),
                          max(c[0] for c in corners), max(c[1] for c in corners)):
            return self
        cx = out.ox + (np.arange(out.w) + 0.5) * out.res
        cy = out.oy + (np.arange(out.h) + 0.5) * out.res
        gx, gy = np.meshgrid(cx, cy)
        c1, s1 = math.cos(datum["yaw_rad"]), math.sin(datum["yaw_rad"])
        east = datum["utm_e"] + c1 * gx - s1 * gy
        north = datum["utm_n"] + s1 * gx + c1 * gy
        c0, s0 = math.cos(old["yaw_rad"]), math.sin(old["yaw_rad"])
        ex, ny = east - old["utm_e"], north - old["utm_n"]
        col = np.floor((c0 * ex + s0 * ny - self.ox) / self.res).astype(np.int64)
        row = np.floor((-s0 * ex + c0 * ny - self.oy) / self.res).astype(np.int64)
        ok = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        for name, arr in self.a.items():
            out.a[name][ok] = arr[row[ok], col[ok]]
        out.dirty = True
        return out


class CoverageGrid(_Grid):
    RES = COVERAGE_RES
    FILENAME = "coverage.npz"
    LAYERS = {"last_mowed": (np.uint32, 0, ())}

    def stamp(self, x0, y0, x1, y1, radius, t):
        """Mark what a deck disc swept from (x0, y0) to (x1, y1)."""
        if not self.ensure(min(x0, x1) - radius, min(y0, y1) - radius,
                           max(x0, x1) + radius, max(y0, y1) + radius):
            return False
        c0 = int(math.floor((min(x0, x1) - radius - self.ox) / self.res))
        c1 = int(math.floor((max(x0, x1) + radius - self.ox) / self.res)) + 1
        r0 = int(math.floor((min(y0, y1) - radius - self.oy) / self.res))
        r1 = int(math.floor((max(y0, y1) + radius - self.oy) / self.res)) + 1
        xs = self.ox + (np.arange(c0, c1) + 0.5) * self.res
        ys = self.oy + (np.arange(r0, r1) + 0.5) * self.res
        dx, dy = x1 - x0, y1 - y0
        n = dx * dx + dy * dy
        px = xs[np.newaxis, :] - x0
        py = ys[:, np.newaxis] - y0
        u = np.clip((px * dx + py * dy) / n, 0.0, 1.0) if n > 0.0 else 0.0
        mask = (px - u * dx) ** 2 + (py - u * dy) ** 2 <= radius * radius
        self.a["last_mowed"][r0:r1, c0:c1][mask] = int(t)
        self.dirty = self.changed = True
        return True


class GnssGrid(_Grid):
    RES = GNSS_RES
    FILENAME = "gnss_quality.npz"
    LAYERS = {
        "hacc_best_mm": (np.uint32, BEST_NONE, ()),
        "class_best": (np.int8, -1, ()),
        "hacc_hist": (np.uint16, 0, (len(HACC_EDGES_MM) + 1,)),
        "class_hist": (np.uint16, 0, (N_CLASS,)),
        "rtcm_age_ds": (np.uint16, RTCM_NONE, ()),
    }
    OPTIONAL = ("rtcm_age_ds",)     # a file from before it stays valid

    def add(self, x, y, h_acc_mm, cls, rtcm_age_s=None):
        if not self.ensure(x, y, x, y):
            return False
        c = int(math.floor((x - self.ox) / self.res))
        r = int(math.floor((y - self.oy) / self.res))
        h_acc_mm = max(1, int(h_acc_mm))
        cls = max(0, min(N_CLASS - 1, int(cls)))
        if h_acc_mm < self.a["hacc_best_mm"][r, c]:
            self.a["hacc_best_mm"][r, c] = h_acc_mm
        if cls > self.a["class_best"][r, c]:
            self.a["class_best"][r, c] = cls
        hist = self.a["hacc_hist"][r, c]
        hist[int(np.searchsorted(HACC_EDGES_MM, h_acc_mm, side="left"))] += 1
        if hist.max() >= HIST_CAP:
            hist //= 2
        chist = self.a["class_hist"][r, c]
        chist[cls] += 1
        if chist.max() >= HIST_CAP:
            chist //= 2
        if rtcm_age_s is not None:
            age = min(RTCM_NONE - 1, int(round(rtcm_age_s * 10.0)))
            old = self.a["rtcm_age_ds"][r, c]
            if old == RTCM_NONE or age > old:
                self.a["rtcm_age_ds"][r, c] = age
        self.dirty = self.changed = True
        return True

    @staticmethod
    def _median_bin(hist):
        """Index of the median bin along the last axis; 0 where hist is empty."""
        total = hist.sum(axis=-1, dtype=np.int64)
        cum = np.cumsum(hist, axis=-1, dtype=np.int64)
        return np.argmax(cum * 2 >= total[..., np.newaxis], axis=-1)

    def class_median(self):
        out = self._median_bin(self.a["class_hist"]).astype(np.int8)
        out[self.a["class_hist"].sum(axis=-1) == 0] = -1
        return out

    def hacc_median_mm(self):
        """Median hAcc, geometric centre of the median bin, 0 = no data."""
        hist = self.a["hacc_hist"]
        edges = HACC_EDGES_MM.astype(np.float64)
        low = np.concatenate(([edges[0] / 1.41], edges))
        high = np.concatenate((edges, [edges[-1] * 2.0]))
        centre = np.sqrt(low * high)
        out = np.round(centre[self._median_bin(hist)]).astype(np.uint32)
        out[hist.sum(axis=-1) == 0] = 0
        return out

    def _extra_arrays(self):
        return {"hacc_median_mm": self.hacc_median_mm(),
                "class_median": self.class_median(),
                "hacc_edges_mm": HACC_EDGES_MM}


# --------------------------------------------------------------------------
# slope
# --------------------------------------------------------------------------
def slope_from_dem(dem_path, res=0.5):
    """(slope_deg [row, col], ox, oy, res) from dem.npz, None when it has no
    data. DEM cells are 5 cm trajectory samples and centimetres of z noise over
    5 cm read as tens of degrees, so they are averaged to `res` first."""
    with np.load(dem_path, allow_pickle=False) as data:
        elev = data["elev"].astype(np.float64)      # DEM axes are [ix, iy]
        dem_res = float(data["res"])
        ox, oy = float(data["ox"]), float(data["oy"])
    if elev.size == 0 or not np.isfinite(elev).any():
        return None
    block = max(1, int(round(res / dem_res)))
    z = elev.T                                       # -> [row = y, col = x]
    h, w = z.shape[0] // block, z.shape[1] // block
    if h == 0 or w == 0:
        return None
    z = z[:h * block, :w * block].reshape(h, block, w, block)
    good = np.isfinite(z)
    n = good.sum(axis=(1, 3))
    summed = np.where(good, z, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = summed / n
    cell = dem_res * block

    def grad(arr, axis):
        d = np.diff(arr, axis=axis) / cell
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        fwd = np.pad(d, pad, constant_values=np.nan)
        pad[axis] = (0, 1)
        back = np.pad(d, pad, constant_values=np.nan)
        cnt = np.isfinite(fwd).astype(np.int8) + np.isfinite(back)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.nan_to_num(fwd) + np.nan_to_num(back)) / cnt

    gx, gy = grad(mean, 1), grad(mean, 0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return slope, ox, oy, cell


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
_COVER = [(86400.0, "#22c55e", "< 1 d"), (3 * 86400.0, "#84cc16", "< 3 d"),
          (7 * 86400.0, "#eab308", "< 7 d"), (14 * 86400.0, "#f97316", "< 14 d"),
          (float("inf"), "#6b7280", "14 d+")]
_GNSS = [(0, "#ef4444", "none"), (1, "#f97316", "3D"), (2, "#f97316", "DGNSS"),
         (3, "#eab308", "RTK float"), (4, "#22c55e", "RTK fix")]
_SLOPE = [(5.0, "#22c55e", "< 5°"), (10.0, "#84cc16", "< 10°"),
          (15.0, "#eab308", "< 15°"), (20.0, "#f97316", "< 20°"),
          (float("inf"), "#ef4444", "20°+")]


def _legend(title, table):
    return {"title": title, "stops": [{"color": c, "label": t} for _v, c, t in table]}


def coverage_layer(grid, site, now):
    legend = _legend("Last mowed", _COVER)
    if grid.h == 0:
        return empty_layer(site, now, grid.res, legend)
    t = grid.a["last_mowed"]
    valid = t > 0
    age = now - t.astype(np.float64)
    idx = np.searchsorted([row[0] for row in _COVER], age, side="right")
    rgba = _paint(idx, valid, [row[1] for row in _COVER])
    return _layer_msg(site, now, grid.ox, grid.oy, grid.res, rgba, valid, legend)


def gnss_layer(grid, site, now):
    legend = _legend("GNSS quality", _GNSS)
    if grid.h == 0:
        return empty_layer(site, now, grid.res, legend)
    cls = grid.class_median()
    rgba = _paint(cls, cls >= 0, [row[1] for row in _GNSS])
    return _layer_msg(site, now, grid.ox, grid.oy, grid.res, rgba, cls >= 0, legend)


def slope_layer(dem_path, site, now, res=0.5):
    legend = _legend("Slope", _SLOPE)
    got = None
    try:
        if os.path.exists(dem_path):
            got = slope_from_dem(dem_path, res)
    except Exception:                                      # noqa: BLE001
        got = None                # dem.npz being replaced or damaged = no data
    if got is None:
        return empty_layer(site, now, res, legend)
    slope, ox, oy, cell = got
    valid = np.isfinite(slope)
    idx = np.searchsorted([row[0] for row in _SLOPE], np.nan_to_num(slope), side="right")
    rgba = _paint(idx, valid, [row[1] for row in _SLOPE])
    return _layer_msg(site, now, ox, oy, cell, rgba, valid, legend)


# --------------------------------------------------------------------------
# site lookup
# --------------------------------------------------------------------------
def read_served_site(root=None):
    """The site mapping_manager serves, from .last_served; None when none is
    (a fresh mapping session drops the file, nothing is known about the frame)."""
    try:
        with open(os.path.join(root or bundle.SITES_ROOT, ".last_served")) as f:
            data = json.load(f)
        return str(data["site"]) if data.get("site") else None
    except (IOError, OSError, ValueError, KeyError, AttributeError, TypeError):
        return None


def read_datum(site):
    """Canonical datum of a site, None when it has none (yet). Files only."""
    try:
        with open(os.path.join(bundle.site_dir(site), "datum.yaml")) as f:
            return geo.from_datum_yaml(yaml.safe_load(f))
    except (IOError, OSError, ValueError, KeyError, TypeError, AttributeError,
            yaml.YAMLError):
        return None


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# --------------------------------------------------------------------------
# reading without ROS
# --------------------------------------------------------------------------
def _poly_mask(ox, oy, res, h, w, ring):
    """[h, w] bool: cell centres inside `ring` (even-odd)."""
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (np.arange(h) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros((h, w), bool)
    for i in range(len(ring)):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]
        if y0 == y1:
            continue
        cross = (y0 > gy) != (y1 > gy)
        inside ^= cross & (gx < (x1 - x0) * (gy - y0) / (y1 - y0) + x0)
    return inside


def _cell(ox, oy, res, h, w, x, y):
    c = int(math.floor((x - ox) / res))
    r = int(math.floor((y - oy) / res))
    return (r, c) if 0 <= r < h and 0 <= c < w else None


def _query_coverage(grid, points, polygon):
    t = grid.a["last_mowed"]
    if points is not None:
        rows = []
        for x, y in points:
            rc = _cell(grid.ox, grid.oy, grid.res, grid.h, grid.w, x, y)
            v = int(t[rc]) if rc is not None else 0
            rows.append({"x": x, "y": y, "last_mowed": v or None})
        return rows
    mask = _poly_mask(grid.ox, grid.oy, grid.res, grid.h, grid.w, polygon)
    cells = int(mask.sum())
    got = t[mask]
    got = got[got > 0]
    return {"cells": cells, "area_m2": round(cells * grid.res ** 2, 2),
            "mowed_frac": round(got.size / float(cells), 4) if cells else None,
            "oldest": int(got.min()) if got.size else None,
            "newest": int(got.max()) if got.size else None}


def _rtcm_age(ds):
    return None if ds == RTCM_NONE else round(ds / 10.0, 1)


def _query_gnss(grid, points, polygon):
    age_ds = grid.a["rtcm_age_ds"]
    median_mm = grid.hacc_median_mm()
    median_cls = grid.class_median()
    best_mm = grid.a["hacc_best_mm"]
    best_cls = grid.a["class_best"]
    if points is not None:
        rows = []
        for x, y in points:
            rc = _cell(grid.ox, grid.oy, grid.res, grid.h, grid.w, x, y)
            if rc is None or median_cls[rc] < 0:
                rows.append({"x": x, "y": y, "hacc_best_mm": None, "hacc_median_mm": None,
                             "class_best": None, "class_median": None,
                             "rtcm_age_max_s": None})
            else:
                rows.append({"x": x, "y": y, "hacc_best_mm": int(best_mm[rc]),
                             "hacc_median_mm": int(median_mm[rc]),
                             "class_best": int(best_cls[rc]),
                             "class_median": int(median_cls[rc]),
                             "rtcm_age_max_s": _rtcm_age(int(age_ds[rc]))})
        return rows
    mask = _poly_mask(grid.ox, grid.oy, grid.res, grid.h, grid.w, polygon)
    seen = mask & (median_cls >= 0)
    n = int(seen.sum())
    ages = age_ds[seen]
    ages = ages[ages != RTCM_NONE]
    return {"cells": int(mask.sum()), "cells_seen": n,
            "rtcm_age_max_s": _rtcm_age(int(ages.max())) if ages.size else None,
            "class_frac": {str(k): round(int((median_cls[seen] == k).sum()) / float(n), 4)
                           for k in range(N_CLASS)} if n else {},
            "hacc_median_mm": int(np.median(median_mm[seen])) if n else None}


def _query_slope(site, res, points, polygon):
    path = os.path.join(bundle.site_dir(site), "dem.npz")
    if not os.path.exists(path):
        return None
    got = slope_from_dem(path, res)
    if got is None:
        return None
    slope, ox, oy, cell = got
    h, w = slope.shape
    if points is not None:
        rows = []
        for x, y in points:
            rc = _cell(ox, oy, cell, h, w, x, y)
            v = float(slope[rc]) if rc is not None else float("nan")
            rows.append({"x": x, "y": y,
                         "slope_deg": round(v, 2) if math.isfinite(v) else None})
        return rows
    mask = _poly_mask(ox, oy, cell, h, w, polygon)
    vals = slope[mask]
    vals = vals[np.isfinite(vals)]
    return {"cells": int(mask.sum()), "cells_seen": int(vals.size),
            "mean_deg": round(float(vals.mean()), 2) if vals.size else None,
            "max_deg": round(float(vals.max()), 2) if vals.size else None}


def layer_query(site, layer, points=None, polygon=None, slope_res=0.5, run=None):
    """Read a recorded layer without ROS. Read-only, never raises.

    layer    coverage | gnss | slope | grass | mow_load | mow_report
    points   [[x, y], ...] map frame -> "result" is a list, one row per point
    polygon  [[x, y], ...] map frame -> "result" is one stats dict

    Returns {"ok", "reason", "site", "layer", "result"}. No file or no data is
    ok:true with empty values: it means "not recorded", never "not mowed".
    last_mowed None = never recorded at that cell. Fix classes as in the module
    docstring.
    mow_load: where the blade worked hard, newest run or `run`, see mowtrail.query;
    without points and polygon the whole run, "heavy" lists the hardest spots,
    "report_path" the zone report of the run when it is written.
    mow_report: the zone report of the run (newest finished or `run`), see
    mowreport; points and polygon are not used.
    grass: grass height measured ahead by nodes/grass_height (median, last,
    count, time, ground source wheels|dem per 0.2 m cell), see grass.query_grass.
    """
    out = {"ok": False, "reason": "", "site": site, "layer": layer, "result": None}
    if layer == "mow_report":
        from . import mowreport          # it imports this module
        rep = mowreport.load(site, run)
        out.update(ok=bool(rep.get("ok")) or rep.get("reason") == "no run",
                   reason="" if rep.get("ok") else str(rep.get("reason") or ""),
                   result=rep if rep.get("ok") else None)
        return out
    if (points is None) == (polygon is None) and not (layer == "mow_load" and points is None):
        out["reason"] = "give points or polygon"
        return out
    if polygon is not None and len(polygon) < 3:
        out["reason"] = "polygon needs 3 points"
        return out
    try:
        pts = None if points is None else [(float(p[0]), float(p[1])) for p in points]
        ring = None if polygon is None else [(float(p[0]), float(p[1])) for p in polygon]
        if layer == "slope":
            out["result"] = _query_slope(site, slope_res, pts, ring)
        elif layer == "mow_load":
            from . import mowtrail           # it imports this module
            out["result"] = mowtrail.query(site, pts, ring, run)
            res = out["result"]
            if isinstance(res, dict) and res.get("run"):
                from . import mowreport
                path = mowreport.report_path(site, res["run"])
                res["report_path"] = path if os.path.exists(path) else None
        elif layer == "grass":
            from . import grass              # it imports this module
            datum = read_datum(site)
            if datum is None:
                out["reason"] = "no datum for site %s" % site
                return out
            grid = grass.GrassGrid.load(bundle.site_dir(site), datum, readonly=True)
            if grid.h:
                out["result"] = grass.query_grass(grid, pts, ring)
        elif layer in ("coverage", "gnss"):
            datum = read_datum(site)
            if datum is None:
                out["reason"] = "no datum for site %s" % site
                return out
            cls = CoverageGrid if layer == "coverage" else GnssGrid
            grid = cls.load(bundle.site_dir(site), datum, readonly=True)
            if grid.h:
                query = _query_coverage if layer == "coverage" else _query_gnss
                out["result"] = query(grid, pts, ring)
        else:
            out["reason"] = "unknown layer %s" % layer
            return out
    except Exception as exc:                               # noqa: BLE001
        out["reason"] = str(exc)
        return out
    out["ok"] = True
    return out


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
class WorldRecorder(object):
    """All recorder logic. The node calls poll/step/maintain/close.

    publish(layer, payload)  layer in coverage|gnss|slope
    log(text)                one line about something that went wrong
    Times are unix seconds passed in, so a test can run a day in a moment.
    """

    def __init__(self, publish=None, log=None, cfg=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.publish = publish or (lambda _layer, _payload: None)
        self.log = log or (lambda _text: None)
        self.site = None
        self.datum = None
        self.coverage = None
        self.gnss = None
        self._datum_mtime = 0.0
        self._dem_mtime = 0.0
        self._next_poll = 0.0
        self._last_flush = 0.0
        self._published = {}          # layer -> last publish time
        self._pending = set()         # layers with something to publish
        self._prev = None             # last deck centre, for the sweep
        self._gnss_at = None          # last GNSS sample position
        self._blade = False

    # ---- site ---------------------------------------------------------------
    def poll(self, now):
        """Cheap watch of the served site, its datum and its DEM (stat calls)."""
        if now < self._next_poll:
            return
        self._next_poll = now + self.cfg["site_poll_s"]
        site = read_served_site()
        datum_mtime = _mtime(os.path.join(bundle.site_dir(site), "datum.yaml")) if site else 0.0
        if site != self.site or datum_mtime != self._datum_mtime:
            self._switch(site, datum_mtime, now)
        elif site:
            dem_mtime = _mtime(os.path.join(bundle.site_dir(site), "dem.npz"))
            if dem_mtime != self._dem_mtime:
                self._dem_mtime = dem_mtime
                self._pending.add("slope")

    def _switch(self, site, datum_mtime, now):
        self.flush()
        had = self.site
        self.site = self.datum = self.coverage = self.gnss = None
        self._prev = self._gnss_at = None
        self._pending.clear()
        self._datum_mtime = datum_mtime
        datum = read_datum(site) if site else None
        if datum is None:
            if had:
                # the UI must not keep drawing the old site's layers
                for layer, legend in (("coverage", _legend("Last mowed", _COVER)),
                                      ("gnss", _legend("GNSS quality", _GNSS)),
                                      ("slope", _legend("Slope", _SLOPE))):
                    self.publish(layer, empty_layer(site, now, 0.0, legend))
            return
        self.site, self.datum = site, datum
        d = bundle.site_dir(site)
        self.coverage = CoverageGrid.load(d, datum)
        self.gnss = GnssGrid.load(d, datum)
        self._dem_mtime = _mtime(os.path.join(d, "dem.npz"))
        self._published.clear()
        self._pending.update(("coverage", "gnss", "slope"))

    # ---- recording ----------------------------------------------------------
    def step(self, now, pose, blade_on, gnss=None, rtcm_age_s=None):
        """One look at the robot. pose = (x, y, yaw) in map or None, blade_on =
        real rpm > 0, gnss = (h_acc_mm, fix_class, ...) or None, rtcm_age_s =
        seconds since the last RTCM message or None."""
        if self._blade and not blade_on:
            self._prev = None
            self.flush()                 # blade stopped: program end or pause
        self._blade = bool(blade_on)
        if self.site is None or pose is None:
            self._prev = None
            return
        x, y, yaw = pose
        off = self.cfg["deck_offset_x_m"]
        cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
        if blade_on:
            prev = self._prev
            if prev is None or math.hypot(cx - prev[0], cy - prev[1]) > self.cfg["max_step_m"]:
                prev = (cx, cy)
            if self.coverage.stamp(prev[0], prev[1], cx, cy,
                                   self.cfg["deck_diameter_m"] / 2.0, now):
                self._pending.add("coverage")
            else:
                self.log("coverage: pose (%.1f, %.1f) outside the grid limit" % (cx, cy))
            self._prev = (cx, cy)
        if gnss is not None:
            at = self._gnss_at
            if at is None or math.hypot(x - at[0], y - at[1]) >= self.cfg["gnss_step_m"]:
                if self.gnss.add(x, y, gnss[0], gnss[1], rtcm_age_s):
                    self._pending.add("gnss")
                self._gnss_at = (x, y)

    # ---- output -------------------------------------------------------------
    def flush(self):
        """Write what changed. A failed write stays dirty for the next try."""
        if self.site is None:
            return
        d = bundle.site_dir(self.site)
        for grid in (self.coverage, self.gnss):
            if grid is not None and grid.dirty:
                try:
                    grid.save(d, self.site, self.datum)
                except (IOError, OSError, ValueError) as exc:
                    self.log("%s: %s" % (grid.FILENAME, exc))

    def _build(self, layer, now):
        if layer == "coverage":
            return coverage_layer(self.coverage, self.site, now)
        if layer == "gnss":
            return gnss_layer(self.gnss, self.site, now)
        return slope_layer(os.path.join(bundle.site_dir(self.site), "dem.npz"),
                           self.site, now, self.cfg["slope_res_m"])

    def maintain(self, now):
        """Timed flush and layer publishing. Does nothing when nothing changed."""
        if self.site is None:
            return
        if now - self._last_flush >= self.cfg["flush_s"]:
            self._last_flush = now
            self.flush()
        last = self._published.get("coverage")
        if (last is not None and now - last >= self.cfg["layer_refresh_s"]
                and self.coverage.h):
            self._pending.add("coverage")
        for layer in list(self._pending):
            last = self._published.get(layer)
            if last is not None and now - last < self.cfg["layer_min_s"]:
                continue
            self._pending.discard(layer)
            self._published[layer] = now
            self.publish(layer, self._build(layer, now))

    def close(self):
        self.flush()
