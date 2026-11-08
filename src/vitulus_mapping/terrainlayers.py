#!/usr/bin/env python3
"""
terrainlayers.py -- the persistent terrain map of a whole site (T30 D1 core).

NO rospy. The future node `terrain_recorder` feeds it the local terrain window of
`traversability_local` (vitulus_traversability) once a second; everything here
runs with plain numpy arrays, so all of it is tested without a ROS master.
Design: ~/vitulus/predavka/t30-globalni-teren-navrh.md (sections 3 and 4).

Grid
    UTM grid of the site, axis aligned to UTM, `RES` 0.05 m. Global cell
    (gx, gy) = (floor(e / RES), floor(n / RES)); its centre ((gx + .5) RES,
    (gy + .5) RES). Tiles of `TILE` x `TILE` cells (12.8 m): tile (tx, ty) =
    (gx >> 8, gy >> 8), arrays [ly, lx] with row 0 = south, column 0 = west.
    Stored in UTM, so a new site datum never touches the data: only the
    transform map <-> UTM changes (datum: p_utm = R(yaw) p_map + (utm_e, utm_n),
    the convention of geo.py).

Window (input, one per batch)
    The rolling window of the local node in its frame (odom): min corner
    (x0, y0) of cell [0, 0], resolution, arrays [iy, ix] (row 0 = lowest y):
    elevation [m], variance [m^2, right after its last fusion], points, t_fused
    [unix s of the last fused point] (optional), cost (0..100, -1 unknown),
    reason (codes of t10-ui-kontrakt: 0..7), slope [rad], step [m], roughness [m];
    epoch (localisation epoch of the local node: +1 at set_pose / a heading
    re-anchoring) and settling (a correction is being pulled in). With the pose
    map <- odom (x, y, yaw) at the stamp (the map frame is flat: no z from TF)
    and base_link (x, y, z, yaw) in the window frame, all sent by the local node.

Localisation (RTK corrections go mostly into odom, not into map <- odom)
    A batch is fused only `accept_delay` s after its stamp; a settling batch
    drops itself and the queue, a new epoch drops the queue, closes the visits
    and finds the offset again. A jump of map <- odom refuses the batch too.
    Every global cell is sampled at its centre in the window (global -> window:
    no holes at any yaw); `edge_margin` of the window border is left out.

Visit
    Successive windows are NOT independent measurements (the local Kalman keeps
    the same points), so a global cell holds an OPEN VISIT that every batch
    OVERWRITES with the newest local state of its cell. When the cell leaves the
    window, or gets no fresh point for `visit_timeout` s, the visit is closed and
    fused into the long-term estimate as ONE measurement. Evidence counters
    count visits (not the per-cloud "runs" of evaluate_bag.py / the mosaic).

Height alignment (z of the odometry drifts and jumps; datum alt and the DEM are
no reference)
    Every batch is shifted vertically by a robust (Huber) mean of window height -
    reference height over the overlap (reference = long-term height where it is
    certain, else the open visit of an earlier batch). The first batch of a
    session / epoch starts from the mode of the residual histogram: any size of
    offset, robust to a wrong part of the overlap. An offset change over
    `align_jump` (an odometry z jump) closes the open visits; there is no limit
    on its size. Without overlap: the offset of the last batch; at a start the
    stored height around the robot minus base_link z (the dock); else an
    optional anchor (trajectory DEM at the robot - base_link z); else 0. The
    height reference is fixed: z_ref in index.json. With overlap the anchor only
    measures the bias of the map (`anchor_residual`, for a warning); pulling the
    stored map to the DEM (only on freshly mown cells) is a later step.
    Slope / step / cost come from the window (one consistent local map), never
    from the global heights, so they do not depend on this offset.

Change over time (long-term estimate per cell)
    Variance grows by `q_day` per day, never below `sigma_min`. A visit within
    the gate (k sigma, with a floor for the grass canopy and alignment) is
    Kalman fused. When the cell was mowed after its last visit (optional
    callable `mowed`, coverage.npz) the visit replaces the height. Outside the
    gate it is a candidate that replaces the height when `change_visits` visits
    agree with it. Obstacles (reason 5) count only from the last visit. A
    terrain lethal cell (step 3 / slope 4) is CONFIRMED after >= `confirm_visits`
    lethal visits, >= `confirm_share` of its visits, in >= `confirm_sessions`
    sessions and from headings >= 45 deg apart (`confirm_dirs`); `clear_visits`
    non-lethal visits in a row clear the evidence. Until then it is tentative
    (cost 99): the map never bans anything by itself.

Files (terrain dir, e.g. ~/.vitulus/mapping_v3/<site>/terrain/)
    index.json        format_version, res, tile, z_ref, sessions, tiles {"tx_ty": {...}}
    t_<tx>_<ty>.npz   compact arrays (TILE_DTYPES), zlib, written atomically
    broken/           tiles that could not be read (moved aside, start empty)
    Heights are int16 mm relative to z_ref (set at the first committed cell).
"""

import collections
import json
import math
import os
import tempfile
import time

import numpy as np

__all__ = ["RES", "TILE", "FORMAT_VERSION", "DEFAULTS", "TILE_DTYPES", "Window",
           "Tile", "TerrainStore", "huber_mean", "hist_mode", "dir_spread", "pack_keys", "unpack_keys",
           "REASON_STEP", "REASON_SLOPE", "REASON_OBSTACLE", "REASON_EXEMPT",
           "F_CONFIRMED", "F_OBST_LAST", "F_EXEMPT", "F_CAND", "F_TENTATIVE"]

FORMAT_VERSION = 1
RES = 0.05
TILE = 256
TILE_SHIFT = 8
TILE_M = RES * TILE

REASON_UNKNOWN = 0
REASON_STEP = 3
REASON_SLOPE = 4
REASON_OBSTACLE = 5
REASON_EXEMPT = 7

F_CONFIRMED = 0x01      # terrain lethal confirmed over visits
F_OBST_LAST = 0x02      # obstacle in the last visit
F_EXEMPT = 0x04         # last visit inside an exempt zone (dock)
F_CAND = 0x08           # a height candidate is pending
F_TENTATIVE = 0x10      # imported / seen too little (set by importers)

H_NONE = -32768
SIG_NONE = 0xFFFF
U8_NONE = 0xFF

DEFAULTS = {
    # selection of window cells
    "max_variance": 2.0e-4,      # = traversability feature_max_variance (sigma 14 mm)
    "min_points": 3,             # = step_min_points
    # visits
    "visit_timeout": 30.0,       # s without a fresh point -> the visit is closed
    # alignment
    "align_min_cells": 200,
    "align_jump": 0.05,          # m: offset change between batches beyond this closes the open visits
                                 # (an odometry z jump: no visit mixes two offsets); no limit on the size
    "align_hist_bin": 0.02,      # m: first offset of a session = mode of the residual histogram, then Huber
    "robot_ground_radius": 0.30, # m: no overlap at the start -> stored height around the robot (the dock)
    "align_huber": 0.02,         # m
    "ref_max_variance": 2.5e-3,  # long-term reference only when this certain (sigma 5 cm; q_day ages
                                 # a fresh cell there in ~3 months)
    "anchor_ema": 0.05,          # averaging of anchor_residual (per batch with overlap and anchor)
    "align_sigma": 0.01,         # m added to a visit's variance (alignment / RTK)
    # pose
    "jump_max_m": 0.10,          # map <- odom change between batches beyond = batch refused, visits closed
    "jump_max_deg": 2.0,
    "accept_delay": 2.0,         # s: a batch is fused only this late, and dropped when a localisation
                                 # correction (settling / new epoch) comes before that
    "edge_margin": 0.25,         # m of the window border left out (half seen, registration edge)
    # long-term estimate
    "q_day": 2.5e-5,             # m^2 per day ((5 mm)^2)
    "gate_k": 3.0,
    "gate_floor": 0.02,          # m: grass canopy between visits
    "sigma_min": 0.005,          # m: the long-term sigma never shrinks below (gate stays open to change)
    "change_visits": 2,
    "confirm_visits": 3,
    "confirm_share": 0.5,
    "confirm_sessions": 2,       # lethal in at least this many sessions (store starts / new_session)
    "confirm_dirs": 2,           # ... seen lethal from headings >= 45 deg apart (8 sectors)
    "clear_visits": 3,
    "obstacle_clear_points": 5,
    # memory
    "max_loaded_tiles": 16,      # x 2.56 MB (39 B per cell, measured): ~41 MB; a window touches <= 4
}

# per-cell arrays of a tile on disk and in RAM (RAM: h, var float32 instead of h_mm, sig_mm)
TILE_DTYPES = collections.OrderedDict([
    ("h_mm", np.int16),        # height mm relative to z_ref, H_NONE unknown
    ("sig_mm", np.uint16),     # sigma mm at t_last, SIG_NONE unknown
    ("t_last", np.uint32),     # unix s of the last committed visit, 0 never
    ("t_change", np.uint32),   # unix s of the last accepted height change
    ("visits", np.uint8),
    ("l_step", np.uint8),      # visits with reason step
    ("l_slope", np.uint8),     # visits with reason slope
    ("streak", np.uint8),      # non-lethal visits in a row
    ("l_sess", np.uint8),      # sessions with a lethal visit
    ("l_last", np.uint16),     # session number of the last lethal visit (mod 65536)
    ("l_dirs", np.uint8),      # heading sectors (8 x 45 deg, UTM) of the lethal visits, bit mask
    ("cost", np.uint8),        # cost of the last visit, U8_NONE unknown
    ("reason", np.uint8),      # reason of the last visit
    ("slope", np.uint8),       # 0.5 deg units of the last visit, U8_NONE unknown
    ("step", np.uint8),        # cm, U8_NONE unknown
    ("rough", np.uint8),       # mm, U8_NONE unknown
    ("flags", np.uint8),
    ("cand_mm", np.int16),     # candidate height, H_NONE none
    ("cand_n", np.uint8),
    ("cand_t", np.uint32),
])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pack_keys(gx, gy):
    """int64 key of global cells (UTM cells are non-negative)."""
    gx = np.asarray(gx, np.int64)
    gy = np.asarray(gy, np.int64)
    return (gx << 32) | gy


def unpack_keys(keys):
    keys = np.asarray(keys, np.int64)
    return keys >> 32, keys & 0xFFFFFFFF


def huber_mean(r, delta, iters=8, start=None):
    """Robust mean of residuals r (Huber weights, start at `start` or the median)."""
    r = np.asarray(r, np.float64)
    if r.size == 0:
        return float("nan")
    m = float(np.median(r)) if start is None else float(start)
    for _ in range(iters):
        a = np.abs(r - m)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
        m_new = float(np.sum(w * r) / np.sum(w))
        if abs(m_new - m) < 1e-5:
            return m_new
        m = m_new
    return m


def hist_mode(r, bin_w, max_bins=20000):
    """Centre of the fullest bin of r (smoothed over 3 bins): a start for the
    offset that a minority of wrong residuals cannot move."""
    r = np.asarray(r, np.float64)
    lo, hi = float(r.min()), float(r.max())
    nb = int((hi - lo) / bin_w) + 1
    if nb <= 1:
        return float(np.median(r))
    if nb > max_bins:
        return float(np.median(r))
    cnt, edges = np.histogram(r, bins=nb, range=(lo, lo + nb * bin_w))
    sm = np.convolve(cnt, [1, 1, 1], mode="same")
    i = int(np.argmax(sm))
    sel = (r >= edges[max(i - 1, 0)]) & (r <= edges[min(i + 2, nb)])
    return float(np.median(r[sel]))


def _atomic_write(path, writer):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Window(object):
    """One local terrain window (see module doc). Arrays [iy, ix]."""

    def __init__(self, x0, y0, res, elevation, variance, points, cost, reason,
                 slope=None, step=None, roughness=None, t_fused=None, stamp=None, epoch=0,
                 settling=False):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.res = float(res)
        self.elevation = np.asarray(elevation, np.float32)
        h, w = self.elevation.shape
        nanf = np.full((h, w), np.nan, np.float32)
        self.variance = np.asarray(variance, np.float32)
        self.points = np.asarray(points, np.float32)
        self.cost = np.asarray(cost, np.float32)
        self.reason = np.asarray(reason, np.float32)
        self.slope = nanf if slope is None else np.asarray(slope, np.float32)
        self.step = nanf if step is None else np.asarray(step, np.float32)
        self.roughness = nanf if roughness is None else np.asarray(roughness, np.float32)
        self.t_fused = None if t_fused is None else np.asarray(t_fused, np.float64)
        self.stamp = stamp
        self.epoch = int(epoch)          # localisation epoch of the local node (set_pose / heading re-anchor)
        self.settling = bool(settling)   # a localisation correction is being pulled in right now
        for a in (self.variance, self.points, self.cost, self.reason, self.slope,
                  self.step, self.roughness):
            if a.shape != (h, w):
                raise ValueError("window layers differ in shape")

    @property
    def shape(self):
        return self.elevation.shape


# --------------------------------------------------------------------------
# tile
# --------------------------------------------------------------------------
class Tile(object):
    """TILE x TILE cells of the long-term estimate."""

    def __init__(self, tx, ty):
        self.tx = int(tx)
        self.ty = int(ty)
        n = (TILE, TILE)
        self.h = np.full(n, np.nan, np.float32)
        self.var = np.full(n, np.nan, np.float32)
        self.t_last = np.zeros(n, np.uint32)
        self.t_change = np.zeros(n, np.uint32)
        self.visits = np.zeros(n, np.uint8)
        self.l_step = np.zeros(n, np.uint8)
        self.l_slope = np.zeros(n, np.uint8)
        self.streak = np.zeros(n, np.uint8)
        self.l_sess = np.zeros(n, np.uint8)
        self.l_last = np.zeros(n, np.uint16)
        self.l_dirs = np.zeros(n, np.uint8)
        self.cost = np.full(n, U8_NONE, np.uint8)
        self.reason = np.zeros(n, np.uint8)
        self.slope = np.full(n, U8_NONE, np.uint8)
        self.step = np.full(n, U8_NONE, np.uint8)
        self.rough = np.full(n, U8_NONE, np.uint8)
        self.flags = np.zeros(n, np.uint8)
        self.cand_h = np.full(n, np.nan, np.float32)
        self.cand_n = np.zeros(n, np.uint8)
        self.cand_t = np.zeros(n, np.uint32)
        self.version = 0
        self.dirty = False

    @property
    def key(self):
        return (self.tx, self.ty)

    @property
    def name(self):
        return "t_%d_%d" % (self.tx, self.ty)

    def known(self):
        return int(np.count_nonzero(np.isfinite(self.h)))

    def ram_bytes(self):
        return sum(getattr(self, a).nbytes for a in
                   ("h", "var", "t_last", "t_change", "visits", "l_step", "l_slope", "streak",
                    "l_sess", "l_last", "l_dirs",
                    "cost", "reason", "slope", "step", "rough", "flags", "cand_h", "cand_n",
                    "cand_t"))

    def var_now(self, ly, lx, t, q_day):
        age = np.maximum(0.0, t - self.t_last[ly, lx].astype(np.float64))
        return self.var[ly, lx].astype(np.float64) + q_day * age / 86400.0

    # ---- commit closed visits ------------------------------------------------
    def commit(self, ly, lx, v, p, mowed_t=None):
        """Fuse closed visits into the long-term estimate.

        ly, lx: unique cell indices in this tile; v: dict of visit arrays (h, var,
        t, cost, reason, slope, step, rough, points); p: parameters; mowed_t:
        last-mowed unix s per cell (0 never) or None."""
        if ly.size == 0:
            return 0
        t = v["t"]
        tu = np.round(t).astype(np.uint32)
        h_v = v["h"].astype(np.float64)
        var_v = v["var"].astype(np.float64)
        h_old = self.h[ly, lx].astype(np.float64)
        new = ~np.isfinite(h_old)
        vn = self.var_now(ly, lx, t, p["q_day"])
        d = h_v - h_old
        gate = p["gate_k"] * np.sqrt(np.where(new, 0.0, vn) + var_v + p["gate_floor"] ** 2)
        mowed = np.zeros(ly.size, bool)
        if mowed_t is not None:      # mowed since the last visit: the new visit replaces the height
            mowed = ~new & (np.asarray(mowed_t, np.float64) > self.t_last[ly, lx])
        agree = ~new & ~mowed & (np.abs(d) <= gate)
        differ = ~new & ~mowed & ~agree
        # candidate
        cand_old = self.cand_h[ly, lx].astype(np.float64)
        has_cand = np.isfinite(cand_old) & (self.cand_n[ly, lx] > 0)
        cand_gate = p["gate_k"] * np.sqrt(2.0 * var_v + p["gate_floor"] ** 2)
        cand_ok = differ & ~mowed & has_cand & (np.abs(h_v - cand_old) <= cand_gate)
        cand_n_new = np.where(cand_ok, self.cand_n[ly, lx].astype(np.int32) + 1, 1)
        replace_cand = cand_ok & (cand_n_new >= p["change_visits"])
        start_cand = differ & ~mowed & ~replace_cand
        if p["change_visits"] <= 1:
            replace_cand = differ & ~mowed
            start_cand[:] = False
        cand_val = np.where(cand_ok, (cand_old * (cand_n_new - 1) + h_v) / cand_n_new, h_v)

        h_new = h_old.copy()
        var_new = self.var[ly, lx].astype(np.float64)
        # new cells / replacements
        put = new | mowed
        h_new[put] = h_v[put]
        var_new[put] = var_v[put]
        h_new[replace_cand] = cand_val[replace_cand]
        var_new[replace_cand] = var_v[replace_cand]
        # Kalman on agreement
        if np.any(agree):
            k = vn[agree] / (vn[agree] + var_v[agree])
            h_new[agree] = h_old[agree] + k * d[agree]
            var_new[agree] = (1.0 - k) * vn[agree]
        # differing, not replaced: height stays, variance aged
        keep = start_cand | (cand_ok & ~replace_cand)
        var_new[keep] = vn[keep]
        self.h[ly, lx] = h_new
        self.var[ly, lx] = np.maximum(var_new, p["sigma_min"] ** 2)
        changed = new | mowed | replace_cand
        tc = self.t_change[ly, lx]
        self.t_change[ly, lx] = np.where(changed, tu, tc)
        # candidate bookkeeping
        ch = self.cand_h[ly, lx]
        cn = self.cand_n[ly, lx]
        ct = self.cand_t[ly, lx]
        clear = agree | changed
        ch = np.where(clear, np.nan, ch)
        cn = np.where(clear, 0, cn)
        ch = np.where(keep, cand_val, ch)
        cn = np.where(keep, np.minimum(cand_n_new, 255), cn)
        ct = np.where(keep, tu, ct)
        self.cand_h[ly, lx] = ch
        self.cand_n[ly, lx] = cn
        self.cand_t[ly, lx] = ct
        self.t_last[ly, lx] = tu

        # features of the visit and evidence
        reason = np.asarray(v["reason"]).astype(np.int32)
        visits = self.visits[ly, lx].astype(np.int32) + 1
        l_step = self.l_step[ly, lx].astype(np.int32) + (reason == REASON_STEP)
        l_slope = self.l_slope[ly, lx].astype(np.int32) + (reason == REASON_SLOPE)
        terr = (reason == REASON_STEP) | (reason == REASON_SLOPE)
        streak = np.where(terr, 0, self.streak[ly, lx].astype(np.int32) + 1)
        session = np.asarray(v["session"]).astype(np.int32) & 0xFFFF
        l_sess = self.l_sess[ly, lx].astype(np.int32)
        l_last = self.l_last[ly, lx].astype(np.int32)
        new_sess = terr & ((l_sess == 0) | (l_last != session))
        l_sess = np.minimum(l_sess + new_sess, 255)
        l_last = np.where(terr, session, l_last)
        sector = np.asarray(v["sector"]).astype(np.int32) & 7
        l_dirs = self.l_dirs[ly, lx].astype(np.int32) | np.where(terr, 1 << sector, 0)
        cleared = streak >= p["clear_visits"]
        l_step = np.where(cleared, 0, l_step)
        l_slope = np.where(cleared, 0, l_slope)
        l_sess = np.where(cleared, 0, l_sess)
        l_dirs = np.where(cleared, 0, l_dirs)
        sat = visits >= 255
        visits = np.where(sat, visits // 2, visits)
        l_step = np.where(sat, l_step // 2, l_step)
        l_slope = np.where(sat, l_slope // 2, l_slope)
        streak = np.minimum(streak, 255)
        lethal = l_step + l_slope
        confirmed = (lethal >= p["confirm_visits"]) & (lethal >= p["confirm_share"] * visits) & \
            (l_sess >= p["confirm_sessions"]) & (dir_spread(l_dirs) >= p["confirm_dirs"])
        flags = self.flags[ly, lx].astype(np.int32)
        flags &= ~(F_CONFIRMED | F_EXEMPT | F_CAND)
        flags |= np.where(confirmed, F_CONFIRMED, 0)
        flags |= np.where(reason == REASON_EXEMPT, F_EXEMPT, 0)
        flags |= np.where(np.isfinite(self.cand_h[ly, lx]), F_CAND, 0)
        # obstacle: from the last visit; cleared only by a visit with enough points
        obst = reason == REASON_OBSTACLE
        enough = np.asarray(v["points"], np.float64) >= p["obstacle_clear_points"]
        was = (flags & F_OBST_LAST) != 0
        obst_flag = obst | (was & ~enough)
        flags = np.where(obst_flag, flags | F_OBST_LAST, flags & ~F_OBST_LAST)
        self.visits[ly, lx] = visits
        self.l_step[ly, lx] = l_step
        self.l_slope[ly, lx] = l_slope
        self.streak[ly, lx] = streak
        self.l_sess[ly, lx] = l_sess
        self.l_last[ly, lx] = l_last
        self.l_dirs[ly, lx] = l_dirs
        self.flags[ly, lx] = flags
        self.reason[ly, lx] = np.clip(reason, 0, 254)
        cost = np.asarray(v["cost"], np.float64)
        self.cost[ly, lx] = np.where(np.isfinite(cost) & (cost >= 0),
                                     np.clip(np.round(cost), 0, 100), U8_NONE)
        self.slope[ly, lx] = _q8(v["slope"], 2.0 * 180.0 / math.pi)
        self.step[ly, lx] = _q8(v["step"], 100.0)
        self.rough[ly, lx] = _q8(v["rough"], 1000.0)
        self.version += 1
        self.dirty = True
        return int(ly.size)

    # ---- derived ---------------------------------------------------------------
    def effective(self):
        """(reason, cost, tentative) for display / queries.

        Obstacle of the last visit -> 5 / 100; confirmed terrain lethal -> 3 or 4 /
        100; terrain lethal in the last visit but not confirmed -> its code / 99,
        tentative; else the last visit."""
        reason = self.reason.copy()
        cost = self.cost.copy()
        conf = (self.flags & F_CONFIRMED) != 0
        obst = (self.flags & F_OBST_LAST) != 0
        last_terr = (self.reason == REASON_STEP) | (self.reason == REASON_SLOPE)
        tent = last_terr & ~conf
        cost[tent] = 99
        reason[conf] = np.where(self.l_step[conf] >= self.l_slope[conf], REASON_STEP, REASON_SLOPE)
        cost[conf] = 100
        reason[obst] = REASON_OBSTACLE
        cost[obst] = 100
        tent |= (self.flags & F_TENTATIVE) != 0
        return reason, cost, tent & ~obst

    # ---- disk ------------------------------------------------------------------
    def to_arrays(self, z_ref):
        fin = np.isfinite(self.h)
        h_mm = np.full(self.h.shape, H_NONE, np.int16)
        h_mm[fin] = np.clip(np.round((self.h[fin] - z_ref) * 1000.0), -32767, 32767)
        sig = np.full(self.h.shape, SIG_NONE, np.uint16)
        vf = fin & np.isfinite(self.var)
        sig[vf] = np.clip(np.round(np.sqrt(np.maximum(self.var[vf], 0.0)) * 1000.0), 0, 65534)
        cf = np.isfinite(self.cand_h)
        cand = np.full(self.h.shape, H_NONE, np.int16)
        cand[cf] = np.clip(np.round((self.cand_h[cf] - z_ref) * 1000.0), -32767, 32767)
        out = {"h_mm": h_mm, "sig_mm": sig, "cand_mm": cand}
        for k in TILE_DTYPES:
            if k not in out:
                out[k] = getattr(self, k).astype(TILE_DTYPES[k])
        return out

    @classmethod
    def from_arrays(cls, tx, ty, arrs, z_ref):
        t = cls(tx, ty)
        for k, dt in TILE_DTYPES.items():
            a = arrs[k]
            if a.shape != (TILE, TILE) or a.dtype != dt:
                raise ValueError("tile layer %s: %s %s" % (k, a.shape, a.dtype))
        h_mm = arrs["h_mm"]
        fin = h_mm != H_NONE
        t.h[fin] = z_ref + h_mm[fin].astype(np.float64) / 1000.0
        sig = arrs["sig_mm"]
        sf = fin & (sig != SIG_NONE)
        t.var[sf] = (sig[sf].astype(np.float64) / 1000.0) ** 2
        cand = arrs["cand_mm"]
        cf = cand != H_NONE
        t.cand_h[cf] = z_ref + cand[cf].astype(np.float64) / 1000.0
        for k in TILE_DTYPES:
            if k not in ("h_mm", "sig_mm", "cand_mm"):
                setattr(t, k, np.array(arrs[k], dtype=TILE_DTYPES[k]))
        return t


_SPREAD = None


def dir_spread(mask):
    """Number of heading sectors (bits of an 8-sector mask) in a set where every two
    chosen sectors are >= 2 apart (>= 45 deg): 0, 1, 2 (enough for confirm_dirs 2)."""
    global _SPREAD
    if _SPREAD is None:
        tab = np.zeros(256, np.int32)
        for m in range(256):
            bits = [b for b in range(8) if m >> b & 1]
            best = min(len(bits), 1)
            for i in bits:
                for j in bits:
                    dd = min((i - j) % 8, (j - i) % 8)
                    if dd >= 2:
                        best = 2
            tab[m] = best
        _SPREAD = tab
    return _SPREAD[np.asarray(mask, np.int64) & 0xFF]


def _q8(x, scale):
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, U8_NONE, np.uint8)
    f = np.isfinite(x)
    out[f] = np.clip(np.round(np.abs(x[f]) * scale), 0, 254)
    return out


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
_VISIT_FIELDS = ("h", "var", "t", "cost", "reason", "slope", "step", "rough", "points", "sector",
                 "session")


class TerrainStore(object):
    """Tiles of one site + open visits + alignment state."""

    def __init__(self, root, datum, params=None, anchor=None, mowed=None):
        """root: terrain dir (created on save); datum: dict utm_e, utm_n, yaw_rad
        (geo.from_datum_yaml); anchor(e, n) -> ground z (map frame) or NaN, optional;
        mowed(e, n) -> last-mowed unix s (0 never), optional."""
        self.root = root
        self.p = dict(DEFAULTS)
        if params:
            unknown = set(params) - set(DEFAULTS)
            if unknown:
                raise ValueError("unknown parameters: %s" % sorted(unknown))
            self.p.update(params)
        self.datum = dict(datum)
        self.anchor = anchor
        self.mowed = mowed
        self.tiles = collections.OrderedDict()
        self.index = {"format_version": FORMAT_VERSION, "res": RES, "tile": TILE,
                      "z_ref": None, "tiles": {}}
        self.z_offset = None
        self.anchor_residual = None
        self.last_stamp = None
        self.last_pose = None
        self.ov_keys = np.zeros(0, np.int64)
        self.ov = {k: np.zeros(0, np.float64) for k in _VISIT_FIELDS}
        self.stats = collections.Counter()
        self.broken = []
        self.last_batch = {}
        self.epoch = None
        self.pending = []
        self._load_index()
        self.session = int(self.index.get("sessions", 0)) + 1
        self.index["sessions"] = self.session

    # ---- index / tiles ---------------------------------------------------------
    def _index_path(self):
        return os.path.join(self.root, "index.json")

    def _tile_path(self, tx, ty):
        return os.path.join(self.root, "t_%d_%d.npz" % (tx, ty))

    def _load_index(self):
        path = self._index_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                idx = json.load(f)
            if idx.get("format_version") != FORMAT_VERSION or idx.get("tile") != TILE \
                    or abs(float(idx.get("res", 0)) - RES) > 1e-9:
                raise ValueError("format")
            idx.setdefault("tiles", {})
            self.index = idx
        except Exception as e:                       # noqa: BLE001
            self._move_broken(path, "index: %s" % e)

    def _move_broken(self, path, why):
        bdir = os.path.join(self.root, "broken")
        os.makedirs(bdir, exist_ok=True)
        dst = os.path.join(bdir, "%s.%d" % (os.path.basename(path), int(time.time() * 1000)))
        try:
            os.replace(path, dst)
        except OSError:
            pass
        self.broken.append({"file": os.path.basename(path), "why": str(why)})
        self.stats["broken"] += 1

    def tile(self, tx, ty, create=False):
        key = (int(tx), int(ty))
        t = self.tiles.get(key)
        if t is not None:
            self.tiles.move_to_end(key)
            return t
        path = self._tile_path(*key)
        if os.path.exists(path):
            try:
                z_ref = self.index.get("z_ref")
                if z_ref is None:
                    raise ValueError("no z_ref in the index")
                with np.load(path, allow_pickle=False) as z:
                    if int(z["format_version"]) != FORMAT_VERSION:
                        raise ValueError("format_version")
                    t = Tile.from_arrays(key[0], key[1], {k: z[k] for k in TILE_DTYPES},
                                         float(z_ref))
                    t.version = int(z["version"])
                self.stats["tiles_loaded"] += 1
            except Exception as e:                   # noqa: BLE001
                self._move_broken(path, e)
                t = None
        if t is None:
            if not create:
                return None
            t = Tile(*key)
            meta = self.index["tiles"].get("%d_%d" % key)
            if meta:
                t.version = int(meta.get("version", 0))
        self.tiles[key] = t
        return t

    def save_tile(self, t):
        if self.index.get("z_ref") is None:
            fin = np.isfinite(t.h)
            self.index["z_ref"] = float(np.round(np.median(t.h[fin]))) if np.any(fin) else 0.0
        os.makedirs(self.root, exist_ok=True)
        arrs = t.to_arrays(self.index["z_ref"])
        arrs["format_version"] = np.array(FORMAT_VERSION, np.int32)
        arrs["version"] = np.array(t.version, np.int64)
        arrs["tx"] = np.array(t.tx, np.int64)
        arrs["ty"] = np.array(t.ty, np.int64)
        _atomic_write(self._tile_path(t.tx, t.ty), lambda f: np.savez_compressed(f, **arrs))
        known = t.known()
        tl = int(t.t_last.max()) if known else 0
        self.index["tiles"]["%d_%d" % t.key] = {"version": t.version, "t_last": tl, "known": known}
        t.dirty = False
        self.stats["tiles_saved"] += 1

    def save_index(self):
        os.makedirs(self.root, exist_ok=True)
        data = json.dumps(self.index, indent=1, sort_keys=True).encode()
        _atomic_write(self._index_path(), lambda f: f.write(data))

    def save_dirty(self):
        """Write dirty tiles and the index. Returns the number of tiles written."""
        n = 0
        for t in list(self.tiles.values()):
            if t.dirty:
                self.save_tile(t)
                n += 1
        if n or not os.path.exists(self._index_path()):
            self.save_index()
        return n

    def _evict(self):
        lim = int(self.p["max_loaded_tiles"])
        if len(self.tiles) <= lim:
            return
        busy = set()
        if self.ov_keys.size:
            gx, gy = unpack_keys(self.ov_keys)
            busy = set(zip((gx >> TILE_SHIFT).tolist(), (gy >> TILE_SHIFT).tolist()))
        saved = False
        for key in list(self.tiles.keys()):
            if len(self.tiles) <= lim:
                break
            if key in busy:
                continue
            t = self.tiles[key]
            if t.dirty:
                self.save_tile(t)
                saved = True
            del self.tiles[key]
            self.stats["tiles_evicted"] += 1
        if saved:
            self.save_index()

    def ram_bytes(self):
        return sum(t.ram_bytes() for t in self.tiles.values()) + \
            self.ov_keys.nbytes + sum(a.nbytes for a in self.ov.values())

    # ---- per-key access across tiles ----------------------------------------------
    @staticmethod
    def _groups(keys):
        gx, gy = unpack_keys(keys)
        tx = gx >> TILE_SHIFT
        ty = gy >> TILE_SHIFT
        tk = (tx << 32) | ty
        uniq, inv = np.unique(tk, return_inverse=True)
        for i, u in enumerate(uniq):
            sel = np.nonzero(inv == i)[0]
            yield (int(u >> 32), int(u & 0xFFFFFFFF)), sel, \
                (gy[sel] - (int(u & 0xFFFFFFFF) << TILE_SHIFT)).astype(np.intp), \
                (gx[sel] - (int(u >> 32) << TILE_SHIFT)).astype(np.intp)

    def lookup(self, keys, t=None):
        """Long-term (h, var_now) of global cells (NaN unknown). Loads tiles, never creates."""
        h = np.full(keys.size, np.nan)
        var = np.full(keys.size, np.nan)
        for (tx, ty), sel, ly, lx in self._groups(keys):
            tile = self.tile(tx, ty)
            if tile is None:
                continue
            h[sel] = tile.h[ly, lx]
            if t is None:
                var[sel] = tile.var[ly, lx]
            else:
                var[sel] = tile.var_now(ly, lx, t, self.p["q_day"])
        return h, var

    def sample_utm(self, e, n):
        """Long-term height and the effective reason / cost at UTM points."""
        e = np.atleast_1d(np.asarray(e, np.float64))
        n = np.atleast_1d(np.asarray(n, np.float64))
        keys = pack_keys(np.floor(e / RES), np.floor(n / RES))
        out = {"h": np.full(e.size, np.nan), "reason": np.zeros(e.size, np.int32),
               "cost": np.full(e.size, -1, np.int32), "tentative": np.zeros(e.size, bool),
               "visits": np.zeros(e.size, np.int32)}
        for (tx, ty), sel, ly, lx in self._groups(keys):
            tile = self.tile(tx, ty)
            if tile is None:
                continue
            reason, cost, tent = tile.effective()
            out["h"][sel] = tile.h[ly, lx]
            out["reason"][sel] = reason[ly, lx]
            c = cost[ly, lx].astype(np.int32)
            out["cost"][sel] = np.where(c == U8_NONE, -1, c)
            out["tentative"][sel] = tent[ly, lx]
            out["visits"][sel] = tile.visits[ly, lx]
        return out

    # ---- transforms ------------------------------------------------------------
    def utm_from_odom(self, pose):
        """(c, s, te, tn) of p_utm = R p_odom + t for pose map <- odom (x, y, yaw).
        The map frame is flat: no z is taken from TF (heights come from the window)."""
        yd = float(self.datum.get("yaw_rad", 0.0))
        cd, sd = math.cos(yd), math.sin(yd)
        x, y, yaw = pose
        te = self.datum["utm_e"] + cd * x - sd * y
        tn = self.datum["utm_n"] + sd * x + cd * y
        a = yd + yaw
        return math.cos(a), math.sin(a), te, tn

    # ---- sessions --------------------------------------------------------------
    def new_session(self):
        """A new session (e.g. a mowing program starts): closes the visits. The
        confirmation of a terrain ban needs lethal visits from confirm_sessions."""
        self.close_all()
        self.session += 1
        self.index["sessions"] = self.session
        return self.session

    # ---- the batch -------------------------------------------------------------
    def fuse(self, win, pose, stamp, robot=None):
        """Queue one window; fuse the queued ones that are `accept_delay` s old.

        pose: map <- odom (x, y, yaw) at the stamp; robot: base_link (x, y, z, yaw) in
        the window frame (z for the start offset and the anchor, yaw for the heading
        of the evidence). win.epoch: localisation epoch of the local node (a new one =
        odom re-anchored: queued batches dropped, visits closed, the offset found
        again); win.settling: a correction is being pulled in (batch and queue
        dropped). Returns {"queued", "dropped", "processed": [infos]}."""
        p = self.p
        self.stats["batches"] += 1
        info = {"stamp": stamp, "queued": False, "dropped": 0, "processed": []}
        if self.epoch is not None and win.epoch != self.epoch:
            info["dropped"] += len(self.pending)
            self.pending = []
            self.close_all()
            self.z_offset = None
            self.stats["epoch_changes"] += 1
        self.epoch = win.epoch
        if win.settling:
            info["dropped"] += len(self.pending) + 1
            self.stats["dropped_settling"] += len(self.pending) + 1
            self.pending = []
            return info
        self.stats["dropped_epoch"] += info["dropped"]
        self.pending.append((win, tuple(pose), float(stamp), robot))
        info["queued"] = True
        while self.pending and self.pending[0][2] <= stamp - p["accept_delay"]:
            w, ps, st, rb = self.pending.pop(0)
            info["processed"].append(self._process(w, ps, st, rb))
        return info

    def drain(self):
        """Fuse every queued batch now (end of a bag, tests). Returns their infos."""
        out = []
        while self.pending:
            w, ps, st, rb = self.pending.pop(0)
            out.append(self._process(w, ps, st, rb))
        return out

    def _process(self, win, pose, stamp, robot):
        p = self.p
        info = {"stamp": stamp, "fused": False, "cells": 0, "fresh": 0, "overlap": 0,
                "offset": None, "offset_src": None, "closed": 0, "refused": None,
                "offset_jump": False}
        # map <- odom jump: refuse this batch, close every open visit (no mixing of poses)
        if self.last_pose is not None:
            dx = pose[0] - self.last_pose[0]
            dy = pose[1] - self.last_pose[1]
            dyaw = math.degrees(abs(math.atan2(math.sin(pose[2] - self.last_pose[2]),
                                               math.cos(pose[2] - self.last_pose[2]))))
            if math.hypot(dx, dy) > p["jump_max_m"] or dyaw > p["jump_max_deg"]:
                self.last_pose = tuple(pose)
                info["closed"] = self.close_all()
                self.z_offset = None
                info["refused"] = "pose_jump"
                self.stats["refused_pose_jump"] += 1
                self.last_batch = info
                return info
        self.last_pose = tuple(pose)
        if self.last_stamp is not None and stamp < self.last_stamp - 1.0:
            # clock went back (bag loop / restart): restart the visits
            info["closed"] += self.close_all()
            self.last_stamp = None
            self.stats["clock_resets"] += 1

        c, s, te, tn = self.utm_from_odom(pose)
        H, W = win.shape
        # global cells covering the window footprint, each sampled at its centre in the
        # window (global cell -> window cell: no holes, no double cells at any yaw)
        cx = np.array([win.x0, win.x0 + W * win.res, win.x0, win.x0 + W * win.res])
        cy = np.array([win.y0, win.y0, win.y0 + H * win.res, win.y0 + H * win.res])
        ce = te + c * cx - s * cy
        cn = tn + s * cx + c * cy
        gx0, gx1 = int(math.floor(ce.min() / RES)), int(math.floor(ce.max() / RES))
        gy0, gy1 = int(math.floor(cn.min() / RES)), int(math.floor(cn.max() / RES))
        GX, GY = np.meshgrid(np.arange(gx0, gx1 + 1, dtype=np.int64),
                             np.arange(gy0, gy1 + 1, dtype=np.int64))
        GX = GX.ravel()
        GY = GY.ravel()
        de = (GX + 0.5) * RES - te
        dn = (GY + 0.5) * RES - tn
        xo = c * de + s * dn
        yo = -s * de + c * dn
        ix = np.floor((xo - win.x0) / win.res).astype(np.int64)
        iy = np.floor((yo - win.y0) / win.res).astype(np.int64)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        GX, GY, ix, iy = GX[inside], GY[inside], ix[inside], iy[inside]
        covered = pack_keys(GX, GY)
        el = win.elevation[iy, ix].astype(np.float64)
        var = win.variance[iy, ix].astype(np.float64)
        pts = win.points[iy, ix].astype(np.float64)
        m = int(round(p["edge_margin"] / win.res))
        valid = np.isfinite(el) & np.isfinite(var) & (var <= p["max_variance"]) & \
            (pts >= p["min_points"]) & (ix >= m) & (ix < W - m) & (iy >= m) & (iy < H - m)
        if win.t_fused is not None and self.last_stamp is not None:
            tf = win.t_fused[iy, ix]
            fresh = valid & np.isfinite(tf) & (tf > self.last_stamp)
        else:
            fresh = valid.copy()
        keys = covered[valid]
        h = el[valid]
        info["cells"] = int(valid.sum())
        info["fresh"] = int(fresh.sum())

        # ---- alignment
        ref_h, ref_var = self.lookup(keys, stamp)
        use_lt = np.isfinite(ref_h) & (ref_var <= p["ref_max_variance"])
        ref = np.where(use_lt, ref_h, np.nan)
        if self.ov_keys.size:
            pos = np.searchsorted(self.ov_keys, keys)
            pos_c = np.minimum(pos, self.ov_keys.size - 1)
            in_ov = (self.ov_keys[pos_c] == keys) & ~use_lt
            ref[in_ov] = self.ov["h"][pos_c[in_ov]]
        ok = np.isfinite(ref)
        info["overlap"] = int(ok.sum())
        offset = None
        if ok.sum() >= p["align_min_cells"]:
            r = h[ok] - ref[ok]
            if self.z_offset is None:
                start = hist_mode(r, p["align_hist_bin"])    # any size, robust to a wrong part
            else:
                start = -self.z_offset
            offset = -huber_mean(r, p["align_huber"], start=start)
            info["offset_src"] = "overlap"
        rz = None
        if robot is not None and len(robot) >= 3 and robot[2] is not None:
            rz = float(robot[2])
        anchor_off = None
        if self.anchor is not None and rz is not None:
            re_ = te + c * robot[0] - s * robot[1]
            rn_ = tn + s * robot[0] + c * robot[1]
            z = np.asarray(self.anchor(np.array([re_]), np.array([rn_])), np.float64).ravel()
            if z.size and np.isfinite(z[0]):
                anchor_off = float(z[0] - rz)
        info["anchor_offset"] = anchor_off
        if offset is not None:
            if anchor_off is not None:
                r_a = anchor_off - offset
                a = self.anchor_residual
                self.anchor_residual = r_a if a is None else a + p["anchor_ema"] * (r_a - a)
        elif self.z_offset is not None:
            offset = self.z_offset
            info["offset_src"] = "carried"
        else:
            ground = self._ground_at_robot(c, s, te, tn, robot) if rz is not None else None
            if ground is not None:
                offset = ground - rz                       # e.g. standing in the dock
                info["offset_src"] = "robot_ground"
            elif anchor_off is not None:
                offset = anchor_off
                info["offset_src"] = "anchor"
            else:
                offset = 0.0
                info["offset_src"] = "none"
        if self.z_offset is not None and abs(offset - self.z_offset) > p["align_jump"]:
            # odometry z jump: no visit may mix the two offsets
            info["closed"] += self.close_all()
            info["offset_jump"] = True
            self.stats["offset_jumps"] += 1
        self.z_offset = offset
        info["offset"] = offset

        # ---- open visits: overwrite with the fresh cells
        sector = 0
        if robot is not None and len(robot) >= 4 and robot[3] is not None:
            yaw_utm = float(self.datum.get("yaw_rad", 0.0)) + pose[2] + float(robot[3])
            sector = int(math.floor((yaw_utm % (2 * math.pi)) / (math.pi / 4))) % 8
        fk = covered[fresh]
        fsel = fresh[valid]                     # fresh among the valid cells
        vals = {
            "h": h[fsel] + offset,
            "var": var[fresh] + p["align_sigma"] ** 2,
            "t": np.full(fk.size, float(stamp)),
            "cost": win.cost[iy[fresh], ix[fresh]].astype(np.float64),
            "reason": win.reason[iy[fresh], ix[fresh]].astype(np.float64),
            "slope": win.slope[iy[fresh], ix[fresh]].astype(np.float64),
            "step": win.step[iy[fresh], ix[fresh]].astype(np.float64),
            "rough": win.roughness[iy[fresh], ix[fresh]].astype(np.float64),
            "points": pts[fresh],
            "sector": np.full(fk.size, float(sector)),
            "session": np.full(fk.size, float(self.session)),
        }
        self._ov_upsert(fk, vals)
        # ---- close visits: left the window or timed out
        if self.ov_keys.size:
            inwin = np.isin(self.ov_keys, covered, assume_unique=False)
            stale = self.ov["t"] < stamp - p["visit_timeout"]
            info["closed"] += self._close(~inwin | stale)
        self.last_stamp = stamp if self.last_stamp is None else max(self.last_stamp, stamp)
        info["fused"] = True
        self.stats["batches_fused"] += 1
        self.last_batch = info
        self._evict()
        return info

    def _ground_at_robot(self, c, s, te, tn, robot):
        """Median stored height within robot_ground_radius of base_link, or None."""
        r = self.p["robot_ground_radius"]
        e = te + c * robot[0] - s * robot[1]
        n = tn + s * robot[0] + c * robot[1]
        k = int(math.ceil(r / RES))
        gx0, gy0 = int(math.floor(e / RES)), int(math.floor(n / RES))
        GX, GY = np.meshgrid(np.arange(gx0 - k, gx0 + k + 1, dtype=np.int64),
                             np.arange(gy0 - k, gy0 + k + 1, dtype=np.int64))
        near = np.hypot((GX + 0.5) * RES - e, (GY + 0.5) * RES - n) <= r
        h, _ = self.lookup(pack_keys(GX[near], GY[near]))
        h = h[np.isfinite(h)]
        return float(np.median(h)) if h.size >= 5 else None

    def _ov_upsert(self, keys, vals):
        if keys.size == 0:
            return
        order = np.argsort(keys)
        keys = keys[order]
        vals = {k: np.asarray(v)[order] for k, v in vals.items()}
        if self.ov_keys.size:
            pos = np.searchsorted(self.ov_keys, keys)
            pos_c = np.minimum(pos, self.ov_keys.size - 1)
            hit = self.ov_keys[pos_c] == keys
            for k in _VISIT_FIELDS:
                self.ov[k][pos_c[hit]] = vals[k][hit]
            keys = keys[~hit]
            vals = {k: v[~hit] for k, v in vals.items()}
        if keys.size:
            allk = np.concatenate([self.ov_keys, keys])
            order = np.argsort(allk, kind="stable")
            self.ov_keys = allk[order]
            for k in _VISIT_FIELDS:
                self.ov[k] = np.concatenate([self.ov[k], vals[k]])[order]

    def _close(self, mask):
        n = int(np.count_nonzero(mask))
        if n == 0:
            return 0
        keys = self.ov_keys[mask]
        vals = {k: self.ov[k][mask] for k in _VISIT_FIELDS}
        keep = ~mask
        self.ov_keys = self.ov_keys[keep]
        for k in _VISIT_FIELDS:
            self.ov[k] = self.ov[k][keep]
        self._commit(keys, vals)
        self.stats["visits_closed"] += n
        return n

    def close_all(self, stamp=None):
        """Close every open visit (shutdown, pose jump, end of a mowing program)."""
        if self.ov_keys.size == 0:
            return 0
        return self._close(np.ones(self.ov_keys.size, bool))

    def _commit(self, keys, vals):
        mowed_all = None
        if self.mowed is not None:
            gx, gy = unpack_keys(keys)
            mowed_all = np.asarray(self.mowed((gx + 0.5) * RES, (gy + 0.5) * RES), np.float64)
        for (tx, ty), sel, ly, lx in self._groups(keys):
            tile = self.tile(tx, ty, create=True)
            v = {k: vals[k][sel] for k in _VISIT_FIELDS}
            tile.commit(ly, lx, v, self.p, None if mowed_all is None else mowed_all[sel])

    # ---- import -------------------------------------------------------------------
    def import_map_grid(self, origin, res, layers, t, overwrite=False, align=True):
        """Seed the store from a grid in the map frame of the store's datum (the
        offline mosaic). Every imported cell is ONE tentative visit (the mosaic's
        per-cloud runs are no visits): flags F_TENTATIVE, visits 1, a terrain lethal
        reason counts as one lethal visit, never confirmed by the import.

        origin: (x, y) min corner of cell [0, 0]; layers: arrays [row, col], row 0 =
        lowest y: elevation [m], sigma [m], cost (0..100, NaN unknown), reason
        (0..7), slope [rad], step [m], roughness [m]; t: unix s (scalar or array).
        Only unknown cells are filled unless `overwrite`. With `align` the grid is
        first shifted in height onto the cells the store already knows (each mosaic
        has its own height reference; >= align_min_cells of overlap, Huber).
        Returns (cell count, height offset applied)."""
        elev = np.asarray(layers["elevation"], np.float64)
        H, W = elev.shape
        c, s, te, tn = self.utm_from_odom((0.0, 0.0, 0.0))     # map frame = "odom" with identity
        x0, y0 = float(origin[0]), float(origin[1])
        cx = np.array([x0, x0 + W * res, x0, x0 + W * res])
        cy = np.array([y0, y0, y0 + H * res, y0 + H * res])
        ce = te + c * cx - s * cy
        cn = tn + s * cx + c * cy
        GX, GY = np.meshgrid(np.arange(int(math.floor(ce.min() / RES)), int(math.floor(ce.max() / RES)) + 1,
                                       dtype=np.int64),
                             np.arange(int(math.floor(cn.min() / RES)), int(math.floor(cn.max() / RES)) + 1,
                                       dtype=np.int64))
        GX = GX.ravel()
        GY = GY.ravel()
        de = (GX + 0.5) * RES - te
        dn = (GY + 0.5) * RES - tn
        col = np.floor((c * de + s * dn - x0) / res).astype(np.int64)
        row = np.floor((-s * de + c * dn - y0) / res).astype(np.int64)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        GX, GY, col, row = GX[ok], GY[ok], col[ok], row[ok]
        ok = np.isfinite(elev[row, col])
        GX, GY, col, row = GX[ok], GY[ok], col[ok], row[ok]
        offset = 0.0
        if align:
            have, _ = self.lookup(pack_keys(GX, GY))
            f = np.isfinite(have)
            if np.count_nonzero(f) >= self.p["align_min_cells"]:
                r = elev[row[f], col[f]] - have[f]
                offset = -huber_mean(r, self.p["align_huber"], start=hist_mode(r, self.p["align_hist_bin"]))
        elev = elev + offset
        tt = np.broadcast_to(np.asarray(t, np.float64), elev.shape)[row, col]
        sig = np.asarray(layers["sigma"], np.float64)[row, col]
        var = np.maximum(np.where(np.isfinite(sig), sig, 0.05) ** 2, self.p["sigma_min"] ** 2) + \
            self.p["align_sigma"] ** 2
        cost = np.asarray(layers["cost"], np.float64)[row, col]
        reason = np.asarray(layers["reason"]).astype(np.int32)[row, col]
        total = 0
        for (tx, ty), sel, ly, lx in self._groups(pack_keys(GX, GY)):
            tile = self.tile(tx, ty, create=True)
            put = np.ones(sel.size, bool) if overwrite else ~np.isfinite(tile.h[ly, lx])
            if not np.any(put):
                continue
            sel, ly, lx = sel[put], ly[put], lx[put]
            r = reason[sel]
            tile.h[ly, lx] = elev[row[sel], col[sel]]
            tile.var[ly, lx] = var[sel]
            tile.t_last[ly, lx] = np.round(tt[sel]).astype(np.uint32)
            tile.t_change[ly, lx] = tile.t_last[ly, lx]
            tile.visits[ly, lx] = 1
            tile.l_step[ly, lx] = (r == REASON_STEP).astype(np.uint8)
            tile.l_slope[ly, lx] = (r == REASON_SLOPE).astype(np.uint8)
            tile.streak[ly, lx] = ((r != REASON_STEP) & (r != REASON_SLOPE)).astype(np.uint8)
            terr = (r == REASON_STEP) | (r == REASON_SLOPE)
            tile.l_sess[ly, lx] = terr.astype(np.uint8)
            tile.l_last[ly, lx] = np.where(terr, self.session & 0xFFFF, 0).astype(np.uint16)
            tile.l_dirs[ly, lx] = 0                       # the heading of the mosaic's views is not known
            cs = cost[sel]
            tile.cost[ly, lx] = np.where(np.isfinite(cs) & (cs >= 0), np.clip(np.round(cs), 0, 100),
                                         U8_NONE).astype(np.uint8)
            tile.reason[ly, lx] = np.clip(r, 0, 254).astype(np.uint8)
            tile.slope[ly, lx] = _q8(np.asarray(layers["slope"], np.float64)[row[sel], col[sel]],
                                     2.0 * 180.0 / math.pi)
            tile.step[ly, lx] = _q8(np.asarray(layers["step"], np.float64)[row[sel], col[sel]], 100.0)
            tile.rough[ly, lx] = _q8(np.asarray(layers["roughness"], np.float64)[row[sel], col[sel]],
                                     1000.0)
            fl = F_TENTATIVE | np.where(r == REASON_EXEMPT, F_EXEMPT, 0) | \
                np.where(r == REASON_OBSTACLE, F_OBST_LAST, 0)
            tile.flags[ly, lx] = fl.astype(np.uint8)
            tile.cand_h[ly, lx] = np.nan
            tile.cand_n[ly, lx] = 0
            tile.version += 1
            tile.dirty = True
            total += int(sel.size)
        self.stats["imported_cells"] += total
        return total, offset

    def open_visit_heights(self):
        """(keys, heights) of the open visits (for display of the live state)."""
        return self.ov_keys.copy(), self.ov["h"].copy()

    def summary(self):
        known = sum(t.known() for t in self.tiles.values())
        return {"tiles_loaded": len(self.tiles), "tiles_indexed": len(self.index["tiles"]),
                "known_loaded": known, "open_visits": int(self.ov_keys.size),
                "z_offset": self.z_offset, "anchor_residual": self.anchor_residual,
                "ram_bytes": self.ram_bytes(),
                "broken": list(self.broken), "stats": dict(self.stats)}
