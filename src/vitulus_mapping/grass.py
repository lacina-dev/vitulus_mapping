#!/usr/bin/env python3
"""
grass.py -- grass height in front of the robot from the D435 depth image, told
apart from terrain that rises or falls. Logic of nodes/grass_height and
nodes/grass_calib. NO rospy: everything runs in tests with plain arrays.

WHAT THE CAMERA SEES. Stereo depth sees the top of the vegetation (the
"carpet"), never the soil under it. A carpet 4 cm above the robot's wheel plane
is either 4 cm of grass on flat ground, or short grass on ground that rises by
4 cm. One frame cannot tell, so the answer is built in three ways:

1. DEFERRED (the map layer). Every frame stores the carpet top of each 0.2 m map
   cell it sees (`Pending`). The ground of the cell comes later: when the wheels
   drive over it (the robot's own altitude, see `Track`), or, if they never do
   within `pending_ttl_s`, from the site DEM of earlier runs (dem.npz,
   trajectory_dem: wheel contact, immune to vegetation). Height = carpet - ground.
   Only resolved cells reach the "grass" layer.

2. IMMEDIATE (/grass/ahead). Ground ahead = the cells the wheels already know,
   else the DEM, else the robot's own plane extrapolated forward
   (`terrain_source` wheels | dem | extrapolated | surface). The uncertainty of
   the extrapolation grows with the distance and with how much the slope changed
   along the last ~2 m of the track.

3. TERRAIN CHANGE. A fitted plane of the carpet in the corridor, against the
   robot plane: terrain is smooth and coherent, grass is rough and patchy.
     * carpet as smooth as bare ground (roughness <= calib rough_bare_mm): the
       visible surface IS the ground; whatever its height or slope, the grass
       is below the cut (terrain_source surface).
     * carpet tilted against the robot plane (slope or side slope beyond the
       limits) and no ground from wheels or DEM: the offset cannot be split
       between grass and terrain -> class unknown, reason terrain_change.
     * otherwise the offset over the extrapolated plane is grass.

ALTITUDE. `Track` integrates the robot's altitude from the attitude (IMU roll and
pitch) and the horizontal motion (map pose). The carpet seen at t1 and the ground
under the wheels at t2 are both expressed with the same attitude, so a constant
pitch bias cancels: carpet_alt - ground_alt = x sin(p + b) + g - x sin(p + b) = g.
Over longer tracks it does not (the blade's vibration tilts the IMU by ~1.3 deg:
2 cm of altitude per metre): `Drift` fits the drift curve from the wheels
coming back over cells they stamped before, and wheel ground from the previous
lane is brought onto the current drift.

CALIBRATION (nodes/grass_calib, file ~/.vitulus/grass/calib.json): concrete gives
the camera zero (offset, pitch and roll of the plane the camera sees under a robot
standing on flat concrete), the two lawn samples give the scale between the
carpet the camera sees (p90 of the points) and the ruler height:
true_mm = scale * raw_mm + bias_mm.

Frames: base_link x forward, y left, z up, origin on the ground between the wheels
(wheel centres 0.105 m above it, radius 0.1025 m). Map cells [row, col], row =
y, col = x, the worldlayers convention.
"""

import json
import math
import os
import tempfile
import time

import numpy as np

from . import worldlayers as wl

__all__ = ["DEFAULTS", "CLASSES", "Calib", "Rays", "Track", "GrassEngine",
           "GrassGrid", "grass_layer", "analyze_corridor", "classify",
           "rot_rpy", "class_probability", "ImuBuffer", "save_merged", "CALIB_PATH", "CALIB_DIR", "fit_calibration", "strip_stats"]

CALIB_DIR = os.path.expanduser("~/.vitulus/grass/calib")
CALIB_PATH = os.path.expanduser("~/.vitulus/grass/calib.json")

CLASSES = ("below_cut", "+2-4cm", ">+4cm", "unknown")

SRC_NONE, SRC_WHEELS, SRC_DEM = 0, 1, 2
SRC_NAMES = {SRC_WHEELS: "wheels", SRC_DEM: "dem"}

# wheel contact points in base_link (vitulus4WD URDF: x +-0.125, y +-0.236)
WHEELS_XY = ((0.0, 0.0), (0.125, 0.236), (0.125, -0.236),
             (-0.125, 0.236), (-0.125, -0.236))
# ground under the body stamped by the wheels: the plane of the four contacts over
# the whole footprint (axles and midline x three, wheels, half-way, centre y five),
# so the lane between the wheel tracks is known too
BODY_XY = tuple((bx, by) for bx in (0.125, 0.0, -0.125)
                for by in (0.236, 0.118, 0.0, -0.118, -0.236))

DEFAULTS = {
    "x_min": 0.8,              # corridor in base_link, m
    "x_max": 1.6,
    "half_width": 0.35,
    "z_lo": -0.25,             # points outside this band (m, robot plane) are not ground/grass
    "z_hi": 0.30,
    "bin_x": 0.10,             # corridor bins for the plane and roughness
    "bins_y": 3,
    "bin_min_pts": 12,
    "min_bins": 8,
    "canopy_pct": 90.0,        # carpet top = this percentile of point heights
    "cell": 0.2,               # map cell, m
    "cell_min_pts": 8,
    "pending_ttl_s": 120.0,    # a cell waits this long for the wheels, then DEM or nothing
    "pending_max": 6000,
    "wheel_max_track_m": 4.0,  # wheels resolve a cell seen at most this far back along the track
    "wheel_ahead_max_m": 3.0,  # /grass/ahead uses wheel ground stamped at most this far back
                               # (coming back over its own track an IMU pitch error adds up)
    "dem_max_slope_deg": 30.0, # DEM ground steeper than this from the robot = DEM wrong there
    "slope_change_deg": 2.5,   # carpet tilted against the robot plane beyond this = terrain change
    "slope_sigma_min_deg": 0.5,
    "track_len_m": 2.0,
    "track_hist_s": 30.0,      # track history also bounded in time and count
    "track_hist_max": 400,
    "slope_smooth_m": 1.0,     # terrain slope ahead = median of the robot's slope over this much track
    "slope_smooth_s": 3.0,     # ... and over at most this much time (slow driving, standing)
    "body_agree_deg": 3.0,     # carpet within this of the body plane: the body already follows the terrain
    "imu_bias_unmeasured_deg": 1.5,  # possible IMU pitch error when calibration did not measure it
    "imu_bias_measured_deg": 0.3,
    "track_jump_m": 1.0,       # a pose step longer than this restarts the altitude
    "track_jump_min_m": 0.15,  # ... and so does one longer than this + vmax * dt
    "track_vmax_mps": 0.8,     # (the robot drives <= 0.5 m/s: 0.35 m per 0.25 s frame)
    "track_gap_s": 2.0,
    "drift_deg": 0.5,          # attitude error assumed for the integrated altitude (sigma per metre)
    "ground_max": 20000,       # wheel ground cells kept (a garden run: ~3500 cells per 200 m)
    # DRIFT of the integrated altitude. The IMU pitch is off while driving (the
    # spinning blade: 1.3 deg on 2026-11-08, 0.94 deg at rest), so the altitude
    # drifts with the track (2.1 cm per metre). Where the wheels drive over a
    # cell they stamped before, the altitude difference over the track between is
    # that drift: a drift curve is fitted to such revisits (class Drift) and
    # removed, and wheel ground stamped long ago (the previous lane) becomes usable.
    "wheel_keep_m": 120.0,     # wheel ground kept this far back along the track
    "drift_min_ds_m": 3.0,     # a revisit counts when this much track lies between
    "drift_window_m": 120.0,   # the drift curve covers this much track back
    "drift_knot_m": 4.0,       # ... a knot every this much track
    "drift_max_pairs": 400,    # revisits kept for the fit (newest)
    "drift_refit_m": 0.5,      # refit after this much track
    "drift_refit_pairs": 30,   # ... or this many new revisits
    "drift_smooth": 1.0,       # curvature penalty of the curve
    "drift_min_pairs": 20,     # revisits needed for a curve
    "drift_min_span_m": 60.0,  # ... and this much summed track between them
    "drift_max_rate": 0.06,    # |slope| above this (3.4 deg) now = no curve (a pose fault)
    "drift_sigma_min_m": 0.008,  # floor of the curve's uncertainty
    "drift_use_max_m": 110.0,  # with a curve: ground stamped up to this far back is used
    "lane_step_mm": 25.0,      # strips side by side differing more = kerb or grass edge (a step)
    "strip_w": 0.07,           # lateral strips for kerbs and mown | uncut edges
    "strip_min_pts": 30,
    "strip_pure_gap": 0.35,    # a strip with a larger decile gap holds two heights (border strip)
    "side_even_mm": 15.0,      # strips on each side of a step must agree this well
    "edge_top_rough": 0.15,    # grass tops vary: (p90-p60)/(p90-p10) below this = flat top (kerb)
    "edge_bottom_mm": 20.0,    # a taller lane is grass when its p10 is this close to the low lanes' p10
    "edge_bottom_frac": 0.5,   # ... or p10 - lo_bot <= frac * (p90 - lo_bot) (lower half from the low lanes' bottom to its top)
    "cell_split_mm": 40.0,     # a map cell whose p90-p10 exceeds this and whose deciles have
    "cell_split_gap": 0.5,     # a gap over this share of it holds two heights (kerb, bank): dropped
    "cell_split_gap_terrain": 0.3,  # ... this share when the frame saw raised terrain beside
    "cell_flat_bottom": 0.15,  # (p40-p10)/(p90-p10) below this = flat bottom under a higher top
    "cell_min_cover": 0.5,     # a look counts for a map cell only when it saw this share of its area
    "bare_offset_mm": 20.0,    # smooth AND this close to the ground plane = bare surface
    "pivot_x": 0.125,          # front wheel contact: where a slope seen ahead starts
    "wheels_min_frac": 0.5,    # share of corridor cells with ground to call it wheels/dem
    "wheels_partial_min_cells": 2,   # fewer known: the plane through them and the footprint
    "wheels_partial_resid_mm": 20.0, # ... if the known cells sit on it this well
    "conf_min": 0.6,           # probability that the class is right
    "cut_default_mm": 60.0,
    "smooth_frames": 3,
    "smooth_window_s": 1.5,
}

# map layer
GRASS_RES = 0.2
HIST_STEP_MM = 10
HIST_BINS = 41                 # 0..400 mm in 10 mm, last bin = 400+
LAST_NONE = -1


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def rot_rpy(roll, pitch, yaw=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll), ROS convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def quat_rpy(qx, qy, qz, qw):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


class Rays(object):
    """Decimated pixel rays of the depth camera, already in base_link.
    points(depth) -> Nx3 base_link (m) of the valid pixels."""

    def __init__(self, K, R_bc, t_bc, shape, stride=4, row_min=0):
        h, w = shape
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        vs = np.arange(row_min, h, stride)
        us = np.arange(0, w, stride)
        uu, vv = np.meshgrid(us, vs)
        self.idx = (vv.ravel(), uu.ravel())
        d = np.stack([(uu.ravel() - cx) / fx, (vv.ravel() - cy) / fy,
                      np.ones(uu.size)], axis=1)
        self.dirs = d @ np.asarray(R_bc, float).T           # per metre of depth
        self.t = np.asarray(t_bc, float)
        self.shape = (h, w)
        self.stride = stride

    def points(self, depth, scale=0.001, z_min=0.15, z_max=4.0):
        z = depth[self.idx].astype(np.float64) * scale
        ok = (z > z_min) & (z < z_max)
        return z[ok, None] * self.dirs[ok] + self.t


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
class Calib(object):
    """Camera zero + scale. Identity until grass_calib fit wrote a file."""

    FIELDS = ("offset_mm", "pitch_deg", "roll_deg", "scale", "bias_mm",
              "rough_bare_mm", "noise_mm", "x_min", "x_max", "imu_pitch_offset_deg")

    def __init__(self, data=None):
        self.offset_mm = 0.0
        self.pitch_deg = 0.0     # tilt of the concrete plane seen, +x up
        self.roll_deg = 0.0      # +y up
        self.scale = 1.0
        self.bias_mm = 0.0
        self.rough_bare_mm = 6.0     # p90-p10 of concrete is 2-3 mm indoors
        self.noise_mm = 5.0
        self.x_min = None        # recommended corridor, None = config
        self.x_max = None
        self.imu_pitch_offset_deg = 0.0   # IMU pitch minus true body pitch (beton both ways)
        self.imu_offset_measured = False
        self.calibrated = False
        self.error = None        # why the file could not be used, None = fine
        self.created = None
        self.path = None
        if data:
            for k in self.FIELDS:
                if data.get(k) is not None:
                    setattr(self, k, float(data[k]))
            self.calibrated = bool(data.get("calibrated", True))
            self.imu_offset_measured = bool(data.get("imu_offset_measured", False))
            self.created = data.get("created")

    def zero(self, P):
        """Points with the concrete plane removed (z relative to true ground)."""
        if not len(P):
            return P
        Q = P.copy()
        Q[:, 2] -= (self.offset_mm / 1000.0
                    + math.tan(math.radians(self.pitch_deg)) * Q[:, 0]
                    + math.tan(math.radians(self.roll_deg)) * Q[:, 1])
        return Q

    def height(self, raw_mm):
        return self.scale * raw_mm + self.bias_mm

    def to_dict(self):
        d = dict((k, getattr(self, k)) for k in self.FIELDS)
        d.update(calibrated=self.calibrated, created=self.created,
                 imu_offset_measured=self.imu_offset_measured)
        return d

    @classmethod
    def load(cls, path=None):
        path = path or CALIB_PATH
        if not os.path.exists(path):
            c = cls()
            c.error = "missing %s" % path
        else:
            try:
                with open(path) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("not an object")
                c = cls(data)
            except (IOError, OSError, ValueError, TypeError, AttributeError) as exc:
                c = cls()
                c.error = "bad %s: %s" % (path, exc)
        c.path = path
        return c

    def save(self, path=None):
        path = path or self.path or CALIB_PATH
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            dfd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.path = path


# --------------------------------------------------------------------------
# corridor analysis (one frame, base_link)
# --------------------------------------------------------------------------
def _robust_plane(x, y, z, w=None, iters=3):
    """z = a + b x + c y, trimmed least squares. Returns (a, b, c, resid)."""
    A = np.c_[np.ones(len(x)), x, y]
    keep = np.ones(len(x), bool)
    coef = np.zeros(3)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        Aw = A[keep]
        zw = z[keep]
        if w is not None:
            sw = np.sqrt(w[keep])
            Aw = Aw * sw[:, None]
            zw = zw * sw
        coef = np.linalg.lstsq(Aw, zw, rcond=None)[0]
        r = z - A @ coef
        mad = np.median(np.abs(r[keep])) * 1.4826
        keep = np.abs(r) <= max(3.0 * mad, 0.004)
    return coef[0], coef[1], coef[2], z - A @ coef


def _groups(key, vals):
    """Sort vals by (key, value): starts, counts, sorted values, key of each group."""
    order = np.lexsort((vals, key))
    k, v = key[order], vals[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[starts, len(k)])
    return starts, counts, v, k[starts]


def _group_pct(v, starts, counts, q):
    """Percentile (linear, numpy default) of each sorted group."""
    pos = (counts - 1) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, counts - 1)
    f = pos - lo
    return v[starts + lo] * (1.0 - f) + v[starts + hi] * f


def _decile_gap(v, starts, counts):
    """Largest step between adjacent deciles (p10..p90) as a share of p90-p10, per
    sorted group: ~0.15 for one spread surface, > 0.5 for two heights (a kerb
    and the lawn beside it in one strip or cell)."""
    d = np.stack([_group_pct(v, starts, counts, q) for q in range(10, 100, 10)])
    span = d[-1] - d[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(span > 1e-6, np.diff(d, axis=0).max(axis=0) / span, 0.0)


def analyze_corridor(P, cfg, x_min=None, x_max=None):
    """Carpet of the corridor from points already zeroed by the calibration
    (base_link, m). Returns None without enough data, else a dict (mm, deg):
      canopy_mm   carpet top at the corridor centre over the robot plane
      slope_rel_deg, roll_rel_deg  tilt of the carpet against the robot plane
      rough_mm    median p90-p10 of the bins after the plane (texture)
      patch_mm    robust spread of the bin tops around the plane (patchiness)
      bins        n, valid_frac
    """
    x0 = cfg["x_min"] if x_min is None else x_min
    x1 = cfg["x_max"] if x_max is None else x_max
    hw = cfg["half_width"]
    m = ((P[:, 0] >= x0) & (P[:, 0] < x1) & (np.abs(P[:, 1]) < hw)
         & (P[:, 2] > cfg["z_lo"]) & (P[:, 2] < cfg["z_hi"]))
    Q = P[m]
    nbx = max(1, int(round((x1 - x0) / cfg["bin_x"])))
    nby = int(cfg["bins_y"])
    if len(Q) < cfg["bin_min_pts"] * 4:
        return None
    ix = np.clip(((Q[:, 0] - x0) / (x1 - x0) * nbx).astype(int), 0, nbx - 1)
    iy = np.clip(((Q[:, 1] + hw) / (2 * hw) * nby).astype(int), 0, nby - 1)
    key = ix * nby + iy
    pct = cfg["canopy_pct"]
    starts, counts, zs, gkey = _groups(key, Q[:, 2])
    good = counts >= cfg["bin_min_pts"]
    # per bin: mean x, y, z and the bin's own x trend (slope), all by sums
    nk = nbx * nby
    cnt = np.bincount(key, minlength=nk).astype(float)
    sx = np.bincount(key, Q[:, 0], nk)
    sy = np.bincount(key, Q[:, 1], nk)
    sz = np.bincount(key, Q[:, 2], nk)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my, mz = sx / cnt, sy / cnt, sz / cnt
        dx = Q[:, 0] - mx[key]
        vxx = np.bincount(key, dx * dx, nk)
        vxz = np.bincount(key, dx * (Q[:, 2] - mz[key]), nk)
        k1 = np.where(vxx > 1e-9, vxz / vxx, 0.0)
    res = Q[:, 2] - mz[key] - k1[key] * dx        # within-bin texture, slope removed
    _s2, _c2, rs, _k2 = _groups(key, res)
    spans_all = _group_pct(rs, _s2, _c2, 90.0) - _group_pct(rs, _s2, _c2, 10.0)
    tops_all = _group_pct(zs, starts, counts, pct)
    bots_all = _group_pct(zs, starts, counts, 10.0)
    gk = gkey[good]
    bx, by, top, nb = mx[gk], my[gk], tops_all[good], counts[good]
    bot = bots_all[good]
    span = spans_all[good]
    spans = list(span)
    n = len(top)
    if n < cfg["min_bins"]:
        return None
    a, b, c, r = _robust_plane(bx, by, top, w=np.sqrt(nb))
    xc = 0.5 * (x0 + x1)
    patch = float(np.median(np.abs(r)) * 1.4826)
    # the lowest lane (mown stripe, path, concrete) is the terrain reference: if
    # it tilts with the rest, the ground tilts; if only the tall lanes rise, grass
    lane = gk % nby
    low_slope, low_top, low_j = None, None, None
    lanes = []                  # (lane, carpet at xc mm, roughness mm, bottom p10 at xc mm)
    for j in range(nby):
        sel = lane == j
        if sel.sum() < 2:
            continue
        lanes.append((j, float(np.median(top[sel] - b * (bx[sel] - xc))) * 1000.0,
                      float(np.median(span[sel])) * 1000.0,
                      float(np.median(bot[sel] - b * (bx[sel] - xc))) * 1000.0))
        if sel.sum() < 3 or np.ptp(bx[sel]) < 0.3 * (x1 - x0):
            continue
        k = np.polyfit(bx[sel], top[sel], 1)
        mean_top = float(np.median(top[sel]))
        if low_top is None or mean_top < low_top:
            low_top, low_slope, low_j = mean_top, math.degrees(math.atan(k[0])), j
    low_lane = next((L for L in lanes if low_top is not None and L[0] == low_j), None)
    # fine lateral strips (0.07 m) over the whole corridor length, x-trend of the
    # plane removed: a kerb or a mown | uncut edge rarely sits on a lane border
    nfs = max(3, int(round(2 * hw / cfg["strip_w"])))
    fs = np.clip(((Q[:, 1] + hw) / (2 * hw) * nfs).astype(int), 0, nfs - 1)
    st, cn, zz, sk = _groups(fs, Q[:, 2] - b * (Q[:, 0] - xc))
    p10s, p90s = _group_pct(zz, st, cn, 10.0), _group_pct(zz, st, cn, pct)
    gaps = _decile_gap(zz, st, cn)
    p60s = _group_pct(zz, st, cn, 60.0)
    strips = [(float(-hw + (k + 0.5) * 2 * hw / nfs), float(lo) * 1000.0, float(hi) * 1000.0,
               float(gp), float(mid) * 1000.0)
              for k, n_, lo, hi, gp, mid in zip(sk, cn, p10s, p90s, gaps, p60s)
              if n_ >= cfg["strip_min_pts"]]
    return {
        "strips": strips,           # (y, p10, p90, decile gap share, p60), mm, y ascending
        "low_lane_slope_deg": low_slope,
        "low_lane": low_lane,       # (lane, carpet at xc mm, roughness mm) of the lowest lane
        "lanes": lanes,
        "canopy_mm": (a + b * xc) * 1000.0,
        "slope_rel_deg": math.degrees(math.atan(b)),
        "roll_rel_deg": math.degrees(math.atan(c)),
        "rough_mm": float(np.median(spans)) * 1000.0,
        "patch_mm": patch * 1000.0,
        "bins": n, "valid_frac": n / float(nbx * nby), "x_c": xc,
        "points": int(len(Q)),
    }


def classify(height_mm, cut_mm):
    if height_mm is None:
        return "unknown"
    excess = height_mm - cut_mm
    if excess < 20.0:
        return "below_cut"
    if excess <= 40.0:
        return "+2-4cm"
    return ">+4cm"


def class_probability(height_mm, sigma_mm, cut_mm):
    """Probability that the true height lies in the class of height_mm when the
    estimate is normal with sigma_mm: the confidence of the class."""
    lo, hi = {"below_cut": (-1e9, cut_mm + 20.0), "+2-4cm": (cut_mm + 20.0, cut_mm + 40.0),
              ">+4cm": (cut_mm + 40.0, 1e9)}[classify(height_mm, cut_mm)]
    s = max(1e-3, sigma_mm) * math.sqrt(2.0)
    return 0.5 * (math.erf((hi - height_mm) / s) - math.erf((lo - height_mm) / s))


def strip_stats(P, x0, x1, half_width, z_lo=-0.25, z_hi=0.45, pct=90.0):
    """Stats of one strip of zeroed or raw points (calibration). mm."""
    m = ((P[:, 0] >= x0) & (P[:, 0] < x1) & (np.abs(P[:, 1]) < half_width)
         & (P[:, 2] > z_lo) & (P[:, 2] < z_hi))
    z = P[m, 2] * 1000.0
    if len(z) < 10:
        return None
    return {"n": int(len(z)), "median_mm": float(np.median(z)),
            "p90_mm": float(np.percentile(z, pct)),
            "p10_mm": float(np.percentile(z, 10))}


# --------------------------------------------------------------------------
# altitude along the track
# --------------------------------------------------------------------------
class Track(object):
    """Robot altitude integrated from attitude and horizontal motion. The zero is
    arbitrary; `epoch` changes when it restarts (a jump, a gap), and altitudes
    of different epochs are never compared."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 0
        self.h = 0.0
        self.last = None            # (t, x, y)
        self.hist = []              # (s, h, pitch_world_deg) along the track
        self.s = 0.0

    def reset(self):
        self.epoch += 1
        self.h = 0.0
        self.last = None
        self.hist = []
        self.s = 0.0

    def update(self, t, x, y, R):
        """R = attitude rotation (base_link -> map axes). Returns altitude."""
        if self.last is not None:
            t0, x0, y0 = self.last
            dx, dy = x - x0, y - y0
            step = math.hypot(dx, dy)
            # a step faster than the robot can drive is a pose jump (GNSS, relocalisation):
            # the altitude, the wheel ground and the drift curve start again
            jump = min(self.cfg["track_jump_m"],
                       self.cfg["track_jump_min_m"] + self.cfg["track_vmax_mps"] * max(0.0, t - t0))
            if step > jump or t - t0 > self.cfg["track_gap_s"] or t < t0:
                self.reset()
            elif step > 1e-4:
                M = np.array([[R[0, 0], R[0, 1]], [R[1, 0], R[1, 1]]])
                try:
                    st = np.linalg.solve(M, [dx, dy])
                except np.linalg.LinAlgError:
                    st = np.array([0.0, 0.0])
                self.h += st[0] * R[2, 0] + st[1] * R[2, 1]
                self.s += step
        self.last = (t, x, y)
        fwd = math.hypot(R[0, 0], R[1, 0])
        uphill = math.degrees(math.atan2(R[2, 0], fwd))     # +x pointing up = uphill
        self.hist.append((self.s, self.h, uphill, t))
        cfg = self.cfg
        while len(self.hist) > 2 and (
                self.hist[-1][0] - self.hist[0][0] > cfg["track_len_m"]
                or t - self.hist[0][3] > cfg["track_hist_s"]
                or len(self.hist) > cfg["track_hist_max"]):
            self.hist.pop(0)
        return self.h

    def uphill_now(self):
        return self.hist[-1][2] if self.hist else 0.0

    def uphill_smooth(self):
        """Terrain slope under the robot, median over the last slope_smooth_m of
        track (rocking of the body on tufts and bumps averages out)."""
        if not self.hist:
            return 0.0
        s_now, t_now = self.hist[-1][0], self.hist[-1][3]
        u = [hh[2] for hh in self.hist if s_now - hh[0] <= self.cfg["slope_smooth_m"]
             and t_now - hh[3] <= self.cfg["slope_smooth_s"]]
        return float(np.median(u))

    def slope_sigma_deg(self):
        """How unsure the forward extrapolation of the robot plane is."""
        base = self.cfg["slope_sigma_min_deg"]
        if len(self.hist) < 3:
            return base * 2.0
        p = np.array([h[2] for h in self.hist])
        return max(base, float(np.std(p)) + abs(float(p[-1] - np.median(p))) * 0.5)


def _solve_spd(M, v):
    """Solve M x = v for a small symmetric positive definite M (Cholesky in numpy
    element-wise operations: no LAPACK/BLAS threads for a 31 x 31 system)."""
    n = len(v)
    L = np.zeros_like(M)
    for j in range(n):
        d = M[j, j] - float((L[j, :j] * L[j, :j]).sum())
        L[j, j] = math.sqrt(max(d, 1e-18))
        if j + 1 < n:
            L[j + 1:, j] = (M[j + 1:, j] - (L[j + 1:, :j] * L[j, :j]).sum(axis=1)) / L[j, j]
    y = np.zeros(n)
    for j in range(n):
        y[j] = (v[j] - float((L[j, :j] * y[:j]).sum())) / L[j, j]
    x = np.zeros(n)
    for j in range(n - 1, -1, -1):
        x[j] = (y[j] - float((L[j + 1:, j] * x[j + 1:]).sum())) / L[j, j]
    return x


class DriftFit(object):
    """Drift curve D(s) of the integrated altitude along the track, D(s_now) = 0.
    A value computed at track position s carries D(s); bringing a value from s_a
    to the drift of s_b adds corr(s_a, s_b) = D(s_b) - D(s_a)."""

    def __init__(self, knots, d, s_min, sigma):
        self.knots, self.d = knots, d
        self.s_min = s_min          # oldest track position the revisits constrain
        self.sigma = sigma          # m, robust residual of the revisits
        self.rate = float((d[-1] - d[-2]) / (knots[-1] - knots[-2]))   # m per m, now

    def at(self, s):
        s = np.asarray(s, float)
        k = self.knots
        v = np.interp(s, k, self.d)
        # ahead of now (a cell in front of the robot): the current slope
        return np.where(s > k[-1], self.d[-1] + self.rate * (s - k[-1]), v)

    def corr(self, s_a, s_b):
        return self.at(s_b) - self.at(s_a)


class Drift(object):
    """Drift of the integrated altitude from wheel revisits.

    The IMU pitch is off while driving (blade vibration), so the integrated
    altitude grows with the track, not quite linearly (2.1 cm/m on 2026-11-08 with
    +-3 cm wander around the line over 15-45 m). Two wheel stamps of the same map
    cell at track positions s_old and s_new differ by D(s_new) - D(s_old) (plus the
    terrain inside the 0.2 m cell). D is a piecewise-linear curve with knots every
    drift_knot_m over the last drift_window_m, fitted to all such revisits by
    least squares with a curvature penalty, D(now) = 0, outliers down-weighted.
    Causal check on 2026-11-08: |error| median 15 mm at any span 3-70 m (a single
    rate: 30-37 mm at 15-45 m)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pairs = []             # (s_old, s_new, dh)
        self._cache = (None, None, 0)   # (s, fit, pairs added since)

    def reset(self):
        self.pairs = []
        self._cache = (None, None, 0)

    def add(self, ds, dh, s_new):
        if ds < self.cfg["drift_min_ds_m"]:
            return
        self.pairs.append((s_new - ds, s_new, dh))
        lim = s_new - self.cfg["drift_window_m"]
        cap = int(self.cfg["drift_max_pairs"])
        if len(self.pairs) > cap + 50 or self.pairs[0][0] < lim:
            self.pairs = [p for p in self.pairs if p[0] >= lim][-cap:]
        s0, fit, n = self._cache
        self._cache = (s0, fit, n + 1)

    def estimate(self, s_now):
        """DriftFit or None (too few revisits, implausible drift). Refitted when
        the robot moved drift_refit_m or drift_refit_pairs revisits came in (a few ms
        each);
        in between the last fit holds (D of now is then its short extrapolation)."""
        s0, fit, n = self._cache
        if s0 is not None and 0.0 <= s_now - s0 < self.cfg["drift_refit_m"] and n < self.cfg["drift_refit_pairs"]:
            return fit
        cfg = self.cfg
        lo = s_now - cfg["drift_window_m"]
        P = [p for p in self.pairs if p[0] >= lo and p[1] <= s_now + 1e-6]
        fit = None
        if len(P) >= cfg["drift_min_pairs"]:
            a = np.array(P)
            if (a[:, 1] - a[:, 0]).sum() >= cfg["drift_min_span_m"]:
                fit = self._fit(a, s_now)
        self._cache = (s_now, fit, 0)
        return fit

    def _fit(self, a, s_now):
        cfg = self.cfg
        step = cfg["drift_knot_m"]
        s0 = float(a[:, 0].min())
        n = max(2, int(math.ceil((s_now - s0) / step)) + 1)
        knots = s_now - step * np.arange(n)[::-1]

        # Each revisit is a sparse row: D(s_new) - D(s_old), two knots each. The
        # normal equations (n x n, n ~ 31) are summed with bincount and solved
        # small, with no BLAS call on the ~500-row system (a multi-threaded
        # OpenBLAS spun its threads there: 58 % of a core at 4 Hz).
        def basis(s):
            s = np.clip(s, knots[0], knots[-1])
            i = np.clip(((s - knots[0]) / step).astype(np.int64), 0, n - 2)
            f = (s - knots[i]) / step
            return i, f
        i_n, f_n = basis(a[:, 1])
        i_o, f_o = basis(a[:, 0])
        C = np.stack([i_n, i_n + 1, i_o, i_o + 1], axis=1)              # m x 4 columns
        V = np.stack([1.0 - f_n, f_n, f_o - 1.0, -f_o], axis=1)          # m x 4 values
        b = a[:, 2]
        pair_idx = (C[:, :, None] * n + C[:, None, :]).ravel()
        vv = (V[:, :, None] * V[:, None, :]).reshape(len(b), 16)
        reg = np.zeros((n, n))
        lam2 = cfg["drift_smooth"] ** 2
        for j in range(n - 2):                  # curvature penalty (1, -2, 1)
            for p_, cp in zip(range(j, j + 3), (1.0, -2.0, 1.0)):
                for q_, cq in zip(range(j, j + 3), (1.0, -2.0, 1.0)):
                    reg[p_, q_] += lam2 * cp * cq
        reg[-1, -1] += 100.0                    # D(now) = 0
        reg += 1e-9 * np.eye(n)
        w = np.ones(len(b))
        x = np.zeros(n)
        mad = 0.0
        for _ in range(3):
            w2 = w * w
            AtA = np.bincount(pair_idx, (vv * w2[:, None]).ravel(), n * n).reshape(n, n)
            Atb = np.bincount(C.ravel(), (V * (w2 * b)[:, None]).ravel(), n)
            x = _solve_spd(AtA + reg, Atb)
            r = b - (V * x[C]).sum(axis=1)
            mad = float(np.median(np.abs(r))) * 1.4826 + 1e-4
            w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * mad))
        fit = DriftFit(knots, x, s0, max(cfg["drift_sigma_min_m"], 1.25 * mad))
        if abs(fit.rate) > cfg["drift_max_rate"]:
            return None
        return fit


# --------------------------------------------------------------------------
# map layer grid
# --------------------------------------------------------------------------
class GrassGrid(wl._Grid):
    RES = GRASS_RES
    FILENAME = "grass.npz"
    MAX_CELLS = 750             # 150 m a side: 51 MB of arrays at full size, a garden fits
    LAYERS = {
        "last_mm": (np.int16, LAST_NONE, ()),
        "hist": (np.uint16, 0, (HIST_BINS,)),
        "count": (np.uint16, 0, ()),
        "last_t": (np.uint32, 0, ()),
        "source": (np.int8, SRC_NONE, ()),
    }

    def add(self, x, y, height_mm, t, source):
        if not self.ensure(x, y, x, y):
            return False
        c = int(math.floor((x - self.ox) / self.res))
        r = int(math.floor((y - self.oy) / self.res))
        hmm = int(round(max(0.0, min(30000.0, height_mm))))
        self.a["last_mm"][r, c] = hmm
        hist = self.a["hist"][r, c]
        hist[min(HIST_BINS - 1, hmm // HIST_STEP_MM)] += 1
        if hist.max() >= wl.HIST_CAP:
            hist //= 2
        cnt = self.a["count"][r, c]
        self.a["count"][r, c] = min(65535, int(cnt) + 1)
        self.a["last_t"][r, c] = int(t)
        self.a["source"][r, c] = source
        self.dirty = self.changed = True
        return True

    def median_mm(self):
        """Median height, centre of the median 10 mm bin; -1 = no data."""
        cnt = self.a["count"]
        out = np.full(cnt.shape, -1, np.int32)
        r, c = np.nonzero(cnt)          # only seen cells: no full-size temporaries
        if len(r):
            b = wl.GnssGrid._median_bin(self.a["hist"][r, c])
            out[r, c] = b * HIST_STEP_MM + HIST_STEP_MM // 2
        return out

    def _extra_arrays(self):
        return {"median_mm": self.median_mm().astype(np.int16),
                "hist_step_mm": np.array(HIST_STEP_MM)}


class ImuBuffer(object):
    """Roll/pitch by message stamp; at(t) interpolates between the two samples
    around t, None when the nearest is farther than max_gap_s (IMU missing)."""

    def __init__(self, size=400, max_gap_s=0.2):
        self.t = []
        self.rp = []
        self.size = size
        self.max_gap_s = max_gap_s
        import threading
        self._lock = threading.Lock()      # callback thread adds, frame loop reads

    def add(self, t, roll, pitch):
        with self._lock:
            if self.t and t <= self.t[-1]:
                return
            self.t.append(t)
            self.rp.append((roll, pitch))
            if len(self.t) > self.size:
                del self.t[0], self.rp[0]

    def at(self, t):
        with self._lock:
            return self._at(t)

    def _at(self, t):
        if not self.t:
            return None
        import bisect
        i = bisect.bisect_left(self.t, t)
        if i < len(self.t) and self.t[i] == t:
            return self.rp[i]
        if 0 < i < len(self.t):
            t0, t1 = self.t[i - 1], self.t[i]
            if t1 - t0 > 2 * self.max_gap_s:
                return None
            f = (t - t0) / (t1 - t0)
            (r0, p0), (r1, p1) = self.rp[i - 1], self.rp[i]
            return (r0 + f * (r1 - r0), p0 + f * (p1 - p0))
        j = 0 if i == 0 else len(self.t) - 1
        return self.rp[j] if abs(self.t[j] - t) <= self.max_gap_s else None


def save_merged(grid, site_dir, site, datum, journal, known_mtime):
    """Write grass.npz under an exclusive file lock. When the file on disk is not
    the one this grid was read from (another instance, a restart), it is read
    again and only `journal` (cells added since) is replayed onto it, so older
    data never overwrites newer. Returns (grid now in memory, new mtime)."""
    import fcntl
    path = os.path.join(site_dir, GrassGrid.FILENAME)
    with open(path + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            try:
                m = os.path.getmtime(path)
            except OSError:
                m = 0.0
            if m != known_mtime:
                grid = GrassGrid.load(site_dir, datum)
                for x, y, h, t, src in journal:
                    grid.add(x, y, h, t, src)
            grid.save(site_dir, site, datum)
            return grid, os.path.getmtime(path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


_GRASS = [("below_cut", "#22c55e", "up to cut + 2 cm"), ("+2-4cm", "#eab308", "+2-4 cm"),
          (">+4cm", "#f97316", "> +4 cm")]


def grass_layer(grid, site, now, cut_mm=None):
    """Cells coloured by the class of their median height against the cut height
    (the same classes as /grass/ahead). Each stop has a `key` for translation."""
    cut = DEFAULTS["cut_default_mm"] if cut_mm is None else float(cut_mm)
    legend = {"title": "Grass vs cut %d mm" % round(cut), "cut_mm": round(cut),
              "stops": [{"key": k, "color": c, "label": t} for k, c, t in _GRASS]}
    if grid is None or grid.h == 0:
        return wl.empty_layer(site, now, GRASS_RES, legend)
    med = grid.median_mm()
    valid = med >= 0
    exc = med - cut
    idx = np.where(exc < 20, 0, np.where(exc <= 40, 1, 2))
    rgba = wl._paint(idx, valid, [row[1] for row in _GRASS])
    return wl._layer_msg(site, now, grid.ox, grid.oy, grid.res, rgba, valid, legend)


def query_grass(grid, points=None, polygon=None):
    """Rows per point or stats of a polygon, worldlayers.layer_query style."""
    med = grid.median_mm()
    if points is not None:
        rows = []
        for x, y in points:
            rc = wl._cell(grid.ox, grid.oy, grid.res, grid.h, grid.w, x, y)
            if rc is None or med[rc] < 0:
                rows.append({"x": x, "y": y, "median_mm": None, "last_mm": None,
                             "count": 0, "last_t": None, "source": None})
            else:
                rows.append({"x": x, "y": y, "median_mm": int(med[rc]),
                             "last_mm": int(grid.a["last_mm"][rc]),
                             "count": int(grid.a["count"][rc]),
                             "last_t": int(grid.a["last_t"][rc]),
                             "source": SRC_NAMES.get(int(grid.a["source"][rc]))})
        return rows
    mask = wl._poly_mask(grid.ox, grid.oy, grid.res, grid.h, grid.w, polygon)
    vals = med[mask & (med >= 0)]
    return {"cells": int(mask.sum()), "cells_seen": int(vals.size),
            "median_mm": int(np.median(vals)) if vals.size else None,
            "p90_mm": int(np.percentile(vals, 90)) if vals.size else None}


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class GrassEngine(object):
    """Frame-by-frame logic of nodes/grass_height.

    frame(t, P_base, pose, rp, cut_mm, dem) -> /grass/ahead dict
      P_base  Nx3 points in base_link (m), raw (not yet zeroed)
      pose    (x, y, yaw) in map or None
      rp      (roll, pitch) rad of base_link against gravity (IMU, raw: the
              calibrated IMU pitch offset is removed here)
      dem     callable(xs, ys) -> absolute elevation (NaN unknown) or None
    on_cell(x, y, height_mm, t, source) is called for each resolved map cell.
    """

    def __init__(self, cfg=None, calib=None, on_cell=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.calib = calib or Calib()
        self.on_cell = on_cell or (lambda *a: None)
        self.track = Track(self.cfg)
        self.drift = Drift(self.cfg)
        self._epoch = self.track.epoch
        # (col, row) -> (alt, t, epoch, track s) ground under the wheels
        self.ground = {}
        # (col, row) -> [t_first, [canopy alts], [DEM offsets], epoch, track s and
        #                robot (x, y) at first sight]; a look counts only when it saw
        #                at least cell_min_cover of the cell (16 sub-squares)
        self.pending = {}
        self.recent = []            # (t, height_mm) for the smoothed output
        self.cut = self.cfg["cut_default_mm"]
        self.stats = {"resolved_wheels": 0, "resolved_dem": 0, "expired": 0,
                      "frames": 0, "uncertain": 0}

    # ---- helpers -------------------------------------------------------------
    def _cells(self, xs, ys):
        c = self.cfg["cell"]
        return np.floor(xs / c).astype(np.int64), np.floor(ys / c).astype(np.int64)

    def _centre(self, col, row):
        c = self.cfg["cell"]
        return (col + 0.5) * c, (row + 0.5) * c

    def corridor(self):
        x0 = self.calib.x_min if self.calib.x_min else self.cfg["x_min"]
        x1 = self.calib.x_max if self.calib.x_max else self.cfg["x_max"]
        return x0, x1

    def reset(self):
        """Robot stopped / asleep: altitudes of this epoch end."""
        self.track.reset()
        self._new_epoch()
        self.pending.clear()

    def _new_epoch(self):
        """The integrated altitude restarted: nothing measured on the old zero
        may meet anything measured on the new one."""
        self._epoch = self.track.epoch
        self.ground.clear()
        self.drift.reset()
        self.recent = []

    # ---- deferred resolution ---------------------------------------------------
    def _side(self, an):
        """Strips differing sharply side by side: (kind, low strips, high strips)
        with kind kerb (smooth raised), edge (mown | uncut grass) or terrain
        (the whole strip lifted: kerb, bed border, bank); None without a step."""
        strips = an["strips"]
        if len(strips) < 3:
            return None
        # A side slope (body rolled against the ground ahead) tilts the strips
        # evenly; a kerb or an edge is a step. Two models of the strip tops: a
        # line and two levels with the best split; a step only when it fits
        # clearly better and jumps more than lane_step_mm.
        ys = np.array([sv[0] for sv in strips])
        tops = np.array([sv[2] for sv in strips])
        n = len(tops)
        A = np.c_[np.ones(n), ys]
        res_line = tops - A @ np.linalg.lstsq(A, tops, rcond=None)[0]
        sse_line = float(res_line @ res_line)
        best = None
        for m in range(1, n):
            a, b_ = tops[:m], tops[m:]
            sse = float(((a - a.mean()) ** 2).sum() + ((b_ - b_.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, m, float(b_.mean() - a.mean()))
        sse_step, m, jump = best
        if abs(jump) <= self.cfg["lane_step_mm"] or sse_step > 0.5 * sse_line:
            return None
        high = strips[m:] if jump > 0 else strips[:m]
        low = [sv for sv in strips if sv not in high]
        # Both sides of the step are judged alike: a strip holding two heights sits
        # on the border (lawn | kerb, mown | uncut) and belongs to neither side, so
        # it is left out of the evenness, the kerb test and the low ground on
        # either side; each side needs at least one clean strip.
        lo_p, hi_p = self._pure(low), self._pure(high)
        if not lo_p or not hi_p:
            return ("terrain", low, high)
        # a clean step has two even sides; uneven sides (a bush, an object, rough
        # ground) are no edge and no kerb -> unknown
        if (np.std([sv[2] for sv in lo_p]) > self.cfg["side_even_mm"]
                or np.std([sv[2] for sv in hi_p]) > max(self.cfg["side_even_mm"], 0.25 * abs(jump))):
            return ("terrain", low, high)
        if all(sv[2] - sv[1] <= 1.5 * self.calib.rough_bare_mm for sv in hi_p):
            return ("kerb", low, high)
        if self._grows_from_low_ground(lo_p, hi_p):
            return ("edge", low, high)
        return ("terrain", low, high)

    def _pure(self, strips):
        """Strips holding one height (decile gap share <= strip_pure_gap)."""
        return [sv for sv in strips if sv[3] <= self.cfg["strip_pure_gap"]]

    def _grows_from_low_ground(self, low, high):
        """Taller strips are grass only if their bottom reaches down to the ground
        of the low strips (both lists clean strips: (y, p10, p90, gap, p60)).
        Measured from the low strips' bottom lo_bot, a tall strip's bottom (p10)
        must lie within edge_bottom_mm, or in the lower edge_bottom_frac of the
        strip's own span (p10 and p90 over lo_bot). Grass keeps its p10/p90 share
        whatever its height (blades of all lengths), so this holds for 13 cm as
        for 25 cm grass and beside bare ground (the old reference, a quarter of
        the step over the low carpet, failed above ~15 cm and beside bare
        ground); a kerb, a bed border or a bank lifts the bottom with the top,
        and its share goes to 1."""
        lo_bot = float(np.median([sv[1] for sv in low]))
        frac = self.cfg["edge_bottom_frac"]
        # a flat top (p60..p90 tight) is a kerb or bank top, not grass blades
        return bool(high) and all(
            (sv[1] - lo_bot <= self.cfg["edge_bottom_mm"]
             or sv[1] - lo_bot <= frac * (sv[2] - lo_bot))
            and (sv[2] - sv[4]) >= self.cfg["edge_top_rough"] * max(1e-6, sv[2] - sv[1])
            for sv in high)

    def _bias_deg(self):
        return self.cfg["imu_bias_measured_deg" if self.calib.imu_offset_measured
                        else "imu_bias_unmeasured_deg"]

    def _resolve(self, key, raw, t, source, sigma_mm=15.0):
        self.pending.pop(key, None)
        h = max(0.0, self.calib.height(raw))        # the result only, never the terrain
        # the layer is read as classes: a class the uncertainty does not carry is not written
        if class_probability(h, math.hypot(sigma_mm, self.calib.noise_mm), self.cut) < self.cfg["conf_min"]:
            self.stats["uncertain"] += 1
            return
        x, y = self._centre(*key)
        self.on_cell(x, y, h, t, source)
        self.stats["resolved_wheels" if source == SRC_WHEELS else "resolved_dem"] += 1

    def _stamp_wheels(self, t, x, y, R, h):
        pts = np.array([[wx, wy, 0.0] for wx, wy in BODY_XY]) @ R.T
        cols, rows = self._cells(pts[:, 0] + x, pts[:, 1] + y)
        ep, s_now = self.track.epoch, self.track.s
        drift = self.drift.estimate(s_now)
        done = set()
        for col, row, dz in zip(cols, rows, pts[:, 2]):
            key = (int(col), int(row))
            if key in done:
                continue                # one stamp per cell and frame (the first, front-most)
            done.add(key)
            old = self.ground.get(key)
            if old is not None and old[2] == ep:
                # the wheels are back on a cell they stamped earlier: the altitude
                # difference over the track between is the drift
                self.drift.add(s_now - old[3], h + dz - old[0], s_now)
            self.ground[key] = (h + dz, t, ep, s_now)
            obs = self.pending.get(key)
            # the altitude is integrated: trust it over a few metres, or as far as
            # the drift curve reaches
            lim = self.cfg["wheel_max_track_m"]
            if drift and obs is not None and obs[4] >= drift.s_min:
                lim = max(lim, self.cfg["drift_use_max_m"])
            if obs is not None and obs[3] == ep and s_now - obs[4] <= lim:
                raw = float(np.median(obs[1])) - (h + dz)
                # An IMU pitch error cancels when the robot drove straight from the
                # look to the cell (levelling and integration tilt alike); what
                # the path wandered off the straight line carries it: removed with
                # the drift curve when known (its uncertainty stays), else a sigma.
                # The tops carry the drift of s_look + direct (where straight driving
                # would have stamped the cell), the ground that of now.
                cx, cy = self._centre(*key)
                direct = math.hypot(cx - obs[5][0], cy - obs[5][1])
                wander = max(0.0, s_now - obs[4] - direct) + 0.1
                if drift and s_now - obs[4] > self.cfg["wheel_max_track_m"]:
                    raw -= float(drift.corr(s_now, obs[4] + direct))
                    sig = 5.0 + 1000.0 * drift.sigma
                else:
                    sig = 5.0 + 1000.0 * wander * math.tan(math.radians(self._bias_deg()))
                if self.cfg["z_lo"] < raw < self.cfg["z_hi"]:
                    self._resolve(key, raw * 1000.0, t, SRC_WHEELS, sig)
                else:
                    self.pending.pop(key, None)
        if len(self.ground) > self.cfg["ground_max"]:
            self._prune_ground()

    def _prune_ground(self):
        lim = max(self.cfg["wheel_keep_m"], self.cfg["wheel_max_track_m"])
        s_now, ep = self.track.s, self.track.epoch
        for k in [k for k, g in self.ground.items() if g[2] != ep or s_now - g[3] > lim]:
            del self.ground[k]
        if len(self.ground) > self.cfg["ground_max"]:          # standing still for ages
            keep = sorted(self.ground.items(), key=lambda kv: kv[1][1])[-self.cfg["ground_max"] // 2:]
            self.ground = dict(keep)

    def _wheel_ground(self, keys, drift=None):
        """(altitude, track distance since stamped) of cells the wheels know in
        this epoch within wheel_ahead_max_m (drift_use_max_m with a drift rate);
        NaN elsewhere."""
        ep, s_now = self.track.epoch, self.track.s
        lim = self.cfg["wheel_ahead_max_m"]
        if drift:
            lim = min(max(lim, self.cfg["drift_use_max_m"]), s_now - drift.s_min)
            lim = max(lim, self.cfg["wheel_ahead_max_m"])
        alt = np.full(len(keys), np.nan)
        dist = np.full(len(keys), np.nan)
        for i, k in enumerate(keys):
            g = self.ground.get(k)
            if g is not None and g[2] == ep and s_now - g[3] <= lim:
                alt[i], dist[i] = g[0], s_now - g[3]
        return alt, dist

    def _partial_ground(self, gw, kn, cxy, fwd_d, pose, R, h):
        """Ground of a corridor the wheels know only partly (the previous lane on
        one side): a plane through the robot's own footprint (the four contacts,
        now) and the known cells ahead, robust. Returns (ground of every cell with
        the known ones kept, extra sigma mm) or None when the known cells do not
        lie on one plane (terrain bends) within wheels_partial_resid_mm."""
        cfg = self.cfg
        yaw = pose[2]
        lat = -(cxy[:, 0] - pose[0]) * math.sin(yaw) + (cxy[:, 1] - pose[1]) * math.cos(yaw)
        foot = np.array([[wx, wy, 0.0] for wx, wy in BODY_XY])
        fz = h + (foot @ R.T)[:, 2]
        d = np.r_[foot[:, 0], fwd_d[kn]]
        l = np.r_[foot[:, 1], lat[kn]]
        z = np.r_[fz, gw[kn]]
        w = np.r_[np.full(len(foot), 1.0), np.full(int(kn.sum()), 3.0)]
        a, b, c, r = _robust_plane(d, l, z, w=w)
        rk = r[len(foot):]
        resid = 1000.0 * float(np.sqrt(np.median(rk ** 2)))
        if resid > cfg["wheels_partial_resid_mm"]:
            return None
        plane = a + b * fwd_d + c * lat
        g = np.where(kn, gw, plane)
        # reach sideways beyond the known cells: a side slope change of up to
        # drift_deg*4 (2 deg) over that distance is not seen
        lo, hi = lat[kn].min(), lat[kn].max()
        gap = float(np.max(np.maximum(0.0, np.maximum(lo - lat, lat - hi))))
        extra = math.hypot(resid, 1000.0 * gap * math.tan(math.radians(4.0 * cfg["drift_deg"])))
        return g, extra

    def _dem_offset(self, dem, x, y, R, h):
        """Track altitude minus DEM elevation under the robot now (so that
        ground_alt = dem(cell) + offset), None without DEM under the wheels.
        Taken per frame: the DEM is used relative to where the robot stands
        when it looks, so no integrated altitude enters a DEM result."""
        if dem is None:
            return None
        foot = np.array([[wx, wy, 0.0] for wx, wy in WHEELS_XY]) @ R.T
        under = dem(foot[:, 0] + x, foot[:, 1] + y)
        if np.isfinite(under).sum() < 2:
            return None
        return h - float(np.nanmedian(under - foot[:, 2]))

    def _dem_cells(self, dem, keys, x=None, y=None, base=None):
        """DEM elevation of cells, NaN where unknown; with the robot position
        and the DEM under it also NaN where the DEM is implausibly steep."""
        cx = np.array([self._centre(*k)[0] for k in keys])
        cy = np.array([self._centre(*k)[1] for k in keys])
        z = np.array(dem(cx, cy), float)
        if base is not None:
            d = np.hypot(cx - x, cy - y)
            lim = d * math.tan(math.radians(self.cfg["dem_max_slope_deg"])) + 0.05
            z[np.abs(z - base) > lim] = np.nan
        return z

    def expire(self, now, dem=None):
        ttl = self.cfg["pending_ttl_s"]
        old = [k for k, v in self.pending.items() if now - v[0] > ttl or v[3] != self.track.epoch]
        if not old:
            return
        if dem is not None:
            z = self._dem_cells(dem, old)
            for k, zc in zip(old, z):
                obs = self.pending[k]
                if not np.isfinite(zc):
                    continue
                tops = np.array(obs[1])
                offs = np.array(obs[2], float)
                ok = np.isfinite(offs)
                if ok.sum() == 0:
                    continue
                # tops and offsets share the altitude zero of their own frame
                raw = float(np.median(tops[ok] - (zc + offs[ok])))
                if self.cfg["z_lo"] < raw < self.cfg["z_hi"]:
                    self._resolve(k, raw * 1000.0, now, SRC_DEM)
        for k in old:
            if k in self.pending:
                del self.pending[k]
                self.stats["expired"] += 1

    def flush_pending(self, now, dem=None):
        """Robot goes to sleep: what the DEM can resolve is resolved, rest dropped."""
        for v in self.pending.values():
            v[0] = -1e18
        self.expire(now, dem)

    # ---- one frame --------------------------------------------------------------
    def frame(self, t, P_base, pose=None, rp=(0.0, 0.0), cut_mm=None, dem=None, rp_imu=True):
        """rp_imu False: roll/pitch came from TF (IMU missing), no IMU offset removed."""
        cfg = self.cfg
        self.stats["frames"] += 1
        cut = cfg["cut_default_mm"] if cut_mm is None else float(cut_mm)
        self.cut = cut
        P = self.calib.zero(P_base)
        x0, x1 = self.corridor()
        out = {"stamp": round(t, 3), "canopy_mm": None, "ground_mm": None, "height_mm": None,
               "class": "unknown", "terrain_source": None, "slope_deg": None,
               "confidence": 0.0, "uncertainty_mm": None, "reason": "",
               "corridor_m": [x0, x1], "cut_mm": cut, "calibrated": self.calib.calibrated}
        roll = rp[0]
        pitch = rp[1] - (math.radians(self.calib.imu_pitch_offset_deg) if rp_imu else 0.0)
        yaw = pose[2] if pose is not None else 0.0
        R = rot_rpy(roll, pitch, yaw)
        h = 0.0
        if pose is not None:
            h = self.track.update(t, pose[0], pose[1], R)
            if self.track.epoch != self._epoch:
                self._new_epoch()
            self._stamp_wheels(t, pose[0], pose[1], R, h)

        def done(o):
            if not o["reason"] and not self.calib.calibrated:
                o["reason"] = "uncalibrated"
            self.expire(t, dem)
            return o

        dem_off = self._dem_offset(dem, pose[0], pose[1], R, h) if pose is not None else None
        an = analyze_corridor(P, cfg, x0, x1)
        if an is None:
            out["reason"] = "no_data"
            return done(out)
        side = self._side(an)
        # beside raised terrain (kerb, border, bank) a cell holding two heights is
        # dropped readily; elsewhere only a clear split
        split_gap = cfg["cell_split_gap_terrain"] if side and side[0] == "terrain" else cfg["cell_split_gap"]
        xc = an["x_c"]
        smooth = an["rough_mm"] <= self.calib.rough_bare_mm
        low = an["low_lane_slope_deg"]
        out.update(rough_mm=round(an["rough_mm"], 1), roll_rel_deg=round(an["roll_rel_deg"], 2),
                   low_lane_slope_deg=None if low is None else round(low, 2),
                   patch_mm=round(an["patch_mm"], 1), slope_rel_deg=round(an["slope_rel_deg"], 2),
                   valid_frac=round(an["valid_frac"], 2))

        # ---- map cells of this frame (world, track altitude) ----
        keys, canopy_alt = [], []
        m = ((P[:, 0] >= x0) & (P[:, 0] < x1) & (np.abs(P[:, 1]) < cfg["half_width"])
             & (P[:, 2] > cfg["z_lo"]) & (P[:, 2] < cfg["z_hi"]))
        if pose is not None and m.any():
            W = P[m] @ R.T
            W[:, 0] += pose[0]
            W[:, 1] += pose[1]
            W[:, 2] += h
            cols, rows = self._cells(W[:, 0], W[:, 1])
            c0, r0 = cols.min(), rows.min()
            ck = (cols - c0) * 100000 + (rows - r0)
            order = np.lexsort((W[:, 2], ck))
            k_s, zs = ck[order], W[order, 2]
            starts = np.flatnonzero(np.r_[True, k_s[1:] != k_s[:-1]])
            counts = np.diff(np.r_[starts, len(k_s)])
            gk = k_s[starts]
            tops = _group_pct(zs, starts, counts, cfg["canopy_pct"])
            p10c = _group_pct(zs, starts, counts, 10.0)
            spread = tops - p10c
            gap = _decile_gap(zs, starts, counts)
            # a flat bottom under a much higher top: lawn and a kerb or bank top
            # (with the kerb's face between). A cell straddling a mown | uncut
            # edge looks the same and is dropped too: the cells on either side
            # carry the edge, a wrong kerb-top "grass" cell would not be undone.
            with np.errstate(invalid="ignore", divide="ignore"):
                flat_bottom = (_group_pct(zs, starts, counts, 40.0) - p10c) / spread
            two_level = ((spread > cfg["cell_split_mm"] / 1000.0)
                         & (flat_bottom < cfg["cell_flat_bottom"]))
            # which quarter-of-a-quarter of the cell the points cover (16 bits)
            q = cfg["cell"] / 4.0
            sx = np.clip(np.floor((W[order, 0] - cols[order] * cfg["cell"]) / q), 0, 3).astype(np.int64)
            sy = np.clip(np.floor((W[order, 1] - rows[order] * cfg["cell"]) / q), 0, 3).astype(np.int64)
            cover = np.bitwise_or.reduceat(np.left_shift(1, sx * 4 + sy), starts)
            for g, n, top, sp, gp, cv, tl in zip(gk, counts, tops, spread, gap, cover, two_level):
                if n < cfg["cell_min_pts"]:
                    continue
                if bin(int(cv)).count("1") < 16 * cfg["cell_min_cover"]:
                    # seen only at a corner (the corridor's edge, a turn sweeping
                    # over it): this look's p90 is not the cell's
                    self.stats["partial"] = self.stats.get("partial", 0) + 1
                    continue
                if (sp > cfg["cell_split_mm"] / 1000.0 and gp > split_gap) or tl:
                    self.stats["split"] = self.stats.get("split", 0) + 1
                    continue           # two heights in one cell (kerb, bank edge): no answer
                key = (int(g // 100000 + c0), int(g % 100000 + r0))
                top = float(top)
                keys.append(key)
                canopy_alt.append(top)
                off = np.nan if dem_off is None else dem_off
                ob = self.pending.get(key)
                if (ob is None or ob[3] != self.track.epoch
                        or self.track.s - ob[4] > cfg["wheel_max_track_m"]):
                    # first look, or a look from a later pass (the next lane): a
                    # fresh observation, the old pass did not get its wheels in time
                    # and its tops sit on another drift of the altitude
                    if ob is not None or len(self.pending) < cfg["pending_max"]:
                        self.pending[key] = [t, [top], [off], self.track.epoch, self.track.s,
                                             (pose[0], pose[1])]
                else:
                    ob[1].append(top)
                    ob[2].append(off)
                    if len(ob[1]) > 40:
                        del ob[1][0], ob[2][0]

        # ---- ground of the corridor: wheels, DEM, extrapolation ----
        source, ground_rel, sigma_g, slope_deg = None, None, None, None
        xc_c = an["x_c"]
        if keys:
            ca = np.array(canopy_alt)
            drift = self.drift.estimate(self.track.s)
            gw, gdist = self._wheel_ground(keys, drift)
            cxy_all = np.array([self._centre(*k) for k in keys])
            fwd_d = ((cxy_all[:, 0] - pose[0]) * math.cos(yaw)
                     + (cxy_all[:, 1] - pose[1]) * math.sin(yaw))
            if drift:
                # the carpet of a cell d ahead carries the drift of s_now + d, its
                # ground the drift of when it was stamped: bring the ground over
                s_now = self.track.s
                gw = np.where(np.isfinite(gw), gw + drift.corr(s_now - np.nan_to_num(gdist), s_now + fwd_d), np.nan)
            bad = ~((ca - gw > cfg["z_lo"]) & (ca - gw < cfg["z_hi"]))
            gw[bad] = np.nan
            frac_w = np.isfinite(gw).mean()
            gd = None
            if dem_off is not None:
                gd = self._dem_cells(dem, keys, pose[0], pose[1], h - dem_off) + dem_off
                gd[~((ca - gd > cfg["z_lo"]) & (ca - gd < cfg["z_hi"]))] = np.nan
            frac_d = np.isfinite(gd).mean() if gd is not None else 0.0
            pick = None
            kn = np.isfinite(gw)
            partial = None
            if (frac_w < cfg["wheels_min_frac"] and kn.sum() >= cfg["wheels_partial_min_cells"]
                    and fwd_d[kn].max() >= x0 + 0.3 * (x1 - x0)):
                partial = self._partial_ground(gw, kn, cxy_all, fwd_d, pose, R, h)
            if frac_w >= cfg["wheels_min_frac"] or partial is not None:
                ok = kn
                pick, source = gw, "wheels"
                # the integrated altitude drifts with the track between the two looks
                span = float(np.median(gdist[ok] + fwd_d[ok]))
                out["ground_track_m"] = round(span, 1)
                if drift:
                    out["drift_rate"] = round(drift.rate, 4)
                    # short spans hardly drift, long ones carry the curve's error
                    sigma_g = 5.0 + 1000.0 * min(drift.sigma, abs(span) * math.tan(
                        math.radians(max(self.cfg["drift_deg"], self._bias_deg()))))
                else:
                    sigma_g = 5.0 + 1000.0 * (float(np.median(gdist[ok])) + xc_c) * math.tan(
                        math.radians(max(self.cfg["drift_deg"], self._bias_deg())))
                if partial is not None:
                    # the rest of the corridor from the plane through the footprint
                    # and the known cells; its fit and the reach sideways add up
                    pick, extra = partial
                    sigma_g = math.hypot(sigma_g, extra)
                    out["ground_cells_frac"] = round(float(frac_w), 2)
            elif frac_d >= cfg["wheels_min_frac"]:
                pick, source, sigma_g = gd, "dem", 15.0
            if pick is not None:
                ok = np.isfinite(pick)
                raw = float(np.median((ca[ok] - pick[ok]) * 1000.0))
                ground_rel = float(np.median(pick[ok]) - h) * 1000.0
                canopy_rel = float(np.median(ca[ok]) - h) * 1000.0
                cxy = np.array([self._centre(*k) for k in keys])[ok]
                d = (cxy[:, 0] - pose[0]) * math.cos(yaw) + (cxy[:, 1] - pose[1]) * math.sin(yaw)
                if np.ptp(d) > 0.3:
                    k = np.polyfit(d, pick[ok], 1)
                    slope_deg = math.degrees(math.atan(k[0]))
                    # Ground from long ago (drift curve) must follow a terrain change
                    # the carpet shows: if the carpet tilts against the body and the
                    # "grass" over that ground grows along the corridor by more than
                    # the terrain-change slope, the old ground does not fit here
                    # (2026-11-08: 4 frames of 120-155 mm into a dip, layer <= 80).
                    # Checked where it would claim grass over the cut: a wrong low
                    # ground there makes false tall grass (below the cut it agreed
                    # with the layer: |error| > 2 sigma in 1-2 %)
                    if (source == "wheels" and out.get("ground_track_m", 0.0) > cfg["wheel_ahead_max_m"]
                            and abs(an["slope_rel_deg"]) > cfg["slope_change_deg"]
                            and classify(self.calib.height(raw), cut) != "below_cut"):
                        kh = np.polyfit(d, ca[ok] - pick[ok], 1)[0]
                        if abs(math.degrees(math.atan(kh))) > cfg["slope_change_deg"]:
                            out.update(terrain_source=source, slope_deg=round(slope_deg, 1),
                                       reason="terrain_change")
                            return done(out)
        if source is None:
            # Extrapolation. The terrain ahead keeps the SMOOTHED slope of the
            # track, not the body's instantaneous plane (the body rocks on tufts).
            # delta = that slope against the body, both from the IMU: its bias cancels.
            if pose is not None:
                u_s, u_b = self.track.uphill_smooth(), self.track.uphill_now()
            else:
                u_b = math.degrees(math.atan2(-math.sin(pitch), math.cos(pitch)))
                u_s = u_b
            # Basis: the smoothed track slope, unless the carpet AND its lowest lane
            # sit with the body's own plane (the body is already on the new slope,
            # the track median lags behind, e.g. climbing onto a rise).
            delta = u_s - u_b
            if (abs(an["slope_rel_deg"]) <= cfg["body_agree_deg"]
                    and (low is None or abs(low) <= cfg["body_agree_deg"])
                    and abs(an["slope_rel_deg"]) < abs(an["slope_rel_deg"] - delta)):
                delta = 0.0
            rel = an["slope_rel_deg"] - delta
            low_rel = None if low is None else low - delta
            # terrain change: the carpet tilts against the expected ground AND its
            # lowest lane tilts too (a grass edge raises only the tall lanes)
            tilted = (abs(rel) > cfg["slope_change_deg"]
                      and (low_rel is None or abs(low_rel) > cfg["slope_change_deg"]))
            g_body = xc * math.tan(math.radians(delta))
            # The lowest lane's tilt is terrain ONLY when that lane is bare ground
            # (smooth and on the ground): a gentle ramp or dip at the front wheels,
            # signed. A rough lane tilting can be grass growing taller ahead: it
            # never lowers the height, it only widens the uncertainty.
            extra_sigma = 0.0
            lowL = an.get("low_lane")
            if low_rel is not None and not tilted:
                corr = (xc - cfg["pivot_x"]) * math.tan(math.radians(low_rel))
                low_bare = (lowL is not None and lowL[2] <= self.calib.rough_bare_mm
                            and abs(lowL[1] - g_body * 1000.0 - corr * 1000.0) <= cfg["bare_offset_mm"])
                if low_bare:
                    g_body += corr
                else:
                    extra_sigma = abs(corr) * 1000.0
                    if corr < 0.0:
                        g_body += corr        # a dip may hide tall grass: count it in, never out
            fwd = (R if pose is not None else rot_rpy(roll, pitch)) @ np.array([xc, 0.0, 0.0])
            ground_rel = fwd[2] * 1000.0 + g_body * 1000.0
            slope_deg = u_b + delta + (low_rel if low_rel is not None and not tilted and not extra_sigma else 0.0)
            canopy = an["canopy_mm"]
            # kerb, bed border or mown | uncut edge beside the track: strips differ
            # sharply side by side
            strips = an["strips"]
            edge = False
            if side is not None:
                kind, low_s, high = side
                # each side's level from its clean strips (the border strip holds both)
                low_s, high = self._pure(low_s) or low_s, self._pure(high) or high
                lo_top = float(np.median([sv[2] for sv in low_s]))
                if kind == "kerb":
                    # smooth raised strips = kerb/pavement: the low strips carry the grass
                    canopy = lo_top
                    out["kerb"] = True
                elif kind == "edge":
                    # mown | uncut edge: each strip on its own, the answer is the
                    # tall part; the robot mows along such edges
                    edge = True
                    sh = [max(0.0, self.calib.height(sv[2] - g_body * 1000.0)) for sv in strips]
                    hi_h = max(0.0, self.calib.height(float(np.median([sv[2] for sv in high]))
                                                      - g_body * 1000.0))
                    lo_h = max(0.0, self.calib.height(lo_top - g_body * 1000.0))
                    out.update(lane_edge=True, lanes_mm=[round(v) for v in sh],
                               max_lane_mm=round(hi_h), min_lane_mm=round(lo_h),
                               min_lane_class=classify(lo_h, cut))
                    canopy = float(np.median([sv[2] for sv in high]))
                else:
                    # the whole strip is lifted (kerb, bed border, bank): terrain
                    # beside the track, its height is not grass -> unknown
                    out.update(terrain_source="extrapolated", lane_step=True,
                               canopy_mm=round(fwd[2] * 1000.0 + an["canopy_mm"]),
                               reason="terrain_change")
                    return done(out)
            source = "extrapolated"
            raw = canopy - g_body * 1000.0
            if smooth and abs(raw) <= cfg["bare_offset_mm"] and not edge:
                source, raw = "surface", 0.0      # the surface seen is the ground
            out["surface"] = "bare" if source == "surface" else "carpet"
            sigma_g = math.hypot(3.0 + xc * math.tan(math.radians(self.track.slope_sigma_deg())) * 1000.0,
                                 extra_sigma)
            if tilted and source != "surface":
                out.update(terrain_source="extrapolated", slope_deg=round(u_s + rel, 1),
                           canopy_mm=round(fwd[2] * 1000.0 + an["canopy_mm"]),
                           reason="terrain_change")
                return done(out)
            canopy_rel = fwd[2] * 1000.0 + canopy

        sigma = math.sqrt(sigma_g ** 2 + self.calib.noise_mm ** 2 + (an["patch_mm"] * 0.5) ** 2)
        height = max(0.0, self.calib.height(raw)) if source != "surface" else 0.0
        self.recent = [(tt, hh) for tt, hh in self.recent if t - tt <= cfg["smooth_window_s"]]
        self.recent.append((t, height))
        self.recent = self.recent[-cfg["smooth_frames"]:]
        hs = float(np.median([hh for _tt, hh in self.recent]))
        conf = class_probability(hs, sigma, cut)
        out.update(canopy_mm=round(canopy_rel), ground_mm=round(ground_rel),
                   height_mm=round(hs), raw_mm=round(raw, 1), terrain_source=source,
                   slope_deg=round(slope_deg, 1) if slope_deg is not None else None,
                   confidence=round(conf, 2), uncertainty_mm=round(sigma, 1))
        if conf < cfg["conf_min"]:
            out.update(height_mm=None, reason="low_confidence")
        else:
            out["class"] = classify(hs, cut)
        return done(out)


# --------------------------------------------------------------------------
# calibration fit
# --------------------------------------------------------------------------
def latest_records(records):
    """Only the newest record of each label and distance (a remeasure replaces)."""
    keep = {}
    for r in sorted(records, key=lambda r: (r.get("created") or "", r.get("_file") or "")):
        yaw = r.get("yaw_deg")
        # concrete turned around is a second record on purpose (IMU offset)
        head = int(round(yaw / 90.0)) % 4 if r["label"] == "beton" and yaw is not None else 0
        keep[(r["label"], round(float(r["x"]), 1), head)] = r
    return list(keep.values())


IMU_STEADY_MAX_DEG = 0.3      # p90-p10 of a record's IMU pitch; more = the IMU was not steady


def imu_steady(r):
    """A record's IMU pitch is usable: present and steady over its frames (a robot
    standing still reads within ~0.1 deg; on 2026-11-08 one record swung +-2 deg
    while its depth strips did not move at all: an IMU fault, not a tilt)."""
    if r.get("imu_pitch_deg") is None:
        return False
    spread = r.get("imu_pitch_spread_deg")
    return spread is None or spread <= IMU_STEADY_MAX_DEG


def fit_calibration(records, noise_max_mm=5.0, valid_min=0.5, allow_single_direction=False):
    """records: dicts written by grass_calib record (label, true_mm, x, plane,
    strips, frames). Returns (Calib, report dict). Raises ValueError when
    concrete is missing."""
    warnings = []
    for r in records:
        if r.get("imu_pitch_deg") is not None and not imu_steady(r):
            warnings.append("IMU pitch unsteady during %s (spread %.1f deg, robot still?): the "
                            "record is left out (no scale, no IMU offset): record it again"
                            % (r.get("_file") or r["label"], r["imu_pitch_spread_deg"]))
    beton = [r for r in records if r["label"] == "beton"]
    if not beton:
        raise ValueError("no beton record: concrete gives the camera zero")
    # 1) camera zero: the plane seen on concrete, all concrete records together
    xs, ys, zs = [], [], []
    for r in beton:
        for s in r["strips"]:
            if s.get("median_mm") is None:
                continue
            xs.append(s["x"])
            ys.append(s.get("y", 0.0))
            zs.append(s["median_mm"] / 1000.0)
    if len(zs) < 3:
        raise ValueError("beton records have too few strips")
    xs, ys, zs = map(np.asarray, (xs, ys, zs))
    if np.ptp(ys) > 0.05:
        a, b, c, _r = _robust_plane(xs, ys, zs)
    else:
        b, a = np.polyfit(xs, zs, 1)
        c = 0.0
    cal = Calib()
    cal.offset_mm = float(a) * 1000.0
    cal.pitch_deg = math.degrees(math.atan(b))
    cal.roll_deg = math.degrees(math.atan(c))
    plane = lambda x, y=0.0: (a + b * x + c * y) * 1000.0   # noqa: E731

    # roughness and noise on concrete
    rough_b = [r["rough_mm"] for r in beton if r.get("rough_mm") is not None]
    noise = [s["noise_mm"] for r in beton for s in r["strips"]
             if s.get("noise_mm") is not None and 0.75 <= s["x"] <= 1.65]

    # IMU: body attitude on the concrete = reference. A lawn record stood on
    # another attitude: the level ground there appears at z = x tan(dpitch).
    pb = [r["imu_pitch_deg"] for r in beton if imu_steady(r)]
    p_ref = float(np.median(pb)) if pb else None
    # IMU pitch offset against the body: concrete both ways (yaw 180 deg apart),
    # the slope of the concrete cancels in the mean
    pairs = []
    for i, r1 in enumerate(beton):
        for r2 in beton[i + 1:]:
            if None in (r1.get("yaw_deg"), r2.get("yaw_deg")) or not (imu_steady(r1) and imu_steady(r2)):
                continue
            dy = abs((r1["yaw_deg"] - r2["yaw_deg"] + 180.0) % 360.0 - 180.0)
            if dy > 150.0:
                pairs.append(0.5 * (r1["imu_pitch_deg"] + r2["imu_pitch_deg"]))
    if pairs:
        cal.imu_pitch_offset_deg = float(np.median(pairs))
        cal.imu_offset_measured = True
    elif allow_single_direction:
        warnings.append("IMU pitch offset not measured (beton one way only): map cells "
                        "resolved by the wheels carry a %.1f deg uncertainty"
                        % DEFAULTS["imu_bias_unmeasured_deg"])
    else:
        raise ValueError("beton recorded in one direction only: turn the robot by 180 deg "
                         "on the same concrete and record beton again (IMU tilt error), "
                         "or fit --allow-single-direction")

    # 2) scale: raw = p90 above the concrete plane at the sample's strip. Concrete
    # is the point (0, 0): no free bias besides the camera zero.
    pts = []
    for r in records:
        if r["label"] == "beton":
            continue
        s = r.get("sample") or {}
        if s.get("p90_mm") is None:
            continue
        if r.get("imu_pitch_deg") is not None and not imu_steady(r):
            continue                 # the robot or the IMU moved: no sample for the scale
        raw = s["p90_mm"] - plane(s["x"])
        dp = 0.0
        if p_ref is not None and imu_steady(r):
            dp = r["imu_pitch_deg"] - p_ref
            raw -= s["x"] * math.tan(math.radians(dp)) * 1000.0
        pts.append((r["label"], float(r["true_mm"]), raw, r["x"], r.get("rough_mm"), dp))
    labels = sorted(set(p[0] for p in pts))
    den = sum(p[2] * p[2] for p in pts)
    k = sum(p[1] * p[2] for p in pts) / den if den > 0 else 1.0
    q = 0.0
    if len(labels) < 2:
        warnings.append("only %s: the scale is not checked by a second height" % (labels or ["beton"]))
    if not (0.2 < k < 5.0):
        raise ValueError("scale %.2f is not plausible: check the true heights and the samples" % k)
    resid = {}
    for lab in labels:
        e = [k * p[2] - p[1] for p in pts if p[0] == lab]
        resid[lab] = round(float(np.mean(e)), 1)
        if len(labels) >= 2 and abs(resid[lab]) > 10.0:
            warnings.append("%s is off the line by %+.1f mm: measure again (ruler, sample "
                            "in the strip, robot still)" % (lab, resid[lab]))
    cal.scale, cal.bias_mm = float(k), float(q)
    rb = float(np.median(rough_b)) if rough_b else 6.0
    grass_rough = [p[4] for p in pts if p[4] is not None]
    thr = max(4.0, rb * 1.5)
    if grass_rough:
        thr = min(thr, 0.5 * (rb + min(grass_rough))) if min(grass_rough) > rb else thr
    cal.rough_bare_mm = float(thr)
    cal.noise_mm = max(2.0, float(np.median(noise))) if noise else 5.0
    cal.calibrated = True
    cal.created = time.strftime("%Y-%m-%d %H:%M:%S")

    errors = []
    for lab, true, raw, x, _rg, dp in pts:
        est = k * raw + q
        errors.append({"label": lab, "x": x, "true_mm": true, "raw_mm": round(raw, 1),
                       "est_mm": round(est, 1), "err_mm": round(est - true, 1),
                       "dpitch_deg": round(dp, 2)})
    for r in beton:
        for s in r["strips"]:
            if s.get("median_mm") is None:
                continue
            errors.append({"label": "beton", "x": s["x"], "true_mm": 0.0,
                           "est_mm": round(s["median_mm"] - plane(s["x"], s.get("y", 0.0)), 1),
                           "err_mm": round(s["median_mm"] - plane(s["x"], s.get("y", 0.0)), 1)})

    # 3) recommended corridor: concrete strips with low noise and enough pixels
    by_x = {}
    for r in beton:                     # sensor noise: on concrete, nothing moves
        for s in r["strips"]:
            sig = s.get("sigma_mm", s.get("noise_mm"))
            if sig is None:
                continue
            by_x.setdefault(round(s["x"], 2), []).append((sig, s.get("valid") or 0.0))
    good = sorted(x for x, v in by_x.items()
                  if np.median([n for n, _ in v]) <= noise_max_mm
                  and np.median([f for _, f in v]) >= valid_min)
    rec = None
    if good:
        # the longest run of consecutive good strips, 0.1 m apart
        runs, cur = [], [good[0]]
        for x in good[1:]:
            if x - cur[-1] <= 0.11:
                cur.append(x)
            else:
                runs.append(cur)
                cur = [x]
        runs.append(cur)
        best = max(runs, key=lambda r: (len(r), r[-1]))
        lo, hi = best[0] - 0.05, best[-1] + 0.05
        if hi - lo >= 0.4:
            rec = [round(max(lo, 0.6), 2), round(min(hi, 1.8), 2)]   # holes past ~2 m
            cal.x_min, cal.x_max = rec
    report = {"calib": cal.to_dict(), "errors": errors, "recommended_corridor_m": rec,
              "noise_by_x": dict((str(x), round(float(np.median([n for n, _ in v])), 1))
                                 for x, v in sorted(by_x.items())),
              "rough_beton_mm": round(rb, 1), "residual_mm": resid, "warnings": warnings,
              "rough_grass_mm": [round(g, 1) for g in grass_rough]}
    return cal, report
