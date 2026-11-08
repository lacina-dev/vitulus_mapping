#!/usr/bin/env python3
"""
terrainfuse.py -- the logic of nodes/terrain_recorder (T30 D2), without rospy.

  decode_gridmap / build_gridmap   the /traversability/fuse message: GridMap +
                              layer fuse_meta (contract ~/vitulus/predavka/t30-kontrakt.md, v1)
  QualityGate                 own localisation gate: /gnss/navpvt (RTK FIXED,
                              hAcc), /nav_tf/loc_status (owner gps, gps_good,
                              heading_ok, anchors not frozen), freshness
  RecorderCore                served site + datum -> terrainlayers.TerrainStore
                              (site/<site>/terrain/), coverage.npz for "mowed",
                              sessions, flushing, statistics; driven by plain
                              calls (feed / poll / flush / close) with times

The node only wires ROS topics to these calls, so all of it is tested without a
ROS master (test/test_terrainfuse.py).
"""

import collections
import glob
import json
import math
import os
import secrets
import shutil
import tempfile
import time

import numpy as np
import yaml

from . import bundle
from . import geo
from . import terrainexport
from . import terrainlayers as tl

__all__ = ["FUSE_VERSION", "LAYERS", "decode_gridmap", "build_gridmap", "QualityGate",
           "RecorderCore", "DEFAULTS"]

FUSE_VERSION = 2          # meta index 12 since 2; version 1 (C1) is still accepted

# data layers of the message, besides fuse_meta (t30-kontrakt.md)
LAYERS = ("elevation", "variance", "points", "t_fused", "cost", "reason", "slope", "step", "roughness")

DEFAULTS = {
    # gate
    "min_fix_class": 4,          # worldlayers.fix_class: 4 = RTK fixed
    "max_hacc_mm": 30.0,
    "pvt_max_age": 1.5,          # s
    "loc_max_age": 3.0,          # s (loc_status is 1 Hz + edges)
    "gate_hold_s": 10.0,         # the gate must have been open over [stamp - hold, now] (a fresh RTK fix
                                 # pulls the pose for a while)
    # messages
    "max_age": 3.0,              # s: message stamp vs now
    # subscription (the local node publishes only while somebody listens)
    "subscribe_after_s": 2.0,    # gate open this long -> subscribe
    "unsubscribe_after_s": 30.0, # gate closed this long -> unsubscribe
    # site, files
    "site_poll_s": 5.0,
    "coverage_reload_s": 60.0,
    "flush_s": 60.0,
    "session_gap_s": 1800.0,     # no fused batch this long -> a new session (besides /mower_smach/busy)
    # smooth localisation corrections pulled into odom (set_pose): /odometry/odom vs the correction-free
    # wheel EKF over corr_window_s, in the body frame
    "corr_window_s": 1.0,
    "corr_xy_m": 0.03,
    "corr_yaw_deg": 0.5,
    "corr_settle_s": 3.0,        # a batch with a stamp in [event - window, event + settle] is not fused
    "corr_max_dt": 0.15,         # s: nearest samples of both odometries at least this close
    "corr_enforce": False,       # True = act on events (freshness boundary, drop unsettled / uncovered batches);
                                 # False (default until measured on the robot) = only count them in ~stats
    "batch_log": "",             # CSV of every message / fused batch (diagnostics, replays); "" = off
    # T31: control from the UI
    "settings_file": "",         # "" = ~/.vitulus/terrain_recorder.json: {"enabled": bool} (owner's choice)
    "export": True,              # write <site>/terrain_mosaic/ (a niced child process, from the stored tiles)
    "export_s": 600.0,           # at most this often while the map changes; also at the end of a program
                                 # (falling edge of /mower_smach/busy) and on rebuild_export / import / delete
    "export_nice": 19,
    "backup_keep": 5,            # terrain_backup/: the newest N are kept
    "export_timeout_s": 300.0,   # an export child running longer is killed (state.export error "timeout")
    "import_max_datum_m": 0.5,   # a mosaic datum further from the site datum needs "force": true
    "import_max_datum_deg": 1.0,
    "test_cmd_delay_s": 0.0,     # TESTS ONLY: long commands sleep this long (non-blocking state check)
    "token_s": 60.0,             # confirmation token lifetime (delete_map, import replace)
}


# --------------------------------------------------------------------------
# message: grid_map_msgs/GridMap with the layer fuse_meta (contract t30-kontrakt.md)
# --------------------------------------------------------------------------
def _arr(layer_data):
    """float32 numpy array of a Float32MultiArray's data (numpy_msg gives one, plain
    genpy a tuple)."""
    return np.asarray(layer_data, np.float32)


def decode_gridmap(msg):
    """Parse one /traversability/fuse GridMap (or any object with the same
    attributes). Returns a dict: win (terrainlayers.Window), pose (x, y, yaw) or
    None, stamp, robot (x, y, z, yaw), epoch, settling, z_correction. Raises
    ValueError with the reason for a refused message."""
    try:
        info = msg.info
        stamp = float(info.header.stamp.to_sec())
        res = float(info.resolution)
        lx, ly = float(info.length_x), float(info.length_y)
        cx, cy = float(info.pose.position.x), float(info.pose.position.y)
        names = list(msg.layers)
        data = msg.data
        outer, inner = int(msg.outer_start_index), int(msg.inner_start_index)
    except (AttributeError, TypeError, ValueError) as e:
        raise ValueError("field: %s" % e)
    if not (0.0 < res < 1.0):
        raise ValueError("resolution %s" % res)
    nx, ny = int(round(lx / res)), int(round(ly / res))
    if not (0 < nx <= 2000 and 0 < ny <= 2000):
        raise ValueError("size %dx%d" % (nx, ny))
    if outer != 0 or inner != 0:
        raise ValueError("start index %d/%d (the contract has 0)" % (outer, inner))
    missing = [k for k in LAYERS + ("fuse_meta",) if k not in names]
    if missing:
        raise ValueError("missing layers %s" % missing)
    if len(data) != len(names):
        raise ValueError("%d layers, %d data arrays" % (len(names), len(data)))
    meta = _arr(data[names.index("fuse_meta")].data)
    if meta.size < 12:
        raise ValueError("fuse_meta has %d values" % meta.size)
    version = float(meta[0])
    if not (math.isfinite(version) and version >= 1):
        raise ValueError("version %r" % meta[0])
    # version 2 (P2): index 12 = time of the last epoch event relative to the stamp (NaN = none)
    ev_rel = float(meta[12]) if version >= 2 and meta.size >= 13 else float("nan")
    meta = meta[:12].astype(np.float64)
    arrs = {}
    for k in LAYERS:
        a = _arr(data[names.index(k)].data)
        if a.size != nx * ny:
            raise ValueError("layer %s: %d values, expected %d" % (k, a.size, nx * ny))
        arrs[k] = np.ascontiguousarray(a.reshape(ny, nx)[::-1, ::-1])   # -> [iy, ix], row 0 = lowest y
    robot = tuple(float(v) for v in meta[7:11])
    pose = (float(meta[4]), float(meta[5]), float(meta[6])) if meta[3] > 0.5 else None
    if not all(math.isfinite(v) for v in (stamp, cx, cy) + robot):
        raise ValueError("non-finite number")
    if pose is not None and not all(math.isfinite(v) for v in pose):
        raise ValueError("non-finite map_from_odom")
    epoch = int(meta[1]) if math.isfinite(meta[1]) else 0
    settling = bool(meta[2] > 0.5)
    pts = np.nan_to_num(arrs["points"], nan=0.0)
    cost = np.where(np.isfinite(arrs["cost"]), arrs["cost"], -1.0).astype(np.float32)
    reason = np.nan_to_num(arrs["reason"], nan=0.0)
    win = tl.Window(cx - lx / 2.0, cy - ly / 2.0, res, arrs["elevation"], arrs["variance"], pts, cost,
                    reason, slope=arrs["slope"], step=arrs["step"], roughness=arrs["roughness"],
                    t_fused=stamp + arrs["t_fused"].astype(np.float64), stamp=stamp, epoch=epoch,
                    settling=settling)
    win.epoch_t = stamp + ev_rel if math.isfinite(ev_rel) else None
    return {"win": win, "pose": pose, "stamp": stamp, "robot": robot, "epoch": epoch,
            "settling": settling, "z_correction": float(meta[11]), "version": int(version),
            "epoch_t": win.epoch_t}


class _NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Stamp(object):
    def __init__(self, t):
        self.t = float(t)

    def to_sec(self):
        return self.t


def build_gridmap(stamp, origin, res, layers, map_from_odom, robot, epoch=0, settling=False,
                  z_correction=0.0, classes=None, epoch_rel=None, version=FUSE_VERSION):
    """A /traversability/fuse message as the C1 node builds it (reference for tests).

    layers: arrays [iy][ix] (row 0 = lowest y) for every key of LAYERS (NaN unknown,
    t_fused relative to stamp). classes: None -> plain objects (no ROS); or
    (GridMap, Float32MultiArray, MultiArrayDimension, rospy.Time) -> a real message."""
    ny, nx = np.asarray(layers["elevation"]).shape
    meta = np.full(nx * ny, np.nan, np.float32)
    mo = map_from_odom
    meta[:12] = [version, epoch, 1.0 if settling else 0.0, 0.0 if mo is None else 1.0,
                 0.0 if mo is None else mo[0], 0.0 if mo is None else mo[1], 0.0 if mo is None else mo[2],
                 robot[0], robot[1], robot[2], robot[3], z_correction]
    if version >= 2:
        meta[12] = np.nan if epoch_rel is None else epoch_rel
    flat = [np.ascontiguousarray(np.asarray(layers[k], np.float32)[::-1, ::-1]).reshape(-1) for k in LAYERS]
    flat.append(meta)
    names = list(LAYERS) + ["fuse_meta"]
    cx = origin[0] + nx * res / 2.0
    cy = origin[1] + ny * res / 2.0
    if classes is None:
        info = _NS(header=_NS(stamp=_Stamp(stamp), frame_id="odom"), resolution=res,
                   length_x=nx * res, length_y=ny * res,
                   pose=_NS(position=_NS(x=cx, y=cy, z=0.0)))
        return _NS(info=info, layers=names, basic_layers=[], outer_start_index=0, inner_start_index=0,
                   data=[_NS(data=a) for a in flat])
    GridMap, F32, Dim, Time = classes
    m = GridMap()
    m.info.header.stamp = Time.from_sec(stamp)
    m.info.header.frame_id = "odom"
    m.info.resolution = res
    m.info.length_x = nx * res
    m.info.length_y = ny * res
    m.info.pose.position.x = cx
    m.info.pose.position.y = cy
    m.info.pose.orientation.w = 1.0
    m.layers = names
    for a in flat:
        d = F32()
        d.layout.dim = [Dim("column_index", nx, nx * ny), Dim("row_index", ny, ny)]
        d.data = a
        m.data.append(d)
    return m


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
def fix_class(fix_type, flags):
    """Fix class from NAV-PVT fixType and flags (= worldlayers.fix_class)."""
    if not flags & 0x01 or fix_type < 2 or fix_type > 4:
        return 0
    carr = (flags >> 6) & 0x03
    if carr == 2:
        return 4
    if carr == 1:
        return 3
    if flags & 0x02:
        return 2
    return 1


class QualityGate(object):
    """Localisation good enough for the persistent map. Times are receive times
    on one clock (the node: rospy.get_time())."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.pvt = None          # (t, class, hacc_mm)
        self.loc = None          # (t, dict)
        self.last_bad = None     # last time seen closed
        self.open_since = None
        self.closed_since = None  # first time seen closed after being open (None: open / never seen)
        self.reason = "no data"

    def feed_pvt(self, now, fix_type, flags, hacc_mm):
        self.pvt = (now, fix_class(fix_type, flags), float(hacc_mm))
        self.check(now)

    def feed_loc(self, now, status):
        self.loc = (now, status if isinstance(status, dict) else {})
        self.check(now)

    def _why(self, now):
        c = self.cfg
        if self.pvt is None:
            return "no navpvt"
        t, cls, hacc = self.pvt
        if now - t > c["pvt_max_age"]:
            return "navpvt stale"
        if cls < c["min_fix_class"]:
            return "fix class %d" % cls
        if hacc > c["max_hacc_mm"]:
            return "hAcc %.0f mm" % hacc
        if self.loc is None:
            return "no loc_status"
        t, st = self.loc
        if now - t > c["loc_max_age"]:
            return "loc_status stale"
        if st.get("owner") != "gps":
            return "owner %s" % st.get("owner")
        if not st.get("gps_good", False):
            return "gps not good"
        if not st.get("heading_ok", False):
            return "heading not ok"
        if st.get("anchor_frozen") or st.get("heading_anchor_frozen"):
            return "anchor frozen"
        return None

    def check(self, now):
        """Evaluate now; returns True when open."""
        why = self._why(now)
        self.reason = why or "ok"
        if why is None:
            if self.open_since is None:
                self.open_since = now
            self.closed_since = None
            return True
        self.last_bad = now
        self.open_since = None
        if self.closed_since is None:
            self.closed_since = now
        return False

    def ok_for(self, stamp, now):
        """Open now and over [stamp - gate_hold_s, now]."""
        if not self.check(now):
            return False
        return self.open_since is not None and self.open_since <= stamp - self.cfg["gate_hold_s"]

    def open_for(self, now):
        """Seconds the gate has been open (0 when closed)."""
        return (now - self.open_since) if (self.check(now) and self.open_since is not None) else 0.0


# --------------------------------------------------------------------------
# smooth corrections
# --------------------------------------------------------------------------
def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _rel(p0, p1):
    """p1 in the body frame of p0 (x, y, yaw)."""
    c, s = math.cos(p0[2]), math.sin(p0[2])
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return c * dx + s * dy, -s * dx + c * dy, _wrap(p1[2] - p0[2])


class CorrectionDetector(object):
    """Corrections pulled smoothly into odom: the motion of /odometry/odom over
    corr_window_s (body frame) against the motion of a correction-free odometry
    (/odometry/ekf_wheel_odom, never set_pose'd) over the same interval. More than
    corr_xy_m / corr_yaw_deg apart = an event at that time."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.odom = collections.deque(maxlen=400)
        self.ref = collections.deque(maxlen=400)
        self.events = collections.deque(maxlen=200)
        self.evals = collections.deque(maxlen=600)      # times of successful comparisons
        self.last_eval = None
        self.n_events = 0
        self.last_diff = None

    @staticmethod
    def _at(hist, t, max_dt):
        if not hist:
            return None
        best = min(hist, key=lambda h: abs(h[0] - t)) if len(hist) < 8 else None
        if best is None:
            # histories are in time order: binary search
            lo, hi = 0, len(hist) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if hist[mid][0] < t:
                    lo = mid + 1
                else:
                    hi = mid
            cand = [hist[i] for i in (lo - 1, lo) if 0 <= i < len(hist)]
            best = min(cand, key=lambda h: abs(h[0] - t))
        return best if abs(best[0] - t) <= max_dt else None

    def feed_ref(self, t, x, y, yaw):
        if self.ref and t <= self.ref[-1][0]:
            if t < self.ref[-1][0] - 1.0:
                self.ref.clear()
            else:
                return
        self.ref.append((t, x, y, yaw))

    def feed_odom(self, t, x, y, yaw):
        """Returns True when this sample shows an event."""
        if self.odom and t <= self.odom[-1][0]:
            if t < self.odom[-1][0] - 1.0:
                self.odom.clear()
            else:
                return False
        self.odom.append((t, x, y, yaw))
        if self.last_eval is not None and t - self.last_eval < 0.1:
            return False
        self.last_eval = t
        c = self.cfg
        w = c["corr_window_s"]
        o0 = self._at(self.odom, t - w, c["corr_max_dt"])
        r0 = self._at(self.ref, t - w, c["corr_max_dt"])
        r1 = self._at(self.ref, t, c["corr_max_dt"])
        if o0 is None or r0 is None or r1 is None:
            return False
        self.evals.append(t)
        a = _rel(o0[1:], (x, y, yaw))
        b = _rel(r0[1:], r1[1:])
        dxy = math.hypot(a[0] - b[0], a[1] - b[1])
        dyaw = abs(math.degrees(_wrap(a[2] - b[2])))
        self.last_diff = (dxy, dyaw)
        if dxy > c["corr_xy_m"] or dyaw > c["corr_yaw_deg"]:
            self.events.append(t)
            self.n_events += 1
            return True
        return False

    def covered(self, stamp, half=0.5):
        """A comparison of both odometries exists within +-half s of the stamp (missing
        samples = not known to be settled: the caller fails closed)."""
        return any(abs(e - stamp) <= half for e in reversed(self.evals))

    def unsettled(self, stamp):
        """A batch with this stamp lies in [event - window, event + settle] of an event."""
        w, st = self.cfg["corr_window_s"], self.cfg["corr_settle_s"]
        return any(e - w <= stamp <= e + st for e in self.events)


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def read_served_site(root):
    try:
        with open(os.path.join(root, ".last_served")) as f:
            data = json.load(f)
        return str(data["site"]) if data.get("site") else None
    except (IOError, OSError, ValueError, KeyError, AttributeError, TypeError):
        return None


def read_datum(site_dir):
    path = os.path.join(site_dir, "datum.yaml")
    try:
        mtime = os.path.getmtime(path)
        with open(path) as f:
            return geo.from_datum_yaml(yaml.safe_load(f)), mtime
    except (IOError, OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        return None, None


class _Coverage(object):
    """last_mowed lookup at UTM points from coverage.npz (map frame of its datum)."""

    def __init__(self, site_dir, datum):
        from . import worldlayers            # numpy + yaml only; imported late for light tests
        self.grid = worldlayers.CoverageGrid.load(site_dir, datum, readonly=True)
        self.datum = datum

    def __call__(self, e, n):
        e = np.asarray(e, np.float64)
        n = np.asarray(n, np.float64)
        out = np.zeros(e.shape)
        g = self.grid
        lm = g.a.get("last_mowed") if hasattr(g, "a") else None
        if lm is None or lm.size == 0:
            return out
        yaw = self.datum["yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = e - self.datum["utm_e"], n - self.datum["utm_n"]
        x = c * dx + s * dy
        y = -s * dx + c * dy
        col = np.floor((x - g.ox) / g.RES).astype(np.int64)
        row = np.floor((y - g.oy) / g.RES).astype(np.int64)
        ok = (row >= 0) & (row < lm.shape[0]) & (col >= 0) & (col < lm.shape[1])
        out[ok] = lm[row[ok], col[ok]]
        return out


class _CmdError(Exception):
    pass


class RecorderCore(object):
    """Everything of terrain_recorder except ROS (see module doc)."""

    def __init__(self, cfg=None, store_params=None, sites_root=None, log=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            unknown = set(cfg) - set(DEFAULTS)
            if unknown:
                raise ValueError("unknown parameters: %s" % sorted(unknown))
            self.cfg.update(cfg)
        self.store_params = dict(store_params or {})
        self.root = sites_root or bundle.SITES_ROOT
        self.log = log or (lambda level, text: None)
        self.gate = QualityGate(self.cfg)
        self.site = None
        self.datum = None
        self.datum_mtime = None
        self.store = None
        self.cov_t = 0.0
        self.poll_t = None
        self.flush_t = None
        self.last_fused_stamp = None
        self.subscribed = False
        self.versions = {}
        self.stats = collections.Counter()
        self.fuse_ms = collections.deque(maxlen=200)
        self.decode_ms = collections.deque(maxlen=200)
        self.last_drop = None
        self.corr = CorrectionDetector(self.cfg)
        self.busy = None
        self.corr_active = False        # the node subscribes a reference odometry (set by the node)
        self.session_pending = False
        self.session_has_data = True    # the stored session may hold data: the first edge starts a new one
        self._log_f = None
        # T31
        self.settings_path = self.cfg["settings_file"] or os.path.join(
            os.path.expanduser("~"), ".vitulus", "terrain_recorder.json")
        self.enabled = self._load_enabled()
        self.paused = False
        self.token = None               # (token, cmd, expires, extra)
        self.last_cmd = None
        self.export_info = {"version": None, "stamp": None, "dir": "terrain_mosaic", "ok": None, "error": None,
                            "running": False, "reason": None}
        self.export_proc = None          # (Popen, output file, reason, start time)
        self.export_req = None
        self.export_dirty = False
        self.export_t = None
        self.nogo = []
        self.last_epoch = None
        self.last_settling = False
        self.last_write = None

    # ---- site ------------------------------------------------------------------
    def site_dir(self):
        return os.path.join(self.root, self.site) if self.site else None

    def poll(self, now, force=False):
        """Served site and its datum (every site_poll_s); coverage reload."""
        if not force and self.poll_t is not None and now - self.poll_t < self.cfg["site_poll_s"]:
            return
        self.poll_t = now
        site = read_served_site(self.root)
        if site != self.site:
            self._close_store()
            self.site = site
            self.datum, self.datum_mtime = (None, None)
            self.versions = {}
            self.log("info", "site %s" % site)
        if self.site is None:
            return
        datum, mtime = read_datum(self.site_dir())
        if datum is None:
            if self.store is not None:
                self._close_store()
            self.datum = None
            return
        if self.store is None:
            self.datum, self.datum_mtime = datum, mtime
            self.store = tl.TerrainStore(os.path.join(self.site_dir(), "terrain"), datum,
                                         params=self.store_params)
            self.versions = {k: v.get("version", 0) for k, v in self.store.index["tiles"].items()}
            self.cov_t = 0.0
            if self.session_pending:
                self.store.new_session()
                self.session_pending = False
            if self.store.broken:
                self.log("warn", "broken terrain files moved aside: %s" % self.store.broken)
        elif mtime != self.datum_mtime and datum != self.datum:
            # the map frame moved under odom: no visit may mix both; stored data is UTM
            self.store.close_all()
            self.store.datum = dict(datum)
            self.store.last_pose = None
            self.datum, self.datum_mtime = datum, mtime
            self.stats["datum_changes"] += 1
            self.log("info", "datum changed, visits closed")
        else:
            self.datum_mtime = mtime
        if now - self.cov_t >= self.cfg["coverage_reload_s"]:
            self.cov_t = now
            try:
                self.store.mowed = _Coverage(self.site_dir(), self.datum)
            except Exception as e:                       # noqa: BLE001
                self.store.mowed = None
                self.log("warn", "coverage.npz: %s" % e)

    def _close_store(self):
        if self.store is not None:
            try:
                self.store.close_all()
                # a removed site (mapping_manager remove_site) is not created again
                if os.path.isdir(os.path.join(self.root, self.site or "")) and self.site:
                    self.store.save_dirty()
            except Exception as e:                       # noqa: BLE001
                self.log("error", "saving terrain: %s" % e)
            self.store = None

    # ---- subscription ------------------------------------------------------------
    def want_subscription(self, now):
        """Subscribe after the gate was open subscribe_after_s, unsubscribe after it
        was closed unsubscribe_after_s (and without a site / datum)."""
        if self.store is None or not self.enabled or self.paused:
            self.subscribed = False
            return False
        if self.gate.check(now):
            if not self.subscribed and self.gate.open_for(now) >= self.cfg["subscribe_after_s"]:
                self.subscribed = True
        elif self.subscribed and now - self.gate.closed_since >= self.cfg["unsubscribe_after_s"]:
            self.subscribed = False
        return self.subscribed

    # ---- sessions and odometry -------------------------------------------------
    def feed_busy(self, now, busy):
        """/mower_smach/busy: a program starting (rising edge) is a new session."""
        busy = bool(busy)
        if busy and self.busy is False or busy and self.busy is None and self.last_fused_stamp is not None:
            self._new_session()
        if not busy and self.busy and self.store is not None:
            # end of a program: the visits into the map, written, exported
            self.store.close_all()
            if self.site and os.path.isdir(self.site_dir()):
                self.store.save_dirty()
            self.request_export("program_end")
        self.busy = busy

    def _new_session(self):
        """A new session, unless the current one has no fused batch yet (a program
        start right after a long pause is one new session, not two)."""
        if not self.session_has_data:
            return
        if self.store is not None:
            self.store.new_session()
        else:
            self.session_pending = True
        self.session_has_data = False
        self.stats["sessions"] += 1

    def feed_odom(self, t, x, y, yaw):
        """/odometry/odom sample (header time)."""
        if self.corr.feed_odom(t, x, y, yaw):
            self.stats["corrections"] += 1
            if self.store is not None and self.cfg["corr_enforce"]:
                # freshness boundary as for an epoch event: cells last fused before the end of the
                # settling time are stale (queued batches too, at their processing); visits stay open
                b = t + self.cfg["corr_settle_s"]
                fa = self.store.fresh_after
                self.store.fresh_after = b if fa is None else max(fa, b)
            self._blog("correction", t, t, "", "", "", "", "")

    def feed_ref(self, t, x, y, yaw):
        """Correction-free odometry sample (/odometry/ekf_wheel_odom)."""
        self.corr.feed_ref(t, x, y, yaw)

    # ---- messages ------------------------------------------------------------
    def _drop(self, why):
        self.stats["drop_" + why] += 1
        self.last_drop = why
        return why

    def _blog(self, kind, stamp, now, result, epoch, settling, x, y):
        path = self.cfg["batch_log"]
        if not path:
            return
        if self._log_f is None:
            new = not os.path.exists(path)
            self._log_f = open(path, "a")
            if new:
                self._log_f.write("kind,stamp,now,result,epoch,settling,map_x,map_y\n")
        self._log_f.write("%s,%.3f,%.3f,%s,%s,%s,%s,%s\n" % (kind, stamp, now, result, epoch, settling, x, y))

    def feed(self, msg, now):
        """One /traversability/fuse GridMap. Returns "queued" or the drop reason."""
        self.stats["msgs"] += 1
        t0 = time.perf_counter()
        try:
            m = decode_gridmap(msg)
            self.decode_ms.append((time.perf_counter() - t0) * 1000.0)
        except ValueError as e:
            self.log("warn", "fuse message refused: %s" % e)
            return self._drop("decode")
        res = self._feed(m, now)
        mx = my = ""
        if m["pose"] is not None:
            x, y, yaw = m["pose"]
            rx, ry = m["robot"][0], m["robot"][1]
            mx = "%.3f" % (x + math.cos(yaw) * rx - math.sin(yaw) * ry)
            my = "%.3f" % (y + math.sin(yaw) * rx + math.cos(yaw) * ry)
        self._blog("msg", m["stamp"], now, res, m["epoch"], int(m["settling"]), mx, my)
        return res

    def _feed(self, m, now):
        self.last_epoch, self.last_settling = m["epoch"], m["settling"]
        if not self.enabled or self.paused:
            return self._drop("off")
        if self.store is None:
            return self._drop("no_site")
        if now - m["stamp"] > self.cfg["max_age"]:
            return self._drop("stale")
        if m["pose"] is None:
            return self._drop("no_tf")
        if not self.gate.ok_for(m["stamp"], now):
            # the queue of the store must not keep batches from before the gate closed
            if self.store.pending:
                self.stats["drop_queue_gate"] += len(self.store.pending)
                self.store.pending = []
            return self._drop("gate")
        if self.cfg["corr_enforce"] and self.corr_active:
            if not self.corr.covered(m["stamp"]):
                return self._drop("corr_nodata")          # fails closed
            if self.corr.unsettled(m["stamp"]):
                return self._drop("correction")
        if self.last_fused_stamp is not None and \
                m["stamp"] - self.last_fused_stamp > self.cfg["session_gap_s"]:
            self._new_session()
        t0 = time.perf_counter()
        r = self.store.fuse(m["win"], m["pose"], m["stamp"], robot=m["robot"])
        self.fuse_ms.append((time.perf_counter() - t0) * 1000.0)
        if r["dropped"]:
            self.stats["drop_localisation"] += r["dropped"]
        for info in r["processed"]:
            if info.get("fused"):
                self.stats["fused"] += 1
                self.last_fused_stamp = info["stamp"]
                self.session_has_data = True
                self._blog("fused", info["stamp"], now, "fused", "", "", "", "")
            elif info.get("refused"):
                self.stats["refused_" + info["refused"]] += 1
        if not r["queued"]:
            return self._drop("settling")
        return "queued"

    # ---- files ------------------------------------------------------------------
    def flush(self, now, force=False):
        """Write dirty tiles (every flush_s). Returns the changed tiles
        [{"tx", "ty", "version"}] (for /terrain_recorder/updates) or []."""
        if self.store is None:
            return []
        if not force and self.flush_t is not None and now - self.flush_t < self.cfg["flush_s"]:
            return []
        self.flush_t = now
        if self._log_f is not None:
            self._log_f.flush()
        n = self.store.save_dirty()
        if not n:
            return []
        self.last_write = now
        changed = []
        for key, meta in self.store.index["tiles"].items():
            v = meta.get("version", 0)
            if self.versions.get(key) != v:
                tx, ty = (int(p) for p in key.split("_"))
                changed.append({"tx": tx, "ty": ty, "version": v})
                self.versions[key] = v
        self.stats["flushes"] += 1
        if changed:
            self.export_dirty = True
        return changed

    def close(self):
        """Shutdown: close the visits and write everything (queued batches are lost:
        they are younger than accept_delay)."""
        self._close_store()
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None

    # ---- T31: control -------------------------------------------------------------
    def _known_cells(self):
        """(known cells, tiles): loaded tiles counted live, the others from the index."""
        if self.store is None:
            return 0, 0
        keys = set(self.store.index["tiles"].keys())
        known = 0
        for key, t in self.store.tiles.items():
            name = "%d_%d" % key
            keys.add(name)
            known += t.known()
        for name, meta in self.store.index["tiles"].items():
            if tuple(int(v) for v in name.split("_")) not in self.store.tiles:
                known += int(meta.get("known", 0))
        return known, len(keys)

    def _load_enabled(self):
        try:
            with open(self.settings_path) as f:
                return bool(json.load(f).get("enabled", False))
        except (IOError, OSError, ValueError, TypeError, AttributeError):
            return False                 # default: off until the owner switches it on

    def _save_enabled(self, value, by="ui"):
        d = os.path.dirname(self.settings_path)
        os.makedirs(d, exist_ok=True)
        data = json.dumps({"enabled": bool(value), "changed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           "by": by}).encode()
        fd, tmp = tempfile.mkstemp(prefix="terrain_recorder.", suffix=".tmp", dir=d)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.settings_path)

    def _stop_store_work(self):
        """Off / paused / a long command: no queue, visits closed and written."""
        if self.store is not None:
            self.store.pending = []
            self.store.close_all()
            if self.site and os.path.isdir(self.site_dir()):
                self.store.save_dirty()
        self.subscribed = False

    # ---- export (T31 §3; E2: outside the lock, a niced child process) ---------------
    def request_export(self, reason):
        if self.cfg["export"] and self.store is not None and self.site:
            self.export_req = reason
            self.stats["export_requests"] += 1

    def export_tick(self, now):
        """From the timer: collect a finished export, start a requested / due one."""
        self._collect_export()
        if self.export_proc is not None or not self.cfg["export"] or self.store is None or not self.site:
            return
        if self.export_req is None and self.export_dirty and \
                (self.export_t is None or now - self.export_t >= self.cfg["export_s"]):
            self.export_req = "interval"
        if self.export_req is None:
            return
        if os.path.isdir(self.site_dir()):
            self.store.save_dirty()       # the child reads the stored tiles
        self._spawn_export()
        self.export_t = now
        self.export_dirty = False

    def _export_args(self):
        d = self.store.datum
        return ["--terrain", os.path.join(self.site_dir(), "terrain"),
                "--datum", json.dumps({"utm_e": d["utm_e"], "utm_n": d["utm_n"], "yaw_rad": d.get("yaw_rad", 0.0)}),
                "--out", os.path.join(self.site_dir(), "terrain_mosaic"), "--site", self.site,
                "--sessions", str(self.store.session)]

    def _spawn_export(self):
        import subprocess
        import sys
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        out = tempfile.TemporaryFile(mode="w+")
        cmd = ["nice", "-n", str(int(self.cfg["export_nice"])), sys.executable, "-m",
               "vitulus_mapping.terrainexport"] + self._export_args()
        try:
            self.export_proc = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env), out,
                                self.export_req, time.time())
            self.export_info.update(running=True, reason=self.export_req)
        except OSError as e:
            out.close()
            self.export_info.update(ok=False, error="spawn: %s" % e, running=False)
        self.export_req = None

    def _test_delay(self):
        if self.cfg["test_cmd_delay_s"] > 0:
            time.sleep(float(self.cfg["test_cmd_delay_s"]))

    def _collect_export(self, wait=False):
        if self.export_proc is None:
            return None
        proc, out, reason, t0 = self.export_proc
        if wait:
            try:
                proc.wait(timeout=self.cfg["export_timeout_s"])
            except Exception:                            # noqa: BLE001  subprocess.TimeoutExpired
                pass
        if proc.poll() is None:
            if time.time() - t0 > self.cfg["export_timeout_s"]:
                self._kill_export()
                self.export_info.update(ok=False, error="timeout", running=False, reason=reason)
                self.stats["export_timeouts"] += 1
                self.log("error", "export killed after %.0f s" % self.cfg["export_timeout_s"])
            return None
        self.export_proc = None
        out.seek(0)
        text = out.read()
        out.close()
        res = None
        for line in reversed(text.strip().splitlines()):
            try:
                res = json.loads(line)
                break
            except ValueError:
                continue
        if res and res.get("ok"):
            self.nogo = res.get("nogo_candidates", [])
            self.export_info.update(version=res.get("version"), stamp=time.time(), ok=True, error=None,
                                    running=False, cpu_s=res.get("cpu_s"), wall_s=round(time.time() - t0, 2),
                                    reason=reason)
            self.stats["exports"] += 1
        else:
            err = (res or {}).get("error") or text.strip()[-300:] or "exit %s" % proc.returncode
            self.export_info.update(ok=False, error=err, running=False, reason=reason)
            self.log("error", "export: %s" % err)
        return res

    def _kill_export(self):
        if self.export_proc is not None:
            try:
                self.export_proc[0].kill()
                self.export_proc[0].wait()
            except OSError:
                pass
            self.export_proc[1].close()
            self.export_proc = None
            self.export_info.update(running=False)

    def export_now(self):
        """Synchronous export (tests, offline): start it and wait for the child."""
        if self.store is None or not self.site:
            return None
        self._collect_export(wait=True)
        self.export_req = self.export_req or "now"
        if os.path.isdir(self.site_dir()):
            self.store.save_dirty()
        self._spawn_export()
        self.export_dirty = False
        return self._collect_export(wait=True)

    def _token(self, cmd, extra, now):
        tok = secrets.token_hex(3)
        self.token = (tok, cmd, now + self.cfg["token_s"], extra)
        return tok

    def _check_token(self, cmd, given, now, extra=None):
        t = self.token
        ok = t is not None and given == t[0] and cmd == t[1] and now <= t[2] and t[3] == extra
        if ok:
            self.token = None
        return ok

    def _backup(self):
        """Move <site>/terrain/ to <site>/terrain_backup/<YYYYmmdd-HHMMSS>/; returns the name."""
        src = os.path.join(self.site_dir(), "terrain")
        if not os.path.isdir(src):
            return None
        name = time.strftime("%Y%m%d-%H%M%S")
        dst_root = os.path.join(self.site_dir(), "terrain_backup")
        os.makedirs(dst_root, exist_ok=True)
        dst = os.path.join(dst_root, name)
        k = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_root, "%s-%d" % (name, k))
            k += 1
        os.replace(src, dst)
        keep = max(1, int(self.cfg["backup_keep"]))
        old = sorted(glob.glob(os.path.join(dst_root, "*")))[:-keep]
        for d in old:
            shutil.rmtree(d, ignore_errors=True)
            self.stats["backups_pruned"] += 1
        return os.path.basename(dst)

    def _fresh_store(self):
        self.store = tl.TerrainStore(os.path.join(self.site_dir(), "terrain"), self.datum,
                                     params=self.store_params)
        self.versions = {}

    def list_imports(self):
        if not self.site:
            return []
        out = []
        for j in sorted(glob.glob(os.path.join(self.site_dir(), "terrain_import", "*", "teren_*.json"))):
            d = os.path.dirname(j)
            out.append({"name": os.path.basename(d), "path": os.path.relpath(d, self.site_dir())})
        return out

    def list_backups(self):
        if not self.site:
            return []
        out = []
        for d in sorted(glob.glob(os.path.join(self.site_dir(), "terrain_backup", "*"))):
            area = None
            try:
                with open(os.path.join(d, "index.json")) as f:
                    idx = json.load(f)
                area = round(sum(int(t.get("known", 0)) for t in idx.get("tiles", {}).values())
                             * tl.RES ** 2, 1)
            except (IOError, OSError, ValueError, TypeError, AttributeError):
                pass
            out.append({"name": os.path.basename(d), "area_m2": area})
        return out

    LONG_CMDS = ("import", "delete_map")

    def mark_running(self, data):
        """Before a long command: state.last_cmd = {"id", "cmd", "ok": None, "running": True}."""
        try:
            c = json.loads(data) if isinstance(data, str) else dict(data)
        except (TypeError, ValueError):
            return
        if isinstance(c, dict) and c.get("cmd") in self.LONG_CMDS and "confirm" in c or \
                isinstance(c, dict) and c.get("cmd") == "import" and c.get("mode", "merge") == "merge":
            self.last_cmd = {"id": c.get("id"), "cmd": c.get("cmd"), "ok": None, "error": None, "result": {},
                             "running": True}

    def command(self, data, now):
        """One /terrain_recorder/cmd message (str or dict). Returns the answer dict
        {"id", "cmd", "ok", "error", "result"} (also kept as last_cmd)."""
        try:
            c = json.loads(data) if isinstance(data, str) else dict(data)
            if not isinstance(c, dict):
                raise ValueError
        except (TypeError, ValueError):
            c = {}
        ans = {"id": c.get("id"), "cmd": c.get("cmd"), "ok": False, "error": None, "result": {}, "running": False}
        try:
            ans["result"] = self._command(c, now)
            ans["ok"] = True
        except _CmdError as e:
            ans["error"] = str(e)
        except Exception as e:                           # noqa: BLE001
            ans["error"] = "internal: %s" % e
            self.log("error", "command %s: %s" % (c.get("cmd"), e))
        self.last_cmd = ans
        self.stats["cmds"] += 1
        return ans

    def _command(self, c, now):
        cmd = c.get("cmd")
        if cmd == "enable":
            self._save_enabled(True)
            self.enabled = True
            self.log("info", "enabled from the UI")
            return {"enabled": True}
        if cmd == "disable":
            self._save_enabled(False)
            self.enabled = False
            self._stop_store_work()
            self.log("info", "disabled from the UI")
            return {"enabled": False}
        if cmd in ("pause", "resume"):
            self.paused = cmd == "pause"
            if self.paused:
                self._stop_store_work()
            return {"paused": self.paused}
        if cmd == "list":
            return {"imports": self.list_imports(), "backups": self.list_backups()}
        if cmd == "rebuild_export":
            if self.store is None:
                raise _CmdError("no_site")
            self.request_export("rebuild_export")
            return {"queued": True}
        if cmd == "delete_map":
            if self.store is None:
                raise _CmdError("no_site")
            if "confirm" not in c:
                known, tiles = self._known_cells()
                return {"confirm_token": self._token("delete_map", None, now), "expires_s": self.cfg["token_s"],
                        "site": self.site, "area_m2": round(known * tl.RES ** 2, 1), "tiles": tiles}
            if not self._check_token("delete_map", c.get("confirm"), now):
                raise _CmdError("bad_token")
            self._test_delay()
            self._stop_store_work()
            self._kill_export()
            name = self._backup()
            self._fresh_store()
            exp = os.path.join(self.site_dir(), "terrain_mosaic")
            if os.path.isdir(exp):
                shutil.rmtree(exp)
            self.nogo = []
            self.export_info.update(version=None, stamp=time.time(), ok=True, error=None)
            self.log("warn", "terrain map of %s deleted from the UI, backup %s" % (self.site, name))
            return {"backup": name}
        if cmd == "import":
            if self.store is None:
                raise _CmdError("no_site")
            src = c.get("source")
            if src == "bag":
                raise _CmdError("not_available")
            if src != "mosaic":
                raise _CmdError("bad_source")
            names = {i["name"]: i for i in self.list_imports()}
            item = names.get(c.get("name"))
            if item is None:
                raise _CmdError("unknown_import")
            mode = c.get("mode", "merge")
            if mode not in ("merge", "replace"):
                raise _CmdError("bad_mode")
            mdir = os.path.join(self.site_dir(), item["path"])
            try:
                meta, layers, t = terrainexport.load_mosaic_dir(mdir)
            except (ValueError, IOError, OSError, KeyError) as e:
                raise _CmdError("bad_mosaic: %s" % e)
            mdatum, _mt = read_datum(mdir)      # the site datum the mosaic was made in, when shipped along
            delta = None
            if mdatum is not None:
                zs, zm = self.datum.get("utm_zone"), mdatum.get("utm_zone")
                if zs is not None and zm is not None and int(zs) != int(zm):
                    raise _CmdError("utm_zone_mismatch: site %s, mosaic %s" % (zs, zm))
                de = mdatum["utm_e"] - self.datum["utm_e"]
                dn = mdatum["utm_n"] - self.datum["utm_n"]
                dyaw = math.degrees(_wrap(mdatum.get("yaw_rad", 0.0) - self.datum.get("yaw_rad", 0.0)))
                delta = {"dE": round(de, 3), "dN": round(dn, 3), "dyaw_deg": round(dyaw, 3)}
                if (math.hypot(de, dn) > self.cfg["import_max_datum_m"] or
                        abs(dyaw) > self.cfg["import_max_datum_deg"]) and not c.get("force"):
                    raise _CmdError("datum_differs: dE %.2f m, dN %.2f m, dyaw %.2f deg (force to import)"
                                    % (de, dn, dyaw))
            if mode == "replace":
                if "confirm" not in c:
                    return {"confirm_token": self._token("import", item["name"], now),
                            "expires_s": self.cfg["token_s"], "site": self.site, "datum_delta": delta}
                if not self._check_token("import", c.get("confirm"), now, extra=item["name"]):
                    raise _CmdError("bad_token")
            self._test_delay()
            self._stop_store_work()
            backup = None
            if mode == "replace":
                self._kill_export()               # the export must not read the files being moved
                backup = self._backup()
                self._fresh_store()
            n, off = self.store.import_map_grid((float(meta["origin_x"]), float(meta["origin_y"])),
                                                float(meta["resolution"]), layers, t, datum=mdatum)
            self.store.save_dirty()
            self.request_export("import")
            self.log("info", "imported %s (%s): %d cells" % (item["name"], mode, n))
            return {"imported_cells": n, "height_offset_m": round(off, 4), "backup": backup,
                    "datum": "mosaic" if mdatum else "site", "datum_delta": delta}
        raise _CmdError("unknown_cmd")

    def state(self, now, load=None):
        """/terrain_recorder/state (T31 §1)."""
        if not self.enabled:
            mode = "off"
        elif self.paused:
            mode = "paused"
        elif self.store is None:
            mode = "no_site"
        elif self.subscribed:
            mode = "recording"
        else:
            mode = "waiting"
        gopen = self.gate.check(now)
        known, tiles = self._known_cells()
        ms = sorted(self.fuse_ms)
        drops = {k[5:]: v for k, v in self.stats.items() if k.startswith("drop_")}
        return {
            "v": 2, "stamp": now, "running": True, "enabled": self.enabled, "paused": self.paused, "mode": mode,
            "site": self.site,
            "gate": {"open": gopen, "reason": self.gate.reason, "open_for_s": round(self.gate.open_for(now), 1)},
            "subscribed": self.subscribed,
            "map": {"area_m2": round(known * tl.RES ** 2, 1), "tiles": tiles, "known_cells": known,
                    "last_fused": self.last_fused_stamp,
                    "last_fused_age_s": round(now - self.last_fused_stamp, 1) if self.last_fused_stamp else None,
                    "last_write": self.last_write,
                    "session": self.store.session if self.store is not None else None},
            "localisation": {"epoch": self.last_epoch, "settling": bool(self.last_settling),
                             "corrections": self.corr.n_events},
            "load": dict(load or {}, fuse_ms_avg=round(sum(ms) / len(ms), 2) if ms else None,
                         fuse_ms_p95=round(ms[int(0.95 * (len(ms) - 1))], 2) if ms else None),
            "counts": {"msgs": self.stats.get("msgs", 0), "fused": self.stats.get("fused", 0), "dropped": drops},
            "export": dict(self.export_info),
            "nogo_candidates": self.nogo,
            "imports": self.list_imports(), "backups": self.list_backups(),
            "last_cmd": self._last_cmd_public(),
        }

    def _last_cmd_public(self):
        """last_cmd for the latched state: without the confirmation token (it goes only to
        cmd_result, to the UI that asked)."""
        if not self.last_cmd:
            return self.last_cmd
        lc = dict(self.last_cmd)
        res = dict(lc.get("result") or {})
        if "confirm_token" in res:
            res["confirm_token"] = None
            res["token_sent"] = True
        lc["result"] = res
        return lc

    # ---- state --------------------------------------------------------------------
    def status(self, now):
        ms = sorted(self.fuse_ms)
        st = {"site": self.site, "gate": self.gate.reason, "subscribed": self.subscribed,
              "stats": dict(self.stats), "last_drop": self.last_drop,
              "fuse_ms_avg": round(sum(ms) / len(ms), 2) if ms else None,
              "fuse_ms_p95": round(ms[int(0.95 * (len(ms) - 1))], 2) if ms else None,
              "fuse_ms_max": round(ms[-1], 2) if ms else None,
              "corrections": self.corr.n_events, "corr_last_diff": self.corr.last_diff,
              "decode_ms_avg": round(sum(self.decode_ms) / len(self.decode_ms), 2) if self.decode_ms else None}
        if self.store is not None:
            s = self.store.summary()
            st.update({"tiles_loaded": s["tiles_loaded"], "tiles_indexed": s["tiles_indexed"],
                       "open_visits": s["open_visits"], "queued": len(self.store.pending),
                       "z_offset": s["z_offset"], "anchor_residual": s["anchor_residual"],
                       "ram_mb": round(s["ram_bytes"] / 1e6, 1), "session": self.store.session,
                       "epoch": self.store.epoch, "store": s["stats"], "broken": s["broken"]})
        return st
