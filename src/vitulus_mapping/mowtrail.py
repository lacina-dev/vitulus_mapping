#!/usr/bin/env python3
"""
mowtrail.py -- the "Mow load" layer: a trail behind the robot that says how hard
the blade worked and what state the robot was in. Contract: docs/MAP_OBJECTS.md
section 11.

NO rospy. `TrailRecorder` holds all the logic, nodes/map_world_recorder feeds it
plain values, so a whole run plays in a test without a ROS master.

LOAD. 0..1, where 1.0 = the firmware cuts the motor. Two real limits of
STMmowerROSControll/src/main.cpp, the higher of the two wins:

  current   isense / trip_a       trip_a = ISENSE_MAX, default 1.54 A. isense comes
                                  only in the text of /mower/stall, 2 Hz, a 200 ms
                                  mean. Mean of the last two readings: one reading
                                  swings 0.09..0.64 A on the same grass.
  rpm sag   (sp - rpm) / (sp * (1 - stall_ratio))
                                  stall_ratio = 0.25: the firmware stalls below a
                                  quarter of the setpoint. The speed PID holds rpm
                                  within +-3 of 3400 until its duty runs out, so
                                  sag alone reads 0 on all but the worst grass.

Blade running free reads about 0.16 A = 0.10. The PID duty, the best signal, is
not published (only a debug print), and battery current includes the wheels.

TEMPERATURE is raw on purpose: a row carries the highest reading since the row
before, and a jump forces a row, so a one-sample spike stays visible and survives
downsampling. /mower/status sends garbage rpm in a bad sample (seen -31966, next
to temp 91 C) -- such rpm is dropped, the temp is kept: telling a spike from real
heating is what the history is for.

STORAGE, ~/.vitulus/mapping_v3/<site>/mow_load/
  <run>.jsonl   while running: header line, then one row or {"e":event} per line.
                Appended every flush_s; a torn last line is dropped when read.
  <run>.npz     finished: rows (int32 NxSTRIDE), events and meta (JSON). Atomic.
  index.json    the list of finished runs; rebuilt from the npz when missing.

ROW on disk, raw. Unknown = NONE, unless noted.
  0 x_cm, 1 y_cm    map frame of the stored datum
  2 load_pct        0..100
  3 amps_ma         motor current, isense
  4 rpm             5 temp_dc   0.1 C
  6 state           STATES index, + BRK when the line must not join the row before
  7 t_ds            0.1 s since the run start
  8 speed_cms       real, from the pose      9 cmd_cms   target, mux output
  10 height_mm      11 setpoint rpm
  12 roll_dd        13 pitch_dd   0.1 deg, IMU
  14 fix            GNSS class, worldlayers.fix_class     15 hacc_mm
  16 batt_cv        0.01 V        17 batt_pct    18 power_w  battery discharge
  19 wheel_cnm      0.01 Nm, sum of |torque| of the four wheels
  20 num_sv         21 pdop_c      0.01, NAV-PVT
  22 rtcm_rate_d    0.1 msg/s      23 rtcm_age_d   0.1 s, base RTCM, RXM-RTCM of /gnss
  24 rtcm_used      % of the messages in the window the receiver used
  25 rtcm_crc       CRC failures in the window
  26 wheel_max_cnm  0.01 Nm, the wheel working hardest
  27 lidar_cm       GPS against lidar on the map     28 lidar_rms_mm    29 lidar_yaw_dd
  30 jump_cm        GPS step against wheels over 1 s
  31 cno_dd         0.1 dB-Hz, mean C/N0 of used SV  32 prres_cm  RMS pseudorange residual
  33 jam            MON-HW jamInd 0..255
Older files have 20 or 26 columns: they load with NONE in the rest. A row of the
open run may be short too, the driver of a restart can be the old version.

ON THE WIRE a message carries only what the map colours by (WIRE columns, flat,
stride 18, downsampled) and the map-worthy events; a tap asks for the raw row.

GPS HONESTY. hAcc is what the receiver thinks of itself: 14 mm almost always, r = 0.03
against the lidar below. Two checks that do not ask the receiver:
  lidar_cm  gloc_server aligns every scan to the map even while GPS owns the pose
            and does not apply it: state gps_owns, corr_m is how far the scan wants
            the pose to move. 17 Oct bag, moving: median 9 cm, p90 36, p99 59.
            Median of the last second, rms <= 0.2 m only. It measures against the
            map, so a bad map reads as a bad GPS.
  jump_cm   the GPS step over 1 s against the wheel step in the same second, as
            vectors. Wheels cannot jump. Same bag, RTK fix, moving: median 3.8 cm,
            p90 12, p99 31. Skipped while the robot turns (the antenna swings).
C/N0, residuals and jamInd go along for the card.

RPM. The speed PID holds rpm so well that a scale over rpm shows one colour, so the
map colours by the dip = setpoint - rpm, only where the motor is at speed. It needs
no new column, old runs have it too.

RTCM. RXM-RTCM of the position receiver, one small message per RTCM message that
reached the receiver: 5.5 msg/s from a 1 Hz base (1005 and four MSM at 1 Hz, 1230
at 0.5 Hz), 10 s window like gnss_status. It carries type and CRC state, no size, so
the rate is msg/s and not bytes. Only counted. rtcm_gap = no message for gap_s
(3 s, gnss_status calls that corr_stale).

EVENT [t_ds, x_cm, y_cm, kind, text]. Point events go to the map; context kinds
(CONTEXT: zone, path, chunk, owner, program) change seldom and stand for a column
in each row, the tap card looks up the last one. The sources are firmware state
edges, /mower_smach/status and the smach's own log lines (feed_log): there is no
event topic, the patterns are pinned by a test.
"""

import json
import math
import os
import re
import tempfile
import time

import numpy as np

from . import bundle
from . import geo
from . import worldlayers

__all__ = ["TrailRecorder", "STATES", "EVENTS", "BRK", "NONE", "STRIDE", "WIRE_STRIDE", "DEFAULTS",
           "parse_isense", "parse_log", "sane_rpm", "compute_load", "classify", "decimate",
           "summarize", "list_runs", "read_run", "query"]

FORMAT_VERSION = 3
OLD_STRIDES = (20, 26)
STRIDE = 34
(X, Y, LOAD, AMPS, RPM, TEMP, STATE, T, SPEED, CMD, HEIGHT, SETPOINT, ROLL, PITCH, FIX, HACC,
 BATT_V, BATT_PCT, POWER, WHEEL, NUM_SV, PDOP, RTCM_RATE, RTCM_AGE, RTCM_USED,
 RTCM_CRC, WHEEL_MAX, LIDAR, LIDAR_RMS, LIDAR_YAW, JUMP, CNO, PRRES, JAM) = range(STRIDE)
# x y load amps rpm temp state t speed slope fix dip hacc age rate wheel lidar jump
WIRE_STRIDE = 18
BRK = 16
NONE = -9999
STATES = ("cutting", "spinup", "transit", "blocked", "overload", "hot", "rule", "hold",
          "link_lost")
CUTTING, SPINUP, TRANSIT, BLOCKED, OVERLOAD, HOT, RULE, HOLD, LINK_LOST = range(len(STATES))
RPM_LIMIT = 6000                # firmware rpm_callback refuses anything above
STATUS_STALE_S = 3.0            # /mower/status and /mower/stall are 2 Hz
RULE_STALE_S = 5.0              # the guard repeats at 1 Hz
# worst first: the map keeps the worst when icons crowd
EVENTS = ("blocked", "link_lost", "hot", "overload", "rule", "temp", "recovery", "backup", "detour",
          "stall", "no_progress", "rewindow", "hold", "rtcm_gap", "fix_down", "fix_up",
          "gps_jump")
CONTEXT = ("ctx", "owner", "program")
STOP_KINDS = ("blocked", "link_lost", "hot", "overload", "rule", "recovery", "backup", "stall",
              "no_progress")
_LOG_EVENTS = (("[MOWER_LINK] Mower link lost", "link_lost"),
               ("[MOWER_LINK] Blade stopped", "link_lost"),
               ("[MOWER_LINK] recovery failed", "link_lost"),
               ("[MOWER_LINK] mower power switched off", "link_lost"),
               ("[EXE] STALL", "stall"), ("[EXE] stalled", "stall"),
               ("[EXE] no goal progress", "no_progress"),
               ("[EXE] mux hold", "hold"), ("[DETOUR]", "detour"), ("[DETOUR_ESCALATE]", "detour"),
               ("[NAV_RECOVERY] Backing up", "backup"), ("[NAV_RECOVERY]", "recovery"),
               ("[NAV_REC]", "recovery"), ("[BLOCKED_RECOVERY]", "recovery"))
FIX_NAMES = ("none", "fix", "dgnss", "float", "rtk")
RTCM_WINDOW_S = 10.0
_EXE_START = re.compile(r"\[EXE\] START zone=(.*) path=(\d+)/(\d+) chunk_idx=(\d+)")
_ID = re.compile(r"^[0-9]{8}_[0-9]{6}(_[0-9]+)?$")
_ISENSE = re.compile(r"isense=(-?[0-9.]+)")

DEFAULTS = {
    "trip_a": 1.54,             # firmware ISENSE_MAX (EEPROM may differ, see README)
    "stall_ratio": 0.25,        # firmware stall_rpm_ratio
    "overload_load": 0.85,      # state "overload" from here on
    "temp_warn_c": 60.0,        # firmware TEMP_WARN
    "temp_max_c": 70.0,         # firmware TEMP_MAX
    "deck_m": 0.29,             # vitulus_planner MapSpec.COVERAGE_DIAMETER
    "spinup_s": 12.0,           # 600 rpm/s soft start to 3400 rpm + margin
    "spinup_ratio": 0.9,        # at speed from here on (firmware setpoint_reached)
    "blade_on_rpm": 300.0,
    "step_m": 0.10,             # one row per this much travel
    "period_s": 0.5,            # or this often if anything moved or changed
    "keepalive_s": 30.0,        # or this often whatever happens
    "load_change": 3,           # percentage points that count as a change
    "temp_jump_c": 2.0,         # a row at once on a jump this big
    "max_step_m": 1.0,          # longer step = pose jump, break the line
    "pose_gap_s": 2.0,          # no pose this long = break the line
    "hold_s": 4.0,              # no motion this long in a program = hold
    "hold_move_m": 0.05,
    "hold_turn_rad": 0.1,
    "end_hold_s": 30.0,         # blade off and not busy this long = end of the run
    "resume_gap_s": 600.0,      # a restart this short continues the same run
    "recover_wait_s": 15.0,     # after a restart, wait this long for busy to come
    "max_run_rows": 400000,     # about 11 h at 10 rows/s; then a new run
    "max_runs": 30,
    "flush_s": 5.0,
    "publish_s": 1.0,
    "snapshot_s": 30.0,
    "snapshot_min_s": 5.0,      # at most this often on request
    "max_msg_rows": 400,        # one append message
    "max_snapshot_rows": 12000, # snapshot or past run
    "event_gap_s": 5.0,         # the same event again within this is the same event
    "max_events": 5000,         # one run
    "max_msg_events": 500,      # one message
    "rtcm_gap_s": 3.0,          # event from here on, gnss_status corr_stale
    "fix_hold_s": 3.0,          # a fix class counts when it held this long
    "lidar_max_rms": 0.2,       # gloc_server eff_max_rms, a worse match says nothing
    "lidar_window_s": 1.5,
    "jump_period_s": 1.0,
    "jump_turn_rad": 0.25,      # more turn than this in the window: the antenna swung
    "jump_event_m": 0.5,        # RTK fix reached 0.42 m at most in the 17 Oct bag
}


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def parse_isense(text):
    """The motor current from a /mower/stall text, None without it."""
    m = _ISENSE.search(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if value >= 0.0 else None


def parse_log(text):
    """What a smach log line says: (kind, text), ("ctx", "zone|path|paths|chunk") or
    None. Only lines of known shape, see the module docstring."""
    if not text or text[0] != "[":
        return None
    m = _EXE_START.match(text)
    if m:
        return "ctx", "|".join(m.groups())
    if "re-window" in text and text.startswith("[EXE]"):
        return "rewindow", text[:120]
    for prefix, kind in _LOG_EVENTS:
        if text.startswith(prefix):
            return kind, text[:120]
    return None


class RtcmMeter(object):
    """Counts RXM-RTCM messages, parses nothing. A callback on any thread."""

    def __init__(self, window_s=RTCM_WINDOW_S):
        self.window_s = window_s
        self.times = []
        self.t0 = None
        self.last = None

    def feed(self, now, used, crc_failed):
        if self.t0 is None:
            self.t0 = now
        self.times.append((now, used, crc_failed))
        self.last = now
        if len(self.times) > 256 and self.times[0][0] < now - self.window_s:
            self._trim(now)

    def _trim(self, now):
        keep = now - self.window_s
        times = self.times
        i = 0
        while i < len(times) and times[i][0] < keep:
            i += 1
        if i:
            self.times = times[i:]

    def read(self, now):
        """(msg/s, age s, used %, crc failures), None before the first message."""
        if self.last is None:
            return None
        self._trim(now)
        win = self.times
        span = max(1.0, min(self.window_s, now - self.t0))
        used = int(round(100.0 * sum(1 for w in win if w[1]) / len(win))) if win else NONE
        return len(win) / span, now - self.last, used, sum(1 for w in win if w[2])


class GpsCheck(object):
    """GPS step against wheel step over about a second, as vectors, in east-north.

    Each feed is one GPS position with the wheel odometry of that moment; the node
    pairs them in the GPS callback, so both carry the same latency. pose_yaw - odom_yaw
    turns odometry into the map, datum_yaw the map into east-north."""

    def __init__(self, period_s=1.0, turn_rad=0.25):
        self.period_s = period_s
        self.turn_rad = turn_rad
        self.anchor = None
        self.value = None
        self.t = 0.0

    def feed(self, now, east, north, ox, oy, odom_yaw, pose_yaw, datum_yaw=0.0):
        """The new residual in metres when a window closed, else None."""
        a = self.anchor
        if a is None or now - a[0] > 5.0 * self.period_s:
            self.anchor = (now, east, north, ox, oy, odom_yaw)
            return None
        if now - a[0] < self.period_s:
            return None
        self.anchor = (now, east, north, ox, oy, odom_yaw)
        if abs(geo.wrap_pi(odom_yaw - a[5])) > self.turn_rad:
            self.value = None
            return None
        dx, dy = ox - a[3], oy - a[4]
        turn = pose_yaw + datum_yaw - odom_yaw
        c, s = math.cos(turn), math.sin(turn)
        self.value = math.hypot(east - a[1] - (c * dx - s * dy), north - a[2] - (s * dx + c * dy))
        self.t = now
        return self.value

    def read(self, now):
        return self.value if self.value is not None and now - self.t <= 3.0 * self.period_s else None


def sane_rpm(value):
    """abs(rpm), None for a garbage sample."""
    try:
        rpm = abs(int(value))
    except (TypeError, ValueError):
        return None
    return rpm if rpm <= RPM_LIMIT else None


def compute_load(amps, rpm, setpoint, running, cfg):
    """Load 0..1 against the firmware's limits, None when nothing is known.
    Sag counts only with the motor on and past the spin-up."""
    parts = []
    if amps is not None and cfg["trip_a"] > 0:
        parts.append(amps / cfg["trip_a"])
    if running and rpm is not None and setpoint and setpoint > 0:
        parts.append(max(0.0, setpoint - rpm) / (setpoint * (1.0 - cfg["stall_ratio"])))
    if not parts:
        return None
    return min(1.0, max(parts))


def classify(status, load, spinning_up, program, stationary, by_rule, cfg):
    """State class of one row. Firmware faults first, they explain everything
    else; a blade stopped by a rule stays that even while it coasts."""
    if status == "BLOCKED":
        return BLOCKED
    if status == "TEMP":
        return HOT
    if status == "ERR":
        return OVERLOAD
    if by_rule:
        return RULE
    if status == "RUN":
        if spinning_up:
            return SPINUP
        if load is not None and load >= cfg["overload_load"]:
            return OVERLOAD
        if stationary and program:
            return HOLD
        return CUTTING
    if stationary and program:
        return HOLD
    return TRANSIT


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------
def decimate(rows, limit):
    """Fewer rows for a message: (rows, factor). Keeps what the eye looks for:
    every break and state change, every temperature jump, and in each bin the
    highest load and the highest temperature."""
    n = len(rows)
    if n <= limit:
        return rows, 1
    k = max(2, int(math.ceil(n / (limit * 0.45))))
    bins = np.arange(n) // k
    keep = [np.array([0, n - 1])]
    brk = rows[:, STATE] >= BRK
    change = np.zeros(n, bool)
    change[1:] = (rows[1:, STATE] & (BRK - 1)) != (rows[:-1, STATE] & (BRK - 1))
    temp = rows[:, TEMP]
    jump = np.zeros(n, bool)
    known = (temp[1:] != NONE) & (temp[:-1] != NONE)
    jump[1:] = known & (np.abs(temp[1:] - temp[:-1]) >= 20)
    jump[:-1] |= jump[1:]                 # the row before: a spike has two edges
    keep.append(np.flatnonzero(brk | change | jump))
    for col in (LOAD, TEMP):
        order = np.lexsort((rows[:, col], bins))
        keep.append(order[np.flatnonzero(np.r_[bins[order][1:] != bins[order][:-1], True])])
    idx = np.unique(np.concatenate(keep))
    if len(idx) > limit:
        hard = idx[brk[idx]]
        rest = idx[~brk[idx]]
        room = max(2, limit - len(hard))
        rest = rest[np.linspace(0, len(rest) - 1, min(room, len(rest))).astype(np.int64)]
        idx = np.unique(np.concatenate((hard, rest)))
    return rows[idx], k


def _known(column):
    return column[column != NONE]


def summarize(rows, events=()):
    """Totals of a run. Load statistics over the cut rows only: spin-up inrush is
    normal and transit has no load. Time goes to the state of the row it ends in;
    a gap over a minute is not time at work."""
    out = {"rows": int(len(rows)), "length_m": 0.0, "cut_m": 0.0, "mean_load": None,
           "peak_load": None, "peak_a": None, "peak_temp": None, "time_mow_s": 0, "time_transit_s": 0,
           "time_stop_s": 0, "stops": sum(1 for e in events if e[3] in STOP_KINDS), "energy_wh": None,
           "rtk_frac": None, "rtcm_gaps": sum(1 for e in events if e[3] == "rtcm_gap"),
           "rtcm_rate": None, "rtcm_age_max": None, "wheel_mean": None, "wheel_peak": None,
           "fix_losses": sum(1 for e in events if e[3] == "fix_down"),
           "gps_jumps": sum(1 for e in events if e[3] == "gps_jump"),
           "lidar_med": None, "lidar_p95": None, "jump_med": None, "jump_p95": None}
    if not len(rows):
        return out
    state = rows[:, STATE] & (BRK - 1)
    cutting = (state == CUTTING) | (state == OVERLOAD)
    if len(rows) > 1:
        step = np.hypot(np.diff(rows[:, X]), np.diff(rows[:, Y])) / 100.0
        joined = rows[1:, STATE] < BRK
        out["length_m"] = round(float(step[joined].sum()), 1)
        out["cut_m"] = round(float(step[joined & cutting[1:]].sum()), 1)
        dt = np.minimum(np.diff(rows[:, T]) / 10.0, 60.0)
        out["time_mow_s"] = int(dt[cutting[1:]].sum())
        out["time_transit_s"] = int(dt[(state[1:] == TRANSIT) | (state[1:] == SPINUP)].sum())
        out["time_stop_s"] = int(dt.sum()) - out["time_mow_s"] - out["time_transit_s"]
        power = rows[1:, POWER]
        if (power != NONE).any():
            out["energy_wh"] = round(float((dt * power)[power != NONE].sum()) / 3600.0, 1)
    load = _known(rows[cutting, LOAD])
    if len(load):
        out["mean_load"] = round(float(load.mean()) / 100.0, 3)
        out["peak_load"] = round(float(load.max()) / 100.0, 3)
    amps = _known(rows[cutting, AMPS])
    if len(amps):
        out["peak_a"] = round(float(amps.max()) / 1000.0, 3)
    temp = _known(rows[:, TEMP])
    if len(temp):
        out["peak_temp"] = round(float(temp.max()) / 10.0, 1)
    fix = _known(rows[:, FIX])
    if len(fix):
        out["rtk_frac"] = round(float((fix == 4).mean()), 3)
    rate = _known(rows[:, RTCM_RATE])
    if len(rate):
        out["rtcm_rate"] = round(float(rate.mean()) / 10.0, 1)
    age = _known(rows[:, RTCM_AGE])
    if len(age):
        out["rtcm_age_max"] = round(float(age.max()) / 10.0, 1)
    # the wheels work in transit too; standing still they do not
    wheel = _known(rows[(state != HOLD), WHEEL])
    if len(wheel):
        out["wheel_mean"] = round(float(wheel.mean()) / 100.0, 1)
        out["wheel_peak"] = round(float(wheel.max()) / 100.0, 1)
    for name, col in (("lidar", LIDAR), ("jump", JUMP)):
        vals = _known(rows[:, col])
        if len(vals):
            out[name + "_med"] = int(np.median(vals))
            out[name + "_p95"] = int(np.percentile(vals, 95))
    return out


def wire(rows):
    """The columns the map colours by, flat."""
    out = np.empty((len(rows), WIRE_STRIDE), np.int32)
    out[:, :8] = rows[:, :8]
    out[:, 8] = rows[:, SPEED]
    bad = (rows[:, ROLL] == NONE) | (rows[:, PITCH] == NONE)
    out[:, 9] = np.where(bad, NONE, np.rint(np.hypot(rows[:, ROLL], rows[:, PITCH])))
    out[:, 10] = rows[:, FIX]
    state = rows[:, STATE] & (BRK - 1)
    # a wait may be with the blade off, then there is nothing to dip from
    at_speed = (state == CUTTING) | (state == OVERLOAD) | ((state == HOLD) & (rows[:, RPM] * 2 >= rows[:, SETPOINT]))
    known = at_speed & (rows[:, RPM] != NONE) & (rows[:, SETPOINT] != NONE) & (rows[:, SETPOINT] > 0)
    out[:, 11] = np.where(known, rows[:, SETPOINT] - rows[:, RPM], NONE)
    out[:, 12] = rows[:, HACC]
    out[:, 13] = rows[:, RTCM_AGE]
    out[:, 14] = rows[:, RTCM_RATE]
    out[:, 15] = rows[:, WHEEL]
    out[:, 16] = rows[:, LIDAR]
    out[:, 17] = rows[:, JUMP]
    return out.ravel().tolist()


def wire_events(events, limit):
    """[t, x, y, kind index] flat; the worst survive a cut."""
    shown = [e for e in events if e[3] in EVENTS]
    if len(shown) > limit:
        shown = sorted(sorted(shown, key=lambda e: EVENTS.index(e[3]))[:limit], key=lambda e: e[0])
    out = []
    for e in shown:
        out.extend((e[0], e[1], e[2], EVENTS.index(e[3])))
    return out


def _rebase(rows, stored, datum):
    """Rows written under datum `stored`, in the map frame of `datum`."""
    if stored is None or datum is None or not len(rows):
        return rows
    de = datum["utm_e"] - stored["utm_e"]
    dn = datum["utm_n"] - stored["utm_n"]
    dyaw = geo.wrap_pi(datum["yaw_rad"] - stored["yaw_rad"])
    if math.hypot(de, dn) < 0.02 and abs(dyaw) < 0.0005:
        return rows
    c0, s0 = math.cos(stored["yaw_rad"]), math.sin(stored["yaw_rad"])
    x, y = rows[:, X] / 100.0, rows[:, Y] / 100.0
    ex, ny = c0 * x - s0 * y - de, s0 * x + c0 * y - dn
    c1, s1 = math.cos(datum["yaw_rad"]), math.sin(datum["yaw_rad"])
    out = rows.copy()
    out[:, X] = np.rint((c1 * ex + s1 * ny) * 100.0).astype(np.int32)
    out[:, Y] = np.rint((-s1 * ex + c1 * ny) * 100.0).astype(np.int32)
    return out


def _datum_row(datum):
    return [float(datum["utm_e"]), float(datum["utm_n"]), float(datum["yaw_rad"])]


def _datum_dict(row):
    if row is None or len(row) != 3:
        return None
    return {"utm_e": float(row[0]), "utm_n": float(row[1]), "yaw_rad": float(row[2])}


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def run_dir(site):
    return os.path.join(bundle.site_dir(site), "mow_load")


def _atomic_json(path, payload):
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


def _full(rows):
    """Rows of any format version as STRIDE columns, the missing ones unknown."""
    if rows.shape[1] >= STRIDE:
        return rows[:, :STRIDE]
    pad = np.full((len(rows), STRIDE - rows.shape[1]), NONE, np.int32)
    return np.concatenate((rows, pad), axis=1)


def _read_jsonl(path):
    """(meta, rows, events, good_bytes). A torn line ends the file there."""
    meta = None
    rows = []
    events = []
    good = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                item = json.loads(line)
            except ValueError:
                break
            if meta is None:
                if not isinstance(item, dict):
                    break
                meta = item
            elif isinstance(item, list) and (len(item) == STRIDE or len(item) in OLD_STRIDES):
                rows.append(item + [NONE] * (STRIDE - len(item)))
            elif isinstance(item, dict) and "e" in item:
                events.append(item["e"])
            else:
                break
            good += len(line)
    arr = np.array(rows, np.int32).reshape(-1, STRIDE) if rows else np.zeros((0, STRIDE), np.int32)
    return meta, arr, events, good


def _write_npz(path, meta, rows, events):
    worldlayers._atomic_npz(path, {"rows": rows.astype(np.int32),
                                   "meta": np.array(json.dumps(meta)),
                                   "events": np.array(json.dumps(events, separators=(",", ":"))),
                                   "version": np.array(FORMAT_VERSION)})


def _read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])), _full(z["rows"].astype(np.int32)), json.loads(str(z["events"]))


def _entry(meta, rows, events, end):
    item = {"id": meta["run"], "program": meta.get("program") or "", "start": meta["start"],
            "end": end, "stop_reason": meta.get("stop_reason") or ""}
    item.update(summarize(rows, events))
    return item


def list_runs(site):
    """Finished runs, newest first. A lost index is rebuilt from the npz files."""
    d = run_dir(site)
    try:
        with open(os.path.join(d, "index.json")) as f:
            runs = json.load(f)["runs"]
        if isinstance(runs, list):
            return runs
    except (IOError, OSError, ValueError, KeyError, TypeError):
        pass
    runs = []
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".npz"))
    except OSError:
        return runs
    for name in names:
        try:
            meta, rows, events = _read_npz(os.path.join(d, name))
            runs.append(_entry(meta, rows, events, meta.get("end")))
        except (IOError, OSError, ValueError, KeyError, TypeError):
            continue
    runs.sort(key=lambda r: r.get("start") or 0, reverse=True)
    return runs


def read_run(site, run, datum=None):
    """(meta, rows, events in the map frame of datum), None when the run is not there."""
    if not _ID.match(str(run or "")):
        return None
    d = run_dir(site)
    try:
        if os.path.exists(os.path.join(d, run + ".npz")):
            meta, rows, events = _read_npz(os.path.join(d, run + ".npz"))
        else:
            meta, rows, events, _good = _read_jsonl(os.path.join(d, run + ".jsonl"))
    except (IOError, OSError, ValueError, KeyError, TypeError):
        return None
    if meta is None:
        return None
    stored = _datum_dict(meta.get("datum"))
    if events and datum is not None:
        at = np.array([[e[1], e[2]] + [0] * (STRIDE - 2) for e in events], np.int32)
        at = _rebase(at, stored, datum)
        events = [[e[0], int(p[0]), int(p[1])] + list(e[3:]) for e, p in zip(events, at)]
    return meta, _rebase(rows, stored, datum), events


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------
def _scaled(value, factor, digits=2):
    if value == NONE:
        return None
    return round(int(value) / factor, digits) if digits else int(int(value) / factor)


def _row_dict(row, start):
    return {"x": round(row[X] / 100.0, 2), "y": round(row[Y] / 100.0, 2),
            "load": _scaled(row[LOAD], 100.0), "amps": _scaled(row[AMPS], 1000.0, 3),
            "rpm": _scaled(row[RPM], 1.0, 0), "setpoint": _scaled(row[SETPOINT], 1.0, 0),
            "temp": _scaled(row[TEMP], 10.0, 1), "state": STATES[int(row[STATE]) & (BRK - 1)],
            "t": round((start or 0.0) + row[T] / 10.0, 1),
            "speed": _scaled(row[SPEED], 100.0), "cmd": _scaled(row[CMD], 100.0),
            "height": _scaled(row[HEIGHT], 1.0, 0), "roll": _scaled(row[ROLL], 10.0, 1),
            "pitch": _scaled(row[PITCH], 10.0, 1), "fix": _scaled(row[FIX], 1.0, 0),
            "hacc": _scaled(row[HACC], 1000.0, 3), "batt_v": _scaled(row[BATT_V], 100.0),
            "batt_pct": _scaled(row[BATT_PCT], 1.0, 0), "power": _scaled(row[POWER], 1.0, 0),
            "wheel": _scaled(row[WHEEL], 100.0), "num_sv": _scaled(row[NUM_SV], 1.0, 0),
            "pdop": _scaled(row[PDOP], 100.0), "rtcm_rate": _scaled(row[RTCM_RATE], 10.0, 1),
            "rtcm_age": _scaled(row[RTCM_AGE], 10.0, 1), "rtcm_used": _scaled(row[RTCM_USED], 1.0, 0),
            "rtcm_crc": _scaled(row[RTCM_CRC], 1.0, 0), "wheel_max": _scaled(row[WHEEL_MAX], 100.0),
            "lidar_m": _scaled(row[LIDAR], 100.0), "lidar_rms": _scaled(row[LIDAR_RMS], 1000.0, 3),
            "lidar_yaw": _scaled(row[LIDAR_YAW], 10.0, 1), "jump_m": _scaled(row[JUMP], 100.0),
            "cno": _scaled(row[CNO], 10.0, 1), "prres": _scaled(row[PRRES], 100.0),
            "jam": _scaled(row[JAM], 1.0, 0)}


def context_at(events, t_ds):
    """zone, path, chunk, owner, program at a moment of the run."""
    out = {"zone": "", "path": "", "chunk": None, "owner": "", "program": ""}
    for e in events:
        if e[0] > t_ds:
            break
        if e[3] == "ctx":
            parts = str(e[4]).split("|")
            if len(parts) == 4:
                out["zone"] = parts[0]
                out["path"] = "%s/%s" % (parts[1], parts[2])
                out["chunk"] = int(parts[3])
        elif e[3] == "owner":
            out["owner"] = e[4]
        elif e[3] == "program":
            out["program"] = e[4]
    return out


def detail(meta, rows, events, t_ds):
    """The raw row nearest a moment, with its context, for the tap card."""
    i = int(np.argmin(np.abs(rows[:, T] - int(t_ds))))
    out = _row_dict(rows[i], meta.get("start"))
    out.update(context_at(events, int(rows[i, T])))
    out["program"] = out["program"] or meta.get("program") or ""
    near = [e for e in events if e[3] in EVENTS and abs(e[0] - int(rows[i, T])) <= 50]
    out["events"] = [{"kind": e[3], "text": e[4]} for e in near[:4]]
    return out


def _inside(xs, ys, ring):
    """Even-odd test of points in a closed ring."""
    inside = np.zeros(len(xs), bool)
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if y0 == y1:
            continue
        cross = ((ys >= min(y0, y1)) & (ys < max(y0, y1))
                 & (xs < x0 + (ys - y0) * (x1 - x0) / (y1 - y0)))
        inside ^= cross
    return inside


def _heavy(rows, start, count=5, apart_m=1.0):
    """The highest loads, at least apart_m from each other."""
    out = []
    for i in np.argsort(-rows[:, LOAD], kind="stable"):
        row = rows[i]
        if row[LOAD] == NONE or len(out) >= count:
            break
        if all(math.hypot(row[X] / 100.0 - h["x"], row[Y] / 100.0 - h["y"]) >= apart_m
               for h in out):
            out.append(_row_dict(row, start))
    return out


def query(site, points=None, polygon=None, run=None, radius_m=0.5):
    """Where was it heavy. run None = the newest run of the site.

    points   -> a list, one row per point: the nearest sample within radius_m or None
    polygon  -> stats of the samples inside; neither -> the same of the whole run
    Rows of the cut states only count as load. Raises nothing the caller must
    handle beyond what layer_query catches."""
    datum = worldlayers.read_datum(site)
    if run is None:
        d = run_dir(site)
        open_runs = sorted(n[:-6] for n in os.listdir(d) if n.endswith(".jsonl")) \
            if os.path.isdir(d) else []
        runs = list_runs(site)
        run = open_runs[-1] if open_runs else (runs[0]["id"] if runs else None)
    got = read_run(site, run, datum) if run else None
    if got is None:
        return [None] * len(points) if points is not None else {"run": None, "rows": 0}
    meta, rows, events = got
    start = meta.get("start")
    if points is not None:
        out = []
        for px, py in points:
            if not len(rows):
                out.append(None)
                continue
            dist = np.hypot(rows[:, X] / 100.0 - px, rows[:, Y] / 100.0 - py)
            i = int(np.argmin(dist))
            if dist[i] > radius_m:
                out.append(None)
            else:
                out.append(dict(_row_dict(rows[i], start), dist=round(float(dist[i]), 2), run=run))
        return out
    if polygon is not None and len(rows):
        rows = rows[_inside(rows[:, X] / 100.0, rows[:, Y] / 100.0, polygon)]
    state = rows[:, STATE] & (BRK - 1)
    cut = rows[(state == CUTTING) | (state == OVERLOAD)]
    out = summarize(rows, events if polygon is None else ())
    out.update({"run": run, "program": meta.get("program") or "", "start": start,
                "states": dict((STATES[i], int((state == i).sum()))
                               for i in range(len(STATES)) if (state == i).any()),
                "heavy": _heavy(cut, start)})
    return out


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
class _Run(object):
    """Rows of one run in a growing int32 array; pending rows wait for the file."""

    def __init__(self, meta):
        self.meta = meta
        self.id = meta["run"]
        self.rows = np.zeros((1024, STRIDE), np.int32)
        self.n = 0
        self.saved = 0                 # rows in the file
        self.sent = 0                  # rows in a message
        self.snapped = -1              # rows in the last snapshot
        self.events = []
        self.events_saved = 0
        self.events_sent = 0
        self.last_event = {}           # kind -> time

    def add(self, row):
        if self.n == len(self.rows):
            self.rows = np.concatenate((self.rows, np.zeros_like(self.rows)))
        self.rows[self.n] = row
        self.n += 1

    def view(self):
        return self.rows[:self.n]


def _deci(degrees):
    return int(round(degrees * 10.0))


def _end_of(meta, rows):
    return meta["start"] + (rows[-1, T] / 10.0 if len(rows) else 0.0)


class TrailRecorder(object):
    """The node feeds signals, calls sync_site + step + maintain in its loop.

    publish(kind, payload)   kind in trail|snapshot|runs|past|detail
    log(text)                one line about something that went wrong
    Times are unix seconds passed in. Signal setters may run on other threads:
    each writes plain attributes, the loop reads them.
    """

    def __init__(self, publish=None, log=None, cfg=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.publish = publish or (lambda _kind, _payload: None)
        self.log = log or (lambda _text: None)
        self.site = None
        self.datum = None
        self.run = None
        self._recover = None           # (meta, rows, path) of an unfinished run
        self._recover_t = 0.0
        self._requests = []
        self._next_publish = 0.0
        self._next_flush = 0.0
        self._next_snapshot = 0.0
        self._last_snapshot = 0.0
        # signals
        self.status = None
        self.status_t = 0.0
        self.link_lost = False          # the mission said so: a board restart is not this
        self.rpm = None
        self.setpoint = 0
        self.temp = None
        self.temp_peak = None          # highest reading since the last row
        self.height = NONE
        self.cmd = None
        self.attitude = None           # (roll, pitch) rad, set by the node each tick
        self.wheel = None              # sum |torque| Nm, set by the node each tick
        self.gnss = None               # (h_acc_mm, fix_class, num_sv, pdop_c), same
        self.rtcm = RtcmMeter()
        self.wheel_max = None          # the hardest wheel, Nm, set by the node each tick
        self.lidar = []                # (t, corr_m, rms, yaw_deg) while GPS owns the pose
        self.check = GpsCheck(self.cfg["jump_period_s"], self.cfg["jump_turn_rad"])
        self.sat = None                # (mean cno, rms residual m), time
        self.sat_t = 0.0
        self.jam = None
        self.jam_t = 0.0
        self._rtcm_gap = False
        self._fix_said = None          # the fix class last told
        self._fix_new = None           # (class, since) waiting to hold
        self.battery = None            # (volts, percent, watts)
        self.battery_t = 0.0
        self.owner = ""
        self.amps = []                 # last two readings
        self.amps_t = 0.0
        self.busy = False
        self.program = ""
        self.stop_reason = ""
        self._stop_seen = ""
        self.rule = False
        self.rule_t = 0.0
        self.run_since = None          # motor came on
        # motion
        self._want_t = 0.0
        self._anchor = None
        self._anchor_t = 0.0
        self._pose_t = 0.0
        self._last = None              # last row: x, y, t, load, state, temp

    # ---- signals -------------------------------------------------------------
    def feed_status(self, now, status, rpm, setpoint, temp, height=None):
        if status == "RUN" and self.status != "RUN":
            self.run_since = now
        elif status != "RUN":
            self.run_since = None
        self.status = status
        self.status_t = now
        self.link_lost = False
        rpm = sane_rpm(rpm)
        if rpm is not None:
            self.rpm = rpm
        self.setpoint = setpoint
        if height is not None:
            self.height = int(height)
        if temp is not None and math.isfinite(temp):
            if (self.temp is not None and self.temp < self.cfg["temp_warn_c"] <= temp):
                self.event(now, "temp", "%.1f C" % temp)
            self.temp = temp
            if self.temp_peak is None or temp > self.temp_peak:
                self.temp_peak = temp

    def feed_rtcm(self, now, used, crc_failed):
        """One RXM-RTCM message of the position receiver. Only counted."""
        self.rtcm.feed(now, used, crc_failed)

    def feed_tracker(self, now, msg):
        """/gloc/track_status while a run is open. Only a match that was not
        applied while GPS owns the pose says something about the GPS."""
        if (msg.get("state") == "gps_owns" and msg.get("corr_m") is not None
                and (msg.get("rms") or 9.0) <= self.cfg["lidar_max_rms"]):
            self.lidar = [e for e in self.lidar if now - e[0] <= self.cfg["lidar_window_s"]]
            self.lidar.append((now, float(msg["corr_m"]), float(msg["rms"]),
                               float(msg.get("corr_yaw_deg") or 0.0)))

    def _read_lidar(self, now):
        """(corr_m, rms, yaw) median of the last moments, None without a match."""
        keep = [e for e in self.lidar if now - e[0] <= self.cfg["lidar_window_s"]]
        if not keep:
            return None
        return tuple(float(np.median([e[i] for e in keep])) for i in (1, 2, 3))

    def feed_check(self, now, east, north, ox, oy, odom_yaw, pose_yaw):
        """One GPS position with the odometry of that moment, in the loop."""
        got = self.check.feed(now, east, north, ox, oy, odom_yaw, pose_yaw,
                              self.datum["yaw_rad"] if self.datum else 0.0)
        if got is not None and got >= self.cfg["jump_event_m"]:
            self.event(now, "gps_jump", "%.2f m" % got)

    def feed_sat(self, now, cno, prres):
        self.sat = (cno, prres)
        self.sat_t = now

    def feed_jam(self, now, jam):
        self.jam = jam
        self.jam_t = now

    def _watch_gnss(self, now):
        """rtcm_gap at the edge, fix changes once they held: a run with one float
        second every minute must not fill the map."""
        got = self.rtcm.read(now)
        if got is not None:
            gap = got[1] >= self.cfg["rtcm_gap_s"]
            if gap and not self._rtcm_gap:
                self.event(now, "rtcm_gap", "%.1f s" % got[1])
            self._rtcm_gap = gap
        fix = self.gnss[1] if self.gnss is not None else None
        if fix is None:
            return
        if self._fix_said is None:
            self._fix_said = fix
        if fix == self._fix_said:
            self._fix_new = None
            return
        if self._fix_new is None or self._fix_new[0] != fix:
            self._fix_new = (fix, now)
        elif now - self._fix_new[1] >= self.cfg["fix_hold_s"]:
            self.event(now, "fix_up" if fix > self._fix_said else "fix_down",
                       "%s > %s" % (FIX_NAMES[self._fix_said], FIX_NAMES[min(fix, 4)]))
            self._fix_said = fix
            self._fix_new = None

    def feed_cmd(self, speed):
        self.cmd = speed

    def feed_power(self, now, volts, percent, watts):
        self.battery = (volts, percent, watts)
        self.battery_t = now

    def feed_owner(self, now, owner):
        if owner and owner != self.owner:
            self.owner = owner
            self.event(now, "owner", owner)

    def feed_log(self, now, text):
        """A smach log line, already filtered to the smach node."""
        got = parse_log(text) if self.run is not None else None
        if got:
            if text.startswith("[MOWER_LINK] Mower link lost"):
                self.link_lost = True
            self.event(now, got[0], got[1])

    def feed_smach(self, now, text):
        """/mower_smach/status text. A detour says so only here."""
        if text and text.startswith("Detour"):
            self.event(now, "detour", text[:120])

    def event(self, now, kind, text=""):
        """One point event at the robot, any thread. Repeats within event_gap_s
        are one event, context is never dropped."""
        run = self.run
        last = self._last
        if run is None or len(run.events) >= self.cfg["max_events"]:
            return
        if kind in EVENTS:
            if now - run.last_event.get(kind, -1e9) < self.cfg["event_gap_s"]:
                return
            run.last_event[kind] = now
        x, y = (last["x"], last["y"]) if last else (0.0, 0.0)
        run.events.append([int(round((now - run.meta["start"]) * 10.0)), int(round(x * 100.0)),
                           int(round(y * 100.0)), kind, text])

    def feed_stall(self, now, text):
        value = parse_isense(text)
        if value is None:
            return
        self.amps = (self.amps + [value])[-2:]
        self.amps_t = now

    def feed_busy(self, busy):
        self.busy = bool(busy)

    def feed_program(self, name):
        self.program = (name or "").strip()
        if self.run is not None and self.program and not self.run.meta.get("program"):
            self.run.meta["program"] = self.program

    def feed_stop_reason(self, text):
        self.stop_reason = (text or "").strip()

    def feed_rule(self, now, blade):
        self.rule = blade in ("stopped_by_rule", "restoring")
        self.rule_t = now

    def _fresh_status(self, now):
        return self.status if now - self.status_t <= STATUS_STALE_S else None

    def blade_running(self, now):
        return self._fresh_status(now) in ("RUN", "ERR", "BLOCKED")

    def wants_pose(self, now):
        """Should the node keep TF on."""
        return self.run is not None or self.busy or self.blade_running(now)

    # ---- site ----------------------------------------------------------------
    def sync_site(self, site, datum, now):
        """Cheap when nothing changed. The run belongs to one site and datum."""
        if site == self.site and datum == self.datum:
            return
        had = self.site
        self._finish(now)
        self._drop_recover()
        self.site, self.datum = site, datum
        self._last = None
        if had or site:
            self._publish_runs()
            self.publish("snapshot", self._empty(now))
        if site and datum:
            self._open_leftovers(now)

    def _limits(self):
        c = self.cfg
        return {"trip_a": c["trip_a"], "overload_load": c["overload_load"], "deck_m": c["deck_m"],
                "temp_warn_c": c["temp_warn_c"], "temp_max_c": c["temp_max_c"]}

    def _empty(self, now):
        return {"site": self.site, "run": None, "active": False, "stamp": now, "n": 0, "ds": 1,
                "stride": WIRE_STRIDE, "r": [], "e": [], "limits": self._limits()}

    def _open_leftovers(self, now):
        """Runs a dead node left open. The newest may go on, the rest are done."""
        d = run_dir(self.site)
        try:
            names = sorted(n for n in os.listdir(d) if n.endswith(".jsonl"))
        except OSError:
            return
        for i, name in enumerate(names):
            path = os.path.join(d, name)
            try:
                meta, rows, events, good = _read_jsonl(path)
                if meta is None or "run" not in meta:
                    os.remove(path)
                    continue
                if i == len(names) - 1:
                    with open(path, "rb+") as f:
                        f.truncate(good)
                    self._recover = (meta, rows, events, path)
                    self._recover_t = now
                else:
                    self._store(meta, rows, events, _end_of(meta, rows))
                    os.remove(path)
            except (IOError, OSError, ValueError, KeyError, TypeError) as exc:
                self.log("mow_load %s: %s" % (name, exc))

    def _drop_recover(self):
        """Close what could not be resumed."""
        got, self._recover = self._recover, None
        if got is None:
            return
        meta, rows, events, path = got
        try:
            self._store(meta, rows, events, _end_of(meta, rows))
            os.remove(path)
        except (IOError, OSError, ValueError, KeyError, TypeError) as exc:
            self.log("mow_load: %s" % exc)

    # ---- run -----------------------------------------------------------------
    def _new_id(self, start):
        base = time.strftime("%Y%m%d_%H%M%S", time.localtime(start))
        d = run_dir(self.site)
        run, n = base, 1
        while os.path.exists(os.path.join(d, run + ".npz")) or os.path.exists(os.path.join(d, run + ".jsonl")):
            run = "%s_%d" % (base, n)
            n += 1
        return run

    def _begin(self, now):
        try:
            os.makedirs(run_dir(self.site), exist_ok=True)
            meta = {"v": FORMAT_VERSION, "run": self._new_id(now), "site": self.site,
                    "program": self.program, "start": round(now, 1), "datum": _datum_row(self.datum)}
            with open(self._path(meta["run"]), "w") as f:
                f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        except (IOError, OSError) as exc:
            self.log("mow_load: %s" % exc)
            return
        self.run = _Run(meta)
        self._stop_seen = self.stop_reason
        self._last = None
        self._anchor = None
        self._rtcm_gap = False
        self._fix_said = self._fix_new = None
        self.check = GpsCheck(self.cfg["jump_period_s"], self.cfg["jump_turn_rad"])
        self.lidar = []
        if self.program:
            self.event(now, "program", self.program)
        if self.owner:
            self.event(now, "owner", self.owner)
        self._publish_runs()
        self._next_snapshot = 0.0

    def _resume(self, now):
        meta, rows, events, _path = self._recover
        self._recover = None
        self.run = _Run(meta)
        self.run.rows = np.concatenate((rows, self.run.rows))
        self.run.n = self.run.saved = self.run.sent = len(rows)
        self.run.events = events
        self.run.events_saved = self.run.events_sent = len(events)
        self._stop_seen = self.stop_reason
        self._last = None              # next row breaks the line: pose unknown between
        self._anchor = None
        self._publish_runs()
        self._next_snapshot = 0.0

    def _path(self, run):
        return os.path.join(run_dir(self.site), run + ".jsonl")

    def _append(self):
        run = self.run
        if run is None or (run.saved >= run.n and run.events_saved >= len(run.events)):
            return
        rows = run.rows[run.saved:run.n].tolist()
        events = run.events[run.events_saved:]
        try:
            with open(self._path(run.id), "a") as f:
                f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
                f.write("".join(json.dumps({"e": e}, separators=(",", ":")) + "\n" for e in events))
                f.flush()
                os.fsync(f.fileno())
            run.saved = run.n
            run.events_saved += len(events)
        except (IOError, OSError) as exc:
            self.log("mow_load: %s" % exc)

    def _store(self, meta, rows, events, end):
        """A finished run goes to its npz and into the index, retention applies."""
        meta = dict(meta, end=round(end, 1))
        d = run_dir(self.site)
        _write_npz(os.path.join(d, meta["run"] + ".npz"), meta, rows, events)
        runs = [r for r in list_runs(self.site) if r.get("id") != meta["run"]]
        runs.insert(0, _entry(meta, rows, events, meta["end"]))
        runs.sort(key=lambda r: r.get("start") or 0, reverse=True)
        for old in runs[int(self.cfg["max_runs"]):]:
            try:
                os.remove(os.path.join(d, str(old["id"]) + ".npz"))
            except OSError:
                pass
        _atomic_json(os.path.join(d, "index.json"),
                     {"v": FORMAT_VERSION, "runs": runs[:int(self.cfg["max_runs"])]})

    def _finish(self, now):
        run, self.run = self.run, None
        if run is None:
            return
        reason = self.stop_reason if self.stop_reason != self._stop_seen else ""
        if reason:
            run.meta["stop_reason"] = reason
        path = self._path(run.id)
        try:
            self.run = run                 # _append works on self.run
            self._append()
            self.run = None
            if run.n:
                self._store(run.meta, run.view(), run.events, now)
            os.remove(path)
        except (IOError, OSError, ValueError, KeyError, TypeError) as exc:
            self.run = None
            self.log("mow_load %s: %s" % (run.id, exc))
            return
        self._publish_runs()
        self.publish("trail", {"site": self.site, "run": run.id, "from": run.n, "n": run.n,
                               "stride": WIRE_STRIDE, "r": [], "end": round(now, 1)})
        self._send_snapshot(run, now, active=False)

    # ---- one tick ------------------------------------------------------------
    def step(self, now, pose):
        """pose = (x, y, yaw) in map, None while TF is off or stale."""
        want = bool(self.site and self.datum and (self.busy or self.blade_running(now)))
        if want:
            self._want_t = now
        if self.run is None:
            if self._recover is not None:
                meta, rows = self._recover[:2]
                if want:
                    if now - _end_of(meta, rows) <= self.cfg["resume_gap_s"]:
                        self._resume(now)
                    else:
                        self._drop_recover()
                        self._publish_runs()
                elif now - self._recover_t >= self.cfg["recover_wait_s"]:
                    self._drop_recover()
                    self._publish_runs()
            if self.run is None:
                if not want:
                    return
                self._begin(now)
                if self.run is None:
                    return
        elif not want and now - self._want_t >= self.cfg["end_hold_s"]:
            self._finish(now)
            return
        elif self.run.n >= self.cfg["max_run_rows"]:
            self._finish(now)
            self._begin(now)
            if self.run is None:
                return
        self._watch_gnss(now)
        self._sample(now, pose)

    def _stationary(self, now, x, y, yaw):
        a = self._anchor
        c = self.cfg
        if (a is None or math.hypot(x - a[0], y - a[1]) >= c["hold_move_m"]
                or abs(geo.wrap_pi(yaw - a[2])) >= c["hold_turn_rad"]):
            self._anchor = (x, y, yaw)
            self._anchor_t = now
            return False
        return now - self._anchor_t >= c["hold_s"]

    def _sample(self, now, pose):
        c = self.cfg
        last = self._last
        if pose is None:
            if last is None:
                return
            x, y, yaw = last["x"], last["y"], last["yaw"]
            stationary = False
        else:
            x, y, yaw = pose
            stationary = self._stationary(now, x, y, yaw)
        status = self._fresh_status(now)
        running = status == "RUN"
        rpm = self.rpm if status is not None else None
        spin_up = (running and self.run_since is not None
                   and now - self.run_since < c["spinup_s"]
                   and (rpm or 0) < c["spinup_ratio"] * (self.setpoint or 0))
        amps = (sum(self.amps) / len(self.amps)
                if self.amps and now - self.amps_t <= STATUS_STALE_S else None)
        load = compute_load(amps, rpm, self.setpoint, running and not spin_up, c)
        rule = self.rule and now - self.rule_t <= RULE_STALE_S
        state = classify(status, load, spin_up, self.busy, stationary, rule, c)
        if self.link_lost and status is None:
            state = LINK_LOST
        if last is not None and state != last["state"] and state in (BLOCKED, HOT, OVERLOAD, RULE):
            self.event(now, STATES[state], "")
        load_pct = -1 if load is None else int(round(load * 100.0))
        temp = self.temp_peak

        brk = last is None
        if last is not None and pose is not None:
            dist = math.hypot(x - last["x"], y - last["y"])
            brk = dist > c["max_step_m"] or now - self._pose_t > c["pose_gap_s"]
        else:
            dist = 0.0
        if pose is not None:
            self._pose_t = now
        if last is None:
            due = True
        else:
            age = now - last["t"]
            temp_jump = (temp is not None and last["temp"] is not None
                         and abs(temp - last["temp"]) >= c["temp_jump_c"])
            changed = (dist >= 0.02 or abs(load_pct - last["load"]) >= c["load_change"]
                       or (temp is not None and last["temp"] is not None
                           and abs(temp - last["temp"]) >= 0.5))
            due = (brk or state != last["state"] or dist >= c["step_m"] or temp_jump
                   or (age >= c["period_s"] and changed) or age >= c["keepalive_s"])
        if not due:
            return
        run = self.run
        speed = NONE
        if last is not None and not brk and now > last["t"] and pose is not None:
            speed = int(round(dist / (now - last["t"]) * 100.0))
        roll = pitch = hacc = fix = num_sv = pdop = NONE
        if self.attitude is not None:
            roll, pitch = _deci(math.degrees(self.attitude[0])), _deci(math.degrees(self.attitude[1]))
        if self.gnss is not None:
            hacc, fix = min(int(self.gnss[0]), 99999), int(self.gnss[1])
            if len(self.gnss) > 3:
                num_sv, pdop = int(self.gnss[2]), int(self.gnss[3])
        lidar = self._read_lidar(now)
        jump = self.check.read(now)
        cno = prres = jam = NONE
        if self.sat is not None and now - self.sat_t <= 12.0:     # NAV-SAT comes every 4 s
            cno, prres = int(round(self.sat[0] * 10.0)), int(round(self.sat[1] * 100.0))
        if self.jam is not None and now - self.jam_t <= 5.0:
            jam = int(self.jam)
        rate = age = used = crc = NONE
        got = self.rtcm.read(now)
        if got is not None:
            rate, age = int(round(got[0] * 10.0)), min(int(round(got[1] * 10.0)), 99999)
            used, crc = got[2], got[3]
        volts = pct = watts = NONE
        if self.battery is not None and now - self.battery_t <= STATUS_STALE_S:
            volts = int(round(self.battery[0] * 100.0))
            pct, watts = int(self.battery[1]), int(round(self.battery[2]))
        run.add((int(round(x * 100.0)), int(round(y * 100.0)),
                 NONE if load is None else load_pct,
                 NONE if amps is None else int(round(amps * 1000.0)),
                 NONE if rpm is None else int(rpm),
                 NONE if temp is None else int(round(temp * 10.0)),
                 state + (BRK if brk else 0),
                 int(round((now - run.meta["start"]) * 10.0)),
                 speed,
                 NONE if self.cmd is None else int(round(self.cmd * 100.0)),
                 NONE if status is None else self.height,
                 NONE if status is None else int(self.setpoint or 0),
                 roll, pitch, fix, hacc, volts, pct, watts,
                 NONE if self.wheel is None else int(round(self.wheel * 100.0)),
                 num_sv, pdop, rate, age, used, crc,
                 NONE if self.wheel_max is None else int(round(self.wheel_max * 100.0)),
                 NONE if lidar is None else int(round(lidar[0] * 100.0)),
                 NONE if lidar is None else int(round(lidar[1] * 1000.0)),
                 NONE if lidar is None else int(round(lidar[2] * 10.0)),
                 NONE if jump is None else min(int(round(jump * 100.0)), 99999),
                 cno, prres, jam))
        self._last = {"x": x, "y": y, "yaw": yaw, "t": now, "load": load_pct, "state": state,
                      "temp": temp}
        self.temp_peak = self.temp     # the next row starts from the current reading

    # ---- output --------------------------------------------------------------
    def _publish_runs(self):
        runs = list_runs(self.site) if self.site else []
        active = None
        if self.run is not None:
            active = self.run.id
            runs = [_entry(self.run.meta, self.run.view(), self.run.events, None)] + runs
        self.publish("runs", {"site": self.site, "active": active, "runs": runs,
                              "limits": self._limits()})

    def _send_snapshot(self, run, now, active=True):
        rows, ds = decimate(run.view(), int(self.cfg["max_snapshot_rows"]))
        run.snapped = run.n
        self._last_snapshot = now
        self.publish("snapshot", {"site": self.site, "run": run.id, "active": active,
                                  "program": run.meta.get("program") or "",
                                  "start": run.meta["start"], "stamp": round(now, 1),
                                  "n": run.n, "ds": ds, "stride": WIRE_STRIDE, "r": wire(rows),
                                  "e": wire_events(run.events, int(self.cfg["max_msg_events"])),
                                  "ne": len(run.events), "limits": self._limits()})

    def request(self, payload):
        """/map_objects/mow_trail/request, any thread. Done in the loop."""
        if len(self._requests) < 8:
            self._requests.append(payload)

    def _is_current(self, run):
        return run in (None, "", "current") or (self.run is not None and run == self.run.id)

    def _answer_detail(self, req, run_id, t_ds):
        if self._is_current(run_id) and self.run is not None:
            meta, rows, events = self.run.meta, self.run.view(), self.run.events
        else:
            got = read_run(self.site, run_id, self.datum) if self.site else None
            if got is None:
                self.publish("detail", {"req": req, "ok": False})
                return
            meta, rows, events = got
        if not len(rows):
            self.publish("detail", {"req": req, "ok": False})
            return
        self.publish("detail", {"req": req, "ok": True, "run": meta["run"],
                                "row": detail(meta, rows, events, t_ds)})

    def _answer(self, item, now):
        req = item.get("req")
        run = item.get("run")
        if item.get("t") is not None:
            self._answer_detail(req, run, int(item["t"]))
            return
        if self._is_current(run):
            if self.run is not None and now - self._last_snapshot >= self.cfg["snapshot_min_s"]:
                self._send_snapshot(self.run, now)
            return
        if run == "list":
            self._publish_runs()
            return
        got = read_run(self.site, run, self.datum) if self.site else None
        if got is None:
            self.publish("past", {"req": req, "ok": False, "run": run, "site": self.site,
                                  "reason": "no such run"})
            return
        meta, rows, events = got
        cut, ds = decimate(rows, int(self.cfg["max_snapshot_rows"]))
        self.publish("past", dict(_entry(meta, rows, events, meta.get("end")), **{
            "req": req, "ok": True, "run": meta["run"], "site": self.site, "n": len(rows),
            "ds": ds, "stride": WIRE_STRIDE, "r": wire(cut),
            "e": wire_events(events, int(self.cfg["max_msg_events"])), "limits": self._limits()}))

    def maintain(self, now):
        """Requests, the append message, the file, the snapshot. Nothing when idle."""
        requests, self._requests = self._requests, []
        for item in requests:
            try:
                self._answer(item, now)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.log("mow_load request: %s" % exc)
        run = self.run
        if run is None:
            return
        if now >= self._next_publish and run.sent < run.n:
            self._next_publish = now + self.cfg["publish_s"]
            end = min(run.n, run.sent + int(self.cfg["max_msg_rows"]))
            fresh = run.events[run.events_sent:]
            run.events_sent = len(run.events)
            self.publish("trail", {"site": self.site, "run": run.id, "from": run.sent, "n": end,
                                   "stride": WIRE_STRIDE, "start": run.meta["start"],
                                   "r": wire(run.rows[run.sent:end]),
                                   "e": wire_events(fresh, int(self.cfg["max_msg_events"]))})
            run.sent = end
        if now >= self._next_flush:
            self._next_flush = now + self.cfg["flush_s"]
            self._append()
        if now >= self._next_snapshot and run.snapped != run.n:
            self._next_snapshot = now + self.cfg["snapshot_s"]
            self._send_snapshot(run, now)

    def close(self):
        """Node going down. The open run stays open: a restart continues it."""
        self._append()
