#!/usr/bin/env python3
"""Driver and checks of terrain_recorder_events_check.sh (T30 P4).

Plays the bags in a row with a good fake localisation gate, then checks against the
offline mosaics (map frame of the same bags):
  1. no batch fused with a stamp inside a settling interval of the local node
     (consecutive messages with settling = 1);
  2. around every localisation event (start of a settling interval / epoch change) the
     recorder's heights are not shifted by more than 0.10 m against the mosaic (best
     horizontal shift of a 3 m patch, searched +-0.30 m);
  3. confirmed terrain bans on the robot's track (cells within 0.25 m of base_link of any
     message) <= 0.1 % of the known cells there.
Prints one JSON summary; exit 1 when a check fails."""

import csv
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rospy
import yaml
from diagnostic_msgs.msg import DiagnosticStatus
from std_msgs.msg import String
from ublox_msgs.msg import NavPVT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from vitulus_mapping import geo, terrainlayers as tl    # noqa: E402
import terrain_import_mosaic as tim                     # noqa: E402

SITE_DIR, REC_PID, BLOG, MOSAICS, RATE = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
BAGS = sys.argv[6:]
REF = os.environ.get("REF", "")
LOC = {"owner": "gps", "gps_good": True, "heading_ok": True, "anchor_frozen": False,
       "heading_anchor_frozen": False}
HZ = os.sysconf("SC_CLK_TCK")
RES = tl.RES


def cpu_s(pid):
    try:
        f = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / float(HZ)
    except (IOError, OSError, IndexError):
        return None


# ---- play -------------------------------------------------------------------------
rospy.init_node("events_check", anonymous=True)
got = {"stats": None, "trav": None}
rospy.Subscriber("/terrain_recorder/stats", String, lambda m: got.__setitem__("stats", json.loads(m.data)))
rospy.Subscriber("/traversability_local/stats", DiagnosticStatus,
                 lambda m: got.__setitem__("trav", {kv.key: kv.value for kv in m.values}))
pub_loc = rospy.Publisher("/nav_tf/loc_status", String, queue_size=1, latch=True)
pub_pvt = rospy.Publisher("/gnss/navpvt", NavPVT, queue_size=1)
time.sleep(1.0)
topics = ["/cloudXYZ", "/obstacles_cloud", "/tf", "/tf_static", "/odometry/odom", "/mower_smach/busy"]
if REF:
    topics.append(REF)
t0 = time.time()
c0 = cpu_s(REC_PID)
for bag in BAGS:
    play = subprocess.Popen(["rosbag", "play", "-q", "--clock", "-r", RATE, bag, "--topics"] + topics,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while play.poll() is None:
        pub_loc.publish(String(data=json.dumps(LOC)))
        p = NavPVT()
        p.fixType, p.flags, p.hAcc = 3, 0x01 | (2 << 6), 14
        pub_pvt.publish(p)
        time.sleep(0.2)
    print("played %s (%.0f s wall)" % (os.path.basename(bag), time.time() - t0), flush=True)
    time.sleep(4.0)
wall = time.time() - t0
cpu = cpu_s(REC_PID) - c0
st = got["stats"] or {}
tr = got["trav"] or {}
os.killpg(REC_PID, signal.SIGINT)
for _ in range(150):
    try:
        os.kill(REC_PID, 0)
    except OSError:
        break
    time.sleep(0.2)

# ---- batch log ----------------------------------------------------------------------
rows = list(csv.DictReader(open(BLOG)))
msgs = [r for r in rows if r["kind"] == "msg"]
fused = np.array([float(r["stamp"]) for r in rows if r["kind"] == "fused"])
corr = [float(r["stamp"]) for r in rows if r["kind"] == "correction"]
intervals = []
cur = None
prev_epoch = None
events = []
for r in msgs:
    s = float(r["stamp"])
    if r["settling"] == "1":
        if cur is None:
            cur = [s, s, r]
        cur[1] = s
    elif cur is not None:
        intervals.append(cur)
        cur = None
    if prev_epoch is not None and r["epoch"] != prev_epoch:
        events.append(("epoch", s, r))
    prev_epoch = r["epoch"]
if cur is not None:
    intervals.append(cur)
for a, b, r in intervals:
    events.append(("settling", a, r))
viol1 = [(a, b, int(np.count_nonzero((fused >= a) & (fused <= b)))) for a, b, _ in intervals]
fail1 = [v for v in viol1 if v[2] > 0]

# ---- store --------------------------------------------------------------------------
datum = geo.from_datum_yaml(yaml.safe_load(open(os.path.join(SITE_DIR, "datum.yaml"))))
store = tl.TerrainStore(os.path.join(SITE_DIR, "terrain"), datum, params={"max_loaded_tiles": 10000})
E0, N0 = datum["utm_e"], datum["utm_n"]


def rec_heights(xs, ys):
    """Recorder heights at map points (datum yaw 0: map = UTM - origin)."""
    keys = tl.pack_keys(np.floor((xs + E0) / RES + 1e-9), np.floor((ys + N0) / RES + 1e-9))
    h, _ = store.lookup(keys)
    return h


mosaics = []
for mdir in [m for m in MOSAICS.split(",") if m]:
    meta, lay, _t = tim.load_mosaic(mdir)
    mosaics.append((os.path.basename(os.path.normpath(mdir)), float(meta["origin_x"]), float(meta["origin_y"]),
                    lay["elevation"]))


def shift_at(x, y, radius=3.0, search=0.30):
    best_out = None
    for name, ox, oy, el in mosaics:
        H, W = el.shape
        c0, c1 = int((x - radius - ox) / RES), int((x + radius - ox) / RES)
        r0, r1 = int((y - radius - oy) / RES), int((y + radius - oy) / RES)
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, W - 1), min(r1, H - 1)
        if c1 <= c0 or r1 <= r0:
            continue
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        mx = ox + (cc + 0.5) * RES
        my = oy + (rr + 0.5) * RES
        inside = np.hypot(mx - x, my - y) <= radius
        m_el = el[rr, cc]
        ok = inside & np.isfinite(m_el)
        if np.count_nonzero(ok) < 200:
            continue
        k = int(round(search / RES))
        res = {}
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                h = rec_heights(mx[ok] + di * RES, my[ok] + dj * RES)
                f = np.isfinite(h)
                if np.count_nonzero(f) < 200:
                    continue
                d = h[f] - m_el[ok][f]
                res[(di, dj)] = (float(np.median(np.abs(d - np.median(d)))), int(f.sum()))
        if (0, 0) not in res:
            continue
        (bi, bj), (berr, bn) = min(res.items(), key=lambda kv: kv[1][0])
        out = {"mosaic": name, "cells": res[(0, 0)][1], "err0_mm": round(1000 * res[(0, 0)][0], 1),
               "best_shift_m": [round(bi * RES, 2), round(bj * RES, 2)],
               "best_err_mm": round(1000 * berr, 1)}
        if best_out is None or out["cells"] > best_out["cells"]:
            best_out = out
    return best_out


ev_out = []
fail2 = []
for kind, s, r in events:
    if not r["map_x"]:
        continue
    x, y = float(r["map_x"]), float(r["map_y"])
    sh = shift_at(x, y)
    item = {"kind": kind, "stamp": round(s, 1), "map": [round(x, 1), round(y, 1)], "shift": sh}
    ev_out.append(item)
    if sh is not None:
        mag = float(np.hypot(*sh["best_shift_m"]))
        # a shift counts only when it clearly fits better than none
        if mag > 0.10 + 1e-9 and sh["best_err_mm"] < 0.8 * sh["err0_mm"]:
            fail2.append(item)

# track
track = np.array([[float(r["map_x"]), float(r["map_y"])] for r in msgs if r["map_x"]])
keys = set()
rad = 0.25
k = int(np.ceil(rad / RES))
for x, y in track:
    gx0, gy0 = int(np.floor((x + E0) / RES)), int(np.floor((y + N0) / RES))
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if np.hypot(di * RES, dj * RES) <= rad:
                keys.add((gx0 + di, gy0 + dj))
keys = np.array(sorted(keys), np.int64)
pk = tl.pack_keys(keys[:, 0], keys[:, 1])
known_t = conf_t = tent_t = 0
conf_all = known_all = 0
for (tx, ty), sel, ly, lx in store._groups(pk):
    t = store.tile(tx, ty)
    if t is None:
        continue
    reason, cost, tent = t.effective()
    f = np.isfinite(t.h[ly, lx])
    terr = ((reason[ly, lx] == 3) | (reason[ly, lx] == 4)) & (cost[ly, lx] == 100)
    known_t += int(f.sum())
    conf_t += int(np.count_nonzero(f & terr))
    tent_t += int(np.count_nonzero(f & tent[ly, lx] & (cost[ly, lx] == 99)))
for key in store.index["tiles"]:
    t = store.tile(*(int(v) for v in key.split("_")))
    reason, cost, tent = t.effective()
    f = np.isfinite(t.h)
    known_all += int(f.sum())
    conf_all += int(np.count_nonzero(f & ((reason == 3) | (reason == 4)) & (cost == 100)))
share = 100.0 * conf_t / known_t if known_t else 0.0
out = {
    "bags": [os.path.basename(b) for b in BAGS], "rate": RATE, "wall_s": round(wall, 0),
    "recorder_cpu_pct_of_core": round(100.0 * cpu / wall, 2),
    "recorder_stats": st.get("stats"), "fuse_ms_avg": st.get("fuse_ms_avg"), "fuse_ms_p95": st.get("fuse_ms_p95"),
    "decode_ms_avg": st.get("decode_ms_avg"), "store": st.get("store"), "session": st.get("session"),
    "trav": {k2: tr.get(k2) for k2 in ("fuse_msgs", "epoch", "epoch_events", "epoch_last_reason")},
    "msgs": len(msgs), "fused": int(fused.size),
    "check1_settling_intervals": len(intervals), "check1_fused_inside": fail1,
    "check2_events": ev_out, "check2_fail": len(fail2),
    "check3_track_known": known_t, "check3_confirmed_on_track": conf_t, "check3_share_pct": round(share, 3),
    "tentative_on_track": tent_t, "confirmed_total": conf_all, "known_total": known_all,
    "area_m2": round(known_all * RES ** 2, 1),
    "p3_corrections_counted": len(corr), "p3_reference": REF or None,
}
print(json.dumps(out, indent=1))
ok = not fail1 and not fail2 and share <= 0.1
print("CHECKS:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
