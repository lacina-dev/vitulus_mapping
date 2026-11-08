#!/usr/bin/env python3
"""Driver of terrain_recorder_bag_check.sh: plays the bag, keeps a good fake localisation
gate on (sim time), measures the CPU / RSS of terrain_recorder and traversability_local,
stops the recorder and summarises the stored terrain (JSON)."""

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
from vitulus_mapping import geo, terrainlayers as tl    # noqa: E402

SITE_DIR, REC_PID, BAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PLAY = sys.argv[4:]
LOC = {"owner": "gps", "gps_good": True, "heading_ok": True, "anchor_frozen": False,
       "heading_anchor_frozen": False}
HZ = os.sysconf("SC_CLK_TCK")


def cpu_s(pid):
    try:
        f = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / float(HZ)
    except (IOError, OSError, IndexError):
        return None


def rss_mb(pid):
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    except (IOError, OSError):
        return None


def trav_pid():
    out = subprocess.run(["pgrep", "-f", "lib/vitulus_traversability/traversability_local"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


rospy.init_node("bag_check", anonymous=True)
got = {"stats": None, "trav": None, "updates": 0}
rospy.Subscriber("/terrain_recorder/stats", String, lambda m: got.__setitem__("stats", json.loads(m.data)))
rospy.Subscriber("/terrain_recorder/updates", String, lambda m: got.__setitem__("updates", got["updates"] + 1))
rospy.Subscriber("/traversability_local/stats", DiagnosticStatus,
                 lambda m: got.__setitem__("trav", {kv.key: kv.value for kv in m.values}))
pub_loc = rospy.Publisher("/nav_tf/loc_status", String, queue_size=1, latch=True)
pub_pvt = rospy.Publisher("/gnss/navpvt", NavPVT, queue_size=1)
time.sleep(1.0)
play = subprocess.Popen(["rosbag", "play"] + PLAY + [BAG, "--topics", "/cloudXYZ", "/obstacles_cloud", "/tf",
                                                       "/tf_static", "/odometry/odom"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t_wall0 = time.time()
c_rec0 = cpu_s(REC_PID)
tp = None
c_trav0 = None
rss_max = 0.0
samples = []
while play.poll() is None:
    pub_loc.publish(String(data=json.dumps(LOC)))
    p = NavPVT()
    p.fixType, p.flags, p.hAcc = 3, 0x01 | (2 << 6), 14
    pub_pvt.publish(p)
    if tp is None:
        tp = trav_pid()
        c_trav0 = cpu_s(tp) if tp else None
    r = rss_mb(REC_PID)
    rss_max = max(rss_max, r or 0.0)
    time.sleep(0.2)
wall = time.time() - t_wall0
c_rec = cpu_s(REC_PID) - c_rec0
c_trav = (cpu_s(tp) - c_trav0) if tp and c_trav0 is not None else None
time.sleep(3.0)
st = got["stats"] or {}
tr = got["trav"] or {}
os.killpg(REC_PID, signal.SIGINT)
for _ in range(100):
    try:
        os.kill(REC_PID, 0)
    except OSError:
        break
    time.sleep(0.2)
datum = geo.from_datum_yaml(yaml.safe_load(open(os.path.join(SITE_DIR, "datum.yaml"))))
store = tl.TerrainStore(os.path.join(SITE_DIR, "terrain"), datum, params={"max_loaded_tiles": 1000})
known = lethal = tent = obst = 0
hs = []
for key in list(store.index["tiles"].keys()):
    t = store.tile(*(int(v) for v in key.split("_")))
    reason, cost, tentative = t.effective()
    f = np.isfinite(t.h)
    known += int(f.sum())
    lethal += int(np.count_nonzero(f & (cost == 100)))
    tent += int(np.count_nonzero(f & tentative & (cost == 99)))
    obst += int(np.count_nonzero(f & (reason == 5)))
    hs.append(t.h[f])
hs = np.concatenate(hs) if hs else np.zeros(0)
disk = sum(os.path.getsize(os.path.join(SITE_DIR, "terrain", f)) for f in os.listdir(os.path.join(SITE_DIR, "terrain"))
           if f.endswith(".npz"))
out = {
    "wall_s": round(wall, 1), "play": " ".join(PLAY),
    "recorder_cpu_pct_of_core": round(100.0 * c_rec / wall, 2), "recorder_rss_max_mb": round(rss_max, 1),
    "trav_cpu_pct_of_core": round(100.0 * c_trav / wall, 2) if c_trav is not None else None,
    "recorder_stats": st.get("stats"), "fuse_ms_avg": st.get("fuse_ms_avg"), "fuse_ms_p95": st.get("fuse_ms_p95"),
    "fuse_ms_max": st.get("fuse_ms_max"), "decode_ms_avg": st.get("decode_ms_avg"), "z_offset": st.get("z_offset"), "store": st.get("store"),
    "updates_msgs": got["updates"],
    "trav": {k: tr.get(k) for k in ("fuse_msgs", "epoch", "epoch_events", "epoch_last_reason", "settling",
                                     "cloud_hz", "cpu_process_pct")},
    "tiles": len(store.index["tiles"]), "known_cells": known, "area_m2": round(known * tl.RES ** 2, 1),
    "confirmed_lethal_cells": lethal, "tentative_lethal_cells": tent, "obstacle_cells": obst,
    "height_p1_p99": [round(float(np.percentile(hs, 1)), 3), round(float(np.percentile(hs, 99)), 3)] if hs.size else None,
    "disk_kb": round(disk / 1e3, 1),
}
print(json.dumps(out, indent=1))
sys.exit(0 if known > 0 and (st.get("stats") or {}).get("fused", 0) > 0 else 1)
