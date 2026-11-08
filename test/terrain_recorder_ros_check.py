#!/usr/bin/env python3
"""Driver of terrain_recorder_ros_check.sh: publishes a good localisation and 1 Hz fuse
windows of a robot driving 0.25 m/s, then checks the node (subscription only with the
gate open, stats, updates, files, writing at shutdown). Made-up coordinates."""

import json
import os
import signal
import sys
import time

import numpy as np
import rosgraph
import rospy
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from ublox_msgs.msg import NavPVT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vitulus_mapping import terrainfuse as tf    # noqa: E402

SITE_DIR, REC_PID = sys.argv[1], int(sys.argv[2])
LOC = {"owner": "gps", "gps_good": True, "heading_ok": True, "anchor_frozen": False,
       "heading_anchor_frozen": False}
fails = []


def check(cond, text):
    print(("ok   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


def fuse_subscribed():
    state = rosgraph.Master('/check').getSystemState()
    return any(t == '/traversability/fuse' and any('terrain_recorder' in n for n in nodes)
               for t, nodes in state[1])


def layers(n=120):
    shp = (n, n)
    return {"elevation": np.zeros(shp, np.float32), "variance": np.full(shp, 1e-4, np.float32),
            "points": np.full(shp, 5, np.float32), "t_fused": np.zeros(shp, np.float32),
            "cost": np.full(shp, 10, np.float32), "reason": np.ones(shp, np.float32),
            "slope": np.full(shp, 0.05, np.float32), "step": np.full(shp, 0.01, np.float32),
            "roughness": np.full(shp, 0.004, np.float32)}


rospy.init_node('check', anonymous=True)
got = {"stats": None, "updates": None, "state": None, "result": []}
rospy.Subscriber('/terrain_recorder/state', String, lambda m: got.__setitem__("state", json.loads(m.data)))
rospy.Subscriber('/terrain_recorder/cmd_result', String, lambda m: got["result"].append(json.loads(m.data)))
pub_cmd = rospy.Publisher('/terrain_recorder/cmd', String, queue_size=5)
rospy.Subscriber('/terrain_recorder/stats', String, lambda m: got.__setitem__("stats", json.loads(m.data)))
rospy.Subscriber('/terrain_recorder/updates', String, lambda m: got.__setitem__("updates", json.loads(m.data)))
pub_loc = rospy.Publisher('/nav_tf/loc_status', String, queue_size=1, latch=True)
pub_pvt = rospy.Publisher('/gnss/navpvt', NavPVT, queue_size=1)
pub_fuse = rospy.Publisher('/traversability/fuse', GridMap, queue_size=1)
CLASSES = (GridMap, Float32MultiArray, MultiArrayDimension, rospy.Time)
time.sleep(2.0)

# 1. docked: no fuse subscription
pub_loc.publish(String(data=json.dumps(dict(LOC, owner="dock"))))
time.sleep(4.0)
check(not fuse_subscribed(), "no /traversability/fuse subscription while docked")

# 2. good localisation: subscribes, fuses
lay = layers()
t_end = time.time() + 26.0       # gate_hold_s 10 + accept_delay 3 before the first fusion
x = 0.0
last_fuse = 0.0
while time.time() < t_end and not rospy.is_shutdown():
    pub_loc.publish(String(data=json.dumps(LOC)))
    p = NavPVT()
    p.fixType = 3
    p.flags = 0x01 | (2 << 6)
    p.hAcc = 14
    pub_pvt.publish(p)
    now = rospy.get_time()
    if now - last_fuse >= 1.0:
        last_fuse = now
        pub_fuse.publish(tf.build_gridmap(now - 0.4, (x - 3.0, -3.0), 0.05, lay, (0.0, 0.0, 0.0),
                                          (x, 0.0, 0.0, 0.0), classes=CLASSES))
        x += 0.25
    time.sleep(0.2)
check(fuse_subscribed(), "subscribed to /traversability/fuse with RTK fixed + loc ok")
st = got["stats"] or {}
print("stats:", json.dumps({k: st.get(k) for k in ("site", "gate", "subscribed", "stats", "fuse_ms_avg",
                                                    "tiles_loaded", "open_visits", "queued")}))
check(st.get("gate") == "ok" and st.get("site") == "testsite", "stats: gate ok, site")
check((st.get("stats") or {}).get("fused", 0) >= 5, "stats: fused batches")
up = got["updates"] or {}
check(up.get("site") == "testsite" and len(up.get("tiles", [])) >= 1, "updates after a flush")
check(os.path.exists(os.path.join(SITE_DIR, "terrain", "index.json")), "index.json written")
sv = got["state"] or {}
print("state:", json.dumps({k: sv.get(k) for k in ("mode", "enabled", "gate", "map", "load", "export")}))
check(sv.get("mode") == "recording" and sv.get("v") == 1, "state: mode recording (T31)")
check(os.path.exists(os.path.join(SITE_DIR, "terrain_mosaic", "teren_testsite.json")) and
      os.path.exists(os.path.join(SITE_DIR, "terrain_mosaic", "teren_testsite_cost_rgba.png")),
      "export terrain_mosaic written (T31)")
before = json.load(open(os.path.join(SITE_DIR, "terrain", "index.json")))
pub_cmd.publish(String(data=json.dumps({"id": "c1", "cmd": "disable"})))
for _ in range(30):
    if got["result"] and (got["state"] or {}).get("mode") == "off":
        break
    time.sleep(0.2)
check(bool(got["result"]) and got["result"][-1].get("ok") and got["result"][-1].get("id") == "c1",
      "cmd disable answered on cmd_result")
check((got["state"] or {}).get("mode") == "off", "state: mode off after disable")
time.sleep(1.5)
check(not fuse_subscribed(), "fuse unsubscribed after disable")

# 3. disable (above) and the shutdown write the open visits
os.killpg(REC_PID, signal.SIGINT)
for _ in range(50):
    try:
        os.kill(REC_PID, 0)
    except OSError:
        break
    time.sleep(0.2)
after = json.load(open(os.path.join(SITE_DIR, "terrain", "index.json")))
known_b = sum(t["known"] for t in before["tiles"].values())
known_a = sum(t["known"] for t in after["tiles"].values())
print("known cells before disable / after shutdown: %d / %d" % (known_b, known_a))
check(known_a > known_b, "disable / shutdown closed the visits and wrote them")
sys.exit(1 if fails else 0)
