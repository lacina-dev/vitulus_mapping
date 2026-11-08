#!/usr/bin/env python3
"""Driver of terrain_align_compare.sh: every /traversability/fuse window goes into two
TerrainStores (align_tilt False / True, else the defaults, accept_delay 0, no gate); every
closed visit is logged; per bag: consistency (RMS of the visit values about the cell mean
over cells with >= 2 visits), the same without the first visit, p95 of |deviation|."""

import collections
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import rospy
from grid_map_msgs.msg import GridMap
from rospy.numpy_msg import numpy_msg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vitulus_mapping import terrainfuse as tf, terrainlayers as tl    # noqa: E402

JOB, RATE, BAGS = sys.argv[1], sys.argv[2], sys.argv[3:]
DATUM = {"utm_e": 500000.0, "utm_n": 5000000.0, "yaw_rad": 0.0}
VARIANTS = {"offset": {"align_tilt": False}, "plane": {"align_tilt": True}}


class Variant(object):
    def __init__(self, name, params):
        self.name = name
        self.params = dict(params, accept_delay=0.0)
        self.reset()

    def reset(self):
        self.store = tl.TerrainStore(tempfile.mkdtemp(prefix="cmp_%s_" % self.name, dir=JOB), DATUM,
                                     params=self.params)
        self.visits = collections.defaultdict(list)
        self.store.visit_log = self.log
        self.tilts = []

    def log(self, keys, h):
        for k, v in zip(keys.tolist(), h.tolist()):
            self.visits[k].append(v)

    def feed(self, m):
        r = self.store.fuse(m["win"], m["pose"], m["stamp"], robot=m["robot"])
        for info in r["processed"]:
            if info.get("tilt_deg") is not None:
                self.tilts.append(info["tilt_deg"])

    def metrics(self):
        self.store.close_all()
        dev_all, dev_later = [], []
        for vs in self.visits.values():
            if len(vs) >= 2:
                a = np.array(vs)
                dev_all.append(a - a.mean())
            if len(vs) >= 3:
                b = np.array(vs[1:])
                dev_later.append(b - b.mean())
        out = {"cells_2plus": len(dev_all)}
        if dev_all:
            d = np.abs(np.concatenate(dev_all))
            out.update(rms_cm=round(100 * float(np.sqrt(np.mean(d ** 2))), 3),
                       p95_cm=round(100 * float(np.percentile(d, 95)), 3))
        if dev_later:
            d = np.abs(np.concatenate(dev_later))
            out.update(later_rms_cm=round(100 * float(np.sqrt(np.mean(d ** 2))), 3),
                       later_p95_cm=round(100 * float(np.percentile(d, 95)), 3))
        if self.tilts:
            out["tilt_deg_p50_p95"] = [round(float(np.percentile(self.tilts, 50)), 3),
                                       round(float(np.percentile(self.tilts, 95)), 3)]
            out["tilt_batches"] = len(self.tilts)
        return out


variants = [Variant(n, p) for n, p in VARIANTS.items()]
count = {"msgs": 0, "bad": 0}


def cb(msg):
    try:
        m = tf.decode_gridmap(msg)
    except ValueError:
        count["bad"] += 1
        return
    count["msgs"] += 1
    if m["pose"] is None:
        return
    for v in variants:
        v.feed(m)


rospy.init_node("align_compare", anonymous=True)
rospy.Subscriber("/traversability/fuse", numpy_msg(GridMap), cb, queue_size=5, buff_size=8 * 1024 * 1024)
time.sleep(2.0)
results = {}
for bag in BAGS:
    t0 = time.time()
    subprocess.run(["rosbag", "play", "-q", "--clock", "-r", RATE, bag, "--topics", "/cloudXYZ", "/obstacles_cloud",
                    "/tf", "/tf_static", "/odometry/odom"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
    name = os.path.basename(bag)
    results[name] = {v.name: v.metrics() for v in variants}
    results[name]["msgs"] = count["msgs"]
    print(json.dumps({name: results[name], "wall_s": round(time.time() - t0)}), flush=True)
    for v in variants:
        v.reset()
    count["msgs"] = 0
print("SUMMARY " + json.dumps(results))
