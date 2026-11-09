#!/usr/bin/env python3
"""Run the localisation-jump detector of terrain_recorder (terrainfuse.CorrectionDetector) over a bag.

    python3 tools/corr_detector_eval.py <bag> [--odom /odometry/ekf_wheel_nav_odom] [--ref /odometry/ekf_wheel_odom]
                                        [--set key=value ...]

The bag needs the corrected odometry (/odometry/odom, or /odometry/ekf_wheel_nav_odom which navi_transform
republishes as /odometry/odom) and the correction-free wheel EKF. Prints events per minute, the events (time,
dxy, dyaw), and how many raw jumps of the corrected odometry (> 10 cm beyond the motion between two messages)
were caught within 1 s. Read-only (ROS Noetic, rosbag API).
"""

import argparse
import math
import os
import sys

import numpy as np
import rosbag

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vitulus_mapping import terrainfuse as tf    # noqa: E402


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--odom", default=None, help="default: /odometry/odom, else /odometry/ekf_wheel_nav_odom")
    ap.add_argument("--ref", default="/odometry/ekf_wheel_odom")
    ap.add_argument("--set", action="append", default=[], help="detector parameter key=value")
    a = ap.parse_args()
    b = rosbag.Bag(a.bag)
    topics = b.get_type_and_topic_info().topics
    odom = a.odom or ("/odometry/odom" if "/odometry/odom" in topics else "/odometry/ekf_wheel_nav_odom")
    cfg = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = float(v)
    det = tf.CorrectionDetector(cfg)
    nav, events = [], []
    for topic, m, _t in b.read_messages(topics=[odom, a.ref]):     # bag order = arrival order
        p = m.pose.pose
        r = (m.header.stamp.to_sec(), p.position.x, p.position.y, yaw(p.orientation))
        if topic == a.ref:
            det.feed_ref(*r)
        else:
            nav.append(r + (abs(m.twist.twist.linear.x),))
            if det.feed_odom(*r):
                events.append((r[0], det.last_diff))
    nav = np.array(nav)
    if nav.size == 0:
        raise SystemExit("no %s in the bag" % odom)
    mins = (nav[-1, 0] - nav[0, 0]) / 60.0
    dt = np.diff(nav[:, 0])
    st = np.hypot(np.diff(nav[:, 1]), np.diff(nav[:, 2])) - nav[1:, 4] * dt
    known = nav[1:, 0][(st > 0.10) & (dt < 0.3)]
    caught = sum(any(abs(k - e[0]) <= 1.0 for e in events) for k in known)
    print("%s: %.1f min, %s vs %s, %d events = %.2f/min, raw jumps > 10 cm caught %d/%d"
          % (os.path.basename(a.bag), mins, odom, a.ref, len(events), len(events) / mins, caught, len(known)))
    for t, d in events[:50]:
        print("  t+%7.1f s  dxy %.3f m  dyaw %.1f deg" % (t - nav[0, 0], d[0], d[1]))


if __name__ == "__main__":
    main()
