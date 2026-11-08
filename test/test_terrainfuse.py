#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.terrainfuse (T30 D2: terrain_recorder logic).

Run: python3 test/test_terrainfuse.py   (from the package root)

Made-up site and coordinates (a round UTM point), temporary site root.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import yaml

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import terrainfuse as tf                       # noqa: E402
from vitulus_mapping import worldlayers                             # noqa: E402

E0, N0 = 500000.0, 5000000.0
DATUM = {"utm_e": E0 + 3.3, "utm_n": N0 - 7.1, "yaw_rad": 0.01, "alt": 300.0, "utm_zone": 33}
T0 = 1790000000.0
RTK_FLAGS = 0x01 | (2 << 6)          # gnssFixOK + carrier fixed
LOC_OK = {"owner": "gps", "gps_good": True, "heading_ok": True, "anchor_frozen": False,
          "heading_anchor_frozen": False}


def layers(z=0.0, n=120, t_rel=0.0, cost=10.0, reason=1.0, var=1e-4, points=5.0):
    shp = (n, n)
    return {"elevation": np.full(shp, z, np.float32), "variance": np.full(shp, var, np.float32),
            "points": np.full(shp, points, np.float32), "t_fused": np.full(shp, t_rel, np.float32),
            "cost": np.full(shp, cost, np.float32), "reason": np.full(shp, reason, np.float32),
            "slope": np.full(shp, 0.05, np.float32), "step": np.full(shp, 0.01, np.float32),
            "roughness": np.full(shp, 0.004, np.float32)}


def msg(stamp, x0=-3.0, y0=-3.0, z=0.0, pose=(0.0, 0.0, 0.0), robot=None, epoch=0, settling=False, **kw):
    robot = robot or (x0 + 3.0, y0 + 3.0, z, 0.0)
    return tf.build_gridmap(stamp, (x0, y0), 0.05, layers(z, **kw), pose, robot, epoch=epoch,
                            settling=settling)


class Message(unittest.TestCase):
    def test_roundtrip(self):
        lay = layers(0.2)
        lay["elevation"][0, 0] = np.nan                 # row 0 = lowest y, column 0 = lowest x
        lay["elevation"][7, 90] = 0.9
        lay["cost"][1, 2] = np.nan
        lay["points"][1, 3] = np.nan
        lay["t_fused"][3, 4] = -0.5
        m = tf.decode_gridmap(tf.build_gridmap(T0, (1.0, 2.0), 0.05, lay, (0.1, 0.2, 0.3),
                                               (4, 5, 0.01, 1.5), epoch=7, settling=True,
                                               z_correction=-0.26))
        w = m["win"]
        self.assertEqual((w.x0, w.y0, w.res, w.shape), (1.0, 2.0, 0.05, (120, 120)))
        self.assertTrue(np.isnan(w.elevation[0, 0]))
        self.assertAlmostEqual(float(w.elevation[7, 90]), 0.9, places=6)
        self.assertAlmostEqual(float(w.elevation[5, 5]), 0.2, places=6)
        self.assertEqual((float(w.cost[1, 2]), float(w.points[1, 3])), (-1.0, 0.0))
        self.assertAlmostEqual(float(w.t_fused[3, 4]), T0 - 0.5, places=3)
        self.assertEqual((m["epoch"], m["settling"], w.epoch, w.settling), (7, True, 7, True))
        np.testing.assert_allclose(m["pose"], (0.1, 0.2, 0.3), atol=1e-6)
        np.testing.assert_allclose(m["robot"], (4.0, 5.0, 0.01, 1.5), atol=1e-6)
        self.assertAlmostEqual(m["z_correction"], -0.26, places=5)
        m2 = tf.decode_gridmap(tf.build_gridmap(T0, (0, 0), 0.05, lay, None, (0, 0, 0, 0)))
        self.assertIsNone(m2["pose"])

    def test_version_2_epoch_boundary(self):
        """Contract v2: meta[12] = last epoch event relative to the stamp; cells last
        fused before it are neither mapped nor used for the alignment."""
        from vitulus_mapping import terrainlayers as tl
        lay = layers(0.0, t_rel=-1.0)
        lay["t_fused"][:, :60] = -5.0                       # western half fused before the event
        g = tf.build_gridmap(T0, (-3.0, -3.0), 0.05, lay, (0, 0, 0), (0, 0, 0, 0), epoch_rel=-2.0)
        m = tf.decode_gridmap(g)
        self.assertEqual(m["version"], 2)
        self.assertAlmostEqual(m["epoch_t"], T0 - 2.0)
        st = tl.TerrainStore(tempfile.mkdtemp(prefix="t30v2_"), DATUM, params={"accept_delay": 0.0})
        try:
            st.fuse(m["win"], m["pose"], m["stamp"], robot=m["robot"])
            info = st.last_batch
            self.assertGreater(info["stale_epoch_cells"], 5000)
            st.close_all()
            e, n = DATUM["utm_e"], DATUM["utm_n"]
            self.assertFalse(np.isfinite(st.sample_utm(e - 1.5, n)["h"][0]))   # west: stale
            self.assertTrue(np.isfinite(st.sample_utm(e + 1.5, n)["h"][0]))    # east: fresh
        finally:
            shutil.rmtree(st.root, ignore_errors=True)
        # NaN = no event since the start; version 1 (C1) still accepted, without index 12
        m = tf.decode_gridmap(tf.build_gridmap(T0, (0, 0), 0.05, layers(0.0), (0, 0, 0), (0, 0, 0, 0)))
        self.assertIsNone(m["epoch_t"])
        m = tf.decode_gridmap(tf.build_gridmap(T0, (0, 0), 0.05, layers(0.0), (0, 0, 0), (0, 0, 0, 0),
                                               version=1))
        self.assertEqual((m["version"], m["epoch_t"]), (1, None))

    def test_storage_order_of_the_contract(self):
        """grid_map stores column-major from the largest x and y: the first value of a
        layer is the cell with the largest x and y (t30-kontrakt.md)."""
        lay = layers(0.0)
        lay["elevation"][119, 119] = 5.0
        g = tf.build_gridmap(T0, (0, 0), 0.05, lay, (0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(float(g.data[0].data[0]), 5.0)

    def test_refused(self):
        def bad(change):
            g = msg(T0)
            change(g)
            return g
        cases = [
            bad(lambda g: g.data[g.layers.index("fuse_meta")].data.__setitem__(0, 0.0)),   # version
            bad(lambda g: g.layers.remove("reason")),
            bad(lambda g: setattr(g.data[0], "data", g.data[0].data[:100])),
            bad(lambda g: setattr(g, "outer_start_index", 5)),
            bad(lambda g: g.data[g.layers.index("fuse_meta")].data.__setitem__(9, np.nan)),  # robot z
            bad(lambda g: setattr(g.info, "resolution", 0.0)),
            object(),
        ]
        for c in cases:
            with self.assertRaises(ValueError):
                tf.decode_gridmap(c)


class Gate(unittest.TestCase):
    def open_gate(self, g, now):
        g.feed_pvt(now, 3, RTK_FLAGS, 14)
        g.feed_loc(now, LOC_OK)

    def test_reasons(self):
        g = tf.QualityGate()
        self.assertFalse(g.check(T0))
        self.assertEqual(g.reason, "no navpvt")
        self.open_gate(g, T0)
        self.assertTrue(g.check(T0))
        g.feed_pvt(T0, 3, 0x01 | (1 << 6), 14)             # RTK float
        self.assertFalse(g.check(T0))
        g.feed_pvt(T0, 3, RTK_FLAGS, 45)                   # hAcc
        self.assertEqual((g.check(T0), g.reason), (False, "hAcc 45 mm"))
        g.feed_pvt(T0, 3, RTK_FLAGS, 14)
        self.assertFalse(g.check(T0 + 2.0))                # navpvt stale
        for key, val, why in (("owner", "dock", "owner dock"), ("gps_good", False, "gps not good"),
                              ("heading_ok", False, "heading not ok"),
                              ("anchor_frozen", True, "anchor frozen"),
                              ("heading_anchor_frozen", True, "anchor frozen")):
            st = dict(LOC_OK)
            st[key] = val
            g.feed_pvt(T0, 3, RTK_FLAGS, 14)
            g.feed_loc(T0, st)
            self.assertEqual((g.check(T0), g.reason), (False, why))

    def test_hold_default_10_s(self):
        g = tf.QualityGate()
        for i in range(25):
            self.open_gate(g, T0 + 0.5 * i)                # open since T0, now T0 + 12
        self.assertFalse(g.ok_for(T0 + 9.0, T0 + 12.0))    # needs open since stamp - 10 s
        self.assertTrue(g.ok_for(T0 + 10.0, T0 + 12.0))

    def test_hold_interval(self):
        g = tf.QualityGate({"gate_hold_s": 1.0})
        self.open_gate(g, T0)
        self.assertFalse(g.ok_for(T0, T0))                 # open only since T0
        self.open_gate(g, T0 + 1.5)
        self.assertTrue(g.ok_for(T0 + 1.0, T0 + 1.5))      # open over [stamp - 1, now]
        g.feed_loc(T0 + 1.6, dict(LOC_OK, heading_ok=False))
        self.open_gate(g, T0 + 1.7)
        self.assertFalse(g.ok_for(T0 + 1.5, T0 + 1.7))     # a closed moment inside the interval


class Core(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="t30d2_")
        self.site = "testsite"
        self.sdir = os.path.join(self.root, self.site)
        os.makedirs(self.sdir)
        self.write_datum(DATUM)
        self.serve(self.site)
        self.logs = []

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_datum(self, d):
        with open(os.path.join(self.sdir, "datum.yaml"), "w") as f:
            yaml.safe_dump(dict(d, version=2), f)

    def serve(self, site):
        with open(os.path.join(self.root, ".last_served"), "w") as f:
            json.dump({"site": site, "raster": "combined"}, f)

    def settings(self, enabled=True):
        path = os.path.join(self.root, "terrain_recorder.json")
        with open(path, "w") as f:
            json.dump({"enabled": enabled}, f)
        return path

    def core(self, enabled=True, cfg=None, **store):
        q = {"accept_delay": 2.0}
        q.update(store)
        cf = {"flush_s": 0.0, "settings_file": self.settings(enabled)}
        cf.update(cfg or {})
        c = tf.RecorderCore(cf, store_params=q,
                            sites_root=self.root, log=lambda lv, tx: self.logs.append((lv, tx)))
        c.poll(T0, force=True)
        return c

    def gate_ok(self, c, now):
        """Good navpvt and loc_status from the last call up to `now` (every 0.5 s, as
        the topics come; a gap longer than their max age closes the gate)."""
        t = getattr(self, "gate_t", None)
        t = now if t is None or t > now else t
        while t < now:
            c.gate.feed_loc(t, LOC_OK)
            c.gate.feed_pvt(t, 3, RTK_FLAGS, 14)
            t += 0.5
        c.gate.feed_loc(now, LOC_OK)
        c.gate.feed_pvt(now, 3, RTK_FLAGS, 14)
        self.gate_t = now

    def drive(self, c, t0, n, x_step=0.25, z=0.0, epoch=0, **kw):
        out = []
        for i in range(n):
            t = t0 + i
            self.gate_ok(c, t + 0.3)
            out.append(c.feed(msg(t, x0=-3.0 + i * x_step, z=z, epoch=epoch, **kw), t + 0.3))
        return out

    def test_record_flush_restart(self):
        c = self.core()
        self.gate_ok(c, T0 - 15)
        res = self.drive(c, T0, 12)
        self.assertEqual(res, ["queued"] * 12)
        self.assertGreaterEqual(c.stats["fused"], 9)        # the first ~2 s wait in the queue
        changed = c.flush(T0 + 20)
        self.assertTrue(changed)
        self.assertTrue(os.path.exists(os.path.join(self.sdir, "terrain", "index.json")))
        st = c.status(T0 + 20)
        self.assertEqual((st["site"], st["gate"]), (self.site, "ok"))
        self.assertIsNotNone(st["fuse_ms_avg"])
        c.close()
        c2 = self.core()
        e, n = E0 + 3.3, N0 - 7.1                        # the map origin, inside the windows
        h = c2.store.sample_utm(e, n)["h"][0]
        self.assertAlmostEqual(float(h), 0.0, delta=0.002)
        self.assertEqual(c2.store.session, 1)                 # a restart is no new session (P1)

    def test_drops(self):
        c = self.core()
        self.assertEqual(c.feed(object(), T0), "decode")
        self.gate_ok(c, T0 - 15)
        self.gate_ok(c, T0)
        self.assertEqual(c.feed(msg(T0 - 10), T0), "stale")
        self.assertEqual(c.feed(msg(T0, pose=None), T0 + 0.2), "no_tf")
        self.assertEqual(c.feed(msg(T0, settling=True), T0 + 0.2), "settling")
        c.gate.feed_loc(T0 + 0.3, dict(LOC_OK, owner="dock"))
        self.assertEqual(c.feed(msg(T0 + 0.3), T0 + 0.4), "gate")
        self.serve("")
        c.poll(T0 + 1, force=True)
        self.assertEqual(c.feed(msg(T0 + 1), T0 + 1.1), "no_site")

    def test_gate_closing_clears_the_queue(self):
        c = self.core()
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 2)                                # queued, not fused yet
        self.assertEqual(len(c.store.pending), 2)
        c.gate.feed_pvt(T0 + 2.2, 3, 0x01 | (1 << 6), 14)    # RTK float
        self.assertEqual(c.feed(msg(T0 + 2), T0 + 2.3), "gate")
        self.assertEqual(c.store.pending, [])
        self.assertEqual(c.stats["fused"], 0)

    def test_epoch_change(self):
        c = self.core()
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 6, epoch=1)
        self.drive(c, T0 + 6, 3, epoch=2)
        self.assertEqual(c.store.stats["epoch_changes"], 1)
        self.assertGreater(c.stats["drop_localisation"], 0)

    def test_datum_change_and_site_switch(self):
        c = self.core()
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 6)
        time.sleep(0.01)
        self.write_datum(dict(DATUM, utm_e=DATUM["utm_e"] + 0.4))
        os.utime(os.path.join(self.sdir, "datum.yaml"), (T0 + 100, T0 + 100))
        c.poll(T0 + 10, force=True)
        self.assertEqual(c.stats["datum_changes"], 1)
        self.assertAlmostEqual(c.store.datum["utm_e"], DATUM["utm_e"] + 0.4)
        self.assertEqual(c.store.ov_keys.size, 0)
        # another site served: the old one is written
        os.makedirs(os.path.join(self.root, "other"))
        self.serve("other")
        c.poll(T0 + 20, force=True)
        self.assertTrue(os.path.exists(os.path.join(self.sdir, "terrain", "index.json")))
        self.assertIsNone(c.store)                          # "other" has no datum yet
        self.assertEqual(c.feed(msg(T0 + 20), T0 + 20.1), "no_site")

    def test_removed_site_not_recreated(self):
        c = self.core()
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 4)
        shutil.rmtree(self.sdir)
        self.serve("")
        c.poll(T0 + 10, force=True)
        self.assertFalse(os.path.exists(self.sdir))

    def test_session_gap(self):
        c = self.core(accept_delay=0.0)
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 3)
        s0 = c.store.session
        c.gate.feed_loc(T0 + 10, dict(LOC_OK, owner="dock"))   # in the dock for an hour
        self.gate_t = T0 + 4000 - 15
        self.gate_ok(c, T0 + 4000 - 15)
        c.feed_busy(T0 + 3999, True)                         # the program starts, then the first batch
        self.drive(c, T0 + 4000, 2)                          # after the long gap: still one new session
        self.assertEqual(c.store.session, s0 + 1)

    def test_sessions_from_the_mowing_program(self):
        c = self.core()
        s0 = c.store.session
        c.feed_busy(T0, True)                               # respawn during a program: latched True first
        self.assertEqual(c.store.session, s0)
        c.feed_busy(T0 + 1, False)
        c.feed_busy(T0 + 2, True)                           # the next program
        self.assertEqual(c.store.session, s0 + 1)
        c.feed_busy(T0 + 3, False)
        c.feed_busy(T0 + 4, True)                           # nothing fused in between: same session
        self.assertEqual(c.store.session, s0 + 1)
        c.close()
        self.assertEqual(self.core().store.session, s0 + 1)

    def odom_pair(self, c, t0, n, slew=0.0, yaw_slew=0.0):
        """Driving 0.3 m/s along x; /odometry/odom gets an extra slew (m/s, rad/s)."""
        for i in range(n):
            t = t0 + 0.05 * i
            d = 0.3 * 0.05 * i                              # the wheel EKF: own frame, heading 0.2 rad
            c.feed_ref(t, 10.0 + d * math.cos(0.2), 5.0 + d * math.sin(0.2), 0.2)
            c.feed_odom(t, 0.3 * 0.05 * i, slew * 0.05 * i, yaw_slew * 0.05 * i)

    def test_correction_only_counted_by_default(self):
        c = self.core(accept_delay=2.0)
        c.corr_active = True
        self.odom_pair(c, T0 - 15, 330)
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 2)
        self.odom_pair(c, T0 + 1.5, 40, slew=0.10)
        self.assertGreater(c.corr.n_events, 0)
        self.assertGreater(c.stats["corrections"], 0)
        self.assertIsNone(c.store.fresh_after)
        self.assertEqual(len(c.store.pending), 2)
        self.gate_ok(c, T0 + 3.3)
        self.assertEqual(c.feed(msg(T0 + 3), T0 + 3.3), "queued")

    def test_correction_enforced_freshness_boundary(self):
        c = self.core(cfg={"corr_enforce": True}, accept_delay=2.0)
        c.corr_active = True
        self.odom_pair(c, T0 - 15, 330)                      # quiet up to T0 + 1.5: no event
        self.assertEqual(c.corr.n_events, 0)
        self.gate_ok(c, T0 - 15)
        self.assertEqual(self.drive(c, T0, 2), ["queued", "queued"])
        self.odom_pair(c, T0 + 1.5, 40, slew=0.10)            # pulled 10 cm/s sideways
        ev = c.corr.events[0]
        self.assertGreaterEqual(c.store.fresh_after, ev + 3.0)
        self.assertEqual(len(c.store.pending), 2)            # the queue stays, its stale cells go at processing
        self.gate_ok(c, T0 + 3.3)
        self.assertEqual(c.feed(msg(T0 + 3), T0 + 3.3), "correction")
        c.store.drain()
        self.assertGreater(c.store.last_batch["stale_epoch_cells"], 10000)
        self.assertEqual(c.store.ov_keys.size, 0)            # nothing fresh entered the map
        # no comparison around the stamp (reference odometry missing): fails closed
        self.gate_ok(c, T0 + 20.3)
        self.assertEqual(c.feed(msg(T0 + 20), T0 + 20.3), "corr_nodata")
        # yaw slew 1 deg/s is an event too; a quiet time after the settle time is not
        c2 = self.core()
        self.odom_pair(c2, T0, 60, yaw_slew=math.radians(1.0))
        self.assertGreater(c2.corr.n_events, 0)
        self.assertFalse(c2.corr.unsettled(T0 + 3 + 3.5 + 1))

    def test_program_end_writes_and_requests_export(self):
        c = self.core(accept_delay=0.0)
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 3)
        self.assertGreater(c.store.ov_keys.size, 0)
        c.feed_busy(T0 + 4, True)
        c.feed_busy(T0 + 5, False)                           # program over
        self.assertEqual(c.store.ov_keys.size, 0)
        self.assertEqual(c.export_req, "program_end")
        self.assertTrue(os.path.exists(os.path.join(self.sdir, "terrain", "index.json")))

    def test_batch_log(self):
        path = os.path.join(self.root, "batches.csv")
        c = tf.RecorderCore({"flush_s": 0.0, "batch_log": path, "settings_file": self.settings()},
                            store_params={"accept_delay": 0.0}, sites_root=self.root)
        c.poll(T0, force=True)
        self.gate_ok(c, T0 - 15)
        self.drive(c, T0, 3)
        c.close()
        with open(path) as f:
            rows = f.read().splitlines()
        self.assertEqual(rows[0], "kind,stamp,now,result,epoch,settling,map_x,map_y")
        self.assertEqual(sum(r.startswith("msg,") for r in rows), 3)
        self.assertEqual(sum(r.startswith("fused,") for r in rows), 3)

    def test_subscription_hysteresis(self):
        c = self.core()
        self.assertFalse(c.want_subscription(T0))
        self.gate_ok(c, T0)
        self.assertFalse(c.want_subscription(T0 + 1))
        self.gate_ok(c, T0 + 2.5)
        self.assertTrue(c.want_subscription(T0 + 2.5))
        c.gate.feed_loc(T0 + 3, dict(LOC_OK, owner="dock"))
        self.assertTrue(c.want_subscription(T0 + 10))       # closed 7 s: stay subscribed
        self.assertFalse(c.want_subscription(T0 + 34))      # closed 31 s

    def test_mowed_from_coverage(self):
        g = worldlayers.CoverageGrid()
        g.stamp(-5.0, 0.0, 5.0, 0.0, 2.0, T0 + 50)          # map frame, mowed at T0 + 50
        g.save(self.sdir, self.site, DATUM)
        c = self.core()
        cov = c.store.mowed
        self.assertIsNotNone(cov)
        e, n = E0 + 3.3, N0 - 7.1                            # map (0, 0)
        self.assertEqual(float(cov(np.array([e]), np.array([n]))[0]), T0 + 50)
        self.assertEqual(float(cov(np.array([e]), np.array([n + 4.0]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
