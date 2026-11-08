#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.grass: synthetic depth images of a D435
mounted like on the robot, ray-cast against a height field.

Run: python3 -m pytest test/test_grass.py   (from the package root)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import bundle, grass  # noqa: E402

# the robot's camera: base_link -> d435_depth_optical_frame (TF, 2026-11)
K = np.array([[388.5, 0.0, 320.0], [0.0, 388.5, 240.0], [0.0, 0.0, 1.0]])
T_BC = (0.21229144, 0.0415, 0.29275525)
Q_BC = (-0.49385429, 0.49385429, -0.50607108, 0.50607108)


def rays(stride=4):
    return grass.Rays(K, grass.quat_to_rot(*Q_BC), T_BC, (480, 640), stride=stride)


def texture(x, y, cell=0.03, seed=0.0):
    """Static per-world-position value in [0, 1) (grass blades)."""
    ix = np.floor(x / cell)
    iy = np.floor(y / cell)
    v = np.sin(ix * 12.9898 + iy * 78.233 + seed) * 43758.5453
    return v - np.floor(v)


def render(R_ray, surface, pose=(0.0, 0.0, 0.0), rp=(0.0, 0.0), z_robot=0.0, noise_mm=1.5, rng=None):
    """Depth image (uint16 mm, only the decimated pixels filled) of `surface`
    (callable xs, ys -> z world) seen from a robot at pose/rp/z_robot."""
    R = grass.rot_rpy(rp[0], rp[1], pose[2])
    o = R @ R_ray.t + np.array([pose[0], pose[1], z_robot])
    d = R_ray.dirs @ R.T
    lo = np.full(len(d), 0.15)
    hi = np.full(len(d), 6.0)

    def above(s):
        p = o + s[:, None] * d
        return p[:, 2] - surface(p[:, 0], p[:, 1])

    hit = above(hi) < 0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a = above(mid) > 0
        lo = np.where(a, mid, lo)
        hi = np.where(a, hi, mid)
    s = 0.5 * (lo + hi)
    if rng is not None and noise_mm:
        s = s + rng.normal(0.0, noise_mm / 1000.0, len(s)) * (s / 1.2) ** 2
    depth = np.zeros((480, 640), np.uint16)
    vals = np.where(hit & (s < 5.0), np.round(s * 1000.0), 0).astype(np.uint16)
    depth[R_ray.idx] = vals
    return depth


def flat(_x=None, _y=None):
    return np.zeros_like(_x)


def grass_on(ground, height_m, lo=0.6):
    """Carpet: ground + height * (lo..1) per 3 cm tuft."""
    def f(x, y):
        return ground(x, y) + height_m * (lo + (1.0 - lo) * texture(x, y))
    return f


RNG = np.random.default_rng(5)


def carpet(ground, grass_mm, n=6000, lo=0.3):
    """Body-frame points of a corridor (no ray casting): fast scenarios."""
    x = RNG.uniform(0.6, 1.8, n)
    y = RNG.uniform(-0.4, 0.4, n)
    z = ground(x, y) + grass_mm / 1000.0 * RNG.uniform(lo, 1.0, n)
    return np.c_[x, y, z]


def flat_ground(x, y):
    return 0.0 * x


def calibrated(measured=True):
    c = grass.Calib()
    c.calibrated = True
    c.imu_offset_measured = measured
    return c


class Frame(unittest.TestCase):
    """One frame, robot standing on flat ground."""

    @classmethod
    def setUpClass(cls):
        cls.R = rays()
        cls.rng = np.random.default_rng(1)

    def run_frames(self, surface, n=3, cut_mm=40.0, dem=None, calib=None):
        eng = grass.GrassEngine(calib=calib)
        out = None
        for i in range(n):
            P = self.R.points(render(self.R, surface, rng=self.rng))
            out = eng.frame(100.0 + 0.25 * i, P, (0.0, 0.0, 0.0), (0.0, 0.0), cut_mm, dem)
        return out

    def test_points_land_on_flat_ground(self):
        P = self.R.points(render(self.R, flat, noise_mm=0))
        near = P[(P[:, 0] > 0.6) & (P[:, 0] < 2.0)]
        self.assertGreater(len(near), 2000)
        self.assertLess(np.abs(near[:, 2]).max(), 0.002)
        self.assertGreater(P[:, 0].min(), 0.35)        # ground visible from ~0.45 m

    def test_concrete_is_bare_ground(self):
        o = self.run_frames(flat)
        self.assertEqual(o["terrain_source"], "surface")
        self.assertEqual(o["class"], "below_cut")
        self.assertEqual(o["height_mm"], 0)
        self.assertLess(o["rough_mm"], 6.0)

    def test_tall_grass_on_flat_ground(self):
        o = self.run_frames(grass_on(flat, 0.10), cut_mm=40.0)
        self.assertEqual(o["terrain_source"], "extrapolated")
        self.assertEqual(o["class"], ">+4cm")
        self.assertAlmostEqual(o["height_mm"], 96, delta=12)
        self.assertGreater(o["rough_mm"], 10.0)

    def test_short_grass_is_below_cut(self):
        o = self.run_frames(grass_on(flat, 0.03), cut_mm=40.0)
        self.assertEqual(o["class"], "below_cut")
        self.assertLess(o["height_mm"], 45)

    def test_concrete_step_is_terrain_not_grass(self):
        step = lambda x, y: np.where(x > 1.0, 0.05, 0.0)       # noqa: E731
        o = self.run_frames(step)
        # a step across the corridor tilts the whole carpet: terrain, never grass
        self.assertIn(o["class"], ("unknown", "below_cut"))
        self.assertNotIn(o["class"], ("+2-4cm", ">+4cm"))

    def test_grass_on_rising_slope_is_unknown_without_ground(self):
        ramp = lambda x, y: np.maximum(0.0, x - 0.6) * math.tan(math.radians(8.0))  # noqa: E731
        o = self.run_frames(grass_on(ramp, 0.03), cut_mm=40.0)
        self.assertEqual(o["class"], "unknown")
        self.assertEqual(o["reason"], "terrain_change")
        self.assertIsNone(o["height_mm"])
        self.assertGreater(o["slope_deg"], 4.0)

    def test_grass_on_slope_with_dem(self):
        ramp = lambda x, y: np.maximum(0.0, x - 0.6) * math.tan(math.radians(8.0))  # noqa: E731
        o = self.run_frames(grass_on(ramp, 0.03), cut_mm=40.0, dem=ramp)
        self.assertEqual(o["terrain_source"], "dem")
        self.assertEqual(o["class"], "below_cut")
        self.assertAlmostEqual(o["height_mm"], 29, delta=10)
        self.assertGreater(o["slope_deg"], 6.0)

    def test_grass_edge_beside_mown_lane_is_grass(self):
        # left lane mown (2 cm), the rest tall (12 cm) from x = 1.0: not terrain
        def f(x, y):
            tall = (x > 1.0) & (y < 0.1)
            return np.where(tall, 0.12, 0.02) * (0.6 + 0.4 * texture(x, y))
        o = self.run_frames(f, cut_mm=40.0)
        # lanes differ sharply and the high ones are rough: grass or a grassy
        # kerb, one frame cannot tell -> unknown, never "mown"
        self.assertNotEqual(o["reason"], "terrain_change")
        self.assertNotEqual(o["class"], "below_cut")

    def test_calibration_zero_removes_camera_tilt(self):
        # the camera looks 1 deg more down than TF says: concrete appears tilted
        tilted = lambda x, y: -0.0175 * x + 0.01               # noqa: E731
        raw = self.run_frames(grass_on(tilted, 0.08), cut_mm=10.0)
        cal = grass.Calib({"offset_mm": 10.0, "pitch_deg": -1.0, "rough_bare_mm": 4.0})
        o = self.run_frames(grass_on(tilted, 0.08), cut_mm=10.0, calib=cal)
        self.assertAlmostEqual(o["height_mm"], 77, delta=8)
        self.assertGreater(abs(raw["raw_mm"] - 77), abs(o["raw_mm"] - 77) + 5)


class Deferred(unittest.TestCase):
    """Driving over terrain that rises: the deferred map cells measure the grass,
    not the terrain."""

    def drive(self, ground, grass_m, x_end=4.0, speed=0.3, rate=4.0, dem=None, calib=None):
        R_ray = rays()
        rng = np.random.default_rng(2)
        cells = []
        eng = grass.GrassEngine(calib=calib or calibrated(), on_cell=lambda *a: cells.append(a))
        surf = grass_on(ground, grass_m) if not callable(grass_m) else grass_m
        t, x = 0.0, 0.0
        outs = []
        while x < x_end:
            zf = float(ground(np.array([x + 0.125]), np.array([0.0]))[0])
            zr = float(ground(np.array([x - 0.125]), np.array([0.0]))[0])
            pitch = -math.atan2(zf - zr, 0.25)              # nose up on a rise
            z = 0.5 * (zf + zr)
            depth = render(R_ray, surf, (x, 0.0, 0.0), (0.0, pitch), z, rng=rng)
            outs.append(eng.frame(t, R_ray.points(depth), (x, 0.0, 0.0), (0.0, pitch), 40.0, dem))
            t += 1.0 / rate
            x += speed / rate
        return eng, cells, outs

    def test_cells_resolved_by_wheels_measure_grass_on_a_ramp(self):
        ramp = lambda x, y: np.maximum(0.0, x - 2.0) * math.tan(math.radians(6.0))  # noqa: E731
        eng, cells, outs = self.drive(ramp, 0.08)
        self.assertGreater(len(cells), 15)
        self.assertTrue(all(c[4] == grass.SRC_WHEELS for c in cells))
        h = np.array([c[2] for c in cells])
        # canopy p90 of a 0.6..1.0 tuft field = 0.96 * 80 mm; no cell reads the ramp
        self.assertAlmostEqual(float(np.median(h)), 77, delta=10)
        self.assertLess(np.abs(h - 77).max(), 30)
        on_ramp = [c for c in cells if c[0] > 2.4]
        self.assertTrue(on_ramp)
        # the immediate output flagged the slope change at least once instead of
        # calling it tall grass
        reasons = [o["reason"] for o in outs]
        self.assertIn("terrain_change", reasons)
        self.assertFalse(any(o["class"] == ">+4cm" and o["height_mm"] > 140 for o in outs))

    def test_wheel_ground_feeds_the_immediate_estimate(self):
        # back and forth: the second pass sees cells the wheels already know
        eng, cells, outs = self.drive(flat, 0.05, x_end=1.0)
        R_ray = rays()
        surf = grass_on(flat, 0.05)
        t = outs[-1]["stamp"] + 0.25
        o = None
        for x in (0.9, 0.6, 0.3, 0.0, -0.3):          # reverse: pose behind, corridor over the track
            depth = render(R_ray, surf, (x, 0.0, 0.0), (0.0, 0.0), 0.0)
            o = eng.frame(t, R_ray.points(depth), (x, 0.0, 0.0), (0.0, 0.0), 40.0)
            t += 0.25
        self.assertEqual(o["terrain_source"], "wheels")
        self.assertAlmostEqual(o["height_mm"], 48, delta=10)

    def test_pending_expire_to_dem(self):
        ramp = lambda x, y: np.maximum(0.0, x - 0.5) * math.tan(math.radians(5.0))  # noqa: E731
        eng, cells, _outs = self.drive(ramp, 0.06, x_end=0.6, dem=ramp)
        n_wheels = len(cells)
        self.assertGreater(len(eng.pending), 5)
        eng.flush_pending(1e6, ramp)
        dem_cells = cells[n_wheels:]
        self.assertTrue(dem_cells)
        self.assertTrue(all(c[4] == grass.SRC_DEM for c in dem_cells))
        h = np.array([c[2] for c in dem_cells])
        self.assertAlmostEqual(float(np.median(h)), 58, delta=12)
        self.assertEqual(len(eng.pending), 0)

    def test_track_altitude_follows_ramp(self):
        tr = grass.Track(dict(grass.DEFAULTS))
        slope = math.radians(10.0)
        for i in range(21):
            x = i * 0.1
            tr.update(i * 0.1, x, 0.0, grass.rot_rpy(0.0, -slope, 0.0))
        self.assertAlmostEqual(tr.h, 2.0 * math.tan(slope), delta=0.01)
        tr.update(2.1, 10.0, 0.0, grass.rot_rpy(0, 0, 0))     # jump restarts
        self.assertEqual(tr.h, 0.0)
        self.assertEqual(tr.epoch, 1)


class Calibration(unittest.TestCase):
    def record(self, label, true_mm, x, plane, raw_p90=None, rough=None, pitch=None, yaw=None,
               created="2026-11-08 10:00:00"):
        strips = []
        for xs in [round(0.5 + 0.1 * i, 2) for i in range(16)]:
            for y in (-0.133, 0.0, 0.133):
                strips.append({"x": xs, "y": y, "median_mm": plane(xs, y), "p90_mm": plane(xs, y) + 1,
                               "noise_mm": 0.5, "sigma_mm": 1.0 + max(0.0, xs - 1.5) * 20,
                               "valid": 0.9 if xs < 1.9 else 0.3})
        sample = {"x": x, "p90_mm": plane(x, 0.0) + (raw_p90 or 0.0)}
        return {"label": label, "true_mm": true_mm, "x": x, "strips": strips, "sample": sample,
                "rough_mm": rough, "imu_pitch_deg": pitch, "yaw_deg": yaw, "created": created}

    def test_fit_recovers_zero_and_scale(self):
        plane = lambda x, y: 12.0 + math.tan(math.radians(-0.8)) * x * 1000 + 0.0 * y  # noqa: E731
        k, q = 1.25, 0.0          # concrete is the point (0, 0): no free bias
        recs = [self.record("beton", 0.0, 1.0, plane, rough=2.5)]
        for x in (0.8, 1.2, 1.6):
            recs.append(self.record("posekano", 40.0, x, plane, (40.0 - q) / k, rough=9.0))
            recs.append(self.record("vysoka", 120.0, x, plane, (120.0 - q) / k, rough=18.0))
        cal, rep = grass.fit_calibration(recs, allow_single_direction=True)
        self.assertAlmostEqual(cal.offset_mm, 12.0, delta=0.5)
        self.assertAlmostEqual(cal.pitch_deg, -0.8, delta=0.05)
        self.assertAlmostEqual(cal.scale, k, delta=0.01)
        self.assertEqual(cal.bias_mm, 0.0)
        self.assertTrue(all(abs(e["err_mm"]) < 1.0 for e in rep["errors"]))
        self.assertFalse([w for w in rep["warnings"] if "off the line" in w])
        self.assertTrue(2.5 < cal.rough_bare_mm < 9.0)
        # sigma > 5 mm from x = 1.8 on: the corridor ends before it
        self.assertEqual(rep["recommended_corridor_m"][1], 1.75)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calib.json")
            cal.save(p)
            back = grass.Calib.load(p)
            self.assertTrue(back.calibrated)
            self.assertAlmostEqual(back.scale, k, delta=0.01)
            self.assertAlmostEqual(back.height(100.0), cal.height(100.0))

    def test_two_labels_off_the_line_warn(self):
        plane = lambda x, y: 0.0  # noqa: E731
        recs = [self.record("beton", 0.0, 1.0, plane),
                self.record("posekano", 40.0, 1.0, plane, 20.0),     # 2x
                self.record("vysoka", 120.0, 1.0, plane, 100.0)]     # 1.2x
        cal, rep = grass.fit_calibration(recs, allow_single_direction=True)
        self.assertTrue([w for w in rep["warnings"] if "off the line" in w], rep)
        self.assertIn("posekano", rep["residual_mm"])

    def test_imu_pitch_of_the_record_is_corrected(self):
        # robot on the lawn stood 1 deg more nose-down than on the concrete:
        # level ground appears at x*tan(1 deg) = 17 mm at 1 m, not grass
        plane = lambda x, y: 0.0  # noqa: E731
        tilt = 1000.0 * math.tan(math.radians(1.0))
        recs = [self.record("beton", 0.0, 1.0, plane, pitch=0.5)]
        recs.append(self.record("posekano", 40.0, 1.0, plane, 40.0 / 1.25 + tilt, pitch=1.5))
        recs.append(self.record("vysoka", 120.0, 1.0, plane, 120.0 / 1.25 + tilt, pitch=1.5))
        cal, rep = grass.fit_calibration(recs, allow_single_direction=True)
        self.assertAlmostEqual(cal.scale, 1.25, delta=0.01)

    def test_imu_offset_from_concrete_both_ways(self):
        # concrete sloping 1 deg, IMU 0.4 deg off: 1.4 one way, -0.6 the other
        plane = lambda x, y: 0.0  # noqa: E731
        recs = [self.record("beton", 0.0, 1.0, plane, pitch=1.4, yaw=10.0),
                self.record("beton", 0.0, 1.0, plane, pitch=-0.6, yaw=-172.0),
                self.record("posekano", 40.0, 1.0, plane, 32.0, pitch=0.4)]
        cal, rep = grass.fit_calibration(grass.latest_records(recs))
        self.assertAlmostEqual(cal.imu_pitch_offset_deg, 0.4, delta=0.01)
        self.assertTrue(cal.imu_offset_measured)
        one, _ = grass.fit_calibration(recs[:1] + recs[2:], allow_single_direction=True)
        self.assertEqual(one.imu_pitch_offset_deg, 0.0)
        self.assertFalse(one.imu_offset_measured)
        # one direction only is an error unless asked for explicitly
        with self.assertRaises(ValueError):
            grass.fit_calibration(recs[:1] + recs[2:])

    def test_latest_record_per_label_and_distance(self):
        plane = lambda x, y: 0.0  # noqa: E731
        a = self.record("posekano", 40.0, 1.0, plane, 10.0, created="2026-11-08 10:00:00")
        b = self.record("posekano", 42.0, 1.0, plane, 10.0, created="2026-11-08 11:00:00")
        c = self.record("posekano", 40.0, 1.6, plane, 10.0)
        got = grass.latest_records([b, a, c])
        self.assertEqual(len(got), 2)
        self.assertIn(b, got)

    def test_fit_needs_concrete(self):
        plane = lambda x, y: 0.0  # noqa: E731
        with self.assertRaises(ValueError):
            grass.fit_calibration([self.record("posekano", 40.0, 1.0, plane, 40.0)])

    def test_single_grass_label_goes_through_zero(self):
        plane = lambda x, y: 0.0  # noqa: E731
        recs = [self.record("beton", 0.0, 1.0, plane), self.record("posekano", 50.0, 1.0, plane, 40.0)]
        cal, _rep = grass.fit_calibration(recs, allow_single_direction=True)
        self.assertAlmostEqual(cal.scale, 1.25, delta=0.01)
        self.assertEqual(cal.bias_mm, 0.0)

    def test_missing_file_is_identity(self):
        c = grass.Calib.load("/nonexistent/calib.json")
        self.assertFalse(c.calibrated)
        self.assertIn("missing", c.error)
        self.assertEqual(c.height(50.0), 50.0)
        o = grass.GrassEngine(calib=c).frame(0.0, carpet(flat_ground, 20.0))
        self.assertEqual(o["reason"], "uncalibrated")

    def test_bad_file_reports(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calib.json")
            with open(p, "w") as f:
                f.write("{broken")
            c = grass.Calib.load(p)
            self.assertFalse(c.calibrated)
            self.assertIn("bad", c.error)
            grass.Calib({"scale": 1.1}).save(p)
            self.assertEqual(os.listdir(d), ["calib.json"])        # no tmp left behind
            self.assertTrue(grass.Calib.load(p).calibrated)


class Classes(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(grass.classify(None, 40), "unknown")
        self.assertEqual(grass.classify(55, 40), "below_cut")
        self.assertEqual(grass.classify(70, 40), "+2-4cm")
        self.assertEqual(grass.classify(81, 40), ">+4cm")

    def test_confidence_is_class_probability(self):
        self.assertGreater(grass.class_probability(30, 10, 80), 0.99)
        self.assertAlmostEqual(grass.class_probability(100, 5, 80), 0.5, delta=0.01)   # on an edge
        self.assertLess(grass.class_probability(110, 30, 80), 0.3)


class Layer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        os.makedirs(bundle.site_dir("a"))

    def tearDown(self):
        bundle.SITES_ROOT = self.old
        self.tmp.cleanup()

    def test_grid_median_save_load_layer(self):
        datum = {"utm_e": 493000.0, "utm_n": 5540000.0, "yaw_rad": 0.0}
        g = grass.GrassGrid()
        for h in (30, 40, 50, 200):
            g.add(1.05, 2.05, h, 1790000000 + h, grass.SRC_WHEELS)
        g.add(-3.0, 0.5, 120, 1790000000, grass.SRC_DEM)
        med = g.median_mm()
        r, c = int((2.05 - g.oy) / g.res), int((1.05 - g.ox) / g.res)
        self.assertIn(int(med[r, c]), (45, 55))
        g.save(bundle.site_dir("a"), "a", datum)
        back = grass.GrassGrid.load(bundle.site_dir("a"), datum)
        rows = grass.query_grass(back, points=[(1.05, 2.05), (9.0, 9.0)])
        self.assertEqual(rows[0]["count"], 4)
        self.assertEqual(rows[0]["last_mm"], 200)
        self.assertEqual(rows[0]["source"], "wheels")
        self.assertIsNone(rows[1]["median_mm"])
        lay = grass.grass_layer(back, "a", 1790000500.0)
        self.assertGreater(lay["w"], 0)
        self.assertTrue(lay["legend"]["title"].startswith("Grass vs cut"))
        self.assertEqual([st["key"] for st in lay["legend"]["stops"]], ["below_cut", "+2-4cm", ">+4cm"])
        st = grass.query_grass(back, polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)])
        self.assertEqual(st["cells_seen"], 2)

    def test_layer_query_reads_grass(self):
        from vitulus_mapping import worldlayers as wl
        with open(os.path.join(bundle.site_dir("a"), "datum.yaml"), "w") as f:
            f.write("utm_e: 493000.0\nutm_n: 5540000.0\nyaw_rad: 0.0\n")
        datum = wl.read_datum("a")
        self.assertIsNotNone(datum)
        g = grass.GrassGrid()
        g.add(0.5, 0.5, 70, 1790000000, grass.SRC_WHEELS)
        g.save(bundle.site_dir("a"), "a", datum)
        r = wl.layer_query("a", "grass", points=[[0.5, 0.5]])
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"][0]["median_mm"], 75)


class Robust(unittest.TestCase):
    """Scenarios of the independent review (2026-11-08)."""

    def test_pose_jump_after_descent_does_not_mix_altitudes(self):
        e = grass.GrassEngine(calib=calibrated())
        p, t = math.radians(5.0), 0.0
        for i in range(60):                       # 6 m down a 5 deg slope
            e.frame(t, carpet(flat_ground, 20.0), (i * 0.1, 0.0, 0.0), (0.0, p))
            t += 0.25
        self.assertLess(e.track.h, -0.4)
        for i in range(4):                        # relocalised 4.5 m back: new epoch
            o = e.frame(t, carpet(flat_ground, 20.0), (1.5 + i * 0.05, 0.0, 0.0), (0.0, p))
            t += 0.25
        self.assertEqual(e.track.epoch, 1)
        self.assertNotEqual(o["terrain_source"], "wheels")
        self.assertEqual(o["class"], "below_cut")
        self.assertLess(o["height_mm"], 35)

    def test_pitch_bias_there_and_back(self):
        # IMU 0.7 deg off: 15 m out integrates 18 cm of false altitude; coming
        # back on the next stripe must not read it as grass
        e = grass.GrassEngine(calib=calibrated())
        b, t, outs = math.radians(0.7), 0.0, []
        for i in range(150):
            e.frame(t, carpet(flat_ground, 20.0), (i * 0.1, 0.0, 0.0), (0.0, -b))
            t += 0.25
        for i in range(150):
            outs.append(e.frame(t, carpet(flat_ground, 20.0), (15 - i * 0.1, 0.25, math.pi), (0.0, -b)))
            t += 0.25
        # uncorrected: the error stays bounded (wheel ground <= 3 m back), no false class
        self.assertEqual({o["class"] for o in outs} - {"unknown"}, {"below_cut"})
        self.assertLess(max(o["height_mm"] or 0 for o in outs), 75)
        # with the offset from the calibration (beton both ways) it is gone
        cal = calibrated()
        cal.imu_pitch_offset_deg = -0.7
        e = grass.GrassEngine(calib=cal)
        t, outs = 0.0, []
        for i in range(150):
            e.frame(t, carpet(flat_ground, 20.0), (i * 0.1, 0.0, 0.0), (0.0, -b))
            t += 0.25
        for i in range(150):
            outs.append(e.frame(t, carpet(flat_ground, 20.0), (15 - i * 0.1, 0.25, math.pi), (0.0, -b)))
            t += 0.25
        self.assertLess(abs(e.track.h), 0.01)
        self.assertLess(max(o["height_mm"] for o in outs), 30)

    def test_rocking_body_uses_smoothed_slope(self):
        e = grass.GrassEngine(calib=calibrated())
        t, hs = 0.0, []
        for i in range(80):
            p = math.radians(1.5 * math.sin(2 * math.pi * 0.7 * t))
            P = carpet(flat_ground, 45.0)
            P[:, 2] += P[:, 0] * math.sin(p)       # body nose-down by p: level ground rises ahead
            o = e.frame(t, P, (100 + i * 0.075, 0.0, 0.0), (0.0, p))
            t += 0.25
            if i > 8:
                hs.append(o["height_mm"])
                self.assertEqual(o["class"], "below_cut")
        self.assertLess(np.ptp(hs), 15)
        self.assertAlmostEqual(float(np.median(hs)), 42, delta=8)

    def test_gentle_ramp_below_the_limit(self):
        def run(deg, grass_mm, lo=0.3):
            e = grass.GrassEngine(calib=calibrated())
            f = lambda x, y: np.maximum(0, x - 0.2) * math.tan(math.radians(deg))  # noqa: E731
            for t in range(4):
                o = e.frame(t * 0.25, carpet(f, grass_mm, lo=lo))
            return o
        # bare ramp: the lowest lane is smooth and on the ground -> terrain, taken off
        o = run(2.4, 2.0, lo=0.9)
        self.assertEqual(o["class"], "below_cut")
        self.assertEqual(o["height_mm"], 0)
        # under grass the same tilt may be grass growing taller: it never lowers
        # the height (only widens the uncertainty); a dip is counted in
        flat = run(0.0, 45.0)["height_mm"]
        for deg in (2.4, -2.4):
            o = run(deg, 45.0)
            self.assertTrue(o["height_mm"] is None or o["height_mm"] >= flat - 8, (deg, flat, o))

    def test_grass_growing_taller_ahead_is_not_lowered(self):
        # review 2: 70 -> 100 and 90 -> 125 mm smoothly ahead, cut 60: tall or unknown
        for a, b in ((70, 100), (90, 125)):
            for lo in (0.3, 0.95):
                e = grass.GrassEngine(calib=calibrated())
                g = lambda x, y, a=a, b=b: a + (b - a) * np.clip((x - 0.8) / 0.8, 0, 1)  # noqa: E731
                for t in range(6):
                    x = RNG.uniform(0.6, 1.8, 6000)
                    y = RNG.uniform(-0.4, 0.4, 6000)
                    P = np.c_[x, y, g(x, y) / 1000.0 * RNG.uniform(lo, 1.0, 6000)]
                    o = e.frame(t * 0.25, P, (t * 0.05, 0.0, 0.0), (0.0, 0.0), cut_mm=60.0)
                self.assertIn(o["class"], ("+2-4cm", ">+4cm", "unknown"), (a, b, lo, o))

    def test_body_already_on_the_new_slope(self):
        # flat track behind, the body now pitched 4 deg nose-up onto a rise, the
        # carpet lies with the body: no terrain change (cap2 frames 24-44)
        e = grass.GrassEngine(calib=calibrated())
        t = 0.0
        for i in range(20):
            e.frame(t, carpet(flat_ground, 20.0), (i * 0.05, 0.0, 0.0), (0.0, 0.0))
            t += 0.25
        p = math.radians(-4.0)
        for i in range(3):
            o = e.frame(t, carpet(flat_ground, 20.0), (1.0 + i * 0.05, 0.0, 0.0), (0.0, p))
            t += 0.25
        self.assertNotEqual(o["reason"], "terrain_change")
        self.assertEqual(o["class"], "below_cut")

    def test_kerb_left(self):
        for kerb in (0.05, 0.08):
            # grass on the kerb too: the whole strip is lifted (its bottom with
            # its top), terrain beside the track -> unknown, never tall grass
            e = grass.GrassEngine(calib=calibrated())
            f = lambda x, y: np.where(y > 0.12, kerb, 0.0)       # noqa: E731
            for t in range(4):
                o = e.frame(t * 0.25, carpet(f, 20.0), cut_mm=40.0)
            self.assertEqual(o["class"], "unknown", (kerb, o))
            self.assertTrue(o.get("lane_step"), o)
            # bare kerb (pavement): the grass is read from the low lanes
            e = grass.GrassEngine(calib=calibrated())
            P = carpet(flat_ground, 20.0)
            kb = P[:, 1] > 0.12
            P[kb, 2] = kerb + RNG.normal(0, 0.001, kb.sum())
            for t in range(4):
                o = e.frame(t * 0.25, P, cut_mm=40.0)
            self.assertTrue(o.get("kerb"), o)
            self.assertEqual(o["class"], "below_cut")
            self.assertLess(o["height_mm"], 30)

    def test_ditch_does_not_hide_tall_grass(self):
        ditch = lambda x, y: np.where((x > 0.7) & (x < 1.7), -0.10, 0.0)   # noqa: E731

        def surf(x, y):
            tall = (x > 0.7) & (x < 1.7)
            return ditch(x, y) + np.where(tall, 0.12, 0.02) * RNG.uniform(0.5, 1.0, len(x))
        e = grass.GrassEngine(calib=calibrated())
        for t in range(4):
            x = RNG.uniform(0.6, 1.8, 6000)
            y = RNG.uniform(-0.4, 0.4, 6000)
            o = e.frame(t * 0.25, np.c_[x, y, surf(x, y)], (0.0, 0.0, 0.0), (0.0, 0.0),
                        cut_mm=40.0, dem=ditch)
        self.assertEqual(o["terrain_source"], "dem")
        self.assertEqual(o["class"], ">+4cm", o)

    def test_ditch_drive_cells_measure_grass(self):
        ditch = lambda x, y: np.where((x > 1.5) & (x < 2.5), -0.08, 0.0)   # noqa: E731
        _eng, cells, _outs = Deferred().drive(ditch, 0.10, x_end=3.0)
        inside = [c[2] for c in cells if 1.7 < c[0] < 2.3]
        self.assertTrue(inside)
        self.assertGreater(float(np.median(inside)), 80)

    def test_memory_is_bounded(self):
        e = grass.GrassEngine(calib=calibrated())
        for i in range(2000):
            e.track.update(i * 0.25, 0.0, 0.0, np.eye(3))
        self.assertLessEqual(len(e.track.hist), e.cfg["track_hist_max"])
        e = grass.GrassEngine(cfg={"ground_max": 200}, calib=calibrated())
        for i in range(3000):
            e._stamp_wheels(i * 0.1, (i % 300) * 0.2, (i // 300) * 0.5, np.eye(3), 0.0)
        self.assertLessEqual(len(e.ground), 200 + 5)
        g = grass.GrassGrid()
        self.assertTrue(g.add(0.0, 0.0, 50, 1, grass.SRC_WHEELS))
        self.assertFalse(g.add(500.0, 0.0, 50, 1, grass.SRC_WHEELS))   # a jump does not grow it
        self.assertLessEqual(max(g.h, g.w), grass.GrassGrid.MAX_CELLS)


class Review2(unittest.TestCase):
    """Second independent review (2026-11-08)."""

    def test_mown_uncut_edge_reports_the_tall_lane(self):
        for side in (1, -1):
            e = grass.GrassEngine(calib=calibrated())
            outs = []
            for t in range(8):
                x = RNG.uniform(0.6, 1.8, 6000)
                y = RNG.uniform(-0.4, 0.4, 6000)
                tall = side * y > 0.12
                z = np.where(tall, 0.130, 0.030) * RNG.uniform(0.3, 1.0, 6000)
                outs.append(e.frame(t * 0.25, np.c_[x, y, z], (t * 0.05, 0.0, 0.0), (0.0, 0.0),
                                    cut_mm=60.0))
            # tall in most frames, never "mown"; an odd frame may stay unknown
            self.assertNotIn("below_cut", [o["class"] for o in outs])
            edges = [o for o in outs if o.get("lane_edge")]
            self.assertGreaterEqual(len(edges), 6)
            self.assertTrue(all(o["class"] == ">+4cm" and o["min_lane_class"] == "below_cut"
                                for o in edges))

    def test_edge_cells_keep_both_sides(self):
        # drive along a mown | uncut edge: map cells come out per side, not averaged
        def surf(x, y):
            return np.where(y > 0.0, 0.13, 0.03) * (0.6 + 0.4 * texture(x, y))
        _e, cells, _o = Deferred().drive(flat, surf, x_end=3.0)
        left = [c[2] for c in cells if c[1] > 0.15]
        right = [c[2] for c in cells if c[1] < -0.15]
        self.assertTrue(left and right)
        self.assertGreater(float(np.median(left)), 100)
        self.assertLess(float(np.median(right)), 45)

    def test_median_only_over_seen_cells(self):
        import tracemalloc
        g = grass.GrassGrid()
        self.assertTrue(g.ensure(0, 0, 138.0, 138.0))
        g.add(1, 1, 50, 0, 1)
        tracemalloc.start()
        med = g.median_mm()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertEqual(int(med[5, 5]), 55)
        self.assertLess(peak, 2 * med.nbytes + 1e6)      # no full-size hist temporaries

    def test_unmeasured_imu_offset_writes_no_unsupported_class(self):
        # IMU 2 deg off, never calibrated both ways, a path with turns over flat
        # 20 mm lawn, cut 60: no written cell may claim more than below_cut
        for measured, bias in ((False, 2.0), (True, 0.0)):
            cells = []
            e = grass.GrassEngine(calib=calibrated(measured), on_cell=lambda *a: cells.append(a))
            path = []
            for x0, y0, yaw, L in ((0, 0, 0, 1.0), (1, 0, math.pi / 2, 0.5), (1, 0.5, 0, 1.0),
                                   (2, 0.5, -math.pi / 2, 1.5)):
                for i in range(int(L / 0.05) + 1):
                    path.append((x0 + math.cos(yaw) * i * 0.05, y0 + math.sin(yaw) * i * 0.05, yaw))
            t = 0.0
            for x, y, yaw in path:
                P = carpet(flat_ground, 20.0, lo=0.6)
                e.frame(t, P, (x, y, yaw), (0.0, math.radians(bias)), cut_mm=60.0)
                t += 0.15
            self.assertTrue(cells, measured)
            self.assertTrue(all(grass.classify(c[2], 60.0) == "below_cut" for c in cells),
                            (measured, max(c[2] for c in cells)))

    def test_save_merged_keeps_newer_file(self):
        with tempfile.TemporaryDirectory() as d:
            datum = {"utm_e": 493000.0, "utm_n": 5540000.0, "yaw_rad": 0.0}
            a = grass.GrassGrid()
            a.add(0.5, 0.5, 40, 1, grass.SRC_WHEELS)
            a, m_a = grass.save_merged(a, d, "s", datum, [(0.5, 0.5, 40, 1, 1)], 0.0)
            b = grass.GrassGrid.load(d, datum)          # another instance, same file
            b.add(3.1, 3.1, 90, 2, grass.SRC_WHEELS)
            b, _m = grass.save_merged(b, d, "s", datum, [(3.1, 3.1, 90, 2, 1)], m_a)
            a.add(1.1, 1.1, 70, 3, grass.SRC_WHEELS)    # a's view is stale now
            a, _m = grass.save_merged(a, d, "s", datum, [(1.1, 1.1, 70, 3, 1)], m_a)
            rows = grass.query_grass(grass.GrassGrid.load(d, datum),
                                     points=[(0.5, 0.5), (3.1, 3.1), (1.1, 1.1)])
            self.assertEqual([r["count"] for r in rows], [1, 1, 1])
            self.assertTrue(os.path.exists(os.path.join(d, "grass.npz.lock")))

    def test_imu_buffer_at_frame_time(self):
        b = grass.ImuBuffer(max_gap_s=0.2)
        b.add(10.0, 0.0, 0.0)
        b.add(10.1, 0.02, 0.04)
        r, p = b.at(10.05)
        self.assertAlmostEqual(r, 0.01)
        self.assertAlmostEqual(p, 0.02)
        self.assertIsNone(b.at(11.0))                   # IMU gone
        b.add(12.0, 0.0, 0.0)
        self.assertIsNone(b.at(11.0))                   # a hole is not interpolated across

    def test_tf_attitude_keeps_offset(self):
        cal = calibrated()
        cal.imu_pitch_offset_deg = 3.0
        P = carpet(flat_ground, 20.0)
        a = grass.GrassEngine(calib=cal).frame(0.0, P, (0.0, 0.0, 0.0), (0.0, 0.0), rp_imu=True)
        b = grass.GrassEngine(calib=cal).frame(0.0, P, (0.0, 0.0, 0.0), (0.0, 0.0), rp_imu=False)
        self.assertNotEqual(a["slope_deg"], b["slope_deg"])
        self.assertAlmostEqual(b["slope_deg"], 0.0, delta=0.5)

    def test_legend_matches_classes(self):
        lay = grass.grass_layer(None, "s", 0.0, 60)
        self.assertEqual(lay["legend"]["stops"][0]["label"], "up to cut + 2 cm")
        self.assertEqual(grass.classify(79.9, 60), "below_cut")


class Review3(unittest.TestCase):
    """Third independent review (2026-11-08): kerbs and borders beside the track,
    cells with two heights, cells seen only at a corner in a turn."""

    @staticmethod
    def side_frames(ground, grass_mm, n=8, tex=(0.3, 1.0)):
        e = grass.GrassEngine(calib=calibrated())
        outs = []
        for t in range(n):
            x = RNG.uniform(0.6, 1.8, 6000)
            y = RNG.uniform(-0.4, 0.4, 6000)
            z = ground(x, y) + grass_mm(x, y) / 1000.0 * RNG.uniform(tex[0], tex[1], 6000)
            outs.append(e.frame(t * 0.25, np.c_[x, y, z], (t * 0.05, 0.0, 0.0), (0.0, 0.0),
                                cut_mm=60.0))
        return outs

    def test_kerb_beside_the_track_is_never_tall_grass(self):
        for kh in (0.05, 0.07, 0.09, 0.10, 0.15):
            for yk in (0.27, 0.32):
                outs = self.side_frames(lambda x, y, kh=kh, yk=yk: np.where(y > yk, kh, 0.0),
                                        lambda x, y: 30.0 + 0 * x)
                bad = [o["class"] for o in outs if o["class"] in ("+2-4cm", ">+4cm")]
                self.assertFalse(bad, (kh, yk, bad))

    def test_bare_stone_border_is_never_tall_grass(self):
        g = lambda x, y: np.where(y > 0.27, 0.08 + 0.02 * np.sin(37 * x) * np.cos(53 * y), 0.0)  # noqa: E731
        h = lambda x, y: np.where(y > 0.27, 0.0, 30.0)                                          # noqa: E731
        outs = self.side_frames(g, h)
        self.assertFalse([o for o in outs if o["class"] in ("+2-4cm", ">+4cm")])

    def test_real_edge_still_tall(self):
        for ye in (0.0, 0.2, -0.2):
            outs = self.side_frames(flat_ground, lambda x, y, ye=ye: np.where(y > ye, 130.0, 30.0))
            cls = [o["class"] for o in outs]
            self.assertGreaterEqual(cls.count(">+4cm"), 5, (ye, cls))
            self.assertNotIn("below_cut", cls)

    def test_cell_with_two_heights_is_dropped(self):
        # drive along a kerb whose edge cuts through a row of cells (y = 0.3 in a
        # 0.2-0.4 cell): those cells hold lawn and kerb top -> never written
        def surf(x, y):
            return np.where(y > 0.30, 0.09, 0.0) + 0.03 * (0.6 + 0.4 * texture(x, y))
        e, cells, _o = Deferred().drive(flat, surf, x_end=3.0)
        mixed = [c for c in cells if 0.2 < c[1] < 0.4]
        self.assertFalse(mixed, mixed[:3])
        self.assertGreater(e.stats.get("split", 0), 0)

    def test_turn_writes_only_cells_seen_enough(self):
        # arc R = 2 m, mown 30 inside the ring R - 0.15, uncut 130 outside: a
        # cell is written only from looks covering at least half of it
        U = lambda x, y: np.where(np.hypot(x, y - 2.0) < 1.85, 130.0, 30.0)   # noqa: E731
        cells = []
        e = grass.GrassEngine(calib=calibrated(), on_cell=lambda *a: cells.append(a))
        t = 0.0
        for i in range(int(2.0 * 2.0 / 0.05)):
            a = i * 0.05 / 2.0
            x0, y0, yaw = 2.0 * math.sin(a), 2.0 - 2.0 * math.cos(a), a
            x = RNG.uniform(0.6, 1.8, 6000)
            y = RNG.uniform(-0.4, 0.4, 6000)
            wx = x0 + math.cos(yaw) * x - math.sin(yaw) * y
            wy = y0 + math.sin(yaw) * x + math.cos(yaw) * y
            z = U(wx, wy) / 1000.0 * RNG.uniform(0.3, 1.0, 6000)
            e.frame(t, np.c_[x, y, z], (x0, y0, yaw), (0.0, 0.0), cut_mm=60.0)
            t += 0.25
        self.assertGreater(e.stats.get("partial", 0), 0)
        wrong = 0
        for (cx, cy, h, _t, _s) in cells:
            xs, ys = np.meshgrid(np.linspace(cx - 0.09, cx + 0.09, 7), np.linspace(cy - 0.09, cy + 0.09, 7))
            v = U(xs.ravel(), ys.ravel())
            if grass.classify(h, 60) not in (grass.classify(v.max(), 60), grass.classify(np.median(v), 60)):
                wrong += 1
        self.assertLessEqual(wrong, 2)


if __name__ == '__main__':
    unittest.main()
