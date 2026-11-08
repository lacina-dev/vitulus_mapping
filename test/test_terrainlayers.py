#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.terrainlayers (T30 D1: the persistent terrain map).

Run: python3 test/test_terrainlayers.py   (from the package root)

Coordinates are made up (a round UTM point), no real site.
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import terrainlayers as tl                     # noqa: E402

E0, N0 = 500000.0, 5000000.0          # made-up UTM point
DATUM = {"utm_e": E0 + 3.3, "utm_n": N0 - 7.1, "yaw_rad": 0.01}
T0 = 1790000000.0
RES = 0.05
NOW = {"accept_delay": 0.0}           # most tests fuse at once; the delay has its own tests
P0 = (0.0, 0.0, 0.0)                  # map <- odom (x, y, yaw)


def flat_window(x0, y0, z=0.0, n=120, var=1e-4, points=5, reason=1, cost=10, t=None, **kw):
    shp = (n, n)
    return tl.Window(x0, y0, RES, np.full(shp, z, np.float32), np.full(shp, var, np.float32),
                     np.full(shp, points, np.float32), np.full(shp, cost, np.float32),
                     np.full(shp, reason, np.float32), slope=np.full(shp, 0.05, np.float32),
                     step=np.full(shp, 0.01, np.float32), roughness=np.full(shp, 0.004, np.float32),
                     t_fused=None if t is None else np.full(shp, t, np.float64), **kw)


def map_to_utm(x, y, d=DATUM):
    c, s = math.cos(d["yaw_rad"]), math.sin(d["yaw_rad"])
    return d["utm_e"] + c * x - s * y, d["utm_n"] + s * x + c * y


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="t30_")
        self.root = os.path.join(self.dir, "terrain")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def store(self, root=None, **params):
        q = dict(NOW)
        q.update(params)
        return tl.TerrainStore(root or self.root, DATUM, params=q)

    def one_visit(self, st, x0, y0, z, t, pose=P0, patch=None, robot=None, **kw):
        """A window fused once, then its visits closed. patch: other height of the
        1 x 1 m square in the middle (a uniform change of the whole window cannot be
        told from a z offset of the odometry: the alignment takes it out)."""
        w = flat_window(x0, y0, z, t=t, **kw)
        if patch is not None:
            w.elevation[50:70, 50:70] = patch
        st.fuse(w, pose, t, robot=robot)
        st.drain()
        st.close_all()

    def cell(self, st, x=0.0, y=0.0):
        e, n = map_to_utm(x, y)
        return st.sample_utm(e, n)


class Keys(unittest.TestCase):
    def test_pack_unpack_and_tile_groups(self):
        gx = np.array([255, 256, 10000000, 3], np.int64)
        gy = np.array([100000000, 100000000, 5, 0], np.int64)
        k = tl.pack_keys(gx, gy)
        bx, by = tl.unpack_keys(k)
        np.testing.assert_array_equal(bx, gx)
        np.testing.assert_array_equal(by, gy)
        groups = list(tl.TerrainStore._groups(k))
        self.assertEqual(len(groups), 4)            # 255 and 256 lie in neighbouring tiles
        for (tx, ty), sel, ly, lx in groups:
            self.assertTrue(np.all((lx >= 0) & (lx < tl.TILE) & (ly >= 0) & (ly < tl.TILE)))
            np.testing.assert_array_equal((tx << 8) + lx, gx[sel])
            np.testing.assert_array_equal((ty << 8) + ly, gy[sel])

    def test_huber_and_histogram_ignore_outliers(self):
        rng = np.random.RandomState(1)
        r = np.concatenate([rng.normal(0.10, 0.005, 500), np.full(100, -0.5)])
        self.assertAlmostEqual(tl.huber_mean(r, 0.02), 0.10, delta=0.005)
        # 40 % of the overlap wrong by 1.2 m (another level): the mode still finds the right one
        r2 = np.concatenate([rng.normal(-3.0, 0.005, 600), rng.normal(-1.8, 0.02, 400)])
        self.assertAlmostEqual(tl.hist_mode(r2, 0.02), -3.0, delta=0.01)

    def test_direction_spread(self):
        self.assertEqual(int(tl.dir_spread(0b00000001)), 1)
        self.assertEqual(int(tl.dir_spread(0b00000011)), 1)     # neighbouring sectors
        self.assertEqual(int(tl.dir_spread(0b00000101)), 2)     # 90 deg apart
        self.assertEqual(int(tl.dir_spread(0b10000001)), 1)     # 0 and 7 are neighbours
        self.assertEqual(int(tl.dir_spread(0)), 0)

    def test_unknown_parameter_refused(self):
        with self.assertRaises(ValueError):
            tl.TerrainStore("/nonexistent", DATUM, params={"no_such": 1})


class Fusion(Base):
    def test_rotated_window_no_holes_and_heights(self):
        st = self.store()
        pose = (1.2, -0.4, 0.7)                    # odom rotated 0.7 rad in map
        st.fuse(flat_window(-3.0, -3.0, z=0.2, t=T0), pose, T0)
        st.close_all()
        # every global cell whose centre lies inside the window minus its edge margin is known
        c, s, te, tn = st.utm_from_odom(pose)
        xs, ys = np.meshgrid(np.arange(-2.7, 2.7, 0.02), np.arange(-2.7, 2.7, 0.02))
        e = te + c * xs.ravel() - s * ys.ravel()
        n = tn + s * xs.ravel() + c * ys.ravel()
        out = st.sample_utm(e, n)
        self.assertEqual(int(np.count_nonzero(~np.isfinite(out["h"]))), 0)
        np.testing.assert_allclose(out["h"], 0.2, atol=1e-5)  # first batch, nothing known: offset 0
        self.assertEqual(st.last_batch["offset_src"], "none")

    def test_edge_margin_left_out(self):
        st = self.store()
        st.fuse(flat_window(-3.0, -3.0, z=0.0, t=T0), P0, T0)
        st.close_all()
        self.assertTrue(np.isfinite(self.cell(st, -2.7, 0.0)["h"][0]))
        self.assertFalse(np.isfinite(self.cell(st, -2.85, 0.0)["h"][0]))   # within 0.25 m of the edge
        self.assertFalse(np.isfinite(self.cell(st, 0.0, 2.85)["h"][0]))

    def test_many_batches_of_one_visit_are_one_measurement(self):
        st = self.store()
        for i in range(50):
            st.fuse(flat_window(-3, -3, 0.0, t=T0 + i, var=1e-4), P0, T0 + i)
        st.close_all()
        e, n = map_to_utm(0.0, 0.0)
        tile = st.tile(int(e / RES) >> 8, int(n / RES) >> 8)
        gx, gy = int(e / RES), int(n / RES)
        ly, lx = gy - (tile.ty << 8), gx - (tile.tx << 8)
        self.assertEqual(int(tile.visits[ly, lx]), 1)
        self.assertAlmostEqual(float(tile.var[ly, lx]), 1e-4 + 0.01 ** 2, places=7)

    def test_visit_closes_when_the_cell_leaves_the_window(self):
        st = self.store()
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        self.assertGreater(st.ov_keys.size, 12000)
        st.fuse(flat_window(-3 + 3.0, -3, 0.0, t=T0 + 1), P0, T0 + 1)
        info = st.last_batch
        self.assertGreater(info["closed"], 5500)    # the left half left the window
        self.assertLess(info["closed"], 7500)
        self.assertEqual(int(self.cell(st, -2.5, 0.0)["visits"][0]), 1)

    def test_visit_timeout(self):
        st = self.store(visit_timeout=30.0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0 + 31)   # no fresh point for 31 s
        self.assertEqual(st.last_batch["fresh"], 0)
        self.assertEqual(st.ov_keys.size, 0)

    def test_only_certain_cells(self):
        st = self.store(edge_margin=0.0)
        w = flat_window(-3, -3, 0.0, t=T0)
        w.variance[:60, :] = 5e-4                    # above max_variance
        w.points[60:, :60] = 1                       # too few points
        st.fuse(w, P0, T0)
        self.assertAlmostEqual(st.last_batch["cells"] / (120 * 60 * 1.0), 0.5, delta=0.05)


class Localisation(Base):
    def test_accept_delay(self):
        st = self.store(accept_delay=2.0)
        r = st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        self.assertEqual((r["queued"], len(r["processed"])), (True, 0))
        r = st.fuse(flat_window(-3, -3, 0.0, t=T0 + 1), P0, T0 + 1)
        self.assertEqual(len(r["processed"]), 0)
        r = st.fuse(flat_window(-3, -3, 0.0, t=T0 + 2), P0, T0 + 2)
        self.assertEqual([i["stamp"] for i in r["processed"]], [T0])

    def test_settling_drops_the_queue(self):
        st = self.store(accept_delay=2.0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0 + 1), P0, T0 + 1)
        r = st.fuse(flat_window(-3, -3, 0.0, t=T0 + 2, settling=True), P0, T0 + 2)
        self.assertEqual((r["queued"], r["dropped"]), (False, 3))
        self.assertEqual(st.pending, [])
        self.assertEqual(st.ov_keys.size, 0)
        self.assertEqual(st.tiles, {})               # nothing reached the map

    def test_new_epoch_drops_queue_and_closes_visits(self):
        st = self.store(accept_delay=1.0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0, epoch=4), P0, T0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0 + 1, epoch=4), P0, T0 + 1)   # T0 fused, open visits
        st.fuse(flat_window(-3, -3, 0.0, t=T0 + 1.5, epoch=4), P0, T0 + 1.5)
        self.assertGreater(st.ov_keys.size, 0)
        r = st.fuse(flat_window(-3, -3, 0.7, t=T0 + 2, epoch=5), P0, T0 + 2)
        self.assertEqual(r["dropped"], 2)            # T0+1 and T0+1.5 of the old epoch
        self.assertEqual(st.ov_keys.size, 0)         # the visits of epoch 4 are in the map
        self.assertEqual(int(self.cell(st)["visits"][0]), 1)
        self.assertIsNone(st.z_offset)               # the offset is found again (wide)
        st.drain()
        self.assertEqual(st.last_batch["offset_src"], "overlap")
        st.close_all()
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.0, delta=0.002)

    def test_pose_jump_refused(self):
        st = self.store()
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        st.fuse(flat_window(-3, -3, 0.0, t=T0 + 1), (0.3, 0.0, 0.0), T0 + 1)
        self.assertEqual(st.last_batch["refused"], "pose_jump")
        self.assertEqual(st.ov_keys.size, 0)

    def test_odometry_z_jump_closes_visits_and_is_aligned(self):
        st = self.store()
        st.fuse(flat_window(-3, -3, 0.0, t=T0), P0, T0)
        st.fuse(flat_window(-3, -3, -0.316, t=T0 + 1), P0, T0 + 1)    # E5-sized jump, any size passes
        self.assertTrue(st.last_batch["offset_jump"])
        self.assertAlmostEqual(st.z_offset, 0.316, delta=0.002)
        st.close_all()
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.0, delta=0.002)

    def test_start_offset_from_the_ground_under_the_robot(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.10, T0)                 # the dock area is known
        st.save_dirty()
        st2 = self.store()                                   # restart: new odom, z 1 m off
        self.assertEqual(st2.session, st.session + 1)
        # first window: no overlap (other place), robot standing at (0, 0) on known cells
        st2.fuse(flat_window(10.0, -3, -1.0, t=T0 + 600), P0, T0 + 600, robot=(0.0, 0.0, -1.0, 0.0))
        self.assertEqual(st2.last_batch["offset_src"], "robot_ground")
        st2.close_all()
        self.assertAlmostEqual(float(self.cell(st2, 13.0, 0.0)["h"][0]), 0.10, delta=0.002)


class ChangeOverTime(Base):
    def test_change_needs_two_visits(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.10, T0)
        self.one_visit(st, -3, -3, 0.10, T0 + 3600, patch=0.0)    # one disagreeing visit
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.10, delta=0.002)
        self.assertAlmostEqual(float(self.cell(st, 2.0, 2.0)["h"][0]), 0.10, delta=0.002)
        self.one_visit(st, -3, -3, 0.10, T0 + 7200, patch=0.0)    # confirmed
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.0, delta=0.002)

    def test_cut_grass_at_once_after_mowing(self):
        st = tl.TerrainStore(self.root, DATUM, params=NOW,
                             mowed=lambda e, n: np.full(np.size(e), T0 + 1800))
        self.one_visit(st, -3, -3, 0.10, T0)
        self.one_visit(st, -3, -3, 0.10, T0 + 3600, patch=0.05)   # inside the gate, still replaced
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.05, delta=0.002)

    def test_single_outlier_visit_leaves_the_height(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.10, T0)
        self.one_visit(st, -3, -3, 0.10, T0 + 3600, patch=0.30)   # a box once
        self.one_visit(st, -3, -3, 0.10, T0 + 7200)               # gone again
        self.one_visit(st, -3, -3, 0.10, T0 + 9000, patch=0.30)   # once more: a new candidate only
        self.assertAlmostEqual(float(self.cell(st)["h"][0]), 0.10, delta=0.002)

    def test_agreeing_visits_are_fused_with_a_sigma_floor(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.100, T0)
        self.one_visit(st, -3, -3, 0.100, T0 + 60, patch=0.110)   # within the gate
        h = float(self.cell(st)["h"][0])
        self.assertTrue(0.101 < h < 0.109, h)
        for i in range(40):
            self.one_visit(st, -3, -3, 0.100, T0 + 120 + 60 * i, var=1e-6)
        e, n = map_to_utm(0.0, 0.0)
        tile = st.tile(int(e / RES) >> 8, int(n / RES) >> 8)
        self.assertGreaterEqual(float(np.nanmin(tile.var)), 0.005 ** 2 * 0.999)

    def test_obstacle_only_from_the_last_visit(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.0, T0, reason=5, cost=100)
        c = self.cell(st)
        self.assertEqual((int(c["reason"][0]), int(c["cost"][0])), (5, 100))
        self.one_visit(st, -3, -3, 0.0, T0 + 60, reason=1, cost=5, points=4)   # too thin to clear
        self.assertEqual(int(self.cell(st)["reason"][0]), 5)
        self.one_visit(st, -3, -3, 0.0, T0 + 120, reason=1, cost=5, points=8)  # moved away
        c = self.cell(st)
        self.assertEqual((int(c["reason"][0]), int(c["cost"][0])), (1, 5))

    def lethal(self, st, t, yaw, reason=3):
        self.one_visit(st, -3, -3, 0.0, t, reason=reason, cost=100, robot=(0.0, 0.0, 0.0, yaw))

    def state(self, st):
        c = self.cell(st)
        return int(c["reason"][0]), int(c["cost"][0]), bool(c["tentative"][0])

    def test_terrain_ban_needs_visits_sessions_and_directions(self):
        st = self.store()
        for i in range(4):                                  # 4 lethal visits, one session, one heading
            self.lethal(st, T0 + 60 * i, 0.0)
        self.assertEqual(self.state(st), (3, 99, True))
        st.new_session()
        self.lethal(st, T0 + 3600, 0.1)                     # 2 sessions, still one heading
        self.assertEqual(self.state(st), (3, 99, True))
        self.lethal(st, T0 + 3700, math.pi / 2)             # from 90 deg: confirmed
        self.assertEqual(self.state(st), (3, 100, False))
        # one free visit: still confirmed; three free in a row clear it
        self.one_visit(st, -3, -3, 0.0, T0 + 3800, reason=1, cost=5)
        self.assertEqual(self.state(st)[1], 100)
        for i in range(2):
            self.one_visit(st, -3, -3, 0.0, T0 + 3900 + 60 * i, reason=1, cost=5)
        self.assertEqual(self.state(st)[:2], (1, 5))

    def test_lethal_share(self):
        st = self.store()
        for i in range(6):                                  # 6 free visits
            self.one_visit(st, -3, -3, 0.0, T0 + 60 * i, reason=1, cost=5)
        yaws = [0.0, math.pi / 2, math.pi]
        for i in range(3):                                  # 3 lethal of 9 < 50 %
            st.new_session()
            self.lethal(st, T0 + 600 + 60 * i, yaws[i], reason=4)
        self.assertEqual(self.state(st)[:2], (4, 99))
        for i in range(2):
            self.lethal(st, T0 + 900 + 60 * i, 0.0, reason=4)   # 5 of 11 < 50 %
        self.assertEqual(self.state(st)[1], 99)
        self.lethal(st, T0 + 1020, 0.0, reason=4)            # 6 of 12
        self.assertEqual(self.state(st)[1], 100)


# --------------------------------------------------------------------------
# drift simulation
# --------------------------------------------------------------------------
def true_z(e, n):
    x = e - E0
    y = n - N0
    return 0.15 * np.sin(x / 3.0) + 0.08 * np.cos(y / 2.0) + 0.02 * x + 0.03 * np.sin(x * 1.7 + y)


class Sim(object):
    """A robot mowing; the local map keeps what it measured (with the odometry z error
    of that moment) in its odom frame; windows of 6 x 6 m around the robot.
    odom: map <- odom (x, y, z, yaw) - z only inside the simulation (odom z vs the
    terrain), the store gets (x, y, yaw). tilt_deg: a pitch error of the camera /
    odometry (heights wrong by tan(tilt) x distance ahead of the robot)."""

    def __init__(self, seed, odom, drift_step=0.003, noise=0.003, tilt_deg=0.0):
        self.rng = np.random.RandomState(seed)
        self.odom = odom
        self.drift_step = drift_step
        self.noise = noise
        self.tilt = math.tan(math.radians(tilt_deg))
        self.L = 1000                         # local map 50 x 50 m in odom
        self.ox = -25.0
        self.h = np.full((self.L, self.L), np.nan)
        self.t = np.zeros((self.L, self.L))
        self.drift = 0.0

    @property
    def pose(self):
        return (self.odom[0], self.odom[1], self.odom[3])

    def map_to_odom(self, xm, ym):
        x, y, _, yaw = self.odom
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xm - x, ym - y
        return c * dx + s * dy, -s * dx + c * dy

    def step(self, xm, ym, head_m, t):
        """Robot at map (xm, ym) heading head_m; measures a cone, returns (window, robot)."""
        self.drift += self.rng.normal(0, self.drift_step)
        xo, yo = self.map_to_odom(xm, ym)
        ho = head_m - self.odom[3]
        ci = int(round((xo - self.ox) / RES))
        cj = int(round((yo - self.ox) / RES))
        i0, j0 = ci - 60, cj - 60
        jj, ii = np.mgrid[j0:j0 + 120, i0:i0 + 120]
        cx = self.ox + (ii + 0.5) * RES
        cy = self.ox + (jj + 0.5) * RES
        dx, dy = cx - xo, cy - yo
        r = np.hypot(dx, dy)
        ang = np.abs(np.arctan2(np.sin(np.arctan2(dy, dx) - ho), np.cos(np.arctan2(dy, dx) - ho)))
        cone = (r > 0.5) & (r < 2.8) & (ang < math.radians(40))
        ahead = dx * math.cos(ho) + dy * math.sin(ho)
        x, y, z, yaw = self.odom
        c, s = math.cos(yaw), math.sin(yaw)
        e, n = map_to_utm(x + c * cx - s * cy, y + s * cx + c * cy)
        meas = true_z(e, n) - z + self.drift + self.tilt * ahead + \
            self.rng.normal(0, self.noise, e.shape)
        sl = (slice(j0, j0 + 120), slice(i0, i0 + 120))
        hw, tw = self.h[sl], self.t[sl]
        hw[cone] = meas[cone]
        tw[cone] = t
        win = tl.Window(self.ox + i0 * RES, self.ox + j0 * RES, RES, hw.astype(np.float32),
                        np.where(np.isfinite(hw), 1e-4, np.nan).astype(np.float32),
                        np.where(np.isfinite(hw), 5, 0).astype(np.float32),
                        np.full(hw.shape, 5, np.float32), np.full(hw.shape, 1, np.float32),
                        t_fused=tw.copy())
        re_, rn_ = map_to_utm(xm, ym)
        zo = float(true_z(np.array([re_]), np.array([rn_]))[0]) - z + self.drift   # base_link z in odom
        return win, (xo, yo, zo, ho)


def mow_rows(x0, x1, y0, y1, row, step):
    pts = []
    y = y0
    d = 1
    while y <= y1 + 1e-9:
        xs = np.arange(x0, x1 + 1e-9, step) if d > 0 else np.arange(x1, x0 - 1e-9, -step)
        for x in xs:
            pts.append((x, y, 0.0 if d > 0 else math.pi))
        y += row
        d = -d
    return pts


def loop_path(x0, x1, y0, y1, step):
    pts = []
    for (ax, ay, bx, by, hd) in ((x0, y0, x1, y0, 0.0), (x1, y0, x1, y1, math.pi / 2),
                                 (x1, y1, x0, y1, math.pi), (x0, y1, x0, y0, -math.pi / 2)):
        nseg = int(round(math.hypot(bx - ax, by - ay) / step))
        for k in range(nseg):
            f = k / float(nseg)
            pts.append((ax + f * (bx - ax), ay + f * (by - ay), hd))
    return pts


class Drift(Base):
    def drive(self, st, sim, path, t0):
        t = t0
        for (x, y, hd) in path:
            win, rob = sim.step(x, y, hd, t)
            st.fuse(win, sim.pose, t, robot=rob)
            t += 1.0
        st.drain()
        st.close_all()
        return t

    def errors(self, st, x0, x1, y0, y1):
        xs, ys = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
        e, n = map_to_utm(xs.ravel(), ys.ravel())
        h = st.sample_utm(e, n)["h"]
        f = np.isfinite(h)
        return h[f] - true_z(e[f], n[f]), f.mean()

    @staticmethod
    def rms0(err):
        return float(np.sqrt(np.mean((err - np.median(err)) ** 2)))

    def test_drift_keeps_the_map_consistent(self):
        st = self.store(accept_delay=2.0)
        sim = Sim(3, (1.2, -0.4, 0.0, 0.6), drift_step=0.003)
        path = mow_rows(0.0, 10.0, 0.0, 6.0, 0.6, 0.25)
        t = self.drive(st, sim, path, T0)
        err, cov = self.errors(st, 0.5, 10.0, 0.5, 6.0)
        rms = self.rms0(err)                               # no anchor: the z reference is free
        self.assertGreater(cov, 0.9)
        self.assertLess(rms, 0.02, "rms %.4f" % rms)
        # the same run fused without alignment is clearly worse
        st2 = self.store(os.path.join(self.dir, "noalign"), align_min_cells=10 ** 9)
        self.drive(st2, Sim(3, (1.2, -0.4, 0.0, 0.6), drift_step=0.003), path, T0)
        err2, _ = self.errors(st2, 0.5, 10.0, 0.5, 6.0)
        rms2 = self.rms0(err2)
        self.assertGreater(rms2, 2 * rms, "aligned %.4f, without %.4f" % (rms, rms2))

        # restart of the stack: new odom (other yaw, z 1.5 m off), driving over the mapped part
        ref = float(np.median(err))
        before = st.sample_utm(*map_to_utm(5.0, 3.0))["h"][0]
        sim3 = Sim(4, (-2.0, 1.0, -1.5, 1.9), drift_step=0.003)
        win, rob = sim3.step(0.0, 0.0, 0.0, t)
        st.fuse(win, sim3.pose, t, robot=rob)
        st.drain()
        self.assertEqual(st.last_batch["refused"], "pose_jump")
        self.drive(st, sim3, mow_rows(1.0, 9.0, 0.3, 5.7, 0.9, 0.25), t + 1)
        err3, _ = self.errors(st, 0.5, 10.0, 0.5, 6.0)
        rms3 = float(np.sqrt(np.mean((err3 - ref) ** 2)))
        self.assertLess(rms3, 0.02, "after restart rms %.4f" % rms3)
        after = st.sample_utm(*map_to_utm(5.0, 3.0))["h"][0]
        self.assertAlmostEqual(after, before, delta=0.03)

    def test_systematic_tilt(self):
        """0.5 deg pitch error: heights wrong by up to 2.4 cm at 2.8 m ahead, the rows
        alternate the direction. Bound: the map stays within 2.5 cm RMS."""
        st = self.store()
        sim = Sim(7, (0.3, 0.2, 0.0, -0.4), drift_step=0.002, tilt_deg=0.5)
        self.drive(st, sim, mow_rows(0.0, 10.0, 0.0, 6.0, 0.6, 0.25), T0)
        err, cov = self.errors(st, 0.5, 10.0, 0.5, 6.0)
        rms = self.rms0(err)
        print("\n  tilt 0.5 deg: rms %.4f m, coverage %.2f" % (rms, cov))
        self.assertGreater(cov, 0.9)
        self.assertLess(rms, 0.025, "rms %.4f" % rms)

    def test_loop_closure(self):
        """A 12 x 8 m loop with 3x the drift of the other tests, back to the start: the
        return finds the start in the map (overlap), no seam grows over 3 cm RMS."""
        st = self.store()
        sim = Sim(8, (0.0, 0.0, 0.0, 0.3), drift_step=0.009)
        path = loop_path(0.0, 12.0, 0.0, 8.0, 0.25)
        path = path + path[:40]                            # past the start again
        self.drive(st, sim, path, T0)
        err, cov = self.errors(st, -1.0, 13.0, -1.0, 9.0)
        rms = self.rms0(err)
        print("\n  loop: rms %.4f m, drift at the end %.3f m, offset jumps %d"
              % (rms, sim.drift, st.stats["offset_jumps"]))
        self.assertLess(rms, 0.03, "rms %.4f" % rms)

    def test_anchor_gives_absolute_heights(self):
        rng = np.random.RandomState(9)
        dem = lambda e, n: true_z(np.asarray(e), np.asarray(n)) + rng.normal(0, 0.01, np.size(e))  # noqa: E731
        st = tl.TerrainStore(self.root, DATUM, params=NOW, anchor=dem)
        sim = Sim(5, (0.5, 0.5, 0.0, -0.3), drift_step=0.003)
        sim.drift = 0.4                                   # odometry z 40 cm off from the start
        self.drive(st, sim, mow_rows(0.0, 8.0, 0.0, 5.0, 0.6, 0.25), T0)
        err, cov = self.errors(st, 0.5, 8.0, 0.5, 5.0)
        self.assertGreater(cov, 0.9)
        self.assertLess(abs(float(np.median(err))), 0.02)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.025)
        self.assertLess(abs(st.anchor_residual), 0.03)     # with overlap: bias of the map only

    def test_anchor_residual_reports_a_biased_map(self):
        st = self.store()
        sim = Sim(6, (0.5, 0.5, 0.0, -0.3), drift_step=0.0)
        sim.drift = 0.25                                  # no anchor at the start: the map is 25 cm high
        self.drive(st, sim, mow_rows(0.0, 4.0, 0.0, 2.0, 0.6, 0.25), T0)
        st.anchor = lambda e, n: true_z(np.asarray(e), np.asarray(n))
        self.drive(st, sim, mow_rows(0.0, 4.0, 0.0, 2.0, 0.6, 0.25), T0 + 1000)
        self.assertAlmostEqual(st.anchor_residual, -0.25, delta=0.02)


class Import(Base):
    def grid(self, z, h=80, w=100, reason=1):
        shp = (h, w)
        lay = {"elevation": np.full(shp, z), "sigma": np.full(shp, 0.01), "cost": np.full(shp, 10.0),
               "reason": np.full(shp, reason, np.uint8), "slope": np.full(shp, 0.1),
               "step": np.full(shp, 0.02), "roughness": np.full(shp, 0.005)}
        lay["elevation"][0, :] = np.nan                     # unknown row (south edge)
        return lay

    def test_import_is_one_tentative_visit(self):
        st = self.store()
        lay = self.grid(0.3, reason=3)
        lay["elevation"][10, 20] = 0.5                      # row 10 = 10th row from the south
        n, off = st.import_map_grid((-2.0, -1.0), 0.05, lay, T0)
        self.assertEqual(off, 0.0)                          # nothing to align to
        self.assertGreater(n, 7500)
        c = self.cell(st, -2.0 + 20.5 * 0.05, -1.0 + 10.5 * 0.05)
        self.assertAlmostEqual(float(c["h"][0]), 0.5, delta=1e-6)
        c = self.cell(st, 0.0, 1.0)
        self.assertEqual((int(c["reason"][0]), int(c["cost"][0]), bool(c["tentative"][0]),
                          int(c["visits"][0])), (3, 99, True, 1))
        self.assertFalse(np.isfinite(self.cell(st, 0.0, -1.0 + 0.5 * 0.05)["h"][0]))
        # live visits confirm it later like any other visit: 2 more lethal visits in another
        # session from another heading
        st.new_session()
        for i in range(2):
            self.one_visit(st, -3, -3, 0.3, T0 + 100 + 60 * i, reason=3, cost=100,
                           robot=(0, 0, 0.3, math.pi / 2 * i))
        c = self.cell(st, 0.0, 1.0)
        self.assertEqual((int(c["cost"][0]), int(c["visits"][0])), (100, 3))

    def test_second_mosaic_aligned_and_known_cells_kept(self):
        st = self.store()
        st.import_map_grid((-2.0, -1.0), 0.05, self.grid(0.3), T0)
        n, off = st.import_map_grid((0.0, -1.0), 0.05, self.grid(1.1), T0 + 3600)  # own reference
        self.assertAlmostEqual(off, -0.8, delta=1e-3)
        self.assertAlmostEqual(float(self.cell(st, 2.8, 1.0)["h"][0]), 0.3, delta=1e-3)  # new cells aligned
        self.assertAlmostEqual(float(self.cell(st, 1.0, 1.0)["h"][0]), 0.3, delta=1e-3)  # kept
        self.assertLess(n, 80 * 100)


class Disk(Base):
    def filled(self):
        st = self.store()
        self.one_visit(st, -3, -3, 0.123, T0, reason=3, cost=100, robot=(0, 0, 0, 0.0))
        self.one_visit(st, -3, -3, 0.123, T0 + 60, reason=3, cost=100, robot=(0, 0, 0, 2.0))
        self.one_visit(st, 9.0, -3, 0.5, T0 + 120)            # another tile
        self.one_visit(st, -3, -3, 0.123, T0 + 180, patch=0.30)   # candidate pending
        return st

    def test_roundtrip(self):
        st = self.filled()
        self.assertGreaterEqual(len(st.tiles), 2)
        st.save_dirty()
        st2 = self.store()
        self.assertEqual(st2.index["tiles"].keys(), st.index["tiles"].keys())
        self.assertEqual(st2.index["z_ref"], st.index["z_ref"])
        for key, t in st.tiles.items():
            t2 = st2.tile(*key)
            self.assertEqual(t2.version, t.version)
            fin = np.isfinite(t.h)
            np.testing.assert_array_equal(fin, np.isfinite(t2.h))
            self.assertLessEqual(float(np.max(np.abs(t.h[fin] - t2.h[fin]))), 0.0005 + 1e-6)
            self.assertLessEqual(float(np.max(np.abs(np.sqrt(t.var[fin]) - np.sqrt(t2.var[fin])))),
                                 0.0005 + 1e-6)
            for k in tl.TILE_DTYPES:
                if k in ("h_mm", "sig_mm", "cand_mm"):
                    continue
                np.testing.assert_array_equal(getattr(t, k), getattr(t2, k), k)
            # loaded -> saved again: identical arrays
            a1 = t2.to_arrays(st2.index["z_ref"])
            a2 = tl.Tile.from_arrays(t2.tx, t2.ty, a1, st2.index["z_ref"]).to_arrays(st2.index["z_ref"])
            for k in a1:
                np.testing.assert_array_equal(a1[k], a2[k], k)
        e, n = map_to_utm(0.0, 0.0)
        t0 = st.tile(int(e / RES) >> 8, int(n / RES) >> 8)
        self.assertGreater(int(np.count_nonzero(t0.cand_n)), 0)      # the pending candidate survives
        c1, c2 = self.cell(st), self.cell(st2)
        self.assertEqual((int(c1["reason"][0]), int(c1["cost"][0])), (int(c2["reason"][0]), int(c2["cost"][0])))

    def test_broken_tile_moved_aside(self):
        st = self.filled()
        st.save_dirty()
        key = next(iter(st.tiles))
        path = st._tile_path(*key)
        with open(path, "wb") as f:
            f.write(b"not a zip")
        st2 = self.store()
        self.assertIsNone(st2.tile(*key))
        self.assertEqual(len(st2.broken), 1)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(len(os.listdir(os.path.join(self.root, "broken"))), 1)
        self.assertIsNotNone(st2.tile(*key, create=True))     # starts empty, the node goes on

    def test_broken_index_moved_aside(self):
        st = self.filled()
        st.save_dirty()
        with open(os.path.join(self.root, "index.json"), "w") as f:
            f.write("{broken")
        st2 = self.store()
        self.assertEqual(st2.index["tiles"], {})
        self.assertEqual(len(st2.broken), 1)

    def test_eviction_saves_dirty_tiles(self):
        st = self.store(max_loaded_tiles=1)
        self.one_visit(st, -3, -3, 0.1, T0)
        self.one_visit(st, 30.0, -3, 0.2, T0 + 60)
        self.assertLessEqual(len(st.tiles), 2)
        st.save_dirty()
        st2 = self.store()
        self.assertAlmostEqual(float(self.cell(st2)["h"][0]), 0.1, delta=0.001)

    def test_new_datum_keeps_the_data(self):
        st = self.filled()
        st.save_dirty()
        e, n = map_to_utm(0.0, 0.0)
        before = st.sample_utm(e, n)
        # the site datum is captured again: map frame moved by (0.4, -0.2) m and rotated 0.02 rad
        d2 = {"utm_e": DATUM["utm_e"] + 0.4, "utm_n": DATUM["utm_n"] - 0.2, "yaw_rad": 0.03}
        st2 = tl.TerrainStore(self.root, d2, params=NOW)
        self.assertAlmostEqual(float(st2.sample_utm(e, n)["h"][0]), float(before["h"][0]), delta=0.0006)
        # the same physical place seen through the new datum maps to the same UTM point
        x_new = math.cos(0.03) * (e - d2["utm_e"]) + math.sin(0.03) * (n - d2["utm_n"])
        y_new = -math.sin(0.03) * (e - d2["utm_e"]) + math.cos(0.03) * (n - d2["utm_n"])
        _, _, te, tn = st2.utm_from_odom((x_new, y_new, 0.0))
        self.assertAlmostEqual(te, e, places=6)
        self.assertAlmostEqual(tn, n, places=6)


if __name__ == "__main__":
    unittest.main()
