#!/usr/bin/env python3
"""Replay of two real D435 captures on the mown lawn (2026-11, site 10_2026):
the robot drives over low grass on gently uneven ground. The output must say
low grass or unknown, never tall grass, and terrain must not read as grass.

The captures (cap1.npz, cap2.npz from cap.py: depth frames, K, base_link->camera,
map->base_link per frame) are not in git. Set GRASS_CAPS to their directory;
without them the test is skipped. The site DEM is used when GRASS_DEM points to
a dem.npz (default: the served site's on this robot).

Run: python3 -m pytest test/test_grass_replay.py   (from the package root)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import grass  # noqa: E402
from vitulus_mapping.demgrid import DemGrid  # noqa: E402

CAPS = os.environ.get('GRASS_CAPS', os.path.expanduser('~/.claude/jobs/04836ce0/tmp'))
DEM = os.environ.get('GRASS_DEM', os.path.expanduser('~/.vitulus/mapping_v3/10_2026/dem.npz'))
CUT_MM = 80.0          # /mower/status current_height on the day


def replay(name, use_dem):
    f = np.load(os.path.join(CAPS, name))
    D, K, T, P, st = f['depth'], f['K'], f['T'], f['pose'], f['stamps']
    t0 = T[~np.isnan(T[:, 0])][0]
    rays = grass.Rays(K, grass.quat_to_rot(*t0[3:]), t0[:3], D.shape[1:], stride=4)
    dem = DemGrid.load_npz(DEM).query_many if use_dem and os.path.exists(DEM) else None
    cells = []
    eng = grass.GrassEngine(on_cell=lambda *a: cells.append(a))
    outs = []
    for i, d in enumerate(D):
        pose, rp = None, (0.0, 0.0)
        if not np.isnan(P[i, 0]):
            r, p, y = grass.quat_rpy(*P[i, 3:])
            pose, rp = (P[i, 0], P[i, 1], y), (r, p)
        outs.append(eng.frame(st[i], rays.points(d), pose, rp, CUT_MM, dem))
    eng.flush_pending(st[-1] + 1000.0, dem)
    return outs, cells


@unittest.skipUnless(os.path.exists(os.path.join(CAPS, 'cap2.npz')), 'captures not available')
class Replay(unittest.TestCase):
    def check(self, outs, cells):
        classes = [o['class'] for o in outs]
        decided = [c for c in classes if c != 'unknown']
        self.assertTrue(decided, 'nothing decided at all')
        self.assertEqual(set(decided), {'below_cut'}, classes)
        # most frames say something (unknown only where terrain changes)
        self.assertGreaterEqual(len(decided), len(outs) // 2)
        for o in outs:
            if o['class'] == 'unknown':
                self.assertIn(o['reason'], ('terrain_change', 'low_confidence', 'no_data'))
        self.assertTrue(cells)
        h = np.array([c[2] for c in cells])
        self.assertLess(float(np.median(h)), CUT_MM + 20.0)
        return classes

    def test_cap1(self):
        outs, cells = replay('cap1.npz', True)
        self.check(outs, cells)

    def test_cap2_extrapolated(self):
        outs, cells = replay('cap2.npz', False)
        classes = self.check(outs, cells)
        # the robot climbs onto a rise here: the start must be held as terrain
        self.assertIn('terrain_change', [o['reason'] for o in outs[:10]], classes)
        # frames 24-44: the body is already on the new slope, the carpet lies with
        # it; the lagging track median must not call that a terrain change
        mid = outs[24:45]
        self.assertGreaterEqual(sum(o['class'] != 'unknown' for o in mid), 18, classes[24:45])

    @unittest.skipUnless(os.path.exists(DEM), 'no site DEM')
    def test_cap2_with_dem(self):
        outs, cells = replay('cap2.npz', True)
        self.check(outs, cells)
        src = [o['terrain_source'] for o in outs]
        self.assertIn('dem', src)
        self.assertTrue(any(c[4] == grass.SRC_DEM for c in cells))
        # where the DEM knows the ground, the rise is not called grass
        for o in outs:
            if o['terrain_source'] == 'dem':
                self.assertEqual(o['class'], 'below_cut')


if __name__ == '__main__':
    unittest.main()
