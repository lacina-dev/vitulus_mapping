#!/usr/bin/env python3
"""DEM ground reference (2026-11-08): base_link is on the wheel contact plane;
DEMs written before (5.5 cm low, no ground_ref key) are lifted on load once.

Run: python3 -m pytest test/test_demgrid_ground.py   (from the package root)
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import demgrid  # noqa: E402
from vitulus_mapping.demgrid import DemGrid  # noqa: E402


class GroundRef(unittest.TestCase):
    def test_new_file_roundtrip_is_not_shifted(self):
        with tempfile.TemporaryDirectory() as d:
            g = DemGrid(0.05)
            g.stamp_disc(1.0, 1.0, 0.30, 1.0, 0.1)
            p = os.path.join(d, 'dem.npz')
            g.save_npz(p)
            b = DemGrid.load_npz(p)
            self.assertFalse(b.migrated)
            self.assertAlmostEqual(float(np.nanmax(b.elev)), 0.30, places=5)

    def test_legacy_file_is_lifted_once(self):
        with tempfile.TemporaryDirectory() as d:
            g = DemGrid(0.05)
            g.stamp_disc(1.0, 1.0, 0.30, 1.0, 0.1)
            p = os.path.join(d, 'dem.npz')
            np.savez_compressed(p, res=g.res, ox=g.ox, oy=g.oy, elev=g.elev, weight=g.weight,
                                last_seen=g.last_seen, source=g.source)       # old format
            b = DemGrid.load_npz(p)
            self.assertTrue(b.migrated)
            self.assertAlmostEqual(float(np.nanmax(b.elev)),
                                   0.30 + demgrid.LEGACY_GROUND_SHIFT_M, places=5)
            b.save_npz(p)
            c = DemGrid.load_npz(p)                    # marker written: no second lift
            self.assertFalse(c.migrated)
            self.assertAlmostEqual(float(np.nanmax(c.elev)), float(np.nanmax(b.elev)), places=5)

    def test_shift_does_not_change_grass_dem_ground(self):
        # grass_height uses the DEM only relative to the DEM under the robot
        sys.path.insert(0, os.path.dirname(__file__))
        from vitulus_mapping import grass
        ramp = lambda x, y: 0.05 * x                                  # noqa: E731
        lifted = lambda x, y: 0.05 * x + 0.055                         # noqa: E731
        outs = []
        for dem in (ramp, lifted):
            e = grass.GrassEngine()
            rng = np.random.default_rng(1)
            x = rng.uniform(0.6, 1.8, 6000)
            y = rng.uniform(-0.4, 0.4, 6000)
            P = np.c_[x, y, 0.05 * x + 0.03 * rng.uniform(0.3, 1.0, 6000)]
            outs.append(e.frame(0.0, P, (0.0, 0.0, 0.0), (0.0, 0.0), 60.0, dem))
        self.assertEqual(outs[0]['terrain_source'], 'dem')
        self.assertEqual(outs[0]['height_mm'], outs[1]['height_mm'])
        self.assertEqual(outs[0]['ground_mm'], outs[1]['ground_mm'])


if __name__ == '__main__':
    unittest.main()
