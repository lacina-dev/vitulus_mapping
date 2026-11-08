#!/usr/bin/env python3
"""geo.ll_to_utm without ROS: python3 test/test_geo.py

The copy of navi_transform's ll_to_utm must give what the site datum was made
with: the origin of 10_2026 (datum.yaml origin_lat/lon) lands on its utm_e/utm_n.
"""
import math
import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import geo                                     # noqa: E402

NAVI_TRANSFORM = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                              'vitulus_navi', 'nodes', 'navi_transform')


class LlToUtm(unittest.TestCase):
    def test_site_origin(self):
        # datum.yaml of 10_2026: origin_lat/lon -> utm_e/utm_n, pyproj agrees to 0.4 mm
        e, n, zone = geo.ll_to_utm(50.01870316, 14.90331712, 33)
        self.assertEqual(zone, 33)
        self.assertAlmostEqual(e, 493073.710, places=2)
        self.assertAlmostEqual(n, 5540714.687, places=2)

    def test_central_meridian_and_equator(self):
        self.assertEqual(geo.ll_to_utm(0.0, 3.0), (500000.0, 0.0, 31))

    def test_zone_auto_and_forced(self):
        self.assertEqual(geo.ll_to_utm(50.0, 14.9)[2], 33)
        e, _n, zone = geo.ll_to_utm(50.0, 15.0, 34)
        self.assertEqual(zone, 34)
        self.assertAlmostEqual(e, 70134.50, places=1)

    def test_south(self):
        _e, n, _z = geo.ll_to_utm(-33.0, 151.0)
        self.assertTrue(6.0e6 < n < 6.5e6)

    def test_into_map(self):
        datum = {'utm_e': 493073.71, 'utm_n': 5540714.687, 'yaw_rad': -0.001292}
        e, n, _z = geo.ll_to_utm(50.01870316, 14.90331712, 33)
        x, y = geo.utm_to_map(e, n, datum)
        self.assertLess(math.hypot(x, y), 0.01)

    @unittest.skipUnless(os.path.exists(NAVI_TRANSFORM), 'vitulus_navi not next to this package')
    def test_same_as_navi_transform(self):
        src = open(NAVI_TRANSFORM).read()
        ns = {}
        exec('import math\n' + src[src.index('_UTM_A, _UTM_E2'):src.index('def utm_to_ll')], ns)
        for lat, lon, zone in ((50.0184923, 14.9033219, 33), (48.2, 16.37, None), (-12.5, -77.0, None)):
            self.assertEqual(geo.ll_to_utm(lat, lon, zone), ns['ll_to_utm'](lat, lon, zone))


if __name__ == '__main__':
    unittest.main()
