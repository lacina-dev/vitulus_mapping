#!/usr/bin/env python3
"""nogo stamping and change-only re-serve in mapping_manager, without a master:
the real node methods on an instance with fake publishers.

Run: python3 -m pytest test/test_rules_stamping.py   (from the package root)
"""

import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

import numpy as np  # noqa: E402
import rospy  # noqa: E402
import yaml  # noqa: E402


from vitulus_mapping import bundle, editmask, geo, objects  # noqa: E402
from vitulus_mapping.pgmio import save_occupancy_pgm  # noqa: E402

_loader = importlib.machinery.SourceFileLoader(
    'mapping_manager_node', os.path.join(HERE, '..', 'nodes', 'mapping_manager'))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
mm = importlib.util.module_from_spec(_spec)
_loader.exec_module(mm)

DATUM = {'utm_e': 500000.0, 'utm_n': 5500000.0, 'yaw_rad': 0.0, 'utm_zone': 33}
RES = 0.1
SIZE = 100          # 10 x 10 m, origin 0,0


class Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


def cell(grid_msg, x, y):
    ix, iy = int(x / RES), int(y / RES)
    return grid_msg.data[iy * grid_msg.info.width + ix]


class Stamping(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        rospy.rostime.set_rostime_initialized(True)
        self.site = 's'
        d = bundle.site_dir(self.site)
        os.makedirs(os.path.join(d, 'rasters', 'r'))
        with open(os.path.join(d, 'datum.yaml'), 'w') as f:
            yaml.safe_dump(DATUM, f)
        save_occupancy_pgm(os.path.join(d, 'rasters', 'r', 'r'),
                           np.zeros((SIZE, SIZE), dtype=np.int8), RES, (0.0, 0.0))
        m = mm.MappingManager.__new__(mm.MappingManager)
        m.base_dir = self.tmp.name
        m.map_frame = 'map'
        m.serving = None
        m.serve_error = None
        m.edits_applied = 0
        m.nogo_stamped = 0
        m._nogo_sig = None
        m._nogo_last = (None, [])
        m._serve_lock = threading.RLock()
        m.pub_site_map = Pub()
        m.pub_site_map_loc = Pub()
        m.pub_edit_list = Pub()
        m.pub_track_baseline = Pub()
        m.pub_rename_result = Pub()
        m.save_last_served = lambda *a: None
        m.link_site_datum = lambda *a: None
        m._publish_track_baseline = lambda *a: None
        self.m = m

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        self.tmp.cleanup()

    def utm(self, pts):
        return [list(geo.map_to_utm(x, y, DATUM)) for x, y in pts]

    def rule(self, name, t, pts, **kw):
        o = objects._blank('place', t, name, self.utm(pts))
        o['id'] = 'o_' + name
        o.update(kw)
        return o

    SQ = [[2, 2], [4, 2], [4, 4], [2, 4]]

    def test_nogo_in_site_map_not_in_loc(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
        self.assertEqual(cell(self.m.pub_site_map_loc.msgs[-1], 3, 3), 0)
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 8, 8), 0)
        self.assertEqual(self.m.nogo_stamped, 1)

    def test_edits_stay_in_both_and_cannot_erase_a_nogo(self):
        bundle.save_edits(self.site, [
            {'name': 'W', 'kind': 'obstacle', 'vertices': self.utm([(6, 6), (8, 6), (8, 8), (6, 8)])},
            {'name': 'F', 'kind': 'free', 'vertices': self.utm(self.SQ)}], 33)
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        for pub in (self.m.pub_site_map, self.m.pub_site_map_loc):
            self.assertEqual(cell(pub.msgs[-1], 7, 7), 100)
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)

    def test_other_rules_do_not_stamp(self):
        objects.save_places(self.site, [self.rule('S', 'slow', self.SQ, props={'speed_mps': 0.2}),
                                        self.rule('B', 'noblade', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 0)

    def test_inactive_nogo_does_not_stamp(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ, valid_until=1.0)], 33)
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 0)

    def test_no_rules_file(self):
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)
        self.assertIs(self.m.pub_site_map.msgs[0], self.m.pub_site_map_loc.msgs[0])

    def test_nothing_changed_no_reserve(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        for _ in range(5):
            self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)

    def test_slow_change_no_reserve(self):
        self.m._publish_site_map(self.site, 'r')
        objects.save_places(self.site, [self.rule('S', 'slow', self.SQ, props={'speed_mps': 0.2})], 33)
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)

    def test_rule_change_reserves_map_only_once(self):
        self.m._publish_site_map(self.site, 'r')
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._check_rules()
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 2)
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
        self.assertEqual(len(self.m.pub_site_map_loc.msgs), 1)     # localization untouched

    def test_expiry_clears_the_stamp(self):
        self.m._publish_site_map(self.site, 'r')
        real = objects.load_rules
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ, valid_until=2000.0)], 33)
        try:
            objects.load_rules = lambda site, now=None: real(site, 1000.0)
            self.m._check_rules()
            self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
            objects.load_rules = lambda site, now=None: real(site, 3000.0)
            self.m._check_rules()
            self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 0)
        finally:
            objects.load_rules = real

    def test_rules_unreadable_keep_the_stamp(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        with open(objects.places_path(self.site)) as f:
            good = f.read()
        with open(objects.places_path(self.site), 'w') as f:
            f.write('{')
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)     # no re-serve without it
        self.m._reserve_current()                              # e.g. an edit saved now
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
        with open(objects.places_path(self.site), 'w') as f:
            f.write(good)
        self.m._check_rules()
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)

    def test_datum_gone_keeps_the_stamp(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        os.remove(os.path.join(bundle.site_dir(self.site), 'datum.yaml'))
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)

    def test_bad_nogo_is_not_stamped_and_blocks_no_retry_loop(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ),
                                        self.rule('B', 'nogo', [[6, 6], [8, 6], [8, 8]])], 33)
        with open(objects.places_path(self.site)) as f:
            fc = json.load(f)
        fc['features'][1]['geometry'] = None
        with open(objects.places_path(self.site), 'w') as f:
            json.dump(fc, f)
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)
        self.assertIsNotNone(self.m._nogo_sig)

    def test_last_rules_stay_with_their_site(self):
        self.m._nogo_last = ('other', [{'type': 'nogo'}])
        self.assertEqual(self.m._active_nogo(self.site), [])

    def test_respawn_serves_stamped(self):
        """A fresh instance knows nothing from before, the file is enough."""
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m._publish_site_map(self.site, 'r')
        self.m._nogo_last = (None, [])
        self.m._nogo_sig = None
        self.m._publish_site_map(self.site, 'r')
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)

    def test_status_starts_after_the_boot_serve(self):
        with open(os.path.join(HERE, '..', 'nodes', 'mapping_manager')) as f:
            src = f.read()
        self.assertLess(src.index('self.auto_serve_last()\n'),
                        src.index('rospy.Duration(1.0), self.tick_status'))

    def test_nothing_served(self):
        self.m._check_rules()
        self.assertEqual(self.m.pub_site_map.msgs, [])

    def test_stamping_failure_still_serves(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        real = editmask.rasterize_edits

        def boom(*a):
            raise RuntimeError('x')
        mm.editmask.rasterize_edits = boom
        try:
            self.m._publish_site_map(self.site, 'r')
        finally:
            mm.editmask.rasterize_edits = real
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)
        self.assertEqual(self.m.nogo_stamped, 0)
        self.assertIsNone(self.m._nogo_sig)

    def test_failed_stamp_is_retried_without_publishing_until_it_works(self):
        self.m._publish_site_map(self.site, 'r')
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        real = mm.editmask.rasterize_edits
        calls = []

        def boom(*a):
            calls.append(1)
            raise RuntimeError('x')
        mm.editmask.rasterize_edits = boom
        try:
            for _ in range(3):
                self.m._check_rules()
        finally:
            mm.editmask.rasterize_edits = real
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.m.pub_site_map.msgs), 1)
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 2)
        self.assertEqual(cell(self.m.pub_site_map.msgs[-1], 3, 3), 100)
        self.m._check_rules()
        self.assertEqual(len(self.m.pub_site_map.msgs), 2)

    def test_stamp_of_zero_is_no_success(self):
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        real = mm.editmask.rasterize_edits
        mm.editmask.rasterize_edits = lambda *a: 0
        try:
            self.m._publish_site_map(self.site, 'r')
        finally:
            mm.editmask.rasterize_edits = real
        self.assertIsNone(self.m._nogo_sig)

    def test_exception_cannot_kill_the_timer(self):
        self.m.serving = {'site': self.site, 'raster': 'r', 'mtime': 0}

        def boom(*a):
            raise RuntimeError('x')
        self.m._publish_site_map = boom
        self.m._nogo_sig = 'stale'
        self.m._rules_tick(None)

    def test_reserve_does_not_bring_an_old_site_back(self):
        """_check_rules holds the lock from reading `serving` to publishing."""
        objects.save_places(self.site, [self.rule('N', 'nogo', self.SQ)], 33)
        self.m.serving = {'site': self.site, 'raster': 'r', 'mtime': 0}
        order = []
        real = self.m._active_nogo

        def slow(site):
            order.append('read')
            threading.Thread(target=other).start()
            time.sleep(0.1)
            return real(site)

        def other():
            with self.m._serve_lock:
                order.append('other serve')

        self.m._active_nogo = slow
        self.m._check_rules()
        self.m._active_nogo = real
        time.sleep(0.05)
        self.assertEqual(order[-1], 'other serve')


if __name__ == '__main__':
    unittest.main()
