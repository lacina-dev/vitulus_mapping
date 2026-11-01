#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.suggestions (contract section 7).

Run: python3 -m pytest test/test_suggestions.py   (from the package root)
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from vitulus_mapping import bundle, suggestions as sg  # noqa: E402

DATUM = {'utm_e': 493073.71, 'utm_n': 5540714.687, 'yaw_rad': 0.3}
SITE = 's1'


def row(x, y, kind='static_new', extent=0.8):
    return {'kind': kind, 'centroid': [x, y], 'extent_m': extent}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        os.makedirs(bundle.site_dir(SITE))

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        self.tmp.cleanup()

    def look(self, t, rx, x=5.0, y=5.0, objects=None, **kw):
        return sg.observe(SITE, [row(x, y, **kw)], {'x': rx, 'y': 0.0}, DATUM,
                          objects, now=t)

    def confirm(self, x=5.0, y=5.0):
        for t, rx in ((1000, 0.0), (1060, 2.0), (1130, 4.0)):
            self.look(t, rx, x, y)
        return sg.public(SITE, DATUM)['suggestions']


class Thresholds(Base):
    def test_confirmed_only_when_all_three_hold(self):
        self.look(1000, 0.0)
        self.look(1060, 2.0)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])
        stats = self.look(1130, 4.0)
        self.assertEqual(stats['confirmed'], 1)
        got = sg.public(SITE, DATUM)['suggestions']
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['state'], 'new')
        self.assertEqual(got[0]['seen'], 3)
        self.assertEqual(sorted(got[0]), sorted(
            ['id', 'kind', 'points', 'size_m', 'seen', 'first', 'last', 'state']))

    def test_one_position_is_not_enough(self):
        for t in (1000, 1100, 1200, 1300):
            self.look(t, 0.0)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])

    def test_short_span_is_not_enough(self):
        for t, rx in ((1000, 0.0), (1030, 2.0), (1060, 4.0), (1090, 6.0)):
            self.look(t, rx)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])

    def test_quick_repeats_count_once(self):
        for t, rx in ((1000, 0.0), (1001, 2.0), (1002, 4.0), (1200, 6.0)):
            self.look(t, rx)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])

    def test_two_rows_on_one_thing_count_once(self):
        sg.observe(SITE, [row(5, 5), row(5.1, 5)], {'x': 0, 'y': 0}, DATUM, None, now=1000)
        sg.observe(SITE, [row(5, 5), row(5.1, 5)], {'x': 2, 'y': 0}, DATUM, None, now=1060)
        self.assertEqual(sg.load(SITE)['entries'][0]['seen'], 2)


class Filtering(Base):
    def test_only_static_new(self):
        for kind in ('static_known', 'moving', 'noise', 'unknown'):
            self.look(1000, 0.0, kind=kind)
        self.assertEqual(sg.load(SITE)['entries'], [])

    def test_huge_extent_is_not_a_thing(self):
        self.look(1000, 0.0, extent=9.0)
        self.assertEqual(sg.load(SITE)['entries'], [])

    def test_matches_objects(self):
        objs = [
            {'type': 'obstacle', 'geometry': 'area',
             'points': [[4, 4], [6, 4], [6, 6], [4, 6]]},
            {'type': 'landmark', 'geometry': 'point', 'points': [[20, 20]]},
            {'type': 'wall', 'geometry': 'line', 'points': [[30, 0], [30, 10]]},
            {'type': 'waypoint', 'geometry': 'point', 'points': [[40, 40]],
             'protected': True},
        ]
        self.look(1000, 0.0, 5, 5, objs)           # inside the area
        self.look(1000, 0.0, 20.3, 20, objs)       # near the point
        self.look(1000, 0.0, 30.4, 5, objs)        # near the line
        self.look(1000, 0.0, 40.8, 40, objs)       # around the dock
        self.assertEqual(sg.load(SITE)['entries'], [])

    def test_zone_and_place_do_not_match(self):
        objs = [{'type': 'zone', 'geometry': 'area',
                 'points': [[0, 0], [10, 0], [10, 10], [0, 10]]},
                {'type': 'place', 'geometry': 'area',
                 'points': [[0, 0], [10, 0], [10, 10], [0, 10]]},
                {'type': 'waypoint', 'geometry': 'point', 'points': [[5, 5]]}]
        self.look(1000, 0.0, 5, 5, objs)
        self.assertEqual(len(sg.load(SITE)['entries']), 1)

    def test_object_drawn_later_removes_entry(self):
        self.confirm()
        objs = [{'type': 'obstacle', 'geometry': 'area',
                 'points': [[4, 4], [6, 4], [6, 6], [4, 6]]}]
        self.look(1200, 0.0, 50, 50, objs)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])


class States(Base):
    def test_ignored_stays_ignored_at_that_place(self):
        sid = self.confirm()[0]['id']
        self.assertEqual(sg.mark(SITE, sid, 'ignored'), (True, ''))
        for t, rx in ((2000, 0.0), (2100, 2.0), (2200, 4.0), (9000000, 6.0)):
            self.look(t, rx, 5.2, 5.2)
        got = sg.public(SITE, DATUM)['suggestions']
        self.assertEqual([g['state'] for g in got], ['ignored'])
        self.assertEqual(got[0]['seen'], 3)

    def test_ignored_does_not_silence_elsewhere(self):
        sid = self.confirm()[0]['id']
        sg.mark(SITE, sid, 'ignored')
        self.look(2000, 0.0, 9, 9)
        self.assertEqual(len(sg.load(SITE)['entries']), 2)

    def test_ignored_survives_expiry(self):
        sid = self.confirm()[0]['id']
        sg.mark(SITE, sid, 'ignored')
        self.look(1130 + 400 * 86400, 0.0, 50, 50)
        self.assertEqual(len(sg.public(SITE, DATUM)['suggestions']), 1)

    def test_expiry(self):
        self.look(1000, 0.0)                       # candidate
        self.look(1000 + 15 * 86400, 0.0, 50, 50)
        ents = sg.load(SITE)['entries']
        self.assertEqual([e['first'] for e in ents], [1000 + 15 * 86400])

    def test_accept_as_landmark_and_obstacle(self):
        sid = self.confirm()[0]['id']
        obj, why = sg.accept_object(SITE, sid, None, DATUM)
        self.assertEqual(why, '')
        self.assertEqual((obj['family'], obj['type'], obj['geometry']),
                         ('edit', 'obstacle', 'area'))
        self.assertEqual(obj['source'], 'robot')
        self.assertNotIn('id', obj)
        self.assertEqual(len(obj['points']), 4)
        self.assertAlmostEqual(sum(p[0] for p in obj['points']) / 4, 5.0, 2)
        obj, _ = sg.accept_object(SITE, sid, {'type': 'landmark', 'name': 'Strom',
                                              'props': {'category': 'strom'}}, DATUM)
        self.assertEqual((obj['family'], obj['geometry'], obj['name']),
                         ('place', 'point', 'Strom'))
        self.assertEqual(sg.load(SITE)['entries'][0]['state'], 'new')

    def test_accept_refusals(self):
        sid = self.confirm()[0]['id']
        self.assertIsNone(sg.accept_object(SITE, 'nope', None, DATUM)[0])
        self.assertIsNone(sg.accept_object(SITE, sid, None, None)[0])
        self.assertIsNone(sg.accept_object(SITE, sid, {'type': 'zone'}, DATUM)[0])
        sg.mark(SITE, sid, 'accepted')
        self.assertIsNone(sg.accept_object(SITE, sid, None, DATUM)[0])

    def test_accepted_is_silent(self):
        sid = self.confirm()[0]['id']
        sg.mark(SITE, sid, 'accepted')
        self.look(5000, 0.0, 5, 5)
        self.assertEqual(len(sg.load(SITE)['entries']), 1)

    def test_candidate_cannot_be_marked(self):
        self.look(1000, 0.0)
        sid = sg.load(SITE)['entries'][0]['id']
        self.assertFalse(sg.mark(SITE, sid, 'ignored')[0])
        self.assertFalse(sg.mark(SITE, sid, 'bogus')[0])


class Files(Base):
    def test_missing_file(self):
        self.assertEqual(sg.mtime(SITE), 0.0)
        self.assertEqual(sg.public(SITE, DATUM)['suggestions'], [])

    def test_no_datum_gives_empty_list(self):
        self.confirm()
        self.assertEqual(sg.public(SITE, None)['suggestions'], [])
        self.assertEqual(sg.observe(SITE, [row(1, 1)], {'x': 0, 'y': 0}, None)['added'], 0)

    def test_file_is_utm_topic_is_map(self):
        got = self.confirm()[0]
        cx = sum(p[0] for p in got['points']) / len(got['points'])
        self.assertAlmostEqual(cx, 5.0, 2)
        with open(sg.path(SITE)) as f:
            self.assertGreater(json.load(f)['entries'][0]['c'][0], 400000)

    def test_nothing_new_writes_nothing(self):
        self.look(1000, 0.0)
        before = sg.load(SITE)['seq']
        sg.observe(SITE, [], {'x': 0, 'y': 0}, DATUM, None, now=1001)
        self.assertEqual(sg.load(SITE)['seq'], before)

    def test_damaged_file_is_set_aside(self):
        with open(sg.path(SITE), 'w') as f:
            f.write('{broken')
        self.assertEqual(sg.load(SITE)['entries'], [])
        self.assertTrue(any('.bad.' in n for n in os.listdir(bundle.site_dir(SITE))))

    def test_concurrent_writers_lose_nothing(self):
        sid = self.confirm()[0]['id']

        def feed(base):
            for i in range(30):
                sg.observe(SITE, [row(20 + base * 3, i * 3.0)], {'x': 0, 'y': 0},
                           DATUM, None, now=2000 + i)

        def flip():
            for i in range(30):
                sg.mark(SITE, sid, 'ignored' if i % 2 else 'new')

        threads = [threading.Thread(target=feed, args=(b,)) for b in range(3)]
        threads.append(threading.Thread(target=flip))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(sg.load(SITE)['entries']), 91)


class RealWorldQuery(Base):
    """The rows really come from world_query.obstacles."""

    def test_static_new_flows_through(self):
        import numpy as np
        path = os.path.join(HERE, '..', '..', '..', '..', '..', 'vitulus_navi', 'src',
                            'vitulus_navi', 'world_query.py')
        path = os.environ.get('WORLD_QUERY_PY', path)
        if not os.path.exists(path):
            self.skipTest('world_query.py not next to this worktree')
        spec = importlib.util.spec_from_file_location('wq_for_test', path)
        wq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wq)

        def grid(a):
            return wq.Grid({'w': a.shape[1], 'h': a.shape[0], 'res': 0.05, 'ox': 0.0,
                            'oy': 0.0, 'data': a.ravel(), 'topic': 't'})

        map_a = np.zeros((200, 200), np.int8)
        map_a[10:190, 20] = 100                       # wall, x = 1.0
        cost = map_a.copy()
        cost[100:108, 120:128] = 100                  # new block at (6.2, 5.2)
        robot = {'x': 5.0, 'y': 5.0, 'yaw': 0.0}
        rows = wq.obstacles(grid(cost), grid(map_a), robot, radius_m=5.0)
        self.assertEqual(sorted(r['kind'] for r in rows)[-1], 'static_new')
        sg.observe(SITE, rows, robot, DATUM, None, now=1000)
        ents = sg.load(SITE)['entries']
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0]['kind'], 'landmark')


if __name__ == '__main__':
    unittest.main()
