#!/usr/bin/env python3
"""Zone report of a mow run without ROS: python3 test/test_mowreport.py

A 4 x 4 m zone mowed in stripes 0.25 m apart (deck 0.29), one stripe block left
out (the gap), one heavy patch, an off-path event; an older 34-column run; the
stored report, its reuse and the layer_query read.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import bundle, mowreport as mr, mowtrail as mt, worldlayers as wl  # noqa: E402

E0, N0 = 500000.0, 5500000.0
T0 = 1793540000.0
ZONE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
GAP = (2.0, 4.0, 1.4, 2.6)          # x0, x1, y0, y1 left out


def stripes(gap=True, heavy=True, stride=mt.STRIDE):
    """Rows of a boustrophedon over ZONE, 0.1 m apart, cutting at load 0.12."""
    rows = []
    t = 0
    y = 0.1
    k = 0
    while y < 4.0:
        xs = np.arange(0.1, 3.95, 0.1)
        if k % 2:
            xs = xs[::-1]
        brk = True
        for x in xs:
            if gap and GAP[0] <= x <= GAP[1] and GAP[2] <= y <= GAP[3]:
                brk = True
                continue
            row = [mt.NONE] * mt.STRIDE
            row[mt.X], row[mt.Y] = int(round(x * 100)), int(round(y * 100))
            row[mt.LOAD] = 12
            if heavy and abs(x - 1.0) <= 0.3 and abs(y - 1.0) <= 0.3:
                row[mt.LOAD] = 60
            row[mt.STATE] = mt.CUTTING + (mt.BRK if brk else 0)
            row[mt.T] = t
            row[mt.FIX] = 4
            row[mt.LIDAR] = 8
            row[mt.JUMP] = 3
            row[mt.XTE] = 2
            rows.append(row)
            brk = False
            t += 5
        y += 0.25
        k += 1
    arr = np.array(rows, np.int32)
    return arr[:, :stride]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        self.addCleanup(lambda: setattr(bundle, 'SITES_ROOT', self.root))
        os.makedirs(bundle.site_dir('a'))
        with open(os.path.join(bundle.site_dir('a'), 'datum.yaml'), 'w') as f:
            f.write('utm_e: %r\nutm_n: %r\nyaw_rad: 0.0\nutm_zone: 33\n' % (E0, N0))
        bundle.save_zones('a', [{'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in ZONE],
                                 'props': {}}], utm_zone=33)
        os.makedirs(mt.run_dir('a'))

    def store(self, run, rows, events=None, program='P'):
        start = T0 + int(run[9:15])
        meta = {'v': mt.FORMAT_VERSION, 'run': run, 'site': 'a', 'program': program,
                'start': start, 'end': start + 600.0, 'stop_reason': 'succeeded',
                'datum': [E0, N0, 0.0]}
        if events is None:
            events = [[0, 10, 10, 'ctx', 'Z|1|1|0']]
        mt._write_npz(os.path.join(mt.run_dir('a'), run + '.npz'), meta, rows, events)


class Report(Base):
    def test_full_cover(self):
        self.store('20261101_100000', stripes(gap=False, heavy=False))
        rep = mr.report('a', '20261101_100000')
        self.assertTrue(rep['ok'], rep)
        z = rep['zones'][0]
        self.assertEqual(z['zone'], 'Z')
        self.assertEqual(rep['zones_from'], 'ctx')
        self.assertGreater(z['coverage']['frac'], 0.97)
        self.assertEqual(z['coverage']['gaps'], [])
        self.assertAlmostEqual(z['load']['p50'], 0.12)
        self.assertEqual(z['heavy'], [])
        self.assertEqual([a['kind'] for a in rep['advice']], [])
        self.assertGreater(z['cut_m'], 50)
        self.assertGreater(z['time_mow_s'], 100)

    def test_gap_and_heavy(self):
        rows = stripes()
        events = [[0, 10, 10, 'ctx', 'Z|1|1|0'], [400, 300, 300, 'off_path', 'xte 0.30 m'],
                  [410, 310, 300, 'off_path', ''], [420, 320, 300, 'off_path', '']]
        self.store('20261101_110000', rows, events)
        rep = mr.report('a', '20261101_110000')
        z = rep['zones'][0]
        cov = z['coverage']
        self.assertEqual(cov['gaps_total'], 1, cov)
        g = cov['gaps'][0]
        # 2 x 1.2 m left out, the neighbouring stripes eat into it by deck/2 - 0.15
        self.assertTrue(GAP[0] <= g['x'] <= GAP[1] and GAP[2] <= g['y'] <= GAP[3], g)
        self.assertTrue(1.0 < g['area_m2'] < 2.6, g)
        self.assertTrue(0.75 < cov['frac'] < 0.92, cov)
        h = z['heavy']
        self.assertEqual(len(h), 1, h)
        self.assertAlmostEqual(h[0]['x'], 1.0, delta=0.15)
        self.assertAlmostEqual(h[0]['y'], 1.0, delta=0.15)
        self.assertEqual(h[0]['max_load'], 0.6)
        kinds = [a['kind'] for a in rep['advice']]
        self.assertEqual(kinds, ['resow_gaps', 'heavy_area', 'path_tracking'], rep['advice'])
        self.assertEqual(rep['advice'][0]['where'], [g['x'], g['y']])
        self.assertEqual(z['loc']['off_path_n'], 3)
        self.assertEqual(z['loc']['xte_p95_cm'], 2)
        self.assertEqual(z['loc']['rtk_frac'], 1.0)

    def test_obstacle_is_not_a_gap(self):
        from vitulus_mapping import objects
        obst = {'family': 'edit', 'type': 'obstacle', 'name': 'strom', 'geometry': 'area',
                'points': [(E0 + x, N0 + y) for x, y in
                           ((1.9, 1.3), (4.1, 1.3), (4.1, 2.7), (1.9, 2.7))]}
        orig = objects.load_all
        objects.load_all = lambda site, now=None, frame='map': [
            o for o in orig(site)] + [dict(obst, points=[(x - E0, y - N0) for x, y in obst['points']])]
        self.addCleanup(lambda: setattr(objects, 'load_all', orig))
        self.store('20261101_120000', stripes())
        z = mr.report('a', '20261101_120000')['zones'][0]
        self.assertEqual(z['coverage']['gaps'], [], z['coverage'])
        self.assertGreater(z['coverage']['excluded_m2'], 2.5)

    def test_old_34_columns(self):
        rows = stripes(stride=34)
        self.assertEqual(rows.shape[1], 34)
        self.store('20261101_130000', rows, events=[])     # no ctx: program zones
        with open(os.path.join(bundle.site_dir('a'), 'programs.yaml'), 'w') as f:
            f.write('- name: P\n  zone_names: [Z]\n')
        rep = mr.report('a', '20261101_130000')
        self.assertTrue(rep['ok'], rep)
        self.assertEqual(rep['zones_from'], 'program')
        z = rep['zones'][0]
        self.assertIsNone(z['loc']['xte_p95_cm'])          # column 37, not in old runs
        self.assertEqual(z['loc']['lidar_p95_cm'], 8)
        self.assertEqual(z['coverage']['gaps_total'], 1)
        json.dumps(rep)                                    # all plain types

    def test_outside_before_ctx(self):
        rows = stripes(gap=False, heavy=False)
        self.store('20261101_140000', rows, [[100, 0, 0, 'ctx', 'Z|1|3|0']])
        rep = mr.report('a', '20261101_140000')
        self.assertEqual(rep['outside']['rows'], int((rows[:, mt.T] < 100).sum()))
        self.assertEqual(len(rep['zones']), 1)

    def test_unknown_zone(self):
        self.store('20261101_150000', stripes(), [[0, 0, 0, 'ctx', 'Gone|1|1|0']])
        z = mr.report('a', '20261101_150000')['zones'][0]
        self.assertFalse(z['known'])
        self.assertIsNone(z['coverage'])


class Files(Base):
    def test_save_load_backfill(self):
        self.store('20261101_100000', stripes())
        self.store('20261101_110000', stripes(gap=False))
        self.assertFalse(mr.load('a', 'nope_bad')['ok'])
        done = mr.backfill('a')
        self.assertEqual(sorted(done), ['20261101_100000', '20261101_110000'])
        self.assertEqual(mr.backfill('a'), [])             # nothing due
        path = mr.report_path('a', '20261101_100000')
        self.assertTrue(os.path.exists(path))
        self.assertFalse([n for n in os.listdir(mt.run_dir('a')) if n.endswith('.tmp')])
        rep = mr.load('a', '20261101_100000')
        self.assertEqual(rep['path'], path)
        # newest finished by default (index rebuilt from the npz files)
        self.assertEqual(mr.load('a')['run'], '20261101_110000')
        # datum moved: the stored one is stale, computed again
        with open(os.path.join(bundle.site_dir('a'), 'datum.yaml'), 'w') as f:
            f.write('utm_e: %r\nutm_n: %r\nyaw_rad: 0.0\nutm_zone: 33\n' % (E0 + 1.0, N0))
        self.assertIsNone(mr.load('a', '20261101_100000')['path'])
        # a run retention removed takes its report along
        os.remove(os.path.join(mt.run_dir('a'), '20261101_100000.npz'))
        mr.backfill('a')
        self.assertFalse(os.path.exists(path))

    def test_layer_query(self):
        got = wl.layer_query('a', 'mow_report')
        self.assertTrue(got['ok'])
        self.assertIsNone(got['result'])                   # no run = not recorded
        self.store('20261101_100000', stripes())
        mr.backfill('a')
        got = wl.layer_query('a', 'mow_report', run='20261101_100000')
        self.assertTrue(got['ok'], got)
        self.assertEqual(got['result']['zones'][0]['zone'], 'Z')
        self.assertFalse(wl.layer_query('a', 'mow_report', run='20990101_000000')['ok'])
        res = wl.layer_query('a', 'mow_load')['result']
        self.assertEqual(res['report_path'], mr.report_path('a', '20261101_100000'))

    def test_cli(self):
        self.store('20261101_100000', stripes())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(mr.main(['--site', 'a', '--save']), 0)
            self.assertEqual(mr.main(['--site', 'a', '20990101_000000']), 1)
        self.assertTrue(os.path.exists(mr.report_path('a', '20261101_100000')))
        self.assertIn('resow_gaps Z', out.getvalue())


def rules(*objs):
    """Patch objects.load_all to add map-frame objects [(family, type, ring)]."""
    from vitulus_mapping import objects
    orig = objects.load_all
    extra = [{'family': f, 'type': ty, 'name': ty, 'geometry': 'area', 'points': ring}
             for f, ty, ring in objs]
    objects.load_all = lambda site, now=None, frame='map': list(orig(site)) + extra
    return lambda: setattr(objects, 'load_all', orig)


def box(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def row(x, y, t, state=mt.CUTTING, load=12, brk=False):
    r = [mt.NONE] * mt.STRIDE
    r[mt.X], r[mt.Y], r[mt.T] = int(round(x * 100)), int(round(y * 100)), t
    r[mt.LOAD], r[mt.STATE] = load, state + (mt.BRK if brk else 0)
    r[mt.FIX], r[mt.LIDAR], r[mt.JUMP], r[mt.XTE] = 4, 8, 3, 2
    return r


class Progress(Base):
    """Resumed and stopped programs, zones never reached, the way home."""

    def split(self):
        rows = stripes(gap=False, heavy=False)
        low = rows[rows[:, mt.Y] < 200]
        high = rows[rows[:, mt.Y] >= 200].copy()
        high[0, mt.STATE] |= mt.BRK
        return low, high

    def store_meta(self, run, rows, events, stop, start):
        meta = {'v': mt.FORMAT_VERSION, 'run': run, 'site': 'a', 'program': 'P',
                'start': start, 'end': start + 600.0, 'stop_reason': stop,
                'datum': [E0, N0, 0.0]}
        mt._write_npz(os.path.join(mt.run_dir('a'), run + '.npz'), meta, rows, events)

    def test_stopped_is_unfinished_not_resow(self):
        low, _high = self.split()
        self.store_meta('20261101_100000', low, [[0, 10, 10, 'ctx', 'Z|1|2|0']],
                        'stopped:user_stop', T0)
        rep = mr.report('a', '20261101_100000')
        self.assertFalse(rep['complete'])
        self.assertFalse(rep['resumed'])
        self.assertEqual(rep['paths_first'], ['Z', 1])
        self.assertEqual(rep['paths_total'], {'Z': 2})
        kinds = [a['kind'] for a in rep['advice']]
        self.assertIn('unfinished', kinds)
        self.assertNotIn('resow_gaps', kinds)
        self.assertEqual(rep['advice'][0]['remaining'][0]['paths'], [1, 2])
        self.assertGreater(rep['zones'][0]['coverage']['gap_m2'], 3)

    def test_resumed_union(self):
        low, high = self.split()
        self.store_meta('20261101_100000', low, [[0, 10, 10, 'ctx', 'Z|1|2|0']],
                        'stopped:user_stop', T0)
        # the second run starts again at path 2, ten hours later
        self.store_meta('20261101_200000', high, [[0, 10, 210, 'ctx', 'Z|2|2|0']],
                        'succeeded', T0 + 36000.0)
        rep = mr.report('a', '20261101_200000')
        self.assertTrue(rep['resumed'])
        self.assertEqual(rep['continues'], ['20261101_100000'])
        self.assertFalse(rep['complete'])          # path 1 was never finished
        self.assertIn('unfinished', [a['kind'] for a in rep['advice']])
        self.assertNotIn('resow_gaps', [a['kind'] for a in rep['advice']])
        self.assertGreater(rep['zones'][0]['coverage']['frac'], 0.97)
        # the first run did finish path 1: the union is complete
        os.remove(os.path.join(mt.run_dir('a'), '20261101_100000.npz'))
        self.store_meta('20261101_100000', low, [[0, 10, 10, 'ctx', 'Z|1|2|0'],
                                                [int(low[-1, mt.T]), 0, 0, 'ctx', 'Z|2|2|0']],
                        'stopped:user_stop', T0)
        rep = mr.report('a', '20261101_200000')
        self.assertTrue(rep['complete_union'])
        self.assertTrue(rep['zones'][0]['complete'])
        self.assertEqual([a['kind'] for a in rep['advice']], [])
        # older than chain_h: alone again, half the zone, no resow
        rep = mr.report('a', '20261101_200000', cfg={'chain_h': 5.0})
        self.assertEqual(rep['continues'], [])
        self.assertLess(rep['zones'][0]['coverage']['frac'], 0.7)
        self.assertNotIn('resow_gaps', [a['kind'] for a in rep['advice']])

    def test_program_zone_not_reached(self):
        bundle.save_zones('a', [{'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in ZONE],
                                 'props': {}},
                                {'name': 'W', 'polygon': [(E0 + x, N0 + y) for x, y in
                                                          box(10, 14, 0, 4)], 'props': {}}],
                          utm_zone=33)
        with open(os.path.join(bundle.site_dir('a'), 'programs.yaml'), 'w') as f:
            f.write('- name: P\n  zone_names: [Z, W]\n')
        self.store_meta('20261101_100000', stripes(gap=False, heavy=False),
                        [[0, 10, 10, 'ctx', 'Z|1|1|0']], 'succeeded', T0)  # W added since
        rep = mr.report('a', '20261101_100000')
        self.assertFalse(rep['complete'])
        names = [z['zone'] for z in rep['zones']]
        self.assertEqual(names, ['Z', 'W'])
        w = rep['zones'][1]
        self.assertFalse(w['visited'])
        self.assertEqual(w['coverage']['frac'], 0.0)
        self.assertTrue(rep['zones'][0]['complete'])
        self.assertAlmostEqual(rep['total']['coverage_frac'], 0.5, delta=0.03)
        un = [a for a in rep['advice'] if a['kind'] == 'unfinished'][0]
        self.assertEqual(un['zone'], 'W')
        self.assertFalse(un['remaining'][0]['visited'])

    def test_way_home_is_outside(self):
        rows = [list(r) for r in stripes(gap=False, heavy=False)]
        t = int(rows[-1][mt.T])
        # after the last path: transit to the dock at x 12, poor localization there
        for i in range(60):
            r = row(4.5 + i * 0.15, 3.85, t + 5 * (i + 1), state=mt.TRANSIT, load=mt.NONE)
            r[mt.LIDAR], r[mt.JUMP], r[mt.FIX] = 90, 80, 1
            rows.append(r)
        rows = np.array(rows, np.int32)
        events = [[0, 10, 10, 'ctx', 'Z|1|1|0'], [t + 290, 1150, 385, 'gps_jump', '1.2 m'],
                  [t + 295, 1160, 385, 'fix_down', '']]
        self.store('20261101_100000', rows, events)
        rep = mr.report('a', '20261101_100000')
        z = rep['zones'][0]
        self.assertGreaterEqual(rep['outside']['rows'], 55)
        self.assertEqual(z['loc']['gps_jumps'], 0)
        self.assertEqual(z['loc']['rtk_frac'], 1.0)
        self.assertNotIn('gnss_check', [a['kind'] for a in rep['advice']])


class Shapes(Base):
    def test_nogo_and_noblade_are_not_gaps(self):
        undo = rules(('place', 'nogo', box(1.9, 4.1, 1.3, 2.0)),
                     ('place', 'noblade', box(1.9, 4.1, 2.0, 2.7)))
        self.addCleanup(undo)
        self.store('20261101_120000', stripes())
        z = mr.report('a', '20261101_120000')['zones'][0]
        self.assertEqual(z['coverage']['gaps'], [], z['coverage'])
        self.assertGreater(z['coverage']['excluded_m2'], 2.5)

    def test_multipolygon_and_holes(self):
        bundle.save_zones('a', [{'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in ZONE],
                                 'props': {}},
                                {'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in
                                                          box(10, 12, 0, 2)], 'props': {}}],
                          utm_zone=33)
        path = bundle.zones_path('a')
        with open(path) as f:
            fc = json.load(f)
        hole = [[E0 + x, N0 + y] for x, y in box(1.0, 2.0, 1.0, 2.0)]
        fc['features'][0]['geometry']['coordinates'].append(hole + [hole[0]])
        with open(path, 'w') as f:
            json.dump(fc, f)
        self.store('20261101_120000', stripes(gap=False, heavy=False))
        rep = mr.report('a', '20261101_120000')
        self.assertEqual(len(rep['zones']), 1)
        cov = rep['zones'][0]['coverage']
        self.assertAlmostEqual(cov['zone_m2'], 16 + 4 - 1, delta=0.6)
        self.assertEqual(cov['gaps_total'], 1)               # the second part, unmowed
        self.assertAlmostEqual(cov['gaps'][0]['x'], 11.0, delta=0.6)

    def test_jumps_and_err_do_not_cut(self):
        rows = [list(r) for r in stripes()]
        t = int(rows[-1][mt.T])
        # a pose jump across the gap and firmware ERR rows inside it
        rows.append(row(1.5, 2.0, t + 5, brk=True))
        rows.append(row(4.0, 2.0, t + 10))
        for i, x in enumerate(np.arange(2.1, 3.9, 0.1)):
            rows.append(row(x, 1.8, t + 15 + 5 * i, state=mt.OVERLOAD, load=mt.NONE,
                            brk=(i == 0)))
        self.store('20261101_120000', np.array(rows, np.int32))
        z = mr.report('a', '20261101_120000')['zones'][0]
        self.assertEqual(z['coverage']['gaps_total'], 1, z['coverage'])
        self.assertGreater(z['coverage']['gap_m2'], 1.0)


class Validity(Base):
    def test_signature(self):
        self.store('20261101_100000', stripes())
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])
        self.assertIsNotNone(mr.load('a', '20261101_100000')['path'])
        self.assertEqual(mr.backfill('a', pause_s=0), [])
        # a rule changes: stale
        undo = rules(('place', 'nogo', box(1.9, 4.1, 1.3, 2.7)))
        self.assertIsNone(mr.load('a', '20261101_100000')['path'])
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])
        undo()
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])
        # the zone changes
        bundle.save_zones('a', [{'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in
                                                          box(0, 4, 0, 4.5)], 'props': {}}],
                          utm_zone=33)
        self.assertIsNone(mr.load('a', '20261101_100000')['path'])
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])
        # another deck
        self.assertIsNone(mr.load('a', '20261101_100000', {'deck_m': 0.4})['path'])
        # the run stored again
        path = os.path.join(mt.run_dir('a'), '20261101_100000.npz')
        os.utime(path, (time.time() + 5, time.time() + 5))
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])

    def test_old_version_recomputed(self):
        self.store('20261101_100000', stripes())
        mr._atomic_write(mr.report_path('a', '20261101_100000'), {'ok': True, 'v': 1})
        self.assertEqual(mr.backfill('a', pause_s=0), ['20261101_100000'])


class Small(unittest.TestCase):
    def test_weighted_percentile(self):
        vals = np.array([0.1, 0.9])
        self.assertEqual(mr._pct(vals, 50, np.array([9.0, 1.0])), 0.1)
        self.assertEqual(mr._pct(vals, 95, np.array([9.0, 1.0])), 0.9)
        self.assertEqual(mr._pct(vals, 50, np.array([0.0, 0.0])), 0.5)

    def test_gnss_value_is_the_trigger(self):
        z = {'visited': True, 'heavy': [], 'coverage': None,
             'loc': {'rtk_frac': 1.0, 'lidar_p95_cm': 10, 'jump_p95_cm': 5, 'gps_jumps': 2,
                     'gps_jump_at': [1.0, 2.0], 'worst': [5.0, 5.0, 10], 'xte_p95_cm': None,
                     'off_path_n': 0, 'off_path': []}}
        a = mr._advice('Z', z, mr._cfg(None))[0]
        self.assertEqual((a['kind'], a['metric'], a['value'], a['where']),
                         ('gnss_check', 'gps_jumps', 2, [1.0, 2.0]))


class WorkerQueue(unittest.TestCase):
    def test_merge_limit_child(self):
        import threading
        gate = threading.Event()
        calls, answers = [], []

        def runner(args, timeout_s):
            gate.wait(5)
            calls.append(args)
            if '--all' in args:
                return 0, 'written: -\n', ''
            return 0, json.dumps({'ok': True, 'run': args[-1]}), ''
        w = mr.Worker(answers.append, limit=3, runner=runner)
        self.addCleanup(w.close)
        w.request('a', 'r1', 1)                    # taken by the thread, waits at the gate
        for _ in range(50):
            if not w.pending():
                break
            time.sleep(0.01)
        w.request('a', 'r2', 2)
        w.request('a', 'r2', 3)                    # merged
        w.backfill('a')
        w.backfill('a')                            # merged
        w.request('a', 'r3', 4)
        w.request('a', 'r4', 5)                    # over the limit: busy
        self.assertEqual(len(w.pending()), 3)
        self.assertEqual(answers[-1]['reason'], 'busy')
        gate.set()
        for _ in range(200):
            if len(calls) == 4:
                break
            time.sleep(0.01)
        time.sleep(0.05)
        self.assertEqual(len(calls), 4)            # r1, r2 once, backfill once, r3
        r2 = [a for a in answers if a.get('run') == 'r2'][0]
        self.assertEqual((r2['req'], r2['reqs']), (3, [2, 3]))
        self.assertIn('--deck', calls[0])

    def test_real_child(self):
        env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get('PYTHONPATH', ''))
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update(env)
        try:
            code, out, _err = mr._child(['--site', 'no_such_site_x', '--json'], 60)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['reason'], 'no run')


class Speed(Base):
    def test_400k_rows(self):
        """A long run on a 40 x 40 m zone: under 3 s CPU."""
        bundle.save_zones('a', [{'name': 'Z', 'polygon': [(E0 + x, N0 + y) for x, y in
                                                          box(0, 40, 0, 40)], 'props': {}}],
                          utm_zone=33)
        n = 400000
        k = np.arange(n)
        lane = k // 4000                      # 100 lanes of 4000 rows, 1 cm apart
        x = np.where(lane % 2, 39.9 - (k % 4000) * 0.00995, 0.1 + (k % 4000) * 0.00995)
        y = 0.2 + lane * 0.39
        rows = np.full((n, mt.STRIDE), mt.NONE, np.int32)
        rows[:, mt.X], rows[:, mt.Y] = np.round(x * 100), np.round(y * 100)
        rows[:, mt.T] = k
        rows[:, mt.LOAD] = 12 + (k % 97 == 0) * 50
        rows[:, mt.STATE] = mt.CUTTING + np.where(k % 4000 == 0, mt.BRK, 0)
        rows[:, mt.FIX], rows[:, mt.LIDAR], rows[:, mt.JUMP], rows[:, mt.XTE] = 4, 8, 3, 2
        events = [[int(i), 100, 100, 'ctx', 'Z|%d|100|0' % (i // 4000 + 1)]
                  for i in range(0, n, 4000)]
        self.store('20261101_100000', rows, events)
        c0 = time.process_time()
        rep = mr.report('a', '20261101_100000')
        cpu = time.process_time() - c0
        self.assertTrue(rep['complete'])
        self.assertGreater(rep['zones'][0]['coverage']['frac'], 0.6)
        self.assertLess(cpu, 3.0, cpu)


if __name__ == '__main__':
    unittest.main()
