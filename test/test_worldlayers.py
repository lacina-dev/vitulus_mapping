#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.worldlayers (contract section 7).

Run: python3 -m pytest test/test_worldlayers.py   (from the package root)
"""

import base64
import json
import math
import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import bundle, worldlayers as wl  # noqa: E402

T0 = 1790000000.0


def decode_png(raw):
    """RGBA array of a PNG written by encode_png (no decoder dependency)."""
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    pos, idat, w, h = 8, b'', 0, 0
    while pos < len(raw):
        n = int.from_bytes(raw[pos:pos + 4], 'big')
        tag = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + n]
        assert zlib.crc32(tag + body) == int.from_bytes(raw[pos + 8 + n:pos + 12 + n], 'big')
        if tag == b'IHDR':
            w = int.from_bytes(body[:4], 'big')
            h = int.from_bytes(body[4:8], 'big')
        elif tag == b'IDAT':
            idat += body
        pos += 12 + n
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 4 + 1)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(h, w, 4)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        self.pubs = []
        self.add_site('a')
        self.serve('a')

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        self.tmp.cleanup()

    def add_site(self, site, e=493000.0, n=5540000.0, yaw=0.0):
        os.makedirs(bundle.site_dir(site), exist_ok=True)
        with open(os.path.join(bundle.site_dir(site), 'datum.yaml'), 'w') as f:
            f.write('utm_e: %r\nutm_n: %r\nyaw_rad: %r\n' % (e, n, yaw))

    def serve(self, site):
        p = os.path.join(self.tmp.name, '.last_served')
        if site is None:
            if os.path.exists(p):
                os.remove(p)
            return
        with open(p, 'w') as f:
            json.dump({'site': site, 'raster': 'combined'}, f)

    def recorder(self, **cfg):
        rec = wl.WorldRecorder(lambda layer, msg: self.pubs.append((layer, msg)), cfg=cfg)
        rec.poll(T0)
        return rec

    def file(self, site, name):
        return os.path.join(bundle.site_dir(site), name)


class Helpers(unittest.TestCase):
    def test_fix_class(self):
        self.assertEqual(wl.fix_class(3, 0x83), 4)      # carrSoln fixed
        self.assertEqual(wl.fix_class(3, 0x43), 3)      # the live robot: float
        self.assertEqual(wl.fix_class(3, 0x03), 2)
        self.assertEqual(wl.fix_class(3, 0x01), 1)
        self.assertEqual(wl.fix_class(0, 0x00), 0)
        self.assertEqual(wl.fix_class(3, 0x80), 0)      # fix not OK
        self.assertEqual(wl.fix_class(5, 0x01), 0)      # time only

    def test_png_roundtrip(self):
        img = np.random.randint(0, 255, (7, 11, 4)).astype(np.uint8)
        self.assertTrue((decode_png(wl.encode_png(img)) == img).all())


class Coverage(Base):
    def test_blade_rule(self):
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), False)
        self.assertEqual(rec.coverage.h, 0)
        rec.step(T0 + 1, (1.0, 1.0, 0.0), True)
        self.assertGreater(int((rec.coverage.a['last_mowed'] > 0).sum()), 0)

    def test_disc_area_and_time(self):
        g = wl.CoverageGrid()
        g.stamp(1.0, 1.0, 1.0, 1.0, 1.0, T0)
        n = int((g.a['last_mowed'] > 0).sum())
        self.assertAlmostEqual(n * 0.01, math.pi, delta=0.1)
        self.assertEqual(int(g.a['last_mowed'].max()), int(T0))

    def test_sweep_has_no_gaps(self):
        rec = self.recorder()
        for i in range(21):
            rec.step(T0 + i, (i * 0.08, 0.0, 0.0), True)
        g = rec.coverage
        row = int(math.floor((0.0 - g.oy) / g.res))
        c0 = int(math.floor((0.0 - g.ox) / g.res))
        c1 = int(math.floor((1.6 - g.ox) / g.res))
        self.assertTrue((g.a['last_mowed'][row, c0:c1] > 0).all())

    def test_stale_pose_or_jump_does_not_paint_across(self):
        rec = self.recorder()
        rec.step(T0, (0.0, 0.0, 0.0), True)
        rec.step(T0 + 1, (20.0, 0.0, 0.0), True)
        g = rec.coverage
        mid = int(math.floor((10.0 - g.ox) / g.res))
        self.assertFalse(g.a['last_mowed'][:, mid].any())

    def test_growth_all_directions_keeps_cells(self):
        g = wl.CoverageGrid()
        g.stamp(0.05, 0.05, 0.05, 0.05, 0.05, 1000)
        for x, y in ((-30.0, 2.0), (40.0, -25.0), (5.0, 60.0)):
            g.stamp(x, y, x, y, 0.1, 2000)
        c = int(math.floor((0.05 - g.ox) / g.res))
        r = int(math.floor((0.05 - g.oy) / g.res))
        self.assertEqual(int(g.a['last_mowed'][r, c]), 1000)
        self.assertAlmostEqual(g.ox / (wl.GROW_CELLS * g.res),
                               round(g.ox / (wl.GROW_CELLS * g.res)), 6)

    def test_absurd_pose_is_refused(self):
        g = wl.CoverageGrid()
        g.stamp(0, 0, 0, 0, 0.1, 1000)
        self.assertFalse(g.stamp(5000.0, 0, 5000.0, 0, 0.1, 1000))
        self.assertEqual(g.w, 2 * wl.GROW_CELLS)

    def test_restart_keeps_data(self):
        rec = self.recorder()
        rec.step(T0, (2.0, 3.0, 0.0), True)
        rec.close()
        rec2 = self.recorder()
        self.assertEqual(int(rec2.coverage.a['last_mowed'].max()), int(T0))
        self.assertEqual(rec2.coverage.ox, rec.coverage.ox)

    def test_flush_period_and_blade_stop(self):
        rec = self.recorder()
        rec.maintain(T0)
        rec.step(T0 + 1, (2.0, 3.0, 0.0), True)
        rec.maintain(T0 + 10)
        self.assertFalse(os.path.exists(self.file('a', 'coverage.npz')))
        rec.maintain(T0 + 31)
        self.assertTrue(os.path.exists(self.file('a', 'coverage.npz')))
        rec.step(T0 + 32, (2.2, 3.0, 0.0), True)
        rec.step(T0 + 33, (2.2, 3.0, 0.0), False)        # program end
        self.assertFalse(rec.coverage.dirty)
        with np.load(self.file('a', 'coverage.npz')) as d:
            self.assertEqual(int(d['last_mowed'].max()), int(T0 + 32))

    def test_no_temp_left_and_failed_write_stays_dirty(self):
        rec = self.recorder()
        rec.step(T0, (2.0, 3.0, 0.0), True)
        site_dir = bundle.site_dir('a')
        os.chmod(site_dir, 0o500)
        if os.access(site_dir, os.W_OK):
            os.chmod(site_dir, 0o700)
            self.skipTest('running as root')
        try:
            rec.flush()
            self.assertTrue(rec.coverage.dirty)
        finally:
            os.chmod(site_dir, 0o700)
        rec.flush()
        self.assertFalse(rec.coverage.dirty)
        self.assertEqual([n for n in os.listdir(site_dir) if n.endswith('.tmp')], [])

    def test_damaged_file_set_aside(self):
        with open(self.file('a', 'coverage.npz'), 'wb') as f:
            f.write(b'junk')
        rec = self.recorder()
        self.assertEqual(rec.coverage.h, 0)
        self.assertTrue(any('.bad.' in n for n in os.listdir(bundle.site_dir('a'))))

    def test_deck_offset(self):
        rec = self.recorder(deck_offset_x_m=1.0)
        rec.step(T0, (0.0, 0.0, math.pi / 2), True)
        got = wl.coverage_layer(rec.coverage, 'a', T0)
        self.assertAlmostEqual(got['origin'][1], 0.8, delta=0.2)


class Gnss(Base):
    def test_best_median_class(self):
        g = wl.GnssGrid()
        for acc, cls in ((15, 4), (18, 4), (19, 4), (900, 3), (5000, 1)):
            g.add(1.1, 1.1, acc, cls)
        self.assertEqual(int(g.a['hacc_best_mm'].min()), 15)
        self.assertEqual(int(g.a['class_best'].max()), 4)
        self.assertEqual(int(g.class_median().max()), 4)
        med = int(g.hacc_median_mm().max())
        self.assertTrue(14 <= med <= 20, med)

    def test_samples_by_travel_not_time(self):
        rec = self.recorder()
        for i in range(50):
            rec.step(T0 + i, (1.0, 1.0, 0.0), False, (20, 4))
        self.assertEqual(int(rec.gnss.a['class_hist'].sum()), 1)
        rec.step(T0 + 60, (1.3, 1.0, 0.0), False, (20, 4))
        self.assertEqual(int(rec.gnss.a['class_hist'].sum()), 2)

    def test_histogram_saturates_without_overflow(self):
        g = wl.GnssGrid()
        g.add(0, 0, 20, 4)
        g.a['hacc_hist'][:] = wl.HIST_CAP - 1
        g.add(0, 0, 20, 4)
        self.assertLess(int(g.a['hacc_hist'].max()), wl.HIST_CAP)

    def test_medians_accumulate_across_restarts(self):
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), False, (2000, 1))
        rec.close()
        rec = self.recorder()
        for i in range(3):
            rec._gnss_at = None
            rec.step(T0 + i, (1.0, 1.0, 0.0), False, (15, 4))
        rec.close()
        with np.load(self.file('a', 'gnss_quality.npz')) as d:
            self.assertEqual(int(d['class_median'].max()), 4)
            self.assertEqual(int(d['class_hist'].sum()), 4)
            self.assertEqual(d['hacc_edges_mm'].size + 1, d['hacc_hist'].shape[2])


class Sites(Base):
    def test_switch_keeps_sites_apart(self):
        self.add_site('b')
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), True)
        self.serve('b')
        rec.poll(T0 + 10)
        self.assertEqual(rec.site, 'b')
        self.assertEqual(rec.coverage.h, 0)
        self.assertTrue(os.path.exists(self.file('a', 'coverage.npz')))
        self.assertFalse(os.path.exists(self.file('b', 'coverage.npz')))
        rec.step(T0 + 11, (5.0, 5.0, 0.0), True)
        self.serve('a')
        rec.poll(T0 + 20)
        self.assertEqual(int(rec.coverage.a['last_mowed'].max()), int(T0))
        self.assertTrue(os.path.exists(self.file('b', 'coverage.npz')))

    def test_no_sweep_across_a_switch(self):
        self.add_site('b')
        rec = self.recorder()
        rec.step(T0, (0.5, 0.0, 0.0), True)
        self.serve('b')
        rec.poll(T0 + 10)
        rec.step(T0 + 11, (0.0, 0.0, 0.0), True)
        self.assertLess(int((rec.coverage.a['last_mowed'] > 0).sum()), 80)

    def test_no_served_site_records_nothing(self):
        self.serve(None)
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), True, (20, 4))
        rec.maintain(T0 + 100)
        rec.close()
        self.assertEqual(self.pubs, [])

    def test_site_without_datum_records_nothing_until_it_has_one(self):
        os.makedirs(bundle.site_dir('c'))
        self.serve('c')
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), True)
        self.assertFalse(os.path.exists(self.file('c', 'coverage.npz')))
        self.add_site('c')
        rec.poll(T0 + 10)
        rec.step(T0 + 11, (1.0, 1.0, 0.0), True)
        rec.close()
        self.assertTrue(os.path.exists(self.file('c', 'coverage.npz')))

    def test_losing_the_site_clears_layers(self):
        rec = self.recorder()
        rec.maintain(T0)
        self.pubs[:] = []
        self.serve(None)
        rec.poll(T0 + 10)
        self.assertEqual(sorted(p[0] for p in self.pubs), ['coverage', 'gnss', 'slope'])
        self.assertTrue(all(p[1]['w'] == 0 for p in self.pubs))

    def test_datum_change_keeps_the_place_on_the_ground(self):
        rec = self.recorder()
        rec.step(T0, (10.0, 0.0, 0.0), True)
        rec.close()
        # new datum: origin 3 m east, frame turned 90 deg
        self.add_site('a', e=493003.0, yaw=math.pi / 2)
        os.utime(self.file('a', 'datum.yaml'), (T0 + 5, T0 + 5))
        rec = self.recorder()
        got = wl.layer_query('a', 'coverage', points=[[0.0, -7.0], [10.0, 0.0]])
        self.assertEqual(got['result'][0]['last_mowed'], int(T0))
        self.assertIsNone(got['result'][1]['last_mowed'])
        self.assertTrue(rec.coverage.dirty)
        rec.close()
        rec = self.recorder()
        self.assertFalse(rec.coverage.dirty)

    def test_small_datum_noise_does_not_resample(self):
        g = wl.CoverageGrid()
        d = {'utm_e': 1.0, 'utm_n': 2.0, 'yaw_rad': 0.0}
        self.assertIs(g._rebased([1.005, 2.0, 0.0001], d), g)


class Layers(Base):
    def test_image_orientation_and_shape(self):
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), True)
        rec.step(T0 + 1, (1.0, 2.0, 0.0), True)
        rec.step(T0 + 2, (1.8, 2.0, 0.0), True)
        msg = wl.coverage_layer(rec.coverage, 'a', T0 + 3)
        for key in ('site', 'stamp', 'origin', 'res', 'w', 'h', 'png_b64', 'legend'):
            self.assertIn(key, msg)
        img = decode_png(base64.b64decode(msg['png_b64']))
        self.assertEqual(img.shape[:2], (msg['h'], msg['w']))
        # an L: bottom stroke at x = 1, top stroke to the east. Top row of the
        # image must be the highest y, so the far top-right corner is painted
        # and the bottom-right corner is empty.
        self.assertGreater(img[1, -2, 3], 0)
        self.assertEqual(img[-1, -1, 3], 0)
        self.assertAlmostEqual(msg['origin'][0], 0.85, delta=0.11)
        self.assertAlmostEqual(msg['origin'][1], 0.85, delta=0.11)

    def test_legend_one_shape_and_age_colours(self):
        rec = self.recorder()
        rec.step(T0, (1.0, 1.0, 0.0), True)
        msg = wl.coverage_layer(rec.coverage, 'a', T0 + 10 * 86400)
        self.assertEqual(sorted(msg['legend']), ['stops', 'title'])
        self.assertEqual(sorted(msg['legend']['stops'][0]), ['color', 'label'])
        img = decode_png(base64.b64decode(msg['png_b64']))
        painted = img[img[:, :, 3] > 0][0]
        self.assertEqual('#%02x%02x%02x' % tuple(painted[:3]), '#f97316')

    def test_empty_layers_are_published_at_start(self):
        rec = self.recorder()
        rec.maintain(T0)
        self.assertEqual(sorted(p[0] for p in self.pubs), ['coverage', 'gnss', 'slope'])
        self.assertTrue(all(p[1]['w'] == 0 and p[1]['png_b64'] == '' for p in self.pubs))

    def test_rate_limit_and_change_only(self):
        rec = self.recorder()
        rec.maintain(T0)
        self.pubs[:] = []
        rec.maintain(T0 + 60)
        self.assertEqual(self.pubs, [])
        rec.step(T0 + 61, (1.0, 1.0, 0.0), True)
        rec.maintain(T0 + 62)
        self.assertEqual(len(self.pubs), 1)
        rec.step(T0 + 63, (1.2, 1.0, 0.0), True)
        rec.maintain(T0 + 70)
        self.assertEqual(len(self.pubs), 1)
        rec.maintain(T0 + 93)
        self.assertEqual(len(self.pubs), 2)

    def test_idle_writes_and_publishes_nothing(self):
        rec = self.recorder()
        rec.maintain(T0)
        self.pubs[:] = []
        before = sorted(os.listdir(bundle.site_dir('a')))
        for i in range(1, 400):
            t = T0 + i
            rec.poll(t)
            rec.step(t, None, False)
            rec.maintain(t)
        self.assertEqual(self.pubs, [])
        self.assertEqual(sorted(os.listdir(bundle.site_dir('a'))), before)

    def test_image_is_capped(self):
        g = wl.CoverageGrid()
        g.stamp(0.0, 0.0, 200.0, 0.0, 0.2, T0)
        g.stamp(0.0, 0.0, 0.0, 150.0, 0.2, T0)
        msg = wl.coverage_layer(g, 'a', T0)
        self.assertLessEqual(max(msg['w'], msg['h']), wl.MAX_PNG)
        self.assertAlmostEqual(msg['res'], 0.2, 5)

    def test_coverage_recolours_as_it_ages(self):
        rec = self.recorder()
        rec.maintain(T0)
        rec.step(T0 + 1, (1.0, 1.0, 0.0), True)
        rec.maintain(T0 + 31)
        self.pubs[:] = []
        rec.maintain(T0 + 31 + 3601)
        self.assertEqual([p[0] for p in self.pubs], ['coverage'])


class Slope(Base):
    def dem(self, tilt=0.1, holes=False, noise=0.0):
        nx, ny = 200, 100
        res = 0.05
        ix = np.arange(nx)[:, np.newaxis]
        elev = (tilt * ix * res) * np.ones((nx, ny), np.float32)
        elev = elev + noise * np.random.RandomState(1).standard_normal((nx, ny))
        if holes:
            elev[:, 50:] = np.nan
        p = self.file('a', 'dem.npz')
        np.savez(p, res=res, ox=-2.0, oy=-1.0, elev=elev.astype(np.float32),
                 weight=np.ones((nx, ny), np.float32),
                 last_seen=np.zeros((nx, ny)), source=np.ones((nx, ny), np.uint8))
        return p

    def test_plane(self):
        slope, ox, oy, cell = wl.slope_from_dem(self.dem(), 0.5)
        self.assertAlmostEqual(cell, 0.5)
        self.assertEqual((ox, oy), (-2.0, -1.0))
        self.assertTrue(np.allclose(slope, math.degrees(math.atan(0.1)), atol=0.01))
        self.assertEqual(slope.shape, (10, 20))                # [row = y, col = x]

    def test_noise_does_not_read_as_a_hill(self):
        slope, *_ = wl.slope_from_dem(self.dem(tilt=0.0, noise=0.01), 0.5)
        self.assertLess(float(np.nanmax(slope)), 2.5)

    def test_holes_stay_holes_and_edges_survive(self):
        slope, *_ = wl.slope_from_dem(self.dem(holes=True), 0.5)
        self.assertTrue(np.isnan(slope[6:, :]).all())
        self.assertTrue(np.isfinite(slope[:4, :]).all())

    def test_layer_and_dem_change_trigger(self):
        p = self.dem()
        rec = self.recorder()
        rec.maintain(T0)
        msg = [m for name, m in self.pubs if name == 'slope'][0]
        self.assertEqual((msg['w'], msg['h']), (20, 10))
        self.assertEqual(msg['res'], 0.5)
        self.pubs[:] = []
        rec.poll(T0 + 10)
        rec.maintain(T0 + 40)
        self.assertEqual(self.pubs, [])
        os.utime(p, (T0 + 50, T0 + 50))
        rec.poll(T0 + 60)
        rec.maintain(T0 + 60)
        self.assertEqual([n for n, _m in self.pubs], ['slope'])

    def test_damaged_dem_is_no_data(self):
        with open(self.file('a', 'dem.npz'), 'wb') as f:
            f.write(b'junk')
        rec = self.recorder()
        rec.maintain(T0)
        msg = [m for name, m in self.pubs if name == 'slope'][0]
        self.assertEqual(msg['w'], 0)


class Query(Base):
    def test_points_and_polygon(self):
        rec = self.recorder()
        for i in range(11):
            rec.step(T0 + i, (i * 0.1, 0.0, 0.0), True, (20, 4))
            rec._gnss_at = None
        rec.close()
        got = wl.layer_query('a', 'coverage', points=[[0.5, 0.0], [9.0, 9.0], [-99, 0]])
        self.assertTrue(got['ok'])
        self.assertIsNotNone(got['result'][0]['last_mowed'])
        self.assertIsNone(got['result'][1]['last_mowed'])
        self.assertIsNone(got['result'][2]['last_mowed'])
        box = [[0, -0.1], [1, -0.1], [1, 0.1], [0, 0.1]]
        stats = wl.layer_query('a', 'coverage', polygon=box)['result']
        self.assertAlmostEqual(stats['area_m2'], 0.2, delta=0.03)
        self.assertEqual(stats['mowed_frac'], 1.0)
        half = [[0, -0.5], [1, -0.5], [1, 0.5], [0, 0.5]]
        self.assertLess(wl.layer_query('a', 'coverage', polygon=half)['result']['mowed_frac'], 0.5)
        gn = wl.layer_query('a', 'gnss', points=[[0.1, 0.1], [40, 40]])['result']
        self.assertEqual(gn[0]['class_median'], 4)
        self.assertIsNone(gn[1]['class_median'])
        wide = [[0, -0.6], [1, -0.6], [1, 0.6], [0, 0.6]]
        stats = wl.layer_query('a', 'gnss', polygon=wide)['result']
        self.assertEqual(stats['class_frac']['4'], 1.0)

    def test_no_data_is_not_a_failure(self):
        got = wl.layer_query('a', 'coverage', points=[[0, 0]])
        self.assertTrue(got['ok'])
        self.assertIsNone(got['result'])
        self.assertTrue(wl.layer_query('a', 'slope', points=[[0, 0]])['ok'])

    def test_refusals_never_raise(self):
        self.assertFalse(wl.layer_query('a', 'coverage')['ok'])
        self.assertFalse(wl.layer_query('a', 'coverage', points=[[0, 0]], polygon=[[0, 0]])['ok'])
        self.assertFalse(wl.layer_query('a', 'bogus', points=[[0, 0]])['ok'])
        self.assertFalse(wl.layer_query('nosite', 'gnss', points=[[0, 0]])['ok'])
        self.assertFalse(wl.layer_query('a', 'coverage', points=[['x', 0]])['ok'])
        self.assertFalse(wl.layer_query('a', 'coverage', polygon=[[0, 0], [1, 1]])['ok'])

    def test_reader_leaves_a_damaged_file_alone(self):
        p = self.file('a', 'coverage.npz')
        with open(p, 'wb') as f:
            f.write(b'junk')
        wl.layer_query('a', 'coverage', points=[[0, 0]])
        self.assertTrue(os.path.exists(p))

    def test_slope_points(self):
        np.savez(self.file('a', 'dem.npz'), res=0.05, ox=0.0, oy=0.0,
                 elev=np.outer(np.arange(100) * 0.005, np.ones(100)).astype(np.float32))
        got = wl.layer_query('a', 'slope', points=[[1.0, 1.0], [50, 50]])['result']
        self.assertAlmostEqual(got[0]['slope_deg'], 5.71, 1)
        self.assertIsNone(got[1]['slope_deg'])


if __name__ == '__main__':
    unittest.main()
