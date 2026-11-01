#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.profiles (mapping profiles 2026-11-01).

Run: python3 -m pytest test/test_profiles.py   (from the package root)
"""

import json
import os
import sys
import tempfile
import unittest

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from vitulus_mapping import presets, profiles as P  # noqa: E402

# what vitulus_ui/nodes/sensor limits and a site of the owner really hold
REAL_OVERRIDES = {
    'cam_cluster_radius_m': 0.2, 'cam_min_cluster_pts': 4,
    'cam_obstacle_max_h': 0.8, 'cam_obstacle_min_h': 0.25, 'free_thresh': -0.4,
    'hi_clamp': 3.5, 'hit_inc': 0.5, 'lidar_free_dec': 0.08,
    'lidar_free_no_hit': True, 'lo_clamp': -2.0, 'min_hits': 4, 'miss_dec': 0.3,
    'occ_erode_factor': 0.3, 'occ_thresh': 0.8, 'use_cam_free': True,
    'use_cam_obstacle': True, 'use_lidar': True}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.wide = os.path.join(self.d, 'store', 'settings.json')
        self.s = P.Store(self.d, wide_store=self.wide)

    def tearDown(self):
        self.tmp.cleanup()

    def site(self, name, overrides=None):
        os.makedirs(os.path.join(self.d, name))
        if overrides is not None:
            P.write_json(os.path.join(self.d, name, P.OVERRIDES_FILE), overrides)

    def st(self, active=P.DEFAULT_PROFILE):
        return {'active': active, 'camera': {}, 'mapper': {}}


class Defaults(Base):
    def test_mapper_defaults_are_the_yaml(self):
        with open(os.path.join(ROOT, 'config', 'mapping_v3.yaml')) as f:
            y = yaml.safe_load(f)['direct_raster']
        for k, v in P.MAPPER_DEFAULTS.items():
            self.assertEqual(y[k], v, k)
        for gone in ('cam_obstacle_min_h', 'cam_obstacle_max_h', 'obstacle_max_range_m',
                     'free_max_range_m', 'cam_min_cluster_pts', 'cam_cluster_radius_m'):
            self.assertNotIn(gone, y)

    def test_camera_defaults_are_the_chain(self):
        path = os.path.join(ROOT, '..', '..', 'vitulus_depth_obstacles', 'config', 'd435_ros1.yaml')
        for up in (path, os.path.join(ROOT, '..', '..', '..', '..', '..', 'vitulus_depth_obstacles', 'config', 'd435_ros1.yaml')):
            if os.path.exists(up):
                with open(up) as f:
                    w = yaml.safe_load(f)['wide']
                for k, v in P.CAMERA_DEFAULTS.items():
                    if k in w:
                        self.assertEqual(w[k], v, k)
                return
        self.skipTest('vitulus_depth_obstacles not next to this repo')

    def test_every_builtin_validates_and_has_a_window(self):
        self.assertEqual(set(P.BUILTIN_PROFILES), set(presets.BUILTIN_PRESETS))
        for name in P.BUILTIN_PROFILES:
            p = self.s.profile(name)
            self.assertEqual(P.validate_mapper(p['mapper'])[1], None, name)
            c = dict(P.CAMERA_DEFAULTS, **p['camera'])
            self.assertLess(max(c['max_ground_height'], c['min_obstacle_height']), c['max_obstacle_height'], name)
            self.assertEqual(c['range_max'], 2.8, name)   # the old default reach stays
            self.assertFalse(set(p['camera']) & {'lidar_min_intensity'})

    def test_validate_refuses(self):
        for bad in ({'lidar_max_range_m': 0.1}, {'min_hits': 0}, {'lo_clamp': 1.0},
                    {'occ_thresh': -1.0}, {'hit_inc': float('nan')}, {'hit_inc': 'x'}):
            self.assertIsNotNone(P.validate_mapper(dict(P.MAPPER_DEFAULTS, **bad))[1], bad)


class Legacy(Base):
    def test_intersection_of_the_two_filters(self):
        # chain 0.15..2.0 and filter 0.12..0.6 let 0.15..0.6 through
        c = P.from_legacy({'cam_obstacle_min_h': 0.12, 'cam_obstacle_max_h': 0.6})['camera']
        self.assertEqual(c, {'max_obstacle_height': 0.6})
        c = P.from_legacy({'cam_obstacle_max_h': 2.5})['camera']   # the chain stops at 2.0
        self.assertEqual(c, {})
        c = P.from_legacy({'cam_min_cluster_pts': 1, 'cam_cluster_radius_m': 0.5})['camera']
        self.assertEqual(c, {})
        c = P.from_legacy({'cam_min_cluster_pts': 4, 'cam_cluster_radius_m': 0.3})['camera']
        self.assertEqual(c, {'min_cluster_size': 4, 'cluster_radius': 0.3})

    def test_reach_is_the_larger_of_the_sources_in_use(self):
        c = P.from_legacy({'obstacle_max_range_m': 2.0, 'free_max_range_m': 3.0})['camera']
        self.assertEqual(c['range_max'], 3.0)
        c = P.from_legacy({'obstacle_max_range_m': 2.0, 'free_max_range_m': 3.0, 'use_cam_free': False})['camera']
        self.assertEqual(c['range_max'], 2.0)
        self.assertEqual(P.from_legacy({'obstacle_max_range_m': 9.0})['camera'], {})   # chain: 4 m

    def test_store_values_stay_and_narrow(self):
        c = P.from_legacy({'cam_obstacle_max_h': 1.0}, {'max_obstacle_height': 0.7, 'voxel_size': 0.1})['camera']
        self.assertEqual(c, {'max_obstacle_height': 0.7, 'voxel_size': 0.1})

    def test_height_window_never_closes(self):
        c = P.from_legacy({'cam_obstacle_min_h': 0.6, 'cam_obstacle_max_h': 0.3})['camera']
        self.assertGreater(c['max_obstacle_height'], c['min_obstacle_height'])


class Migration(Base):
    def test_real_site_keeps_its_effective_values(self):
        self.site('10_2026', REAL_OVERRIDES)
        self.assertEqual(self.s.migrate_all(), ['10_2026'])
        st = self.s.load_state('10_2026')
        self.assertEqual(st['active'], 'Lawn tall')
        self.assertEqual(st['camera'], {})
        # the file had no intensity: 0 until now, the profile says 6
        self.assertEqual(st['mapper'], {'lidar_min_intensity': 0.0})
        e = self.s.effective(st)
        for k, v in REAL_OVERRIDES.items():
            if k in P.MAPPER_KEYS:
                self.assertEqual(e['mapper'][k], v, k)
        self.assertEqual(e['camera']['min_obstacle_height'], 0.25)
        self.assertEqual(json.load(open(os.path.join(self.d, '10_2026', P.OVERRIDES_FILE + P.PRE_SUFFIX))), REAL_OVERRIDES)

    def test_site_without_overrides_is_the_default(self):
        self.site('fresh')
        self.s.migrate_all()
        self.assertEqual(self.s.load_state('fresh'), self.st())

    def test_no_site_at_all_works(self):
        self.assertEqual(self.s.migrate_all(), [])
        self.assertIsNone(self.s.load_state(None))

    def test_unknown_values_become_changes(self):
        self.site('x', dict(REAL_OVERRIDES, min_hits=11, obstacle_max_range_m=1.5, free_max_range_m=1.5))
        self.s.migrate_all()
        st = self.s.load_state('x')
        self.assertEqual(st['mapper']['min_hits'], 11)
        self.assertEqual(st['camera'], {'range_max': 1.5})

    def test_custom_presets_become_profiles(self):
        P.write_json(os.path.join(self.d, 'direct_presets.json'),
                     {'Mine': {'cam_obstacle_min_h': 0.3, 'cam_obstacle_max_h': 0.9, 'hit_inc': 0.7, 'min_hits': 6}})
        self.site('a', dict(P.LEGACY_DEFAULTS, cam_obstacle_min_h=0.3, cam_obstacle_max_h=0.9, hit_inc=0.7, min_hits=6))
        self.s.migrate_all()
        self.assertFalse(os.path.exists(os.path.join(self.d, 'direct_presets.json')))
        self.assertTrue(os.path.exists(os.path.join(self.d, 'direct_presets.json' + P.PRE_SUFFIX)))
        p = self.s.profile('Mine')
        self.assertEqual(p['camera']['min_obstacle_height'], 0.3)
        self.assertEqual(p['mapper']['min_hits'], 6)
        self.assertEqual(self.s.load_state('a'), self.st('Mine'))
        again = P.Store(self.d, wide_store=self.wide)
        self.assertIn('Mine', again.names())

    def test_broken_presets_file_stays(self):
        with open(os.path.join(self.d, 'direct_presets.json'), 'w') as f:
            f.write('{broken')
        self.s.migrate_all()
        self.assertTrue(os.path.exists(os.path.join(self.d, 'direct_presets.json')))

    def test_global_wide_store_goes_into_the_site(self):
        P.write_json(self.wide, {'version': 1, 'profiles': {'wide': {'voxel_size': 0.1}, 'nav': {'max_depth': 3.05}}})
        self.site('a', {})
        self.s.migrate_all()
        st = self.s.load_state('a')
        self.assertEqual(st['camera'], {'voxel_size': 0.1})     # nav is not ours

    def test_idempotent(self):
        self.site('a', REAL_OVERRIDES)
        self.s.migrate_all()
        st = self.s.load_state('a')
        st['mapper']['min_hits'] = 9
        self.s.save_state('a', st)
        self.assertEqual(self.s.migrate_all(), [])
        self.assertEqual(self.s.load_state('a')['mapper']['min_hits'], 9)

    def test_pending_is_taken_once(self):
        path = os.path.join(self.d, P.PENDING_FILE)
        P.write_json(path, {'lidar_max_range_m': 6.0, 'use_lidar': False})
        st = self.st()
        self.assertTrue(self.s.take_pending(st))
        self.assertEqual(st['mapper'], {'lidar_max_range_m': 6.0, 'use_lidar': False})
        self.assertFalse(os.path.exists(path))
        self.assertFalse(self.s.take_pending(st))

    def test_bad_pending_stays(self):
        path = os.path.join(self.d, P.PENDING_FILE)
        P.write_json(path, {'lidar_max_range_m': 99.0})
        self.assertFalse(self.s.take_pending(self.st()))
        self.assertTrue(os.path.exists(path))


class Changes(Base):
    def test_override_and_back(self):
        st = self.st('Lawn tall')
        self.assertEqual(self.s.set_mapper(st, {'min_hits': 7}), (True, None))
        self.assertEqual(st['mapper'], {'min_hits': 7})
        self.s.set_mapper(st, {'min_hits': 4})         # the profile's value
        self.assertEqual(st['mapper'], {})

    def test_refused_changes_nothing(self):
        st = self.st()
        ok, err = self.s.set_mapper(st, {'min_hits': 0})
        self.assertFalse(ok)
        self.assertIn('min_hits', err)
        self.assertEqual(st['mapper'], {})
        self.assertFalse(self.s.set_mapper(st, {'cam_obstacle_min_h': 0.2})[0])

    def test_camera_against_profile_and_default(self):
        st = self.st('Lawn tall')
        self.s.set_camera(st, {'range_max': 3.5, 'voxel_size': 0.07})
        self.assertEqual(st['camera'], {'range_max': 3.5, 'voxel_size': 0.07})
        self.s.set_camera(st, {'range_max': 2.8})
        self.assertEqual(st['camera'], {'voxel_size': 0.07})
        self.s.camera_defaults['voxel_size'] = 0.05
        self.s.set_camera(st, {'voxel_size': 0.05})
        self.assertEqual(st['camera'], {})

    def test_discard_all_and_keys(self):
        st = self.st()
        st['camera'] = {'range_max': 3.0, 'voxel_size': 0.1}
        st['mapper'] = {'min_hits': 9}
        self.s.discard(st, ['range_max'], [])
        self.assertEqual(st['camera'], {'voxel_size': 0.1})
        self.assertEqual(st['mapper'], {'min_hits': 9})
        self.s.discard(st)
        self.assertEqual((st['camera'], st['mapper']), ({}, {}))

    def test_save_as_then_it_is_active_and_clean(self):
        st = self.st('Lawn tall')
        self.s.set_mapper(st, {'min_hits': 7})
        self.s.set_camera(st, {'range_max': 3.5})
        self.assertEqual(self.s.save_profile(st, ' Zahrada '), (True, None))
        self.assertEqual(st, self.st('Zahrada'))
        p = self.s.effective(st)
        self.assertEqual((p['mapper']['min_hits'], p['camera']['range_max']), (7, 3.5))
        self.assertEqual(P.Store(self.d, self.wide).profile('Zahrada')['camera']['range_max'], 3.5)

    def test_save_refuses_bad_names(self):
        for bad in ('x' * 41, 'a\nb'):
            self.assertFalse(self.s.save_profile(self.st(), bad)[0])

    def test_builtin_shadow_and_reset(self):
        st = self.st('Lawn tall')
        self.s.set_mapper(st, {'min_hits': 7})
        self.s.save_profile(st, None)
        self.assertEqual(self.s.profile('Lawn tall')['mapper']['min_hits'], 7)
        self.assertEqual([p for p in self.s.listing() if p['name'] == 'Lawn tall'],
                         [{'name': 'Lawn tall', 'builtin': True, 'custom': True}])
        self.assertTrue(self.s.delete_profile('Lawn tall', keep=st)[0])
        self.assertEqual(self.s.profile('Lawn tall')['mapper']['min_hits'], 4)
        self.assertEqual(st['active'], 'Lawn tall')

    def test_delete_keeps_the_values_of_every_site(self):
        self.site('a')
        self.site('b')
        st = self.st()
        self.s.set_mapper(st, {'min_hits': 7})
        self.s.save_profile(st, 'Mine')
        self.s.save_state('a', self.st('Mine'))
        self.s.save_state('b', self.st('Lawn tall'))
        self.s.save_state(None, st)
        self.assertTrue(self.s.delete_profile('Mine', keep=st)[0])
        for x in (st, self.s.load_state('a'), self.s.load_state(None)):
            self.assertEqual(x['active'], P.DEFAULT_PROFILE)
            self.assertEqual(x['mapper'], {'min_hits': 7})
        self.assertEqual(self.s.load_state('b')['active'], 'Lawn tall')

    def test_delete_refuses_builtin(self):
        self.assertFalse(self.s.delete_profile('Lawn tall')[0])

    def test_unknown_active_falls_back(self):
        self.assertEqual(self.s.profile('gone'), self.s.profile(P.DEFAULT_PROFILE))

    def test_overrides_file_is_for_direct_raster(self):
        self.site('a')
        st = self.st('Lawn tall')
        self.s.write_overrides('a', st)
        doc = json.load(open(os.path.join(self.d, 'a', P.OVERRIDES_FILE)))
        self.assertEqual(doc['_profile']['name'], 'Lawn tall')
        self.assertEqual(doc['min_hits'], 4)
        self.assertNotIn('cam_obstacle_min_h', doc)


if __name__ == '__main__':
    unittest.main()
