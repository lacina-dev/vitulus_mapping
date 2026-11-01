#!/usr/bin/env python3
"""DOCK, UNDOCK, DOCKED: uppercase everywhere, and what a stop in the dock saves
(owner 2026-11-01). No ROS master.

Run: python3 test/test_dock_points.py
"""
import importlib.machinery
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:1'     # never the robot's master

from vitulus_mapping import bundle, dockpoints, objnames  # noqa: E402

DOCK_CONFIG = os.path.join(ROOT, '..', '..', '..', '..', 'vitulus_dock', '.claude', 'worktrees',
                           'map-editor', 'config', 'docking_controller.yaml')
if not os.path.exists(DOCK_CONFIG):
    DOCK_CONFIG = os.path.join(ROOT, '..', 'vitulus_dock', 'config', 'docking_controller.yaml')

# ~/.vitulus/mapping_v3/10_2026/waypoints.geojson, 2026-11-01
LIVE_DOCKED = (493076.1831607195, 5540714.79979186, -0.27565775841350854)
LIVE_DOCK = (493077.3303992618, 5540714.52691138, -0.2543647415391783)


def feature(name, e=1.0, n=2.0, yaw=0.5):
    return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [e, n, 3.0]},
            'properties': {'name': name, 'yaw_rad': yaw, 'z': 3.0, 'extra': 'kept'}}


class Names(unittest.TestCase):
    def test_canonical(self):
        for raw, want in (('docked', 'DOCKED'), ('Docked', 'DOCKED'), (' DOCKED ', 'DOCKED'),
                          ('dock', 'DOCK'), ('undock', 'UNDOCK'), ('UnDock', 'UNDOCK'),
                          ('dock1', 'dock1'), ('Bazén', 'Bazén'), (None, None)):
            self.assertEqual(objnames.canonical_name(raw), want)

    def test_reserved_is_uppercase_and_folds(self):
        self.assertEqual(objnames.RESERVED, ('DOCK', 'UNDOCK', 'DOCKED'))
        for n in ('docked', 'DOCKED', 'Undock'):
            self.assertTrue(objnames.is_dock_point(n))
            self.assertEqual(objnames.check_name(n, []), 'reserved')
        self.assertFalse(objnames.is_dock_point('TEST'))
        self.assertFalse(objnames.is_dock_point(None))

    def test_normalize(self):
        norm = objnames.normalize_dock_names
        self.assertEqual(norm(['docked', 'DOCK']), [(0, 'DOCKED'), (1, 'DOCK')])
        self.assertEqual(norm(['docked', 'DOCKED', 'A']), [(1, 'DOCKED'), (2, 'A')])
        self.assertEqual(norm(['DOCKED', 'docked']), [(0, 'DOCKED')])
        self.assertEqual(norm(['docked', 'Docked']), [(0, 'DOCKED')])
        self.assertEqual(norm(['A', 'A']), [(0, 'A'), (1, 'A')])     # not ours to judge
        self.assertEqual(norm([]), [])


class Site(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='dock_points_')
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.root
        os.makedirs(bundle.site_dir('S'))

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, names):
        with open(bundle.waypoints_path('S'), 'w') as f:
            json.dump({'type': 'FeatureCollection', 'crs': {'keep': 1},
                       'features': [feature(*n) if isinstance(n, tuple) else feature(n)
                                    for n in names]}, f)

    def raw(self):
        with open(bundle.waypoints_path('S')) as f:
            return json.load(f)

    def meta_path(self):
        return os.path.join(bundle.site_dir('S'), 'objects_meta.json')

    def test_reader_is_canonical_before_any_migration(self):
        self.write(['docked', 'TEST'])
        self.assertEqual([w['name'] for w in bundle.load_waypoints('S')], ['DOCKED', 'TEST'])
        self.assertEqual(self.raw()['features'][0]['properties']['name'], 'docked')

    def test_migration_keeps_the_point_and_everything_else(self):
        self.write([('docked', 7.0, 8.0, -0.27), 'DOCK', 'UNDOCK'])
        self.assertTrue(bundle.migrate_dock_names('S'))
        fc = self.raw()
        self.assertEqual(fc['crs'], {'keep': 1})
        self.assertEqual([f['properties']['name'] for f in fc['features']],
                         ['DOCKED', 'DOCK', 'UNDOCK'])
        first = fc['features'][0]
        self.assertEqual(first['geometry']['coordinates'], [7.0, 8.0, 3.0])
        self.assertEqual(first['properties']['yaw_rad'], -0.27)
        self.assertEqual(first['properties']['extra'], 'kept')

    def test_migration_is_idempotent(self):
        self.write(['docked'])
        self.assertTrue(bundle.migrate_dock_names('S'))
        before = os.stat(bundle.waypoints_path('S')).st_mtime_ns
        self.assertFalse(bundle.migrate_dock_names('S'))
        self.assertEqual(os.stat(bundle.waypoints_path('S')).st_mtime_ns, before)

    def test_both_cases_become_one_and_the_exact_one_stays(self):
        self.write([('docked', 1.0), ('DOCKED', 9.0)])
        self.assertTrue(bundle.migrate_dock_names('S'))
        wps = bundle.load_waypoints('S')
        self.assertEqual([(w['name'], w['e']) for w in wps], [('DOCKED', 9.0)])

    def test_missing_files_are_fine(self):
        self.assertFalse(bundle.migrate_dock_names('S'))
        self.assertFalse(os.path.exists(bundle.waypoints_path('S')))

    def test_sidecar_keys(self):
        self.write(['DOCKED'])
        with open(self.meta_path(), 'w') as f:
            json.dump({'waypoint:docked': {'id': 'o_1'}, 'waypoint:TEST': {'id': 'o_2'},
                       'zone:docked': {'id': 'o_3'}}, f)
        self.assertTrue(bundle.migrate_dock_names('S'))
        with open(self.meta_path()) as f:
            self.assertEqual(json.load(f), {'waypoint:DOCKED': {'id': 'o_1'},
                                            'waypoint:TEST': {'id': 'o_2'},
                                            'zone:docked': {'id': 'o_3'}})
        self.assertFalse(bundle.migrate_dock_names('S'))

    def test_sidecar_never_overwrites_the_canonical_key(self):
        with open(self.meta_path(), 'w') as f:
            json.dump({'waypoint:docked': {'id': 'old'}, 'waypoint:DOCKED': {'id': 'new'}}, f)
        self.assertTrue(bundle.migrate_dock_names('S'))
        with open(self.meta_path()) as f:
            self.assertEqual(json.load(f), {'waypoint:DOCKED': {'id': 'new'}})

    def test_a_writer_cannot_bring_the_old_name_back(self):
        bundle.save_waypoints('S', [{'name': 'docked', 'e': 1.0, 'n': 2.0, 'yaw_rad': 0.0}])
        self.assertEqual(self.raw()['features'][0]['properties']['name'], 'DOCKED')

    def test_virgin_site_no_manifest_no_file(self):
        self.assertEqual(bundle.load_waypoints('S'), [])
        points = dockpoints.points_to_save([], 1.0, 2.0, 0.0, 1.02)
        self.assertEqual([p['name'] for p in points], ['DOCKED', 'UNDOCK', 'DOCK'])
        bundle.save_waypoints('S', [{'name': p['name'], 'e': p['x'], 'n': p['y'],
                                     'yaw_rad': p['yaw']} for p in points])
        self.assertEqual(sorted(w['name'] for w in bundle.load_waypoints('S')),
                         ['DOCK', 'DOCKED', 'UNDOCK'])


class Geometry(unittest.TestCase):
    def test_drive_out_is_what_the_controller_drives(self):
        with open(DOCK_CONFIG) as f:
            self.assertIn('undock_distance', f.read())
        self.assertAlmostEqual(dockpoints.drive_out_m_from_yaml(DOCK_CONFIG), 1.02, places=6)
        params = {dockpoints.PARAM_NS + 'undock_distance': 1.5,
                  dockpoints.PARAM_NS + 'charge_point_distance': 0.25}
        self.assertAlmostEqual(
            dockpoints.drive_out_m_from_params(lambda n, d=None: params.get(n, d)), 1.25)

    def test_unknown_or_nonsense_distance(self):
        self.assertIsNone(dockpoints.drive_out_m_from_params(lambda n, d=None: d))
        self.assertIsNone(dockpoints.drive_out_m(0.2, 0.23))
        self.assertIsNone(dockpoints.drive_out_m_from_yaml('/nonexistent.yaml'))
        self.assertIsNone(dockpoints.drive_out_m_from_yaml(None))

    def test_ahead_of_docked_facing_away(self):
        points = {p['name']: p for p in dockpoints.points_to_save([], 10.0, 20.0, math.pi / 2, 1.02)}
        self.assertEqual((points['DOCKED']['x'], points['DOCKED']['y']), (10.0, 20.0))
        for name in ('DOCK', 'UNDOCK'):
            self.assertAlmostEqual(points[name]['x'], 10.0)
            self.assertAlmostEqual(points[name]['y'], 21.02)
            self.assertEqual(points[name]['yaw'], math.pi / 2)

    def test_docked_heading_points_out_on_the_live_site(self):
        x, y, yaw = LIVE_DOCKED
        bearing = math.atan2(LIVE_DOCK[1] - y, LIVE_DOCK[0] - x)
        self.assertLess(abs(bearing - yaw), 0.05)
        self.assertLess(abs(LIVE_DOCK[2] - yaw), 0.05)
        made = {p['name']: p for p in dockpoints.points_to_save([], x, y, yaw, 1.02)}
        self.assertLess(math.hypot(made['DOCK']['x'] - LIVE_DOCK[0],
                                   made['DOCK']['y'] - LIVE_DOCK[1]), 0.2)

    def test_existing_points_stay(self):
        names = lambda ex: [p['name'] for p in dockpoints.points_to_save(ex, 0, 0, 0, 1.02)]
        self.assertEqual(names(['DOCK', 'UNDOCK', 'DOCKED']), ['DOCKED'])
        self.assertEqual(names(['DOCK']), ['DOCKED', 'UNDOCK'])
        self.assertEqual(names(['UNDOCK']), ['DOCKED', 'DOCK'])
        self.assertEqual(names(['undock', 'dock']), ['DOCKED'])       # old case too
        self.assertEqual(names(['TEST']), ['DOCKED', 'UNDOCK', 'DOCK'])

    def test_not_knowing_saves_docked_only(self):
        names = lambda ex, out: [p['name'] for p in dockpoints.points_to_save(ex, 0, 0, 0, out)]
        self.assertEqual(names(None, 1.02), ['DOCKED'])
        self.assertEqual(names([], None), ['DOCKED'])


class Manager(unittest.TestCase):
    """The method of the real mapping_manager, on a fake node."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(ROOT, 'nodes', 'mapping_manager')
        spec = importlib.util.spec_from_loader(
            'mapping_manager_node', importlib.machinery.SourceFileLoader('mapping_manager_node', path))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='dock_points_')
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.root
        self.real_get_param = self.mod.rospy.get_param
        self.mod.rospy.get_param = lambda name, default=None: default

    def tearDown(self):
        self.mod.rospy.get_param = self.real_get_param
        bundle.SITES_ROOT = self.old_root
        shutil.rmtree(self.root, ignore_errors=True)

    def save(self, config=DOCK_CONFIG):
        node = types.SimpleNamespace(_dock_config_path=lambda: config)
        cls = next(c for c in vars(self.mod).values()
                   if isinstance(c, type) and hasattr(c, '_dock_points_to_save'))
        return cls._dock_points_to_save(node, 'S', {'x': 2.0, 'y': 3.0, 'yaw_deg': 0.0})

    def test_virgin_site_gets_all_three_from_the_config_file(self):
        points = self.save()
        self.assertEqual([p['name'] for p in points], ['DOCKED', 'UNDOCK', 'DOCK'])
        self.assertAlmostEqual(points[1]['x'], 3.02)

    def test_keeps_what_the_owner_placed(self):
        bundle.save_waypoints('S', [{'name': 'UNDOCK', 'e': 1.0, 'n': 1.0, 'yaw_rad': 0.0}], 33)
        self.assertEqual([p['name'] for p in self.save()], ['DOCKED', 'DOCK'])

    def test_no_distance_known(self):
        self.assertEqual([p['name'] for p in self.save(config=None)], ['DOCKED'])

    def test_unreadable_file_overwrites_nothing(self):
        os.makedirs(bundle.site_dir('S'))
        with open(bundle.waypoints_path('S'), 'w') as f:
            f.write('{broken')
        self.assertEqual([p['name'] for p in self.save()], ['DOCKED'])


if __name__ == '__main__':
    unittest.main()
