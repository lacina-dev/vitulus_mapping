#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.objects.

Run: python3 -m pytest test/test_objects.py   (from the package root)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yaml  # noqa: E402

from vitulus_mapping import bundle, objects  # noqa: E402

DATUM = {'utm_e': 500000.0, 'utm_n': 5500000.0, 'yaw_rad': 0.1, 'utm_zone': 33}
SQUARE = [[0, 0], [2, 0], [2, 2], [0, 2]]


def place(name, obj_type, points, **kw):
    o = objects._blank('place', obj_type, name, points)
    o['id'] = kw.pop('id', 'o_' + name)
    o.update(kw)
    return o


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        self.site = 's'
        os.makedirs(bundle.site_dir(self.site))

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        self.tmp.cleanup()

    def datum(self):
        with open(os.path.join(bundle.site_dir(self.site), 'datum.yaml'), 'w') as f:
            yaml.safe_dump(DATUM, f)

    def utm(self, pts):
        return [list(objects.geo.map_to_utm(x, y, DATUM)) for x, y in pts]

    def waypoint(self, name, x, y):
        e, n = objects.geo.map_to_utm(x, y, DATUM)
        wps = bundle.load_waypoints(self.site)
        wps.append({'name': name, 'e': e, 'n': n, 'yaw_rad': 0.2})
        bundle.save_waypoints(self.site, wps, utm_zone=33)


class Loading(Base):
    def test_site_without_files(self):
        self.assertEqual(objects.load_all(self.site), [])
        self.assertEqual(objects.active_rules(self.site), [])
        fams = objects.load_families(self.site)
        self.assertTrue(all(v is None for v in fams.values()))

    def test_empty_file_differs_from_missing_file(self):
        bundle.save_waypoints(self.site, [], utm_zone=33)
        fams = objects.load_families(self.site)
        self.assertEqual(fams['waypoint'], [])
        self.assertIsNone(fams['zone'])

    def test_broken_file_is_no_information(self):
        with open(bundle.zones_path(self.site), 'w') as f:
            f.write('{nope')
        self.assertIsNone(objects.load_families(self.site)['zone'])

    def test_map_frame_needs_datum(self):
        self.waypoint('A', 1.0, 2.0)
        self.assertEqual(objects.load_all(self.site), [])
        self.assertEqual(len(objects.load_all(self.site, frame='utm')), 1)

    def test_frames(self):
        self.datum()
        self.waypoint('A', 1.0, 2.0)
        o = objects.load_all(self.site)[0]
        self.assertAlmostEqual(o['points'][0][0], 1.0)
        self.assertAlmostEqual(o['points'][0][1], 2.0)
        self.assertAlmostEqual(o['yaw'], 0.2 - 0.1)
        self.assertEqual((o['family'], o['type'], o['geometry']),
                         ('waypoint', 'waypoint', 'point'))

    def test_object_has_every_contract_field(self):
        self.datum()
        self.waypoint('A', 1.0, 2.0)
        o = objects.load_all(self.site)[0]
        for k in ('id', 'family', 'type', 'name', 'geometry', 'points', 'yaw', 'aliases',
                  'note', 'tags', 'source', 'created', 'updated', 'valid_from',
                  'valid_until', 'active', 'protected', 'props'):
            self.assertIn(k, o)
        self.assertRegex(o['id'], r'^o_[0-9a-f]{8}$')

    def test_id_stable_without_sidecar(self):
        self.datum()
        self.waypoint('A', 1.0, 2.0)
        a = objects.load_all(self.site)[0]['id']
        self.assertEqual(a, objects.load_all(self.site)[0]['id'])

    def test_meta_is_merged(self):
        self.datum()
        self.waypoint('A', 1.0, 2.0)
        objects.save_meta(self.site, {'waypoint:A': {'id': 'o_12345678', 'note': 'x'}})
        o = objects.load_all(self.site)[0]
        self.assertEqual((o['id'], o['note']), ('o_12345678', 'x'))

    def test_protected_any_case(self):
        self.datum()
        bundle.save_waypoints(self.site, [
            {'name': n, 'e': DATUM['utm_e'], 'n': DATUM['utm_n'], 'yaw_rad': 0.0}
            for n in ('DOCK', 'UNDOCK', 'DOCKED', 'TEST')], utm_zone=33)
        got = {o['name']: o['protected'] for o in objects.load_all(self.site)}
        self.assertEqual(got, {'DOCK': True, 'UNDOCK': True, 'DOCKED': True, 'TEST': False})
        self.assertTrue(objects.is_protected('waypoint', 'docked'))
        self.assertFalse(objects.is_protected('zone', 'DOCK'))

    def test_places_round_trip(self):
        self.datum()
        p = place('Bazen', 'nogo', self.utm(SQUARE), tags=['voda'], note='n',
                  valid_until=100.0, props={})
        l = place('Strom', 'landmark', self.utm([[5, 5]]), props={'category': 'strom'})
        c = place('C', 'corridor', self.utm([[0, 0], [5, 0]]),
                  props={'width_m': 0.8, 'connects': ['A', 'B']})
        objects.save_places(self.site, [p, l, c], 33)
        got = objects.load_places(self.site)
        self.assertEqual([o['name'] for o in got], ['Bazen', 'Strom', 'C'])
        self.assertEqual(len(got[0]['points']), 4)         # ring not closed in memory
        self.assertEqual(got[0]['tags'], ['voda'])
        self.assertEqual(got[0]['valid_until'], 100.0)
        self.assertEqual(got[1]['geometry'], 'point')
        self.assertEqual(got[2]['props']['connects'], ['A', 'B'])
        with open(objects.places_path(self.site)) as f:
            fc = json.load(f)
        self.assertEqual(fc['features'][0]['geometry']['type'], 'Polygon')
        self.assertIn('crs', fc)

    def test_effective_passes_through(self):
        self.datum()
        eff = [{'program': 'P', 'cut_height': 45, 'rpm': 3300}]
        bundle.save_zones(self.site, [{'name': 'Z', 'polygon': self.utm(SQUARE),
                                       'props': {'cut_height': 60, 'effective': eff}}], 33)
        z = objects.load_all(self.site)[0]
        self.assertEqual(z['props']['effective'], eff)


class Activation(Base):
    def setUp(self):
        super().setUp()
        self.datum()

    def test_is_active_window(self):
        o = {'type': 'nogo', 'valid_from': 10.0, 'valid_until': 20.0}
        self.assertFalse(objects.is_active(o, 9.9))
        self.assertTrue(objects.is_active(o, 10.0))
        self.assertTrue(objects.is_active(o, 19.9))
        self.assertFalse(objects.is_active(o, 20.0))

    def test_open_ended(self):
        self.assertTrue(objects.is_active({'type': 'slow'}, 5.0))
        self.assertTrue(objects.is_active({'type': 'slow', 'valid_until': 9.0}, 5.0))
        self.assertTrue(objects.is_active({'type': 'slow', 'valid_from': 1.0}, 5.0))

    def test_other_types_ignore_validity(self):
        self.assertTrue(objects.is_active({'type': 'landmark', 'valid_until': 1.0}, 5.0))

    def test_active_rules(self):
        objects.save_places(self.site, [
            place('N', 'nogo', self.utm(SQUARE), valid_from=10.0),
            place('S', 'slow', self.utm(SQUARE), props={'speed_mps': 0.1}),
            place('B', 'noblade', self.utm(SQUARE)),
            place('L', 'landmark', self.utm([[1, 1]])),
        ], 33)
        self.assertEqual([r['type'] for r in objects.active_rules(self.site, 5.0)],
                         ['noblade', 'slow'])
        rules = {r['type']: r for r in objects.active_rules(self.site, 15.0)}
        self.assertEqual(set(rules), {'nogo', 'slow', 'noblade'})
        self.assertEqual(rules['slow']['speed_mps'], 0.1)
        self.assertIsNone(rules['nogo']['speed_mps'])
        self.assertAlmostEqual(rules['nogo']['points'][2][0], 2.0)

    def test_signature_follows_the_set(self):
        a = [{'id': 'x', 'type': 'nogo', 'name': 'n', 'points': [[0.0, 0.0]], 'speed_mps': None}]
        b = [dict(a[0], points=[[0.0, 0.5]])]
        self.assertEqual(objects.rules_signature(a), objects.rules_signature(list(a)))
        self.assertNotEqual(objects.rules_signature(a), objects.rules_signature(b))

    def test_stamp_edits_are_nogo_only(self):
        rules = [{'type': 'nogo', 'name': 'n', 'points': SQUARE},
                 {'type': 'slow', 'name': 's', 'points': SQUARE}]
        self.assertEqual([e['name'] for e in objects.nogo_stamp_edits(rules)], ['n'])


class BadRules(Base):
    """Hand-edited or damaged places.geojson: one object must not cost the rest."""

    def setUp(self):
        super().setUp()
        self.datum()
        objects.save_places(self.site, [
            place('Good', 'nogo', self.utm(SQUARE)),
            place('Slow', 'slow', self.utm(SQUARE), props={'speed_mps': 0.1}),
            place('Blade', 'noblade', self.utm(SQUARE))], 33)

    def edit(self, fn):
        with open(objects.places_path(self.site)) as f:
            fc = json.load(f)
        fn(fc['features'])
        with open(objects.places_path(self.site), 'w') as f:
            json.dump(fc, f)

    def by_name(self, problems=None):
        return {r['name']: r for r in objects.load_rules(self.site, 5.0, problems)}

    def test_null_speed_takes_the_default(self):
        self.edit(lambda f: f[1]['properties']['props'].update(speed_mps=None))
        problems = []
        rules = self.by_name(problems)
        self.assertEqual(rules['Slow']['speed_mps'], objects.DEFAULT_SLOW_MPS)
        self.assertNotIn('bad', rules['Slow'])
        self.assertEqual(len(rules), 3)
        self.assertEqual(problems, [{'id': 'o_Slow', 'name': 'Slow', 'reason': 'bad_speed'}])

    def test_speed_of_wrong_kind_or_range(self):
        for v in ('fast', 9.0, True):
            self.edit(lambda f: f[1]['properties']['props'].update(speed_mps=v))
            self.assertEqual(self.by_name()['Slow']['speed_mps'], objects.DEFAULT_SLOW_MPS)

    def test_bad_validity_stays_on(self):
        self.edit(lambda f: f[0]['properties'].update(valid_until='soon'))
        problems = []
        self.assertIn('Good', self.by_name(problems))
        self.assertEqual(problems[0]['reason'], 'bad_validity')
        self.assertEqual(len(objects.load_all(self.site)), 3)

    def test_unreadable_nogo_is_marked(self):
        self.edit(lambda f: f[0].update(geometry=None))
        problems = []
        r = self.by_name(problems)['Good']
        self.assertEqual((r['type'], r['points'], r['bad']), ('nogo', [], 'bad_geometry'))
        self.assertEqual(r['id'], 'o_Good')
        self.assertEqual(problems, [{'id': 'o_Good', 'name': 'Good', 'reason': 'bad_geometry'}])
        self.assertEqual(objects.stampable(list(self.by_name().values())), [])

    def test_unreadable_noblade_is_marked(self):
        self.edit(lambda f: f[2]['geometry'].update(coordinates=[[['a', 1]]]))
        r = self.by_name()['Blade']
        self.assertEqual((r['type'], r['bad']), ('noblade', 'bad_geometry'))

    def test_nan_point_is_unreadable(self):
        self.edit(lambda f: f[0]['geometry']['coordinates'][0].__setitem__(1, [float('nan'), 1.0]))
        self.assertEqual(self.by_name()['Good']['bad'], 'bad_geometry')

    def test_unknown_type_keeps_its_outline(self):
        self.edit(lambda f: f[0]['properties'].update(type='nog'))
        r = self.by_name()['Good']
        self.assertEqual((r['type'], r['bad']), ('nogo', 'bad_type'))
        self.assertEqual(len(r['points']), 4)
        self.assertAlmostEqual(r['points'][2][0], 2.0)

    def test_unreadable_slow_is_only_reported(self):
        self.edit(lambda f: f[1].update(geometry=None))
        problems = []
        self.assertNotIn('Slow', self.by_name(problems))
        self.assertEqual(problems[0]['reason'], 'bad_geometry')

    def test_unreadable_landmark_is_no_rule(self):
        objects.save_places(self.site, [place('L', 'landmark', self.utm([[1, 1]]))], 33)
        self.edit(lambda f: f[0].update(geometry=None))
        problems = []
        self.assertEqual(objects.load_rules(self.site, 5.0, problems), [])
        self.assertEqual(len(problems), 1)

    def test_garbage_feature_counts_as_nogo(self):
        self.edit(lambda f: f.append('junk'))
        rules = objects.load_rules(self.site, 5.0)
        self.assertEqual([r['bad'] for r in rules if 'bad' in r], ['bad_type'])

    def test_signature_sees_the_marker(self):
        a = objects.load_rules(self.site, 5.0)
        self.edit(lambda f: f[0].update(geometry=None))
        self.assertNotEqual(objects.rules_signature(a),
                            objects.rules_signature(objects.load_rules(self.site, 5.0)))

    def test_wrong_kinds_are_dropped_for_readers(self):
        self.edit(lambda f: f[0]['properties'].update(tags='x', props=[], yaw='n', note=5, id=7))
        o = next(p for p in objects.load_places(self.site) if p['name'] == 'Good')
        self.assertEqual((o['tags'], o['props'], o['yaw'], o['note']), ([], {}, None, ''))
        self.assertTrue(objects.is_active(o, 5.0))
        self.assertRegex(objects.load_rules(self.site, 5.0)[0]['id'], r'^o_')

    def test_broken_file_is_still_no_information(self):
        with open(objects.places_path(self.site), 'w') as f:
            f.write('{')
        self.assertIsNone(objects.load_rules(self.site))


class Validator(Base):
    def obj(self, t='nogo', points=None, **kw):
        fam = objects.family_of_type(t)
        o = objects._blank(fam, t, kw.pop('name', 'x'), points or SQUARE)
        if t == 'slow':
            o['props'] = {'speed_mps': 0.2}
        o.update(kw)
        return o

    def reason(self, o, site=None):
        return objects.validate(o, site)['reason']

    def test_good(self):
        self.assertTrue(objects.validate(self.obj())['ok'])

    def test_self_crossing(self):
        self.assertEqual(self.reason(self.obj(points=[[0, 0], [2, 2], [2, 0], [0, 2]])),
                         'self_crossing')

    def test_concave_is_fine(self):
        self.assertTrue(objects.validate(
            self.obj(points=[[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]]))['ok'])

    def test_too_small(self):
        s = 0.2                  # 0.04 m2
        self.assertEqual(self.reason(self.obj(points=[[0, 0], [s, 0], [s, s], [0, s]])),
                         'too_small')

    def test_too_few_points(self):
        self.assertEqual(self.reason(self.obj(points=[[0, 0], [1, 1]])), 'too_few_points')

    def test_bad_points(self):
        self.assertEqual(self.reason(self.obj(points=[[0, 0], [1, 'a'], [2, 2]])), 'bad_points')
        self.assertEqual(self.reason(self.obj(points=[[0, 0], [1, float('nan')], [2, 2]])),
                         'bad_points')

    def test_bad_type(self):
        self.assertEqual(self.reason({'type': 'pond', 'points': SQUARE}), 'bad_type')

    def test_speed_range(self):
        for v, ok in ((0.05, True), (0.40, True), (0.04, False), (0.41, False)):
            o = self.obj('slow')
            o['props']['speed_mps'] = v
            self.assertEqual(objects.validate(o)['ok'], ok, v)

    def test_validity(self):
        self.assertEqual(self.reason(self.obj('landmark', points=[[1, 1]], valid_until=9.0)),
                         'not_temporary')
        self.assertEqual(self.reason(self.obj(valid_from=9.0, valid_until=9.0)), 'bad_validity')

    def test_cut_height(self):
        o = self.obj('zone', props={'cut_height': 90})
        self.assertEqual(self.reason(o), 'bad_cut_height')

    def test_nogo_over_each_dock_point(self):
        self.datum()
        for name in ('DOCK', 'UNDOCK', 'docked'):
            bundle.save_waypoints(self.site, [
                {'name': name, 'e': objects.geo.map_to_utm(1, 1, DATUM)[0],
                 'n': objects.geo.map_to_utm(1, 1, DATUM)[1], 'yaw_rad': 0.0}], 33)
            self.assertTrue(self.reason(self.obj(), self.site).startswith(
                'nogo_contains_protected:'), name)

    def test_nogo_beside_dock_passes(self):
        self.datum()
        self.waypoint('DOCK', 9.0, 9.0)
        self.assertTrue(objects.validate(self.obj(), self.site)['ok'])

    def test_nogo_without_datum_refused(self):
        self.assertEqual(self.reason(self.obj(), self.site), 'no_datum')

    def test_nogo_over_zone_is_a_warning(self):
        self.datum()
        bundle.save_zones(self.site, [{'name': 'Z', 'polygon': self.utm(
            [[1, 1], [5, 1], [5, 5], [1, 5]]), 'props': {}}], 33)
        v = objects.validate(self.obj(), self.site)
        self.assertTrue(v['ok'])
        self.assertEqual(v['warnings'], ['nogo_overlaps_zone:Z'])

    def test_name_rules(self):
        self.datum()
        self.waypoint('A', 1.0, 1.0)
        self.waypoint('DOCK', 2.0, 2.0)
        wp = self.obj('waypoint', points=[[3, 3]], name='a')
        self.assertEqual(self.reason(wp, self.site), 'exists')
        wp = self.obj('waypoint', points=[[3, 3]], name='dock')
        self.assertEqual(self.reason(wp, self.site), 'reserved')

    def test_moving_a_dock_point_is_allowed_renaming_is_not(self):
        self.datum()
        self.waypoint('DOCK', 2.0, 2.0)
        o = objects.load_all(self.site)[0]
        o['points'] = [[4.0, 4.0]]
        self.assertTrue(objects.validate(o, self.site)['ok'])
        o['name'] = 'Garage'
        self.assertEqual(self.reason(o, self.site), 'reserved')


class Lookup(Base):
    def test_find_id(self):
        self.datum()
        objects.save_places(self.site, [
            place('Bazen', 'nogo', self.utm(SQUARE), aliases=['pool'], id='o_1'),
            place('Dub', 'landmark', self.utm([[5, 5]]), props={'category': 'strom'}, id='o_2'),
            place('Buk', 'landmark', self.utm([[6, 6]]), tags=['stin'], id='o_3'),
        ], 33)
        self.assertEqual(objects.find_id(self.site, 'bazen'), 'o_1')
        self.assertEqual(objects.find_id(self.site, 'POOL'), 'o_1')
        self.assertEqual(objects.find_id(self.site, 'strom'), 'o_2')
        self.assertEqual(objects.find_id(self.site, 'stin'), 'o_3')
        self.assertIsNone(objects.find_id(self.site, 'nic'))
        self.assertIsNone(objects.find_id(self.site, 'bazen', family='zone'))


class OwnerRule(unittest.TestCase):
    def nogo(self, pts=SQUARE, **kw):
        return dict(objects._blank('place', 'nogo', 'N', pts), **kw)

    def test_covers(self):
        self.assertTrue(objects.polygon_covers(SQUARE, SQUARE))
        self.assertTrue(objects.polygon_covers([[-1, -1], [3, -1], [3, 3], [-1, 3]], SQUARE))
        self.assertFalse(objects.polygon_covers([[0, 0], [1, 0], [1, 2], [0, 2]], SQUARE))

    def test_covers_tolerates_round_trip_noise(self):
        noisy = [[x + 0.002, y - 0.002] for x, y in SQUARE]
        self.assertTrue(objects.polygon_covers(noisy, SQUARE))

    def test_covers_notch_between_vertices(self):
        notch = [[-1, -1], [3, -1], [3, 3], [1.1, 3], [1, 1.9], [0.9, 3], [-1, 3]]
        self.assertFalse(objects.polygon_covers(notch, SQUARE))

    def test_codes(self):
        b = self.nogo()
        self.assertEqual(objects.owner_nogo_problem(b, None), 'owner_nogo_delete:N')
        self.assertIsNone(objects.owner_nogo_problem(self.nogo(source='agent'), None))
        self.assertIsNone(objects.owner_nogo_problem(None, b))
        self.assertIsNone(objects.owner_nogo_problem(
            objects._blank('place', 'slow', 'S', SQUARE), None))

    def test_switching_off_by_validity_already_past(self):
        b = self.nogo(valid_until=100.0)
        self.assertEqual(objects.owner_nogo_problem(b, dict(b, valid_from=50.0), 10.0),
                         'owner_nogo_validity:N')


class Programs(unittest.TestCase):
    def test_zone_back_in_order(self):
        f = objects.zone_back_in_order
        self.assertEqual(f(['A', 'B', 'C'], ['A', 'C'], 'B'), ['A', 'B', 'C'])
        self.assertEqual(f(['A', 'B'], ['A'], 'B'), ['A', 'B'])
        self.assertEqual(f(['A', 'B'], ['B'], 'A'), ['A', 'B'])
        self.assertEqual(f(['A', 'B', 'C'], ['C', 'A'], 'B'), ['C', 'A', 'B'])
        self.assertEqual(f(['B'], [], 'B'), ['B'])


class Diff(unittest.TestCase):
    def wp(self, name, x, **kw):
        o = objects._blank('waypoint', 'waypoint', name, [[x, 0.0]])
        o['yaw'] = 0.0
        o.update(kw)
        return o

    def zone(self, name, **props):
        o = objects._blank('zone', 'zone', name, [[0, 0], [1, 0], [1, 1]])
        o['props'] = props
        return o

    def test_noise_is_no_change(self):
        self.assertEqual(objects.diff_family([self.wp('A', 1.0)], [self.wp('A', 1.002)]), [])

    def test_move(self):
        d = objects.diff_family([self.wp('A', 1.0)], [self.wp('A', 1.5)])
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0][0]['points'], [[1.0, 0.0]])

    def test_create_delete(self):
        a, b = self.wp('A', 1.0), self.wp('B', 5.0)
        self.assertEqual(objects.diff_family([a], [a, b]), [(None, b)])
        self.assertEqual(objects.diff_family([a, b], [a]), [(b, None)])

    def test_rename(self):
        a, b = self.wp('A', 1.0), self.wp('B', 1.0)
        self.assertEqual(objects.diff_family([a], [b]), [(a, b)])

    def test_computed_zone_props(self):
        a = self.zone('Z', area=10, length=50, cut_height=60)
        b = self.zone('Z', area=11, length=52, cut_height=60, effective=[{'program': 'P'}])
        self.assertEqual(objects.diff_family([a], [b]), [])
        c = self.zone('Z', area=11, length=52, cut_height=50)
        self.assertEqual(len(objects.diff_family([a], [c])), 1)

    def test_dock_migration_is_no_delete(self):
        old, new = self.wp('docked', 1.0), self.wp('DOCKED', 1.0)
        self.assertEqual(objects.diff_family([old], [new]), [])
        moved = self.wp('DOCKED', 3.0)
        self.assertEqual(objects.diff_family([old], [moved]), [(old, moved)])


if __name__ == '__main__':
    unittest.main()
