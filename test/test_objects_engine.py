#!/usr/bin/env python3
"""map_objects_server logic with fake owners: no ROS master, nothing published.

Run: python3 -m pytest test/test_objects_engine.py   (from the package root)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yaml  # noqa: E402

from vitulus_mapping import bundle, geo, objects, objects_engine  # noqa: E402
from vitulus_mapping.objects_engine import Engine, Refuse, compute_stacks  # noqa: E402

DATUM = {'utm_e': 500000.0, 'utm_n': 5500000.0, 'yaw_rad': 0.1, 'utm_zone': 33}
SQUARE = [[0, 0], [2, 0], [2, 2], [0, 2]]


def utm(pts):
    return [geo.map_to_utm(x, y, DATUM) for x, y in pts]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeOwners:
    """navi_man, node_planner and mapping_manager: they get the message, write
    their file (or do not) and say nothing, like the real ones."""

    def __init__(self, site):
        self.served = site
        self.sent = []
        self.ready = True
        self.deaf = False
        self.refusal = None       # the planner's text for a zone save
        self.running = None
        self.busy = False
        self.program_deaf = False

    def site(self):
        return self.served

    def not_ready(self, family, site):
        return None if self.ready else 'navi_not_ready'

    def failure(self, family, name, since):
        return self.refusal if family == 'zone' else None

    def running_program(self):
        return self.running

    def mission_busy(self):
        return self.busy

    def save(self, family, obj):
        self.sent.append(('save', family, obj['name']))
        if self.deaf or self.refusal:
            return
        name = obj['name']
        if family == 'waypoint':
            items = [w for w in bundle.load_waypoints(self.served) if w['name'] != name]
            e, n = geo.map_to_utm(*obj['points'][0], DATUM)
            items.append({'name': name, 'e': e, 'n': n,
                          'yaw_rad': geo.map_yaw_to_utm(obj['yaw'], DATUM)})
            bundle.save_waypoints(self.served, items, 33)
        elif family == 'path':
            items = [p for p in bundle.load_paths(self.served) if p['name'] != name]
            items.append({'name': name, 'vertices': utm(obj['points']), 'yaws': [0.0] * len(obj['points'])})
            bundle.save_paths(self.served, items, 33)
        elif family == 'edit':
            items = [e for e in bundle.load_edits(self.served) if e['name'] != name]
            items.append({'name': name, 'kind': obj['type'], 'vertices': utm(obj['points']),
                          'width_m': obj['props'].get('width_m', 0.1)})
            bundle.save_edits(self.served, items, 33)
        else:
            items = [z for z in bundle.load_zones(self.served) if z['name'] != name]
            props = dict(obj['props'], area=12, length=40)   # the planner computes these
            items.append({'name': name, 'polygon': utm(obj['points']), 'props': props})
            bundle.save_zones(self.served, items, 33)

    def remove(self, family, name):
        self.sent.append(('remove', family, name))
        if self.deaf:
            return
        if family == 'waypoint':
            bundle.save_waypoints(self.served, [w for w in bundle.load_waypoints(self.served)
                                                if w['name'] != name], 33)
        elif family == 'path':
            bundle.save_paths(self.served, [p for p in bundle.load_paths(self.served)
                                            if p['name'] != name], 33)
        elif family == 'edit':
            left = [e for e in bundle.load_edits(self.served) if e['name'] != name]
            if left:
                bundle.save_edits(self.served, left, 33)
            else:
                os.remove(bundle.edits_path(self.served))   # as cb_remove_edit does
        else:
            bundle.save_zones(self.served, [z for z in bundle.load_zones(self.served)
                                            if z['name'] != name], 33)
            programs = bundle.load_programs(self.served)
            for p in programs:
                p['zone_names'] = [z for z in p['zone_names'] if z != name]
            if programs:
                bundle.save_programs(self.served, programs)

    def save_program(self, name, zone_names, added):
        self.sent.append(('program', name, list(zone_names)))
        if self.program_deaf:
            return
        programs = bundle.load_programs(self.served)
        for p in programs:
            if p['name'] == name:
                p['zone_names'] = list(zone_names)
        bundle.save_programs(self.served, programs)

    def rename(self, family, old, new):
        self.sent.append(('rename', family, old, new))
        if family == 'waypoint':
            items = bundle.load_waypoints(self.served)
            for w in items:
                if w['name'] == old:
                    w['name'] = new
            bundle.save_waypoints(self.served, items, 33)
        elif family == 'zone':
            items = bundle.load_zones(self.served)
            for z in items:
                if z['name'] == old:
                    z['name'] = new
            bundle.save_zones(self.served, items, 33)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        self.make_site('s')
        self.owners = FakeOwners('s')
        self.clock = Clock()
        self.out = []
        self.engine = self.new_engine()
        self.engine.tick()

    def tearDown(self):
        bundle.SITES_ROOT = self.old_root
        self.tmp.cleanup()

    def make_site(self, site, datum=True):
        os.makedirs(bundle.site_dir(site))
        if datum:
            with open(os.path.join(bundle.site_dir(site), 'datum.yaml'), 'w') as f:
                yaml.safe_dump(DATUM, f)

    def new_engine(self):
        return Engine(self.owners, lambda k, v: self.out.append((k, v)), clock=self.clock,
                      sleep=self.clock.sleep)

    def op(self, **kw):
        kw.setdefault('req', 'r')
        kw.setdefault('actor', 'owner')
        return self.engine.handle(kw)

    def save(self, **obj):
        return self.op(op='save', object=obj)

    def nogo(self, name='N', points=None, **kw):
        return self.save(type='nogo', name=name, points=points or SQUARE, **kw)

    def wp(self, name='A', x=1.0, y=1.0):
        return self.save(type='waypoint', name=name, points=[[x, y]], yaw=0.3)

    def zone(self, name='Z', **props):
        return self.save(type='zone', name=name, points=[[0, 0], [8, 0], [8, 8], [0, 8]], props=props)

    def journal(self, site='s'):
        return objects_engine.read_journal(
            os.path.join(objects.history_dir(site), 'journal.jsonl'))

    def last(self, what):
        return [v for k, v in self.out if k == what][-1]

    def count(self, what):
        return len([1 for k, _ in self.out if k == what])

    def obj(self, name, site='s'):
        return next((o for o in objects.load_all(site) if o['name'] == name), None)

    def tick(self, dt=1.0):
        self.clock.t += dt
        self.engine.tick()


class Places(Base):
    def test_create_writes_journals_and_answers(self):
        r = self.nogo()
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['req'], 'r')
        o = self.obj('N')
        self.assertEqual(r['id'], o['id'])
        self.assertRegex(o['id'], r'^o_[0-9a-f]{8}$')
        self.assertEqual(o['source'], 'owner')
        j = self.journal()
        self.assertEqual(len(j), 1)
        self.assertEqual((j[0]['op'], j[0]['actor'], j[0]['family'], j[0]['before']),
                         ('save', 'owner', 'place', None))
        self.assertEqual(self.owners.sent, [])     # place family is written directly

    def test_topics_are_map_frame(self):
        self.nogo()
        o = self.last('objects')['objects'][0]
        self.assertAlmostEqual(o['points'][2][0], 2.0)
        self.assertAlmostEqual(self.last('rules')['rules'][0]['points'][2][1], 2.0)
        self.assertAlmostEqual(self.last('history')['entries'][0]['after']['points'][2][0], 2.0)

    def test_files_are_utm(self):
        self.nogo()
        self.assertGreater(objects.load_places('s')[0]['points'][0][0], 1000.0)
        self.assertGreater(self.journal()[0]['after']['points'][0][0], 1000.0)

    def test_refused_by_validator(self):
        r = self.nogo(points=[[0, 0], [2, 2], [2, 0], [0, 2]])
        self.assertFalse(r['ok'])
        self.assertEqual(r['reason'], 'self_crossing')
        self.assertEqual(self.journal(), [])

    def test_update_merges_props(self):
        r = self.save(type='slow', name='S', points=SQUARE, props={'speed_mps': 0.1, 'x': 1})
        self.op(op='update', id=r['id'], props={'speed_mps': 0.3})
        o = self.obj('S')
        self.assertEqual(o['props'], {'speed_mps': 0.3, 'x': 1})

    def test_id_survives_rename_and_reshape(self):
        i = self.nogo()['id']
        self.assertTrue(self.op(op='update', id=i, name='M')['ok'])
        self.assertTrue(self.op(op='update', id=i, points=[[0, 0], [3, 0], [3, 3]])['ok'])
        self.assertEqual(self.obj('M')['id'], i)

    def test_delete(self):
        i = self.nogo()['id']
        self.assertTrue(self.op(op='delete', id=i)['ok'])
        self.assertIsNone(self.obj('N'))

    def test_save_with_gone_id_recreates_it(self):
        i = self.nogo()['id']
        self.op(op='delete', id=i)
        self.assertTrue(self.save(id=i, type='nogo', name='N', points=SQUARE)['ok'])
        self.assertEqual(self.obj('N')['id'], i)

    def test_auto_name(self):
        self.save(type='landmark', points=[[1, 1]])
        self.save(type='landmark', points=[[2, 1]])
        self.assertIsNotNone(self.obj('landmark_2'))

    def test_no_change_no_entry(self):
        i = self.nogo()['id']
        self.op(op='update', id=i, name='N')
        self.assertEqual(len(self.journal()), 1)

    def test_nonexistent(self):
        self.assertEqual(self.op(op='delete', id='o_x')['reason'], 'not_found')
        self.assertEqual(self.op(op='fly')['reason'], 'bad_op')
        self.assertEqual(self.op(op='save', actor='cat', object={})['reason'], 'bad_actor')

    def test_garbage_message(self):
        self.assertEqual(self.engine.handle(None)['reason'], 'bad_request')

    def test_broken_places_is_not_overwritten(self):
        self.nogo()
        with open(objects.places_path('s'), 'w') as f:
            f.write('{broken')
        self.assertEqual(self.nogo('M')['reason'], 'places_unreadable')
        with open(objects.places_path('s')) as f:
            self.assertEqual(f.read(), '{broken')


class Forwarding(Base):
    def test_waypoint_goes_to_owner_once(self):
        r = self.wp()
        self.assertTrue(r['ok'], r)
        self.assertEqual(self.owners.sent, [('save', 'waypoint', 'A')])
        self.tick()
        self.tick()
        self.assertEqual(len(self.journal()), 1)    # the watcher does not journal it again
        self.assertEqual(self.journal()[0]['actor'], 'owner')

    def test_id_goes_to_sidecar(self):
        i = self.wp()['id']
        self.assertEqual(objects.load_meta('s')['waypoint:A']['id'], i)
        self.assertEqual(self.obj('A')['id'], i)

    def test_meta_only_stays_off_the_owner(self):
        i = self.wp()['id']
        self.owners.sent.clear()
        self.owners.ready = False
        self.assertTrue(self.op(op='update', id=i, note='u vrat', tags=['x'])['ok'])
        self.assertEqual(self.owners.sent, [])
        self.assertEqual(objects.load_meta('s')['waypoint:A']['note'], 'u vrat')
        self.assertEqual(self.obj('A')['tags'], ['x'])

    def test_rename_then_move(self):
        i = self.wp()['id']
        self.owners.sent.clear()
        r = self.op(op='update', id=i, name='B', points=[[4.0, 4.0]])
        self.assertTrue(r['ok'], r)
        self.assertEqual(self.owners.sent[0], ('rename', 'waypoint', 'A', 'B'))
        self.assertEqual(self.owners.sent[1], ('save', 'waypoint', 'B'))
        self.assertEqual(self.obj('B')['id'], i)
        self.assertNotIn('waypoint:A', objects.load_meta('s'))

    def test_owner_silent_is_timeout(self):
        self.owners.deaf = True
        r = self.wp()
        self.assertEqual(r['reason'], 'owner_timeout')
        self.assertEqual(self.journal(), [])

    def test_late_write_keeps_the_actor(self):
        self.owners.deaf = True
        self.op(op='save', actor='agent', object=dict(type='waypoint', name='A', points=[[1.0, 1.0]]))
        self.owners.deaf = False
        self.owners.save('waypoint', {'name': 'A', 'points': [[1.0, 1.0]], 'yaw': 0.0})
        self.tick()
        self.assertEqual(self.journal()[0]['actor'], 'agent')

    def test_owner_not_ready_after_switch(self):
        self.owners.ready = False
        r = self.wp()
        self.assertEqual(r['reason'], 'navi_not_ready')
        self.assertEqual(self.owners.sent, [])     # nothing went to the old site's memory

    def test_ready_arrives_while_waiting(self):
        self.owners.ready = False
        real = self.clock.sleep

        def sleep(dt):
            real(dt)
            self.owners.ready = True
        self.engine.sleep = sleep
        self.assertTrue(self.wp()['ok'])

    def test_protected(self):
        i = self.seed_dock()
        self.assertEqual(self.op(op='delete', id=i)['reason'], 'protected')
        self.assertEqual(self.op(op='update', id=i, name='X')['reason'], 'reserved')
        self.assertTrue(self.op(op='update', id=i, points=[[2.0, 2.0]])['ok'])

    def seed_dock(self):
        self.owners.save('waypoint', {'name': 'DOCKED', 'points': [[1.0, 1.0]], 'yaw': 0.0})
        self.tick()
        return self.obj('DOCKED')['id']

    def test_nogo_over_dock_refused(self):
        self.seed_dock()
        self.assertTrue(self.nogo()['reason'].startswith('nogo_contains_protected'))

    def test_zone_computed_props_arrive(self):
        self.assertTrue(self.zone(cut_height=60)['ok'])
        self.assertEqual(self.obj('Z')['props']['area'], 12)
        self.assertEqual(self.journal()[0]['after']['props']['area'], 12)

    def test_zone_refusal_is_the_reason(self):
        self.owners.refusal = "Zone 'Z' NOT saved: program 'P' is running on it."
        r = self.zone()
        self.assertIn('running', r['reason'])

    def test_zone_delete_in_running_program(self):
        i = self.zone()['id']
        bundle.save_programs('s', [{'name': 'P', 'zone_names': ['Z']}])
        self.owners.running = 'P'
        self.assertEqual(self.op(op='delete', id=i)['reason'], 'zone_in_running_program:P')
        self.owners.running = None
        r = self.op(op='delete', id=i)
        self.assertTrue(r['ok'])
        self.assertEqual(r['warnings'], [])

    def test_last_edit_removed_file_gone(self):
        i = self.save(type='obstacle', name='E', points=SQUARE)['id']
        self.assertTrue(self.op(op='delete', id=i)['ok'])
        self.assertFalse(os.path.exists(bundle.edits_path('s')))


class Watcher(Base):
    def seed(self):
        self.owners.save('waypoint', {'name': 'A', 'points': [[1.0, 1.0]], 'yaw': 0.0})

    def test_outside_change_is_journaled_with_id(self):
        self.seed()
        self.owners.sent.clear()
        self.tick()
        j = self.journal()
        self.assertEqual((j[0]['actor'], j[0]['op'], j[0]['name']), ('unknown', 'save', 'A'))
        self.assertEqual(objects.load_meta('s')['waypoint:A']['id'], j[0]['id'])

    def test_only_mtime_changed(self):
        self.seed()
        self.tick()
        os.utime(bundle.waypoints_path('s'), (1, 1))
        self.tick()
        self.assertEqual(len(self.journal()), 1)

    def test_computed_zone_props_are_no_change(self):
        self.zone()
        z = bundle.load_zones('s')
        z[0]['props']['area'] = 99
        z[0]['props']['length'] = 77
        z[0]['props']['effective'] = [{'program': 'P', 'cut_height': 45, 'rpm': 3300}]
        bundle.save_zones('s', z, 33)
        self.tick()
        self.assertEqual(len(self.journal()), 1)
        self.assertEqual(self.obj('Z')['props']['area'], 99)    # shown, though

    def test_real_zone_prop_change(self):
        self.zone()
        z = bundle.load_zones('s')
        z[0]['props']['cut_height'] = 40
        bundle.save_zones('s', z, 33)
        self.tick()
        self.assertEqual(self.journal()[-1]['op'], 'update')

    def test_missing_file_wipes_nothing(self):
        i = self.wp()['id']
        before = open(objects.meta_path('s')).read()
        os.remove(bundle.waypoints_path('s'))
        self.tick()
        self.assertEqual(len(self.journal()), 1)     # no delete journaled
        self.assertEqual(open(objects.meta_path('s')).read(), before)
        self.owners.save('waypoint', {'name': 'A', 'points': [[1.0, 1.0]], 'yaw': 0.3})
        self.tick()
        self.assertEqual(len(self.journal()), 1)
        self.assertEqual(self.obj('A')['id'], i)

    def test_broken_file_wipes_nothing(self):
        self.wp()
        with open(bundle.waypoints_path('s'), 'w') as f:
            f.write('{')
        self.tick()
        self.assertEqual(len(self.journal()), 1)

    def test_empty_file_is_real_information(self):
        self.wp()
        bundle.save_waypoints('s', [], 33)
        self.tick()
        self.assertEqual(self.journal()[-1]['op'], 'delete')

    def test_outside_rename_keeps_id_and_note(self):
        i = self.wp()['id']
        self.op(op='update', id=i, note='n')
        self.owners.rename('waypoint', 'A', 'B')
        self.tick()
        b = self.obj('B')
        self.assertEqual((b['id'], b['note']), (i, 'n'))
        self.assertEqual(self.journal()[-1]['op'], 'update')

    def test_mass_change_is_one_undo_step(self):
        self.seed()
        self.owners.save('waypoint', {'name': 'B', 'points': [[2.0, 1.0]], 'yaw': 0.0})
        self.tick()
        self.assertEqual(len(self.engine.undo), 1)
        self.assertEqual(len(self.engine.undo[0]), 2)

    def test_change_made_while_down_is_seen(self):
        self.wp()
        self.owners.save('waypoint', {'name': 'A', 'points': [[5.0, 5.0]], 'yaw': 0.3})
        self.owners.rename('waypoint', 'A', 'B')
        self.owners.save('waypoint', {'name': 'C', 'points': [[7.0, 5.0]], 'yaw': 0.3})
        self.engine = self.new_engine()
        self.engine.tick()
        self.assertGreater(len(self.journal()), 1)
        self.assertEqual(self.journal()[-1]['actor'], 'unknown')

    def test_first_contact_journals_nothing(self):
        self.make_site('old')
        bundle.save_waypoints('old', [{'name': 'A', 'e': DATUM['utm_e'], 'n': DATUM['utm_n'],
                                       'yaw_rad': 0.0}], 33)
        self.owners.served = 'old'
        self.tick()
        self.assertEqual(self.journal('old'), [])
        self.assertIn('waypoint:A', objects.load_meta('old'))

    def test_dock_migration_keeps_id(self):
        self.owners.save('waypoint', {'name': 'DOCKED', 'points': [[1.0, 1.0]], 'yaw': 0.0})
        self.tick()
        i = self.obj('DOCKED')['id']
        entries = len(self.journal())
        # the site as it was before the rename: file, sidecar and our snapshot
        for path in (bundle.waypoints_path('s'), objects.meta_path('s'),
                     os.path.join(objects.history_dir('s'), 'snapshot.json')):
            with open(path) as f:
                text = f.read()
            with open(path, 'w') as f:
                f.write(text.replace('DOCKED', 'docked'))
        self.engine = self.new_engine()
        self.engine.tick()
        self.assertEqual(self.obj('DOCKED')['id'], i)
        self.assertTrue(self.obj('DOCKED')['protected'])
        self.assertEqual(len(self.journal()), entries)


class Undo(Base):
    def test_undo_create_and_redo(self):
        i = self.nogo()['id']
        r = self.op(op='undo')
        self.assertTrue(r['ok'])
        self.assertEqual(r['id'], i)
        self.assertIsNone(self.obj('N'))
        self.assertEqual(self.journal()[-1]['op'], 'undo')
        self.assertTrue(self.op(op='redo')['ok'])
        self.assertEqual(self.obj('N')['id'], i)

    def test_undo_delete_restores_id_and_meta(self):
        i = self.wp()['id']
        self.op(op='update', id=i, note='n')
        self.op(op='delete', id=i)
        self.assertTrue(self.op(op='undo')['ok'])
        o = self.obj('A')
        self.assertEqual((o['id'], o['note']), (i, 'n'))

    def test_undo_move_goes_through_owner(self):
        i = self.wp()['id']
        self.op(op='update', id=i, points=[[5.0, 5.0]])
        self.owners.sent.clear()
        self.assertTrue(self.op(op='undo')['ok'])
        self.assertEqual(self.owners.sent, [('save', 'waypoint', 'A')])
        self.assertAlmostEqual(self.obj('A')['points'][0][0], 1.0)

    def test_undo_rename(self):
        i = self.wp()['id']
        self.op(op='update', id=i, name='B')
        self.assertTrue(self.op(op='undo')['ok'])
        self.assertEqual(self.obj('A')['id'], i)

    def test_undo_outside_change(self):
        self.owners.save('waypoint', {'name': 'A', 'points': [[1.0, 1.0]], 'yaw': 0.0})
        self.tick()
        self.assertTrue(self.op(op='undo', actor='agent')['ok'])
        self.assertIsNone(self.obj('A'))

    def test_history_is_never_rewritten(self):
        self.nogo()
        n = len(self.journal())
        self.op(op='undo')
        self.op(op='redo')
        self.assertEqual(len(self.journal()), n + 2)
        self.assertEqual([e['seq'] for e in self.journal()], [1, 2, 3])

    def test_new_change_clears_redo(self):
        self.nogo()
        self.op(op='undo')
        self.nogo('M')
        self.assertEqual(self.op(op='redo')['reason'], 'nothing_to_redo')

    def test_empty(self):
        self.assertEqual(self.op(op='undo')['reason'], 'nothing_to_undo')

    def test_zone_undo_uses_planner_save_and_can_fail(self):
        i = self.zone(cut_height=60)['id']
        self.op(op='update', id=i, props={'cut_height': 40})
        self.owners.sent.clear()
        self.owners.refusal = "Zone 'Z' NOT saved: program 'P' is running on it — stop it first."
        r = self.op(op='undo')
        self.assertFalse(r['ok'])
        self.assertIn('running', r['reason'])
        self.assertEqual(self.owners.sent, [('save', 'zone', 'Z')])
        self.assertEqual(len(self.engine.undo), 2)     # still there to try again
        self.owners.refusal = None
        self.assertTrue(self.op(op='undo')['ok'])
        self.assertEqual(self.obj('Z')['props']['cut_height'], 60)

    def test_restored_nogo_rechecked_against_moved_dock(self):
        i = self.nogo()['id']
        self.op(op='delete', id=i)
        self.owners.save('waypoint', {'name': 'DOCK', 'points': [[1.0, 1.0]], 'yaw': 0.0})
        self.tick()                  # dock flow's write is not an undo step
        self.assertTrue(self.op(op='undo')['reason'].startswith('nogo_contains_protected'))

    def test_dock_flow_is_no_undo_step(self):
        self.nogo()
        self.owners.save('waypoint', {'name': 'DOCKED', 'points': [[9.0, 9.0]], 'yaw': 0.0})
        self.tick()
        self.op(op='undo')
        self.assertIsNone(self.obj('N'))
        self.assertIsNotNone(self.obj('DOCKED'))

    def test_survives_restart(self):
        self.nogo()
        self.nogo('M')
        self.op(op='undo')
        self.engine = self.new_engine()
        self.engine.tick()
        self.assertEqual((len(self.engine.undo), len(self.engine.redo)), (1, 1))
        self.assertTrue(self.op(op='redo')['ok'])
        self.assertIsNotNone(self.obj('M'))

    def test_reach(self):
        e = [{'seq': i, 'op': 'save', 'group': None, 'family': 'place', 'name': 'x'}
             for i in range(250)]
        undo, _ = compute_stacks(e)
        self.assertEqual(len(undo), 200)
        self.assertEqual(undo[0][0]['seq'], 50)

    def test_history_topic(self):
        for i in range(105):
            self.nogo('N%d' % i, points=[[0, 0], [2, 0], [2, 2]])
        h = self.last('history')
        self.assertEqual(len(h['entries']), 100)
        self.assertEqual(h['seq'], 105)
        self.assertTrue(h['can_undo'])
        self.assertFalse(h['can_redo'])
        self.assertEqual(len(self.journal()), 105)

    def test_torn_journal_line(self):
        self.nogo()
        p = os.path.join(objects.history_dir('s'), 'journal.jsonl')
        with open(p, 'a') as f:
            f.write('{"seq": 2, "op"')
        self.engine = self.new_engine()
        self.engine.tick()
        self.nogo('M')
        self.assertEqual([e['name'] for e in self.journal()], ['N', 'M'])


PROGRAM = {'name': 'P', 'speed': 'slow', 'rpm': 3400, 'cut_height': 70, 'override_zone': True}


class ZoneInPrograms(Base):
    def setUp(self):
        super().setUp()
        self.ids = {n: self.zone(n)['id'] for n in 'ABC'}
        bundle.save_programs('s', [dict(PROGRAM, zone_names=['A', 'B', 'C']),
                                   dict(PROGRAM, name='Q', zone_names=['B']),
                                   dict(PROGRAM, name='R', zone_names=['C'])])
        self.tick()

    def zones_of(self, name):
        return objects.program_zone_names('s')[name]

    def test_journal_holds_membership_in_order(self):
        self.op(op='delete', id=self.ids['B'])
        before = self.journal()[-1]['before']
        self.assertEqual(before['props']['programs'],
                         [{'name': 'P', 'zone_names': ['A', 'B', 'C']},
                          {'name': 'Q', 'zone_names': ['B']}])

    def test_undo_restores_membership_and_order(self):
        self.op(op='delete', id=self.ids['B'])
        self.assertEqual(self.zones_of('P'), ['A', 'C'])
        self.owners.sent.clear()
        r = self.op(op='undo')
        self.assertTrue(r['ok'], r)
        self.assertEqual(self.owners.sent[0], ('save', 'zone', 'B'))
        self.assertEqual(self.zones_of('P'), ['A', 'B', 'C'])
        self.assertEqual(self.zones_of('Q'), ['B'])
        self.assertEqual(self.zones_of('R'), ['C'])

    def test_program_parameters_stay(self):
        self.op(op='delete', id=self.ids['B'])
        self.op(op='undo')
        p = next(p for p in bundle.load_programs('s') if p['name'] == 'P')
        for k, v in PROGRAM.items():
            self.assertEqual(p[k], v)

    def test_undo_of_outside_delete(self):
        self.owners.remove('zone', 'B')
        self.tick()
        self.assertEqual(len(self.journal()[-1]['before']['props']['programs']), 2)
        self.assertTrue(self.op(op='undo')['ok'])
        self.assertEqual(self.zones_of('P'), ['A', 'B', 'C'])

    def test_running_program_blocks_before_anything(self):
        self.op(op='delete', id=self.ids['B'])
        self.owners.running = 'P'
        self.owners.sent.clear()
        self.assertEqual(self.op(op='undo')['reason'], 'program_running:P')
        self.assertEqual(self.owners.sent, [])
        self.assertIsNone(self.obj('B'))
        self.assertEqual(len(self.engine.redo), 0)
        self.owners.running = None
        self.assertTrue(self.op(op='undo')['ok'])

    def test_program_changed_meanwhile_keeps_its_change(self):
        self.op(op='delete', id=self.ids['B'])
        programs = bundle.load_programs('s')
        programs[0]['zone_names'] = ['C', 'A']
        bundle.save_programs('s', programs)
        self.op(op='undo')
        self.assertEqual(self.zones_of('P'), ['C', 'A', 'B'])

    def test_program_gone_is_a_warning(self):
        self.op(op='delete', id=self.ids['B'])
        bundle.save_programs('s', [p for p in bundle.load_programs('s') if p['name'] != 'Q'])
        r = self.op(op='undo')
        self.assertTrue(r['ok'])
        self.assertEqual(r['warnings'], ['program_gone:Q'])

    def test_planner_silent_takes_the_zone_back(self):
        self.op(op='delete', id=self.ids['B'])
        self.owners.program_deaf = True
        r = self.op(op='undo')
        self.assertEqual(r['reason'], 'owner_timeout')
        self.assertIsNone(self.obj('B'))
        self.tick()
        self.assertEqual(len(self.engine.undo[-1]), 1)
        self.assertEqual(self.engine.undo[-1][0]['op'], 'delete')

    def test_programs_survive_redo_cycle(self):
        self.op(op='delete', id=self.ids['B'])
        self.op(op='undo')
        self.op(op='redo')
        self.assertEqual(self.zones_of('P'), ['A', 'C'])
        self.op(op='undo')
        self.assertEqual(self.zones_of('P'), ['A', 'B', 'C'])

    def test_computed_key_never_reaches_the_planner_or_file(self):
        self.op(op='delete', id=self.ids['B'])
        self.op(op='undo')
        self.assertNotIn('programs', self.obj('B')['props'])


BIG = [[-1, -1], [3, -1], [3, 3], [-1, 3]]


class OwnerNogo(Base):
    def setUp(self):
        super().setUp()
        self.i = self.nogo()['id']        # actor owner, SQUARE

    def agent(self, **kw):
        return self.op(actor='agent', **kw)

    def test_delete(self):
        self.assertEqual(self.agent(op='delete', id=self.i)['reason'], 'owner_nogo_delete:N')
        self.assertIsNotNone(self.obj('N'))
        self.assertEqual(len(self.journal()), 1)

    def test_shrink(self):
        r = self.agent(op='update', id=self.i, points=[[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(r['reason'], 'owner_nogo_shrink:N')

    def test_move_is_a_shrink(self):
        r = self.agent(op='update', id=self.i, points=[[1, 0], [3, 0], [3, 2], [1, 2]])
        self.assertEqual(r['reason'], 'owner_nogo_shrink:N')

    def test_concave_cut_between_vertices(self):
        # every old vertex stays inside, a notch cuts the top edge
        r = self.agent(op='update', id=self.i, points=[
            [-1, -1], [3, -1], [3, 3], [1.2, 3], [1, 1], [0.8, 3], [-1, 3]])
        self.assertEqual(r['reason'], 'owner_nogo_shrink:N')

    def test_enlarge_passes(self):
        self.assertTrue(self.agent(op='update', id=self.i, points=BIG)['ok'])

    def test_same_outline_passes_rename_and_note(self):
        self.assertTrue(self.agent(op='update', id=self.i, name='M', note='x')['ok'])

    def test_type(self):
        r = self.agent(op='update', id=self.i, type='slow', props={'speed_mps': 0.2})
        self.assertEqual(r['reason'], 'owner_nogo_type:N')

    def test_validity(self):
        t = self.clock.t
        for kw in ({'valid_until': t + 100}, {'valid_from': t + 100}):
            self.assertEqual(self.agent(op='update', id=self.i, **kw)['reason'],
                             'owner_nogo_validity:N')

    def test_extending_validity_passes(self):
        t = self.clock.t
        self.op(op='update', id=self.i, valid_until=t + 100)
        self.assertTrue(self.agent(op='update', id=self.i, valid_until=t + 200)['ok'])
        self.assertEqual(self.agent(op='update', id=self.i, valid_until=t + 150)['reason'],
                         'owner_nogo_validity:N')

    def test_owner_is_free(self):
        self.assertTrue(self.op(op='delete', id=self.i)['ok'])

    def test_save_with_id_cannot_go_around(self):
        r = self.agent(op='save', object=dict(id=self.i, type='nogo', name='N',
                                              points=[[0, 0], [1, 0], [1, 1]]))
        self.assertEqual(r['reason'], 'owner_nogo_shrink:N')

    def test_own_nogo_is_free(self):
        i = self.agent(op='save', object=dict(type='nogo', name='A', points=[[5, 5], [7, 5], [7, 7]],
                                              source='owner'))['id']
        self.assertEqual(self.obj('A')['source'], 'agent')
        self.assertTrue(self.agent(op='delete', id=i)['ok'])

    def test_robot_origin_counts_as_protected(self):
        i = self.op(op='save', actor='robot', object=dict(
            type='nogo', name='R', points=[[5, 5], [7, 5], [7, 7]]))['id']
        self.assertEqual(self.agent(op='delete', id=i)['reason'], 'owner_nogo_delete:R')

    def test_undo_of_owner_create(self):
        self.assertEqual(self.agent(op='undo')['reason'], 'owner_nogo_delete:N')
        self.assertIsNotNone(self.obj('N'))
        self.assertTrue(self.op(op='undo')['ok'])

    def test_undo_of_owner_enlarge(self):
        self.op(op='update', id=self.i, points=BIG)
        self.assertEqual(self.agent(op='undo')['reason'], 'owner_nogo_shrink:N')

    def test_redo_of_owner_delete(self):
        self.op(op='delete', id=self.i)
        self.assertTrue(self.op(op='undo')['ok'])       # restored by the owner
        self.assertEqual(self.agent(op='redo')['reason'], 'owner_nogo_delete:N')
        self.assertIsNotNone(self.obj('N'))

    def test_undo_of_owner_delete_gives_it_back(self):
        self.op(op='delete', id=self.i)
        self.assertTrue(self.agent(op='undo')['ok'])

    def test_group_refused_as_a_whole(self):
        r = self.agent(op='group', ops=[
            {'op': 'save', 'object': dict(type='landmark', name='L', points=[[1, 1]])},
            {'op': 'delete', 'id': self.i}])
        self.assertEqual(r['reason'], 'owner_nogo_delete:N')
        self.assertIsNone(self.obj('L'))
        self.assertIsNotNone(self.obj('N'))
        self.assertEqual(len(self.journal()), 1)

    def test_undo_step_refused_before_any_entry(self):
        self.op(op='group', ops=[
            {'op': 'save', 'object': dict(type='landmark', name='L', points=[[1, 1]])},
            {'op': 'delete', 'id': self.i}])
        self.op(op='undo')
        r = self.agent(op='redo')
        self.assertEqual(r['reason'], 'owner_nogo_delete:N')
        self.assertIsNone(self.obj('L'))

    def test_rollback_is_not_stopped_by_the_rule(self):
        r = self.agent(op='group', ops=[
            {'op': 'update', 'id': self.i, 'points': BIG},
            {'op': 'save', 'object': dict(type='nogo', name='X', points=[[0, 0]])}])
        self.assertFalse(r['ok'])
        self.assertEqual(len(self.obj('N')['points']), 4)
        self.assertAlmostEqual(self.obj('N')['points'][0][0], 0.0)


class Groups(Base):
    def test_group_is_one_step(self):
        r = self.op(op='group', ops=[
            {'op': 'save', 'object': dict(type='nogo', name='A', points=SQUARE)},
            {'op': 'save', 'object': dict(type='landmark', name='B', points=[[1, 1]])}])
        self.assertTrue(r['ok'], r)
        j = self.journal()
        self.assertEqual(len(j), 2)
        self.assertEqual(j[0]['group'], j[1]['group'])
        self.assertIsNotNone(j[0]['group'])
        self.assertEqual(j[0]['actor'], 'owner')
        self.assertTrue(self.op(op='undo')['ok'])
        self.assertIsNone(self.obj('A'))
        self.assertIsNone(self.obj('B'))
        self.assertTrue(self.op(op='redo')['ok'])
        self.assertIsNotNone(self.obj('A'))
        self.assertIsNotNone(self.obj('B'))

    def test_failed_group_leaves_nothing(self):
        r = self.op(op='group', ops=[
            {'op': 'save', 'object': dict(type='nogo', name='A', points=SQUARE)},
            {'op': 'save', 'object': dict(type='waypoint', name='W', points=[[1, 1]])},
            {'op': 'save', 'object': dict(type='nogo', name='C', points=[[0, 0]])}])
        self.assertFalse(r['ok'])
        self.assertEqual(r['reason'], 'too_few_points')
        self.assertIsNone(self.obj('A'))
        self.assertIsNone(self.obj('W'))
        self.assertEqual(self.journal(), [])

    def test_no_nesting(self):
        self.assertEqual(self.op(op='group', ops=[{'op': 'undo'}])['reason'], 'bad_op')


class Sites(Base):
    def test_switch_reloads_everything(self):
        self.nogo()
        self.make_site('t')
        self.owners.served = 't'
        self.tick()
        self.assertEqual(self.last('objects'), dict(self.last('objects'), site='t', objects=[]))
        self.assertEqual(self.last('rules')['rules'], [])
        self.assertFalse(self.last('history')['can_undo'])
        self.assertEqual(self.op(op='undo')['reason'], 'nothing_to_undo')
        self.owners.served = 's'
        self.tick()
        self.assertEqual(len(self.last('rules')['rules']), 1)
        self.assertTrue(self.last('history')['can_undo'])

    def test_switch_journals_nothing(self):
        self.nogo()
        self.make_site('t')
        self.owners.served = 't'
        self.tick()
        self.assertEqual(self.journal('t'), [])
        self.assertEqual(len(self.journal()), 1)

    def test_op_during_switch_is_refused_not_misfiled(self):
        self.make_site('t')
        self.owners.served = 't'           # known to owners, tick has not run yet
        r = self.wp()
        self.assertEqual(r['reason'], 'site_changed')
        self.assertEqual(self.owners.sent, [])

    def test_switch_while_waiting_for_owner(self):
        self.owners.deaf = True
        self.owners.save = lambda *a: setattr(self.owners, 'served', None)
        self.assertEqual(self.wp()['reason'], 'site_changed')

    def test_no_site(self):
        self.owners.served = None
        self.tick()
        self.tick(31.0)
        self.assertEqual(self.nogo()['reason'], 'no_site')
        self.assertEqual(self.last('objects')['objects'], [])

    def test_no_datum(self):
        self.make_site('nod', datum=False)
        bundle.save_waypoints('nod', [{'name': 'A', 'e': 1.0, 'n': 2.0, 'yaw_rad': 0.0}], 33)
        self.owners.served = 'nod'
        self.tick()
        self.assertEqual(self.last('objects')['objects'], [])
        self.assertEqual(self.nogo()['reason'], 'no_datum')
        self.assertEqual(self.last('result')['reason'], 'no_datum')
        with open(os.path.join(bundle.site_dir('nod'), 'datum.yaml'), 'w') as f:
            yaml.safe_dump(DATUM, f)
        self.tick()
        self.assertEqual(len(self.last('objects')['objects']), 1)
        self.assertTrue(self.nogo()['ok'])


class RulesTopic(Base):
    def test_activates_on_time_republishes_on_change_only(self):
        t = self.clock.t
        self.save(type='nogo', name='N', points=SQUARE, valid_from=t + 20, valid_until=t + 40)
        self.tick(5.1)
        n = self.count('rules')
        self.assertEqual(self.last('rules')['rules'], [])
        for _ in range(2):
            self.tick(5.1)
        self.assertEqual(self.count('rules'), n)
        self.tick(5.1)
        self.tick(5.1)
        self.assertEqual(self.count('rules'), n + 1)
        self.assertEqual(self.last('rules')['rules'][0]['name'], 'N')
        o = self.last('objects')['objects'][0]
        self.assertTrue(o['active'])
        for _ in range(4):
            self.tick(5.1)
        self.assertEqual(self.last('rules')['rules'], [])
        self.assertEqual(self.count('rules'), n + 2)
        self.assertFalse(self.last('objects')['objects'][0]['active'])

    def test_ticks_without_change_publish_nothing(self):
        self.nogo()
        n = len(self.out)
        for _ in range(10):
            self.tick(3.0)
        self.assertEqual(len(self.out), n)


class NoSite(Base):
    """mapping_manager's respawn says "nothing served" for a moment."""

    def setUp(self):
        super().setUp()
        self.nogo()
        self.rules_sent = self.count('rules')

    def test_transient_none_clears_nothing(self):
        self.owners.served = None
        for _ in range(5):
            self.tick(5.0)
        self.owners.served = 's'
        self.tick()
        self.assertEqual(self.count('rules'), self.rules_sent)
        self.assertEqual(len(self.last('objects')['objects']), 1)

    def test_lasting_none_clears_after_the_hold(self):
        self.owners.served = None
        self.tick()
        self.tick(29.0)
        self.assertEqual(self.count('rules'), self.rules_sent)
        self.tick(2.0)
        self.assertEqual(self.last('rules')['rules'], [])
        self.assertIsNone(self.last('rules')['site'])

    def test_never_during_a_mission(self):
        self.owners.served = None
        self.owners.busy = True
        self.tick()
        for _ in range(100):
            self.tick(10.0)
        self.assertEqual(self.count('rules'), self.rules_sent)
        self.owners.busy = False
        self.tick()
        self.assertEqual(self.last('rules')['rules'], [])

    def test_hold_restarts_after_a_site_came_back(self):
        self.owners.served = None
        self.tick()
        self.tick(25.0)
        self.owners.served = 's'
        self.tick()
        self.owners.served = None
        self.tick(1.0)
        self.tick(25.0)
        self.assertEqual(self.count('rules'), self.rules_sent)

    def test_real_switch_is_immediate_with_its_site(self):
        self.make_site('t')
        self.owners.served = 't'
        self.tick()
        r = self.last('rules')
        self.assertEqual((r['site'], r['rules']), ('t', []))

    def test_switch_out_of_a_hold(self):
        self.make_site('t')
        self.owners.served = None
        self.tick()
        self.owners.served = 't'
        self.owners.busy = True
        self.tick()
        self.assertEqual(self.last('rules')['site'], 't')

    def test_every_rules_message_has_both_keys(self):
        self.owners.served = None
        self.tick()
        self.tick(31.0)
        for k, v in self.out:
            if k == 'rules':
                self.assertIn('site', v)
                self.assertIn('rules', v)

    def test_own_start_publishes_no_empty_rules(self):
        self.owners.served = None
        self.out.clear()
        self.engine = self.new_engine()
        self.engine.tick()
        self.tick(5.0)
        self.assertEqual(self.count('rules'), 0)
        self.owners.served = 's'
        self.tick()
        self.assertEqual(len(self.last('rules')['rules']), 1)
        self.assertEqual(self.last('rules')['site'], 's')

    def test_own_start_mission_no_site_stays_quiet(self):
        self.owners.served = None
        self.owners.busy = True
        self.out.clear()
        self.engine = self.new_engine()
        self.engine.tick()
        self.tick(60.0)
        self.assertEqual(self.out, [])

    def test_ops_refused_during_the_hold(self):
        self.owners.served = None
        self.tick()
        self.assertEqual(self.nogo('M')['reason'], 'site_changed')

    def test_lost_places_file_keeps_rules(self):
        os.remove(objects.places_path('s'))
        self.tick(6.0)
        self.tick(6.0)
        self.assertEqual(self.count('rules'), self.rules_sent)

    def test_lost_datum_keeps_rules(self):
        d = os.path.join(bundle.site_dir('s'), 'datum.yaml')
        os.rename(d, d + '.x')
        self.tick(6.0)
        self.assertEqual(self.count('rules'), self.rules_sent)
        os.rename(d + '.x', d)
        self.tick(6.0)
        self.assertEqual(self.count('rules'), self.rules_sent)

    def test_last_rules_do_not_follow_to_another_site(self):
        self.make_site('t')
        with open(objects.places_path('t'), 'w') as f:
            f.write('{')
        self.owners.served = 't'
        self.tick()
        self.assertEqual(self.last('rules')['rules'], [])


class BadRuleInFile(Base):
    def setUp(self):
        super().setUp()
        self.nogo('Good')
        self.save(type='slow', name='Slow', points=[[5, 5], [7, 5], [7, 7]])
        self.logged = []
        self.engine.log = self.logged.append
        self.tick(6.0)

    def damage(self, fn):
        with open(objects.places_path('s')) as f:
            fc = json.load(f)
        fn({x['properties']['name']: x for x in fc['features']})
        with open(objects.places_path('s'), 'w') as f:
            json.dump(fc, f)

    def test_null_speed_keeps_rules_flowing(self):
        self.damage(lambda f: f['Slow']['properties']['props'].update(speed_mps=None))
        self.tick(6.0)
        m = self.last('rules')
        self.assertEqual(sorted(r['name'] for r in m['rules']), ['Good', 'Slow'])
        self.assertEqual(m['problems'][0]['reason'], 'bad_speed')
        self.assertEqual(m['problems'][0]['name'], 'Slow')
        self.assertTrue(any('bad_speed' in t for t in self.logged))

    def test_marked_nogo_and_result_report(self):
        self.damage(lambda f: f['Good'].update(geometry=None))
        self.tick(6.0)
        r = next(r for r in self.last('rules')['rules'] if r['name'] == 'Good')
        self.assertEqual((r['bad'], r['points']), ('bad_geometry', []))
        res = self.wp('A')
        self.assertTrue(res['ok'])
        self.assertTrue(any(w.startswith('bad_rule:') and w.endswith(':Good:bad_geometry')
                            for w in res['warnings']))

    def test_damaged_feature_is_no_delete_and_survives_a_write(self):
        self.damage(lambda f: f['Good'].update(geometry=None))
        n = len(self.journal())
        self.tick(6.0)
        self.assertEqual(len(self.journal()), n)
        self.assertTrue(self.nogo('Other', points=[[5, 0], [7, 0], [7, 2]])['ok'])
        with open(objects.places_path('s')) as f:
            names = [x['properties']['name'] for x in json.load(f)['features']]
        self.assertIn('Good', names)

    def test_one_failing_topic_does_not_stop_rules(self):
        real = self.engine._publish_objects

        def boom(site, now):
            raise RuntimeError('x')
        self.engine._publish_objects = boom
        self.nogo('Third', points=[[5, 0], [7, 0], [7, 2]])
        self.engine._publish_objects = real
        self.assertTrue(any(r['name'] == 'Third' for r in self.last('rules')['rules']))
        self.assertTrue(any('boom failed' in t for t in self.logged))

    def test_scan_failure_does_not_stop_the_tick(self):
        def boom():
            raise RuntimeError('x')
        self.engine.scan = boom
        self.tick(6.0)
        self.assertTrue(any('scan failed' in t for t in self.logged))


class Suggestions(Base):
    def test_ignore_and_accept(self):
        calls = []

        class Fake:
            @staticmethod
            def mtime(site):
                return 0.0

            @staticmethod
            def public(site, datum):
                return {'site': site, 'seq': 0, 'stamp': 0, 'suggestions': []}

            @staticmethod
            def mark(site, sid, state):
                calls.append((sid, state))
                return True, None

            @staticmethod
            def accept_object(site, sid, as_, datum):
                return {'type': 'landmark', 'name': 'Sud', 'points': [[1, 1]],
                        'source': 'robot'}, None
        real = objects_engine.suggestions
        objects_engine.suggestions = Fake
        try:
            self.assertTrue(self.op(op='suggestion', id='x', action='ignore')['ok'])
            self.assertTrue(self.op(op='suggestion', id='y', action='accept', **{'as': {}})['ok'])
            self.assertEqual(calls, [('x', 'ignored'), ('y', 'accepted')])
            self.assertEqual(self.obj('Sud')['source'], 'robot')
            self.assertEqual(self.journal()[-1]['op'], 'save')
        finally:
            objects_engine.suggestions = real


if __name__ == '__main__':
    unittest.main()
