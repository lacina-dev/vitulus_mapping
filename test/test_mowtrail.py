#!/usr/bin/env python3
"""The Mow load trail without ROS: python3 test/test_mowtrail.py

A whole run plays through TrailRecorder on plain values. Load and state from the
real numbers of the run of 2026-11-01 (3400 rpm, 0.16 A free, trip 1.54 A, the
garbage sample -31966 with 91 C), storage, restart, pose jumps, site switch,
bounds of the messages, the query for the agent.
"""
import json
import math
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
from vitulus_mapping import bundle, mowtrail as mt, worldlayers as wl  # noqa: E402

T0 = 1793540000.0
STALL = 'sg=-50 off=50 cs=214 isense=%.3f dbg=1'
CFG = dict(mt.DEFAULTS)
TIMED_LOGS = {
    '[EXE] no goal progress 3.1s (dist_to_goal=1.00, moved=0.01m) -> re-window from current pos': 'rewindow',
    '[EXE] no goal progress 12.0s + obstacle on line': 'no_progress',
    '[EXE] STALL: no motion for 8.0s': 'stall',
    '[EXE] mux hold at chunk_idx=4: waiting': 'hold',
    '[DETOUR] FAILED no feasible detour around obstacle (start_idx 3)': 'detour',
    '[NAV_RECOVERY] Backing up 0.40m at 0.10m/s': 'backup',
    '[NAV_RECOVERY] Stopping blade, waiting 10s': 'recovery',
    '[BLOCKED_RECOVERY] Phase 2: raising blade to max height': 'recovery',
}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        self.addCleanup(lambda: setattr(bundle, 'SITES_ROOT', self.root))
        self.make_site('a')
        self.datum = wl.read_datum('a')
        self.out = []
        self.t = T0

    def make_site(self, site, e=500000.0, n=5500000.0, yaw=0.0):
        os.makedirs(bundle.site_dir(site), exist_ok=True)
        with open(os.path.join(bundle.site_dir(site), 'datum.yaml'), 'w') as f:
            f.write('utm_e: %r\nutm_n: %r\nyaw_rad: %r\n' % (e, n, yaw))

    def recorder(self, **cfg):
        rec = mt.TrailRecorder(lambda k, p: self.out.append((k, p)), cfg=cfg)
        rec.sync_site('a', self.datum, self.t)
        return rec

    def sent(self, kind):
        return [p for k, p in self.out if k == kind]

    def tick(self, rec, pose, status='RUN', rpm=3400, amps=0.16, temp=20.0, dt=0.5):
        self.t += dt
        rec.feed_status(self.t, status, rpm, 3400, temp, 60)
        if amps is not None:
            rec.feed_stall(self.t, STALL % amps)
        rec.step(self.t, pose)
        rec.maintain(self.t)

    def drive(self, rec, n=20, x0=0.0, step=0.15, **kw):
        for i in range(n):
            self.tick(rec, (x0 + i * step, 0.0, 0.0), **kw)
        return x0 + n * step

    def close_run(self, rec):
        rec.feed_busy(False)
        for _ in range(70):
            self.tick(rec, (0.0, 0.0, 0.0), status='OFF', rpm=0, amps=0.0)
        self.assertIsNone(rec.run)

    def rows(self, rec):
        return rec.run.view() if rec.run else None


class Signals(unittest.TestCase):
    def test_isense(self):
        self.assertAlmostEqual(mt.parse_isense(STALL % 0.64), 0.64)
        self.assertIsNone(mt.parse_isense('sg=1'))
        self.assertIsNone(mt.parse_isense(''))
        self.assertIsNone(mt.parse_isense(None))

    def test_garbage_rpm(self):
        self.assertIsNone(mt.sane_rpm(-31966))
        self.assertEqual(mt.sane_rpm(-3400), 3400)
        self.assertEqual(mt.sane_rpm(6000), 6000)

    def test_load_reaches_one_at_the_firmware_limits(self):
        self.assertAlmostEqual(mt.compute_load(1.54, 3400, 3400, True, CFG), 1.0)
        self.assertAlmostEqual(mt.compute_load(None, 850, 3400, True, CFG), 1.0)   # stall ratio
        self.assertAlmostEqual(mt.compute_load(0.16, 3400, 3400, True, CFG), 0.104, places=3)
        self.assertIsNone(mt.compute_load(None, None, 3400, True, CFG))
        self.assertAlmostEqual(mt.compute_load(0.2, 1000, 3400, False, CFG), 0.2 / 1.54)  # no sag at rest

    def test_classes(self):
        c = mt.classify
        self.assertEqual(c('BLOCKED', 1.0, False, True, True, True, CFG), mt.BLOCKED)
        self.assertEqual(c('TEMP', 0, False, True, False, False, CFG), mt.HOT)
        self.assertEqual(c('ERR', 0, False, True, False, False, CFG), mt.OVERLOAD)
        self.assertEqual(c('OFF', 0, False, True, False, True, CFG), mt.RULE)
        self.assertEqual(c('RUN', 0.1, True, True, False, False, CFG), mt.SPINUP)
        self.assertEqual(c('RUN', 0.9, False, True, False, False, CFG), mt.OVERLOAD)
        self.assertEqual(c('RUN', 0.1, False, True, True, False, CFG), mt.HOLD)
        self.assertEqual(c('RUN', 0.1, False, True, False, False, CFG), mt.CUTTING)
        self.assertEqual(c('OFF', 0, False, True, True, False, CFG), mt.HOLD)
        self.assertEqual(c('OFF', 0, False, True, False, False, CFG), mt.TRANSIT)
        self.assertEqual(c(None, None, False, False, False, False, CFG), mt.TRANSIT)

    def test_log_lines(self):
        for text, kind in TIMED_LOGS.items():
            self.assertEqual(mt.parse_log(text)[0], kind, text)
        got = mt.parse_log('[EXE] START zone=Zone A path=3/12 chunk_idx=40 len=5/9 last_chunk=False '
                           'goal=(1.00,2.00) robot=(0.00,0.00)')
        self.assertEqual(got, ('ctx', 'Zone A|3|12|40'))
        self.assertIsNone(mt.parse_log('[EXE] END chunk_idx=1 outcome=ok'))
        self.assertIsNone(mt.parse_log('Moving'))
        self.assertEqual(mt.parse_log('[MOWER_LINK] Mower link lost: robot stops, blade unknown')[0],
                         'link_lost')
        self.assertEqual(mt.parse_log('[MOWER_LINK] Blade stopped during mowing: robot stops')[0],
                         'link_lost')
        self.assertIsNone(mt.parse_log('[MOWER_LINK] hold released'))

    def test_the_smach_still_says_this(self):
        """The patterns live in another package: fail here when that one changes."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..',
                            'vitulus_navi', 'src', 'mower_unit_smach')
        if not os.path.isdir(path):
            self.skipTest('no vitulus_navi next to this package')
        text = ''.join(open(os.path.join(path, n), errors='ignore').read() for n in ('states.py', 'recovery.py'))
        for needle in ('[EXE] START zone=%s path=%d/%d chunk_idx=%d', '[EXE] no goal progress', '[EXE] STALL',
                       '[EXE] mux hold', '[NAV_RECOVERY] Backing up', '[BLOCKED_RECOVERY]', 'Detour around obstacle'):
            self.assertIn(needle, text)


class LinkLost(Base):
    """1. 11. 2026: minutes of rows with status gone read as transit."""

    def _silent(self, rec, n, x):
        for i in range(n):
            self.t += 0.5
            rec.step(self.t, (x + i * 0.15, 0.0, 0.0))
            rec.maintain(self.t)

    def _states(self, rec):
        return [mt.STATES[int(s) & (mt.BRK - 1)] for s in self.rows(rec)[:, mt.STATE]]

    def test_the_mission_said_so(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec)
        rec.feed_log(self.t, '[MOWER_LINK] Mower link lost: robot stops, blade unknown')
        self._silent(rec, 16, x)
        self.assertEqual(self._states(rec)[-1], 'link_lost')
        self.assertIn('link_lost', [e[3] for e in rec.run.events])
        self.tick(rec, (x, 0.0, 0.0))
        self.tick(rec, (x + 0.3, 0.0, 0.0))
        self.assertNotEqual(self._states(rec)[-1], 'link_lost')

    def test_a_board_restart_is_not_a_lost_link(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec)
        self._silent(rec, 16, x)
        self.assertNotIn('link_lost', self._states(rec))


class Recording(Base):
    def test_cut_and_load(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 30, rpm=3400, amps=0.32)
        rows = self.rows(rec)
        self.assertGreater(len(rows), 20)
        self.assertEqual(int(rows[-1, mt.STATE]), mt.CUTTING)
        self.assertEqual(int(rows[-1, mt.LOAD]), 21)
        self.assertEqual(int(rows[-1, mt.AMPS]), 320)
        self.assertEqual(int(rows[-1, mt.HEIGHT]), 60)

    def test_spinup_then_cutting(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.tick(rec, (0, 0, 0), status='OFF', rpm=0)
        for i in range(8):
            self.tick(rec, (0.2 * i, 0, 0), rpm=300 * (i + 1))
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.SPINUP)
        self.tick(rec, (2.0, 0, 0), rpm=3400)
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.CUTTING)

    def test_transit_blade_off(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 10, status='OFF', rpm=0, amps=0.0)
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.TRANSIT)

    def test_hold_when_not_moving(self):
        rec = self.recorder()
        rec.feed_busy(True)
        for _ in range(12):
            self.tick(rec, (1.0, 1.0, 0.0))
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.HOLD)

    def test_rule_state_goes_stale(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_rule(self.t, 'stopped_by_rule')
        self.drive(rec, 4, status='OFF', rpm=0, amps=0.0)
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.RULE)
        self.t += 10
        self.drive(rec, 4, x0=5.0, status='OFF', rpm=0, amps=0.0)
        self.assertEqual(int(self.rows(rec)[-1, mt.STATE]) & 15, mt.TRANSIT)

    def test_a_one_sample_spike_stays(self):
        rec = self.recorder()
        rec.feed_busy(True)
        for i in range(4):
            self.tick(rec, (1.0, 0.0, 0.0), temp=21.0)
        # between two ticks only the status callback runs
        rec.feed_status(self.t + 0.1, 'RUN', -31966, 3400, 90.7, 60)
        rec.feed_status(self.t + 0.2, 'RUN', 3400, 3400, 21.0, 60)
        self.tick(rec, (1.0, 0.0, 0.0), temp=21.0)
        self.tick(rec, (1.0, 0.0, 0.0), temp=21.0, dt=1.0)
        rows = self.rows(rec)
        self.assertEqual(int(rows[:, mt.TEMP].max()), 907)
        self.assertEqual(int(rows[-1, mt.TEMP]), 210)
        self.assertEqual(int(rows[:, mt.RPM].max()), 3400)      # garbage rpm is not a value
        self.assertEqual(rec.rpm, 3400)
        self.assertIn('temp', [e[3] for e in rec.run.events])

    def test_garbage_rpm_keeps_the_last(self):
        rec = self.recorder()
        rec.feed_status(self.t, 'RUN', 3400, 3400, 20.0)
        rec.feed_status(self.t, 'RUN', -31966, 3400, 20.0)
        self.assertEqual(rec.rpm, 3400)

    def test_pose_jump_breaks_the_line(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec, 6)
        self.drive(rec, 6, x0=x + 5.0)
        rows = self.rows(rec)
        breaks = np.flatnonzero(rows[:, mt.STATE] >= mt.BRK)
        self.assertEqual(len(breaks), 2)                        # the first row, the jump
        self.assertEqual(int(rows[breaks[1], mt.SPEED]), mt.NONE)

    def test_lost_pose_breaks_the_line(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec, 6)
        for _ in range(6):
            self.tick(rec, None)
        self.drive(rec, 3, x0=x + 0.2)
        self.assertEqual(int((self.rows(rec)[:, mt.STATE] >= mt.BRK).sum()), 2)

    def test_quiet_when_idle(self):
        rec = self.recorder()
        count = len(self.out)
        for i in range(100):
            self.tick(rec, None, status='OFF', rpm=0, amps=0.0)
        self.assertEqual(len(self.out), count)
        self.assertFalse(os.path.isdir(mt.run_dir('a')))

    def test_signals_in_the_row(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_cmd(0.3)
        rec.gnss = (14, 4)
        rec.attitude = (0.1, -0.05)
        rec.wheel = 2.5
        rec.feed_power(self.t + 100, 22.4, 67, 88.0)
        self.t += 99.0
        self.drive(rec, 4)
        row = self.rows(rec)[-1]
        self.assertEqual(int(row[mt.CMD]), 30)
        self.assertEqual(int(row[mt.FIX]), 4)
        self.assertEqual(int(row[mt.HACC]), 14)
        self.assertEqual(int(row[mt.ROLL]), 57)
        self.assertEqual(int(row[mt.WHEEL]), 250)
        self.assertEqual(int(row[mt.BATT_PCT]), 67)
        self.assertEqual(int(row[mt.POWER]), 88)
        self.assertEqual(int(row[mt.SPEED]), 30)

    def test_stale_battery_is_unknown(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_power(self.t - 100, 22.4, 67, 88.0)
        self.drive(rec, 3)
        self.assertEqual(int(self.rows(rec)[-1, mt.POWER]), mt.NONE)


class Events(Base):
    def test_state_edge_makes_one_event(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec, 4)
        self.drive(rec, 4, x0=x, status='BLOCKED', rpm=0, amps=0.0)
        kinds = [e[3] for e in rec.run.events]
        self.assertEqual(kinds.count('blocked'), 1)
        self.assertGreater(rec.run.events[-1][1], 0)            # at the robot

    def test_repeat_is_one(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 3)
        rec.feed_log(self.t, '[EXE] STALL: no motion for 8.0s')
        rec.feed_log(self.t + 1, '[EXE] STALL: no motion for 9.0s')
        rec.feed_log(self.t + 6, '[EXE] STALL: no motion for 9.0s')
        self.assertEqual([e[3] for e in rec.run.events].count('stall'), 2)

    def test_no_run_no_event(self):
        rec = self.recorder()
        rec.feed_log(self.t, '[EXE] STALL: no motion for 8.0s')
        rec.feed_smach(self.t, 'Detour around obstacle')
        self.assertFalse(os.path.isdir(mt.run_dir('a')))

    def test_context_in_the_card(self):
        rec = self.recorder()
        rec.feed_owner(self.t, 'gps')
        rec.feed_busy(True)
        rec.feed_program('Bigger')
        self.drive(rec, 2)
        rec.feed_log(self.t, '[EXE] START zone=Zone A path=3/12 chunk_idx=40 len=5/9 ')
        self.drive(rec, 4, x0=1.0)
        rec.feed_owner(self.t, 'rtabmap')
        self.drive(rec, 4, x0=2.0)
        rows = self.rows(rec)
        rec.request({'req': 'q', 'run': 'current', 't': int(rows[-1, mt.T])})
        rec.maintain(self.t)
        row = self.sent('detail')[-1]['row']
        self.assertEqual((row['zone'], row['path'], row['chunk'], row['owner'], row['program']),
                         ('Zone A', '3/12', 40, 'rtabmap', 'Bigger'))
        rec.request({'req': 'q', 'run': 'current', 't': 0})
        rec.maintain(self.t)
        self.assertEqual(self.sent('detail')[-1]['row']['owner'], 'gps')


class Storage(Base):
    def test_finish_and_read_back(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_program('Bigger')
        rec.feed_power(self.t, 22.4, 67, 100.0)
        self.drive(rec, 40)
        run = rec.run.id
        n = rec.run.n
        self.close_run(rec)
        self.assertFalse(os.path.exists(os.path.join(mt.run_dir('a'), run + '.jsonl')))
        entry = mt.list_runs('a')[0]
        self.assertEqual((entry['id'], entry['program'], entry['end'] is not None), (run, 'Bigger', True))
        self.assertGreater(entry['length_m'], 4)
        self.assertGreater(entry['time_mow_s'], 10)
        self.assertEqual(entry['peak_temp'], 20.0)
        self.assertGreater(mt.read_run('a', run, self.datum)[1].shape[0], n - 1)
        self.assertEqual(self.sent('runs')[-1]['active'], None)

    def test_index_lost(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 10)
        self.close_run(rec)
        os.remove(os.path.join(mt.run_dir('a'), 'index.json'))
        self.assertEqual(len(mt.list_runs('a')), 1)

    def test_retention(self):
        rec = self.recorder(max_runs=3)
        for _ in range(5):
            self.t += 1000
            rec.feed_busy(True)
            self.drive(rec, 5)
            self.close_run(rec)
        self.assertEqual(len(mt.list_runs('a')), 3)
        self.assertEqual(len([n for n in os.listdir(mt.run_dir('a')) if n.endswith('.npz')]), 3)

    def test_restart_resumes_the_same_run(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_program('Bigger')
        x = self.drive(rec, 20)
        rec.feed_log(self.t, '[EXE] STALL: no motion for 8.0s')
        run = rec.run.id
        n = rec.run.n
        self.t += rec.cfg['flush_s'] + 1
        rec.maintain(self.t)
        rec.close()
        with open(os.path.join(mt.run_dir('a'), run + '.jsonl'), 'a') as f:
            f.write('[1,2,3')                                  # torn by the crash
        self.t += 60
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 5, x0=x)
        self.assertEqual(rec.run.id, run)
        self.assertGreater(rec.run.n, n)
        self.assertEqual(int(rec.run.rows[n, mt.STATE]) >= mt.BRK, True)
        self.assertIn('stall', [e[3] for e in rec.run.events])
        self.t += rec.cfg['flush_s'] + 1
        rec.maintain(self.t)
        meta, rows, _ev, _good = mt._read_jsonl(os.path.join(mt.run_dir('a'), run + '.jsonl'))
        self.assertEqual(len(rows), rec.run.n)

    def test_late_restart_starts_a_new_run(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 12)
        run = rec.run.id
        rec.close()
        self.t += 7200
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 3)
        self.assertNotEqual(rec.run.id, run)
        self.assertEqual(mt.list_runs('a')[0]['id'], run)

    def test_restart_with_nothing_running_closes_it(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 12)
        run = rec.run.id
        rec.close()
        rec = self.recorder()
        for _ in range(40):
            self.tick(rec, None, status='OFF', rpm=0, amps=0.0)
        self.assertEqual(mt.list_runs('a')[0]['id'], run)

    def test_site_switch_closes_the_run(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 10)
        run = rec.run.id
        self.make_site('b', e=600000.0)
        rec.sync_site('b', wl.read_datum('b'), self.t)
        self.assertIsNone(rec.run)
        self.assertEqual(mt.list_runs('a')[0]['id'], run)
        self.assertEqual(self.sent('snapshot')[-1]['run'], None)
        self.assertEqual(self.sent('snapshot')[-1]['site'], 'b')

    def test_datum_change_rebases(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 10, x0=2.0, step=0.1)
        run = rec.run.id
        self.close_run(rec)
        moved = dict(self.datum, utm_e=self.datum['utm_e'] + 1.0)
        rows = mt.read_run('a', run, moved)[1]
        raw = mt.read_run('a', run, self.datum)[1]
        self.assertEqual(int(raw[0, mt.X]) - int(rows[0, mt.X]), 100)

    def test_bad_run_id(self):
        self.assertIsNone(mt.read_run('a', '../../etc/passwd'))


class Messages(Base):
    def test_append_covers_every_row_in_order(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 60)
        msgs = self.sent('trail')
        self.assertGreater(len(msgs), 20)
        self.assertLessEqual(max(len(m['r']) / m['stride'] for m in msgs), rec.cfg['max_msg_rows'])
        for a, b in zip(msgs, msgs[1:]):
            self.assertEqual(a['n'], b['from'])
        self.assertEqual(msgs[0]['stride'], mt.WIRE_STRIDE)

    def test_snapshot_for_a_late_page(self):
        rec = self.recorder(snapshot_s=1.0)
        rec.feed_busy(True)
        self.drive(rec, 40)
        snap = self.sent('snapshot')[-1]
        self.assertTrue(snap['active'])
        self.assertEqual(snap['run'], rec.run.id)
        self.assertEqual(len(snap['r']) % mt.WIRE_STRIDE, 0)
        self.assertGreaterEqual(snap['n'], len(snap['r']) // mt.WIRE_STRIDE)

    def test_decimate_keeps_what_matters(self):
        n = 100000
        rows = np.zeros((n, mt.STRIDE), np.int32)
        rows[:, mt.X] = np.arange(n)
        rows[:, mt.T] = np.arange(n)
        rows[:, mt.LOAD] = 12
        rows[:, mt.TEMP] = 210
        rows[50000, mt.TEMP] = 907                             # one sample
        rows[70000, mt.LOAD] = 95
        rows[60000, mt.STATE] = mt.CUTTING + mt.BRK
        rows[80000:80050, mt.STATE] = mt.BLOCKED
        cut, ds = mt.decimate(rows, 12000)
        self.assertLessEqual(len(cut), 12000)
        self.assertGreater(ds, 1)
        self.assertEqual(int(cut[:, mt.TEMP].max()), 907)
        self.assertEqual(int(cut[:, mt.LOAD].max()), 95)
        self.assertEqual(int((cut[:, mt.STATE] >= mt.BRK).sum()), 1)
        self.assertTrue(((cut[:, mt.STATE] & 15) == mt.BLOCKED).any())
        self.assertEqual(int(cut[-1, mt.X]), n - 1)

    def test_decimate_never_over_with_noise(self):
        n = 50000
        rows = np.zeros((n, mt.STRIDE), np.int32)
        rows[:, mt.STATE] = np.arange(n) % 3
        rows[:, mt.TEMP] = (np.arange(n) % 2) * 100
        cut, _ = mt.decimate(rows, 12000)
        self.assertLessEqual(len(cut), 12000)

    def test_snapshot_size_bound(self):
        rows = np.zeros((400000, mt.STRIDE), np.int32) - 1234
        cut, _ = mt.decimate(rows, 12000)
        # 26 wire columns since the positions (18 before: 800 kB)
        self.assertLess(len(json.dumps(mt.wire(cut))), 1200000)

    def test_past_run_and_unknown(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 10)
        run = rec.run.id
        self.close_run(rec)
        rec.request({'req': 'x', 'run': run})
        rec.request({'req': 'y', 'run': '20000101_000000'})
        rec.maintain(self.t)
        past = self.sent('past')
        self.assertEqual((past[0]['req'], past[0]['ok'], past[0]['stride']), ('x', True, mt.WIRE_STRIDE))
        self.assertEqual((past[1]['req'], past[1]['ok']), ('y', False))

    def test_events_on_the_wire_are_bounded(self):
        ev = [[i, 0, 0, 'hold' if i % 2 else 'blocked', ''] for i in range(2000)]
        flat = mt.wire_events(ev, 500)
        self.assertEqual(len(flat), 2000)
        self.assertTrue(all(k == 0 for k in flat[3::4]))      # the worst stay


class Gnss(Base):
    def rtcm(self, rec, seconds, rate=5):
        for i in range(int(seconds * rate)):
            rec.feed_rtcm(self.t - seconds + i / float(rate), True, False)

    def test_meter(self):
        m = mt.RtcmMeter()
        self.assertIsNone(m.read(T0))
        for i in range(55):
            m.feed(T0 + i / 5.5, i != 3, i == 4)
        rate, age, used, crc = m.read(T0 + 10.1)
        self.assertAlmostEqual(rate, 5.5, delta=0.3)
        self.assertLess(age, 0.5)
        self.assertEqual(crc, 1)
        self.assertEqual(used, 98)
        rate, age, used, _crc = m.read(T0 + 40)
        self.assertEqual((rate, used), (0.0, mt.NONE))
        self.assertGreater(age, 30)

    def test_in_the_row_and_on_the_wire(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.gnss = (14, 4, 19, 178)
        self.t += 10
        self.rtcm(rec, 10)
        self.drive(rec, 4)
        row = self.rows(rec)[-1]
        self.assertEqual((int(row[mt.NUM_SV]), int(row[mt.PDOP]), int(row[mt.RTCM_USED])), (19, 178, 100))
        self.assertGreater(int(row[mt.RTCM_RATE]), 30)
        self.assertLess(int(row[mt.RTCM_AGE]), 60)
        wired = mt.wire(self.rows(rec))[-mt.WIRE_STRIDE:]
        self.assertEqual(wired[12], 14)
        self.assertEqual(wired[13], int(row[mt.RTCM_AGE]))

    def test_gap_is_one_event_at_the_edge(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.rtcm(rec, 2)
        x = self.drive(rec, 4)
        self.assertNotIn('rtcm_gap', [e[3] for e in rec.run.events])            # under 3 s
        x = self.drive(rec, 4, x0=x)
        x = self.drive(rec, 4, x0=x)
        self.assertEqual([e[3] for e in rec.run.events].count('rtcm_gap'), 1)
        rec.feed_rtcm(self.t, True, False)
        self.drive(rec, 4, x0=x)
        self.assertEqual([e[3] for e in rec.run.events].count('rtcm_gap'), 1)

    def test_fix_changes_only_when_they_hold(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.gnss = (14, 4)
        x = self.drive(rec, 4)
        rec.gnss = (27, 3)
        x = self.drive(rec, 2, x0=x)                            # a second, not held
        rec.gnss = (14, 4)
        x = self.drive(rec, 4, x0=x)
        self.assertNotIn('fix_down', [e[3] for e in rec.run.events])
        rec.gnss = (27, 3)
        x = self.drive(rec, 10, x0=x)
        rec.gnss = (14, 4)
        self.drive(rec, 10, x0=x)
        kinds = [e[3] for e in rec.run.events]
        self.assertEqual((kinds.count('fix_down'), kinds.count('fix_up')), (1, 1))

    def test_rpm_dip_on_the_wire(self):
        rows = np.zeros((4, mt.STRIDE), np.int32)
        rows[:, mt.SETPOINT] = 3400
        rows[:, mt.RPM] = (3388, 3403, 0, 0)
        rows[:, mt.STATE] = (mt.CUTTING, mt.CUTTING, mt.HOLD, mt.TRANSIT)
        dip = mt.wire(rows)[11::mt.WIRE_STRIDE]
        self.assertEqual(dip, [12, -3, mt.NONE, mt.NONE])

    def test_old_run_loads(self):
        os.makedirs(mt.run_dir('a'))
        old = np.zeros((5, 20), np.int32)
        old[:, mt.X] = np.arange(5) * 10
        old[:, mt.SETPOINT] = 3400
        old[:, mt.RPM] = 3395
        meta = {'v': 1, 'run': '20261101_184310', 'site': 'a', 'program': 'P', 'start': T0,
                'datum': [self.datum['utm_e'], self.datum['utm_n'], self.datum['yaw_rad']], 'end': T0 + 9}
        mt._write_npz(os.path.join(mt.run_dir('a'), meta['run'] + '.npz'), meta, old, [])
        meta2, rows, _ev = mt.read_run('a', meta['run'], self.datum)
        self.assertEqual(rows.shape, (5, mt.STRIDE))
        self.assertEqual(int(rows[0, mt.RTCM_AGE]), mt.NONE)
        self.assertIsNone(mt._row_dict(rows[0], T0)['rtcm_rate'])
        wired = mt.wire(rows)
        self.assertEqual(wired[11], 5)                          # the dip needs no new column
        self.assertEqual(wired[13], mt.NONE)

    def test_old_open_run_resumes(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 6)
        run = rec.run.id
        rec.close()
        path = os.path.join(mt.run_dir('a'), run + '.jsonl')
        with open(path) as f:
            lines = f.readlines()
        with open(path, 'w') as f:                              # as the version before wrote it
            f.write(lines[0])
            for line in lines[1:]:
                item = json.loads(line)
                f.write(json.dumps(item[:20] if isinstance(item, list) else item) + '\n')
        self.t += 30
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 4, x0=2.0)
        self.assertEqual(rec.run.id, run)
        self.assertEqual(int(rec.run.rows[0, mt.NUM_SV]), mt.NONE)

    def test_summary(self):
        rows = np.full((3, mt.STRIDE), mt.NONE, np.int32)
        rows[:, mt.STATE] = 0
        rows[:, mt.T] = (0, 10, 20)
        rows[:, mt.X] = rows[:, mt.Y] = 0
        rows[:, mt.FIX] = (4, 4, 3)
        rows[:, mt.RTCM_RATE] = 55
        rows[:, mt.RTCM_AGE] = (3, 4, 35)
        s = mt.summarize(rows, [[1, 0, 0, 'rtcm_gap', '']])
        self.assertEqual((s['rtk_frac'], s['rtcm_gaps'], s['rtcm_rate'], s['rtcm_age_max']), (0.667, 1, 5.5, 3.5))

    def test_grid_takes_the_age_and_old_files_stay_valid(self):
        grid = wl.GnssGrid()
        grid.add(1.0, 1.0, 14, 4, 0.4)
        grid.add(1.0, 1.0, 14, 4, 3.5)
        grid.add(1.0, 1.0, 14, 4, 0.2)
        grid.add(9.0, 9.0, 14, 4)
        d = bundle.site_dir('a')
        grid.save(d, 'a', self.datum)
        got = wl.layer_query('a', 'gnss', points=[[1.0, 1.0], [9.0, 9.0]])['result']
        self.assertEqual((got[0]['rtcm_age_max_s'], got[1]['rtcm_age_max_s']), (3.5, None))
        path = os.path.join(d, 'gnss_quality.npz')
        with np.load(path) as z:
            old = dict((k, z[k]) for k in z.files if k != 'rtcm_age_ds')
        np.savez(path.replace('.npz', '_tmp.npz'), **old)
        os.replace(path.replace('.npz', '_tmp.npz'), path)
        again = wl.GnssGrid.load(d, self.datum)
        self.assertGreater(again.h, 0)
        self.assertTrue((again.a['rtcm_age_ds'] == wl.RTCM_NONE).all())
        self.assertIsNone(wl.layer_query('a', 'gnss', points=[[1.0, 1.0]])['result'][0]['rtcm_age_max_s'])


class Honesty(Base):
    def test_tracker_only_while_gps_owns(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 1)
        rec.feed_tracker(self.t, {'state': 'degenerate', 'corr_m': 1.0, 'rms': 0.05})
        rec.feed_tracker(self.t, {'state': 'gps_owns', 'corr_m': 1.0, 'rms': 0.6})
        rec.feed_tracker(self.t, {'state': 'ok', 'corr_m': 1.0, 'rms': 0.05})
        self.assertIsNone(rec._read_lidar(self.t))
        for corr in (0.10, 0.12, 0.90):
            rec.feed_tracker(self.t, {'state': 'gps_owns', 'corr_m': corr, 'rms': 0.05, 'corr_yaw_deg': 1.5})
        self.drive(rec, 1, x0=0.5, dt=0.1)
        row = self.rows(rec)[-1]
        self.assertEqual((int(row[mt.LIDAR]), int(row[mt.LIDAR_RMS]), int(row[mt.LIDAR_YAW])), (12, 50, 15))
        self.t += 5
        self.drive(rec, 1, x0=3.0)
        self.assertEqual(int(self.rows(rec)[-1, mt.LIDAR]), mt.NONE)

    def test_wheels_agree(self):
        c = mt.GpsCheck()
        self.assertIsNone(c.feed(T0, 100.0, 200.0, 0.0, 0.0, 0.0, 0.0))
        got = c.feed(T0 + 1, 100.3, 200.0, 0.3, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(got, 0.0, places=6)

    def test_rotated_frames(self):
        # odometry starts 90 degrees from east, the pose knows the true heading
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 5.0, 5.0, 1.0, 0.0)
        # the robot moves north; in odometry that is along its own yaw 1.0
        got = c.feed(T0 + 1, 0.0, 0.3, 5.0 + 0.3 * math.cos(1.0), 5.0 + 0.3 * math.sin(1.0), 1.0, math.pi / 2)
        self.assertAlmostEqual(got, 0.0, places=6)

    def test_datum_yaw(self):
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
        # pose yaw 0.1 in a map turned 0.2 from east: the robot heads 0.3 from east
        got = c.feed(T0 + 1, 0.3 * math.cos(0.3), 0.3 * math.sin(0.3), 0.3, 0.0, 0.0, 0.1, 0.2)
        self.assertAlmostEqual(got, 0.0, places=6)

    def test_lateral_jump_shows(self):
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(c.feed(T0 + 1, 0.3, 0.2, 0.3, 0.0, 0.0, 0.0), 0.2, places=6)

    def test_gps_moves_robot_stands(self):
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(c.feed(T0 + 1, 3.8, 0.0, 0.0, 0.0, 0.0, 0.0), 3.8, places=6)

    def test_turn_is_skipped(self):
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIsNone(c.feed(T0 + 1, 0.1, 0.1, 0.0, 0.0, 0.5, 0.5))
        self.assertIsNone(c.read(T0 + 1))

    def test_old_value_expires(self):
        c = mt.GpsCheck()
        c.feed(T0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        c.feed(T0 + 1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIsNotNone(c.read(T0 + 2))
        self.assertIsNone(c.read(T0 + 9))

    def test_jump_makes_an_event_and_a_row(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 2)
        rec.feed_check(self.t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.t += 1.0
        rec.feed_check(self.t, 0.9, 0.0, 0.2, 0.0, 0.0, 0.0)
        self.drive(rec, 3, x0=1.0)
        self.assertIn('gps_jump', [e[3] for e in rec.run.events])
        self.assertEqual(int(self.rows(rec)[-1, mt.JUMP]), 70)

    def test_sat_and_jam_in_the_row(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_sat(self.t + 1, 25.1, 4.32)
        rec.feed_jam(self.t + 1, 21)
        self.drive(rec, 3)
        row = self.rows(rec)[-1]
        self.assertEqual((int(row[mt.CNO]), int(row[mt.PRRES]), int(row[mt.JAM])), (251, 432, 21))

    def test_torque_totals(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.wheel, rec.wheel_max = 7.1, 2.6
        self.drive(rec, 6)
        rec.wheel = 21.0
        self.drive(rec, 3, x0=2.0)
        s = mt.summarize(self.rows(rec))
        self.assertEqual(s['wheel_peak'], 21.0)
        self.assertGreater(s['wheel_mean'], 7.0)
        self.assertEqual(mt.wire(self.rows(rec))[-mt.WIRE_STRIDE:][15], 2100)
        self.close_run(rec)
        got = wl.layer_query('a', 'mow_load')['result']
        self.assertEqual(got['wheel_peak'], 21.0)
        got = wl.layer_query('a', 'mow_load', points=[[0.15, 0.0]])['result'][0]
        self.assertEqual((got['wheel'], got['wheel_max']), (7.1, 2.6))

    def test_every_old_width_loads(self):
        os.makedirs(mt.run_dir('a'))
        for i, width in enumerate((20, 26)):
            run = '20261101_18431%d' % i
            old = np.zeros((3, width), np.int32)
            old[:, mt.WHEEL] = 710
            meta = {'v': 1, 'run': run, 'site': 'a', 'start': T0, 'end': T0 + 9,
                    'datum': [self.datum['utm_e'], self.datum['utm_n'], self.datum['yaw_rad']]}
            mt._write_npz(os.path.join(mt.run_dir('a'), run + '.npz'), meta, old, [])
            rows = mt.read_run('a', run, self.datum)[1]
            self.assertEqual(rows.shape[1], mt.STRIDE)
            self.assertEqual(int(rows[0, mt.LIDAR]), mt.NONE)
            wired = mt.wire(rows)
            self.assertEqual((wired[15], wired[16]), (710, mt.NONE))    # torque was there before


def stripes(n=2, length=6.0, gap=0.29, step=0.03):
    """A boustrophedon chunk: stripe 0 along +x at y=0, stripe 1 back at y=gap..."""
    pts = []
    for k in range(n):
        xs = np.arange(0.0, length + 1e-9, step)
        if k % 2:
            xs = xs[::-1]
        pts.extend((float(x), k * gap) for x in xs)
    return pts


class Path(unittest.TestCase):
    def test_window_keeps_the_stripe(self):
        tr = mt.PathTracker()
        tr.set_path('g', stripes())
        got = tr.project(0.5, 0.0)
        self.assertAlmostEqual(got[1], 0.0)
        # drift toward the next stripe, 0.29 m away: nearer to it than to its own
        for i in range(1, 21):
            got = tr.project(0.5 + i * 0.05, 0.01 * i)
        self.assertAlmostEqual(got[1], 0.0)
        self.assertAlmostEqual(got[3], 0.20, places=6)
        self.assertAlmostEqual(got[2], 0.0)
        # without the window the same point lands on the neighbour
        fresh = mt.PathTracker()
        fresh.set_path('g', stripes())
        self.assertAlmostEqual(fresh.project(1.5, 0.20)[1], 0.29)

    def test_xte_sign_left_is_plus(self):
        tr = mt.PathTracker()
        tr.set_path('g', [(0.0, 0.0), (5.0, 0.0)])
        self.assertAlmostEqual(tr.project(1.0, -0.1)[3], -0.1)
        tr.set_path('g', [(0.0, 0.0), (0.0, 5.0)])           # heading +y: left is -x
        got = tr.project(-0.1, 1.0)
        self.assertAlmostEqual(got[3], 0.1)
        self.assertAlmostEqual(got[2], math.pi / 2)

    def test_old_result_keeps_the_new_path(self):
        tr = mt.PathTracker()
        tr.set_path('g1', [(0.0, 0.0), (5.0, 0.0)])
        tr.set_path('g2', [(0.0, 1.0), (5.0, 1.0)])
        self.assertFalse(tr.end('g1'))
        self.assertAlmostEqual(tr.project(1.0, 0.0)[1], 1.0)
        self.assertTrue(tr.end('g2'))
        self.assertIsNone(tr.project(1.0, 0.0))

    def test_zero_length_segments(self):
        tr = mt.PathTracker()
        tr.set_path('g', [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.assertEqual(len(tr.seg), 2)
        self.assertAlmostEqual(tr.project(1.5, 0.1)[3], 0.1)
        tr.set_path('g', [(1.0, 1.0), (1.0, 1.0)])
        self.assertIsNone(tr.project(1.0, 1.0))

    def test_lost_searches_all(self):
        tr = mt.PathTracker()
        tr.set_path('g', [(float(x), 0.0) for x in range(21)])
        tr.project(0.5, 0.0)
        got = tr.project(15.0, 0.1)                           # far past the window
        self.assertAlmostEqual(got[0], 15.0)
        self.assertAlmostEqual(tr.s_last, 15.0)


class Geometry(unittest.TestCase):
    def test_lever_and_alignment(self):
        lever = (-0.16, 0.21)
        # robot at (1, 2) facing +y: the antenna is at x + 0.21 * -1 ... rotated lever
        yaw = math.pi / 2
        ant = (1.0 - 0.21, 2.0 - 0.16)
        bx, by = mt.gps_base_link(ant[0], ant[1], lever, (1.0, 2.0, yaw))
        self.assertAlmostEqual(bx, 1.0)
        self.assertAlmostEqual(by, 2.0)
        # the fix is 0.2 s old and the pose went 0.06 m since: no error appears
        bx, by = mt.gps_base_link(ant[0], ant[1], lever, (1.0, 2.06, yaw), (1.0, 2.0, yaw))
        self.assertAlmostEqual(by, 2.06)

    def test_heading(self):
        q = (0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4))     # yaw -90 deg
        yaw, acc = mt.heading_in_map(*q, cov_yaw=0.0004, datum_yaw=0.1)
        self.assertAlmostEqual(yaw, -0.1)
        self.assertAlmostEqual(acc, 0.02)


class Positions(Base):
    def run_on_path(self, offsets, path=None, **feed):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_path(self.t, 'g1', 'map', path or [(x * 0.03, 0.0) for x in range(400)])
        for i, off in enumerate(offsets):
            for key, value in feed.items():
                getattr(rec, key)(self.t + 0.5, *value)
            self.tick(rec, (i * 0.15, off, 0.0))
        return rec

    def test_xte_in_the_row_and_the_event(self):
        rec = self.run_on_path([0.0] * 5 + [0.05, 0.3, 0.32, 0.3, 0.0, 0.3, 0.0])
        rows = self.rows(rec)
        self.assertEqual(int(rows[0, mt.PATH_Y]), 0)
        self.assertEqual(int(rows[6, mt.XTE]), 30)
        self.assertEqual(int(rows[6, mt.PATH_X]), 90)
        self.assertEqual(int(rows[6, mt.PATH_YAW]), 0)
        kinds = [e[3] for e in rec.run.events]
        self.assertEqual(kinds.count('off_path'), 1)          # an edge, the second within the gap
        self.t += 10.0
        self.tick(rec, (2.0, 0.0, 0.0))
        self.tick(rec, (2.1, -0.4, 0.0))
        self.assertEqual([e[3] for e in rec.run.events].count('off_path'), 2)
        self.assertEqual(rec.run.events[-1][4], '-0.40 m')

    def test_far_from_the_path_is_none(self):
        rec = self.run_on_path([0.0, 2.5])
        self.assertEqual(int(self.rows(rec)[-1, mt.XTE]), mt.NONE)
        self.assertEqual(int(self.rows(rec)[-1, mt.PATH_X]), mt.NONE)

    def test_end_and_other_frame(self):
        logs = []
        rec = self.recorder()
        rec.log = logs.append
        rec.feed_busy(True)
        rec.feed_path(self.t, 'g1', 'map', [(0.0, 0.0), (5.0, 0.0)])
        rec.feed_path_end('g0')
        self.tick(rec, (0.5, 0.1, 0.0))
        self.assertEqual(int(self.rows(rec)[-1, mt.XTE]), 10)
        rec.feed_path(self.t, 'g2', 'odom', [(0.0, 0.0), (5.0, 0.0)])
        rec.feed_path(self.t, 'g3', 'odom', [(0.0, 0.0), (5.0, 0.0)])
        self.assertEqual(len(logs), 1)
        self.tick(rec, (0.7, 0.1, 0.0))
        self.assertEqual(int(self.rows(rec)[-1, mt.XTE]), mt.NONE)
        rec.feed_path(self.t, 'g4', '/map', [(0.0, 0.0), (5.0, 0.0)])
        rec.feed_path_end('g4')
        self.tick(rec, (0.9, 0.1, 0.0))
        self.assertEqual(int(self.rows(rec)[-1, mt.XTE]), mt.NONE)

    def test_sources_in_the_row(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_owner(self.t, 'tracker')
        rec.feed_path(self.t, 'g', 'map', [(0.0, 0.0), (9.0, 0.0)])
        self.tick(rec, (0.85, 0.02, 0.0))                     # the run opens, signals reset
        rec.feed_odom(self.t + 0.4, 3.0, -1.0, 0.5)
        rec.feed_gps(self.t + 0.4, 1.02, 0.05)
        rec.feed_heading(self.t + 0.4, math.radians(-2.0), math.radians(0.8))
        rec.feed_tracker(self.t + 0.4, {'state': 'gps_owns', 'corr_m': 0.05, 'rms': 0.05,
                                        'corr_yaw_deg': 1.0, 'corr_dx': -0.03, 'corr_dy': 0.04,
                                        'corr_dyaw_deg': -1.0})
        self.tick(rec, (1.0, 0.02, 0.0))
        row = self.rows(rec)[-1]
        self.assertEqual(tuple(int(v) for v in row[mt.GPS_X:mt.TRACK + 1]),
                         (102, 5, -20, 8, 97, 6, -10, 300, -100, 286, 0, 4, 0))
        d = mt.detail(rec.run.meta, self.rows(rec), rec.run.events, int(row[mt.T]))
        self.assertEqual((d['owner'], d['track']), ('tracker', 'gps_owns'))
        self.assertEqual((d['xte_m'], d['gps_xte_m'], d['lidar_xte_m']), (0.02, 0.05, 0.06))
        self.assertEqual((d['gps_along_m'], d['gps_across_m']), (0.02, 0.03))
        self.assertEqual((d['lidar_along_m'], d['lidar_across_m']), (-0.03, 0.04))
        self.assertEqual(d['head_err_deg'], 2.0)
        self.assertEqual((d['gps_head'], d['head_acc'], d['odom_yaw']), (-2.0, 0.8, 28.6))
        for key in ('path_x', 'path_y', 'path_yaw', 'gps_x', 'gps_y', 'lidar_x', 'lidar_y',
                    'lidar_dyaw', 'odom_x', 'odom_y', 'yaw'):
            self.assertIn(key, d)

    def test_stale_and_old_gloc(self):
        rec = self.recorder()
        rec.feed_busy(True)
        rec.feed_owner(self.t, 'someone_new')
        rec.feed_gps(self.t, 1.0, 0.0)
        self.tick(rec, (0.9, 0.0, 0.0), dt=0.0)
        rec.feed_heading(self.t + 0.3, 0.0, 31.6)             # the receiver's no-heading
        rec.feed_tracker(self.t + 0.5, {'state': 'gps_owns', 'corr_m': 0.05, 'rms': 0.05})
        rec.feed_tracker(self.t + 0.5, {'state': 'weird_new', 'corr_m': 0.05, 'rms': 0.05})
        self.tick(rec, (1.0, 0.0, 0.0), dt=0.5)
        row = self.rows(rec)[-1]
        self.assertEqual(int(row[mt.GPS_HEAD]), mt.NONE)
        self.assertEqual(int(row[mt.HEAD_ACC]), 18105)
        self.assertEqual(int(row[mt.LIDAR]), 5)               # magnitude as before
        self.assertEqual(int(row[mt.LIDAR_X]), mt.NONE)       # no vector from an old gloc
        self.assertEqual(mt.OWNERS[int(row[mt.OWNER])], 'other')
        self.assertEqual(mt.TRACK_STATES[int(row[mt.TRACK])], 'other')
        self.tick(rec, (1.5, 0.0, 0.0), dt=2.0)
        row = self.rows(rec)[-1]
        self.assertEqual((int(row[mt.GPS_X]), int(row[mt.HEAD_ACC])), (mt.NONE, mt.NONE))

    def test_wire_and_head_err(self):
        rows = np.full((2, mt.STRIDE), mt.NONE, np.int32)
        rows[:, mt.STATE] = 0
        rows[0, mt.YAW], rows[0, mt.GPS_HEAD] = 1790, -1790   # 179 against -179: 2 deg
        rows[0, mt.XTE] = -12
        rows[0, mt.PATH_X:mt.PATH_Y + 1] = (5, 6)
        rows[0, mt.GPS_X:mt.GPS_Y + 1] = (7, 8)
        rows[0, mt.LIDAR_X:mt.LIDAR_Y + 1] = (9, 10)
        rows[1, mt.YAW] = 100
        wired = mt.wire(rows, np.array([False, False]))
        self.assertEqual(mt.WIRE_STRIDE, 26)
        self.assertEqual(len(wired), 2 * 26)
        self.assertEqual(wired[18:26], [-12, 5, 6, 7, 8, 9, 10, -20])
        self.assertEqual(wired[26 + 25], mt.NONE)
        self.assertEqual(mt.wire(rows)[25], mt.NONE)          # alone, the first row has no rate

    def test_summary(self):
        rows = np.full((4, mt.STRIDE), mt.NONE, np.int32)
        rows[:, mt.STATE] = 0
        rows[:, mt.T] = np.arange(4)
        rows[:, mt.X] = rows[:, mt.PATH_X] = rows[:, mt.GPS_X] = np.arange(4) * 10
        rows[:, mt.Y] = rows[:, mt.PATH_Y] = 0
        rows[:, mt.PATH_YAW] = 0
        rows[:, mt.XTE] = (2, -4, 6, -8)
        rows[:, mt.GPS_Y] = (-1, 3, -5, 7)
        rows[:, mt.T] = np.arange(4) * 10                     # a second apart
        rows[:, mt.YAW] = (10, 20, 30, 40)                    # 1 deg/s, slow
        rows[:, mt.GPS_HEAD] = (0, 0, 0, 0)
        rows[:, mt.FIX] = 4
        s = mt.summarize(rows)
        self.assertEqual((s['xte_med'], s['xte_p95']), (5, 7))
        self.assertEqual((s['gps_xte_med'], s['gps_xte_p95']), (4, 6))
        self.assertEqual((s['head_med'], s['head_p95']), (30, 39))   # the first row has no rate
        # GPS against the path counts only on RTK fixed
        rows[:2, mt.FIX] = 3
        self.assertEqual(mt.summarize(rows)['gps_xte_med'], 6)
        # a fast turn: no heading error there
        rows[2, mt.YAW] = 400                                  # 37 deg in a second
        s = mt.summarize(rows)
        self.assertEqual((s['head_med'], s['head_p95']), (20, 20))
        empty = mt.summarize(rows[:0])
        self.assertIsNone(empty['xte_med'])

    def test_every_old_width_loads(self):
        os.makedirs(mt.run_dir('a'))
        for i, width in enumerate((20, 26, 34)):
            run = '20261108_10000%d' % i
            old = np.zeros((3, width), np.int32)
            old[:, mt.X] = 100
            meta = {'v': 3, 'run': run, 'site': 'a', 'start': T0, 'end': T0 + 9,
                    'datum': [self.datum['utm_e'], self.datum['utm_n'], self.datum['yaw_rad']]}
            mt._write_npz(os.path.join(mt.run_dir('a'), run + '.npz'), meta, old, [])
            rows = mt.read_run('a', run, self.datum)[1]
            self.assertEqual(rows.shape, (3, mt.STRIDE))
            self.assertTrue((rows[:, mt.PATH_X:] == mt.NONE).all())
            self.assertEqual(len(mt.wire(rows)), 3 * mt.WIRE_STRIDE)
            self.assertIsNone(mt.detail(meta, rows, [], 0)['xte_m'])
            # the open run of an older driver, as jsonl
            jrun = '20261108_11000%d' % i
            with open(os.path.join(mt.run_dir('a'), jrun + '.jsonl'), 'w') as f:
                f.write(json.dumps(dict(meta, run=jrun)) + '\n')
                f.write(json.dumps([7] * width) + '\n')
                f.write(json.dumps([7] * mt.STRIDE) + '\n')
            rows = mt.read_run('a', jrun, self.datum)[1]
            self.assertEqual(rows.shape, (2, mt.STRIDE))
            self.assertEqual(int(rows[0, mt.PATH_X]), mt.NONE)
            self.assertEqual(int(rows[1, mt.PATH_X]), 7)

    def test_rebase_moves_positions(self):
        rows = np.full((1, mt.STRIDE), mt.NONE, np.int32)
        rows[0, mt.X], rows[0, mt.Y] = 100, 0
        rows[0, mt.GPS_X], rows[0, mt.GPS_Y] = 100, 0
        rows[0, mt.YAW] = 0
        rows[0, mt.ODOM_X] = 55
        stored = {'utm_e': 0.0, 'utm_n': 0.0, 'yaw_rad': 0.0}
        datum = {'utm_e': 0.0, 'utm_n': 0.0, 'yaw_rad': math.pi / 2}
        out = mt._rebase(rows, stored, datum)
        self.assertEqual((int(out[0, mt.GPS_X]), int(out[0, mt.GPS_Y])), (0, -100))
        self.assertEqual(int(out[0, mt.YAW]), -900)
        self.assertEqual(int(out[0, mt.PATH_X]), mt.NONE)
        self.assertEqual(int(out[0, mt.ODOM_X]), 55)

    def test_gps_gate(self):
        rec = self.recorder()
        self.assertTrue(rec.gps_usable(3, 900, 50.0, 14.9))
        self.assertTrue(rec.gps_usable(4, 14, 50.0, 14.9))
        self.assertFalse(rec.gps_usable(2, 14, 50.0, 14.9))
        self.assertFalse(rec.gps_usable(5, 14, 50.0, 14.9))             # time only
        self.assertFalse(rec.gps_usable(3, 1001, 50.0, 14.9))
        self.assertFalse(rec.gps_usable(3, 14, 0.0, 0.0))
        rec = self.recorder(gps_max_hacc_m=2.0)
        self.assertTrue(rec.gps_usable(3, 1500, 50.0, 14.9))

    def test_fix_time_less_latency(self):
        self.assertAlmostEqual(self.recorder().fix_time(100.0), 99.9)
        self.assertAlmostEqual(self.recorder(gps_latency_s=0.25).fix_time(100.0), 99.75)

    def test_past_the_end_is_none(self):
        tr = mt.PathTracker()
        tr.set_path('g', [(0.0, 0.0), (5.0, 0.0)])
        got = tr.project(6.0, 0.0)
        self.assertAlmostEqual(got[3], 0.0)                    # on the line of the path
        self.assertAlmostEqual(got[4], 1.0)
        rec = self.run_on_path([0.0, 0.0], path=[(0.0, 0.0), (0.2, 0.0)])
        self.assertEqual(int(self.rows(rec)[0, mt.XTE]), 0)
        self.t += 1.0
        self.tick(rec, (2.5, 0.0, 0.0))                        # 2.3 m past the end, xte 0
        self.assertEqual(int(self.rows(rec)[-1, mt.PATH_X]), mt.NONE)
        self.assertEqual(int(self.rows(rec)[-1, mt.XTE]), mt.NONE)

    def test_projection_on_an_old_path_does_not_steer_the_new(self):
        tr = mt.PathTracker()
        tr.set_path('g1', [(float(x), 0.0) for x in range(21)])
        tr.project(15.0, 0.0)
        old = tr._path
        tr.set_path('g2', stripes())
        self.assertIsNone(tr.s_last)                           # bound to g1
        # a project() that started on g1 finishes after the swap: it must not store
        tr._s = (old, 15.0)
        self.assertIsNone(tr.s_last)
        self.assertAlmostEqual(tr.project(0.5, 0.0)[1], 0.0)
        self.assertAlmostEqual(tr.s_last, 0.5)

    def test_set_and_end_from_threads(self):
        import threading
        tr = mt.PathTracker()
        line = [(0.0, 0.0), (5.0, 0.0)]
        stop, bad = [], []

        def loop():
            while not stop:
                got = tr.project(1.0, 0.1)
                if not (got is None or (len(got) == 5 and abs(got[3] - 0.1) < 1e-9)):
                    bad.append(got)
        th = threading.Thread(target=loop)
        th.start()
        for i in range(300):
            tr.set_path('g%d' % i, line)
            tr.end('g%d' % (i - 1))                             # an older result: no effect
        stop.append(1)
        th.join()
        self.assertEqual(bad, [])
        self.assertEqual(tr.goal_id, 'g299')
        self.assertIsNotNone(tr.project(1.0, 0.1))

    def test_lidar_when_gps_good_but_corr_too_big(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 1)
        msg = {'state': 'corr_too_big', 'corr_m': 0.9, 'rms': 0.05, 'corr_yaw_deg': 2.0,
               'corr_dx': 0.0, 'corr_dy': 0.9, 'corr_dyaw_deg': 2.0}
        rec.feed_tracker(self.t, dict(msg, gps_good=False))
        rec.feed_tracker(self.t, dict(msg, rms=0.5, gps_good=True))   # rms gate stays
        self.assertIsNone(rec._read_lidar(self.t))
        rec.feed_tracker(self.t, dict(msg, gps_good=True))
        self.drive(rec, 1, x0=0.5, dt=0.1)
        row = self.rows(rec)[-1]
        self.assertEqual((int(row[mt.LIDAR]), int(row[mt.LIDAR_X]), int(row[mt.LIDAR_Y]),
                          int(row[mt.LIDAR_DYAW])), (90, 50, 90, 20))
        self.assertEqual(mt.TRACK_STATES[int(row[mt.TRACK])], 'corr_too_big')

    def test_heading_must_be_fresh(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 1)
        rec.feed_heading(self.t, 0.1, 0.01)
        self.tick(rec, (0.5, 0.0, 0.0), dt=0.3)
        self.assertNotEqual(int(self.rows(rec)[-1, mt.GPS_HEAD]), mt.NONE)
        self.tick(rec, (1.0, 0.0, 0.0), dt=0.3)                # 0.6 s old
        self.assertEqual(int(self.rows(rec)[-1, mt.GPS_HEAD]), mt.NONE)

    def test_no_head_err_in_a_turn(self):
        rows = np.full((3, mt.STRIDE), mt.NONE, np.int32)
        rows[:, mt.STATE] = 0
        rows[:, mt.T] = (0, 5, 10)                             # 0.5 s apart
        rows[:, mt.GPS_HEAD] = 0
        rows[:, mt.YAW] = (0, 50, 250)                         # 0.17 rad/s, then 0.7 rad/s
        self.assertEqual(mt.wire(rows)[25::26], [mt.NONE, 50, mt.NONE])
        # an append message gets the row before
        self.assertEqual(mt.wire(rows[1:], mt._turning(rows[1:], 0.3, rows[0]))[25::26], [50, mt.NONE])
        self.assertEqual(mt.wire(rows, mt._turning(rows, 1.0))[25::26], [mt.NONE, 50, 250])
        # a line break: no rate
        rows[1, mt.STATE] += mt.BRK
        self.assertEqual(mt.wire(rows)[25::26][1], mt.NONE)

    def test_live_messages_use_the_row_before(self):
        rec = self.recorder()
        rec.feed_busy(True)
        for i in range(6):
            rec.feed_heading(self.t + 0.5, 0.0, 0.01)
            self.tick(rec, (i * 0.15, 0.0, 0.0))
        heads = [v for m in self.sent('trail') for v in m['r'][25::26]]
        self.assertEqual(heads[0], mt.NONE)                     # the very first row only
        self.assertTrue(all(v == 0 for v in heads[1:]))
        self.assertGreater(len(heads), 2)

    def test_off_path_at_the_row(self):
        rec = self.run_on_path([0.0, 0.0, 0.4])
        e = [e for e in rec.run.events if e[3] == 'off_path'][0]
        self.assertEqual((e[1], e[2]), (30, 40))

    def test_old_odometry_is_none(self):
        rec = self.recorder()
        rec.feed_busy(True)
        self.drive(rec, 1)
        rec.feed_odom(self.t, 1.0, 2.0, 0.0)
        self.tick(rec, (0.5, 0.0, 0.0), dt=0.5)
        self.assertEqual(int(self.rows(rec)[-1, mt.ODOM_X]), 100)
        self.tick(rec, (1.0, 0.0, 0.0), dt=0.8)
        self.assertEqual(int(self.rows(rec)[-1, mt.ODOM_X]), mt.NONE)

    def test_new_event_goes_last(self):
        self.assertEqual(mt.EVENTS[-1], 'off_path')
        self.assertEqual(mt.EVENTS.index('gps_jump'), 16)


class Query(Base):
    def make(self):
        rec = self.recorder()
        rec.feed_busy(True)
        x = self.drive(rec, 30, amps=0.2)
        x = self.drive(rec, 8, x0=x, amps=1.0)
        self.drive(rec, 10, x0=x, amps=0.2)
        self.close_run(rec)

    def test_where_it_was_heavy(self):
        self.make()
        got = wl.layer_query('a', 'mow_load')
        self.assertTrue(got['ok'])
        heavy = got['result']['heavy']
        self.assertGreaterEqual(heavy[0]['load'], 0.6)
        self.assertTrue(4.0 < heavy[0]['x'] < 7.0)

    def test_polygon_and_points(self):
        self.make()
        got = wl.layer_query('a', 'mow_load', polygon=[[-1, -1], [2, -1], [2, 1], [-1, 1]])
        self.assertLess(got['result']['peak_load'], 0.3)
        got = wl.layer_query('a', 'mow_load', points=[[1.0, 0.0], [50.0, 50.0]])
        self.assertEqual(got['result'][0]['state'], 'cutting')
        self.assertIsNone(got['result'][1])

    def test_nothing_recorded(self):
        got = wl.layer_query('a', 'mow_load')
        self.assertTrue(got['ok'])
        self.assertEqual(got['result']['rows'], 0)

    def test_other_layers_still_ask_for_a_place(self):
        self.assertFalse(wl.layer_query('a', 'slope')['ok'])


if __name__ == '__main__':
    unittest.main()
