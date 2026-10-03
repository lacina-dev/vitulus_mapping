#!/usr/bin/env python3
"""ROS-free tests of the layer include/exclude rules in
vitulus_mapping.composite (vitulus-field#47).

Run: python3 -m pytest test/test_composite_layers.py   (from the package root)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import composite  # noqa: E402

A = 'direct_20260918_173928'
B = 'direct_20260918_175202'
C = 'direct_20260918_181108'


def _add_session(site_dir, name):
    rdir = os.path.join(site_dir, 'rasters', name)
    os.makedirs(rdir)
    open(os.path.join(rdir, name + '.pgm'), 'w').close()


class LayerRulesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.site = os.path.join(self.tmp.name, 'LADA')
        for name in (A, B, C):
            _add_session(self.site, name)
        os.makedirs(os.path.join(self.site, 'rasters',
                                 composite.COMBINED_NAME))

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_enabled_by_default(self):
        self.assertEqual(composite.enabled_versions(self.site), [A, B, C])
        for name in (A, B, C):
            self.assertFalse(composite.is_last_enabled(self.site, name))

    def test_last_enabled_with_excluded_siblings(self):
        composite.set_layer_enabled(self.site, B, False)
        composite.set_layer_enabled(self.site, C, False)
        self.assertEqual(composite.enabled_versions(self.site), [A])
        self.assertTrue(composite.is_last_enabled(self.site, A))
        # an excluded session can always go
        self.assertFalse(composite.is_last_enabled(self.site, B))

    def test_two_enabled_is_not_last(self):
        composite.set_layer_enabled(self.site, C, False)
        self.assertFalse(composite.is_last_enabled(self.site, A))
        self.assertFalse(composite.is_last_enabled(self.site, B))

    def test_only_session_of_site_is_not_guarded(self):
        site = os.path.join(self.tmp.name, 'T1')
        _add_session(site, A)
        self.assertFalse(composite.is_last_enabled(site, A))

    def test_stale_exclude_entry_is_ignored(self):
        # the exclude list may still name a session that was deleted
        composite.set_layer_enabled(self.site, 'direct_20200101_000000', False)
        self.assertEqual(composite.enabled_versions(self.site), [A, B, C])
        self.assertFalse(composite.is_last_enabled(self.site, A))


if __name__ == '__main__':
    unittest.main()
