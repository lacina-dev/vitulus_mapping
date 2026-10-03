#!/usr/bin/env python3
"""ROS-free tests of bundle.program_names (vitulus-field#46): the programs
that belong to a site, used by the status and the delete warning.

Run: python3 -m pytest test/test_bundle_program_names.py   (from the package root)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import bundle  # noqa: E402


class ProgramNamesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = bundle.SITES_ROOT
        bundle.SITES_ROOT = self.tmp.name
        os.makedirs(os.path.join(self.tmp.name, 'BACK'))

    def tearDown(self):
        bundle.SITES_ROOT = self._root
        self.tmp.cleanup()

    def test_site_without_programs(self):
        self.assertEqual(bundle.program_names('BACK'), [])
        self.assertEqual(bundle.program_names('NOSUCH'), [])

    def test_names_of_saved_programs(self):
        bundle.save_programs('BACK', [{'name': 'B1', 'zone_names': ['Z']},
                                      {'name': 'BB (New)', 'zone_names': []},
                                      {'zone_names': []}])
        self.assertEqual(bundle.program_names('BACK'), ['B1', 'BB (New)'])

    def test_unreadable_file_never_raises(self):
        with open(bundle.programs_path('BACK'), 'w') as f:
            f.write('programs: [unclosed\n')
        self.assertEqual(bundle.program_names('BACK'), [])


if __name__ == '__main__':
    unittest.main()
