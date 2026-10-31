#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.objnames.

Run: python3 -m pytest test/test_objnames.py   (from the package root)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import objnames  # noqa: E402


class ObjNamesTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(objnames.parse_request('{"from":" a ","to":" Bazén "}'), ('a', 'Bazén'))
        with self.assertRaises(ValueError):
            objnames.parse_request('nope')
        with self.assertRaises(ValueError):
            objnames.parse_request('[1]')

    def test_rename_rules(self):
        ex = ['edit_1', 'Bazén', 'DOCK']
        self.assertIsNone(objnames.check_rename('edit_1', 'Plot u sousedů', ex))
        self.assertEqual(objnames.check_rename('nic', 'x', ex), 'not_found')
        self.assertEqual(objnames.check_rename('edit_1', 'edit_1', ex), 'same')
        self.assertEqual(objnames.check_rename('edit_1', 'bazén', ex), 'exists')
        self.assertEqual(objnames.check_rename('edit_1', '', ex), 'empty')
        self.assertEqual(objnames.check_rename('edit_1', 'x' * 61, ex), 'too_long')
        self.assertEqual(objnames.check_rename('edit_1', '*', ex), 'bad_chars')
        self.assertEqual(objnames.check_rename('edit_1', 'a|||b', ex), 'bad_chars')
        self.assertEqual(objnames.check_rename('edit_1', 'a\nb', ex), 'bad_chars')
        self.assertEqual(objnames.check_rename('edit_1', 'undock', ex), 'reserved')
        self.assertEqual(objnames.check_rename('DOCK', 'Garáž', ex), 'reserved')

    def test_case_only_change_of_itself(self):
        self.assertIsNone(objnames.check_rename('Bazén', 'BAZÉN', ['Bazén']))

    def test_apostrophe_is_a_valid_name(self):
        self.assertIsNone(objnames.check_name("Robert's", []))

    def test_result(self):
        r = json.loads(objnames.result_json('edit', 'a', 'b', 'ok'))
        self.assertEqual((r['kind'], r['from'], r['to'], r['ok'], r['code']),
                         ('edit', 'a', 'b', True, 'ok'))
        self.assertTrue(json.loads(objnames.result_json('edit', 'a', 'a', 'same'))['ok'])
        self.assertFalse(json.loads(objnames.result_json('edit', 'a', 'b', 'exists'))['ok'])


if __name__ == '__main__':
    unittest.main()
