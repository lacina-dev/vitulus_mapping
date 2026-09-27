#!/usr/bin/env python3
"""ROS-free tests of vitulus_mapping.presets (vitulus-field#23).

Run: python3 -m pytest test/test_presets.py   (from the package root)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import presets  # noqa: E402


class PresetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'sub', 'direct_presets.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_builtins_only_without_file(self):
        self.assertEqual(presets.load_custom(self.path), {})
        lst = presets.presets_list({})
        self.assertEqual([p['name'] for p in lst],
                         list(presets.BUILTIN_PRESETS))
        self.assertTrue(all(p['builtin'] for p in lst))

    def test_builtin_keys_are_preset_keys(self):
        for name, vals in presets.BUILTIN_PRESETS.items():
            self.assertTrue(set(vals) <= set(presets.PRESET_KEYS), name)

    def test_custom_shadows_builtin_and_is_appended(self):
        custom = {'Lawn tall': {'min_hits': 9}, 'Mine': {'hit_inc': 0.5}}
        lst = presets.presets_list(custom)
        names = [p['name'] for p in lst]
        self.assertEqual(names.count('Lawn tall'), 1)
        self.assertEqual(names[-2:], ['Lawn tall', 'Mine'])
        tall = [p for p in lst if p['name'] == 'Lawn tall'][0]
        self.assertFalse(tall['builtin'])
        self.assertEqual(tall['values'], {'min_hits': 9})

    def test_save_load_roundtrip_creates_dir(self):
        custom = {'Mine': {'hit_inc': 0.5}}
        presets.save_custom(self.path, custom)
        self.assertEqual(presets.load_custom(self.path), custom)

    def test_non_dict_file_is_ignored(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'w') as f:
            json.dump(['not', 'a', 'dict'], f)
        self.assertEqual(presets.load_custom(self.path), {})

    def test_invalid_json_raises(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'w') as f:
            f.write('{broken')
        with self.assertRaises(ValueError):
            presets.load_custom(self.path)


if __name__ == '__main__':
    unittest.main()
