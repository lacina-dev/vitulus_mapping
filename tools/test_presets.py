#!/usr/bin/env python3
"""Unit tests for vitulus_mapping.presets (vitulus-field#23). Run:

    python3 tools/test_presets.py

No ROS required. Exercises: builtins alone, a custom preset shadowing a
builtin of the same name, extra custom presets appended, and the JSON file
round trip (missing / corrupt / non-dict file -> {}).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping import presets   # noqa: E402

FAIL = []


def check(cond, msg):
    print(('  ok  ' if cond else ' FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


def names(lst):
    return [p['name'] for p in lst]


print('builtins only')
lst = presets.presets_list({})
check(names(lst) == list(presets.BUILTIN_PRESETS), 'all builtins, in order')
check(len(lst) == 5, '5 builtins')
check(all(p['builtin'] for p in lst), 'all flagged builtin')
check(presets.presets_list(None) == lst, 'None == {}')
check(all(set(p['values']) <= set(presets.PRESET_KEYS) for p in lst),
      'builtin values use only PRESET_KEYS')

print('custom shadows builtin of the same name')
custom = {'Lawn tall': {'min_hits': 9}, 'My garden': {'hit_inc': 0.3}}
lst = presets.presets_list(custom)
check(names(lst).count('Lawn tall') == 1, "'Lawn tall' listed once")
lt = [p for p in lst if p['name'] == 'Lawn tall'][0]
check(lt['builtin'] is False and lt['values'] == {'min_hits': 9},
      "'Lawn tall' is the custom one")
check(len(lst) == 6, '4 builtins + 2 custom')
check(names(lst)[-2:] == ['Lawn tall', 'My garden'], 'custom ones last')

print('file round trip')
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'sub', 'direct_presets.json')
    logs = []
    log = lambda f, *a: logs.append(f % a)   # noqa: E731
    check(presets.load_custom(path, log=log) == {}, 'missing file -> {}')
    check(presets.save_custom(path, custom, log=log), 'save creates dirs')
    check(presets.load_custom(path, log=log) == custom, 'load == saved')
    with open(path, 'w') as f:
        f.write('[1, 2]')
    check(presets.load_custom(path, log=log) == {}, 'non-dict -> {}')
    with open(path, 'w') as f:
        f.write('{broken')
    check(presets.load_custom(path, log=log) == {}, 'corrupt -> {}')
    check(len(logs) == 1 and 'could not load' in logs[0], 'corrupt is logged')

print()
print('FAILED: %d' % len(FAIL) if FAIL else 'ALL OK')
sys.exit(1 if FAIL else 0)
