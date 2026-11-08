#!/usr/bin/env python3
"""Driver of terrain_recorder_busy_check.sh: the state goes on every ~2 s while an import
(slowed to 8 s) holds the recorder, with last_cmd running, then the answer arrives."""

import json
import sys
import time

import rospy
from std_msgs.msg import String

fails = []


def check(cond, text):
    print(("ok   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


rospy.init_node("busy_check", anonymous=True)
states, results = [], []
rospy.Subscriber("/terrain_recorder/state", String, lambda m: states.append((time.time(), json.loads(m.data))))
rospy.Subscriber("/terrain_recorder/cmd_result", String, lambda m: results.append((time.time(), json.loads(m.data))))
pub = rospy.Publisher("/terrain_recorder/cmd", String, queue_size=2)
t_end = time.time() + 20
while time.time() < t_end and not (states and states[-1][1].get("imports")):
    time.sleep(0.2)
check(bool(states) and states[-1][1].get("imports"), "state with the import to run")
time.sleep(4.5)
idle = [t for t, _ in states]
gaps = [b - a for a, b in zip(idle[-3:], idle[-2:] + idle[-1:])][:-1]
check(bool(gaps) and max(gaps) <= 2.6, "idle state every ~2 s (gaps %s)" % [round(g, 2) for g in gaps])
t0 = time.time()
pub.publish(String(data=json.dumps({"id": "imp1", "cmd": "import", "source": "mosaic", "name": "m1",
                                    "mode": "merge"})))
while time.time() < t0 + 20 and not results:
    time.sleep(0.1)
t1 = results[0][0] if results else time.time()
during = [(t, s) for t, s in states if t0 < t < t1]
check(t1 - t0 >= 7.5, "the import took the test delay (%.1f s)" % (t1 - t0))
check(any((s.get("last_cmd") or {}).get("running") for _, s in during), "state.last_cmd running during the import")
times = [t0] + [t for t, _ in during] + [t1]
worst = max(b - a for a, b in zip(times[:-1], times[1:]))
check(len(during) >= 3 and worst <= 3.0, "state went on during the 8 s command: %d messages, longest gap %.2f s"
      % (len(during), worst))
check(bool(results) and results[0][1].get("ok") and results[0][1]["result"].get("imported_cells") == 4800,
      "the answer: ok, 4800 cells")
time.sleep(2.5)
check((states[-1][1].get("last_cmd") or {}).get("running") is False, "state after: last_cmd running false")
sys.exit(1 if fails else 0)
