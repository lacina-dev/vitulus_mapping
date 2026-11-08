#!/bin/bash
# terrain_recorder keeps publishing its state while a long command runs (T31 / T30 F5): private roscore (11894),
# a small mosaic to import, the import slowed to 8 s by the test-only parameter test_cmd_delay_s.
# Run in ROS Noetic (vitulus-dev): test/terrain_recorder_busy_check.sh <private dir>
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
JOB=$(mktemp -d "${1:?a private dir}/run.XXXXXX")
SITES=$JOB/home/.vitulus/mapping_v3
mkdir -p "$JOB/home/.ros" "$SITES/testsite/terrain_import/m1"
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11894 ROS_LOG_DIR=$JOB/home/.ros/log
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$HERE/src:$PYTHONPATH
printf 'utm_e: 500000.0\nutm_n: 5000000.0\nyaw_rad: 0.0\nalt: 300.0\nutm_zone: 33\nversion: 2\n' >"$SITES/testsite/datum.yaml"
echo '{"site": "testsite", "raster": "combined"}' >"$SITES/.last_served"
python3 - "$SITES/testsite/terrain_import/m1" <<'PY'
import json, os, sys
import numpy as np
d = sys.argv[1]; h, w = 60, 80
np.savez_compressed(os.path.join(d, "teren_m1_layers.npz"), elevation=np.zeros((h, w), np.float32),
                    elevation_std=np.full((h, w), 0.01, np.float32), cost=np.zeros((h, w), np.float32),
                    reason=np.ones((h, w), np.uint8), slope_deg=np.zeros((h, w), np.float32),
                    step=np.zeros((h, w), np.float32), roughness=np.zeros((h, w), np.float32))
json.dump({"frame": "map", "yaw": 0.0, "resolution": 0.05, "width": w, "height": h, "origin_x": 0.0, "origin_y": 0.0,
           "files": {"layers": "teren_m1_layers.npz"}}, open(os.path.join(d, "teren_m1.json"), "w"))
PY
setsid roscore -p 11894 >"$JOB/roscore.log" 2>&1 & core=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
rosparam load "$HERE/config/terrain_recorder.yaml" /terrain_recorder
rosparam set /terrain_recorder/sites_root "$SITES"
rosparam set /terrain_recorder/test_cmd_delay_s 8.0
setsid python3 "$HERE/nodes/terrain_recorder" __name:=terrain_recorder >"$JOB/recorder.log" 2>&1 & rec=$!
timeout 120 python3 "$HERE/test/terrain_recorder_busy_check.py"; rc=$?
echo "logs: $JOB"
kill -INT -- -$rec 2>/dev/null; kill -TERM -- -$core 2>/dev/null
exit $rc
