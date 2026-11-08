#!/bin/bash
# terrain_recorder against a private roscore (11891): fake loc_status, navpvt and /traversability/fuse.
# Fresh private HOME and site root per run, code of THIS tree; touches no robot process.
# Run in ROS Noetic (vitulus-dev): test/terrain_recorder_ros_check.sh <private dir>
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
JOB=$(mktemp -d "${1:?a private dir}/run.XXXXXX")
SITES=$JOB/home/.vitulus/mapping_v3
mkdir -p "$JOB/home/.ros" "$SITES/testsite"
echo '{"enabled": true}' >"$JOB/home/.vitulus/terrain_recorder.json"   # the owner's switch (T31), on for the test
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11891 ROS_LOG_DIR=$JOB/home/.ros/log
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$HERE/src:$PYTHONPATH
printf 'utm_e: 500003.3\nutm_n: 4999992.9\nyaw_rad: 0.01\nalt: 300.0\nutm_zone: 33\nversion: 2\n' >"$SITES/testsite/datum.yaml"
echo '{"site": "testsite", "raster": "combined"}' >"$SITES/.last_served"
pids=()
setsid roscore -p 11891 >"$JOB/roscore.log" 2>&1 & pids+=($!)
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
rosparam load "$HERE/config/terrain_recorder.yaml" /terrain_recorder
rosparam set /terrain_recorder/sites_root "$SITES"
rosparam set /terrain_recorder/flush_s 3.0
rosparam set /terrain_recorder/stats_period 1.0
setsid python3 "$HERE/nodes/terrain_recorder" __name:=terrain_recorder >"$JOB/recorder.log" 2>&1 & rec=$!
timeout 120 python3 "$HERE/test/terrain_recorder_ros_check.py" "$SITES/testsite" "$rec"; rc=$?
echo "logs: $JOB"
kill -TERM -- -$rec 2>/dev/null
for p in "${pids[@]}"; do kill -TERM -- -$p 2>/dev/null; done
exit $rc
