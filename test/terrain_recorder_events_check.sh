#!/bin/bash
# T30 P4: several terrain bags in a row (each its own mowing program = session, /mower_smach/busy from the bag)
# -> traversability_local (/traversability/fuse, a build of vitulus_traversability) -> terrain_recorder (THIS tree)
# -> tiles + batch log; then the checks of terrain_recorder_events_check.py against offline mosaics.
# Private roscore (11893), private HOME / site root, fake localisation gate, made-up datum (yaw 0: map = UTM - origin).
#   test/terrain_recorder_events_check.sh <devel/setup.bash> <private dir> <rate> <mosaic dirs, comma> <bag>...
# REF=/licp/odom: reference odometry for the smooth-correction detector, counted only (the bags hold no
# /odometry/ekf_wheel_odom); empty = detector off.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
WS_SETUP=${1:?devel/setup.bash}; PRIV=${2:?private dir}; RATE=${3:?rate}; MOSAICS=${4:?mosaics}; shift 4
JOB=$(mktemp -d "$PRIV/run.XXXXXX")
SITES=$JOB/home/.vitulus/mapping_v3
mkdir -p "$JOB/home/.ros" "$SITES/testsite"
echo '{"enabled": true}' >"$JOB/home/.vitulus/terrain_recorder.json"   # the owner's switch (T31), on for the test
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11893 ROS_LOG_DIR=$JOB/home/.ros/log
source "$WS_SETUP"
export PYTHONPATH=$HERE/src:$PYTHONPATH
printf 'utm_e: 500000.0\nutm_n: 5000000.0\nyaw_rad: 0.0\nalt: 300.0\nutm_zone: 33\nversion: 2\n' >"$SITES/testsite/datum.yaml"
echo '{"site": "testsite", "raster": "combined"}' >"$SITES/.last_served"
setsid roscore -p 11893 >"$JOB/roscore.log" 2>&1 & core=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
rosparam set /use_sim_time true
setsid roslaunch vitulus_traversability traversability.launch >"$JOB/trav.log" 2>&1 & trav=$!
rosparam load "$HERE/config/terrain_recorder.yaml" /terrain_recorder
rosparam set /terrain_recorder/sites_root "$SITES"
rosparam set /terrain_recorder/stats_period 2.0
rosparam set /terrain_recorder/batch_log "$JOB/batches.csv"
rosparam set /terrain_recorder/odom_ref_topic "${REF:-}"
rosparam set /terrain_recorder/corr_enforce false
setsid python3 "$HERE/nodes/terrain_recorder" __name:=terrain_recorder >"$JOB/recorder.log" 2>&1 & rec=$!
timeout 7200 python3 "$HERE/test/terrain_recorder_events_check.py" "$SITES/testsite" "$rec" "$JOB/batches.csv" \
  "$MOSAICS" "$RATE" "$@"; rc=$?
echo "logs: $JOB"
kill -INT -- -$rec 2>/dev/null; kill -INT -- -$trav 2>/dev/null; kill -TERM -- -$core 2>/dev/null
exit $rc
