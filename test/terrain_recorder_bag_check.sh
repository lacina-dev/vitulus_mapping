#!/bin/bash
# End to end on a terrain bag: bag (--clock) -> traversability_local with /traversability/fuse (T30 C1 build)
# -> terrain_recorder (THIS tree) -> tiles. Private roscore (11892), private HOME and site root, fake
# localisation gate (the bags hold no navpvt / loc_status), made-up datum. Touches no robot process.
# Run in ROS Noetic (vitulus-dev):
#   test/terrain_recorder_bag_check.sh <C1 workspace devel/setup.bash> <bag> <private dir> [rate] [start s] [dur s]
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
WS_SETUP=${1:?C1 devel/setup.bash}; BAG=${2:?bag}; PRIV=${3:?private dir}
RATE=${4:-2.0}; START=${5:-0}; DUR=${6:-0}
JOB=$(mktemp -d "$PRIV/run.XXXXXX")
SITES=$JOB/home/.vitulus/mapping_v3
mkdir -p "$JOB/home/.ros" "$SITES/testsite"
echo '{"enabled": true}' >"$JOB/home/.vitulus/terrain_recorder.json"   # the owner's switch (T31), on for the test
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11892 ROS_LOG_DIR=$JOB/home/.ros/log
source "$WS_SETUP"
export PYTHONPATH=$HERE/src:$PYTHONPATH
printf 'utm_e: 500003.3\nutm_n: 4999992.9\nyaw_rad: 0.0\nalt: 300.0\nutm_zone: 33\nversion: 2\n' >"$SITES/testsite/datum.yaml"
echo '{"site": "testsite", "raster": "combined"}' >"$SITES/.last_served"
setsid roscore -p 11892 >"$JOB/roscore.log" 2>&1 & core=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
rosparam set /use_sim_time true
setsid roslaunch vitulus_traversability traversability.launch >"$JOB/trav.log" 2>&1 & trav=$!
rosparam load "$HERE/config/terrain_recorder.yaml" /terrain_recorder
rosparam set /terrain_recorder/sites_root "$SITES"
rosparam set /terrain_recorder/flush_s "${FLUSH_S:-20.0}"
rosparam set /terrain_recorder/stats_period 2.0
setsid python3 "$HERE/nodes/terrain_recorder" __name:=terrain_recorder >"$JOB/recorder.log" 2>&1 & rec=$!
PLAY=(-q --clock -r "$RATE" -s "$START")
[ "$DUR" != "0" ] && PLAY+=(-u "$DUR")
timeout 900 python3 "$HERE/test/terrain_recorder_bag_check.py" "$SITES/testsite" "$rec" "$BAG" "${PLAY[@]}"; rc=$?
echo "logs: $JOB"
kill -INT -- -$rec 2>/dev/null; kill -INT -- -$trav 2>/dev/null; kill -TERM -- -$core 2>/dev/null
exit $rc
