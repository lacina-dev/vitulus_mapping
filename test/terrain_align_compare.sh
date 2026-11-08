#!/bin/bash
# T30 F7: registration of the windows by the offset only vs by a plane (offset + tilt), on terrain bags, with the
# consistency metric of T10 (RMS of the visit values of a cell about the cell mean; "later" without the first
# visit; p95). Bags (--clock) -> traversability_local (a vitulus_traversability build) -> two stores in one
# process (terrain_align_compare.py). Private roscore (11895), made-up datum, nothing written outside a temp dir.
#   test/terrain_align_compare.sh <devel/setup.bash> <private dir> <rate> <bag>...
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
WS_SETUP=${1:?devel/setup.bash}; PRIV=${2:?private dir}; RATE=${3:?rate}; shift 3
JOB=$(mktemp -d "$PRIV/run.XXXXXX")
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11895 ROS_LOG_DIR=$JOB/home/.ros/log
mkdir -p "$HOME/.ros"
source "$WS_SETUP"
export PYTHONPATH=$HERE/src:$PYTHONPATH
setsid roscore -p 11895 >"$JOB/roscore.log" 2>&1 & core=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
rosparam set /use_sim_time true
setsid roslaunch vitulus_traversability traversability.launch >"$JOB/trav.log" 2>&1 & trav=$!
timeout 7200 python3 "$HERE/test/terrain_align_compare.py" "$JOB" "$RATE" "$@"; rc=$?
echo "logs: $JOB"
kill -INT -- -$trav 2>/dev/null; kill -TERM -- -$core 2>/dev/null
exit $rc
