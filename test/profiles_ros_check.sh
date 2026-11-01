#!/bin/bash
# Private roscore (11890), fresh private HOME per run, code of THIS tree; touches no robot process.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
WS=/home/vitulus/catkin_ws
JOB=$(mktemp -d "${1:?a private dir under a jobs tmp}/run.XXXXXX")
mkdir -p "$JOB/home/.ros" "$JOB/home/.vitulus/mapping_v3/siteA/rasters" "$JOB/home/.vitulus/mapping_v3/siteB/rasters" "$JOB/home/.vitulus/depth_obstacles"
export HOME=$JOB/home ROS_HOME=$JOB/home/.ros ROS_MASTER_URI=http://localhost:11890 ROS_LOG_DIR=$JOB/home/.ros/log
source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=$HERE:$WS/src/vitulus_depth_obstacles:/opt/ros/noetic/share
export PYTHONPATH=$HERE/src:$WS/devel/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$WS/devel/lib:$LD_LIBRARY_PATH
B=$HOME/.vitulus/mapping_v3
python3 - <<PY
import json, os
A={'cam_cluster_radius_m':0.2,'cam_min_cluster_pts':4,'cam_obstacle_max_h':0.8,'cam_obstacle_min_h':0.25,'free_thresh':-0.4,'hi_clamp':3.5,'hit_inc':0.5,'lidar_free_dec':0.08,'lidar_free_no_hit':True,'lo_clamp':-2.0,'min_hits':4,'miss_dec':0.3,'occ_erode_factor':0.3,'occ_thresh':0.8,'use_cam_free':True,'use_cam_obstacle':True,'use_lidar':True}
PB={'cam_obstacle_min_h':0.10,'cam_obstacle_max_h':1.5,'cam_min_cluster_pts':1,'cam_cluster_radius_m':0.15,'lidar_min_intensity':0.0,'hit_inc':0.7,'miss_dec':0.4,'occ_thresh':0.85,'free_thresh':-0.4,'min_hits':2,'lidar_free_dec':0.2,'occ_erode_factor':0.5}
json.dump(A,open('$B/siteA/direct_overrides.json','w')); json.dump(PB,open('$B/siteB/direct_overrides.json','w'))
json.dump({'version':1,'profiles':{'nav':{'max_depth':3.05}}},open('$HOME/.vitulus/depth_obstacles/settings.json','w'))
for s in 'AB':
    d='$B/site%s/rasters/r1'%s
    os.makedirs(d)
    open(d+'/r1.pgm','wb').write(b'P5\n4 4\n255\n'+bytes([254]*16))
    open(d+'/r1.yaml','w').write('image: r1.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
PY
pids=()
setsid roscore -p 11890 >"$JOB/roscore.log" 2>&1 & pids+=($!)
sleep 3
rosparam load $WS/src/vitulus_depth_obstacles/config/d435_ros1.yaml /depth_obstacles
setsid $WS/devel/lib/vitulus_depth_obstacles/depth_obstacles __name:=depth_obstacles depth/image:=/cam/depth depth/camera_info:=/cam/info >"$JOB/depth.log" 2>&1 & pids+=($!)
setsid python3 $HERE/nodes/mapping_manager __name:=mapping_manager >"$JOB/manager.log" 2>&1 & pids+=($!)
sleep 2
timeout 200 python3 $HERE/test/profiles_ros_check.py; rc=$?
echo "logs: $JOB"
for p in "${pids[@]}"; do kill -TERM -- -$p 2>/dev/null; done
sleep 1
for p in "${pids[@]}"; do kill -KILL -- -$p 2>/dev/null; done
exit $rc
