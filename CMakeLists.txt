cmake_minimum_required(VERSION 3.0.2)
project(vitulus_mapping)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs
)

# DIRECT-RASTER-ONLY (2026-07-19): the octomap/band chain (octomap_server +
# band_projector + the ot_dump C++ helper) has been retired. octomap
# find_package + the ot_dump target are gone; direct_raster is the mapper.

catkin_install_python(PROGRAMS
  nodes/insertion_gate
  nodes/trajectory_dem
  nodes/direct_raster
  nodes/mapping_manager
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
