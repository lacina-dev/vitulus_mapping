cmake_minimum_required(VERSION 3.0.2)
project(vitulus_mapping)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

# classifier v2 (2026-07-19): the ot_dump helper reads octomap FREE leaves off
# the saved garden.ot so band_projector can build an honest FREE mask.
find_package(octomap REQUIRED)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs
  DEPENDS OCTOMAP
)

include_directories(${OCTOMAP_INCLUDE_DIRS})

# ot_dump: dump all octomap leaf centres (free + occupied) to a flat binary.
# Lands at devel/lib/vitulus_mapping/ot_dump (rosrun vitulus_mapping ot_dump).
add_executable(ot_dump src/tools/ot_dump.cpp)
target_link_libraries(ot_dump ${OCTOMAP_LIBRARIES})

catkin_install_python(PROGRAMS
  nodes/insertion_gate
  nodes/trajectory_dem
  nodes/band_projector
  nodes/mapping_manager
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ot_dump
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
