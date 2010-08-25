cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set(ROS_BUILD_TYPE RelWithDebInfo)
#set(ROS_BUILD_TYPE Release)
#set(ROS_BUILD_TYPE Debug)

rosbuild_init()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
rosbuild_add_boost_directories()

# Dynamic reconfigure
rosbuild_find_ros_package(dynamic_reconfigure)
include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
gencfg()

#add_definitions(-Wall -O3 -mfpmath=sse -mmmx -msse -msse2 -msse3)

#####################################################################
# download data

rosbuild_download_data(http://pr.willowgarage.com/data/vocabulary_tree/holidays.tree data/holidays.tree bf1d267dfc0b3b1953e4e82c8b2d2b7e)
rosbuild_download_data(http://pr.willowgarage.com/data/vocabulary_tree/holidays.weights data/holidays.weights 9705a970c83999258a57bacea105fea7)

#####################################################################
# libraries
rosbuild_add_library(vo src/vo.cpp src/place_recognizer.cpp src/vslam.cpp src/vslam_mono.cpp)

rosbuild_add_library(vo_ptcloud src/vo_ptcloud.cpp src/place_recognizer.cpp src/vslam_ptcloud.cpp)

# various compile flags.  -O3 doesn't seem to help; -DNDEBUG seemst to hurt(???)
#rosbuild_add_compile_flags(vo -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(vo -g -O0 -DDEBUG=1 -Wno-missing-field-initializers -msse3)

#####################################################################
# unit tests
#


######################################################################
# test for run_mono pipeline using simulated data
rosbuild_add_executable(run_simulated_mono test/run_simulated_mono.cpp)
target_link_libraries(run_simulated_mono vo)

######################################################################
# executables
#

# read stereo sequence and SBA it
rosbuild_add_executable(run_stereo src/run_stereo.cpp)
target_link_libraries(run_stereo vo)
#rosbuild_add_compile_flags(bin/run_stereo -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(bin/run_stereo -g -O0 -DDEBUG=1 -Wno-missing-field-initializers -msse3)

# read stereo sequence for an object and SBA it
rosbuild_add_executable(run_object src/run_object.cpp)
target_link_libraries(run_object vo)
rosbuild_add_compile_flags(run_object -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)


# read stereo sequence and run mono VO
rosbuild_add_executable(run_mono src/run_mono.cpp)
target_link_libraries(run_mono vo)

# stereo VSLAM as a ROS node
rosbuild_add_executable(stereo_vslam_node src/nodes/stereo_vslam_node.cpp)
target_link_libraries(stereo_vslam_node vo)

# monocular VSLAM as a ROS node
rosbuild_add_executable(mono_vslam_node src/nodes/mono_vslam_node.cpp)
target_link_libraries(mono_vslam_node vo)


# stereo & ptcloud VSLAM as a ROS node
rosbuild_add_executable(ptcloud_vslam_node src/nodes/ptcloud_vslam_node.cpp)
target_link_libraries(ptcloud_vslam_node vo)

# stereo VO as a ROS node
rosbuild_add_executable(vo_node src/nodes/vo_node.cpp)
target_link_libraries(vo_node vo)

# publish NewCollege (or other) static dataset over ROS
rosbuild_add_executable(fake_stereo_cam src/nodes/fake_stereo_cam.cpp)

######################################################################
# visualization
#

# SBA on intel files
rosbuild_add_executable(vis-intel src/vis-intel.cpp src/read_spa.cpp)
