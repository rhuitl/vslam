cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
set(ROS_BUILT_TYPE RelWithDebInfo)
rosbuild_init()

rosbuild_add_boost_directories()

rosbuild_check_for_sse()

add_definitions(-Wall -Wno-missing-field-initializers -fopenmp)

#####################################################################
# library
rosbuild_add_library(frame_common src/frame.cpp src/stereo.cpp src/draw.cpp src/stereolib.c)
rosbuild_add_compile_flags(frame_common ${SSE_FLAGS})

#####################################################################
# unit tests
#



######################################################################
# executables
#
rosbuild_add_executable (test_frame src/test_frame.cpp)
target_link_libraries (test_frame gomp)
######################################################################
# visualization
#


