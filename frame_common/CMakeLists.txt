cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init(frame_common)

rosbuild_add_boost_directories()

add_definitions(-Wall -O3 -mfpmath=sse -mmmx -msse -msse2 -msse3)

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

#####################################################################
# library
rosbuild_add_library(frame_common src/frame.cpp src/stereo.cpp src/draw.cpp src/stereolib.c)

rosbuild_add_compile_flags(frame_common -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(frame_common -g -O0 -DDEBUG=1 -Wno-missing-field-initializers -msse3)

rosbuild_add_executable (test_frame src/test_frame.cpp)

#####################################################################
# unit tests
#



######################################################################
# executables
#


######################################################################
# visualization
#


