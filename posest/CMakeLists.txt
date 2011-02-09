cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
set(ROS_BUILT_TYPE RelWithDebInfo)
rosbuild_init(posest)

rosbuild_add_boost_directories()

rosbuild_check_for_sse()

add_definitions(-Wall -Wno-missing-field-initializers -fopenmp)

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

#####################################################################
# PE library
rosbuild_add_library(posest src/pe.cpp src/pe3d.cpp src/planarSFM.cpp src/pe2d.cpp src/pnp_ransac.cpp)
target_link_libraries(posest tbb) # Intel Thread Building Blocks

#####################################################################
# PE test library: artificial data generation
rosbuild_add_library(pe_simulated test/simulated.cpp)
target_link_libraries(pe_simulated posest)

#####################################################################
# unit tests
#



######################################################################
# executables
#

# test pose estimation
rosbuild_add_executable(test/run_posest test/run_posest.cpp)
target_link_libraries(test/run_posest posest)

# test pose estimation for mono camera
rosbuild_add_executable(test/run_simulated2d test/run_simulated2d.cpp)
target_link_libraries(test/run_simulated2d pe_simulated)

# read sequence and SBA it
rosbuild_add_executable(test/run_sequence test/run_sequence.cpp)
target_link_libraries(test/run_sequence posest)



######################################################################
# visualization
#

######################################################################
# ESM tracking
#

rosbuild_add_library(esm src/esm.cpp)
rosbuild_add_executable(test/run_esm test/run_esm.cpp)
target_link_libraries(test/run_esm esm)

