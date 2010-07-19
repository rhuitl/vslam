cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init(sba)

rosbuild_add_boost_directories()

add_definitions(-Wall -O3 -mfpmath=sse -mmmx -msse -msse2 -msse3)
##
## NOTE: if you include these, you must make sure the manifest 
## exports them!!!
#add_definitions(-DSBA_CHOLMOD -DSBA_DSIF)
add_definitions(-DSBA_CHOLMOD)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#####################################################################
# SBA library
rosbuild_add_library(sba src/sba.cpp src/spa.cpp src/spa2d.cpp src/csparse.cpp src/proj.cpp src/node.cpp src/sba_file_io.cpp)
target_link_libraries(sba blas lapack cholmod amd camd colamd ccolamd csparse)

# SBA library with ROS & utilities, including reading from file and visualization.
rosbuild_add_library(sba_ros src/visualization.cpp)
target_link_libraries(sba_ros sba)

# various compile flags.  -O3 doesn't seem to help; -DNDEBUG seemst to hurt(???)
rosbuild_add_compile_flags(sba -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(sba -g -O0 -DDEBUG=1 -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(sba -O3 -DNDEBUG -Wno-missing-field-initializers )


#####################################################################
# unit tests
#

# basic transforms and jacobians
#rosbuild_add_gtest(test/sba_jac_test test/sba_jac.cpp)
#target_link_libraries(test/sba_jac_test sba)
#rosbuild_add_gtest(test/sbast_jac_test test/sbast_jac.cpp)
#target_link_libraries(test/sbast_jac_test sba)

# SBA
rosbuild_add_gtest(test/sba_test test/sba_test.cpp)
target_link_libraries(test/sba_test sba)
rosbuild_add_gtest(test/sbast_test test/sbast_test.cpp)
target_link_libraries(test/sbast_test sba)

# Spiral SBA
rosbuild_add_gtest(test/spiral_test test/spiral_test.cpp test/spiral_setup.cpp TIMEOUT 1000.0)
target_link_libraries(test/spiral_test sba)

# SPA 2D
rosbuild_add_gtest(test/spa2d_test test/spa2d_test.cpp test/spiral_setup.cpp)
target_link_libraries(test/spa2d_test sba)

# Monocular
rosbuild_add_gtest(test/mono_test test/mono_test.cpp test/spiral_setup.cpp)
target_link_libraries(test/mono_test sba)

# File IO
rosbuild_add_gtest(test/file_io_test test/file_io_test.cpp)
target_link_libraries(test/file_io_test  sba)

######################################################################
# executables
#

# test cholmod timing
rosbuild_add_executable(test/cholmod_timing test/cholmod_timing.cpp)
target_link_libraries(test/cholmod_timing sba)

# SBA on Bundler-type files
rosbuild_add_executable(test/run_sba_bundler_file test/run_sba_bundler_file.cpp)
target_link_libraries(test/run_sba_bundler_file sba)

# SPA2D on Freiburg constraint files
rosbuild_add_executable(test/run_spa2d test/run_spa2d.cpp test/read_spa.cpp)
target_link_libraries(test/run_spa2d sba)

# SPA on Freiburg constraint files
rosbuild_add_executable(test/run_spa test/run_spa.cpp test/read_spa.cpp)
target_link_libraries(test/run_spa sba)

# Spiral SBA for results
rosbuild_add_executable(test/run_sba_spiral test/run_sba_spiral.cpp test/spiral_setup.cpp)
target_link_libraries(test/run_sba_spiral sba)

# Sphere SBA for results
rosbuild_add_executable(test/run_sba_sphere test/run_sba_sphere.cpp test/spiral_setup.cpp)
target_link_libraries(test/run_sba_sphere sba)

# Test Cholesky timing
#rosbuild_add_executable(test/choldemo test/choldemo.cpp)
#target_link_libraries(test/choldemo lapack blas f2c)

# Stand-alone SBA ROS Node
rosbuild_add_executable(bin/sba_node src/nodes/sba_node.cpp)
target_link_libraries(bin/sba_node sba_ros)

######################################################################
# visualization
#

# SPA2D
rosbuild_add_executable(test/vis-2d test/vis-2d.cpp test/read_spa.cpp)
target_link_libraries(test/vis-2d sba)

# SBA on bundler files
rosbuild_add_executable(test/vis-bundler test/vis-bundler.cpp)
target_link_libraries(test/vis-bundler sba)

# SBA on monocular spiral
rosbuild_add_executable(test/vis-mono test/vis-mono.cpp test/spiral_setup.cpp)
target_link_libraries(test/vis-mono sba)

