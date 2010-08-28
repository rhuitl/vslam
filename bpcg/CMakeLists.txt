cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init(bpcg)

rosbuild_add_boost_directories()

add_definitions(-Wall -O3 -mfpmath=sse -mmmx -msse -msse2 -msse3)

#####################################################################
# BPCG library
rosbuild_add_library(bpcg src/bpcg.cpp)

# various compile flags.  -O3 doesn't seem to help; -DNDEBUG seemst to hurt(???)
rosbuild_add_compile_flags(bpcg -O3 -DNDEBUG -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(bpcg -g -O0 -DDEBUG=1 -Wno-missing-field-initializers -msse3)
#rosbuild_add_compile_flags(bpcg -O3 -DNDEBUG -Wno-missing-field-initializers )

#####################################################################
# unit tests
#

rosbuild_add_gtest(bin/test_bpcg src/test_bpcg.cpp)
target_link_libraries(bin/test_bpcg bpcg)


######################################################################
# executables
#

# test pose estimation
#rosbuild_add_executable(test/run_bpcg test/run_bpcg.cpp)
#target_link_libraries(test/run_bpcg bpcg)




######################################################################
# visualization
#


