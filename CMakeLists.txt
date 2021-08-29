# Copyright (c) 2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V3.3.2
cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

# explicitly set certain policies
cmake_policy(VERSION 3.3.2)

set(CMAKE_CXX_STANDARD 17)

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)")
endif()

################################################################################
# project metadata
################################################################################
project(HalidePhylanxPlugin CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Debug library postfix
set(CMAKE_DEBUG_POSTFIX "d")
set(PHYLANX_DEBUG_POSTFIX "d")

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

find_package(Phylanx)
find_package(Halide REQUIRED)

add_subdirectory(plugin)
