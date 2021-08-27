# Copyright (c) 2021 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_executable(harris.generator harris_generator.cpp)
target_link_libraries(harris.generator PRIVATE Halide::Generator Halide::Tools)

# Halide filters
add_halide_library(harris FROM harris.generator)
add_halide_library(
    harris_auto_schedule FROM harris.generator
    GENERATOR harris
    AUTOSCHEDULER Halide::Mullapudi2016)

# Phylanx plugin
set(plugin_headers
    ${CMAKE_CURRENT_LIST_DIR}/halide_plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/harris.hpp)

set(plugin_sources
    ${CMAKE_CURRENT_LIST_DIR}/hpx_runtime.cpp
    ${CMAKE_CURRENT_LIST_DIR}/halide_plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/harris.cpp)

add_phylanx_primitive_plugin(
    halide_plugin
    HEADERS ${plugin_headers}
    SOURCES ${plugin_sources}
    DEPENDENCIES
        Halide::ImageIO
        harris
        harris_auto_schedule
)
