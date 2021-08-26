# Copyright (c) 2021 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(plugin_headers
    ${CMAKE_CURRENT_LIST_DIR}/halide_plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/harries.hpp)

set(plugin_sources
    ${CMAKE_CURRENT_LIST_DIR}/halide_plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/harries.cpp)

add_phylanx_primitive_plugin(
    halide_plugin
    HEADERS ${plugin_headers}
    SOURCES ${plugin_sources})
