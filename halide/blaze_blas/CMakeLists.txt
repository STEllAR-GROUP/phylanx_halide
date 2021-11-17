# Copyright (c) 2021 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


# Phylanx plugin
set(plugin_headers
    ${CMAKE_CURRENT_LIST_DIR}/blaze_blas_plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/blaze_blas.hpp)

set(plugin_sources
    ${CMAKE_CURRENT_LIST_DIR}/blaze_blas_plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/blaze_blas.cpp)

add_phylanx_primitive_plugin(
    blaze_blas_plugin
    HEADERS ${plugin_headers}
    SOURCES ${plugin_sources}
)
