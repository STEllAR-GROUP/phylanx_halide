// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "blaze_blas_plugin.hpp"

#include <phylanx/config.hpp>
#include <phylanx/plugins/plugin_factory.hpp>

PHYLANX_REGISTER_PLUGIN_MODULE();

PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dscal_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[0]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dasum_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[1]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dnrm2_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[2]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_daxpy_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[3]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dgemv_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[4]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dger_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[5]);
PHYLANX_REGISTER_PLUGIN_FACTORY(blaze_dgemm_plugin,
    phylanx_blaze_blas_plugin::blaze_blas::match_data[6]);
