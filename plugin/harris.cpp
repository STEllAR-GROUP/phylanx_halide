// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<hpx/config.hpp>

#include <Halide.h>

#include "harris.h"
#include "harris.hpp"

#include <phylanx/config.hpp>

#include <hpx/exception.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_halide_plugin {

    constexpr char const* const help_string = R"(
        harris(input)
        Args:

            input (array) : image array to process

        Returns:

            the processed image
        )";

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::match_pattern_type const harris::match_data = {
        hpx::make_tuple("harris",
            std::vector<std::string>{"harris(_1)"},
            &create_harris,
            &phylanx::execution_tree::create_primitive<harris>,
            help_string)};

    ///////////////////////////////////////////////////////////////////////////
    harris::harris(primitive_arguments_type&& operands, std::string const& name,
        std::string const& codename)
      : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
    {
    }

    phylanx::execution_tree::primitive_argument_type harris::filter(
        primitive_argument_type&& val, eval_context ctx) const
    {
        auto data = extract_numeric_value(std::move(val), name_, codename_);

        if (phylanx::execution_tree::extract_numeric_value_dimension(
                data, name_, codename_) != 3)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "harris::filter",
                generate_error_message("the harris filter primitive accepts "
                                       "only 3D data as it's input",
                    ctx));
        }

        auto img = data.tensor();
        auto input = Halide::Runtime::Buffer<double>::make_interleaved(
            img.data(), img.columns(), img.rows(), img.pages());

        blaze::DynamicMatrix<double> outimg(
            input.width() - 6, input.height() - 6);

        {
            auto output = Halide::Runtime::Buffer<double>::make_interleaved(
                outimg.data(), outimg.columns(), outimg.rows(), 1);
            output.set_min(3, 3);

            ::harris(input, output);
            output.device_sync();
        }

        return primitive_argument_type(std::move(outimg));
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type> harris::eval(
        primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (operands.size() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "harris::eval",
                generate_error_message(
                    "harris accepts exactly one argument", ctx));
        }

        auto this_ = this->shared_from_this();
        auto ctx_ = ctx;
        return hpx::dataflow(
            hpx::launch::sync,
            [this_ = std::move(this_), ctx = std::move(ctx_)](
                hpx::future<primitive_argument_type>&& val)
                -> primitive_argument_type {
                return this_->filter(val.get(), ctx);
            },
            phylanx::execution_tree::value_operand(
                operands[0], args, name_, codename_, std::move(ctx)));
    }
}
