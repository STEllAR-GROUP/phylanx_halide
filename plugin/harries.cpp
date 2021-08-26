// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/exception.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "harries.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_halide_plugin {

    constexpr char const* const help_string = R"(
        harries(input)
        Args:

            input (array) : image array to process

        Returns:

            the processed image
        )";

    ///////////////////////////////////////////////////////////////////////////
    phylanx::execution_tree::match_pattern_type const harries::match_data = {
        hpx::make_tuple("harries",
            std::vector<std::string>{"harries(_1)"},
            &create_harries,
            &phylanx::execution_tree::create_primitive<harries>,
            help_string)};

    ///////////////////////////////////////////////////////////////////////////
    harries::harries(primitive_arguments_type&& operands,
        std::string const& name, std::string const& codename)
      : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type> harries::eval(
        primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (operands.size() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "harries::eval",
                generate_error_message(
                    "harries accepts exactly one argument", ctx));
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(
            hpx::launch::sync,
            [this_ = std::move(this_)](hpx::future<primitive_argument_type>&& val)
                -> primitive_argument_type { return val.get(); },
            phylanx::execution_tree::value_operand(
                operands[0], args, name_, codename_, std::move(ctx)));
    }
}
