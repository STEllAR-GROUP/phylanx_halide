// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/base_primitive.hpp>
#include <phylanx/execution_tree/primitives/primitive_component_base.hpp>

#include <hpx/future.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace phylanx_halide_plugin {

    class harris
      : public phylanx::execution_tree::primitives::primitive_component_base
      , public std::enable_shared_from_this<harris>
    {
    private:
        using primitive_argument_type =
            phylanx::execution_tree::primitive_argument_type;
        using primitive_arguments_type =
            phylanx::execution_tree::primitive_arguments_type;
        using eval_context = phylanx::execution_tree::eval_context;

    protected:
        // hpx::future<primitive_argument_type> eval(
        //     primitive_arguments_type const& operands,
        //     primitive_arguments_type const& args) const;
        hpx::future<primitive_argument_type> eval(
            primitive_arguments_type const& params,
            primitive_arguments_type const& args,
            eval_context ctx) const override;

        primitive_argument_type filter(
            primitive_argument_type&& val, eval_context ctx) const;

    public:
        static phylanx::execution_tree::match_pattern_type const match_data;

        harris() = default;

        harris(primitive_arguments_type&& operands, std::string const& name,
            std::string const& codename);

        // hpx::future<primitive_argument_type> eval(
        //     primitive_arguments_type const& params,
        //     primitive_arguments_type const& args,
        //     eval_context ctx) const override;
    };

    inline phylanx::execution_tree::primitive create_harris(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "harris", std::move(operands), name, codename);
    }
}
