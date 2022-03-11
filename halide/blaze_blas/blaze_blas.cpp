// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include "blaze_blas.hpp"

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
namespace phylanx_blaze_blas_plugin {

    constexpr char const* const dscal_string = R"(
        a, x
        Args:
            a (scalar): scaling factor
            x (array): 1d array

        Returns:

            Integer. Status.
        )";

    constexpr char const* const dasum_string = R"(
        N, x, incX
        Args:
            N (scalar): int
            x (array): 1d 
            incX (scalar): int

        Returns:

            Integer. Status.
        )";

    constexpr char const* const dnrm2_string = R"(
        N, x, incX
        Args:
            N (scalar): int
            x (array): 1d 
            incX (scalar): int

        Returns:

            Integer. Status.
        )";

    constexpr char const* const daxpy_string = R"(
        a, x, y
        Args:
            a (scalar): double
            x (array): 1d 
            y (array): 1d

        Returns:

            Integer. Status.
        )";

    constexpr char const* const dgemv_string = R"(
        is_trans, a, A, x, b, y
        Args:
            is_trans (bool) transpose?
            a (scalar): double
            A (array): 2d
            x (array): 1d 
            b (scalar): double
            y (array): 1d

        Returns:

            Integer. Status.
        )";

    constexpr char const* const dger_string = R"(
        is_trans, a, x, y, A
        Args:
            a (scalar): double
            x (array): 1d 
            y (array): 1d
            A (array): 2d

        Returns:

            Integer. Status.
        )";

    constexpr char const* const dgemm_string = R"(
        is_a_trans, is_b_trans, a, A, B, b, C
        Args:
            is_a_trans (bool) transpose A?
            is_b_trans (bool) transpose B?
            a (scalar): double
            A (array): 2d
            B (array): 2d
            b (scalar): double
            C (array): 2d

        Returns:

            Integer. Status.
        )";

    ///////////////////////////////////////////////////////////////////////////
    std::vector<phylanx::execution_tree::match_pattern_type> const
        blaze_blas::match_data = {
            phylanx::execution_tree::match_pattern_type{"blaze_dscal",
                std::vector<std::string>{"blaze_dscal(_1, _2)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>, dscal_string},

            phylanx::execution_tree::match_pattern_type{"blaze_dasum",
                std::vector<std::string>{"blaze_dasum(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>, dasum_string},

            phylanx::execution_tree::match_pattern_type{"blaze_dnrm2",
                std::vector<std::string>{"blaze_dnrm2(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>, dnrm2_string},

            phylanx::execution_tree::match_pattern_type{"blaze_daxpy",
                std::vector<std::string>{"blaze_daxpy(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>, daxpy_string},

            phylanx::execution_tree::match_pattern_type{"blaze_dgemv",
                std::vector<std::string>{"blaze_dgemv(_1, _2, _3, _4, _5, _6)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>,
                dgemv_string},

            phylanx::execution_tree::match_pattern_type{"blaze_dger",
                std::vector<std::string>{"blaze_dger(_1, _2, _3, _4)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>,
                dger_string},

            phylanx::execution_tree::match_pattern_type{"blaze_dgemm",
                std::vector<std::string>{"blaze_dgemm(_1, _2, _3, _4, _5, _6, _7)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blaze_blas>,
                dgemm_string} };

    blaze_blas::blas_mode extract_blas_mode(std::string const& name)
    {
        blaze_blas::blas_mode blas_op = blaze_blas::DGEMM;
        if (name.find("blaze_dscal") != std::string::npos) {
            blas_op = blaze_blas::DSCAL;
        }
        else if (name.find("blaze_dasum") != std::string::npos) {
            blas_op = blaze_blas::DASUM;
        }
        else if (name.find("blaze_dnrm2") != std::string::npos) {
            blas_op = blaze_blas::DNRM2;
        }
        else if (name.find("blaze_daxpy") != std::string::npos) {
            blas_op = blaze_blas::DAXPY;
        }
        else if (name.find("dgemv") != std::string::npos) {
            blas_op = blaze_blas::DGEMV;
        }
        else if (name.find("blaze_dger") != std::string::npos) {
            blas_op = blaze_blas::DGER;
        }
        else if (name.find("blaze_dgemm") != std::string::npos) {
            blas_op = blaze_blas::DGEMM;
        }
        else {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                name,
                phylanx::util::generate_error_message("BLAS operation not recognized."));
        }
        return blas_op;
    }
    ///////////////////////////////////////////////////////////////////////////
    blaze_blas::blaze_blas(primitive_arguments_type&& operands, std::string const& name,
        std::string const& codename)
        : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
        , mode_(extract_blas_mode(name_))
    {
    }

    phylanx::execution_tree::primitive_argument_type blaze_blas::blaze_dscal(
        primitive_argument_type&& a, primitive_argument_type&& x) const
    {
        float a_value = static_cast<float> (extract_scalar_numeric_value(std::move(a), name_, codename_));
        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(x), name_, codename_);
        auto in_vector = x_value.vector();
        int in_size = in_vector.size();
        in_vector = a_value * in_vector;
        return primitive_argument_type(std::move(x_value));
    }

    phylanx::execution_tree::primitive_argument_type blaze_blas::blaze_dgemm(
        primitive_argument_type&& is_a_trans,
        primitive_argument_type&& is_b_trans,
        primitive_argument_type&& a,
        primitive_argument_type&& A,
        primitive_argument_type&& B,
        primitive_argument_type&& b,
        primitive_argument_type&& C)  const
    {
        bool is_a = static_cast<bool> (extract_boolean_value(std::move(is_a_trans), name_, codename_));
        bool is_b = static_cast<bool> (extract_boolean_value(std::move(is_b_trans), name_, codename_));
        double a_value = extract_scalar_numeric_value(std::move(a), name_, codename_);
        auto A_value = phylanx::execution_tree::extract_numeric_value(std::move(A), name_, codename_);
        auto matrix_A = A_value.matrix();
        auto B_value = phylanx::execution_tree::extract_numeric_value(std::move(B), name_, codename_);
        auto matrix_B = B_value.matrix();
        double b_value = extract_scalar_numeric_value(std::move(b), name_, codename_);
        auto C_value = phylanx::execution_tree::extract_numeric_value(std::move(C), name_, codename_);
        auto matrix_C = C_value.matrix();

        C_value = (a_value * matrix_A) * matrix_B + (b_value * matrix_C);

        return primitive_argument_type(std::move(C_value));
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type> blaze_blas::eval(
        primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        auto this_ = this->shared_from_this();
        auto ctx_ = ctx;

        if (2 == operands.size() && this_->mode_ == DSCAL)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& a,
                    hpx::future<primitive_argument_type>&& x)
                ->primitive_argument_type {
                return this_->blaze_dscal(a.get(), x.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx));
        }

        if (6 == operands.size() && this_->mode_ == DGEMV)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& is_trans,
                    hpx::future<primitive_argument_type>&& a,
                    hpx::future<primitive_argument_type>&& A,
                    hpx::future<primitive_argument_type>&& x,
                    hpx::future<primitive_argument_type>&& b,
                    hpx::future<primitive_argument_type>&& y)
                ->primitive_argument_type {
                return this_->blaze_dgemv(is_trans.get(), a.get(), A.get(),
                    x.get(), b.get(), y.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[3], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[4], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[5], args, name_, codename_, ctx));
        }

        if (7 == operands.size() && this_->mode_ == DGEMM)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& is_a_trans,
                    hpx::future<primitive_argument_type>&& is_b_trans,
                    hpx::future<primitive_argument_type>&& a,
                    hpx::future<primitive_argument_type>&& A,
                    hpx::future<primitive_argument_type>&& B,
                    hpx::future<primitive_argument_type>&& b,
                    hpx::future<primitive_argument_type>&& C)
                ->primitive_argument_type {
                return this_->blaze_dgemm(is_a_trans.get(), is_b_trans.get(), a.get(), A.get(),
                    B.get(), b.get(), C.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[3], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[4], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[5], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[6], args, name_, codename_, ctx));
        }
        HPX_THROW_EXCEPTION(hpx::bad_parameter,
            "Non BLAS function",
            generate_error_message("Function not recognized.", ctx));
    }
}
