// Copyright (c) 2021 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <Halide.h>

#include "blas.hpp"
#include "halide_blas.h"

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

using Halide::Runtime::Buffer;

///////////////////////////////////////////////////////////////////////////////
namespace phylanx_halide_plugin {

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
        blas::match_data = {
            phylanx::execution_tree::match_pattern_type{"dscal",
                std::vector<std::string>{"dscal(_1, _2)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blas>, dscal_string},

            phylanx::execution_tree::match_pattern_type{"dasum",
                std::vector<std::string>{"dasum(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blas>, dasum_string},

            phylanx::execution_tree::match_pattern_type{"dnrm2",
                std::vector<std::string>{"dnrm2(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blas>, dnrm2_string},

            phylanx::execution_tree::match_pattern_type{"daxpy",
                std::vector<std::string>{"daxpy(_1, _2, _3)"}, &create_dscal_op,
                &phylanx::execution_tree::create_primitive<blas>, daxpy_string},

            phylanx::execution_tree::match_pattern_type{"dgemv",
                std::vector<std::string>{"dgemv(_1, _2, _3, _4, _5, _6)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blas>,
                dgemv_string},

            phylanx::execution_tree::match_pattern_type{"dger",
                std::vector<std::string>{"dger(_1, _2, _3, _4)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blas>,
                dger_string},

            phylanx::execution_tree::match_pattern_type{"dgemm",
                std::vector<std::string>{"dgemm(_1, _2, _3, _4, _5, _6, _7)"},
                &create_dgemv_op,
                &phylanx::execution_tree::create_primitive<blas>,
                dgemm_string} };

    blas::blas_mode extract_blas_mode(std::string const& name)
    {
        blas::blas_mode blas_op = blas::DGEMM;
        if (name.find("dscal") != std::string::npos) {
            blas_op = blas::DSCAL;
        }
        else if (name.find("dasum") != std::string::npos) {
            blas_op = blas::DASUM;
        }
        else if (name.find("dnrm2") != std::string::npos) {
            blas_op = blas::DNRM2;
        }
        else if (name.find("daxpy") != std::string::npos) {
            blas_op = blas::DAXPY;
        }
        else if (name.find("dgemv") != std::string::npos) {
            blas_op = blas::DGEMV;
        }
        else if (name.find("dger") != std::string::npos) {
            blas_op = blas::DGER;
        }
        else if (name.find("dgemm") != std::string::npos) {
            blas_op = blas::DGEMM;
        }
        else {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                name,
                phylanx::util::generate_error_message("BLAS operation not recognized."));
        }
        return blas_op;
    }
    ///////////////////////////////////////////////////////////////////////////
    blas::blas(primitive_arguments_type&& operands, std::string const& name,
        std::string const& codename)
        : phylanx::execution_tree::primitives::primitive_component_base(
            std::move(operands), name, codename)
        , mode_(extract_blas_mode(name_))
    {
    }

    phylanx::execution_tree::primitive_argument_type blas::dscal(
        primitive_argument_type&& a, primitive_argument_type&& x) const
    {
        float a_value = static_cast<float> (extract_scalar_numeric_value(std::move(a), name_, codename_));
        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(x), name_, codename_);
        auto in_vector = x_value.vector();
        int in_size = in_vector.size();
        Buffer<double> x_buffer(in_vector.data(), in_size);
        halide_dscal_impl(a_value, x_buffer, nullptr, x_buffer);
        return primitive_argument_type(std::move(x_value));
    }

    phylanx::execution_tree::primitive_argument_type blas::dasum(
        primitive_argument_type&& N, primitive_argument_type&& X, primitive_argument_type&& incX) const
    {
        double result;
        int n_value = static_cast<int> (extract_scalar_numeric_value(std::move(N), name_, codename_));
        int inc_value = static_cast<int> (extract_scalar_numeric_value(std::move(incX), name_, codename_));
        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(X), name_, codename_);
        auto x_vector = x_value.vector();
        halide_dimension_t shape = { 0, n_value, inc_value };
        auto buff_x = Buffer<double>(x_vector.data(), 1, &shape);
        auto buff_sum = Buffer<double>::make_scalar(&result);
        halide_dasum(buff_x, buff_sum);

        return primitive_argument_type(std::move(result));
    }

    phylanx::execution_tree::primitive_argument_type blas::dnrm2(
        primitive_argument_type&& N, primitive_argument_type&& X, primitive_argument_type&& incX) const
    {
        double result;
        int n_value = static_cast<int> (extract_scalar_numeric_value(std::move(N), name_, codename_));
        int inc_value = static_cast<int> (extract_scalar_numeric_value(std::move(incX), name_, codename_));
        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(X), name_, codename_);
        auto x_vector = x_value.vector();
        halide_dimension_t shape = { 0, n_value, inc_value };
        auto buff_x = Buffer<double>(x_vector.data(), 1, &shape);
        auto buff_nrm = Buffer<double>::make_scalar(&result);
        halide_ddot(buff_x, buff_x, buff_nrm);

        return primitive_argument_type(std::sqrt(result));
    }

    phylanx::execution_tree::primitive_argument_type blas::daxpy(
        primitive_argument_type&& a, primitive_argument_type&& x, primitive_argument_type&& y) const
    {
        double a_value = static_cast<double> (extract_scalar_numeric_value(std::move(a), name_, codename_));

        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(x), name_, codename_);
        auto x_vector = x_value.vector();
        int x_size = x_vector.size();
        Buffer<double> x_buffer(x_vector.data(), x_size);

        auto y_value = phylanx::execution_tree::extract_numeric_value(std::move(y), name_, codename_);
        auto y_vector = y_value.vector();
        int y_size = y_vector.size();
        Buffer<double> y_buffer(y_vector.data(), y_size);

        halide_daxpy_impl(a_value, x_buffer, y_buffer, y_buffer);


        return primitive_argument_type(std::move(y_value));
    }

    phylanx::execution_tree::primitive_argument_type blas::dgemv(
        primitive_argument_type&& is_trans,
        primitive_argument_type&& a,
        primitive_argument_type&& A,
        primitive_argument_type&& x,
        primitive_argument_type&& b,
        primitive_argument_type&& y)  const
    {
        bool is_transpose = static_cast<bool> (extract_boolean_value(std::move(is_trans), name_, codename_));
        double a_value = extract_scalar_numeric_value(std::move(a), name_, codename_);
        double b_value = extract_scalar_numeric_value(std::move(b), name_, codename_);

        auto A_value = phylanx::execution_tree::extract_numeric_value(std::move(A), name_, codename_);
        auto vector_A = A_value.matrix();
        Buffer<double> A_buffer(vector_A.data(), vector_A.rows(), vector_A.columns());

        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(x), name_, codename_);
        auto x_vector = x_value.vector();
        int x_size = x_vector.size();
        Buffer<double> x_buffer(x_vector.data(), x_size);

        auto y_value = phylanx::execution_tree::extract_numeric_value(std::move(y), name_, codename_);
        auto y_vector = y_value.vector();
        int y_size = y_vector.size();
        Buffer<double> y_buffer(y_vector.data(), y_size);

        halide_dgemv(is_transpose, a_value, A_buffer, x_buffer, b_value, y_buffer);

        return primitive_argument_type(std::move(y_value));
    }

    phylanx::execution_tree::primitive_argument_type blas::dger(
        primitive_argument_type&& a,
        primitive_argument_type&& x,
        primitive_argument_type&& y,
        primitive_argument_type&& A)  const
    {
        double a_value = extract_scalar_numeric_value(std::move(a), name_, codename_);

        auto x_value = phylanx::execution_tree::extract_numeric_value(std::move(x), name_, codename_);
        auto x_vector = x_value.vector();
        int x_size = x_vector.size();
        Buffer<double> x_buffer(x_vector.data(), x_size);

        auto y_value = phylanx::execution_tree::extract_numeric_value(std::move(y), name_, codename_);
        auto y_vector = y_value.vector();
        int y_size = y_vector.size();
        Buffer<double> y_buffer(y_vector.data(), y_size);


        auto A_value = phylanx::execution_tree::extract_numeric_value(std::move(A), name_, codename_);
        auto vector_A = A_value.matrix();
        Buffer<double> A_buffer(vector_A.data(), vector_A.rows(), vector_A.columns());

        halide_dger(a_value, x_buffer, y_buffer, A_buffer);

        return primitive_argument_type(std::move(A_value));
    }

    phylanx::execution_tree::primitive_argument_type blas::dgemm(
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
        auto vector_A = A_value.matrix();
        auto B_value = phylanx::execution_tree::extract_numeric_value(std::move(B), name_, codename_);
        auto vector_B = B_value.matrix();
        double b_value = extract_scalar_numeric_value(std::move(b), name_, codename_);
        auto C_value = phylanx::execution_tree::extract_numeric_value(std::move(C), name_, codename_);
        auto vector_C = C_value.matrix();

        Buffer<double> A_buffer(vector_A.data(), vector_A.rows(), vector_A.columns());
        Buffer<double> B_buffer(vector_B.data(), vector_B.rows(), vector_B.columns());
        Buffer<double> C_buffer(vector_C.data(), vector_C.rows(), vector_C.columns());

        halide_dgemm(is_a, is_b, a_value, A_buffer, B_buffer, b_value, C_buffer);

        return primitive_argument_type(std::move(C_value));
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<phylanx::execution_tree::primitive_argument_type> blas::eval(
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
                return this_->dscal(a.get(), x.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx));
        }

        if (3 == operands.size() && this_->mode_ == DASUM)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& N,
                    hpx::future<primitive_argument_type>&& x,
                    hpx::future<primitive_argument_type>&& incX)
                ->primitive_argument_type {
                return this_->dasum(N.get(), x.get(), incX.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx));
        }

        if (3 == operands.size() && this_->mode_ == DNRM2)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& N,
                    hpx::future<primitive_argument_type>&& x,
                    hpx::future<primitive_argument_type>&& incX)
                ->primitive_argument_type {
                return this_->dnrm2(N.get(), x.get(), incX.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx));
        }

        if (3 == operands.size() && this_->mode_ == DAXPY)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& a,
                    hpx::future<primitive_argument_type>&& x,
                    hpx::future<primitive_argument_type>&& y)
                ->primitive_argument_type {
                return this_->daxpy(a.get(), x.get(), y.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx));
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
                return this_->dgemv(is_trans.get(), a.get(), A.get(),
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

        if (4 == operands.size() && this_->mode_ == DGER)
        {
            return hpx::dataflow(
                hpx::launch::sync,
                [this_ = std::move(this_), ctx = std::move(ctx_)](
                    hpx::future<primitive_argument_type>&& a,
                    hpx::future<primitive_argument_type>&& x,
                    hpx::future<primitive_argument_type>&& y,
                    hpx::future<primitive_argument_type>&& A)
                ->primitive_argument_type {
                return this_->dger(a.get(), x.get(), y.get(), A.get());
            },
                phylanx::execution_tree::value_operand(
                    operands[0], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[1], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[2], args, name_, codename_, ctx),
                phylanx::execution_tree::value_operand(
                    operands[3], args, name_, codename_, ctx));
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
                return this_->dgemm(is_a_trans.get(), is_b_trans.get(), a.get(), A.get(),
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
