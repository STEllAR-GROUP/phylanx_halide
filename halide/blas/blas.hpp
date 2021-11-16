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

    class blas
      : public phylanx::execution_tree::primitives::primitive_component_base
      , public std::enable_shared_from_this<blas>
    {
    private:
        using primitive_argument_type =
            phylanx::execution_tree::primitive_argument_type;
        using primitive_arguments_type =
            phylanx::execution_tree::primitive_arguments_type;
        using eval_context = phylanx::execution_tree::eval_context;

    protected:
        hpx::future<primitive_argument_type> eval(
            primitive_arguments_type const& operands,
            primitive_arguments_type const& args,
            eval_context ctx) const override;

    ///////////////////////////////////////////////////////////////////////////
    // DSCAL scales a vector by a constant.
        primitive_argument_type dscal(primitive_argument_type&& a /* double */,
            primitive_argument_type&& x /* halide_buffer_t */) const;

    // DASUM sums the absolute values of the elements of a double precision 
    // vector
        primitive_argument_type dasum(
            primitive_argument_type&& N /* const int */,
            primitive_argument_type&& X /* const double* */,
            primitive_argument_type&& incX /* const int */) const;
    ///////////////////////////////////////////////////////////////////////////
    // DNRM2 returns the euclidean norm of a vector via the function name, so 
    // thDNRM2 := sqrt( x'*x )
        primitive_argument_type dnrm2(
            primitive_argument_type&& N /* const int */,
            primitive_argument_type&& X /* const double* */,
            primitive_argument_type&& incX /* const int */) const;

    ///////////////////////////////////////////////////////////////////////////
    // DAXPY constant times a vector plus a vector.
        primitive_argument_type daxpy(
            primitive_argument_type&& a /* double */,
            primitive_argument_type&& x /* halide_buffer_t */,
            primitive_argument_type&& y /* halide_buffer_t */) const;

    ///////////////////////////////////////////////////////////////////////////
    // DGEMV  performs one of the matrix-vector operations
    // y := alpha*A*x + beta*y,   
    // or
    // y := alpha*A**T*x + beta*y,
    // where alpha and beta are scalars, x and y are vectors and 
    // A is an m by n matrix.
        primitive_argument_type dgemv(
            primitive_argument_type&& is_trans /* bool */,
            primitive_argument_type&& a /* double */,
            primitive_argument_type&& A /* halide_buffer_t */,
            primitive_argument_type&& x /* halide_buffer_t */,
            primitive_argument_type&& b /* double */,
            primitive_argument_type&& y /* halide_buffer_t */) const;

    ///////////////////////////////////////////////////////////////////////////
    // DGER   performs the rank 1 operation
    // A := alpha*x*y**T + A,
    //  where alpha is a scalar, x is an m element vector, y is an n element
    //  vector and A is an m by n matrix.
        primitive_argument_type dger(
            primitive_argument_type&& a /* double */,
            primitive_argument_type&& x /* halide_buffer_t */,
            primitive_argument_type&& y /* halide_buffer_t */,
            primitive_argument_type&& A /* halide_buffer_t */) const;

    ///////////////////////////////////////////////////////////////////////////
    // DGEMM  performs one of the matrix-matrix operations
    // C := alpha*op( A )*op( B ) + beta*C,
    // where  op( X ) is one of
    //      op( X ) = X
    //      or
    //      op( X ) = X**T,
    // alpha and beta are scalars, and A, B and C are matrices, with op( A )
    // an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
        primitive_argument_type dgemm(
            primitive_argument_type&& is_a_trans /* bool */,
            primitive_argument_type&& is_b_trans /* bool */,
            primitive_argument_type&& a /* double */,
            primitive_argument_type&& A /* halide_buffer_t */,
            primitive_argument_type&& B /* halide_buffer_t */,
            primitive_argument_type&& b /* double */,
            primitive_argument_type&& C /* halide_buffer_t */) const;


    

    public:
        enum blas_mode
        {
            DSCAL,
            DASUM,
            DNRM2,
            DAXPY,
            DGEMV,
            DGER,
            DGEMM
        };

        static std::vector<phylanx::execution_tree::match_pattern_type> const
            match_data;

        blas() = default;

        blas(primitive_arguments_type&& operands, std::string const& name,
            std::string const& codename);

    private:
        blas_mode mode_;
    };

    inline phylanx::execution_tree::primitive create_dscal_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dscal", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_dsum_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dsum", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_dnrm2_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dnrm2", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_daxpy_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "daxpy", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_dgemv_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dgemv", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_dger_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dger", std::move(operands), name, codename);
    }

    inline phylanx::execution_tree::primitive create_dgemm_op(
        hpx::id_type const& locality,
        phylanx::execution_tree::primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return phylanx::execution_tree::create_primitive_component(
            locality, "dgemm", std::move(operands), name, codename);
    }
}
