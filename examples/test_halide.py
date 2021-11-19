# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from phylanx import Phylanx, PhylanxSession
import numpy as np
import time

PhylanxSession.init(16)

N = 2048


@Phylanx
def dgemm_halide(alpha, A, B, beta, C):
    return dgemm(False, False, alpha, A, B, beta, C)


A = np.ones((N, N))
B = np.ones((N, N))
C = np.ones((N, N))

c_halide = dgemm_halide(2, A, B, 1, C)

b_halide = time.time()
c_halide = dgemm_halide(2, A, B, 1, C)
e_halide = time.time()

print('halide', e_halide - b_halide)


@Phylanx
def dgemm_halide_in(N):
    alpha = 2
    beta = 1
    A = np.ones((N, N))
    B = np.ones((N, N))
    C = np.ones((N, N))
    return dgemm(False, False, alpha, A, B, beta, C)


c_halide_in = dgemm_halide_in(N)

b_halide_in = time.time()
c_halide_in = dgemm_halide_in(N)
e_halide_in = time.time()

print('halide_in', e_halide_in - b_halide_in)


@Phylanx
def dscal_halide():
    return dscal(2, np.ones(4))


print("dscal", dscal_halide())


@Phylanx
def dasum_halide():
    return dasum(2, np.ones(4), 1)


print("dasum", dasum_halide())


@Phylanx
def dnrm2_halide():
    a = np.array([1, -2, 1, 1])
    return dnrm2(4, a, 1)


print("dnrm2", dnrm2_halide())


@Phylanx
def daxpy_halide():
    x = np.array([1, 1, 1, 1])
    y = np.array([1, -2, 1, 1])
    return daxpy(2, x, y)


print("daxpy", daxpy_halide())


@Phylanx
def dgemv_halide(N):
    alpha = 2
    beta = 1
    A = np.ones((N, N))
    x = np.ones(N)
    y = np.ones(N)
    return dgemv(False, alpha, A, x, beta, y)


print("dgemv", dgemv_halide(2))


@Phylanx
def dger_halide(N):
    alpha = 2
    A = np.ones((N, N))
    x = np.ones(N)
    y = np.ones(N)
    return dger(alpha, x, y, A)


print("dger", dger_halide(2))
