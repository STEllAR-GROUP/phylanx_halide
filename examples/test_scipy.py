# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import scipy.linalg.blas
import numpy as np
import time

N = 2048


def dgemm_scipy(alpha, A, B, beta, C):
    return scipy.linalg.blas.dgemm(alpha,
                                   A,
                                   B,
                                   beta=beta,
                                   c=C,
                                   trans_a=False,
                                   trans_b=False,
                                   overwrite_c=True)


A = np.ones((N, N))
B = np.ones((N, N))
C = np.ones((N, N))

c_scipy = dgemm_scipy(2, A, B, 1, C)

b_scipy = time.time()
c_scipy = dgemm_scipy(2, A, B, 1, C)
e_scipy = time.time()

print('scipy', e_scipy - b_scipy)


def dgemm_scipy_in(N):
    alpha = 2
    beta = 1
    A = np.ones((N, N))
    B = np.ones((N, N))
    C = np.ones((N, N))
    return scipy.linalg.blas.dgemm(alpha,
                                   A,
                                   B,
                                   beta=beta,
                                   c=C,
                                   trans_a=False,
                                   trans_b=False,
                                   overwrite_c=True)


c_scipy_in = dgemm_scipy_in(N)

b_scipy_in = time.time()
c_scipy_in = dgemm_scipy_in(N)
e_scipy_in = time.time()

print('scipy', e_scipy_in - b_scipy_in)
