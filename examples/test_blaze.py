# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from phylanx import Phylanx
import numpy as np
import time

N = 2048

@Phylanx
def dgemm_blaze(alpha, A, B, beta, C):
    return blaze_dgemm(False, False, alpha, A, B, beta, C)

A = np.ones(N*N).reshape(N,N)
B = np.ones(N*N).reshape(N,N)
C = np.ones(N*N).reshape(N,N)

c_halide = dgemm_blaze(2, A, B, 1, C)

b_halide = time.time()
c_halide = dgemm_blaze(2, A, B, 1, C)
e_halide = time.time()

print('halide', e_halide - b_halide)

