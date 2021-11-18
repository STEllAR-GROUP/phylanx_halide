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


A = np.ones((N, N))
B = np.ones((N, N))
C = np.ones((N, N))

c_blaze = dgemm_blaze(2, A, B, 1, C)

b_blaze = time.time()
c_blaze = dgemm_blaze(2, A, B, 1, C)
e_blaze = time.time()

print('blaze', e_blaze - b_blaze)


@Phylanx
def dgemm_blaze_in(N):
    alpha = 2
    beta = 1
    A = np.ones((N, N))
    B = np.ones((N, N))
    C = np.ones((N, N))
    return blaze_dgemm(False, False, alpha, A, B, beta, C)


c_blaze_in = dgemm_blaze_in(N)

b_blaze_in = time.time()
c_blaze_in = dgemm_blaze_in(N)
e_blaze_in = time.time()

print('blaze_in', e_blaze_in - b_blaze_in)
