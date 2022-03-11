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
def dgemm_phy(alpha, A, B, beta, C):
    return np.dot(alpha * A, B) + beta * C


A = np.ones((N, N))
B = np.ones((N, N))
C = np.ones((N, N))

c_phy = dgemm_phy(2, A, B, 1, C)

b_phy = time.time()
c_phy = dgemm_phy(2, A, B, 1, C)
e_phy = time.time()

print('phy', e_phy - b_phy)


@Phylanx
def dgemm_phy_in(N):
    alpha = 2
    beta = 1
    A = np.ones((N, N))
    B = np.ones((N, N))
    C = np.ones((N, N))
    return np.dot(alpha * A, B) + beta * C


c_phy_in = dgemm_phy_in(N)

b_phy_in = time.time()
c_phy_in = dgemm_phy_in(N)
e_phy_in = time.time()

print('phy_in', e_phy_in - b_phy_in)
