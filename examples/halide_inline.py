from phylanx import Phylanx, PhylanxSession
import numpy as np
import time

PhylanxSession.init(16)

N = 2048


@Phylanx
def dgemm_halide_in(N):
    alpha = 2
    beta = 1
    return dgemm(False, False, alpha, np.ones((N, N)), np.ones((N, N)), beta, np.ones((N, N)))

a_halide_in = time.time()
c_halide_in = dgemm_halide_in(N)
b_halide_in = time.time()

d_halide_in = time.time()
c_halide_in = dgemm_halide_in(N)
e_halide_in = time.time()

print('halide_in_first', b_halide_in - a_halide_in)
print('halide_in_first', e_halide_in - d_halide_in)

