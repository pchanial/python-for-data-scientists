"""
Matrix condition number and error propagation.

"""
from __future__ import division, print_function
import numpy as np

s3 = np.sqrt(3)
A = np.array([[ 3,-s3,   1, -s3],
              [s3,  3, -s3,  -1],
              [s3,  1,  s3,   3],
              [ 1,-s3,  -3,  s3]]) / 4
B = np.array([[10, 10,  7, 8],
              [ 9,  2,  7, 7],
              [ 1,  5, 11, 1],
              [10, 11,  4, 8]])
x = np.array([1, 1, 1, 1])
bA = np.dot(A, x)
bB = np.dot(B, x)


# 1) check A is orthogonal
print(np.dot(A.T, A))


# 2) compute condition number
eigA = np.linalg.eigvals(A)
condA = max(abs(_) for _ in eigA) / min(abs(_) for _ in eigA)
# or
condA = np.linalg.cond(A)
condB = np.linalg.cond(B)


# 3) compute A^-1 and B^-1
invA = A.T
invB = np.linalg.inv(B)


# 4) check error propagation
N = 10000


def montecarlo(invM, b, x, N):
    norm_b = np.linalg.norm(b)
    norm_x = np.linalg.norm(x)
    deltab_b = np.empty(N)
    deltax_x = np.empty(N)
    for i in xrange(N):
        deltab = np.random.standard_normal(4) / np.sqrt(4) * norm_b * 1e-3
        deltab_b[i] = np.linalg.norm(deltab) / norm_b
        deltax = np.dot(invM, deltab)
        deltax_x[i] = np.linalg.norm(deltax) / norm_x
    return deltab_b, deltax_x

deltab_bA, deltax_xA = montecarlo(invA, bA, x, N)
deltab_bB, deltax_xB = montecarlo(invB, bB, x, N)
print(np.mean(deltax_xA / deltab_bA))
print(np.mean(deltax_xB / deltab_bB))
