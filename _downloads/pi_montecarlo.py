"""
Compute Pi by Monte-Carlo sampling.

"""
from __future__ import division
import numpy as np

NTOTAL = 1000000

np.random.seed(0)
x = np.random.uniform(-1, 1, NTOTAL)
y = np.random.uniform(-1, 1, NTOTAL)
ninside = np.sum(x**2 + y**2 < 1)
pi = 4 * ninside / NTOTAL

print pi, np.abs(np.pi - pi) / np.pi
