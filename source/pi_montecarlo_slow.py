"""
Compute Pi by Monte-Carlo sampling.

"""
from __future__ import division
import math
import random

NTOTAL = 1000000

random.seed(0)
ninside = 0
for i in xrange(NTOTAL):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    ninside += math.sqrt(x**2 + y**2) < 1
pi = 4 * ninside / NTOTAL

print pi, abs(math.pi - pi) / math.pi
